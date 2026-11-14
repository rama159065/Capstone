"""FastAPI backend for the Track B Agentic RAG capstone.

Run locally:
    uvicorn scripts.api.api_app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import hashlib
import hmac
import json
import contextvars
import logging
import os
import time
from collections import OrderedDict, deque
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.routing import APIRoute
from scripts.rag.retrieval_metrics import response_metrics
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from scripts.observability.langsmith_utils import configure_langsmith, env_flag, run_traced
from scripts.common.runtime_profile import apply_profile, describe as describe_runtime
from scripts.agents.financial_clarifier import Clarifier, ConversationStore, name as company_name
from scripts.common.agent_limits import max_steps
from scripts.common.companies import NAMES as COMPANY_NAMES, SECTOR_NAMES, SECTOR_OF
from scripts.common.sector_corpus import parse_sectors, sector_slug
from scripts.agents.financial_prompting_agent import (
    AgentConfig,
    deterministic_input_guard,
    retrieve_evidence,
    run_technique,
    summarize_result,
)


FINANCIAL_DOMAIN = "financial_report_qa"
ALL_DOMAINS = (FINANCIAL_DOMAIN,)      # this service is the financial report copilot only
logger = logging.getLogger(__name__)
RUNTIME = apply_profile()   # local or colab defaults, before any retrieval runs
request_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "api_request_id", default=None
)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


# Deployment settings: every one is an environment variable (see .env.example).
RATE_LIMIT_REQUESTS = env_int("RATE_LIMIT_PER_MINUTE", 60)
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_CLIENTS = 10_000
CORS_ALLOW_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",") if o.strip()] or ["*"]
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "").strip()          # empty = no authentication (local runs)
PUBLIC_PATHS = {"/health", "/docs", "/redoc", "/openapi.json", "/docs/oauth2-redirect"}
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "mock")
DEFAULT_EMBEDDING_PROVIDER = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "hashing")
if DEFAULT_LLM_PROVIDER not in {"mock", "hosted"} or DEFAULT_EMBEDDING_PROVIDER not in {"hashing", "hosted"}:
    raise ValueError("DEFAULT_LLM_PROVIDER must be mock|hosted and DEFAULT_EMBEDDING_PROVIDER hashing|hosted")
ANSWER_CACHE_SECONDS = env_int("ANSWER_CACHE_SECONDS", 0)            # 0 = off
ANSWER_CACHE_MAX = env_int("ANSWER_CACHE_MAX", 256)


class RetrievalEvaluationRequest(BaseModel):
    qrels: dict[str, float] | None = Field(None, description="Complete relevance judgments keyed by evidence chunk/page ID; nonnegative grades.")
    relevant_chunk_ids: list[str] | None = Field(None, description="Binary ground truth when graded qrels are unavailable.")


    @field_validator("qrels")
    @classmethod
    def validate_qrels(cls, value):
        from scripts.rag.retrieval_metrics import retrieval_metrics
        retrieval_metrics([], value)
        return value


class MetricsRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()

        async def handler(request):
            response = await original(request)
            if request.method == "POST" and response.status_code == 200:
                payload = json.loads(response.body)
                if isinstance(payload, dict) and ("evidence" in payload or "evidence_chunk_ids" in payload):
                    inputs = await request.json()
                    payload["retrieval_metrics"] = response_metrics(
                        payload.get("evidence") or [{"chunk_id": key} for key in payload.get("evidence_chunk_ids", [])], inputs.get("qrels"),
                        inputs.get("relevant_chunk_ids"), inputs.get("top_k", 5))
                    return JSONResponse(payload, status_code=response.status_code,
                                        headers={key: value for key, value in response.headers.items()
                                                 if key.lower() != "content-length"})
            return response

        return handler


class ChatRequest(RetrievalEvaluationRequest):
    """Input shared by the non-financial and financial chat endpoints."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Question to answer using the selected domain's indexed evidence.",
    )
    llm_provider: Literal["mock", "hosted"] = Field(
        default_factory=lambda: DEFAULT_LLM_PROVIDER,
        description="Chat backend. Hosted mode requires the LLM endpoint environment variables.",
    )
    embedding_provider: Literal["hashing", "hosted"] = Field(
        default_factory=lambda: DEFAULT_EMBEDDING_PROVIDER,
        description="Embedding backend used for dense retrieval.",
    )
    hash_dimensions: int = Field(
        384, ge=32, le=4096, description="Vector size when hashing embeddings are selected."
    )
    top_k: int = Field(5, ge=1, le=20, description="Maximum evidence chunks returned to the answer step.")
    dense_top_k: int = Field(30, ge=1, le=100, description="Candidate count for dense retrieval.")
    bm25_top_k: int = Field(30, ge=1, le=100, description="Candidate count for BM25 retrieval.")
    source_type: str | None = Field(
        None, description="Optional source category used to filter retrieval."
    )
    max_context_chars: int = Field(
        12000, ge=1000, le=64000, description="Maximum retrieved context supplied to the LLM."
    )
    enable_guardrails: bool = Field(True, description="Enable input, evidence, and output safety checks.")
    enable_human_review: bool = Field(True, description="Enable routing for cases requiring human review.")
    enable_langsmith: bool = Field(False, description="Request LangSmith tracing for this operation.")
    langsmith_project: str | None = Field(
        None, description="Optional LangSmith project name; uses a domain-specific default if omitted."
    )


class FinancialChatRequest(ChatRequest):
    """Input for financial report QA, including technique and filing filters."""

    technique: Literal[
        "baseline_rag",
        "structured_few_shot",
        "decomposition",
        "self_consistency",
        "agentic_graph",
        "trend_graph",
        "sector_trend_graph",
        "fund_graph",
        "router",
    ] = Field("agentic_graph", description=(
        "Financial QA workflow to execute. agentic_graph: the Q&A agent (planner, per-company searches, calculator, "
        "critic). trend_graph: the multi-year trend agent (one search per fiscal year, verified series, changes and "
        "CAGR computed in code). sector_trend_graph: the sector trend agent (one measure across one sector's "
        "companies, year by year, over a fixed panel; totals, changes and CAGR computed in code). fund_graph: the "
        "fund agent (a mutual fund's SEC N-PORT holdings as cited rows, linked to the holdings' annual reports; needs "
        "loaded fund data). router: a supervisor that sends each question to the Q&A agent, the trend agent, the "
        "sector trend agent, the fund agent (when fund data is loaded), a report handoff or a refusal. The others are "
        "single-prompt baselines."))
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = Field(
        "hybrid", description="hybrid (default): dense + BM25 fused by RRF, on Qdrant or Chroma; "
                              "dense or bm25: one retriever only (for ablations)."
    )
    ticker: str | None = Field(None, description="Optional company ticker to filter financial evidence.")
    year: int | None = Field(None, description="Optional fiscal/reporting year filter.")
    sector: str | None = Field(None, max_length=200, description=(
        "Optional sector filter for the annual-report corpus, e.g. 'banking' or 'Oil and gas' (comma-separated "
        "for several). Applies to the financial_qdrant and financial_vlm backends."))
    self_consistency_samples: int = Field(
        3, ge=1, le=5, description="Number of answer samples for the self-consistency technique."
    )


class ConversationRequest(FinancialChatRequest):
    """One chat turn; incomplete questions are clarified before any agent runs."""

    conversation_id: str | None = Field(
        None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9-]+$",
        description="Returned by the first turn; send it back with every follow-up message.",
    )
    clarify: bool = Field(True, description="Ask for missing company, year or measure before running agents.")


class RetrieveRequest(RetrievalEvaluationRequest):
    """Input shared by the financial and non-financial retrieval endpoints."""

    query: str = Field(..., min_length=1, max_length=4000, description="Search query.")
    embedding_provider: Literal["hashing", "hosted"] = Field(
        "hashing", description="Embedding backend used for dense retrieval."
    )
    hash_dimensions: int = Field(
        384, ge=32, le=4096, description="Vector size when hashing embeddings are selected."
    )
    top_k: int = Field(5, ge=1, le=20, description="Maximum number of evidence records to return.")
    dense_top_k: int = Field(30, ge=1, le=100, description="Candidate count for dense retrieval.")
    bm25_top_k: int = Field(30, ge=1, le=100, description="Candidate count for BM25 retrieval.")
    source_type: str | None = Field(None, description="Optional source category filter.")
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = Field(
        "hybrid", description="hybrid (default): dense + BM25 fused by RRF, on Qdrant or Chroma; "
                              "dense or bm25: one retriever only (for ablations)."
    )
    ticker: str | None = Field(None, description="Optional company ticker filter for financial evidence.")
    year: int | None = Field(None, description="Optional fiscal/reporting year filter.")
    sector: str | None = Field(None, max_length=200, description="Optional sector filter for financial evidence (annual-report corpus).")
    enable_langsmith: bool = Field(False, description="Request LangSmith tracing for this operation.")
    langsmith_project: str | None = Field(
        None, description="Optional LangSmith project name; uses a domain-specific default if omitted."
    )


class ApiResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class MultimodalRequest(RetrievalEvaluationRequest):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(5, ge=1, le=10)
    provider: Literal["extractive", "ollama", "transformers", "llamacpp"] = Field(
        default_factory=lambda: "llamacpp" if os.getenv("RAG_VISION_API_BASE") else "extractive")
    source_type: str | None = None
    ticker: str | None = None
    year: int | None = None
    idempotency_key: str | None = Field(None, min_length=1, max_length=128)
    timeout_seconds: float = Field(60, gt=0, le=600)


class HealthResponse(ApiResponse):
    status: Literal["ok"]
    service: str
    domains: list[str]


class DomainInfoResponse(ApiResponse):
    domain: str
    display_name: str
    description: str
    default_query: str
    source_types: list[str] | None = None


class DomainsResponse(ApiResponse):
    domains: list[DomainInfoResponse]


class LangSmithResponse(ApiResponse):
    enabled: bool
    reason: str | None = None
    project: str | None = None
    endpoint: str | None = None
    hint: str | None = None


class EvidenceResponse(ApiResponse):
    chunk_id: str | None = None
    title: str | None = None
    source_type: str | None = None
    ticker: str | None = None
    year: int | None = None
    page: int | None = None
    source_url: str | None = None
    image_path: str | None = None
    rank: int | None = None
    score: float | None = None
    preview: str | None = None


class CitationCheckResponse(ApiResponse):
    valid: bool
    cited_chunk_ids: list[str] | None = None
    missing_chunk_ids: list[str] | None = None
    allowed_chunk_ids: list[str] | None = None


class GroundingResponse(BaseModel):
    status: Literal["grounded", "not_grounded", "not_applicable"]
    evidence_count: int
    cited_chunk_ids: list[str]
    reason: str


class ChatResponse(ApiResponse):
    domain: str
    display_name: str | None = None
    final_status: str
    query: str | None = None
    safe_query: str | None = None
    answer: str | None = None
    explanation: str | None = Field(
        None,
        description=(
            "Concise explanation of the workflow outcome based on retrieved evidence, "
            "citation checks, safety checks, or human-review triggers."
        ),
    )
    grounding: GroundingResponse | None = None
    langsmith: LangSmithResponse | None = None
    security: dict[str, Any] | None = None
    plain_text: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    evidence_guard: dict[str, Any] | None = None
    output_filter: dict[str, Any] | None = None
    citation_check: CitationCheckResponse | None = None
    human_review: dict[str, Any] | None = None
    evidence: list[EvidenceResponse] | None = None
    orchestration: dict[str, Any] | None = None
    elapsed_seconds: float | None = None


class FinancialChatResponse(ChatResponse):
    technique: str | None = None
    parsed: dict[str, Any] | None = None
    subquestions: list[str] | None = None
    evidence_chunk_ids: list[str | None] | None = None
    evidence_preview: list[EvidenceResponse] | None = None
    support_check: dict[str, Any] | None = Field(
        default=None,
        description="agentic_graph, trend_graph and router: the critic's support check. `verdict` is supported, partially_supported, "
                    "unsupported or unverified; `claims` are the judge's per-claim decisions; `numeric` lists figures "
                    "missing from the cited chunks. A hosted-LLM answer is shown only when the verdict is supported.",
    )
    route: dict[str, Any] | None = Field(
        default=None,
        description="router and trend_graph: the chosen route (lookup, comparison, trend, report, out_of_scope), the "
                    "LLM's and the rules' choices, who decided and any rule adjustments.",
    )
    trend: dict[str, Any] | None = Field(
        default=None,
        description="Trend route: the per-year series (each figure verified in that year's evidence), year-over-year "
                    "changes, CAGR, missing years and chart rows.",
    )
    sector_trend: dict[str, Any] | None = Field(
        default=None,
        description="Sector trend route: the sector, measure and years; the panel (companies verified in every "
                    "year) and the ones left out with the reason; per year the sector total in USD millions, its "
                    "change, the median company change and how many companies grew; the CAGRs; every company's "
                    "verified figures with their chunks.",
    )
    mutual_fund: dict[str, Any] | None = Field(
        default=None,
        description="Fund route: the fund, holdings date and accession; the lookup kind; the rows shown (top holdings, "
                    "sector mix, exposure or holders); corpus coverage; for questions about the holdings' annual "
                    "reports, each company's verified figures, growth and the weighted growth with its coverage.",
    )
    handoff: dict[str, Any] | None = Field(
        default=None,
        description="Report route: the company and year a full report would cover and whether a report profile exists.",
    )
    agent_run: dict[str, Any] | None = Field(
        default=None,
        description="agentic_graph, trend_graph and router: the node path, step budget, every tool call (coverage_lookup, text_search, "
                    "visual_search, calculator), retrieval attempts, repairs and calculations.",
    )


class RetrieveResponse(ApiResponse):
    domain: str
    display_name: str | None = None
    query: str
    langsmith: LangSmithResponse
    count: int
    evidence: list[EvidenceResponse]


class ApiErrorResponse(BaseModel):
    error: str
    message: str
    details: list[dict[str, str]] | None = None
    request_id: str | None = None


API_ERROR_RESPONSES = {
    404: {"model": ApiErrorResponse, "description": "The requested domain was not found."},
    422: {"model": ApiErrorResponse, "description": "The request failed validation."},
    429: {"model": ApiErrorResponse, "description": "The client exceeded the request rate limit."},
    500: {"model": ApiErrorResponse, "description": "An unexpected server error occurred."},
}
MULTIMODAL_RESPONSES = {
    **API_ERROR_RESPONSES,
    503: {"model": ApiErrorResponse, "description": "The local vision model or visual index is unavailable."},
}
UNAUTHORIZED_RESPONSE = {
    "description": "Missing or invalid API key (only when the server sets API_AUTH_TOKEN).",
    "content": {"application/json": {"schema": {"$ref": "#/components/schemas/ApiErrorResponse"}}},
}


class InMemoryRateLimiter:
    def __init__(
        self,
        *,
        limit: int = RATE_LIMIT_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
        max_clients: int = RATE_LIMIT_MAX_CLIENTS,
    ) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.max_clients = max_clients
        self._requests: OrderedDict[str, deque[float]] = OrderedDict()
        self._lock = Lock()

    def check(self, client_id: str) -> int | None:
        now = time.monotonic()
        with self._lock:
            requests = self._requests.get(client_id)
            if requests is None:
                if len(self._requests) >= self.max_clients:
                    self._requests.popitem(last=False)
                requests = deque()
                self._requests[client_id] = requests
            else:
                self._requests.move_to_end(client_id)

            while requests and now - requests[0] >= self.window_seconds:
                requests.popleft()

            if len(requests) >= self.limit:
                return max(1, int(self.window_seconds - (now - requests[0]) + 0.999))

            requests.append(now)
            return None


app = FastAPI(
    title="Track B Multi-Agent Operations Copilot API",
    version="0.1.0",
    summary="Cited, domain-aware retrieval and Agentic RAG services.",
    description=(
        "FastAPI backend for the Track B Operations Copilot. Provides health and domain discovery, "
        "citation-grounded chat, evidence retrieval, guardrails, human-review routing, and optional "
        "LangSmith tracing. Request and response schemas below are the API contract."
    ),
    openapi_tags=[
        {"name": "System", "description": "Service health and available-domain discovery."},
        {"name": "Chat", "description": "Run a guarded, citation-grounded answer workflow."},
        {"name": "Retrieval", "description": "Retrieve and return evidence without generating an answer."},
        {"name": "Mutual funds", "description": "Loaded SEC N-PORT fund data: funds, holdings, holders and fund-card "
                                                "search (read-only; questions go through the chat routes)."},
        {"name": "Multimodal RAG", "description": "Local multimodal profile: Qdrant text + ColPali page search "
                                                  "and local vision-model answers (`run_local_multimodal`)."},
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.rate_limiter = InMemoryRateLimiter()


def custom_openapi() -> dict[str, Any]:
    """The generated schema plus the API-key security the auth middleware enforces.

    Swagger UI then shows an **Authorize** button: enter the key once and "Try it out" sends it.
    Public paths (``/health``) are marked as needing no key; every other operation documents the 401.
    """
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    schema = get_openapi(title=app.title, version=app.version, summary=app.summary, description=app.description,
                         routes=app.routes, tags=app.openapi_tags)
    schema.setdefault("components", {}).setdefault("securitySchemes", {}).update({
        "ApiKeyHeader": {"type": "apiKey", "in": "header", "name": "X-API-Key",
                         "description": "The value of API_AUTH_TOKEN. Not needed when the server leaves it empty."},
        "BearerAuth": {"type": "http", "scheme": "bearer",
                       "description": "The same key as `Authorization: Bearer <key>`."},
    })
    schema["security"] = [{"ApiKeyHeader": []}, {"BearerAuth": []}]
    for path, operations in schema.get("paths", {}).items():
        for operation in operations.values():
            if path in PUBLIC_PATHS:
                operation["security"] = []
            else:
                operation.setdefault("responses", {}).setdefault("401", UNAUTHORIZED_RESPONSE)
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


def warm_up_retrieval() -> str:
    """Open the served index and load its embedding model once, so the first question isn't the slow one.

    Only the financial ingest backends are warmed (they load a local model and open Qdrant on first use).
    ``API_WARMUP=false`` turns it off. Failures are logged, never raised: a request still retries the load.
    """
    backend = os.getenv("RAG_BACKEND", "")
    if os.getenv("API_WARMUP", "true").strip().lower() in {"0", "false", "no", "off"} or backend not in {"financial_qdrant", "financial_vlm"}:
        return "skipped"
    started = time.perf_counter()
    try:
        if backend == "financial_qdrant":
            from scripts.rag.financial_ingest_qdrant_db import retrieve_for_agent
        else:
            from scripts.rag.financial_ingest_chroma_db import retrieve_for_agent
        retrieve_for_agent("total revenue", top_k=1)
    except Exception as exc:  # noqa: BLE001 - warm-up is best effort
        logger.warning("retrieval warm-up failed: %s", exc)
        return "failed"
    logger.info("retrieval warm-up done in %.1fs (backend=%s)", time.perf_counter() - started, backend)
    return "done"


@app.on_event("startup")
def start_warm_up() -> None:
    import threading
    threading.Thread(target=warm_up_retrieval, name="retrieval-warmup", daemon=True).start()


app.router.route_class = MetricsRoute


def authorized(request: Request) -> bool:
    """True when no API key is configured, the path is public, or the request carries the key."""
    if not API_AUTH_TOKEN or request.url.path in PUBLIC_PATHS or request.method == "OPTIONS":
        return True
    supplied = request.headers.get("X-API-Key") or ""
    bearer = request.headers.get("Authorization") or ""
    if not supplied and bearer.lower().startswith("bearer "):
        supplied = bearer[7:].strip()
    return hmac.compare_digest(supplied.encode("utf-8"), API_AUTH_TOKEN.encode("utf-8"))


class AnswerCache:
    """Answers to identical requests for ``ANSWER_CACHE_SECONDS`` (per process; saves LLM quota on repeats)."""

    NOT_CACHED = {"error", "tool_error", "step_limit"}

    def __init__(self, ttl_seconds: int, max_items: int) -> None:
        self.ttl, self.max = ttl_seconds, max_items
        self.items: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self.lock = Lock()

    @staticmethod
    def key(route: str, request: BaseModel) -> str:
        body = request.model_dump(exclude={"enable_langsmith", "langsmith_project"}, mode="json")
        return hashlib.sha256(json.dumps([route, body], sort_keys=True).encode("utf-8")).hexdigest()

    def get_or_run(self, route: str, request: BaseModel, run) -> dict[str, Any]:
        if self.ttl <= 0:
            return run()
        key, now = self.key(route, request), time.monotonic()
        with self.lock:
            hit = self.items.get(key)
            if hit and hit[0] > now:
                self.items.move_to_end(key)
                return {**json.loads(hit[1]), "cache": "hit"}
        result = run()
        if result.get("final_status") not in self.NOT_CACHED:
            with self.lock:
                self.items[key] = (now + self.ttl, json.dumps(result, default=str))
                while len(self.items) > self.max:
                    self.items.popitem(last=False)
        return result


ANSWER_CACHE = AnswerCache(ANSWER_CACHE_SECONDS, ANSWER_CACHE_MAX)


@app.middleware("http")
async def log_request_and_enforce_rate_limit(request: Request, call_next):
    supplied_request_id = request.headers.get("X-Request-ID")
    try:
        request_id = str(UUID(supplied_request_id)) if supplied_request_id else str(uuid4())
    except ValueError:
        request_id = str(uuid4())

    request_id_token = request_id_context.set(request_id)
    request.state.request_id = request_id
    started_at = time.perf_counter()
    client_id = request.client.host if request.client is not None else "unknown"
    response = None
    try:
        retry_after = request.app.state.rate_limiter.check(client_id)
        if not authorized(request):
            response = JSONResponse(
                status_code=401,
                content=ApiErrorResponse(
                    error="unauthorized",
                    message="Missing or invalid API key: send it as 'X-API-Key' or 'Authorization: Bearer <key>'.",
                    request_id=request_id,
                ).model_dump(exclude_none=True),
                headers={"WWW-Authenticate": "Bearer"},
            )
        elif retry_after is not None:
            response = JSONResponse(
                status_code=429,
                content=ApiErrorResponse(
                    error="rate_limit_exceeded",
                    message="Rate limit exceeded. Please try again later.",
                    details=[
                        {
                            "field": "request",
                            "message": (
                                f"Limit: {RATE_LIMIT_REQUESTS} requests per "
                                f"{RATE_LIMIT_WINDOW_SECONDS} seconds."
                            ),
                            "type": "rate_limit",
                        }
                    ],
                    request_id=request_id,
                ).model_dump(exclude_none=True),
                headers={"Retry-After": str(retry_after)},
            )
        else:
            response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        log_method = logger.warning if response.status_code >= 400 else logger.info
        log_method(
            "api_request_completed request_id=%s method=%s path=%s status_code=%d duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
    except Exception:
        logger.exception(
            "api_request_failed request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        raise
    finally:
        request_id_context.reset(request_id_token)


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    detail = str(exc.detail)
    if exc.status_code == 404 and detail == "Not Found":
        message = "The requested endpoint was not found. Check the route path in the API docs."
        details = [{"field": "path", "message": request.url.path, "type": "unknown_endpoint"}]
    elif exc.status_code == 404 and detail.startswith("Unknown domain:"):
        message = f"{detail}. Use GET /domains to see supported domains."
        details = [{"field": "domain", "message": detail.split(":", 1)[-1].strip(), "type": "unknown_domain"}]
    elif exc.status_code == 405:
        message = "This HTTP method is not supported for the requested endpoint."
        details = [{"field": "method", "message": request.method, "type": "method_not_allowed"}]
    elif exc.status_code == 401:
        message = "Authentication is required to access this resource."
        details = None
    elif exc.status_code == 403:
        message = "You are not authorized to access this resource."
        details = None
    else:
        message = detail
        details = None
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiErrorResponse(
            error="http_error",
            message=message,
            details=details,
            request_id=getattr(request.state, "request_id", None),
        ).model_dump(exclude_none=True),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    details = [
        {
            "field": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=422,
        content=ApiErrorResponse(
            error="validation_error",
            message="Request validation failed.",
            details=details,
            request_id=request_id,
        ).model_dump(exclude_none=True),
        headers={"X-Request-ID": request_id} if request_id else None,
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "api_unhandled_exception request_id=%s method=%s path=%s",
        request_id_context.get(),
        request.method,
        request.url.path,
        exc_info=exc,
    )
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=500,
        content=ApiErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred. Please try again later.",
            request_id=request_id,
        ).model_dump(exclude_none=True),
        headers={"X-Request-ID": request_id} if request_id else None,
    )


def configure_request_tracing(
    *,
    enabled: bool,
    project_name: str | None,
) -> dict[str, Any]:
    tracing = configure_langsmith(
        enabled=enabled or env_flag("LANGSMITH_TRACING"),
        project_name=project_name,
    )
    logger.info(
        "langsmith_trace_configured request_id=%s enabled=%s reason=%s project=%s",
        request_id_context.get(),
        tracing.get("enabled", False),
        tracing.get("reason", "enabled" if tracing.get("enabled") else "disabled"),
        tracing.get("project"),
    )
    return tracing


def trace_request_metadata() -> dict[str, str]:
    request_id = request_id_context.get()
    return {"request_id": request_id} if request_id is not None else {}


def explain_chat_result(result: dict[str, Any]) -> str:
    """Summarize the observable evidence and checks behind a chat outcome."""
    status = str(result.get("final_status") or "unknown")
    grounding = result.get("grounding") or {}
    if status == "not_available":
        return str(grounding.get("reason") or "The filings can't answer this question.")
    if status == "report_handoff":
        return str(grounding.get("reason") or "The request was handed to the report pipeline.")
    if grounding.get("status") == "not_grounded":
        return str(grounding["reason"])
    security = result.get("security") or {}
    plain_text = result.get("plain_text") or {}
    evidence_guard = result.get("evidence_guard") or {}
    output_filter = result.get("output_filter") or {}
    review = result.get("human_review") or {}
    citation_check = result.get("citation_check") or {}

    if status == "refused":
        if security.get("decision") == "block":
            labels = ", ".join(security.get("risk_labels") or [])
            return f"The request was refused by the input security checks ({labels or 'blocked input'})."
        if plain_text and not plain_text.get("allowed", True):
            reason = plain_text.get("reason")
            return f"The request was refused by the domain-intent checks: {reason}" if reason else (
                "The request was refused because it did not pass the domain-intent checks."
            )
        if evidence_guard and not evidence_guard.get("safe", True):
            return "The request was refused because retrieved evidence failed the safety checks."
        if output_filter.get("decision") == "block":
            return "The generated response was withheld because it failed output safety checks."
        return "The workflow did not produce an answer because a safety or policy check refused the request."

    if review.get("required"):
        triggers = ", ".join(review.get("triggers") or [])
        return (
            "The answer is routed for human review"
            f"{f' because of: {triggers}' if triggers else ''}."
        )

    if not result.get("answer"):
        return "No answer was produced by the workflow."

    evidence = result.get("evidence") or result.get("evidence_preview") or []
    cited_ids = citation_check.get("cited_chunk_ids") or result.get("evidence_chunk_ids") or []
    if citation_check.get("valid") and cited_ids:
        return (
            f"The answer is based on {len(evidence)} retrieved evidence item(s); "
            f"citation validation passed for: {', '.join(str(item) for item in cited_ids)}."
        )
    if not evidence:
        return "The workflow found no retrieved evidence to support the answer."
    return (
        f"The answer used {len(evidence)} retrieved evidence item(s), "
        "but citation validation did not confirm its citations."
    )


def apply_grounding_policy(result: dict[str, Any]) -> dict[str, Any]:
    """Withhold answers unless citations validate against retrieved evidence IDs.

    Keeps how the agent grounded the answer (``method``, ``semantic_answer_verified``: the critic's support
    check) next to the citation-ID status this policy adds.
    """
    prior = {key: value for key, value in (result.get("grounding") or {}).items()
             if key in ("method", "semantic_answer_verified")}
    result = _citation_grounding(result)
    verified = bool(prior.get("semantic_answer_verified")) and result["grounding"]["status"] == "grounded"
    result["grounding"] = {**result["grounding"], **prior, "semantic_answer_verified": verified}
    if verified:
        result["grounding"]["reason"] = ("Citations matched retrieved evidence, and the critic's support check found "
                                         "every claim and figure stated in the cited chunks.")
    return result


def _citation_grounding(result: dict[str, Any]) -> dict[str, Any]:
    """Withhold answers unless citations validate against retrieved evidence IDs."""
    if result.get("final_status") == "refused":
        result["grounding"] = {
            "status": "not_applicable",
            "evidence_count": 0,
            "cited_chunk_ids": [],
            "reason": "The request was refused before an answer was provided.",
        }
        return result
    if result.get("final_status") == "report_handoff":
        result["grounding"] = {
            "status": "not_applicable",
            "evidence_count": 0,
            "cited_chunk_ids": [],
            "reason": "The router handed a report request to the report pipeline, which runs after confirmation.",
        }
        return result
    if result.get("final_status") == "not_available" and (result.get("route") or {}).get("route") == "fund":
        result["grounding"] = {
            "status": "not_applicable",
            "evidence_count": 0,
            "cited_chunk_ids": [],
            "reason": "The fund agent could not answer from the loaded N-PORT data; the answer says why.",
        }
        return result
    if result.get("final_status") == "not_available":
        coverage = result.get("coverage") or {}
        result["grounding"] = {
            "status": "not_applicable",
            "evidence_count": 0,
            "cited_chunk_ids": [],
            "reason": f"No figure was given because the filings can't answer this ({coverage.get('reason', 'not available')}).",
        }
        return result
    if result.get("final_status") == "insufficient_evidence" and result.get("sector_trend"):
        # The sector agent stopped before answering (too few companies verified in every year): its message
        # states only what could not be verified, so it stays; the per-company table shows what was found.
        result["grounding"] = {
            "status": "not_applicable",
            "evidence_count": 0,
            "cited_chunk_ids": [],
            "reason": "No sector trend was computed: too few companies had a verified figure for every year.",
        }
        return result

    evidence = result.get("evidence") or result.get("evidence_preview") or []
    evidence_ids = {
        str(item["chunk_id"])
        for item in evidence
        if isinstance(item, dict) and item.get("chunk_id") is not None
    }
    evidence_ids.update(
        str(chunk_id) for chunk_id in result.get("evidence_chunk_ids", []) if chunk_id is not None
    )
    citation_check = result.get("citation_check") or {}
    cited_ids = {
        str(chunk_id)
        for chunk_id in citation_check.get("cited_chunk_ids", [])
        if chunk_id is not None
    }
    allowed_ids = {
        str(chunk_id)
        for chunk_id in citation_check.get("allowed_chunk_ids", [])
        if chunk_id is not None
    }
    valid = (
        bool(result.get("answer"))
        and bool(evidence_ids)
        and citation_check.get("valid") is True
        and bool(cited_ids)
        and cited_ids <= evidence_ids
        and (not allowed_ids or cited_ids <= allowed_ids)
    )
    if valid:
        result["grounding"] = {
            "status": "grounded",
            "evidence_count": len(evidence_ids),
            "cited_chunk_ids": sorted(cited_ids),
            "reason": "Citation validation matched the answer citations to retrieved evidence.",
        }
        return result

    result["answer"] = None
    result["final_status"] = "not_grounded"
    company_check = result.get("company_check") or {}
    if company_check.get("valid") is False:
        reason = "The answer was withheld because it cited filings from a company the question didn't ask about."
    elif not evidence_ids:
        reason = "The answer was withheld because no retrieved evidence was available."
    elif not cited_ids:
        reason = "The answer was withheld because it did not cite any retrieved evidence."
    else:
        reason = "The answer was withheld because its citations did not validate against retrieved evidence."
    result["grounding"] = {
        "status": "not_grounded",
        "evidence_count": len(evidence_ids),
        "cited_chunk_ids": sorted(cited_ids),
        "reason": reason,
    }
    return result


def financial_config_from_request(request: FinancialChatRequest | RetrieveRequest) -> AgentConfig:
    return AgentConfig(
        query=request.query,
        ticker=getattr(request, "ticker", None),
        year=getattr(request, "year", None),
        sector=getattr(request, "sector", None),
        source_type=request.source_type,
        top_k=request.top_k,
        dense_top_k=request.dense_top_k,
        bm25_top_k=request.bm25_top_k,
        retrieval_mode=getattr(request, "retrieval_mode", "hybrid"),
        embedding_provider=request.embedding_provider,
        hash_dimensions=request.hash_dimensions,
        max_context_chars=getattr(request, "max_context_chars", 12000),
        self_consistency_samples=getattr(request, "self_consistency_samples", 3),
        enable_guardrails=getattr(request, "enable_guardrails", True),
        enable_human_review=getattr(request, "enable_human_review", True),
    )


@app.get(
    "/health",
    summary="Check API health",
    description="Returns the service status and the domain identifiers accepted by domain-specific routes.",
    tags=["System"],
    response_model=HealthResponse,
    response_model_exclude_unset=True,
    responses=API_ERROR_RESPONSES,
)
def health() -> HealthResponse:
    return {
        "status": "ok",
        "service": "capstone-agentic-rag-api",
        "domains": list(ALL_DOMAINS),
        "runtime": describe_runtime(),
        "auth_required": bool(API_AUTH_TOKEN),
        "conversation_store": "sqlite" if CONVERSATIONS.path else "memory",
        "agent_max_steps": max_steps(),
        "answer_cache_seconds": ANSWER_CACHE.ttl,
        "defaults": {"llm_provider": DEFAULT_LLM_PROVIDER, "embedding_provider": DEFAULT_EMBEDDING_PROVIDER},
    }


@app.get(
    "/domains",
    summary="List available domains",
    description="Returns the domain this service answers (financial report QA) with its display metadata.",
    tags=["System"],
    response_model=DomainsResponse,
    response_model_exclude_unset=True,
    responses=API_ERROR_RESPONSES,
)
def domains() -> DomainsResponse:
    return {"domains": [{
        "domain": FINANCIAL_DOMAIN,
        "display_name": "Financial Report Copilot",
        "description": "Cited answers, comparisons and checked calculations from SEC annual reports.",
        "default_query": "What drove revenue growth in the latest annual report?",
    }]}


@app.get(
    "/companies",
    summary="List the companies the financial corpus covers",
    description=(
        "Tickers with company name, sector and the fiscal years that have a filing in the corpus or the served "
        "index. Optional `sector` (names or slugs, comma-separated) narrows the list. The UI builds its ticker "
        "and year pickers from this."
    ),
    tags=["System"],
    response_model=ApiResponse,
    responses=API_ERROR_RESPONSES,
)
def companies(sector: str | None = None) -> dict[str, Any]:
    wanted = parse_sectors(sector)
    rows = [
        {"ticker": ticker, "company": COMPANY_NAMES.get(ticker, ticker), "sector": SECTOR_OF.get(ticker),
         "years": years}
        for ticker, years in get_clarifier().directory.items()
        if not wanted or sector_slug(SECTOR_OF.get(ticker) or "") in wanted
    ]
    return {"count": len(rows), "sectors": list(SECTOR_NAMES), "companies": rows}


@app.post(
    "/chat/financial_report_qa",
    summary="Answer a financial-report question",
    description=(
        "Runs the selected financial QA technique over annual-report evidence. "
        "Optional ticker and year fields constrain the source evidence. The response includes "
        "a concise explanation of its evidence and citation checks."
    ),
    tags=["Chat"],
    response_model=FinancialChatResponse,
    response_model_exclude_unset=True,
    responses=API_ERROR_RESPONSES,
)
def chat_financial(request: FinancialChatRequest) -> dict[str, Any]:
    return ANSWER_CACHE.get_or_run("chat_financial", request, lambda: _chat_financial(request))


def _chat_financial(request: FinancialChatRequest) -> dict[str, Any]:
    langsmith = configure_request_tracing(
        enabled=request.enable_langsmith,
        project_name=request.langsmith_project or "capstone-financial-report-qa-api",
    )
    config = financial_config_from_request(request)
    result = run_traced(
        f"api.financial_report_qa.{request.technique}",
        run_technique,
        request.technique,
        request.llm_provider,
        config,
        run_type="chain",
        tags=[
            "capstone",
            "fastapi",
            FINANCIAL_DOMAIN,
            request.technique,
            request.llm_provider,
            request.embedding_provider,
        ],
        metadata={
            "domain": FINANCIAL_DOMAIN,
            "top_k": request.top_k,
            "dense_top_k": request.dense_top_k,
            "bm25_top_k": request.bm25_top_k,
            "retrieval_mode": request.retrieval_mode,
            "guardrails_enabled": request.enable_guardrails,
            "human_review_enabled": request.enable_human_review,
            **trace_request_metadata(),
        },
    )
    summarized_result = apply_grounding_policy(summarize_result(result))
    return {
        "domain": FINANCIAL_DOMAIN,
        "display_name": "Financial Report QA",
        "langsmith": langsmith,
        **summarized_result,
        "explanation": explain_chat_result(summarized_result),
    }


CONVERSATIONS = ConversationStore(ttl_seconds=env_int("CONVERSATION_TTL_SECONDS", 1800),
                                  path=os.getenv("CONVERSATION_DB") or None)
_clarifier: Clarifier | None = None


def get_clarifier() -> Clarifier:
    global _clarifier
    if _clarifier is None:
        _clarifier = Clarifier()
    return _clarifier


def run_financial_report(spec: dict[str, Any]) -> dict[str, Any]:
    """Run the deterministic report pipeline for a confirmed report request."""
    clarifier = get_clarifier()
    if not spec.get("profile"):
        available = ", ".join(clarifier.available_reports()) or "none yet"
        return {"status": "report_unavailable", "message": (
            f"A full automated report for {company_name(spec['ticker'])} FY{spec['year']} is not available yet: "
            f"report profiles exist for {available}. You can still ask for specific figures, trends or "
            "explanations for this company.")}
    from scripts.reports.generate_financial_report import generate
    try:
        manifest = json.loads(Path(generate(spec["profile"])).read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - reported to the user, details in the report trace.
        logger.warning("report_generation_failed error_type=%s", type(exc).__name__)
        return {"status": "report_failed", "message": f"Report generation failed ({type(exc).__name__})."}
    return {"status": "report_generated", "report_manifest": manifest, "message": (
        f"Report generated for {manifest['company']} FY{manifest['year']} ({manifest.get('pdf_pages')} pages, "
        f"{manifest.get('checks_passed')} reconciliation checks passed). It is pending visual review before publishing.")}


@app.post(
    "/conversation/financial_report_qa",
    summary="Chat turn with clarification",
    description=(
        "Checks whether the question names a company, fiscal year and measure that the corpus covers. "
        "If something is missing, ambiguous or not available it returns `needs_clarification` with questions and "
        "options, plus `coverage` with the reason when the filings can't answer it, and "
        "keeps the partial request under `conversation_id` (30-minute TTL; in SQLite when `CONVERSATION_DB` is set, "
        "so it survives restarts, otherwise in memory). "
        "A complete question runs the financial agents; a report request runs the report pipeline after "
        "the user confirms. Advice, prediction and prompt-injection messages go straight to the guarded agent."
    ),
    tags=["Chat"],
    response_model=ApiResponse,
    responses=API_ERROR_RESPONSES,
)
def converse_financial(request: ConversationRequest) -> dict[str, Any]:
    conversation = CONVERSATIONS.get(request.conversation_id)
    if request.clarify:
        outcome = get_clarifier().step(conversation, request.query, ticker=request.ticker, year=request.year)
        CONVERSATIONS.save(conversation)
    else:
        outcome = {"status": "ready", "action": "answer", "conversation_id": conversation.conversation_id,
                   "resolved_query": request.query, "ticker": request.ticker, "year": request.year, "questions": []}
    if outcome["status"] in {"needs_clarification", "cancelled"}:
        return outcome
    if outcome["action"] == "report":
        return {**outcome, "clarifier": outcome["status"], **run_financial_report(outcome["report"])}
    chat_request = FinancialChatRequest(**{
        **request.model_dump(exclude={"conversation_id", "clarify"}),
        "query": outcome["resolved_query"], "ticker": outcome["ticker"], "year": outcome["year"]})
    result = chat_financial(chat_request)
    return {**outcome, "clarifier": outcome["status"], "status": "answered", "result": result}


@app.post(
    "/retrieve/financial_report_qa",
    summary="Retrieve financial-report evidence",
    description="Searches annual-report evidence and returns ranked records without generating an answer.",
    tags=["Retrieval"],
    response_model=RetrieveResponse,
    response_model_exclude_unset=True,
    responses=API_ERROR_RESPONSES,
)
def retrieve_financial(request: RetrieveRequest) -> dict[str, Any]:
    langsmith = configure_request_tracing(
        enabled=request.enable_langsmith,
        project_name=request.langsmith_project or "capstone-financial-retrieval-api",
    )
    config = financial_config_from_request(request)
    hits = run_traced(
        "api.financial_report_qa.retrieve",
        retrieve_evidence,
        config,
        run_type="retriever",
        tags=["capstone", "fastapi", FINANCIAL_DOMAIN, request.embedding_provider, request.retrieval_mode],
        metadata={
            "domain": FINANCIAL_DOMAIN,
            "top_k": request.top_k,
            "dense_top_k": request.dense_top_k,
            "bm25_top_k": request.bm25_top_k,
            "source_type": request.source_type,
            "ticker": request.ticker,
            "year": request.year,
            "sector": request.sector,
            **trace_request_metadata(),
        },
    )
    return {
        "domain": FINANCIAL_DOMAIN,
        "query": request.query,
        "langsmith": langsmith,
        "count": len(hits),
        "evidence": [
            {
                "chunk_id": hit.get("chunk_id"),
                "source_type": hit.get("source_type"),
                "title": hit.get("title"),
                "ticker": hit.get("ticker"),
                "year": hit.get("year"),
                "sector": hit.get("sector"),
                "page": hit.get("page"),
                "source_url": hit.get("source_url"),
                "image_path": hit.get("image_path"),
                "rank": hit.get("rank"),
                "score": hit.get("rerank_score"),
                "retrieval_ranks": hit.get("retrieval_ranks"),
                "preview": hit.get("preview"),
            }
            for hit in hits
        ],
    }


@app.post(
    "/multimodal/{domain}/retrieve",
    tags=["Multimodal RAG"],
    summary="Retrieve multimodal evidence",
    description="Hybrid dense + sparse search over the local Qdrant `docs_text` collection (text and OCR'd "
                "pages). `final_status` is `evidence_found`, `insufficient_evidence` or `tool_error`.",
    responses=MULTIMODAL_RESPONSES,
)
def retrieve_multimodal(domain: str, request: MultimodalRequest) -> dict[str, Any]:
    if domain not in ALL_DOMAINS:
        raise HTTPException(status_code=404, detail="Unknown domain")
    from scripts.rag.multimodal_rag import retrieve_local
    from scripts.common.resilience import Calls, CallFailure
    calls = Calls(request.timeout_seconds)
    try:
        hits = calls.run("retrieve", lambda: retrieve_local(domain, request.query, request.top_k,
                         source_type=request.source_type, ticker=request.ticker, year=request.year))
        return {"domain": domain, "count": len(hits), "evidence": hits,
                "trace_id": calls.trace_id, "trace": calls.events,
                "final_status": "evidence_found" if hits else "insufficient_evidence"}
    except CallFailure:
        return {"domain": domain, "count": 0, "evidence": [], "final_status": "tool_error",
                "warnings": ["retrieval_unavailable"], "trace_id": calls.trace_id, "trace": calls.events}


@app.post(
    "/multimodal/{domain}/chat",
    tags=["Multimodal RAG"],
    summary="Answer with the local multimodal pipeline",
    description="Input guard, retrieval, then a fail-closed answer from the configured provider "
                "(`extractive`, `ollama`, `transformers` or `llamacpp`). Unsafe input returns `refused`.",
    responses=MULTIMODAL_RESPONSES,
)
def chat_multimodal(domain: str, request: MultimodalRequest) -> dict[str, Any]:
    if domain not in ALL_DOMAINS:
        raise HTTPException(status_code=404, detail="Unknown domain")
    from scripts.rag.multimodal_rag import LOCK, get_rag
    guard = deterministic_input_guard(request.query)
    if guard["decision"] != "allow":
        return {"domain": domain, "final_status": "refused", "security": guard,
                "answer": "The query requires review.", "evidence": []}
    try:
        from scripts.common.resilience import timed_lock
        started = time.monotonic()
        with timed_lock(LOCK, request.timeout_seconds):
            return get_rag().answer(domain, guard["sanitized_query"], request.top_k,
                                    request.provider, source_type=request.source_type,
                                    ticker=request.ticker, year=request.year,
                                    idempotency_key=request.idempotency_key,
                                    timeout_seconds=max(.001, request.timeout_seconds - (time.monotonic() - started)))
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Local vision service unavailable; check the configured model and runtime logs.") from exc


@app.post(
    "/multimodal/{domain}/visual/retrieve",
    tags=["Multimodal RAG"],
    summary="Retrieve page images with ColPali",
    description="ColPali MaxSim search over the `visual_pages` collection. Needs the ColPali model and a "
                "visual index (`colpali_index backfill`).",
    responses=MULTIMODAL_RESPONSES,
)
def retrieve_visual_multimodal(domain: str, request: MultimodalRequest) -> dict[str, Any]:
    if domain not in ALL_DOMAINS:
        raise HTTPException(status_code=404, detail="Unknown domain")
    from scripts.rag.multimodal_rag import LOCK, get_rag
    from scripts.common.resilience import Calls, CallFailure
    calls = Calls(request.timeout_seconds)
    def retrieve():
        from scripts.common.resilience import timed_lock
        with timed_lock(LOCK, request.timeout_seconds):
            return get_rag().retrieve_visual(domain, request.query, request.top_k,
                source_type=request.source_type, ticker=request.ticker, year=request.year)
    try:
        hits = calls.run("visual_retrieve", retrieve)
        return {"domain": domain, "retriever": "colpali_maxsim", "count": len(hits), "evidence": hits,
                "trace_id": calls.trace_id, "trace": calls.events}
    except (ValueError, OSError, CallFailure) as exc:
        raise HTTPException(status_code=503, detail="ColPali retrieval unavailable; check model configuration and visual index.") from exc


# Mutual fund endpoints (read-only; 404 until fund data is loaded). Kept in their own module.
from scripts.api.routes_mutual_fund import router as mutual_fund_router  # noqa: E402 - after the app is built

app.include_router(mutual_fund_router)
