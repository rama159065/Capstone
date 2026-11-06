"""FastAPI backend for the Track B Agentic RAG capstone.

Run locally:
    uvicorn scripts.api.api_app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

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

from scripts.domains.competitive_research.domain_config import DOMAIN_CONFIG as COMPETITIVE_CONFIG
from scripts.observability.langsmith_utils import configure_langsmith, env_flag, run_traced
from scripts.agents.rag_agent import (
    DomainConfig,
    retrieve_hybrid,
    run_domain_agent,
    summarize_evidence,
)
from scripts.common.runtime_profile import apply_profile, describe as describe_runtime
from scripts.agents.financial_clarifier import Clarifier, ConversationStore, name as company_name
from scripts.agents.financial_prompting_agent import (
    AgentConfig,
    retrieve_evidence,
    run_technique,
    summarize_result,
)
from scripts.domains.hr_policy.domain_config import DOMAIN_CONFIG as HR_POLICY_CONFIG
from scripts.domains.incident_postmortem.domain_config import DOMAIN_CONFIG as INCIDENT_CONFIG
from scripts.domains.procurement_vendor.domain_config import DOMAIN_CONFIG as PROCUREMENT_CONFIG
from scripts.domains.travel_expense.domain_config import DOMAIN_CONFIG as TRAVEL_CONFIG


NON_FINANCIAL_DOMAINS: dict[str, DomainConfig] = {
    config.domain: config
    for config in [
        PROCUREMENT_CONFIG,
        TRAVEL_CONFIG,
        COMPETITIVE_CONFIG,
        INCIDENT_CONFIG,
        HR_POLICY_CONFIG,
    ]
}
FINANCIAL_DOMAIN = "financial_report_qa"
ALL_DOMAINS = (FINANCIAL_DOMAIN, *NON_FINANCIAL_DOMAINS.keys())
logger = logging.getLogger(__name__)
RUNTIME = apply_profile()   # local or colab defaults, before any retrieval runs
request_id_context: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "api_request_id", default=None
)
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_CLIENTS = 10_000


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
        "mock",
        description="Chat backend. Hosted mode requires the LLM endpoint environment variables.",
    )
    embedding_provider: Literal["hashing", "hosted"] = Field(
        "hashing",
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
    ] = Field("agentic_graph", description="Financial QA workflow to execute.")
    retrieval_mode: Literal["dense", "hybrid"] = Field(
        "hybrid", description="Use dense-only or hybrid dense-plus-BM25 retrieval."
    )
    ticker: str | None = Field(None, description="Optional company ticker to filter financial evidence.")
    year: int | None = Field(None, description="Optional fiscal/reporting year filter.")
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
    retrieval_mode: Literal["dense", "hybrid"] = Field(
        "hybrid", description="Financial retrieval mode; non-financial domains always use hybrid search."
    )
    ticker: str | None = Field(None, description="Optional company ticker filter for financial evidence.")
    year: int | None = Field(None, description="Optional fiscal/reporting year filter.")
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
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.rate_limiter = InMemoryRateLimiter()


app.router.route_class = MetricsRoute


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
        if retry_after is not None:
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
    """Withhold answers unless citations validate against retrieved evidence IDs."""
    if result.get("final_status") == "refused":
        result["grounding"] = {
            "status": "not_applicable",
            "evidence_count": 0,
            "cited_chunk_ids": [],
            "reason": "The request was refused before an answer was provided.",
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


def get_domain_config(domain: str) -> DomainConfig:
    try:
        return NON_FINANCIAL_DOMAINS[domain]
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown domain: {domain}") from exc


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
    }


@app.get(
    "/domains",
    summary="List available domains",
    description="Returns display metadata and default questions for financial and non-financial domains.",
    tags=["System"],
    response_model=DomainsResponse,
    response_model_exclude_unset=True,
    responses=API_ERROR_RESPONSES,
)
def domains() -> DomainsResponse:
    rows = [
        {
            "domain": FINANCIAL_DOMAIN,
            "display_name": "Financial Report QA",
            "description": "Cited financial-report Agentic RAG over annual-report PDFs.",
            "default_query": "What does Apple say can affect quarterly net sales?",
        }
    ]
    rows.extend(
        {
            "domain": config.domain,
            "display_name": config.display_name,
            "description": config.description,
            "default_query": config.default_query,
            "source_types": list(config.source_types),
        }
        for config in NON_FINANCIAL_DOMAINS.values()
    )
    return {"domains": rows}


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


CONVERSATIONS = ConversationStore()
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
        "keeps the partial request under `conversation_id` (in memory, per process, 30-minute TTL). "
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
    "/chat/{domain}",
    summary="Answer a domain question",
    description=(
        "Runs the shared guarded Agentic RAG workflow for a non-financial domain. "
        "Supported domain identifiers are returned by `GET /domains`. "
        "Financial report QA uses the dedicated `/chat/financial_report_qa` operation. "
        "The response includes a concise explanation of its evidence and workflow checks."
    ),
    tags=["Chat"],
    response_model=ChatResponse,
    response_model_exclude_unset=True,
    responses=API_ERROR_RESPONSES,
)
def chat_domain(domain: str, request: ChatRequest) -> dict[str, Any]:
    if domain == FINANCIAL_DOMAIN:
        return chat_financial(FinancialChatRequest(**request.model_dump()))
    domain_config = get_domain_config(domain)
    langsmith = configure_request_tracing(
        enabled=request.enable_langsmith,
        project_name=request.langsmith_project or f"capstone-{domain}-api",
    )
    result = run_traced(
        f"api.{domain}.agent",
        run_domain_agent,
        domain_config,
        run_type="chain",
        tags=[
            "capstone",
            "fastapi",
            domain,
            request.llm_provider,
            request.embedding_provider,
        ],
        metadata={
            "domain": domain,
            "top_k": request.top_k,
            "dense_top_k": request.dense_top_k,
            "bm25_top_k": request.bm25_top_k,
            "source_type": request.source_type,
            "guardrails_enabled": request.enable_guardrails,
            "human_review_enabled": request.enable_human_review,
            **trace_request_metadata(),
        },
        query=request.query,
        llm_provider=request.llm_provider,
        embedding_provider=request.embedding_provider,
        hash_dimensions=request.hash_dimensions,
        top_k=request.top_k,
        dense_top_k=request.dense_top_k,
        bm25_top_k=request.bm25_top_k,
        source_type=request.source_type,
        max_context_chars=request.max_context_chars,
        enable_guardrails=request.enable_guardrails,
        enable_human_review=request.enable_human_review,
    )
    result["langsmith"] = langsmith
    apply_grounding_policy(result)
    result["explanation"] = explain_chat_result(result)
    return result


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
                "page": hit.get("page"),
                "source_url": hit.get("source_url"),
                "image_path": hit.get("image_path"),
                "rank": hit.get("rank"),
                "score": hit.get("rerank_score"),
                "preview": hit.get("preview"),
            }
            for hit in hits
        ],
    }


@app.post(
    "/retrieve/{domain}",
    summary="Retrieve domain evidence",
    description=(
        "Searches a non-financial domain's corpus and returns ranked evidence records. "
        "Supported domain identifiers are returned by `GET /domains`; financial retrieval uses "
        "the dedicated `/retrieve/financial_report_qa` operation."
    ),
    tags=["Retrieval"],
    response_model=RetrieveResponse,
    response_model_exclude_unset=True,
    responses=API_ERROR_RESPONSES,
)
def retrieve_domain(domain: str, request: RetrieveRequest) -> dict[str, Any]:
    if domain == FINANCIAL_DOMAIN:
        return retrieve_financial(request)
    domain_config = get_domain_config(domain)
    langsmith = configure_request_tracing(
        enabled=request.enable_langsmith,
        project_name=request.langsmith_project or f"capstone-{domain}-retrieval-api",
    )
    hits = run_traced(
        f"api.{domain}.retrieve",
        retrieve_hybrid,
        domain_config,
        run_type="retriever",
        tags=["capstone", "fastapi", domain, request.embedding_provider],
        metadata={
            "domain": domain,
            "top_k": request.top_k,
            "dense_top_k": request.dense_top_k,
            "bm25_top_k": request.bm25_top_k,
            "source_type": request.source_type,
            **trace_request_metadata(),
        },
        query=request.query,
        top_k=request.top_k,
        dense_top_k=request.dense_top_k,
        bm25_top_k=request.bm25_top_k,
        embedding_provider=request.embedding_provider,
        hash_dimensions=request.hash_dimensions,
        source_type=request.source_type,
    )
    return {
        "domain": domain,
        "display_name": domain_config.display_name,
        "query": request.query,
        "langsmith": langsmith,
        "count": len(hits),
        "evidence": summarize_evidence(hits),
    }


@app.post("/multimodal/{domain}/retrieve", tags=["Multimodal RAG"])
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


@app.post("/multimodal/{domain}/chat", tags=["Multimodal RAG"])
def chat_multimodal(domain: str, request: MultimodalRequest) -> dict[str, Any]:
    if domain not in ALL_DOMAINS:
        raise HTTPException(status_code=404, detail="Unknown domain")
    from scripts.agents.rag_agent import input_guard
    from scripts.rag.multimodal_rag import LOCK, get_rag
    guard = input_guard(request.query)
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


@app.post("/multimodal/{domain}/visual/retrieve", tags=["Multimodal RAG"])
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
