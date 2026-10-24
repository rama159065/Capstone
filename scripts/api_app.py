"""FastAPI backend for the Track B Agentic RAG capstone.

Run locally:
    uvicorn scripts.api_app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict, deque
from threading import Lock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from scripts.competitive_research.domain_config import DOMAIN_CONFIG as COMPETITIVE_CONFIG
from scripts.domain_common.langsmith_utils import configure_langsmith, env_flag, run_traced
from scripts.domain_common.rag_agent import (
    DomainConfig,
    retrieve_hybrid,
    run_domain_agent,
    summarize_evidence,
)
from scripts.financial_prompting_agent import (
    AgentConfig,
    retrieve_evidence,
    run_technique,
    summarize_result,
)
from scripts.hr_policy.domain_config import DOMAIN_CONFIG as HR_POLICY_CONFIG
from scripts.incident_postmortem.domain_config import DOMAIN_CONFIG as INCIDENT_CONFIG
from scripts.procurement_vendor.domain_config import DOMAIN_CONFIG as PROCUREMENT_CONFIG
from scripts.travel_expense.domain_config import DOMAIN_CONFIG as TRAVEL_CONFIG


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
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_CLIENTS = 10_000


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    llm_provider: Literal["mock", "hosted"] = "mock"
    embedding_provider: Literal["hashing", "hosted"] = "hashing"
    hash_dimensions: int = Field(384, ge=32, le=4096)
    top_k: int = Field(5, ge=1, le=20)
    dense_top_k: int = Field(30, ge=1, le=100)
    bm25_top_k: int = Field(30, ge=1, le=100)
    source_type: str | None = None
    max_context_chars: int = Field(12000, ge=1000, le=64000)
    enable_guardrails: bool = True
    enable_human_review: bool = True
    enable_langsmith: bool = False
    langsmith_project: str | None = None


class FinancialChatRequest(ChatRequest):
    technique: Literal[
        "baseline_rag",
        "structured_few_shot",
        "decomposition",
        "self_consistency",
        "agentic_graph",
    ] = "agentic_graph"
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    ticker: str | None = None
    year: int | None = None
    self_consistency_samples: int = Field(3, ge=1, le=5)


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    embedding_provider: Literal["hashing", "hosted"] = "hashing"
    hash_dimensions: int = Field(384, ge=32, le=4096)
    top_k: int = Field(5, ge=1, le=20)
    dense_top_k: int = Field(30, ge=1, le=100)
    bm25_top_k: int = Field(30, ge=1, le=100)
    source_type: str | None = None
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    ticker: str | None = None
    year: int | None = None
    enable_langsmith: bool = False
    langsmith_project: str | None = None


class ApiResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


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


class ChatResponse(ApiResponse):
    domain: str
    display_name: str | None = None
    final_status: str
    query: str | None = None
    safe_query: str | None = None
    answer: str | None = None
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
    description="FastAPI backend for LangGraph Agentic RAG, hybrid retrieval, guardrails, HITL, and LangSmith tracing.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.rate_limiter = InMemoryRateLimiter()


@app.middleware("http")
async def enforce_rate_limit(request: Request, call_next):
    client_id = request.client.host if request.client is not None else "unknown"
    retry_after = request.app.state.rate_limiter.check(client_id)
    if retry_after is not None:
        return JSONResponse(
            status_code=429,
            content=ApiErrorResponse(
                error="rate_limit_exceeded",
                message="Rate limit exceeded. Please try again later.",
            ).model_dump(exclude_none=True),
            headers={"Retry-After": str(retry_after)},
        )
    return await call_next(request)


@app.exception_handler(StarletteHTTPException)
async def handle_http_exception(
    request: Request,
    exc: StarletteHTTPException,
) -> JSONResponse:
    del request
    return JSONResponse(
        status_code=exc.status_code,
        content=ApiErrorResponse(
            error="http_error",
            message=str(exc.detail),
        ).model_dump(exclude_none=True),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    del request
    details = [
        {
            "field": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=ApiErrorResponse(
            error="validation_error",
            message="Request validation failed.",
            details=details,
        ).model_dump(exclude_none=True),
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "Unhandled API exception for %s %s",
        request.method,
        request.url.path,
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content=ApiErrorResponse(
            error="internal_server_error",
            message="An unexpected error occurred. Please try again later.",
        ).model_dump(exclude_none=True),
    )


def configure_request_tracing(
    *,
    enabled: bool,
    project_name: str | None,
) -> dict[str, Any]:
    return configure_langsmith(
        enabled=enabled or env_flag("LANGSMITH_TRACING"),
        project_name=project_name,
    )


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
    response_model=HealthResponse,
    response_model_exclude_unset=True,
    responses=API_ERROR_RESPONSES,
)
def health() -> HealthResponse:
    return {
        "status": "ok",
        "service": "capstone-agentic-rag-api",
        "domains": list(ALL_DOMAINS),
    }


@app.get(
    "/domains",
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
        },
    )
    return {
        "domain": FINANCIAL_DOMAIN,
        "display_name": "Financial Report QA",
        "langsmith": langsmith,
        **summarize_result(result),
    }


@app.post(
    "/chat/{domain}",
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
    return result


@app.post(
    "/retrieve/financial_report_qa",
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
