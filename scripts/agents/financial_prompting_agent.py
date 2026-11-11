"""Prompting and agentic workflows for the financial-report RAG corpus.

Track B usage with hosted OpenAI-compatible <=8B endpoint:
    set LLM_API_BASE=https://your-hosted-endpoint/v1
    set LLM_API_KEY=...
    set LLM_MODEL=your-chat-model
    python scripts\\agents\\financial_prompting_agent.py --llm-provider hosted --technique agentic_graph --query "What does Apple say can affect quarterly net sales?" --ticker AAPL

Offline smoke test:
    python scripts\\agents\\financial_prompting_agent.py --llm-provider mock --technique all --query "What does Apple say can affect quarterly net sales?" --ticker AAPL
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common import hosted_llm
from scripts.rag.hybrid import matched_by
from scripts.common.agent_limits import STEP_LIMIT_ANSWER, graph_config, max_steps
from scripts.agents.financial_clarifier import question_scope
from scripts.guardrails import checks as guard_checks, patterns as guard_patterns
from scripts.prompts import financial as prompts
from scripts.observability.langsmith_utils import (
    add_langsmith_args,
    maybe_enable_from_args,
    run_traced,
)


DEFAULT_CHUNKS = Path("data/processed/financial_report_qa/pdf_chunks.jsonl")
DEFAULT_INDEX_DIR = Path("indexes/chroma/financial_report_qa_pdfs")
DEFAULT_COLLECTION = "financial_report_qa_pdfs"


@dataclass
class AgentConfig:
    query: str
    ticker: str | None
    year: int | None
    source_type: str | None
    top_k: int
    dense_top_k: int
    bm25_top_k: int
    retrieval_mode: str
    embedding_provider: str
    hash_dimensions: int
    max_context_chars: int
    self_consistency_samples: int
    enable_guardrails: bool = True
    enable_human_review: bool = True
    max_query_chars: int = 4000
    sector: str | None = None        # e.g. "banking" or "Oil and gas"; only the sector-corpus backends filter on it
    visual: bool | None = None       # ColPali page search: None = RAG_FIN_VISUAL decides (financial_qdrant only)


class MockResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class MockLLM:
    """Deterministic local LLM stand-in for pipeline testing."""

    def invoke(self, messages: list[Any]) -> MockResponse:
        joined = "\n".join(getattr(message, "content", str(message)) for message in messages)
        if "Support check JSON" in joined:
            # Mock judge: supported when the answer cites a chunk that is in the cited evidence. Not a real check.
            answer_part = joined.split("Answer to check:", 1)[-1].split("Cited evidence:", 1)[0]
            evidence_part = joined.split("Cited evidence:", 1)[-1]
            evidence_ids = set(re.findall(r"chunk_id:\s*([A-Za-z0-9_.=-]+)", evidence_part))
            ids = sorted(chunk_id for chunk_id in evidence_ids if chunk_id in answer_part)
            ok = bool(ids)
            return MockResponse(json.dumps({
                "claims": [{"claim": "Mock claim.", "chunk_ids": ids, "supported": ok,
                            "reason": "Mock judge: citation present in the evidence." if ok else "Mock judge: no cited evidence."}],
                "verdict": "supported" if ok else "unsupported"}))
        chunk_ids = [
            chunk_id
            for chunk_id in re.findall(r"chunk_id:\s*([A-Za-z0-9_.=-]+)", joined)
            if "EXAMPLE" not in chunk_id
        ]
        citations = sorted(set(chunk_ids))[:3]
        if "Break the question" in joined:
            return MockResponse(
                json.dumps(
                    {
                        "subquestions": [
                            "Which retrieved annual-report chunks discuss the user question?",
                            "What evidence should be cited in the final answer?",
                        ]
                    }
                )
            )
        if "Plan JSON" in joined:
            return MockResponse(
                json.dumps(
                    {
                        "intent": "annual_report_qa",
                        "tools": ["hybrid_financial_search"],
                        "source_type": None,
                        "needs_calculation": False,
                    }
                )
            )
        if "plain-text classifier" in joined.lower():
            query_match = re.search(r"Message:\s*(.*?)\n\nReturn JSON", joined, flags=re.DOTALL)
            query = query_match.group(1).strip() if query_match else joined
            lowered = query.lower()
            financial_terms = {
                "annual report", "revenue", "sales", "cash", "income", "risk",
                "ticker", "xbrl", "financial", "apple", "microsoft", "amazon",
                "aapl", "msft", "amzn", "nvda", "earnings", "profit", "margin", "eps", "dividend",
                "growth", "debt", "assets", "operating", "expenses",
            }
            blocked_terms = {"<script", "```", "select *", "ignore previous"}
            return MockResponse(
                json.dumps(
                    {
                        "is_plain_text": not any(term in lowered for term in blocked_terms),
                        "financial_intent": any(term in lowered for term in financial_terms),
                        "risk_level": "low",
                        "reason": "Mock classifier decision.",
                        "safe_query": query,
                    }
                )
            )
        if "Return JSON" in joined or '"answer"' in joined:
            return MockResponse(
                json.dumps(
                    {
                        "answer": "Mock answer generated from retrieved financial-report evidence.",
                        "citations": citations,
                        "confidence": "medium",
                        "limitations": "Mock mode does not perform real language reasoning.",
                    },
                    indent=2,
                )
            )
        calc = ""
        if "Calculator: enabled" in joined:
            evidence_part = joined.split("Evidence:", 1)[-1].split("Calculator: enabled", 1)[0]
            figures = re.findall(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?", evidence_part)[:2]
            if len(figures) == 2:
                calc = f" Change: CALC[({figures[1]} - {figures[0]}) / {figures[0]} * 100]%."
        return MockResponse(
            "Mock answer: the retrieved evidence discusses the question. "
            f"Citations: {', '.join(citations) if citations else 'none'}.{calc}"
        )


class HostedChatLLM:
    """Small OpenAI-compatible chat client for hosted Track B inference APIs."""

    def __init__(
        self,
        *,
        api_base: str,
        api_key: str,
        model: str,
        temperature: float,
        timeout_seconds: int = 90,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.timeout_seconds = min(timeout_seconds, float(os.getenv("RAG_HTTP_TIMEOUT_SECONDS", "45")))

    def invoke(self, messages):
        from scripts.common.resilience import Calls, fingerprint
        calls = Calls(float(os.getenv("RAG_HTTP_TIMEOUT_SECONDS", "45")))
        key = fingerprint({"endpoint": self.api_base, "model": self.model,
                           "credential": hashlib.sha256(self.api_key.encode()).hexdigest(),
                           "temperature": self.temperature,
                           "messages": [str(getattr(m, "content", m)) + str(type(m)) for m in messages]})
        return calls.run("hosted_model", lambda: self._invoke(messages), key=key, retry=True)

    def _invoke(self, messages: list[Any]) -> MockResponse:
        payload = {
            "model": self.model,
            "messages": [self._message_to_dict(message) for message in messages],
            "temperature": self.temperature,
            **hosted_llm.payload_extras(),
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        request.add_header("Idempotency-Key", hashlib.sha256(request.data).hexdigest())
        try:
            with hosted_llm.open_url(request, self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Hosted LLM request failed: {exc.reason}") from exc

        data = json.loads(raw)
        try:
            return MockResponse(hosted_llm.clean_content(data["choices"][0]["message"]["content"]))
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected hosted LLM response shape: {raw[:1000]}") from exc

    @staticmethod
    def _message_to_dict(message: Any) -> dict[str, str]:
        role = "user"
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif getattr(message, "type", None) == "ai":
            role = "assistant"
        content = getattr(message, "content", str(message))
        return {"role": role, "content": str(content)}


def load_indexer_module():
    module_path = Path("scripts/rag/index_financial_report_pdfs.py")
    spec = importlib.util.spec_from_file_location("financial_pdf_indexer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["financial_pdf_indexer"] = module
    spec.loader.exec_module(module)
    return module


def get_llm(provider: str, temperature: float = 0.0):
    if provider == "mock":
        return MockLLM()
    if provider == "hosted":
        api_base = os.environ.get("LLM_API_BASE")
        api_key = os.environ.get("LLM_API_KEY")
        model = os.environ.get("LLM_MODEL")
        missing = [
            name for name, value in {
                "LLM_API_BASE": api_base,
                "LLM_API_KEY": api_key,
                "LLM_MODEL": model,
            }.items()
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing hosted LLM env var(s): {', '.join(missing)}")
        return HostedChatLLM(
            model=str(model),
            api_key=str(api_key),
            api_base=str(api_base),
            temperature=temperature,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


# Guardrail patterns and checks live in scripts/guardrails; this agent uses the financial pattern set.
PROMPT_INJECTION_PATTERNS = guard_patterns.FINANCIAL_PROMPT_INJECTION_PATTERNS
CODE_OR_TOOL_PATTERNS = guard_patterns.FINANCIAL_CODE_OR_TOOL_PATTERNS
SECRET_PATTERNS = guard_patterns.SECRET_PATTERNS
SENSITIVE_DATA_PATTERNS = guard_patterns.SENSITIVE_DATA_PATTERNS
BASE64_LIKE_PATTERN = guard_patterns.BASE64_LIKE_PATTERN
FINANCIAL_INTENT_TERMS = {
    "annual report", "10-k", "10-q", "revenue", "net sales", "income", "cash",
    "risk", "xbrl", "sec", "filing", "ticker", "table", "chart", "financial",
    "balance sheet", "cash flow", "operating", "margin", "aapl", "msft", "amzn",
    "nvda", "apple", "microsoft", "amazon", "nvidia",
}
INVESTMENT_REVIEW_TERMS = {
    "buy", "sell", "hold", "investment advice", "recommend", "price target",
    "undervalued", "overvalued", "credit rating", "risk rating",
}
PUBLICATION_REVIEW_TERMS = {
    "board summary", "investor memo", "publish", "external report",
    "send to investors", "compliance report",
}


canonicalize_user_message = guard_checks.canonicalize
redact_secrets = guard_checks.redact
regex_labels = guard_checks.regex_labels


def deterministic_input_guard(message: str, max_chars: int = 4000) -> dict[str, Any]:
    result = guard_checks.input_guard(message, PROMPT_INJECTION_PATTERNS, CODE_OR_TOOL_PATTERNS,
                                      detect_encoded=True, max_chars=max_chars)
    labels = result["risk_labels"]
    return {**result, "reason": "No deterministic risks found." if not labels else f"Detected: {', '.join(labels)}."}


def llm_plain_text_classifier(llm, message: str) -> dict[str, Any]:
    system = prompts.CLASSIFIER_SYSTEM
    human = prompts.CLASSIFIER_HUMAN.format(message=message)
    text = invoke_prompt(llm, system, human)
    fallback = {
        "is_plain_text": True,
        "financial_intent": any(term in message.lower() for term in FINANCIAL_INTENT_TERMS),
        "risk_level": "low",
        "reason": "Fallback classifier decision.",
        "safe_query": message,
    }
    parsed = safe_json_loads(text, fallback=fallback)
    parsed["is_plain_text"] = bool(parsed.get("is_plain_text"))
    parsed["financial_intent"] = bool(parsed.get("financial_intent"))
    parsed["risk_level"] = str(parsed.get("risk_level", "medium")).lower()
    parsed["safe_query"] = canonicalize_user_message(str(parsed.get("safe_query") or message))
    parsed["allowed"] = (
        parsed["is_plain_text"]
        and parsed["financial_intent"]
        and parsed["risk_level"] not in {"medium", "high", "critical"}
    )
    return parsed


def prompt_injection_detector(text: str) -> list[str]:
    return guard_checks.injection_labels(text, PROMPT_INJECTION_PATTERNS, CODE_OR_TOOL_PATTERNS)


def evidence_prompt_injection_guard(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    return guard_checks.evidence_guard(evidence, PROMPT_INJECTION_PATTERNS, CODE_OR_TOOL_PATTERNS)


def output_content_filter(answer: str) -> dict[str, Any]:
    return guard_checks.output_filter(answer, PROMPT_INJECTION_PATTERNS, CODE_OR_TOOL_PATTERNS, [
        (INVESTMENT_REVIEW_TERMS, "investment_or_decision_support_output"),
        (PUBLICATION_REVIEW_TERMS, "external_publication_output")])


def human_review_queue(
    *,
    config: AgentConfig,
    triggers: list[str],
    draft_answer: str,
    evidence: list[dict[str, Any]],
    citation_check: dict[str, Any] | None,
    security: dict[str, Any] | None,
    plain_text: dict[str, Any] | None,
    evidence_guard: dict[str, Any] | None,
    output_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    basis = "|".join([config.query, ",".join(sorted(triggers))])
    queue_id = f"human_review_{hashlib.sha256(basis.encode('utf-8')).hexdigest()[:12]}"
    return {
        "required": True,
        "status": "pending_human_review",
        "queue_id": queue_id,
        "triggers": sorted(set(triggers)),
        "reviewer_payload": {
            "query": redact_secrets(config.query),
            "draft_answer": draft_answer,
            "citation_check": citation_check,
            "security": security,
            "plain_text": plain_text,
            "evidence_guard": evidence_guard,
            "output_filter": output_filter,
            "evidence_chunk_ids": [hit.get("chunk_id") for hit in evidence],
        },
        "allowed_actions": ["approve", "reject", "request_revision", "mark_out_of_scope"],
    }


def governance_review_router(
    *,
    config: AgentConfig,
    draft_answer: str,
    evidence: list[dict[str, Any]],
    citation_check: dict[str, Any] | None,
    security: dict[str, Any] | None,
    plain_text: dict[str, Any] | None,
    evidence_guard: dict[str, Any] | None,
    output_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    triggers: list[str] = []
    query = config.query.lower()
    answer = draft_answer.lower()
    if any(term in query for term in INVESTMENT_REVIEW_TERMS):
        triggers.append("investment_or_decision_support_language")
    if any(term in query for term in PUBLICATION_REVIEW_TERMS):
        triggers.append("external_publication_or_export")
    if not evidence:
        triggers.append("low_retrieval_confidence")
    if citation_check and not citation_check.get("valid"):
        triggers.append("citation_failure")
    if security and security.get("decision") == "review":
        triggers.append("security_sensitive_input")
    if evidence_guard and not evidence_guard.get("safe", True):
        triggers.append("retrieved_evidence_prompt_injection")
    if output_filter and output_filter.get("decision") == "review":
        triggers.append("output_content_filter_review")
    if "conflict" in answer or "contradict" in answer:
        triggers.append("conflicting_evidence")
    if re.search(r"\b(calculate|ratio|growth|change|difference)\b", query) and re.search(
        r"\b(unclear|ambiguous|not specified|insufficient)\b", answer
    ):
        triggers.append("calculation_ambiguity")

    if triggers and config.enable_human_review:
        return human_review_queue(
            config=config,
            triggers=triggers,
            draft_answer=draft_answer,
            evidence=evidence,
            citation_check=citation_check,
            security=security,
            plain_text=plain_text,
            evidence_guard=evidence_guard,
            output_filter=output_filter,
        )
    return {
        "required": False,
        "status": "auto_approved",
        "triggers": sorted(set(triggers)),
    }


def refusal_result(
    technique: str,
    reason: str,
    security: dict[str, Any] | None = None,
    plain_text: dict[str, Any] | None = None,
    evidence_guard: dict[str, Any] | None = None,
    output_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "technique": technique,
        "final_status": "refused",
        "answer": reason,
        "security": security,
        "plain_text": plain_text,
        "evidence_guard": evidence_guard,
        "output_filter": output_filter,
        "citation_check": {"valid": False, "cited_chunk_ids": [], "missing_chunk_ids": [], "allowed_chunk_ids": []},
        "evidence": [],
    }


def apply_governance(
    result: dict[str, Any],
    config: AgentConfig,
    evidence: list[dict[str, Any]],
    security: dict[str, Any] | None,
    plain_text: dict[str, Any] | None,
    evidence_guard: dict[str, Any] | None,
) -> dict[str, Any]:
    output_filter = output_content_filter(str(result.get("answer") or "")) if config.enable_guardrails else {
        "decision": "allow",
        "risk_labels": [],
        "redacted_preview": str(result.get("answer") or "")[:800],
    }
    result["output_filter"] = output_filter
    if output_filter["decision"] == "block":
        result["answer"] = "Generated answer was blocked by the output content filter."
        result["final_status"] = "refused"
        result["human_review"] = {"required": False, "status": "not_required", "triggers": []}
        result["security"] = security
        result["plain_text"] = plain_text
        result["evidence_guard"] = evidence_guard
        return result

    review = governance_review_router(
        config=config,
        draft_answer=str(result.get("answer") or ""),
        evidence=evidence,
        citation_check=result.get("citation_check"),
        security=security,
        plain_text=plain_text,
        evidence_guard=evidence_guard,
        output_filter=output_filter,
    )
    result["human_review"] = review
    result["security"] = security
    result["plain_text"] = plain_text
    result["evidence_guard"] = evidence_guard
    result["final_status"] = review["status"] if review.get("required") else "answered"
    return result


def prepare_guarded_config(llm, config: AgentConfig, technique: str) -> tuple[AgentConfig | None, dict[str, Any] | None]:
    if not config.enable_guardrails:
        return config, None
    security = deterministic_input_guard(config.query, max_chars=config.max_query_chars)
    if security["decision"] == "block":
        return None, refusal_result(
            technique,
            "Request blocked by security guardrails. Please rephrase as a plain financial-report question.",
            security=security,
        )

    plain_text = llm_plain_text_classifier(llm, security["sanitized_query"])
    if not plain_text.get("allowed"):
        return None, refusal_result(
            technique,
            "Request was not routed because it is not a safe plain-text financial-report question.",
            security=security,
            plain_text=plain_text,
        )

    guarded_config = AgentConfig(**{**config.__dict__, "query": plain_text["safe_query"]})
    return guarded_config, {"security": security, "plain_text": plain_text}


def safe_json_loads(text: str, fallback: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return fallback


def answer_evidence(config: AgentConfig) -> list[dict[str, Any]]:
    """Evidence to answer from: /retrieve keeps showing ranked hits, but answering needs a relevant one."""
    return guard_checks.relevance_floor(retrieve_evidence(config), float(os.getenv("RAG_MIN_RERANK_SCORE", "-4")))


def retrieve_evidence(config: AgentConfig) -> list[dict[str, Any]]:
    if os.getenv("RAG_BACKEND") == "financial_qdrant":   # collections built by scripts/rag/financial_ingest.py
        from scripts.rag.financial_ingest import retrieve_for_agent
        return retrieve_for_agent(config.query, config.top_k, ticker=config.ticker, year=config.year,
                                  sector=config.sector, visual=config.visual, mode=config.retrieval_mode)
    if os.getenv("RAG_BACKEND") == "financial_vlm":      # Chroma index built by scripts/rag/vlm_ingest.py
        from scripts.rag import vlm_ingest
        return vlm_ingest.retrieve_for_agent(config.query, config.top_k, ticker=config.ticker, year=config.year,
                                             sector=config.sector, mode=config.retrieval_mode)
    if os.getenv("RAG_BACKEND") == "qdrant":
        from scripts.rag.multimodal_rag import retrieve_local
        return retrieve_local("financial_report_qa", config.query, config.top_k,
                              source_type=config.source_type, ticker=config.ticker, year=config.year)
    indexer = load_indexer_module()
    client = chromadb.PersistentClient(path=str(DEFAULT_INDEX_DIR))
    collection = client.get_collection(DEFAULT_COLLECTION)
    embedder = indexer.get_embedder(config.embedding_provider, config.hash_dimensions)

    if config.retrieval_mode == "dense":
        hits = indexer.retrieve_top_k(
            collection=collection,
            embedder=embedder,
            query=config.query,
            top_k=config.top_k,
            source_type=config.source_type,
            ticker=config.ticker,
            year=config.year,
        )
        return [{**hit, "retrieval_ranks": {"dense": rank}} for rank, hit in enumerate(hits, 1)]

    chunk_rows = indexer.load_chunk_rows(DEFAULT_CHUNKS)
    if config.retrieval_mode == "bm25":
        rows = indexer.bm25_search(chunk_rows, config.query, config.top_k, source_type=config.source_type,
                                   ticker=config.ticker, year=config.year)
        return [{**indexer.hit_from_chunk(chunk_id=str(row["chunk_id"]), text=str(row.get("text", "")),
                                          metadata=row.get("metadata", {}), rank=rank, bm25_score=row.get("bm25_score")),
                 "retrieval_ranks": {"bm25": rank}} for rank, row in enumerate(rows, 1)]
    hits = indexer.retrieve_hybrid_top_k(
        collection=collection,
        embedder=embedder,
        chunk_rows=chunk_rows,
        query=config.query,
        top_k=config.top_k,
        dense_top_k=config.dense_top_k,
        bm25_top_k=config.bm25_top_k,
        rrf_k=60,
        reranker="simple",
        source_type=config.source_type,
        ticker=config.ticker,
        year=config.year,
    )
    return [{**hit, "retrieval_ranks": {name: hit[f"{name}_rank"] for name in ("dense", "bm25") if hit.get(f"{name}_rank")}}
            for hit in hits]


def evidence_to_context(evidence: list[dict[str, Any]], max_chars: int) -> str:
    blocks = []
    used = 0
    for hit in evidence:
        metadata = hit.get("metadata", {})
        text = str(hit.get("text") or "")
        block = "\n".join(
            [
                "---",
                f"chunk_id: {hit.get('chunk_id')}",
                f"source_type: {hit.get('source_type')}",
                f"title: {hit.get('title')}",
                f"ticker: {hit.get('ticker')}",
                f"year: {hit.get('year')}",
                f"page: {hit.get('page')}",
                f"source_url: {hit.get('source_url')}",
                f"local_path: {hit.get('local_path')}",
                f"image_path: {hit.get('image_path')}",
                f"scores: distance={hit.get('distance')} bm25={hit.get('bm25_score')} rrf={hit.get('rrf_score')} rerank={hit.get('rerank_score')}",
                *([f"matched_by: {matched_by(hit['retrieval_ranks'])}"] if hit.get("retrieval_ranks") else []),
                "text:",
                text,
            ]
        )
        if used + len(block) > max_chars:
            break
        blocks.append(block)
        used += len(block)
    return "\n".join(blocks)


def extract_citations(answer: str) -> list[str]:
    patterns = [
        r"annual_report_[A-Za-z0-9_.=-]+",
        r"sec_facts_[A-Za-z0-9_.=-]+",
    ]
    citations: set[str] = set()
    for pattern in patterns:
        citations.update(match.rstrip(".,;:)]}\"'") for match in re.findall(pattern, answer))
    return sorted(citations)


def validate_citations(answer: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    allowed = {str(hit.get("chunk_id")) for hit in evidence}
    cited = set(extract_citations(answer))
    missing = sorted(cited - allowed)
    return {
        "valid": not missing and bool(cited),
        "cited_chunk_ids": sorted(cited),
        "missing_chunk_ids": missing,
        "allowed_chunk_ids": sorted(allowed),
    }


def invoke_prompt(llm, system: str, human: str) -> str:
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])
    return str(response.content)


def baseline_prompt(llm, config: AgentConfig, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    context = evidence_to_context(evidence, config.max_context_chars)
    answer = invoke_prompt(llm, prompts.BASELINE_SYSTEM, prompts.BASELINE_HUMAN.format(query=config.query, context=context))
    return {"technique": "baseline_rag", "answer": answer, "citation_check": validate_citations(answer, evidence)}


def structured_few_shot_prompt(llm, config: AgentConfig, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    context = evidence_to_context(evidence, config.max_context_chars)
    prompt = ChatPromptTemplate.from_messages([("system", prompts.FEW_SHOT_SYSTEM), ("human", prompts.FEW_SHOT_HUMAN)])
    messages = prompt.format_messages(query=config.query, context=context)
    response = llm.invoke(messages)
    answer = str(response.content)
    parsed = safe_json_loads(answer, fallback={"answer": answer, "citations": []})
    return {
        "technique": "structured_few_shot",
        "answer": answer,
        "parsed": parsed,
        "citation_check": validate_citations(answer, evidence),
    }


def decomposition_prompt(llm, config: AgentConfig, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    context = evidence_to_context(evidence, config.max_context_chars)
    plan_text = invoke_prompt(llm, prompts.DECOMPOSE_SYSTEM, prompts.DECOMPOSE_HUMAN.format(query=config.query))
    plan = safe_json_loads(plan_text, fallback={"subquestions": [config.query]})
    answer_human = prompts.DECOMPOSITION_ANSWER_HUMAN.format(
        query=config.query, subquestions=json.dumps(plan.get("subquestions", []), ensure_ascii=False), context=context)
    answer = invoke_prompt(llm, prompts.DECOMPOSITION_ANSWER_SYSTEM, answer_human)
    return {
        "technique": "decomposition",
        "subquestions": plan.get("subquestions", []),
        "answer": answer,
        "citation_check": validate_citations(answer, evidence),
    }


def self_consistency_prompt(llm_factory, config: AgentConfig, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    samples = []
    for index in range(config.self_consistency_samples):
        llm = llm_factory(0.4 + 0.1 * index)
        result = baseline_prompt(llm, config, evidence)
        samples.append(result["answer"])

    context = evidence_to_context(evidence, config.max_context_chars)
    judge = llm_factory(0.0)
    human = prompts.CONSOLIDATOR_HUMAN.format(query=config.query, context=context,
                                              candidates=json.dumps(samples, ensure_ascii=False, indent=2))
    answer = invoke_prompt(judge, prompts.CONSOLIDATOR_SYSTEM, human)
    return {
        "technique": "self_consistency",
        "samples": samples,
        "answer": answer,
        "citation_check": validate_citations(answer, evidence),
    }


class AgentState(TypedDict, total=False):
    query: str
    final_status: str
    security: dict[str, Any]
    plain_text: dict[str, Any]
    refusal_reason: str
    plan: dict[str, Any]
    evidence: list[dict[str, Any]]
    evidence_guard: dict[str, Any]
    evidence_check: dict[str, Any]
    output_filter: dict[str, Any]
    answer: str
    citation_check: dict[str, Any]
    support_check: dict[str, Any]
    human_review: dict[str, Any]
    repair_count: int
    retrieval_attempts: int
    tool_calls: list[dict[str, Any]]
    calculations: list[dict[str, Any]]
    path: list[str]


STOPPED = {"refused", "insufficient_evidence", "not_available"}
# Nodes every run still needs after the evidence guard: answer, critic, governance (+ calculator when planned).
MIN_TAIL = 3


def agentic_graph_prompt(llm, config: AgentConfig) -> dict[str, Any]:
    """Planner -> executor -> critic graph with planner-chosen tools and budget-aware loops.

    Routing is decided per request from the state: the plan's tools (coverage check, ColPali, calculator),
    missing evidence for a company/year (one broadened retrieval retry), CALC[...] in the answer (calculator
    step), invalid citations (one repair with the critic's feedback). A loop is taken only if the rest of the
    run still fits AGENT_MAX_STEPS, so the cap shortens a run instead of failing it. ``agent_run`` records the
    path, every tool call and every calculation.
    """
    from scripts.agents import financial_tools as tools
    from scripts.agents.financial_clarifier import METRICS, default_clarifier

    clarifier = default_clarifier()
    budget = max_steps()
    support_enabled = os.getenv("RAG_SUPPORT_CHECK", "1").strip().lower() not in {"0", "false", "off", "no"}

    def step(name, fn):
        def node(state: AgentState) -> AgentState:
            update = fn(state) or {}
            return {**update, "path": [*state.get("path", []), name]}
        return node

    def used(state: AgentState) -> int:
        return len(state.get("path", []))

    def tail(state: AgentState) -> int:
        return MIN_TAIL + (1 if "calculator" in state.get("plan", {}).get("tools", []) else 0)

    def can_retry(state: AgentState) -> bool:
        """Another retrieval round fits: retrieve + guard, then the tail (path already includes this guard)."""
        missing = state.get("evidence_check", {}).get("missing", [])
        return (bool(missing) and state.get("retrieval_attempts", 1) < tools.MAX_RETRIEVAL_ATTEMPTS
                and used(state) + 2 + tail(state) <= budget)

    def security_node(state: AgentState) -> AgentState:
        if not config.enable_guardrails:
            return {"query": state["query"], "final_status": "guardrails_disabled"}
        security = deterministic_input_guard(state["query"], max_chars=config.max_query_chars)
        if security["decision"] == "block":
            return {
                "security": security,
                "final_status": "refused",
                "answer": "Request blocked by security guardrails. Please rephrase as a plain financial-report question.",
            }
        return {"security": security, "query": security["sanitized_query"]}

    def plain_text_node(state: AgentState) -> AgentState:
        if state.get("final_status") == "refused":
            return {}
        if not config.enable_guardrails:
            return {"plain_text": None}
        plain_text = llm_plain_text_classifier(llm, state["query"])
        if not plain_text.get("allowed"):
            return {
                "plain_text": plain_text,
                "final_status": "refused",
                "answer": "Request was not routed because it is not a safe plain-text financial-report question.",
            }
        return {"plain_text": plain_text, "query": plain_text["safe_query"]}

    def planner_node(state: AgentState) -> AgentState:
        plan_text = invoke_prompt(llm, prompts.PLANNER_SYSTEM, prompts.PLANNER_HUMAN.format(query=state["query"]))
        raw = safe_json_loads(plan_text, fallback={})
        plan = tools.normalize_plan(raw, state["query"], ticker=config.ticker, year=config.year,
                                    source_type=config.source_type, clarifier=clarifier)
        return {"plan": plan}

    def executor_node(state: AgentState) -> AgentState:
        """Run the planned tools: coverage check, then one filtered search per target. A retry searches only
        the targets that found nothing, with a broadened query and twice the depth."""
        plan = dict(state.get("plan", {}))
        attempt = state.get("retrieval_attempts", 0) + 1
        calls_log = list(state.get("tool_calls", []))
        targets = plan.get("targets") or [{"ticker": config.ticker, "year": config.year}]
        if attempt == 1 and "coverage_lookup" in plan.get("tools", []):
            started = time.perf_counter()
            covered = tools.coverage(targets, clarifier.directory)
            calls_log.append({"tool": "coverage_lookup", "attempt": attempt, "known": covered["known"],
                              "searchable": [tools.target_label(t) for t in covered["searchable"]],
                              "skipped": covered["skipped"], "ms": round((time.perf_counter() - started) * 1000)})
            targets = covered["searchable"]
            plan.update(targets=targets, skipped_targets=covered["skipped"])
        per_target = max(3, -(-config.top_k // max(1, len(targets)))) * (2 if attempt > 1 else 1)   # retry: 2x deeper
        if attempt > 1:
            missing = set(state.get("evidence_check", {}).get("missing", []))
            targets = [t for t in targets if tools.target_label(t) in missing]
        metric = clarifier.metric_in(state["query"])
        query = state["query"] if attempt == 1 else tools.broaden_query(state["query"], metric, METRICS.get(metric, []))
        visual = "visual_search" in plan.get("tools", [])
        evidence = list(state.get("evidence", [])) if attempt > 1 else []
        seen = {hit.get("chunk_id") for hit in evidence}
        for target in targets:
            started = time.perf_counter()
            run_config = AgentConfig(**{**config.__dict__, "query": query, "ticker": target.get("ticker"),
                                        "year": target.get("year"), "top_k": per_target, "visual": visual,
                                        "source_type": plan.get("source_type") or config.source_type})
            hits = answer_evidence(run_config)
            fresh = []
            for hit in hits:
                if hit.get("chunk_id") not in seen:
                    seen.add(hit.get("chunk_id"))
                    fresh.append({**hit, "target": tools.target_label(target)})
            evidence.extend(fresh)
            calls_log.append({"tool": "text_search+visual_search" if visual else "text_search", "attempt": attempt,
                              "target": tools.target_label(target), "query": query, "top_k": per_target,
                              "hits": len(hits), "new": len(fresh),
                              "ms": round((time.perf_counter() - started) * 1000)})
        return {"plan": plan, "evidence": evidence, "retrieval_attempts": attempt, "tool_calls": calls_log}

    def evidence_guard_node(state: AgentState) -> AgentState:
        evidence = state.get("evidence", [])
        guard = evidence_prompt_injection_guard(evidence)
        if not guard["safe"]:
            return {
                "evidence_guard": guard,
                "final_status": "refused",
                "answer": "Retrieved evidence contained unsafe instructions, so the answer was not generated.",
            }
        plan = state.get("plan", {})
        targets = plan.get("targets") or []
        found = {hit.get("target") for hit in evidence}
        missing = [tools.target_label(t) for t in targets if tools.target_label(t) not in found]
        check = {"attempt": state.get("retrieval_attempts", 1), "evidence": len(evidence), "missing": missing,
                 "skipped": [tools.target_label(t) for t in plan.get("skipped_targets") or []]}
        update: AgentState = {"evidence_guard": guard, "evidence_check": check}
        if not evidence and not can_retry({**state, "evidence_check": check, "path": [*state.get("path", []), "x"]}):
            update.update(final_status="not_available" if check["skipped"] and not targets else "insufficient_evidence",
                          answer="No answer found in the available evidence.")
        return update

    def answer_node(state: AgentState) -> AgentState:
        if state.get("final_status") in STOPPED:
            return {}
        evidence = state.get("evidence", [])
        context = evidence_to_context(evidence, config.max_context_chars)
        plan = state.get("plan", {})
        human = prompts.GRAPH_ANSWER_HUMAN.format(
            query=state["query"], context=context,
            plan=json.dumps({key: plan.get(key) for key in ("intent", "tools", "targets")}, ensure_ascii=False))
        if "calculator" in plan.get("tools", []):
            human += "\n\n" + prompts.CALCULATOR_INSTRUCTION
        feedback = critic_feedback(state) if state.get("repair_count", 0) else ""
        if feedback:
            human += "\n\n" + prompts.REPAIR_INSTRUCTION.format(
                feedback=feedback, allowed=", ".join(str(hit.get("chunk_id")) for hit in evidence))
        return {"answer": invoke_prompt(llm, prompts.GRAPH_ANSWER_SYSTEM, human)}

    def critic_feedback(state: AgentState) -> str:
        check, support = state.get("citation_check") or {}, state.get("support_check") or {}
        notes = []
        if check and not check.get("valid", True):
            notes.append("it cited chunk IDs that are not in the evidence ("
                         + (", ".join(check.get("missing_chunk_ids") or []) or "no citations") + ")")
        if support.get("verdict") not in (None, "supported"):
            claims = support.get("unsupported_claims") or []
            if claims:
                notes.append("these claims are not stated by the cited evidence: " + "; ".join(claims[:5]))
            numbers = (support.get("numeric") or {}).get("ungrounded") or []
            if numbers:
                notes.append("these figures appear in no cited chunk: " + ", ".join(numbers[:10])
                             + " (copy figures exactly, cite the chunk that states them, use CALC[...] for arithmetic)")
            if not claims and not numbers:
                notes.append("its support could not be verified")
        return "; ".join(notes)

    def support_check(state: AgentState, answer: str, cited: list[str]) -> dict[str, Any]:
        """Critic, part 2: does the cited evidence state what the answer says? A code check on every figure,
        then an LLM judge on every claim. Both must pass for 'supported'."""
        evidence = state.get("evidence", [])
        cited_hits = [hit for hit in evidence if str(hit.get("chunk_id")) in set(cited)]
        numeric = tools.numeric_grounding(answer, cited_hits, state.get("calculations", []))
        raw = invoke_prompt(llm, prompts.SUPPORT_SYSTEM, prompts.SUPPORT_HUMAN.format(
            query=state["query"], answer=answer, context=evidence_to_context(cited_hits, config.max_context_chars)))
        judged = tools.parse_support(raw)
        verdict = judged["verdict"]
        if not numeric["grounded"] and verdict in ("supported", "partially_supported"):
            verdict = "unsupported" if not judged.get("claims") else "partially_supported"
        return {**judged, "verdict": verdict, "numeric": numeric, "method": "llm_judge+numeric",
                "attempt": state.get("repair_count", 0) + 1}

    def calculator_node(state: AgentState) -> AgentState:
        answer, records = tools.run_calculations(str(state.get("answer") or ""), state.get("evidence", []))
        calls_log = [*state.get("tool_calls", []), {"tool": "calculator", "expressions": len(records),
                                                    "grounded": all(r["grounded"] for r in records)}]
        return {"answer": answer, "calculations": [*state.get("calculations", []), *records], "tool_calls": calls_log}

    def critic_node(state: AgentState) -> AgentState:
        if state.get("final_status") in STOPPED:
            return {
                "citation_check": {
                    "valid": False,
                    "cited_chunk_ids": [],
                    "missing_chunk_ids": [],
                    "allowed_chunk_ids": [str(hit.get("chunk_id")) for hit in state.get("evidence", [])],
                }
            }
        answer = tools.strip_calculations(str(state.get("answer") or ""))     # CALC[...] the calculator didn't run
        check = validate_citations(answer, state.get("evidence", []))
        support = None
        if check["valid"] and support_enabled:
            support = support_check(state, answer, check["cited_chunk_ids"])
        accepted = check["valid"] and (support is None or support["verdict"] == "supported")
        return {
            "answer": answer,
            "citation_check": check,
            "support_check": support,
            "repair_count": state.get("repair_count", 0) + (0 if accepted else 1),
        }

    def governance_node(state: AgentState) -> AgentState:
        if state.get("final_status") in STOPPED:
            return {
                "human_review": {"required": False, "status": "not_required", "triggers": []},
                "final_status": state["final_status"],
            }
        run_config = AgentConfig(**{**config.__dict__, "query": state.get("query", config.query)})
        output_filter = output_content_filter(str(state.get("answer") or "")) if config.enable_guardrails else {
            "decision": "allow",
            "risk_labels": [],
            "redacted_preview": str(state.get("answer") or "")[:800],
        }
        if output_filter["decision"] == "block":
            return {
                "output_filter": output_filter,
                "answer": "Generated answer was blocked by the output content filter.",
                "human_review": {"required": False, "status": "not_required", "triggers": []},
                "final_status": "refused",
            }
        review = governance_review_router(
            config=run_config,
            draft_answer=str(state.get("answer") or ""),
            evidence=state.get("evidence", []),
            citation_check=state.get("citation_check"),
            security=state.get("security"),
            plain_text=state.get("plain_text"),
            evidence_guard=state.get("evidence_guard"),
            output_filter=output_filter,
        )
        extra = []
        if any(not c.get("grounded") for c in state.get("calculations", [])):
            extra.append("unverified_calculation")
        if (state.get("support_check") or {}).get("verdict") not in (None, "supported"):
            extra.append("unsupported_claims")
        if extra and config.enable_human_review:
            review = {**review, "required": True, "status": "pending_human_review",
                      "triggers": [*review.get("triggers", []), *extra]}
        return {
            "output_filter": output_filter,
            "human_review": review,
            "final_status": review["status"] if review.get("required") else state.get("final_status", "answered"),
        }

    # ------------------------------------------------------------------ routing, decided per request
    def should_continue_after_security(state: AgentState) -> Literal["stop", "continue"]:
        return "stop" if state.get("final_status") == "refused" else "continue"

    def after_evidence(state: AgentState) -> Literal["stop", "retry", "answer"]:
        if state.get("final_status") in STOPPED:
            return "stop"
        return "retry" if can_retry(state) else "answer"

    def after_answer(state: AgentState) -> Literal["calculate", "critic"]:
        wants = ("calculator" in state.get("plan", {}).get("tools", [])
                 and tools.CALC_TOKEN.search(str(state.get("answer") or "")))
        return "calculate" if wants and used(state) + 3 <= budget else "critic"     # calculator, critic, governance

    def should_repair(state: AgentState) -> Literal["repair", "finish"]:
        check, support = state.get("citation_check", {}), state.get("support_check") or {}
        accepted = check.get("valid") and support.get("verdict", "supported") == "supported"
        if state.get("final_status") in STOPPED or accepted or state.get("repair_count", 0) > 1:
            return "finish"
        return "repair" if used(state) + tail(state) <= budget else "finish"      # answer (+calc), critic, governance

    graph = StateGraph(AgentState)
    graph.add_node("security_step", step("security_step", security_node))
    graph.add_node("plain_text_step", step("plain_text_step", plain_text_node))
    graph.add_node("plan_step", step("plan_step", planner_node))
    graph.add_node("retrieve_step", step("retrieve_step", executor_node))
    graph.add_node("evidence_guard_step", step("evidence_guard_step", evidence_guard_node))
    graph.add_node("answer_step", step("answer_step", answer_node))
    graph.add_node("calculator_step", step("calculator_step", calculator_node))
    graph.add_node("critic_step", step("critic_step", critic_node))
    graph.add_node("governance_step", step("governance_step", governance_node))
    graph.set_entry_point("security_step")
    graph.add_conditional_edges("security_step", should_continue_after_security, {"stop": "critic_step", "continue": "plain_text_step"})
    graph.add_conditional_edges("plain_text_step", should_continue_after_security, {"stop": "critic_step", "continue": "plan_step"})
    graph.add_edge("plan_step", "retrieve_step")
    graph.add_edge("retrieve_step", "evidence_guard_step")
    graph.add_conditional_edges("evidence_guard_step", after_evidence,
                                {"stop": "critic_step", "retry": "retrieve_step", "answer": "answer_step"})
    graph.add_conditional_edges("answer_step", after_answer, {"calculate": "calculator_step", "critic": "critic_step"})
    graph.add_edge("calculator_step", "critic_step")
    graph.add_conditional_edges("critic_step", should_repair, {"repair": "answer_step", "finish": "governance_step"})
    graph.add_edge("governance_step", END)
    app = graph.compile()
    try:
        final_state = app.invoke({"query": config.query, "repair_count": 0, "path": []}, config=graph_config())
    except GraphRecursionError:
        final_state = {"final_status": "step_limit", "answer": STEP_LIMIT_ANSWER,
                       "citation_check": {"valid": False, "reason": f"step limit of {max_steps()} reached"}}
    path = final_state.get("path", [])
    return {
        "technique": "agentic_graph",
        "final_status": final_state.get("final_status"),
        "security": final_state.get("security"),
        "plain_text": final_state.get("plain_text"),
        "plan": final_state.get("plan"),
        "answer": final_state.get("answer"),
        "citation_check": final_state.get("citation_check"),
        "support_check": final_state.get("support_check"),
        "evidence_guard": final_state.get("evidence_guard"),
        "output_filter": final_state.get("output_filter"),
        "human_review": final_state.get("human_review"),
        "evidence": final_state.get("evidence", []),
        "agent_run": {
            "path": path,
            "steps": len(path),
            "max_steps": budget,
            "retrieval_attempts": final_state.get("retrieval_attempts", 0),
            "repairs": max(0, path.count("answer_step") - 1),
            "evidence_check": final_state.get("evidence_check"),
            "tool_calls": final_state.get("tool_calls", []),
            "calculations": final_state.get("calculations", []),
        },
    }


def _run_technique(technique: str, llm_provider: str, config: AgentConfig) -> dict[str, Any]:
    def llm_factory(temperature: float = 0.0):
        return get_llm(llm_provider, temperature=temperature)

    if technique == "agentic_graph":
        return agentic_graph_prompt(llm_factory(0.0), config)

    llm = llm_factory(0.0)
    guarded_config, guard_context = prepare_guarded_config(llm, config, technique)
    if guarded_config is None:
        return guard_context or refusal_result(technique, "Request refused.")

    evidence = answer_evidence(guarded_config)
    evidence_guard = evidence_prompt_injection_guard(evidence) if guarded_config.enable_guardrails else {"safe": True, "flagged_chunks": [], "risk_labels": []}
    security = (guard_context or {}).get("security")
    plain_text = (guard_context or {}).get("plain_text")
    if not evidence_guard["safe"]:
        return refusal_result(
            technique,
            "Retrieved evidence contained unsafe instructions, so the answer was not generated.",
            security=security,
            plain_text=plain_text,
            evidence_guard=evidence_guard,
        ) | {"evidence": evidence}

    if technique == "baseline_rag":
        result = baseline_prompt(llm, guarded_config, evidence) | {"evidence": evidence}
    if technique == "structured_few_shot":
        result = structured_few_shot_prompt(llm, guarded_config, evidence) | {"evidence": evidence}
    if technique == "decomposition":
        result = decomposition_prompt(llm, guarded_config, evidence) | {"evidence": evidence}
    if technique == "self_consistency":
        result = self_consistency_prompt(llm_factory, guarded_config, evidence) | {"evidence": evidence}
    if technique not in {"baseline_rag", "structured_few_shot", "decomposition", "self_consistency"}:
        raise ValueError(f"Unsupported technique: {technique}")
    return apply_governance(result, guarded_config, evidence, security, plain_text, evidence_guard)


def run_technique(technique: str, llm_provider: str, config: AgentConfig) -> dict[str, Any]:
    """Bound legacy comparison workflows and label unverified generation honestly."""
    from scripts.common.resilience import Calls, CallFailure
    calls = Calls(float(os.getenv("RAG_REQUEST_TIMEOUT_SECONDS", "60")))
    try:
        result = calls.run("financial_workflow", lambda: _run_technique(technique, llm_provider, config))
    except CallFailure:
        result = {"technique": technique, "final_status": "tool_error",
                  "answer": "No answer found because a required service failed or timed out.", "evidence": [],
                  "citation_check": {"valid": False, "cited_chunk_ids": []}}
    scope = {"expected": [], "coverage": None}
    verified = False
    if result.get("final_status") not in {"refused", "tool_error"}:
        scope = question_scope(config.query, config.ticker)
    if scope["coverage"]:
        # The filings can't answer this, so nothing retrieved or generated may stand in for it.
        for key in ("parsed", "samples", "subquestions"):
            result.pop(key, None)
        result.update(answer=scope["coverage"]["message"], final_status="not_available", coverage=scope["coverage"],
                      evidence=[], citation_check={"valid": False, "cited_chunk_ids": []})
    elif result.get("final_status") not in {"refused", "tool_error"}:
        evidence = result.get("evidence", [])
        if scope["expected"] and llm_provider != "mock":
            # An unfiltered search can return other companies; their excerpts must not become the answer.
            evidence = result["evidence"] = [h for h in evidence if str(h.get("ticker") or "").upper() in scope["expected"]]
        if not evidence:
            result.update(answer="No answer found in the available evidence.", final_status="insufficient_evidence",
                          citation_check={"valid": False, "cited_chunk_ids": []})
        elif llm_provider != "mock" and answer_verified(result, evidence):
            verified = True        # the critic's support check passed: the generated answer stands
        elif llm_provider != "mock":
            # Unverified prose (legacy prompting experiments, or an answer the critic couldn't verify) is not
            # promoted to fact: return only the source excerpts.
            rows = [h for h in evidence if str(h.get("text", "")).strip()]
            result.update(answer="Unverified generation withheld. Available evidence excerpts:\n" + "\n".join(
                str(h["text"])[:1000] + " [" + str(h["chunk_id"]) + "]" for h in rows),
                final_status="evidence_only" if rows else "insufficient_evidence",
                citation_check={"valid": bool(rows), "cited_chunk_ids": [h["chunk_id"] for h in rows]})
        if scope["expected"]:
            cited = (result.get("citation_check") or {}).get("cited_chunk_ids") or []
            result["company_check"] = guard_checks.company_scope_check(evidence, cited, scope["expected"])
            if not result["company_check"]["valid"]:
                for key in ("parsed", "samples", "subquestions"):
                    result.pop(key, None)
                result.update(answer="Answer withheld: it cited filings from a company the question didn't ask about.",
                              final_status="not_grounded", citation_check={"valid": False, "cited_chunk_ids": cited})
    if llm_provider != "mock":
        result.pop("parsed", None)
        result.pop("samples", None)
    verified = verified and result.get("final_status") not in {"not_grounded", "not_available", "refused", "tool_error"}
    method = "mock" if llm_provider == "mock" else "llm_support_check" if verified else "source_excerpts"
    result.update(trace_id=calls.trace_id, trace=calls.events,
                  grounding={"method": method, "semantic_answer_verified": verified})
    return result


def answer_verified(result: dict[str, Any], evidence: list[dict[str, Any]]) -> bool:
    """The agent graph's critic found every claim and figure supported by evidence it still cites."""
    support, check = result.get("support_check") or {}, result.get("citation_check") or {}
    allowed = {str(hit.get("chunk_id")) for hit in evidence}
    return (result.get("technique") == "agentic_graph" and support.get("verdict") == "supported"
            and bool(check.get("valid")) and set(check.get("cited_chunk_ids") or []) <= allowed)


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    evidence = result.get("evidence", [])
    return {
        "trace_id": result.get("trace_id"),
        "trace": result.get("trace"),
        "grounding": result.get("grounding"),
        "technique": result.get("technique"),
        "final_status": result.get("final_status"),
        "answer": result.get("answer"),
        "coverage": result.get("coverage"),
        "company_check": result.get("company_check"),
        "parsed": result.get("parsed"),
        "subquestions": result.get("subquestions"),
        "plan": result.get("plan"),
        "agent_run": result.get("agent_run"),
        "support_check": result.get("support_check"),
        "security": result.get("security"),
        "plain_text": result.get("plain_text"),
        "evidence_guard": result.get("evidence_guard"),
        "output_filter": result.get("output_filter"),
        "human_review": result.get("human_review"),
        "citation_check": result.get("citation_check"),
        "evidence_chunk_ids": [hit.get("chunk_id") for hit in evidence],
        "evidence_preview": [
            {
                "chunk_id": hit.get("chunk_id"),
                "source_type": hit.get("source_type"),
                "ticker": hit.get("ticker"),
                "year": hit.get("year"),
                "page": hit.get("page"),
                "preview": hit.get("preview"),
                "image_path": hit.get("image_path"),
                "retrieval_ranks": hit.get("retrieval_ranks"),
            }
            for hit in evidence[:5]
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument("--ticker")
    parser.add_argument("--year", type=int)
    parser.add_argument("--source-type", help="annual_report_pdf, table_summary, or chart_summary")
    parser.add_argument(
        "--technique",
        choices=[
            "baseline_rag",
            "structured_few_shot",
            "decomposition",
            "self_consistency",
            "agentic_graph",
            "all",
        ],
        default="all",
    )
    parser.add_argument("--llm-provider", choices=["mock", "hosted"], default="mock")
    parser.add_argument("--embedding-provider", choices=["hashing", "hosted"], default="hashing")
    parser.add_argument("--hash-dimensions", type=int, default=384)
    parser.add_argument("--retrieval-mode", choices=["hybrid", "dense", "bm25"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--dense-top-k", type=int, default=30)
    parser.add_argument("--bm25-top-k", type=int, default=30)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--self-consistency-samples", type=int, default=3)
    parser.add_argument("--disable-guardrails", action="store_true")
    parser.add_argument("--disable-human-review", action="store_true")
    parser.add_argument("--output", type=Path)
    add_langsmith_args(parser, default_project="capstone-financial-report-qa")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    langsmith_status = maybe_enable_from_args(args)
    config = AgentConfig(
        query=args.query,
        ticker=args.ticker,
        year=args.year,
        source_type=args.source_type,
        top_k=args.top_k,
        dense_top_k=args.dense_top_k,
        bm25_top_k=args.bm25_top_k,
        retrieval_mode=args.retrieval_mode,
        embedding_provider=args.embedding_provider,
        hash_dimensions=args.hash_dimensions,
        max_context_chars=args.max_context_chars,
        self_consistency_samples=args.self_consistency_samples,
        enable_guardrails=not args.disable_guardrails,
        enable_human_review=not args.disable_human_review,
    )
    techniques = [
        "baseline_rag",
        "structured_few_shot",
        "decomposition",
        "self_consistency",
        "agentic_graph",
    ] if args.technique == "all" else [args.technique]

    results = {
        "query": args.query,
        "ticker": args.ticker,
        "year": args.year,
        "llm_provider": args.llm_provider,
        "embedding_provider": args.embedding_provider,
        "retrieval_mode": args.retrieval_mode,
        "langsmith": langsmith_status,
        "results": [
            summarize_result(
                run_traced(
                    f"financial_report_qa.{technique}",
                    run_technique,
                    technique,
                    args.llm_provider,
                    config,
                    run_type="chain",
                    tags=[
                        "capstone",
                        "financial_report_qa",
                        technique,
                        args.llm_provider,
                        args.embedding_provider,
                        args.retrieval_mode,
                        *args.langsmith_tag,
                    ],
                    metadata={
                        "ticker": args.ticker,
                        "year": args.year,
                        "source_type": args.source_type,
                        "top_k": args.top_k,
                        "dense_top_k": args.dense_top_k,
                        "bm25_top_k": args.bm25_top_k,
                        "guardrails_enabled": not args.disable_guardrails,
                        "human_review_enabled": not args.disable_human_review,
                    },
                )
            )
            for technique in techniques
        ],
    }
    text = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
