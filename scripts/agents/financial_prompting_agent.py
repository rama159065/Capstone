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
import logging
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

from scripts.agents import financial_checks
from scripts.common import hosted_llm
from scripts.common.companies import filing_year, fiscal_label
from scripts.rag.hybrid import matched_by
from scripts.common.agent_limits import STEP_LIMIT_ANSWER, graph_config, max_steps
from scripts.agents.financial_clarifier import EXPLAIN, question_scope
from scripts.guardrails import checks as guard_checks, patterns as guard_patterns
from scripts.prompts import financial as prompts
from scripts.observability.langsmith_utils import (
    traced_llm,
    traced_retrieval,
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
    def __init__(self, content: str, usage: dict[str, Any] | None = None) -> None:
        self.content = content
        self.usage = usage or {}          # OpenAI-format token counts from a hosted response (for LangSmith)


def as_role_messages(messages: list[Any]) -> list[dict[str, str]]:
    """LangChain messages (or strings) as role/content dicts, the form LangSmith shows for an llm run."""
    roles = {"system": "system", "human": "user", "ai": "assistant"}
    return [{"role": roles.get(getattr(m, "type", ""), "user"), "content": str(getattr(m, "content", m))}
            for m in messages]


class MockLLM:
    """Deterministic local LLM stand-in for pipeline testing."""

    def invoke(self, messages: list[Any]) -> MockResponse:
        return traced_llm("mock_llm", lambda: self._respond(messages), as_role_messages(messages),
                          provider="mock", model="mock")

    def _respond(self, messages: list[Any]) -> MockResponse:
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
        if "Route JSON" in joined:
            # Mock router: always proposes a lookup, so the router's rules decide trend, comparison and report.
            return MockResponse(json.dumps({"route": "lookup", "tools": ["hybrid_financial_search"],
                                            "needs_calculation": False, "reason": "Mock router."}))
        if "Series JSON" in joined:
            # Mock extractor: the first comma-grouped figure in the evidence and the chunk that prints it.
            evidence_part = joined.split("Evidence:", 1)[-1]
            for block in evidence_part.split("\n---")[0:]:
                chunk = re.search(r"chunk_id:\s*([A-Za-z0-9_.=-]+)", block)
                text = block.split("text:", 1)[-1]
                figure = re.search(r"(?<![\d.])\d{1,3}(?:,\d{3})+(?:\.\d+)?", text)
                if chunk and figure:
                    return MockResponse(json.dumps({"value": figure.group(0), "unit": "millions", "chunk_id": chunk.group(1),
                                                    "quote": text.strip()[:80]}))
            return MockResponse(json.dumps({"value": None, "unit": "", "chunk_id": None, "quote": ""}))
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
                "fund", "etf", "holdings", "n-port", "nport", "shareholder", "stockholder", "repurchase", "buyback",
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


logger = logging.getLogger(__name__)


NET_OF_ISSUES = re.compile(r"net sales", re.I)
PRIOR_YEAR = re.compile(r"\b(prior|previous|preceding|last)\s+(?:fiscal\s+)?year\b|\byear[- ]over[- ]year\b|\byoy\b|\bchange[ds]?\b|"
                        r"\bgr[eo]w(?:th|n|s)?\b|\bincreased?\b|\bdecreased?\b", re.I)


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
        return traced_llm(
            "hosted_llm", lambda: calls.run("hosted_model", lambda: self._invoke(messages), key=key, retry=True),
            as_role_messages(messages), provider="openai-compatible", model=self.model, temperature=self.temperature)

    def _invoke(self, messages: list[Any]) -> MockResponse:
        role_messages = [self._message_to_dict(message) for message in messages]
        for step in range(hosted_llm.SHRINK_STEPS + 1):
            try:
                return self._post(role_messages)
            except urllib.error.HTTPError as exc:
                if step == hosted_llm.SHRINK_STEPS or not hosted_llm.context_too_long(exc):
                    raise
                role_messages = hosted_llm.shrink_messages(role_messages)      # longer than the model's context

    def _post(self, role_messages: list[dict[str, str]]) -> MockResponse:
        payload = {
            "model": self.model,
            "messages": role_messages,
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
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
            return MockResponse(hosted_llm.clean_content(data["choices"][0]["message"]["content"]), usage=usage)
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
    "mutual fund", "etf", "n-port", "fund holdings",
    "dividend", "shareholder", "stockholder", "repurchase", "buyback",
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


def policy_refusal(message: str) -> dict[str, Any]:
    """Advice, forecasts, personal or non-public data and creative requests: refused by rule, before any LLM call."""
    return guard_checks.policy_refusal(message, guard_patterns.FINANCIAL_POLICY_REFUSALS,
                                       guard_patterns.POLICY_REFUSAL_MESSAGES)


def names_reported_measure(message: str) -> bool:
    """The question names a company or sector the copilot knows and a measure companies report."""
    from scripts.agents.financial_clarifier import default_clarifier
    clarifier = default_clarifier()
    return bool(clarifier.metric_in(message) and (clarifier.companies_in(message) or clarifier.sectors_in(message)))


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
    # Hosted Qwen3-8B marked ordinary questions is_plain_text=false while calling them low-risk financial questions,
    # which refused them. Code, markup, encoded payloads and injection are what the rule guard looks for; when it
    # finds none, the message is plain text. The LLM's own view is kept for audit.
    rule_clean = not deterministic_input_guard(message)["risk_labels"]
    parsed["llm_is_plain_text"] = parsed.get("is_plain_text")
    parsed["is_plain_text"] = bool(parsed.get("is_plain_text")) or rule_clean
    # Hosted Qwen3-8B called "Compare Costco and Walmart revenue in 2025" not a financial-report question and refused it.
    # A message the rule guard finds clean that names a known company or sector and a reported measure is one.
    parsed["llm_financial_intent"] = bool(parsed.get("financial_intent"))
    parsed["financial_intent"] = bool(parsed.get("financial_intent")) or (rule_clean and names_reported_measure(message))
    parsed["risk_level"] = str(parsed.get("risk_level", "medium")).lower()
    # The classifier only decides; it never rewrites the question. Hosted Qwen3-8B returned safe_query=true (-> "True")
    # and the graph then planned and answered "True". The LLM's suggestion is kept for audit only.
    suggested = parsed.get("safe_query")
    parsed["llm_safe_query"] = suggested if isinstance(suggested, str) else None
    parsed["safe_query"] = canonicalize_user_message(message)
    # Hosted Qwen3-8B rated "What was Bank of America's average trading VaR in 2025?" risk_level=medium every time:
    # it judged the financial topic (value at risk), not the request. Injection, code and encoded payloads are the rule
    # guard's job and advice/forecasts the policy rules', both before this step; so an LLM "medium" refuses only when
    # the rule guard also flagged the message. "high" and "critical" still refuse.
    parsed["allowed"] = (
        parsed["is_plain_text"]
        and parsed["financial_intent"]
        and parsed["risk_level"] not in {"high", "critical"}
        and (parsed["risk_level"] != "medium" or rule_clean)
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
    if guard_checks.has_term(query, INVESTMENT_REVIEW_TERMS):      # whole words: not "stockholders'" for "hold"
        triggers.append("investment_or_decision_support_language")
    if guard_checks.has_term(query, PUBLICATION_REVIEW_TERMS):
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
    policy = policy_refusal(config.query)
    if policy["refuse"]:
        return None, refusal_result(technique, policy["message"], security={**security, "policy": policy})

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
    hits = guard_checks.relevance_floor(retrieve_evidence(config), float(os.getenv("RAG_MIN_RERANK_SCORE", "-4")))
    return financial_checks.clean_chart_tables(hits)


def statement_page_hits(pages: list[tuple[str, int]], hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The other chunks of statement pages the search found by their title (Chroma backends only; [] elsewhere)."""
    if not pages or os.getenv("RAG_BACKEND") not in ("financial_vlm", "financial_hybrid"):
        return []
    from scripts.rag.financial_ingest_chroma_db import page_hits
    return financial_checks.clean_chart_tables([name_fiscal_year(hit) for hit in page_hits(pages, hits)])


def page_text_before(hit: dict[str, Any]) -> str:
    """The text of a prose chunk's page above it (Chroma backends; "" elsewhere), for the heading it continues under."""
    meta = hit.get("metadata") or {}
    index = meta.get("chunk_index")
    if hit.get("content_type") != "text" or not meta.get("doc_id") or hit.get("page") is None or not index:
        return ""
    page = statement_page_hits([(str(meta["doc_id"]), int(hit["page"]))], [])
    return "\n".join(str(h.get("text") or "") for h in page
                     if int((h.get("metadata") or {}).get("chunk_index") or 0) < int(index))


def across_targets(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Evidence in turns across the searches (company-years), each in its own rank order: the context limit then cuts
    every target's weakest hits, not the last targets entirely (Chevron's 2024 and 2025 pages of a 4-year question)."""
    queues: dict[Any, list] = {}
    for hit in evidence:
        queues.setdefault(hit.get("target"), []).append(hit)
    if len(queues) <= 1:
        return evidence
    out, position = [], 0
    while len(out) < len(evidence):
        out += [queue[position] for queue in queues.values() if position < len(queue)]
        position += 1
    return out


def name_fiscal_year(hit: dict[str, Any]) -> dict[str, Any]:
    """The hit's fiscal year as the company names it (Home Depot's filing year 2026 is its fiscal 2025)."""
    year = hit.get("year")
    return hit if not isinstance(year, int) or fiscal_label(hit.get("ticker"), year) == year else {
        **hit, "year": fiscal_label(hit.get("ticker"), year)}


def retrieve_evidence(config: AgentConfig) -> list[dict[str, Any]]:
    """One search on the served backend, traced to LangSmith as a retriever run ``retrieve.<backend>``."""
    backend = os.getenv("RAG_BACKEND") or "chroma"
    asked = config
    if config.ticker and config.year and filing_year(config.ticker, config.year) != config.year:
        config = AgentConfig(**{**config.__dict__, "year": filing_year(config.ticker, config.year)})
    return traced_retrieval(
        f"retrieve.{backend}", lambda: [name_fiscal_year(hit) for hit in _retrieve_evidence(config)], query=asked.query,
        filters={key: getattr(config, attr, None) for key, attr in (
            ("ticker", "ticker"), ("year", "year"), ("sector", "sector"), ("top_k", "top_k"), ("visual", "visual"),
            ("mode", "retrieval_mode"))},
        metadata={"rag_backend": backend})


def _retrieve_evidence(config: AgentConfig) -> list[dict[str, Any]]:
    if os.getenv("RAG_BACKEND") == "financial_qdrant":   # collections built by scripts/rag/financial_ingest_qdrant_db.py
        from scripts.rag.financial_ingest_qdrant_db import retrieve_for_agent
        return retrieve_for_agent(config.query, config.top_k, ticker=config.ticker, year=config.year,
                                  sector=config.sector, visual=config.visual, mode=config.retrieval_mode)
    if os.getenv("RAG_BACKEND") == "financial_hybrid":   # Chroma page readings + ColPali page finding (both indexes)
        from scripts.rag import financial_hybrid
        return financial_hybrid.retrieve_for_agent(config.query, config.top_k, ticker=config.ticker, year=config.year,
                                                   sector=config.sector, visual=config.visual, mode=config.retrieval_mode)
    if os.getenv("RAG_BACKEND") == "financial_vlm":      # Chroma index built by scripts/rag/financial_ingest_chroma_db.py
        from scripts.rag import financial_ingest_chroma_db
        return financial_ingest_chroma_db.retrieve_for_agent(config.query, config.top_k, ticker=config.ticker, year=config.year,
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
        r"nport_[A-Za-z0-9_.=-]+",                 # fund rows (agent_mutual_fund)
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
    route: dict[str, Any]
    trend: dict[str, Any]
    sector_trend: dict[str, Any]
    shareholder_returns: dict[str, Any]
    dividend_income: dict[str, Any]
    mutual_fund: dict[str, Any]
    handoff: dict[str, Any]


STOPPED = {"refused", "insufficient_evidence", "not_available", "report_handoff"}
# Nodes every run still needs after the evidence guard: answer, critic, governance (+ calculator when planned).
MIN_TAIL = 3
AGENT_FLOWS = ("agentic_graph", "trend_graph", "sector_trend_graph", "fund_graph", "shareholder_returns_graph",
               "dividend_income_graph", "router")
ACTION_ROUTES = ("shareholder_returns", "dividend_income")      # agents whose answer is a table computed in code


def agentic_graph_prompt(llm, config: AgentConfig) -> dict[str, Any]:
    """Q&A agent: planner -> executor -> critic (see ``financial_graph``)."""
    return financial_graph(llm, config, "agentic_graph")


def trend_graph_prompt(llm, config: AgentConfig) -> dict[str, Any]:
    """Multi-year trend agent: one search per fiscal year, a verified series, changes computed in code."""
    return financial_graph(llm, config, "trend_graph")


def sector_trend_graph_prompt(llm, config: AgentConfig) -> dict[str, Any]:
    """Sector trend agent: one measure across one sector's companies, year by year, over a fixed panel."""
    return financial_graph(llm, config, "sector_trend_graph")


def fund_graph_prompt(llm, config: AgentConfig) -> dict[str, Any]:
    """Fund agent: a mutual fund's N-PORT holdings (fixed SQL rows as evidence), linked to the holdings' annual reports."""
    return financial_graph(llm, config, "fund_graph")


def shareholder_returns_graph_prompt(llm, config: AgentConfig) -> dict[str, Any]:
    """Shareholder-returns agent: dividends plus buybacks per company for one fiscal year, ranked in code."""
    return financial_graph(llm, config, "shareholder_returns_graph")


def dividend_income_graph_prompt(llm, config: AgentConfig) -> dict[str, Any]:
    """Dividend-income agent: the user's share counts times each company's verified dividends per share."""
    return financial_graph(llm, config, "dividend_income_graph")


def router_graph_prompt(llm, config: AgentConfig) -> dict[str, Any]:
    """Supervisor: routes each question to the Q&A agent, the trend agent, the report handoff or a refusal."""
    return financial_graph(llm, config, "router")


def financial_graph(llm, config: AgentConfig, flow: str = "agentic_graph") -> dict[str, Any]:
    """The financial agent graphs. All three share the input guards, the critic (citations + support check, one
    repair) and governance; they differ in how a request is planned:

    * ``agentic_graph``: security -> plain_text -> plan -> retrieve <-> evidence_guard -> answer -> calculator? ->
      critic <-> governance. The planner LLM picks the tools; a company/year with no evidence gets one broadened
      retrieval retry; CALC[...] goes to the calculator; failed citations or support get one repair.
    * ``router``: security -> plain_text -> route, where the router LLM (validated by ``financial_router.decide``)
      sends the question to the Q&A path above (from retrieve), to the trend path, to a report handoff or to a
      refusal. The route step also plans, so a routed run costs no extra step.
    * ``trend_graph``: the router with the trend route forced (falls back to Q&A when no trend can be built).
      Trend path: trend_fetch (one search per fiscal year) -> trend_series (LLM copies each year's figure, code
      checks it is in that year's evidence, computes changes and CAGR) <-> retry missing years -> answer -> critic.
    * ``sector_trend_graph``: the router with the sector trend route forced. Sector path: sector_fetch (one search
      per company x year) -> sector_series (statement row read in code, else the LLM copies it; code checks each
      figure is in that company-year's evidence; one broadened search per figure still missing, inside the node so
      the run stays within AGENT_MAX_STEPS; fixed panel, totals, changes and CAGR in code) -> answer -> critic.
    * ``fund_graph``: the router with the fund route forced. Fund path: fund_lookup (fixed SQL over the N-PORT tables,
      each row an evidence item) -> fund_company (only for questions about the holdings' annual reports: one filtered
      search per company, statement row in code, LLM last, figures cached; weighted growth in code) -> answer -> critic.

    * ``shareholder_returns_graph``: the router with the shareholder-returns route forced. action_fetch (one search
      per company x measure: dividends paid, share repurchases) -> action_series (cash-flow row read in code; one
      broadened search per figure still missing, then the row again and only then the LLM, kept only on a line
      naming the measure; dividends + buybacks, ranking and shares computed in code) -> answer (written in code) ->
      critic (figures checked in code) -> governance.
    * ``dividend_income_graph``: the router with the dividend-income route forced. action_fetch (one search per
      holding for dividends declared per share) -> action_series (per-share row in code, retry, LLM last; then the
      action: shares x dividend per share, in code, the share count recorded as the user's input) -> answer (in
      code) -> critic -> governance.

    A loop is taken only if the rest of the run still fits AGENT_MAX_STEPS, so the cap shortens a run instead of
    failing it. ``agent_run`` records the path, every tool call and every calculation.
    """
    from scripts.agents import financial_router as router, financial_tools as tools, financial_trend as trend
    from scripts.agents import financial_sector_trend as sector_agent
    from scripts.agents import financial_shareholder_returns as returns_agent
    from scripts.agents import financial_dividend_income as income_agent
    from scripts.agents import agent_mutual_fund as fund_agent
    from scripts.prompts import prompts_mutual_fund as fund_prompts
    from scripts.common.companies import NAMES
    from scripts.agents.financial_clarifier import METRICS, default_clarifier

    if flow not in AGENT_FLOWS:
        raise ValueError(f"Unsupported agent flow: {flow}")

    clarifier = default_clarifier()
    budget = max_steps()
    fan_out_log: list[dict] = []          # parallel search rounds of this run (agent_run.search_fan_out)
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

    subject = clarifier.subject_in(config.query) if hasattr(clarifier, "subject_in") else None   # "data center" revenue
    term = financial_checks.defined_term(config.query)                  # "how does X define conduct risk"

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
        policy = policy_refusal(state["query"])
        if policy["refuse"]:                   # decided by rule: the model is never asked whether to refuse
            return {"security": {**security, "policy": policy}, "final_status": "refused", "answer": policy["message"]}
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

    def route_node(state: AgentState) -> AgentState:
        """Router (supervisor): choose the flow, then plan it. The router LLM's tools become the Q&A plan."""
        query = state["query"]
        raw = None
        if flow == "router":
            raw = safe_json_loads(invoke_prompt(llm, prompts.ROUTER_SYSTEM, prompts.ROUTER_HUMAN.format(query=query)),
                                  fallback={})
        started = time.perf_counter()
        trend_plan = trend.trend_plan(query, config.ticker, config.year, clarifier)
        sector_plan = sector_agent.sector_plan(query, clarifier, sector_filter=config.sector)
        forced = {"trend_graph": "trend", "sector_trend_graph": "sector_trend", "fund_graph": "fund",
                  "shareholder_returns_graph": "shareholder_returns", "dividend_income_graph": "dividend_income"}.get(flow)
        decision = router.decide(raw, query, clarifier, trend_plan, ticker=config.ticker, forced=forced,
                                 sector_plan=sector_plan, sector=config.sector)
        route = decision["route"]
        if route == "fund":
            fund_plan = fund_agent.fund_plan(query, clarifier, ticker=config.ticker)
            calls_log = [*state.get("tool_calls", []), {
                "tool": "fund_search", "attempt": 1, "kind": fund_plan.get("kind"), "fund": fund_plan.get("fund"),
                "period_end": fund_plan.get("period_end"),
                "method": (fund_plan.get("fund_search") or {}).get("method", "names"),
                "ms": round((time.perf_counter() - started) * 1000)}]
            if not fund_plan["feasible"]:
                return {"route": decision, "plan": fund_plan, "tool_calls": calls_log,
                        "final_status": "refused" if fund_plan.get("advice") else "not_available",
                        "answer": fund_agent.not_available_message(fund_plan)}
            return {"route": decision, "plan": fund_plan, "tool_calls": calls_log}
        if route in ACTION_ROUTES:
            if route == "shareholder_returns":
                action_plan = returns_agent.returns_plan(query, clarifier, ticker=config.ticker,
                                                         sector_filter=config.sector, year=config.year)
                message = returns_agent.not_available_message
                logged = {"scope": action_plan["scope"], "sector": action_plan["sector"],
                          "companies": action_plan["companies"], "year": action_plan["year"]}
            else:
                action_plan = income_agent.income_plan(query, clarifier, ticker=config.ticker, year=config.year)
                message = income_agent.not_available_message
                logged = {"holdings": [{k: h.get(k) for k in ("ticker", "shares", "year")}
                                       for h in action_plan["holdings"]]}
            calls_log = [*state.get("tool_calls", []), {
                "tool": "coverage_lookup", "attempt": 1, "known": bool(clarifier.directory), **logged,
                "excluded": action_plan["excluded"], "ms": round((time.perf_counter() - started) * 1000)}]
            if not action_plan["feasible"]:
                return {"route": decision, "plan": action_plan, "tool_calls": calls_log,
                        "final_status": "not_available", "answer": message(action_plan)}
            return {"route": decision, "plan": action_plan, "tool_calls": calls_log}
        if route == "sector_trend":
            calls_log = [*state.get("tool_calls", []), {
                "tool": "coverage_lookup", "attempt": 1, "known": bool(clarifier.directory),
                "sector": sector_plan["sector"], "companies": sector_plan["companies"], "years": sector_plan["years"],
                "excluded": sector_plan["excluded"], "ms": round((time.perf_counter() - started) * 1000)}]
            if not sector_plan["feasible"]:
                return {"route": decision, "plan": sector_plan, "tool_calls": calls_log, "final_status": "not_available",
                        "answer": sector_agent.not_available_message(sector_plan)}
            return {"route": decision, "plan": sector_plan, "tool_calls": calls_log}
        if route == "trend":
            calls_log = [*state.get("tool_calls", []), {
                "tool": "coverage_lookup", "attempt": 1, "known": bool(clarifier.directory),
                "searchable": [tools.target_label(t) for t in trend_plan["targets"]],
                "skipped": trend_plan["skipped_targets"], "ms": round((time.perf_counter() - started) * 1000)}]
            return {"route": decision, "plan": trend_plan, "tool_calls": calls_log}
        if route == "report":
            handoff = router.report_handoff(query, clarifier, ticker=config.ticker, year=config.year)
            return {"route": decision, "handoff": handoff, "final_status": "report_handoff", "answer": handoff["message"]}
        if route == "out_of_scope":
            return {"route": decision, "final_status": "refused", "answer": router.OUT_OF_SCOPE_ANSWER}
        plan = tools.normalize_plan(raw, query, ticker=config.ticker, year=config.year,
                                    source_type=config.source_type, clarifier=clarifier)
        plan["intent"] = route
        return {"route": decision, "plan": plan}

    def search_job(label: str, query: str, top_k: int, tool: str = "text_search", attempt: int = 1,
                   **overrides) -> dict:
        """One filtered search for ``run_searches``: its target label, config and log fields."""
        run_config = AgentConfig(**{**config.__dict__, "query": query, "top_k": top_k, "visual": False, **overrides})
        return {"label": label, "config": run_config, "tool": tool, "attempt": attempt, "query": query, "top_k": top_k}

    def run_searches(jobs: list[dict], seen: set, calls_log: list[dict]) -> list[list[dict]]:
        """Fetch every job's hits (up to RAG_SEARCH_WORKERS at once, ``tools.fan_out``), then, in job order, keep the
        hits not seen before and log each search: the evidence and its order are what a one-by-one loop gives."""
        def fetch(job):
            started = time.perf_counter()
            hits = answer_evidence(job["config"])
            return hits, round((time.perf_counter() - started) * 1000)
        results, fan_log = tools.fan_out(fetch, jobs)
        fresh_all = []
        for job, (hits, ms) in zip(jobs, results):
            fresh = []
            for hit in hits:
                if hit.get("chunk_id") not in seen:
                    seen.add(hit.get("chunk_id"))
                    fresh.append({**hit, "target": job["label"]})
            fresh_all.append(fresh)
            calls_log.append({"tool": job["tool"], "attempt": job["attempt"], "target": job["label"],
                              "query": job["query"], "top_k": job["top_k"], "hits": len(hits), "new": len(fresh),
                              "ms": ms})
        if fan_log["workers"] > 1:
            fan_out_log.append(fan_log)
        return fresh_all

    def trend_fetch_node(state: AgentState) -> AgentState:
        """One filtered search per fiscal year. A retry searches only the years without a verified figure, with a
        broadened query and twice the depth."""
        plan = state.get("plan", {})
        attempt = state.get("retrieval_attempts", 0) + 1
        targets = plan.get("targets") or []
        per_target = max(3, -(-config.top_k // max(1, len(targets)))) * (2 if attempt > 1 else 1)
        if attempt > 1:
            missing = set(state.get("evidence_check", {}).get("missing", []))
            targets = [t for t in targets if tools.target_label(t) in missing]
        evidence = list(state.get("evidence", []))
        seen = {hit.get("chunk_id") for hit in evidence}
        calls_log = list(state.get("tool_calls", []))
        phrases = METRICS.get(plan.get("metric"), [])
        jobs = [search_job(tools.target_label(t), trend.year_query(plan, t["year"], phrases, broadened=attempt > 1),
                           per_target, attempt=attempt, ticker=t["ticker"], year=t["year"]) for t in targets]
        if attempt > 1 and sector_agent.rows_query(plan.get("metric")):
            jobs += [rows_job(plan, t) for t in targets]
            targets = [*targets, *targets]
        for target, fresh in zip(targets, run_searches(jobs, seen, calls_log)):
            evidence.extend(fresh)
            evidence.extend(statement_hits(evidence, plan.get("metric"), tools.target_label(target), seen))
        return {"evidence": evidence, "retrieval_attempts": attempt, "tool_calls": calls_log}

    def trend_series_node(state: AgentState) -> AgentState:
        """Evidence guard, then each year's figure: the statement row read in code, else copied by the LLM and kept only
        if that year's evidence states it. A figure another year's filing contradicts is dropped.
        Changes and CAGR are computed in code over the verified years."""
        evidence = state.get("evidence", [])
        guard = evidence_prompt_injection_guard(evidence)
        if not guard["safe"]:
            return {"evidence_guard": guard, "final_status": "refused",
                    "answer": "Retrieved evidence contained unsafe instructions, so the answer was not generated."}
        plan = state.get("plan", {})
        previous = {p["year"]: p for p in (state.get("trend") or {}).get("points", [])}
        attempt = state.get("retrieval_attempts", 1)
        calls_log = list(state.get("tool_calls", []))
        points = []
        for target in plan.get("targets") or []:
            year, label = target["year"], tools.target_label(target)
            if previous.get(year, {}).get("status") == "ok":
                points.append(previous[year])
                continue
            hits = [hit for hit in evidence if hit.get("target") == label]
            if not hits:
                points.append({"year": year, "value": None, "value_text": None, "unit": "", "chunk_id": None,
                               "status": "no_evidence"})
                continue
            # The statement row in code; the LLM only for measures without a known row, or once the retry found none.
            last = attempt >= tools.MAX_RETRIEVAL_ATTEMPTS or not can_retry(
                {**state, "evidence_check": {"missing": [label]}, "path": [*state.get("path", []), "x"]})
            point, log = sector_figure(plan, plan["company"], year, hits, query=state["query"],
                                       use_llm=last or not has_statement_row(plan.get("metric")))
            points.append(point)
            calls_log.append({**log, "attempt": attempt, "target": label})
        points = reject_conflicts(points, calls_log, plan["company"])
        if attempt >= tools.MAX_RETRIEVAL_ATTEMPTS:
            points = fill_from_later_filing(points, evidence, calls_log, plan["company"])
        summary = trend.summarize(plan, points)
        calls_log.append({"tool": "trend_calculator", "attempt": attempt, "verified_years": summary["verified_points"],
                          "calculations": len(summary["calculations"]), "cagr": summary["cagr"]})
        check = {"attempt": attempt, "evidence": len(evidence), "missing": summary["missing"],
                 "skipped": [tools.target_label(t) for t in plan.get("skipped_targets") or []]}
        update: AgentState = {"evidence_guard": guard, "evidence_check": check, "tool_calls": calls_log,
                              "trend": {k: v for k, v in summary.items() if k != "calculations"},
                              "calculations": summary["calculations"]}
        if not evidence and not can_retry({**state, "evidence_check": check, "path": [*state.get("path", []), "x"]}):
            update.update(final_status="insufficient_evidence", answer="No answer found in the available evidence.")
        return update

    def sector_job(plan: dict, target: dict, attempt: int) -> dict:
        """The filtered search for a company-year; a retry broadens the query and searches twice as deep."""
        query = sector_agent.company_query(plan, target["ticker"], target["year"], METRICS.get(plan.get("metric"), []),
                                           broadened=attempt > 1)
        return search_job(tools.target_label(target), query, 3 * (2 if attempt > 1 else 1), attempt=attempt,
                          ticker=target["ticker"], year=target["year"], sector=None)

    def rows_job(plan: dict, target: dict) -> dict:
        """The retry's other search: the statement's own rows, in that company-year."""
        return search_job(tools.target_label(target), sector_agent.rows_query(plan.get("metric")), 4, attempt=2,
                          tool="statement_search", ticker=target["ticker"], year=target["year"], sector=None)

    def sector_fetch_node(state: AgentState) -> AgentState:
        """One filtered search per company x fiscal year of the plan, run in parallel (``run_searches``)."""
        plan = state.get("plan", {})
        evidence, calls_log = [], list(state.get("tool_calls", []))
        seen: set = set()
        targets = plan.get("targets") or []
        found = run_searches([sector_job(plan, t, 1) for t in targets], seen, calls_log)
        pages = statement_hits_all([(fresh, tools.target_label(t)) for t, fresh in zip(targets, found)],
                                   plan.get("metric"), seen)
        for fresh, more in zip(found, pages):
            evidence.extend(fresh)
            evidence.extend(more)
        return {"evidence": evidence, "retrieval_attempts": 1, "tool_calls": calls_log}

    def has_statement_row(metric: str | None) -> bool:
        return metric in sector_agent.STATEMENT_LABELS or metric in returns_agent.CASH_FLOW_LABELS

    def statement_row(hits: list[dict], metric: str | None, year: int, query: str = "",
                      exclude: frozenset | set = frozenset()) -> dict | None:
        """The measure's row for ``year`` read in code: a statement row, or a cash-flow row for dividends and buybacks
        (read and checked as the returns agent reads it; the figure is printed without the minus sign)."""
        if metric in returns_agent.CASH_FLOW_LABELS:
            row = returns_agent.read_row([h for h in hits if str(h.get("chunk_id")) not in exclude], metric, year)
            if row and NET_OF_ISSUES.match(str(row.get("quote") or "")):
                return None                       # Chevron's row nets shares issued (11,855); paid for repurchases: 12,079
            return {**row, "value": row["value"].strip("()")} if row else None
        return sector_agent.label_match(hits, metric, year, sector_agent.labels_for(metric, query), exclude)

    def statement_hits(hits: list[dict], metric: str | None, label: str, seen: set) -> list[dict]:
        """The rest of the statement pages among a target's hits (``sector_agent.statement_pages``): the chunk with the
        statement's title ranks high, the one with the figures often doesn't. Marks the pages' own hits too."""
        return statement_hits_all([(hits, label)], metric, seen)[0]

    def statement_hits_all(groups: list[tuple[list[dict], str]], metric: str | None, seen: set) -> list[list[dict]]:
        """``statement_hits`` of several targets: the pages are read in parallel (a sector reads up to six for each of
        50 company-years, ~0.13 s each), then chunks already ``seen`` are dropped in target order, as in a loop."""
        if not has_statement_row(metric):
            return [[] for _ in groups]
        looked_up, _ = tools.fan_out(lambda group: statement_lookup(group[0], metric, group[1]), groups)
        out = []
        for (_, label), found in zip(groups, looked_up):
            more = [h for h in found if h.get("chunk_id") not in seen]
            seen.update(h.get("chunk_id") for h in more)
            out.append([{**h, "target": label, "statement_page": True} for h in drop_flagged(more)[0]])
        return out

    def statement_lookup(hits: list[dict], metric: str | None, label: str) -> list[dict]:
        """One target's statement-page chunks, before duplicates are dropped."""
        own = [h for h in hits if h.get("target") == label]
        pages = sector_agent.statement_pages(own, metric)
        for hit in (own[-6:] if not pages else []):
            # A later chunk of the statement ranks, its title is in the page's first one (Bank of America 2024, p136).
            meta = hit.get("metadata") or {}
            if meta.get("doc_id") and hit.get("page") is not None and hit.get("content_type") in ("text", "table"):
                pages = sector_agent.statement_pages(statement_page_hits([(str(meta["doc_id"]), int(hit["page"]))], []), metric)
                pages = [p for p in pages if p == (str(meta["doc_id"]), int(hit["page"]))]
            if pages:
                break
        for hit in own:
            if ((hit.get("metadata") or {}).get("doc_id"), hit.get("page")) in pages:
                hit["statement_page"] = True
        return statement_page_hits(pages, own)

    def reject_conflicts(points: list[dict], calls_log: list[dict], ticker: str) -> list[dict]:
        """Drop a figure that another year's filing contradicts (``sector_agent.prior_conflicts``)."""
        points = sector_agent.reconcile_units(points)
        doubted = sector_agent.prior_conflicts({p["year"]: p for p in points})
        for year in sorted(doubted):
            calls_log.append({"tool": "prior_year_check", "target": f"{ticker} FY{year}", "status": "conflict"})
        return [{**p, "value": None, "status": "contradicted_by_another_filing", "rejected_chunk": p.get("chunk_id")}
                if p["year"] in doubted else p for p in points]

    def fill_from_later_filing(points: list[dict], evidence: list[dict], calls_log: list[dict], ticker: str) -> list[dict]:
        """A year whose own filing can't be read (NVIDIA's and Walmart's FY2023 statements have a damaged text layer):
        the figure the next year's filing prints for it, in the same verified row, cited to that filing."""
        ok = {p["year"]: p for p in points if p.get("status") == "ok" and p.get("columns")}
        out = []
        for point in points:
            year = point["year"]
            later = next((ok[y] for y in (year + 1, year + 2) if y in ok and year in ok[y]["columns"]), None)
            if point.get("status") == "ok" or later is None:
                out.append(point)
                continue
            hit = next((h for h in evidence if str(h.get("chunk_id")) == str(later.get("chunk_id"))), None)
            filled = trend.validate_point({"value": later["columns"][year], "unit": later.get("unit"),
                                           "chunk_id": later.get("chunk_id"), "quote": later.get("quote")},
                                          [hit] if hit else [], year)
            if filled.get("status") == "ok":
                filled.update(method="later_filing", reported_in=later["year"])
                calls_log.append({"tool": "prior_year_column", "target": f"{ticker} FY{year}", "status": "ok",
                                  "value": filled["value_text"], "chunk_id": filled["chunk_id"],
                                  "reported_in": later["year"]})
                point = filled
            out.append(point)
        return out

    def sector_figure(plan: dict, ticker: str, year: int, hits: list[dict], use_llm: bool, query: str = "",
                      exclude: frozenset | set = frozenset()) -> tuple[dict, dict]:
        """The statement row read in code; with ``use_llm`` and no verified row, the LLM copies the figure, kept only
        if that company-year's evidence states it on a line naming the measure."""
        started = time.perf_counter()
        hits = [h for h in hits if str(h.get("chunk_id")) not in exclude]
        raw = statement_row(hits, plan.get("metric"), year, query)
        point = trend.validate_point(raw, hits, year) if raw else {"year": year, "value": None, "value_text": None,
                                                                   "unit": "", "chunk_id": None, "status": "not_found"}
        method = "label"
        if point.get("status") != "ok" and use_llm:
            raw = safe_json_loads(invoke_prompt(llm, prompts.SERIES_SYSTEM, prompts.SERIES_HUMAN.format(
                company=NAMES.get(ticker, ticker), metric=plan.get("metric"), year=year,
                context=evidence_to_context(hits, min(config.max_context_chars, trend.SERIES_CONTEXT_CHARS)))),
                fallback={})
            point, method = trend.validate_point(raw, hits, year), "llm"
            if (point.get("status") == "ok" and has_statement_row(plan.get("metric"))
                    and not sector_agent.on_metric_line(point, hits, METRICS.get(plan.get("metric"), []))):
                point = {**point, "value": None, "status": "not_on_metric_line"}
            if (point.get("status") == "ok" and has_statement_row(plan.get("metric"))
                    and not sector_agent.row_position_ok(point, hits, year)):
                point = {**point, "value": None, "status": "not_this_years_column"}
            if point.get("status") == "ok" and sector_agent.other_companys_row(point, hits):
                point = {**point, "value": None, "status": "segment_or_parent_row"}
        point["method"] = method
        return point, {"tool": "series_extract", "target": f"{ticker} FY{year}", "method": method,
                       "status": point["status"], "value": point.get("value_text"), "chunk_id": point.get("chunk_id"),
                       "ms": round((time.perf_counter() - started) * 1000)}

    def drop_flagged(evidence: list[dict]) -> tuple[list[dict], dict]:
        """Chunks the injection guard flags are dropped (never shown to the LLM); the rest of the sector stands."""
        guard = evidence_prompt_injection_guard(evidence)
        if guard["safe"]:
            return evidence, guard
        flagged = {str(f.get("chunk_id") if isinstance(f, dict) else f) for f in guard.get("flagged_chunks") or []}
        kept = [hit for hit in evidence if str(hit.get("chunk_id")) not in flagged]
        return kept, {**guard, "safe": True, "action": "dropped_flagged_chunks", "dropped": sorted(flagged)}

    def sector_series_node(state: AgentState) -> AgentState:
        """Every company-year's figure, code first: the statement row; then one broadened, deeper search and the row
        again; only then the LLM. Flagged chunks are dropped. Then the fixed panel and the sector table in code."""
        evidence, guard = drop_flagged(list(state.get("evidence", [])))
        plan = state.get("plan", {})
        calls_log = list(state.get("tool_calls", []))
        seen = {hit.get("chunk_id") for hit in state.get("evidence", [])}
        dropped = list(guard.get("dropped") or [])
        # Round 1: every company-year's statement row, in code. Round 2: one broadened search for each figure still
        # missing, all in parallel, then the row again and only then the LLM (one call at a time).
        found: dict[tuple, dict] = {}
        for ticker in plan.get("companies") or []:
            for year in plan.get("years") or []:
                label = tools.target_label({"ticker": ticker, "year": year})
                point, log = sector_figure(plan, ticker, year, [h for h in evidence if h.get("target") == label],
                                           use_llm=False, query=state["query"])
                calls_log.append({**log, "attempt": 1})
                found[(ticker, year)] = point
        rejected: dict[tuple, str] = {}
        for ticker in plan.get("companies") or []:
            for point in reject_conflicts([found[(ticker, y)] for y in plan.get("years") or []], calls_log, ticker):
                if point.get("rejected_chunk"):
                    rejected[(ticker, point["year"])] = str(point["rejected_chunk"])
                found[(ticker, point["year"])] = point
        missing = [{"ticker": t, "year": y} for (t, y), point in found.items() if point.get("status") != "ok"]
        retried = len(missing)
        extra = [rows_job(plan, t) for t in missing] if sector_agent.rows_query(plan.get("metric")) else []
        retries = [*missing, *(missing if extra else [])]
        fetched = run_searches([*(sector_job(plan, t, 2) for t in missing), *extra], seen, calls_log)
        pages = statement_hits_all([(fresh, tools.target_label(t)) for t, fresh in zip(retries, fetched)],
                                   plan.get("metric"), seen)
        for fresh, more in zip(fetched, pages):
            fresh, fresh_guard = drop_flagged([*fresh, *more])
            dropped += fresh_guard.get("dropped") or []
            evidence.extend(fresh)
        for target in missing:
            hits = [hit for hit in evidence if hit.get("target") == tools.target_label(target)]
            if hits:
                # The row in code, then the next filing's column for this year, and only then the LLM.
                key = (target["ticker"], target["year"])
                found[key], log = sector_figure(plan, target["ticker"], target["year"], hits, use_llm=False,
                                                query=state["query"],
                                                exclude={rejected[key]} if key in rejected else frozenset())
                calls_log.append({**log, "attempt": 2})
        for ticker in {t["ticker"] for t in missing}:
            points = reject_conflicts([found[(ticker, y)] for y in plan.get("years") or []], calls_log, ticker)
            for point in fill_from_later_filing(points, evidence, calls_log, ticker):
                found[(ticker, point["year"])] = point
        for target in missing:
            hits = [hit for hit in evidence if hit.get("target") == tools.target_label(target)]
            if hits and found[(target["ticker"], target["year"])].get("status") != "ok":
                key = (target["ticker"], target["year"])
                point, log = sector_figure(plan, target["ticker"], target["year"], hits, use_llm=True, query=state["query"],
                                           exclude={rejected[key]} if key in rejected else frozenset())
                calls_log.append({**log, "attempt": 2})
                found[key] = point
        for ticker in plan.get("companies") or []:                # a retried figure must agree with its neighbours too
            for point in reject_conflicts([found[(ticker, y)] for y in plan.get("years") or []], calls_log, ticker):
                found[(ticker, point["year"])] = point
        series = {}
        for ticker in plan.get("companies") or []:
            company_plan = {"company": ticker, "company_name": NAMES.get(ticker, ticker), "metric": plan.get("metric"),
                            "years": plan.get("years") or []}
            series[ticker] = trend.summarize(company_plan, [found[(ticker, y)] for y in plan.get("years") or []])
        if dropped:
            guard = {**guard, "action": "dropped_flagged_chunks", "dropped": sorted(set(dropped))}
        table = sector_agent.aggregate(plan, series)
        calls_log.append({"tool": "sector_calculator", "panel": table["panel"], "years": table["years"],
                          "calculations": len(table["calculations"]), "cagr_total": table["cagr_total"],
                          "unusual_changes": len(table["unusual_changes"])})
        check = {"attempt": 2 if retried else 1, "evidence": len(evidence), "missing": table["missing"],
                 "retried": retried, "dropped_chunks": len(guard.get("dropped") or []),
                 "skipped": [f"FY{y}" for y in plan.get("skipped_years") or []]}
        update: AgentState = {"evidence": evidence, "evidence_guard": guard, "evidence_check": check,
                              "tool_calls": calls_log, "retrieval_attempts": 2 if retried else 1,
                              "sector_trend": {k: v for k, v in table.items() if k != "calculations"},
                              "calculations": table["calculations"]}
        if len(table["panel"]) < sector_agent.MIN_PANEL:
            update.update(final_status="insufficient_evidence", answer=(
                f"No sector trend: {plan.get('metric')} could be verified for every fiscal year for "
                f"{len(table['panel'])} {plan.get('sector')} compan{'y' if len(table['panel']) == 1 else 'ies'}, and "
                f"a sector trend needs at least {sector_agent.MIN_PANEL}."))
        return update

    def action_agent(state: AgentState):
        return returns_agent if state.get("plan", {}).get("intent") == "shareholder_returns" else income_agent

    def action_query(agent, target: dict, broadened: bool) -> str:
        if agent is returns_agent:
            return returns_agent.search_query(target["ticker"], target["year"], target["measure"], broadened)
        return income_agent.search_query(target["ticker"], target["year"], broadened)

    def action_fetch_node(state: AgentState) -> AgentState:
        """One filtered search per target (company x measure, or holding), run in parallel (``run_searches``)."""
        agent, plan = action_agent(state), state.get("plan", {})
        evidence, calls_log = [], list(state.get("tool_calls", []))
        jobs = [search_job(agent.target_label(t), action_query(agent, t, False), 3, ticker=t["ticker"], year=t["year"],
                           sector=None) for t in plan.get("targets") or []]
        seen: set = set()
        for target, fresh in zip(plan.get("targets") or [], run_searches(jobs, seen, calls_log)):
            evidence.extend(fresh)
            evidence.extend(statement_hits(fresh, target.get("measure"), agent.target_label(target), seen))
        return {"evidence": evidence, "retrieval_attempts": 1, "tool_calls": calls_log}

    def action_figures(state: AgentState, agent) -> tuple[list[dict], dict, list[dict], dict[int, dict], int]:
        """Every target's figure, code first (the statement row); then one broadened, deeper search per figure still
        missing, the row again and only then the LLM, kept only on a line naming the measure. Flagged chunks are
        dropped, never shown to the LLM."""
        evidence, guard = drop_flagged(list(state.get("evidence", [])))
        plan = state.get("plan", {})
        targets = plan.get("targets") or []
        calls_log = list(state.get("tool_calls", []))
        seen = {hit.get("chunk_id") for hit in state.get("evidence", [])}
        dropped = list(guard.get("dropped") or [])
        returns = agent is returns_agent
        tool = "cash_flow_extract" if returns else "per_share_extract"

        def target_hits(target: dict) -> list[dict]:
            """The target's own hits first, then the same company-year's hits found for its other measure: one
            cash-flow chunk holds both dividends and repurchases, and a chunk is kept once (``run_searches``)."""
            label = agent.target_label(target)
            own = [h for h in evidence if h.get("target") == label]
            prefix = f"{target['ticker']} FY{target['year']} "
            return own + [h for h in evidence if h.get("target") != label and str(h.get("target") or "").startswith(prefix)]

        def figure(target: dict, hits: list[dict], use_llm: bool, attempt: int) -> dict:
            started, year = time.perf_counter(), target["year"]
            raw = (returns_agent.read_row(hits, target["measure"], year) if returns
                   else income_agent.read_dps(hits, year))
            point = trend.validate_point(raw, hits, year) if raw else {"year": year, "value": None, "value_text": None,
                                                                       "unit": "", "chunk_id": None, "status": "not_found"}
            method = "label"
            if point.get("status") != "ok" and use_llm and hits:
                metric = (f"{target['measure']} (statement of cash flows)" if returns
                          else "cash dividends declared per common share (USD per share)")
                raw = safe_json_loads(invoke_prompt(llm, prompts.SERIES_SYSTEM, prompts.SERIES_HUMAN.format(
                    company=NAMES.get(target["ticker"], target["ticker"]), metric=metric, year=year,
                    context=evidence_to_context(hits, min(config.max_context_chars, trend.SERIES_CONTEXT_CHARS)))),
                    fallback={})
                point, method = trend.validate_point(raw, hits, year), "llm"
                if returns:
                    if point.get("status") == "ok" and not returns_agent.cash_figure_ok(point, hits, target["measure"]):
                        point = {**point, "value": None, "status": "not_a_cash_flow_row"}
                elif point.get("status") == "ok" and not sector_agent.on_metric_line(point, hits, income_agent.PHRASES):
                    point = {**point, "value": None, "status": "not_on_metric_line"}
                if point.get("status") == "ok" and not returns and not income_agent.PER_SHARE.match(
                        str(point.get("value_text") or "").strip()):
                    point = {**point, "value": None, "status": "not_a_per_share_amount"}
            point["method"] = method
            calls_log.append({"tool": tool, "attempt": attempt, "target": agent.target_label(target), "method": method,
                              "status": point["status"], "value": point.get("value_text"),
                              "chunk_id": point.get("chunk_id"), "ms": round((time.perf_counter() - started) * 1000)})
            return point

        found = {i: figure(t, target_hits(t), False, 1) for i, t in enumerate(targets)}
        missing = [i for i, point in found.items() if point.get("status") != "ok"]
        jobs = [search_job(agent.target_label(targets[i]), action_query(agent, targets[i], True), 6, attempt=2,
                           ticker=targets[i]["ticker"], year=targets[i]["year"], sector=None) for i in missing]
        if returns:                  # the retry's other search: the cash-flow statement's own rows
            jobs += [search_job(agent.target_label(targets[i]), sector_agent.rows_query(targets[i]["measure"]), 5,
                                attempt=2, ticker=targets[i]["ticker"], year=targets[i]["year"], sector=None)
                     for i in missing]
        for at, fresh in zip([*missing, *(missing if returns else [])], run_searches(jobs, seen, calls_log)):
            fresh = [*fresh, *statement_hits(fresh, targets[at].get("measure"), agent.target_label(targets[at]), seen)]
            fresh, fresh_guard = drop_flagged(fresh)
            dropped += fresh_guard.get("dropped") or []
            evidence.extend(fresh)
        for i in missing:
            hits = target_hits(targets[i])
            if hits:
                found[i] = figure(targets[i], hits, True, 2)
        if dropped:
            guard = {**guard, "action": "dropped_flagged_chunks", "dropped": sorted(set(dropped))}
        return evidence, guard, calls_log, found, len(missing)

    def action_series_node(state: AgentState) -> AgentState:
        """The verified figures, then the agent's action in code: the shareholder-returns ranking, or the user's
        dividend income (shares x dividend per share)."""
        agent, plan = action_agent(state), state.get("plan", {})
        evidence, guard, calls_log, found, retried = action_figures(state, agent)
        targets = plan.get("targets") or []
        if agent is returns_agent:
            table = returns_agent.rank(plan, {(t["ticker"], t["measure"]): found[i] for i, t in enumerate(targets)})
            calls_log.append({"tool": "returns_calculator", "ranking": table["ranking"], "year": table["year"],
                              "calculations": len(table["calculations"])})
            key, done = "shareholder_returns", bool(table["ranking"])
            empty = ("No ranking: dividends paid and share repurchases could not both be verified for any company "
                     "asked about.")
        else:
            table = income_agent.income(plan, {t["ticker"]: found[i] for i, t in enumerate(targets)})
            calls_log.append({"tool": "dividend_calculator", "holdings": len(targets), "calculated": table["calculated"],
                              "total": table["total"], "calculations": len(table["calculations"])})
            key, done = "dividend_income", bool(table["calculated"])
            empty = "No dividend income calculated: no holding's dividends per share could be verified in its filing."
        check = {"attempt": 2 if retried else 1, "evidence": len(evidence), "retried": retried,
                 "missing": [agent.target_label(targets[i]) for i, p in found.items() if p.get("status") != "ok"],
                 "dropped_chunks": len(guard.get("dropped") or [])}
        update: AgentState = {"evidence": evidence, "evidence_guard": guard, "evidence_check": check,
                              "tool_calls": calls_log, "retrieval_attempts": 2 if retried else 1,
                              key: {k: v for k, v in table.items() if k != "calculations"},
                              "calculations": table["calculations"]}
        if not done:
            update.update(final_status="insufficient_evidence", answer=empty)
        return update

    def code_answer(state: AgentState) -> str | None:
        """The answer a table agent writes in code (sector trend, shareholder returns, dividend income), or None."""
        if (state.get("sector_trend") or {}).get("panel") and not state.get("mutual_fund"):
            return sector_agent.answer_text(state["sector_trend"])
        if (state.get("trend") or {}).get("verified_points"):
            return trend.answer_text(state["trend"])
        if (state.get("mutual_fund") or {}).get("kind") in fund_agent.CODED_KINDS:
            return fund_agent.answer_text(state["mutual_fund"], state.get("query", ""))
        if (state.get("shareholder_returns") or {}).get("ranking"):
            return returns_agent.answer_text(state["shareholder_returns"])
        if (state.get("dividend_income") or {}).get("calculated"):
            return income_agent.answer_text(state["dividend_income"])
        return None

    def fund_lookup_node(state: AgentState) -> AgentState:
        """fund_lookup: the plan's fixed SQL queries; every row is an evidence item with its own ID."""
        plan = state.get("plan", {})
        started = time.perf_counter()
        found = fund_agent.lookup(plan)
        evidence, guard = drop_flagged(found["evidence"])
        ms = round((time.perf_counter() - started) * 1000)
        calls_log = [*state.get("tool_calls", []), *[{**call, "rows": len(evidence), "ms": ms} for call in found["calls"]]]
        update: AgentState = {"evidence": evidence, "evidence_guard": guard, "mutual_fund": found["table"],
                              "tool_calls": calls_log, "retrieval_attempts": 1,
                              "evidence_check": {"attempt": 1, "evidence": len(evidence), "missing": [], "skipped": []}}
        if not evidence:
            company = found["table"].get("company") or "the company"
            update.update(final_status="not_available",
                          answer=f"None of the loaded funds reports an equity holding of {company}.")
        return update

    def fund_company_node(state: AgentState) -> AgentState:
        """The holdings' annual reports: for each top holding, the report filed by the holdings date. Growth questions
        read the measure for that fiscal year and the one before from the same filing (statement row in code, one
        broadened search, the LLM last; a cached figure is reused when its chunk is retrieved again); text questions
        add a filtered search per company. Weighted growth and coverage are computed in code."""
        plan = state.get("plan", {})
        table = {**(state.get("mutual_fund") or {})}
        table["excluded"] = list(table.get("excluded") or [])
        store = fund_agent.default_store()
        evidence, calls_log = list(state.get("evidence", [])), list(state.get("tool_calls", []))
        seen = {hit.get("chunk_id") for hit in evidence}
        targets = fund_agent.company_targets(table, clarifier.directory or {})
        table["companies"] = [{"ticker": t["ticker"], "year": t["year"], "weight": t["weight"], "report": t["report"]}
                              for t in targets]
        calculations, dropped, retried = list(state.get("calculations", [])), [], False
        metric = plan.get("metric")
        if plan.get("kind") == "holdings_growth":
            metric_plan = {"metric": metric}
            figures: dict[str, dict[int, dict]] = {t["ticker"]: {} for t in targets}
            pending = list(targets)
            for attempt in (1, 2):           # each round: every company's search in parallel, then the reads in order
                if not pending:
                    break
                retried = retried or attempt == 2
                fetched = run_searches([sector_job(metric_plan, {"ticker": t["ticker"], "year": t["year"]}, attempt)
                                        for t in pending], seen, calls_log)
                for t, fresh in zip(pending, fetched):
                    label = tools.target_label({"ticker": t["ticker"], "year": t["year"]})
                    fresh = [*fresh, *statement_hits(fresh, metric, label, seen)]
                    fresh, fresh_guard = drop_flagged(fresh)
                    dropped += fresh_guard.get("dropped") or []
                    evidence.extend(fresh)
                for t in pending:
                    ticker, year = t["ticker"], t["year"]
                    points = figures[ticker]
                    hits = [hit for hit in evidence if hit.get("target") == tools.target_label({"ticker": ticker, "year": year})]
                    ids = {str(hit.get("chunk_id")) for hit in hits}
                    for fiscal_year in (year - 1, year):
                        if fiscal_year in points:
                            continue
                        cached = store.cached_figure(ticker, fiscal_year, metric)
                        if cached and str(cached["chunk_id"]) in ids:
                            points[fiscal_year] = cached
                            calls_log.append({"tool": "company_figures", "target": f"{ticker} FY{fiscal_year}",
                                              "cached": True, "chunk_id": cached["chunk_id"]})
                            continue
                        point, figure_log = sector_figure(metric_plan, ticker, fiscal_year, hits, use_llm=attempt == 2)
                        calls_log.append({**figure_log, "attempt": attempt})
                        if point.get("status") == "ok":
                            points[fiscal_year] = point
                            store.save_figure(ticker, fiscal_year, metric, point)
                pending = [t for t in pending if len(figures[t["ticker"]]) < 2]
            calculations += fund_agent.growth(table, figures)
            calls_log.append({"tool": "fund_calculator", "companies": len(targets),
                              "weighted_growth_pct": table.get("weighted_growth_pct"),
                              "coverage_share": table.get("coverage_share")})
        else:
            covered = 0.0
            jobs = [search_job(tools.target_label({"ticker": t["ticker"], "year": t["year"]}),
                               f"{NAMES.get(t['ticker'], t['ticker'])}: {state['query']}", 3, ticker=t["ticker"],
                               year=t["year"], sector=None) for t in targets]
            for t, fresh in zip(targets, run_searches(jobs, seen, calls_log)):
                fresh, fresh_guard = drop_flagged(fresh)
                dropped += fresh_guard.get("dropped") or []
                evidence.extend(fresh)
                covered += t["weight"] if fresh else 0.0
            asked = sum(h["pct_of_net_assets"] or 0 for h in table.get("holdings") or [])
            table.update(covered_weight=round(covered, 4), asked_weight=round(asked, 4),
                         coverage_share=round(covered / asked, 4) if asked else 0.0)
        guard = state.get("evidence_guard") or {"safe": True}
        if dropped:
            guard = {**guard, "action": "dropped_flagged_chunks", "dropped": sorted(set(dropped))}
        return {"evidence": evidence, "mutual_fund": table, "calculations": calculations, "tool_calls": calls_log,
                "evidence_guard": guard, "retrieval_attempts": 2 if retried else 1}

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
        query = (tools.expand_query(state["query"]) if attempt == 1
                 else tools.broaden_query(state["query"], metric, METRICS.get(metric, [])))
        visual = "visual_search" in plan.get("tools", [])
        evidence = list(state.get("evidence", [])) if attempt > 1 else []
        seen = {hit.get("chunk_id") for hit in evidence}
        jobs = [search_job(tools.target_label(t), query, per_target,
                           tool="text_search+visual_search" if visual else "text_search", attempt=attempt,
                           ticker=t.get("ticker"), year=t.get("year"), visual=visual,
                           source_type=plan.get("source_type") or config.source_type) for t in targets]
        statement = statement_metric(state["query"]) if attempt == 1 else None
        margin = margin_metric(state["query"]) if attempt == 1 else None
        statement = statement or margin
        if statement:
            # One more search per company-year for its income statement, so the reported row can be read in code
            # (statement_figures): the question's own wording found JPM's pro forma note, not its statement.
            statement_plan = {"metric": statement}

            def statement_query(t):
                if statement in returns_agent.CASH_FLOW_LABELS:      # dividends paid, repurchases: the cash flows
                    return returns_agent.search_query(t["ticker"], int(t["year"]), statement)
                return sector_agent.company_query(statement_plan, t["ticker"], int(t["year"]), METRICS.get(statement, []))
            jobs += [search_job(tools.target_label(t), statement_query(t), 3,
                                tool="statement_search", ticker=t["ticker"], year=t["year"])
                     for t in targets if t.get("ticker") and t.get("year")]
        for fresh in run_searches(jobs, seen, calls_log):
            evidence.extend(fresh)
        calculations = list(state.get("calculations", []))
        if statement:
            evidence = add_statement_rows(state["query"], statement, targets, evidence, seen, calls_log)
            if margin:
                evidence = add_statement_rows(state["query"], "revenue", targets, evidence, seen, calls_log)
                calculations += margins({**state, "plan": plan, "evidence": evidence})
            else:
                calculations += statement_changes({**state, "plan": plan, "evidence": evidence})
        elif financial_checks.CHART_WORDS.search(state["query"]):
            evidence = chart_tables(evidence, seen)
        elif subject or term:
            pages = subject_pages(evidence, seen, subject or term)
            # The table or paragraph asked about comes first, so the context limit can't cut it.
            evidence = [*pages, *evidence]
        return {"plan": plan, "evidence": evidence, "retrieval_attempts": attempt, "tool_calls": calls_log,
                "calculations": calculations}

    def statement_changes(state: AgentState) -> list[dict]:
        """One company's figure read in code for two or more years: the change and the percent change, computed here so
        the answer can state "8.1%" without working it out ("up 8%" was withheld as an unsupported inference)."""
        records = []
        figures = statement_figures(state)
        for ticker in {f["ticker"] for f in figures}:
            rows = sorted((f for f in figures if f["ticker"] == ticker), key=lambda f: f["year"])
            before_text = (rows[0].get("columns") or {}).get(rows[0]["year"] - 1)
            if len(rows) == 1 and before_text and PRIOR_YEAR.search(state["query"]):
                # "compared with the prior year": the same row prints it.
                rows = [{**rows[0], "year": rows[0]["year"] - 1, "value": before_text}, rows[0]]
            for a, b in zip(rows, rows[1:]):
                before, now = trend.parse_figure(a["value"]), trend.parse_figure(b["value"])
                if before and now is not None and trend.scale_of(a["unit"]) == trend.scale_of(b["unit"]):
                    span = {"ticker": ticker, "from": a["year"], "to": b["year"], "grounded": True}
                    records += [{"expression": f"{b['value']} - {a['value']}", "value": round(now - before, 4), **span,
                                 "kind": "change"},
                                {"expression": f"({b['value']} - {a['value']}) / {a['value']} * 100", **span,
                                 "value": round((now - before) / abs(before) * 100, 4), "kind": "pct_change"}]
        return records

    def margin_metric(query: str) -> str | None:
        """The numerator of the margin a figure question asks for (operating margin -> operating income), else None."""
        metric = clarifier.metric_in(query)
        if metric not in sector_agent.MARGINS or subject or financial_checks.CHART_WORDS.search(query):
            return None
        return sector_agent.MARGINS[metric] if not EXPLAIN.search(query) else None

    def margins(state: AgentState) -> list[dict]:
        """The margin, divided in code from two rows of the same company's statement in the same unit."""
        numerator, records = margin_metric(state["query"]), []
        for target in (state.get("plan") or {}).get("targets") or []:
            if not (target.get("ticker") and target.get("year")):
                continue
            hits = [h for h in state.get("evidence", []) if h.get("target") == tools.target_label(target)]
            top, bottom = (statement_row(hits, m, int(target["year"])) for m in (numerator, "revenue"))
            a, b = (trend.parse_figure(row["value"]) if row else None for row in (top, bottom))
            if a is not None and b and trend.scale_of(top["unit"]) == trend.scale_of(bottom["unit"]):
                records.append({"expression": f"{top['value']} / {bottom['value']} * 100", "value": round(a / b * 100, 4),
                                "grounded": True, "kind": "margin", "ticker": target["ticker"], "year": int(target["year"]),
                                "metric": clarifier.metric_in(state["query"]), "parts": [
                                    {**row, "metric": m} for row, m in ((top, numerator), (bottom, "revenue"))]})
        return records

    def add_statement_rows(query: str, metric: str, targets: list[dict], evidence: list[dict], seen: set,
                           calls_log: list[dict]) -> list[dict]:
        """For each company-year: the chunk of the statement page that states the measure, found through the page's
        title; one broadened search when no row was read; and no parent-company-only chunk, whose rows have the same
        names ("Total assets 559,733" is Bank of America's parent company; consolidated: 3,261,519)."""
        for retry in (False, True):
            unread = []
            for t in [t for t in targets if t.get("ticker") and t.get("year")]:
                label = tools.target_label(t)
                added = statement_hits(evidence, metric, label, seen)
                own = [h for h in evidence if h.get("target") == label]
                row = statement_row([*added, *own], metric, int(t["year"]), query)
                evidence += [h for h in added if row and h.get("chunk_id") == row.get("chunk_id")]
                seen -= {h.get("chunk_id") for h in added if not row or h.get("chunk_id") != row.get("chunk_id")}
                if not row:
                    unread.append(t)
            if retry or not unread:
                break
            jobs = [search_job(tools.target_label(t), (
                        returns_agent.search_query(t["ticker"], int(t["year"]), metric, True)
                        if metric in returns_agent.CASH_FLOW_LABELS else sector_agent.company_query(
                            {"metric": metric}, t["ticker"], int(t["year"]), METRICS.get(metric, []), broadened=True)),
                        6, tool="statement_search", attempt=2, ticker=t["ticker"], year=t["year"]) for t in unread]
            if sector_agent.rows_query(metric):
                jobs += [rows_job({"metric": metric}, t) for t in unread]
            for fresh in run_searches(jobs, seen, calls_log):
                evidence.extend(drop_flagged(fresh)[0])        # an extra search never brings a flagged chunk in
        if not re.search(r"\bparent\b", query, re.I):
            kept = [h for h in evidence if not sector_agent.parent_only(str(h.get("text") or ""))]
            evidence = kept or evidence
        return evidence

    def chart_tables(evidence: list[dict], seen: set, limit: int = 2) -> list[dict]:
        """A question about a chart: the tables read from the pages of the first chart descriptions, right after them.
        A page of five charts is five tables; search returned the description and one of the other tables."""
        pages: dict[tuple, Any] = {}
        for hit in evidence:
            doc_id = (hit.get("metadata") or {}).get("doc_id")
            if hit.get("content_type") == "page_description" and doc_id and hit.get("page") is not None:
                pages.setdefault((str(doc_id), int(hit["page"])), hit.get("target"))
        out = list(evidence)
        for key, label in list(pages.items())[:limit]:
            tables = [{**h, "target": label} for h in statement_page_hits([key], []) if h.get("content_type") == "table"]
            ids = {h["chunk_id"] for h in tables}
            # The description and its page's readings first: a paragraph elsewhere (Chevron's U.S. Upstream text on p37)
            # ranked above the chart the question asks about.
            same = [h for h in out if ((h.get("metadata") or {}).get("doc_id"), h.get("page")) == key
                    and h.get("chunk_id") not in ids]
            rest = [h for h in out if h.get("chunk_id") not in ids and h not in same]
            placed = [h for h in rest if h.get("chart_page")]
            out = [*placed, *({**h, "chart_page": True} for h in [*same[:1], *tables[:8], *same[1:]]),
                   *(h for h in rest if not h.get("chart_page"))]
            seen |= ids
        return out

    def subject_pages(evidence: list[dict], seen: set, phrase: str) -> list[dict]:
        """Per target, the rest of the page headed by what the question is about (else the first page naming it):
        Citi's U.S. Personal Banking table is several chunks, and only the first carries its title. Of those chunks,
        the ones that print the measure (or, for a defined term, the term)."""
        metric = clarifier.metric_in(config.query)
        wanted = METRICS.get(metric, []) if subject else [phrase]
        best: dict[str, tuple] = {}
        for hit in evidence:
            text, meta = str(hit.get("text") or ""), hit.get("metadata") or {}
            if not meta.get("doc_id") or hit.get("page") is None or not financial_checks.names(text, phrase):
                continue
            headed = any(financial_checks.names(line, phrase) and len(line.split()) <= 18 for line in text.splitlines()[:2])
            label = str(hit.get("target"))
            if label not in best or (headed and not best[label][0]):
                best[label] = (headed, str(meta["doc_id"]), int(hit["page"]))
        out = []
        for label, (_, doc_id, page) in best.items():
            for hit in statement_page_hits([(doc_id, page)], evidence):
                if hit.get("chunk_id") not in seen and any(
                        re.search(rf"(?<![a-z]){re.escape(p)}(?![a-z])", str(hit.get("text") or "").lower()) for p in
                        (w.lower() for w in wanted)):
                    seen.add(hit.get("chunk_id"))
                    out.append({**hit, "target": label})
        return out[:6]

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
        plan = state.get("plan", {})
        fund_table = state.get("mutual_fund") or {}
        coded = code_answer(state)
        if coded is not None:
            # Sector trend, shareholder returns, dividend income: written in code from the verified table (no LLM
            # call, nothing for a judge to check).
            return {"answer": coded}
        if fund_table:
            # Fund: the rows (and, for linked questions, only the company chunks that state a verified figure).
            used = evidence
            if fund_table.get("kind") == "holdings_growth":
                used_ids = {p.get("chunk_id") for c in fund_table.get("companies", []) for p in c.get("points", [])
                            if p.get("status") == "ok"}
                used = [h for h in evidence if h.get("source_type") == "nport" or h.get("chunk_id") in used_ids]
            system = fund_prompts.FUND_ANSWER_SYSTEM
            human = fund_prompts.FUND_ANSWER_HUMAN.format(
                query=state["query"], fund=fund_table.get("fund") or fund_table.get("company") or "loaded funds",
                period_end=fund_table.get("period_end") or "each fund's latest loaded filing",
                accession=fund_table.get("accession") or "per row", kind=fund_table.get("kind"),
                table=fund_agent.table_text(fund_table), context=evidence_to_context(used, config.max_context_chars))
        else:
            system = prompts.GRAPH_ANSWER_SYSTEM
            figures = statement_figures(state)
            # The chunk that states a figure read in code comes first, so the context limit can't cut it.
            stated = {f["chunk_id"] for f in figures}
            evidence = sorted(across_targets(evidence), key=lambda h: h.get("chunk_id") not in stated)
            human = prompts.GRAPH_ANSWER_HUMAN.format(
                query=state["query"], context=evidence_to_context(evidence, config.max_context_chars),
                plan=json.dumps({key: plan.get(key) for key in ("intent", "tools", "targets")}, ensure_ascii=False))
            if "calculator" in plan.get("tools", []):
                human += "\n\n" + prompts.CALCULATOR_INSTRUCTION
            if subject:
                human += "\n\n" + prompts.SUBJECT_INSTRUCTION.format(subject=subject)
            for c in (c for c in state.get("calculations", []) if c.get("kind") == "margin"):
                name = NAMES.get(c["ticker"], c["ticker"])
                human += "\n\n" + prompts.MARGIN_INSTRUCTION.format(
                    name=name, year=c["year"], metric=c["metric"], value=f"{c['value']:.2f}", expression=c["expression"],
                    rows="; ".join(f"{p['metric']} {p['value']} {p['unit']} [{p['chunk_id']}]" for p in c["parts"]))
            if figures:
                human += "\n\n" + prompts.STATEMENT_FIGURES_INSTRUCTION.format(figures="\n".join(
                    f"- {NAMES.get(f['ticker'], f['ticker'])} ({f['ticker']}) FY{f['year']} {f['metric']}: "
                    f"{f['value']} {f['unit']} [{f['chunk_id']}]".replace("  ", " ") for f in figures))
                changes = [c for c in state.get("calculations", []) if c.get("kind") == "pct_change" and c.get("ticker")]
                if changes:
                    human += "\n" + prompts.STATEMENT_CHANGES_INSTRUCTION.format(changes="\n".join(
                        f"- {NAMES.get(c['ticker'], c['ticker'])} FY{c['from']} to FY{c['to']}: {c['value']:+.2f}%"
                        for c in changes))
        feedback = critic_feedback(state) if state.get("repair_count", 0) else ""
        if feedback:
            human += "\n\n" + prompts.REPAIR_INSTRUCTION.format(
                feedback=feedback, allowed=", ".join(str(hit.get("chunk_id")) for hit in evidence))
        answer = invoke_prompt(llm, system, human)
        # A figure the code read and the model copied without its citation: the citation is known, so it is added.
        uncited = [f["chunk_id"] for f in (statement_figures(state) if not fund_table else [])
                   if tools.states(answer, f["value"]) and str(f["chunk_id"]) not in answer]
        if fund_table:
            uncited = fund_agent.uncited_rows(answer, evidence)
        if uncited:
            answer += "\n\nSources: " + " ".join(f"[{chunk_id}]" for chunk_id in dict.fromkeys(uncited))
        return {"answer": answer}

    def statement_metric(query: str) -> str | None:
        """The statement measure a figure or comparison question asks for (net income, revenue, ...), else None."""
        metric = clarifier.metric_in(query)
        if metric not in sector_agent.STATEMENT_LABELS and metric not in returns_agent.CASH_FLOW_LABELS:
            return None
        if financial_checks.CHART_WORDS.search(query):        # what a chart shows is not the statement's row
            return None
        if subject:                       # a segment's, product's or region's figure is not the statement's
            return None
        companies, years = clarifier.companies_in(query), clarifier.years_in(query)
        intent = clarifier.intent_in(query, companies, years)
        if intent == "trend" and EXPLAIN.search(query):          # "why did net income change": an explanation
            return None
        return metric if intent in ("figure", "comparison", "trend") else None

    def statement_figures(state: AgentState) -> list[dict]:
        """Q&A figure and comparison questions: the asked measure's statement row, read in code for each company-year
        target from that target's evidence (``label_match`` skips pro forma tables). On hosted Qwen3-8B the answer to
        "JPMorgan or Bank of America, higher net income in FY2025?" quoted JPM's pro forma 2023 note (48,665) although
        the consolidated statement (57,048) was retrieved too."""
        metric = statement_metric(state.get("query", ""))
        if not metric:
            return []
        figures = []
        for target in (state.get("plan") or {}).get("targets") or []:
            if not (target.get("ticker") and target.get("year")):
                continue
            hits = [h for h in state.get("evidence", []) if h.get("target") == tools.target_label(target)]
            row = statement_row(hits, metric, int(target["year"]), state.get("query", ""))
            if row:
                figures.append({"ticker": target["ticker"], "year": int(target["year"]), "metric": metric, **row})
        return figures

    def ranked_figures(state: AgentState) -> list[str]:
        """"Walmart 680,985 is higher than Costco 275,235": the comparison of figures read in code, for the judge."""
        by_year: dict[int, list[dict]] = {}
        for figure in statement_figures(state):
            by_year.setdefault(figure["year"], []).append(figure)
        lines = []
        for year, rows in by_year.items():
            if len(rows) < 2 or len({r["unit"] for r in rows}) != 1:
                continue
            ordered = sorted(rows, key=lambda r: -(trend.parse_figure(r["value"]) or 0))
            lines.append(f"FY{year} {rows[0]['metric']}, highest first: " + " > ".join(
                f"{NAMES.get(r['ticker'], r['ticker'])} {r['value']}" for r in ordered))
        return lines

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
                cited = set((state.get("citation_check") or {}).get("cited_chunk_ids") or [])
                exact = tools.exact_changes(str(state.get("answer") or ""), tools.evidence_numbers(
                    [h for h in state.get("evidence", []) if str(h.get("chunk_id")) in cited]), numbers)
                if exact:
                    notes.append("the exact values, computed in code: " + "; ".join(exact))
            if not claims and not numbers:
                notes.append("its support could not be verified")
        return "; ".join(notes)

    def support_check(state: AgentState, answer: str, cited: list[str]) -> dict[str, Any]:
        """Critic, part 2: does the cited evidence state what the answer says? A code check on every figure,
        then an LLM judge on every claim. Both must pass for 'supported'."""
        evidence = state.get("evidence", [])
        cited_hits = [hit for hit in evidence if str(hit.get("chunk_id")) in set(cited)]
        numeric = tools.numeric_grounding(answer, cited_hits, state.get("calculations", []))
        derived = tools.derived_figures(answer, tools.evidence_numbers(cited_hits))
        calculations = "\n".join([*(f"{c['expression']} = {c['value']:,}" for c in state.get("calculations", [])
                                    if c.get("grounded") and c.get("value") is not None),
                                  *(text for _, text in derived), *ranked_figures(state)]) or "none"
        raw = invoke_prompt(llm, prompts.SUPPORT_SYSTEM, prompts.SUPPORT_HUMAN.format(
            query=state["query"], answer=answer, context=evidence_to_context(cited_hits, config.max_context_chars),
            calculations=calculations))
        judged = tools.parse_support(raw)
        # "a consistent upward trend" is no figure the evidence prints; the answer's own figures show it or they don't.
        left = [c for c in judged.get("unsupported_claims") or []
                if not tools.direction_follows(c, answer, tools.evidence_numbers(cited_hits))
                and not (state.get("mutual_fund") and fund_agent.CAVEAT.search(c))]
        if judged.get("unsupported_claims") and not left:
            judged = {**judged, "verdict": "supported", "unsupported_claims": [],
                      "direction_claims": judged["unsupported_claims"]}
        verdict = judged["verdict"]
        if not numeric["grounded"] and verdict in ("supported", "partially_supported"):
            verdict = "unsupported" if not judged.get("claims") else "partially_supported"
        return {**judged, "verdict": verdict, "numeric": numeric, "method": "llm_judge+numeric",
                "attempt": state.get("repair_count", 0) + 1}

    def calculator_node(state: AgentState) -> AgentState:
        answer, records = tools.run_calculations(str(state.get("answer") or ""), state.get("evidence", []))
        calls_log = [*state.get("tool_calls", []), {"tool": "calculator", "expressions": len(records),
                                                    "grounded": all(r["grounded"] for r in records)}]
        kept = [c for c in state.get("calculations", []) if c not in records]
        return {"answer": answer, "calculations": [*kept, *records], "tool_calls": calls_log}

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
        coded = code_answer(state)
        if check["valid"] and answer != coded:
            cited_ids = set(check["cited_chunk_ids"])
            answer, corrected = tools.correct_changes(answer, tools.evidence_numbers(
                [h for h in state.get("evidence", []) if str(h.get("chunk_id")) in cited_ids]))
            if corrected:
                state = {**state, "calculations": [*state.get("calculations", []), *corrected]}
        if check["valid"] and support_enabled and coded is not None and answer == coded:
            # Written in code from verified figures: the figures are checked in code; no LLM judge, no repair.
            cited_hits = [h for h in state.get("evidence", []) if str(h.get("chunk_id")) in set(check["cited_chunk_ids"])]
            numeric = tools.numeric_grounding(answer, cited_hits, state.get("calculations", []))
            support = {"verdict": "supported" if numeric["grounded"] else "unsupported", "claims": [],
                       "numeric": numeric, "method": "code_table", "attempt": state.get("repair_count", 0) + 1}
        elif check["valid"] and support_enabled:
            support = support_check(state, answer, check["cited_chunk_ids"])
            # The reported statement figure, read in code, must be the one the answer gives (not a pro forma note,
            # a segment or another year that also sits in the evidence).
            ignored = [f for f in statement_figures(state) if not tools.states(answer, f["value"])]
            if ignored:
                notes = [f"the reported {f['metric']} of {NAMES.get(f['ticker'], f['ticker'])} for FY{f['year']} is "
                         f"{f['value']} [{f['chunk_id']}] (consolidated statement row); use that figure" for f in ignored]
                support = {**support, "verdict": "partially_supported" if support.get("claims") else "unsupported",
                           "unsupported_claims": [*(support.get("unsupported_claims") or []), *notes],
                           "statement_check": {"ignored": [f"{f['ticker']} FY{f['year']}" for f in ignored]}}
            if not state.get("mutual_fund"):
                cited_hits = [h for h in state.get("evidence", []) if str(h.get("chunk_id")) in set(check["cited_chunk_ids"])]
                notes = financial_checks.problems(state["query"], answer, cited_hits, state.get("evidence", []), subject,
                                                  sector_agent.parent_only, page_text_before)
                if statement_metric(state["query"]):         # a segment's or parent's row of a statement measure
                    notes += [f"{raw} is from a segment or parent-company table; use the consolidated statement"
                              for raw, _, _ in financial_checks.answer_figures(answer)
                              if (rows := [h for h in cited_hits if raw in str(h.get("text") or "")]) and all(
                                  sector_agent.other_companys_row({"chunk_id": h.get("chunk_id"), "value_text": raw}, [h])
                                  for h in rows)][:3]
                asked = statement_metric(state["query"])
                if asked and not statement_figures(state):   # no row read in code: the model's figure must name the measure
                    notes += [f"{raw} is not on a line about {asked}" for raw in financial_checks.off_measure(
                        answer, cited_hits, [*METRICS.get(asked, []), *sector_agent.STATEMENT_LABELS.get(asked, ()),
                                             *returns_agent.PHRASES.get(asked, [])])]
                notes += financial_checks.ranked_without_figure(
                    answer, {t["ticker"]: NAMES.get(t["ticker"], t["ticker"]) for t in
                             (state.get("plan") or {}).get("targets") or [] if t.get("ticker")}, cited_hits)
                if notes:                 # a figure of another segment, region, product, axis or statement
                    support = {**support, "verdict": "partially_supported" if support.get("claims") else "unsupported",
                               "unsupported_claims": [*(support.get("unsupported_claims") or []), *notes],
                               "measure_check": {"problems": notes}}
        accepted = check["valid"] and (support is None or support["verdict"] == "supported")
        return {
            "answer": answer,
            "calculations": state.get("calculations", []),
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
        if (state.get("route") or {}).get("route") in ("fund", *ACTION_ROUTES):
            # "I hold 100 shares" is not a request for advice: the investment-language trigger stays only for advice
            # wording (buy, sell, should I, recommend).
            kept = fund_agent.review_triggers(state.get("query", ""), review.get("triggers", []))
            if kept != review.get("triggers", []):
                review = {**review, "triggers": kept}
                if not kept:
                    review.update(required=False, status="not_required")
        extra = []
        if any(not c.get("grounded") for c in state.get("calculations", [])):
            extra.append("unverified_calculation")
        if (state.get("support_check") or {}).get("verdict") not in (None, "supported"):
            extra.append("unsupported_claims")
        if ((state.get("support_check") or {}).get("numeric") or {}).get("unverified"):
            extra.append("unverified_vlm_figure")          # a figure only a page reader wrote; text layer and OCR don't
        table = state.get("sector_trend") or {}
        if table and table.get("coverage_share", 1) < 0.5:
            extra.append("low_sector_coverage")            # the panel holds under half the companies planned
        if table.get("unusual_changes"):
            extra.append("unusual_sector_change")          # a company figure moved more than RAG_SECTOR_OUTLIER_PCT
        returns_table = state.get("shareholder_returns") or {}
        if returns_table and returns_table.get("coverage_share", 1) < 0.5:
            extra.append("low_returns_coverage")           # under half the companies read could be ranked
        if fund_agent.low_coverage(state.get("mutual_fund") or {}):
            extra.append("low_fund_coverage")              # the companies used carry under half the fund weight asked
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

    def after_route(state: AgentState) -> Literal["stop", "trend", "sector", "fund", "action", "qa"]:
        if state.get("final_status") in STOPPED:
            return "stop"
        route = state.get("route", {}).get("route")
        return {"trend": "trend", "sector_trend": "sector", "fund": "fund", "shareholder_returns": "action",
                "dividend_income": "action"}.get(route, "qa")

    def after_fund(state: AgentState) -> Literal["stop", "company", "answer"]:
        if state.get("final_status") in STOPPED:
            return "stop"
        return "company" if state.get("plan", {}).get("needs_companies") else "answer"

    def after_sector(state: AgentState) -> Literal["stop", "answer"]:
        return "stop" if state.get("final_status") in STOPPED else "answer"

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
        # A rewrite can't help when the answer was written in code, or when the judge's own reply was cut off (the
        # banking run spent ~30 s on a repair that hit the same 1,024-token limit).
        if support.get("method") == "code_table" or support.get("judge_cut_off"):
            return "finish"
        return "repair" if used(state) + tail(state) <= budget else "finish"      # answer (+calc), critic, governance

    graph = StateGraph(AgentState)
    first = "plan_step" if flow == "agentic_graph" else "route_step"
    nodes = {"security_step": security_node, "plain_text_step": plain_text_node,
             first: planner_node if flow == "agentic_graph" else route_node,
             "retrieve_step": executor_node, "evidence_guard_step": evidence_guard_node,
             "answer_step": answer_node, "calculator_step": calculator_node, "critic_step": critic_node,
             "governance_step": governance_node}
    if flow != "agentic_graph":
        nodes.update(trend_fetch_step=trend_fetch_node, trend_series_step=trend_series_node,
                     sector_fetch_step=sector_fetch_node, sector_series_step=sector_series_node,
                     fund_lookup_step=fund_lookup_node, fund_company_step=fund_company_node,
                     action_fetch_step=action_fetch_node, action_series_step=action_series_node)
    for name, fn in nodes.items():
        graph.add_node(name, step(name, fn))
    graph.set_entry_point("security_step")
    graph.add_conditional_edges("security_step", should_continue_after_security, {"stop": "critic_step", "continue": "plain_text_step"})
    graph.add_conditional_edges("plain_text_step", should_continue_after_security, {"stop": "critic_step", "continue": first})
    if flow == "agentic_graph":
        graph.add_edge("plan_step", "retrieve_step")
    else:
        graph.add_conditional_edges("route_step", after_route,
                                    {"stop": "critic_step", "trend": "trend_fetch_step", "sector": "sector_fetch_step",
                                     "fund": "fund_lookup_step", "action": "action_fetch_step",
                                     "qa": "retrieve_step"})
        graph.add_edge("action_fetch_step", "action_series_step")
        graph.add_conditional_edges("action_series_step", after_sector, {"stop": "critic_step", "answer": "answer_step"})
        graph.add_conditional_edges("fund_lookup_step", after_fund,
                                    {"stop": "critic_step", "company": "fund_company_step", "answer": "answer_step"})
        graph.add_edge("fund_company_step", "answer_step")
        graph.add_edge("sector_fetch_step", "sector_series_step")
        graph.add_conditional_edges("sector_series_step", after_sector, {"stop": "critic_step", "answer": "answer_step"})
        graph.add_edge("trend_fetch_step", "trend_series_step")
        graph.add_conditional_edges("trend_series_step", after_evidence,
                                    {"stop": "critic_step", "retry": "trend_fetch_step", "answer": "answer_step"})
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
        "technique": flow,
        "final_status": final_state.get("final_status"),
        "security": final_state.get("security"),
        "plain_text": final_state.get("plain_text"),
        "route": final_state.get("route"),
        "trend": final_state.get("trend"),
        "sector_trend": final_state.get("sector_trend"),
        "shareholder_returns": final_state.get("shareholder_returns"),
        "dividend_income": final_state.get("dividend_income"),
        "mutual_fund": final_state.get("mutual_fund"),
        "handoff": final_state.get("handoff"),
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
            "search_fan_out": fan_out_log,
        },
    }


def _run_technique(technique: str, llm_provider: str, config: AgentConfig) -> dict[str, Any]:
    def llm_factory(temperature: float = 0.0):
        return get_llm(llm_provider, temperature=temperature)

    if technique in AGENT_FLOWS:
        return financial_graph(llm_factory(0.0), config, technique)

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


def request_deadline(technique: str, config: AgentConfig) -> float:
    """Seconds a request may take: RAG_REQUEST_TIMEOUT_SECONDS, or RAG_SECTOR_TIMEOUT_SECONDS (default 300) for a
    sector trend, which reads one figure per company x year."""
    seconds = float(os.getenv("RAG_REQUEST_TIMEOUT_SECONDS", "60"))
    sector = technique in ("sector_trend_graph", "fund_graph", "shareholder_returns_graph", "dividend_income_graph")
    if technique == "router":
        from scripts.agents import financial_router
        from scripts.agents.financial_clarifier import default_clarifier
        route = financial_router.rule_route(config.query, default_clarifier(), config.ticker, config.sector)[0]
        sector = route in ("sector_trend", "fund", *ACTION_ROUTES)   # one search per company (or company x measure)
    return max(seconds, float(os.getenv("RAG_SECTOR_TIMEOUT_SECONDS", "300"))) if sector else seconds


# Outcomes decided before any evidence: no coverage check, no evidence fallback.
NOT_SCOPED = {"refused", "tool_error", "report_handoff"}


def run_technique(technique: str, llm_provider: str, config: AgentConfig) -> dict[str, Any]:
    """Bound legacy comparison workflows and label unverified generation honestly."""
    from scripts.common.resilience import Calls, CallFailure
    calls = Calls(request_deadline(technique, config))
    try:
        result = calls.run("financial_workflow", lambda: _run_technique(technique, llm_provider, config))
    except CallFailure as exc:
        logger.warning("financial workflow failed: technique=%s reason=%s", technique, exc)
        result = {"technique": technique, "final_status": "tool_error", "error": str(exc),
                  "answer": "No answer found because a required service failed or timed out.", "evidence": [],
                  "citation_check": {"valid": False, "cited_chunk_ids": []}}
    scope = {"expected": [], "coverage": None}
    verified = False
    route_name = (result.get("route") or {}).get("route")
    # A fund answer cites N-PORT rows, not one company's filings: the company scope rules don't apply to it.
    # The shareholder-returns and dividend-income agents plan their own coverage and cite several companies.
    if result.get("final_status") not in NOT_SCOPED and route_name not in ("fund", *ACTION_ROUTES):
        scope = question_scope(config.query, config.ticker)
    if scope["coverage"]:
        # The filings can't answer this, so nothing retrieved or generated may stand in for it.
        for key in ("parsed", "samples", "subquestions"):
            result.pop(key, None)
        result.update(answer=scope["coverage"]["message"], final_status="not_available", coverage=scope["coverage"],
                      evidence=[], citation_check={"valid": False, "cited_chunk_ids": []})
    elif result.get("final_status") == "not_available" and route_name in ("sector_trend", "fund", *ACTION_ROUTES):
        pass                     # the sector or fund plan already said why; keep its reason
    elif result.get("final_status") == "insufficient_evidence" and route_name in ("sector_trend", "fund", *ACTION_ROUTES):
        pass                     # the agent says which figures it could not verify; no excerpts stand in for a table
    elif result.get("final_status") not in NOT_SCOPED:
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
    verified = verified and result.get("final_status") not in {"not_grounded", "not_available", "refused", "tool_error",
                                                               "report_handoff"}
    method = "mock" if llm_provider == "mock" else "llm_support_check" if verified else "source_excerpts"
    result.update(trace_id=calls.trace_id, trace=calls.events,
                  grounding={"method": method, "semantic_answer_verified": verified})
    return result


def answer_verified(result: dict[str, Any], evidence: list[dict[str, Any]]) -> bool:
    """The agent graph's critic found every claim and figure supported by evidence it still cites."""
    support, check = result.get("support_check") or {}, result.get("citation_check") or {}
    allowed = {str(hit.get("chunk_id")) for hit in evidence}
    return (result.get("technique") in AGENT_FLOWS and support.get("verdict") == "supported"
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
        "route": result.get("route"),
        "trend": result.get("trend"),
        "sector_trend": result.get("sector_trend"),
        "shareholder_returns": result.get("shareholder_returns"),
        "dividend_income": result.get("dividend_income"),
        "mutual_fund": result.get("mutual_fund"),
        "handoff": result.get("handoff"),
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
            "trend_graph",
            "sector_trend_graph",
            "fund_graph",
            "shareholder_returns_graph",
            "dividend_income_graph",
            "router",
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
        "trend_graph",
        "sector_trend_graph",
        "router",
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
