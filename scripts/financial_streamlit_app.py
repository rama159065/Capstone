"""Streamlit chat UI for the financial-report agent.

Run:
    streamlit run scripts/financial_streamlit_app.py

The UI reuses scripts/financial_prompting_agent.py and displays the full
governed agent trace: answer, citations, evidence, security guardrails,
plain-text classifier, evidence guard, and human-review routing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.financial_prompting_agent import AgentConfig, run_technique, summarize_result  # noqa: E402


TECHNIQUES = [
    "agentic_graph",
    "baseline_rag",
    "structured_few_shot",
    "decomposition",
    "self_consistency",
]
SOURCE_TYPES = ["", "annual_report_pdf", "table_summary", "chart_summary"]
STATUSES = {
    "answered": "Answered",
    "auto_approved": "Auto approved",
    "pending_human_review": "Pending human review",
    "refused": "Refused",
}


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_result", None)


def build_config(query: str, settings: dict[str, Any]) -> AgentConfig:
    return AgentConfig(
        query=query,
        ticker=settings["ticker"] or None,
        year=settings["year"] or None,
        source_type=settings["source_type"] or None,
        top_k=settings["top_k"],
        dense_top_k=settings["dense_top_k"],
        bm25_top_k=settings["bm25_top_k"],
        retrieval_mode=settings["retrieval_mode"],
        embedding_provider=settings["embedding_provider"],
        hash_dimensions=settings["hash_dimensions"],
        max_context_chars=settings["max_context_chars"],
        self_consistency_samples=settings["self_consistency_samples"],
        enable_guardrails=settings["enable_guardrails"],
        enable_human_review=settings["enable_human_review"],
    )


def sidebar_settings() -> dict[str, Any]:
    st.sidebar.title("Financial Copilot")
    st.sidebar.caption("RAG + agents + guardrails + HITL")

    technique = st.sidebar.selectbox("Prompting technique", TECHNIQUES, index=0)
    llm_provider = st.sidebar.selectbox("LLM provider", ["mock", "hosted"], index=0)
    embedding_provider = st.sidebar.selectbox("Embedding provider", ["hashing", "hosted"], index=0)
    retrieval_mode = st.sidebar.selectbox("Retrieval mode", ["hybrid", "dense"], index=0)

    st.sidebar.divider()
    ticker = st.sidebar.text_input("Ticker filter", value="AAPL").strip().upper()
    year_value = st.sidebar.text_input("Year filter", value="").strip()
    source_type = st.sidebar.selectbox("Source type filter", SOURCE_TYPES, index=0)

    st.sidebar.divider()
    top_k = st.sidebar.slider("Final top k", min_value=1, max_value=12, value=5)
    dense_top_k = st.sidebar.slider("Dense candidates", min_value=5, max_value=80, value=30, step=5)
    bm25_top_k = st.sidebar.slider("BM25 candidates", min_value=5, max_value=80, value=30, step=5)
    max_context_chars = st.sidebar.slider(
        "Max context chars",
        min_value=2000,
        max_value=24000,
        value=12000,
        step=1000,
    )
    hash_dimensions = st.sidebar.selectbox("Hash embedding dimensions", [384, 768, 1024, 1536], index=0)
    self_consistency_samples = st.sidebar.slider(
        "Self-consistency samples",
        min_value=1,
        max_value=5,
        value=3,
    )

    st.sidebar.divider()
    enable_guardrails = st.sidebar.toggle("Security guardrails", value=True)
    enable_human_review = st.sidebar.toggle("Human review routing", value=True)

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["last_result"] = None
        st.rerun()

    try:
        year = int(year_value) if year_value else None
    except ValueError:
        st.sidebar.warning("Year filter must be a number. Ignoring it.")
        year = None

    return {
        "technique": technique,
        "llm_provider": llm_provider,
        "embedding_provider": embedding_provider,
        "retrieval_mode": retrieval_mode,
        "ticker": ticker,
        "year": year,
        "source_type": source_type,
        "top_k": top_k,
        "dense_top_k": dense_top_k,
        "bm25_top_k": bm25_top_k,
        "max_context_chars": max_context_chars,
        "hash_dimensions": hash_dimensions,
        "self_consistency_samples": self_consistency_samples,
        "enable_guardrails": enable_guardrails,
        "enable_human_review": enable_human_review,
    }


def render_status(result: dict[str, Any]) -> None:
    status = result.get("final_status") or "unknown"
    label = STATUSES.get(str(status), str(status))
    if status == "refused":
        st.error(label)
    elif status == "pending_human_review":
        st.warning(label)
    else:
        st.success(label)


def render_citations(result: dict[str, Any]) -> None:
    citation_check = result.get("citation_check") or {}
    cited = citation_check.get("cited_chunk_ids") or []
    missing = citation_check.get("missing_chunk_ids") or []
    valid = citation_check.get("valid")

    cols = st.columns(3)
    cols[0].metric("Citation valid", "yes" if valid else "no")
    cols[1].metric("Cited chunks", len(cited))
    cols[2].metric("Missing chunks", len(missing))

    if cited:
        st.markdown("**Cited chunk IDs**")
        st.code("\n".join(str(chunk_id) for chunk_id in cited), language="text")
    if missing:
        st.markdown("**Missing citation IDs**")
        st.code("\n".join(str(chunk_id) for chunk_id in missing), language="text")


def render_evidence(result: dict[str, Any]) -> None:
    evidence = result.get("evidence_preview") or []
    if not evidence:
        st.info("No evidence preview available.")
        return

    for index, hit in enumerate(evidence, start=1):
        title = f"{index}. {hit.get('chunk_id')} | {hit.get('ticker')} {hit.get('year')} p.{hit.get('page')}"
        with st.expander(title):
            st.write(
                {
                    "source_type": hit.get("source_type"),
                    "image_path": hit.get("image_path"),
                }
            )
            st.write(hit.get("preview") or "")


def render_guardrails(result: dict[str, Any]) -> None:
    st.markdown("**Security guardrail**")
    st.json(result.get("security") or {}, expanded=False)
    st.markdown("**Plain-text classifier**")
    st.json(result.get("plain_text") or {}, expanded=False)
    st.markdown("**Evidence guard**")
    st.json(result.get("evidence_guard") or {}, expanded=False)


def render_human_review(result: dict[str, Any]) -> None:
    review = result.get("human_review") or {}
    if not review:
        st.info("No human-review decision was recorded.")
        return

    st.json(review, expanded=False)
    if review.get("required"):
        st.warning("This response is queued for analyst review before release.")
        cols = st.columns(3)
        with cols[0]:
            if st.button("Approve draft", key=f"approve_{review.get('queue_id')}"):
                st.success("Marked approved in this UI session.")
        with cols[1]:
            if st.button("Request revision", key=f"revise_{review.get('queue_id')}"):
                st.info("Marked for revision in this UI session.")
        with cols[2]:
            if st.button("Reject draft", key=f"reject_{review.get('queue_id')}"):
                st.error("Marked rejected in this UI session.")


def render_result_details(result: dict[str, Any]) -> None:
    with st.expander("Citations", expanded=True):
        render_citations(result)
    with st.expander("Evidence", expanded=False):
        render_evidence(result)
    with st.expander("Security and guardrails", expanded=False):
        render_guardrails(result)
    with st.expander("Human review", expanded=result.get("final_status") == "pending_human_review"):
        render_human_review(result)
    with st.expander("Raw agent result", expanded=False):
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


def run_chat_turn(prompt: str, settings: dict[str, Any]) -> dict[str, Any]:
    config = build_config(prompt, settings)
    result = run_technique(settings["technique"], settings["llm_provider"], config)
    return summarize_result(result)


def render_chat(settings: dict[str, Any]) -> None:
    st.title("Financial Report Operations Copilot")
    st.caption("Ask financial-report questions with citation-grounded RAG, security guardrails, and HITL routing.")

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("result"):
                render_status(message["result"])
                render_result_details(message["result"])

    prompt = st.chat_input("Ask about annual reports, tables, charts, XBRL facts, or financial risks")
    if not prompt:
        return

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running financial agent..."):
            try:
                result = run_chat_turn(prompt, settings)
                answer = result.get("answer") or "No answer was generated."
            except Exception as exc:  # noqa: BLE001 - show app-safe error to analyst.
                result = {
                    "final_status": "error",
                    "answer": f"Agent run failed: {exc}",
                    "error": str(exc),
                }
                answer = result["answer"]

        st.markdown(answer)
        render_status(result)
        render_result_details(result)

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": answer,
            "result": result,
        }
    )
    st.session_state["last_result"] = result


def main() -> None:
    st.set_page_config(
        page_title="Financial Report Copilot",
        page_icon=None,
        layout="wide",
    )
    init_state()
    settings = sidebar_settings()
    render_chat(settings)


if __name__ == "__main__":
    main()
