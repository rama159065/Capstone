"""Streamlit frontend for the FastAPI Agentic RAG backend.

Start the backend first:
    uvicorn scripts.api_app:app --host 127.0.0.1 --port 8000

Then launch this UI:
    streamlit run scripts/api_streamlit_app.py
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import streamlit as st


DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")   # set by the runtime profile
DOMAINS = {
    "financial_report_qa": "Financial Report QA",
    "procurement_vendor": "Procurement and Vendor Review",
    "travel_expense": "Travel and Expense Policy",
    "competitive_research": "Competitive Research",
    "incident_postmortem": "Incident Postmortem",
    "hr_policy": "HR Policy Navigator",
}
DEFAULT_QUERIES = {
    "financial_report_qa": "What does Apple say can affect quarterly net sales?",
    "procurement_vendor": "What evidence describes vendor evaluation criteria?",
    "travel_expense": "What documentation is required for travel expenses?",
    "competitive_research": "Compare competitor risk language about market demand.",
    "incident_postmortem": "What should be included in an incident postmortem?",
    "hr_policy": "What does the FMLA guide say about requesting leave?",
}


def post_json(api_base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach API backend: {exc}") from exc


def get_json(api_base: str, path: str) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(f"{api_base.rstrip('/')}{path}", timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - shown as UI status.
        return {"status": "unavailable", "error": str(exc)}


def build_payload(domain: str, query: str, settings: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "llm_provider": settings["llm_provider"],
        "embedding_provider": settings["embedding_provider"],
        "hash_dimensions": settings["hash_dimensions"],
        "top_k": settings["top_k"],
        "dense_top_k": settings["dense_top_k"],
        "bm25_top_k": settings["bm25_top_k"],
        "source_type": settings["source_type"] or None,
        "max_context_chars": settings["max_context_chars"],
        "enable_guardrails": settings["enable_guardrails"],
        "enable_human_review": settings["enable_human_review"],
        "enable_langsmith": settings["enable_langsmith"],
        "langsmith_project": settings["langsmith_project"] or None,
    }
    if domain == "financial_report_qa":
        payload.update(
            {
                "technique": settings["technique"],
                "retrieval_mode": settings["retrieval_mode"],
                "ticker": settings["ticker"] or None,
                "year": settings["year"],
                "self_consistency_samples": settings["self_consistency_samples"],
            }
        )
    return payload


def render_result(result: dict[str, Any]) -> None:
    status = result.get("final_status", "unknown")
    if status in {"refused", "error"}:
        st.error(status)
    elif status == "pending_human_review":
        st.warning(status)
    else:
        st.success(status)

    st.markdown(result.get("answer") or "No answer was returned.")

    with st.expander("Citations and Human Review", expanded=True):
        st.json(
            {
                "citation_check": result.get("citation_check"),
                "human_review": result.get("human_review"),
                "langsmith": result.get("langsmith"),
            },
            expanded=False,
        )

    evidence = result.get("evidence") or result.get("evidence_preview") or []
    with st.expander("Evidence", expanded=False):
        if not evidence:
            st.info("No evidence returned.")
        for row in evidence:
            st.markdown(f"**{row.get('chunk_id')}**")
            st.write(
                {
                    "title": row.get("title"),
                    "source_type": row.get("source_type"),
                    "page": row.get("page"),
                    "source_url": row.get("source_url"),
                    "image_path": row.get("image_path"),
                }
            )
            st.caption(row.get("preview") or "")

    with st.expander("Guardrails and Trace", expanded=False):
        st.json(
            {
                "security": result.get("security"),
                "plain_text": result.get("plain_text"),
                "plan": result.get("plan"),
                "evidence_guard": result.get("evidence_guard"),
                "output_filter": result.get("output_filter"),
                "orchestration": result.get("orchestration"),
            },
            expanded=False,
        )

    with st.expander("Raw API response", expanded=False):
        st.code(json.dumps(result, ensure_ascii=False, indent=2), language="json")


def render_turn(turn: dict[str, Any]) -> None:
    """Render one /conversation response: questions, an answer, or a report outcome."""
    status = turn.get("status")
    if status == "needs_clarification":
        st.info(turn.get("message") or "I need a few more details.")
    elif status == "answered":
        if turn.get("resolved_query") and turn.get("resolved_query") != turn.get("original"):
            st.caption(f"Answering: {turn['resolved_query']}")
        for note in turn.get("assumptions") or []:
            st.caption(f"Assumption: {note}")
        render_result(turn.get("result") or {})
    elif status == "report_generated":
        st.success(turn.get("message"))
        manifest = turn.get("report_manifest") or {}
        for item in manifest.get("artifacts", []):
            st.write(item.get("path") or item)
    elif status in {"report_unavailable", "report_failed", "cancelled"}:
        (st.error if status == "report_failed" else st.warning)(turn.get("message"))
    else:
        render_result(turn)


def answer_options(turn: dict[str, Any], key: str) -> None:
    """Clickable answers for the latest clarification; a click sends the option as the next message."""
    for index, question in enumerate(turn.get("questions") or []):
        options = question.get("options") or []
        st.markdown(f"**{question.get('question')}**")
        if len(options) > 8:
            choice = st.selectbox("Choose", options, key=f"{key}-{index}-select", label_visibility="collapsed")
            if st.button("Send", key=f"{key}-{index}-send"):
                st.session_state["queued_prompt"] = choice
                st.rerun()
        elif options:
            for column, option in zip(st.columns(len(options)), options):
                if column.button(option, key=f"{key}-{index}-{option}"):
                    st.session_state["queued_prompt"] = option
                    st.rerun()
    if turn.get("questions"):
        st.caption("Or type your own answer. Say 'continue' to answer as asked, or 'start over' for a new question.")


def sidebar() -> dict[str, Any]:
    st.sidebar.title("API Settings")
    api_base = st.sidebar.text_input("FastAPI base URL", DEFAULT_API_BASE)
    domain = st.sidebar.selectbox("Domain", list(DOMAINS), format_func=lambda value: DOMAINS[value])
    health = get_json(api_base, "/health")
    if health.get("status") == "ok":
        runtime = health.get("runtime") or {}
        st.sidebar.success(f"Backend connected ({runtime.get('profile', '?')}, "
                           f"financial retrieval: {runtime.get('financial_backend', '?')})")
    else:
        st.sidebar.warning("Backend unavailable")

    st.sidebar.divider()
    settings = {
        "api_base": api_base,
        "domain": domain,
        "technique": st.sidebar.selectbox(
            "Financial technique",
            ["agentic_graph", "baseline_rag", "structured_few_shot", "decomposition", "self_consistency"],
            index=0,
        ),
        "llm_provider": st.sidebar.selectbox("LLM provider", ["mock", "hosted"], index=0),
        "embedding_provider": st.sidebar.selectbox("Embedding provider", ["hashing", "hosted"], index=0),
        "retrieval_mode": st.sidebar.selectbox("Financial retrieval mode", ["hybrid", "dense"], index=0),
        "clarify": st.sidebar.toggle("Ask clarifying questions (financial)", True,
                                     help="Ask for a missing company, year or measure before running the agents."),
        "ticker": st.sidebar.text_input("Financial ticker (optional)", "").strip().upper(),
        "year": None,
        "source_type": st.sidebar.text_input("Source type filter", "").strip(),
        "top_k": st.sidebar.slider("Final top k", 1, 20, 5),
        "dense_top_k": st.sidebar.slider("Dense candidates", 1, 100, 30),
        "bm25_top_k": st.sidebar.slider("BM25 candidates", 1, 100, 30),
        "hash_dimensions": st.sidebar.selectbox("Hash dimensions", [384, 768, 1024, 1536], index=0),
        "max_context_chars": st.sidebar.slider("Max context chars", 1000, 64000, 12000, step=1000),
        "self_consistency_samples": st.sidebar.slider("Self-consistency samples", 1, 5, 3),
        "enable_guardrails": st.sidebar.toggle("Guardrails", True),
        "enable_human_review": st.sidebar.toggle("Human review routing", True),
        "enable_langsmith": st.sidebar.toggle("LangSmith tracing", False),
        "langsmith_project": st.sidebar.text_input("LangSmith project", ""),
    }
    year_value = st.sidebar.text_input("Financial year", "").strip()
    if year_value:
        try:
            settings["year"] = int(year_value)
        except ValueError:
            st.sidebar.warning("Year must be numeric; ignoring filter.")

    if st.sidebar.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state.pop("conversation_id", None)
        st.rerun()
    return settings


def main() -> None:
    st.set_page_config(page_title="Capstone Agentic RAG API UI", layout="wide")
    st.title("Track B Agentic RAG Copilot")
    st.caption("Streamlit frontend -> FastAPI backend -> LangGraph Agentic RAG + LangSmith tracing")
    st.session_state.setdefault("messages", [])

    settings = sidebar()
    domain = settings["domain"]

    messages = st.session_state["messages"]
    for position, message in enumerate(messages):
        with st.chat_message(message["role"]):
            if message.get("turn"):
                render_turn(message["turn"])
                if position == len(messages) - 1:
                    answer_options(message["turn"], key=f"q{position}")
            else:
                st.markdown(message["content"])
                if message.get("result"):
                    render_result(message["result"])

    prompt = st.chat_input(DEFAULT_QUERIES[domain]) or st.session_state.pop("queued_prompt", None)
    if not prompt:
        return
    conversational = domain == "financial_report_qa" and settings["clarify"]

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if conversational:
        with st.chat_message("assistant"):
            with st.spinner("Checking the question..."):
                try:
                    payload = {**build_payload(domain, prompt, settings), "clarify": True,
                               "conversation_id": st.session_state.get("conversation_id")}
                    turn = post_json(settings["api_base"], "/conversation/financial_report_qa", payload)
                    st.session_state["conversation_id"] = turn.get("conversation_id")
                except Exception as exc:  # noqa: BLE001 - display safe UI error.
                    turn = {"status": "error", "final_status": "error", "answer": str(exc), "error": str(exc)}
            render_turn(turn)
            # Same key the history loop gives this message on the next run, so a click is not lost.
            answer_options(turn, key=f"q{len(messages)}")
        messages.append({"role": "assistant", "content": turn.get("message") or "", "turn": turn})
        return

    with st.chat_message("assistant"):
        with st.spinner("Calling FastAPI backend..."):
            try:
                payload = build_payload(domain, prompt, settings)
                result = post_json(settings["api_base"], f"/chat/{domain}", payload)
                answer = result.get("answer") or "No answer returned."
            except Exception as exc:  # noqa: BLE001 - display safe UI error.
                result = {"final_status": "error", "answer": str(exc), "error": str(exc)}
                answer = result["answer"]
        render_result(result)

    st.session_state["messages"].append({"role": "assistant", "content": answer, "result": result})


if __name__ == "__main__":
    main()
