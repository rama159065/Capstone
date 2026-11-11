"""Streamlit frontend of the Financial Report Copilot (financial_report_qa only).

The API still serves the other domains (`/chat/{domain}`); this UI is the financial-report product only.

Start the backend first:
    uvicorn scripts.api.api_app:app --host 127.0.0.1 --port 8000

Then launch this UI:
    streamlit run scripts/ui/api_streamlit_app.py

Deployment settings (environment variables, see .env.example):
    API_BASE_URL            FastAPI address (default http://127.0.0.1:8000)
    API_AUTH_TOKEN          sent as X-API-Key when the API requires a key
    API_TIMEOUT_SECONDS     per-request timeout (default 180)
    DEFAULT_LLM_PROVIDER    mock | hosted, preselected in the sidebar
    DEFAULT_EMBEDDING_PROVIDER  hashing | hosted
    UI_SHOW_ADVANCED        1 = open the retrieval and tracing settings by default
    UI_APP_TITLE            page heading (default "Financial Report Copilot")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from typing import Any

import streamlit as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.companies import NAMES as COMPANY_NAMES, SECTOR_NAMES  # noqa: E402 - after the repo is on sys.path


DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")   # set by the runtime profile
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "").strip()
API_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "180"))
LLM_PROVIDERS = ["mock", "hosted"]
EMBEDDING_PROVIDERS = ["hashing", "hosted"]
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "mock")
DEFAULT_EMBEDDING_PROVIDER = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "hashing")
SHOW_ADVANCED = os.getenv("UI_SHOW_ADVANCED", "0").lower() in {"1", "true", "yes"}
_COMPANIES: dict[str, tuple[float, list[dict[str, Any]]]] = {}
DOMAIN = "financial_report_qa"        # the only domain this UI serves
APP_TITLE = os.getenv("UI_APP_TITLE", "").strip() or "Financial Report Copilot"
APP_TAGLINE = ("Ask about company performance, compare peers and check growth. "
               "Every figure is cited to its page in the SEC annual report.")
DEFAULT_QUERY = "What drove revenue growth in the latest annual report?"


def api_headers() -> dict[str, str]:
    return {"X-API-Key": API_AUTH_TOKEN} if API_AUTH_TOKEN else {}


def post_json(api_base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **api_headers()},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=API_TIMEOUT_SECONDS) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach API backend: {exc}") from exc


def get_json(api_base: str, path: str) -> dict[str, Any]:
    request = urllib.request.Request(f"{api_base.rstrip('/')}{path}", headers=api_headers())
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - shown as UI status.
        return {"status": "unavailable", "error": str(exc)}


def covered_companies(api_base: str, ttl_seconds: float = 300) -> list[dict[str, Any]]:
    """Tickers, names, sectors and years from GET /companies, cached for a few minutes ([] if unavailable)."""
    cached = _COMPANIES.get(api_base)
    if cached and cached[0] > time.monotonic():
        return cached[1]
    rows = get_json(api_base, "/companies").get("companies") or []
    if rows:
        _COMPANIES[api_base] = (time.monotonic() + ttl_seconds, rows)
    return rows


def ticker_choices(companies: list[dict[str, Any]], sectors: list[str]) -> list[str]:
    """'' (any company) plus the covered tickers, limited to the chosen sectors."""
    return [""] + [row["ticker"] for row in companies if not sectors or row.get("sector") in sectors]


def year_choices(companies: list[dict[str, Any]], ticker: str) -> list[int | None]:
    """None (any year) plus the years with a filing for the ticker (all covered years without a ticker)."""
    years = {year for row in companies if not ticker or row["ticker"] == ticker for year in row.get("years") or []}
    return [None] + sorted(years, reverse=True)


# ----------------------------------------------------------------------------- feature guide

# Well-known companies used to show a "not available" answer when they are not in the served index.
NOT_COVERED_CANDIDATES = ("AAPL", "MSFT", "NVDA", "JPM", "XOM", "TSLA", "GOOGL", "META")

STATUS_HELP = {
    "answered": "Answered from the filings; every citation points to a retrieved page.",
    "evidence_only": "A hosted model's answer couldn't be verified, so the cited excerpts are shown instead.",
    "pending_human_review": "Drafted, but flagged for a person to check (advice wording, weak evidence, "
                            "an unsupported claim or an unverified calculation).",
    "not_available": "The indexed filings don't cover that company, year or measure. No figure is given.",
    "insufficient_evidence": "The searches found nothing usable for this question.",
    "not_grounded": "Withheld: the citations didn't check out or named a company you didn't ask about.",
    "refused": "Blocked by a guardrail: advice, prompt injection or an unsafe request.",
}
READING_TIPS = (
    "**Critic support check**: *supported* means the cited pages state every claim and every figure.",
    "**Agent run**: the steps the agent took, the tools it used, and any retry or repair.",
    "**Evidence** and **Citations**: the exact chunks and pages the answer relies on.",
)


def short_name(ticker: str, companies: list[dict[str, Any]]) -> str:
    """'Costco' rather than 'Costco Wholesale Corporation': the clarifier's short name when it has one."""
    row = next((r for r in companies if r["ticker"] == ticker), {})
    name = COMPANY_NAMES.get(ticker) or row.get("company") or ticker
    for suffix in (" Corporation", " Corp.", " Corp", " Inc.", " Inc", " Company", " Co.", " plc", " Ltd."):
        name = name.removesuffix(suffix)
    return name.strip(" ,")


def coverage_summary(companies: list[dict[str, Any]]) -> str | None:
    """'5 filings · 4 companies · Retail · FY2024–2025' from GET /companies (None when unknown)."""
    if not companies:
        return None
    filings = sum(len(row.get("years") or []) for row in companies)
    sectors = sorted({row.get("sector") for row in companies if row.get("sector")})
    years = sorted({year for row in companies for year in row.get("years") or []})
    sector_text = (", ".join(sectors) if len(sectors) <= 3 else f"{len(sectors)} sectors") if sectors else ""
    year_text = (f"FY{years[0]}–{years[-1]}" if len(years) > 1 else f"FY{years[0]}") if years else ""
    parts = [f"{filings:,} filing{'s' * (filings != 1)}", f"{len(companies):,} compan{'ies' if len(companies) != 1 else 'y'}",
             sector_text, year_text]
    return " · ".join(part for part in parts if part)


def example_questions(companies: list[dict[str, Any]]) -> list[dict[str, str]]:
    """One working example per capability, built from the companies the served index actually holds."""
    if not companies:
        return [{"capability": "Cited answer", "question": DEFAULT_QUERY,
                 "look_for": "Citations to the filing pages"}]
    by_years = sorted(companies, key=lambda r: (-len(r.get("years") or []), r["ticker"]))
    first = by_years[0]
    first_name, latest = short_name(first["ticker"], companies), max(first.get("years") or [0])
    examples = [{"capability": "Cited answer from one filing",
                 "question": f"What was {first_name}'s net income in fiscal {latest}?",
                 "look_for": "Citations to the filing pages; critic support check"}]
    if len(first.get("years") or []) > 1:
        years = sorted(first["years"])
        examples.append({"capability": "Growth with a checked calculation",
                         "question": f"How much did {first_name}'s revenue grow from {years[-2]} to {years[-1]}?",
                         "look_for": "Agent run: one search per year, then the calculator"})
    peers = [r for r in companies if r["ticker"] != first["ticker"] and latest in (r.get("years") or [])]
    if peers:
        peer = short_name(peers[0]["ticker"], companies)
        examples.append({"capability": "Comparison across companies",
                         "question": f"Compare {first_name} and {peer} revenue in {latest}",
                         "look_for": "Agent run: one filtered search per company"})
        examples.append({"capability": "Follow-up questions",
                         "question": f"Compare {first_name} and {peer} revenue",
                         "look_for": "It asks which year, with options to click"})
    examples.append({"capability": "Risks and explanations",
                     "question": f"What are the main risk factors {first_name} describes?",
                     "look_for": "Answer from the risk-factor pages, with citations"})
    covered = {row["ticker"] for row in companies}
    missing = next((t for t in NOT_COVERED_CANDIDATES if t not in covered), None)
    if missing:
        examples.append({"capability": "Honest \"not available\"",
                         "question": f"What was {short_name(missing, [])}'s net income in fiscal {latest}?",
                         "look_for": "Says the filings aren't indexed; gives no figure"})
    examples.append({"capability": "Guardrails",
                     "question": f"Should I buy {first_name} stock?",
                     "look_for": "Refuses investment advice"})
    return examples


def render_guide(companies: list[dict[str, Any]], expanded: bool) -> None:
    """'What can I ask?' with clickable examples, plus how to read an answer."""
    with st.expander("What can I ask?", expanded=expanded):
        st.caption("Click an example to send it, or type your own question below.")
        for index, example in enumerate(example_questions(companies)):
            left, right = st.columns([2, 3])
            left.markdown(f"**{example['capability']}**  \n<small>{example['look_for']}</small>",
                          unsafe_allow_html=True)
            if right.button(example["question"], key=f"example-{index}", width="stretch"):
                st.session_state["queued_prompt"] = example["question"]
                st.rerun()
    with st.expander("How to read an answer", expanded=False):
        st.markdown("\n".join(f"- `{status}`: {text}" for status, text in STATUS_HELP.items()))
        st.markdown("\n".join(f"- {tip}" for tip in READING_TIPS))


def build_payload(query: str, settings: dict[str, Any]) -> dict[str, Any]:
    """Request body for /conversation and /chat/financial_report_qa."""
    return {
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
        "technique": settings["technique"],
        "retrieval_mode": settings["retrieval_mode"],
        "ticker": settings["ticker"] or None,
        "year": settings["year"],
        "sector": ",".join(settings.get("sectors") or []) or None,
        "self_consistency_samples": settings["self_consistency_samples"],
    }


def render_result(result: dict[str, Any]) -> None:
    status = result.get("final_status", "unknown")
    if status in {"refused", "error"}:
        st.error(status)
    elif status in {"pending_human_review", "not_available", "not_grounded"}:
        st.warning(status)
    else:
        st.success(status)

    answer = result.get("answer")
    if answer:
        st.markdown(answer)
    else:
        coverage = result.get("coverage") or {}
        reason = coverage.get("message") or coverage.get("reason") or result.get("explanation")
        st.markdown(f"No answer was returned. {reason}" if reason else "No answer was returned.")

    support = result.get("support_check") or {}
    if support:
        numeric = support.get("numeric") or {}
        claims = support.get("claims") or []
        summary = (f"Critic support check: **{support.get('verdict')}** · "
                   f"{sum(1 for c in claims if c.get('supported') is True)}/{len(claims)} claims supported · "
                   f"{numeric.get('checked', 0) - len(numeric.get('ungrounded') or [])}/{numeric.get('checked', 0)} "
                   "figures found in cited evidence")
        if (result.get("grounding") or {}).get("method") == "mock":
            summary += " (mock judge)"
        st.caption(summary)

    agent_run = result.get("agent_run")
    if agent_run:
        with st.expander(f"Agent run: {agent_run.get('steps')}/{agent_run.get('max_steps')} steps", expanded=False):
            st.markdown(" → ".join(f"`{name.removesuffix('_step')}`" for name in agent_run.get("path", [])))
            plan = result.get("plan") or {}
            st.caption(f"Planner ({plan.get('planner', '?')}) tools: {', '.join(plan.get('tools') or [])} · "
                       f"retrieval attempts: {agent_run.get('retrieval_attempts')} · repairs: {agent_run.get('repairs')}")
            if plan.get("adjustments"):
                st.caption("Rule adjustments: " + "; ".join(plan["adjustments"]))
            st.dataframe([{key: json.dumps(value) if isinstance(value, (list, dict)) else value
                           for key, value in call.items()} for call in agent_run.get("tool_calls") or []],
                         width="stretch")
            if agent_run.get("calculations"):
                st.json(agent_run["calculations"], expanded=False)

    with st.expander("Citations and Human Review", expanded=True):
        st.json(
            {
                "citation_check": result.get("citation_check"),
                "support_check": result.get("support_check"),
                "grounding": result.get("grounding"),
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


def option_index(options: list[str], value: str) -> int:
    return options.index(value) if value in options else 0


def sidebar() -> dict[str, Any]:
    st.sidebar.title("Filters and settings")
    advanced = st.sidebar.expander("Advanced: backend, retrieval and tracing", expanded=SHOW_ADVANCED)
    api_base = advanced.text_input("FastAPI base URL", DEFAULT_API_BASE)
    if get_json(api_base, "/health").get("status") != "ok":      # say nothing while it works
        st.sidebar.warning("The copilot service is not reachable right now. Please try again shortly.")

    settings: dict[str, Any] = {"api_base": api_base, "ticker": "", "year": None, "sectors": []}
    companies = covered_companies(api_base)
    names = {row["ticker"]: row.get("company") or row["ticker"] for row in companies}
    settings["clarify"] = st.sidebar.toggle(
        "Ask clarifying questions", True, help="Ask for a missing company, year or measure before running the agents.")
    settings["sectors"] = st.sidebar.multiselect(
        "Sectors (optional)", list(SECTOR_NAMES),
        help="Limit evidence to these sectors of the annual-report corpus.")
    if companies:
        settings["ticker"] = st.sidebar.selectbox(
            "Company (optional)", ticker_choices(companies, settings["sectors"]),
            format_func=lambda t: "Any company" if not t else f"{t} - {names.get(t, t)}")
        settings["year"] = st.sidebar.selectbox(
            "Fiscal year (optional)", year_choices(companies, settings["ticker"]),
            format_func=lambda y: "Any year" if y is None else str(y))
    else:                                    # older API without /companies: free text
        settings["ticker"] = st.sidebar.text_input("Ticker (optional)", "").strip().upper()
        year_value = st.sidebar.text_input("Fiscal year (optional)", "").strip()
        if year_value:
            try:
                settings["year"] = int(year_value)
            except ValueError:
                st.sidebar.warning("Year must be numeric; ignoring filter.")
    settings["technique"] = advanced.selectbox(
        "Technique", ["agentic_graph", "baseline_rag", "structured_few_shot", "decomposition", "self_consistency"],
        help="agentic_graph is the full agent; the others are single-prompt baselines for comparison.")

    settings.update({
        "llm_provider": advanced.selectbox("LLM provider", LLM_PROVIDERS,
                                           index=option_index(LLM_PROVIDERS, DEFAULT_LLM_PROVIDER)),
        "embedding_provider": advanced.selectbox("Embedding provider", EMBEDDING_PROVIDERS,
                                                 index=option_index(EMBEDDING_PROVIDERS, DEFAULT_EMBEDDING_PROVIDER)),
        "retrieval_mode": advanced.selectbox("Retrieval mode", ["hybrid", "dense", "bm25"], index=0,
                                             help="hybrid = dense + BM25 fused by RRF (recommended); dense or bm25 = one retriever only"),
        "source_type": advanced.text_input("Source type filter", "").strip(),
        "top_k": advanced.slider("Final top k", 1, 20, 5),
        "dense_top_k": advanced.slider("Dense candidates", 1, 100, 30),
        "bm25_top_k": advanced.slider("BM25 candidates", 1, 100, 30),
        "hash_dimensions": advanced.selectbox("Hash dimensions", [384, 768, 1024, 1536], index=0),
        "max_context_chars": advanced.slider("Max context chars", 1000, 64000, 12000, step=1000),
        "self_consistency_samples": advanced.slider("Self-consistency samples", 1, 5, 3),
        "enable_guardrails": advanced.toggle("Guardrails", True),
        "enable_human_review": advanced.toggle("Human review routing", True),
        "enable_langsmith": advanced.toggle("LangSmith tracing", False),
        "langsmith_project": advanced.text_input("LangSmith project", ""),
    })

    if st.sidebar.button("Clear chat", width="stretch"):
        st.session_state["messages"] = []
        st.session_state.pop("conversation_id", None)
        st.rerun()
    return settings


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    st.title(APP_TITLE)
    st.markdown(APP_TAGLINE)
    st.session_state.setdefault("messages", [])

    settings = sidebar()
    companies = covered_companies(settings["api_base"])
    coverage = coverage_summary(companies)
    st.caption((f"Coverage: **{coverage}** of SEC annual reports." if coverage
                else "Coverage: unknown (the API is not reachable).")
               + " Streamlit → FastAPI → LangGraph agents → hybrid RAG.")
    if settings.get("llm_provider") == "mock":
        st.caption(":orange[Test mode: answers come from the mock LLM, so their wording is placeholder text. "
                   "Retrieval, citations, guardrails and the agent steps are real.]")

    messages = st.session_state["messages"]
    render_guide(companies, expanded=not messages)
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

    prompt = st.chat_input("Ask about a company's annual report...") or st.session_state.pop("queued_prompt", None)
    if not prompt:
        return
    conversational = settings["clarify"]

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if conversational:
        with st.chat_message("assistant"):
            with st.spinner("Checking the question..."):
                try:
                    payload = {**build_payload(prompt, settings), "clarify": True,
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
        with st.spinner("Reading the filings..."):
            try:
                payload = build_payload(prompt, settings)
                result = post_json(settings["api_base"], f"/chat/{DOMAIN}", payload)
                answer = result.get("answer") or "No answer returned."
            except Exception as exc:  # noqa: BLE001 - display safe UI error.
                result = {"final_status": "error", "answer": str(exc), "error": str(exc)}
                answer = result["answer"]
        render_result(result)

    st.session_state["messages"].append({"role": "assistant", "content": answer, "result": result})


if __name__ == "__main__":
    main()
