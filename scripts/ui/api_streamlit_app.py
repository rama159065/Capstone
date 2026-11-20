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
    UI_SHOW_ADVANCED        1 = open the "Demo settings" panel by default
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
import uuid
from typing import Any

import streamlit as st

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.companies import NAMES as COMPANY_NAMES, SECTOR_NAMES  # noqa: E402 - after the repo is on sys.path
from scripts.ui.ui_mutual_fund import render_mutual_fund  # noqa: E402
from scripts.observability import langsmith_utils  # noqa: E402


DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")   # set by the runtime profile
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "").strip()
API_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "180"))
LLM_PROVIDERS = ["mock", "hosted"]
EMBEDDING_PROVIDERS = ["hashing", "hosted"]
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "mock")
DEFAULT_EMBEDDING_PROVIDER = os.getenv("DEFAULT_EMBEDDING_PROVIDER", "hashing")
SHOW_ADVANCED = os.getenv("UI_SHOW_ADVANCED", "0").lower() in {"1", "true", "yes"}
INDEX_BACKENDS = {"financial_vlm", "financial_qdrant", "financial_hybrid"}            # sector-corpus indexes (see uses_corpus_index)
BASELINE_TECHNIQUES = {"baseline_rag", "structured_few_shot", "decomposition", "self_consistency"}
# Request values for settings the sidebar doesn't show (same as the API's own defaults).
DEMO_DEFAULTS: dict[str, Any] = {
    "embedding_provider": DEFAULT_EMBEDDING_PROVIDER, "hash_dimensions": 384, "dense_top_k": 30, "bm25_top_k": 30,
    "source_type": "", "max_context_chars": 12000, "self_consistency_samples": 3, "langsmith_project": "",
}
_COMPANIES: dict[str, tuple[float, list[dict[str, Any]]]] = {}
DOMAIN = "financial_report_qa"        # the only domain this UI serves
APP_TITLE = os.getenv("UI_APP_TITLE", "").strip() or "Financial Report Copilot"
APP_TAGLINE = ("Your cited analyst for SEC annual reports and mutual fund holdings: answer questions, compare peers, "
               "track company and sector trends, look through fund portfolios and generate full reports. "
               "Every figure is traced to its page in the filing and checked before you see it.")
# Capability cards under the heading: (icon, title, what it does). "funds" is shown only when fund data is loaded.
CAPABILITIES: tuple[tuple[str, str, str, str], ...] = (
    ("core", "📄", "Cited answers",
     "Figures, risks and strategy from any indexed filing. Every claim links to its page; a critic checks each number."),
    ("core", "⚖️", "Peer comparisons",
     "Companies side by side for a year: one filtered search per company, arithmetic done by a calculator."),
    ("core", "📈", "Multi-year trends",
     "A measure across fiscal years, each year from its own filing, with changes and CAGR computed in code."),
    ("core", "🏭", "Sector trends",
     "One measure across a sector's companies, year by year: totals, medians and a fixed panel verified every year."),
    ("funds", "💼", "Mutual funds",
     "Holdings, sector mix, returns and net assets from SEC N-PORT filings, plus look-through growth of the top "
     "holdings from their own annual reports."),
    ("core", "📑", "Full reports",
     "A deterministic financial report (statements, ratios, charts) whose numbers come straight from the filing."),
    ("core", "🛡️", "Safe by design",
     "Refuses advice and prompt injection, says \"not available\" instead of guessing, and flags weak answers "
     "for human review."),
)
DEFAULT_QUERY = "What drove revenue growth in the latest annual report?"


def api_headers() -> dict[str, str]:
    return {"X-API-Key": API_AUTH_TOKEN} if API_AUTH_TOKEN else {}


def post_json(api_base: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to the API. With LangSmith tracing asked for (sidebar toggle or LANGSMITH_TRACING) and a key in this
    process, the call is a ``streamlit.<route>`` run and the API continues that trace (distributed tracing), so one
    trace runs Streamlit -> FastAPI -> agents -> retrieval -> LLM. Without a key here the trace starts at the API."""
    wanted = bool(payload.get("enable_langsmith")) or langsmith_utils.tracing_requested_by_env()
    if not (wanted and langsmith_utils.has_api_key()):
        return _post_json(api_base, path, payload)
    request_id = str(uuid.uuid4())
    route = path.strip("/").split("/")[0] or "api"
    project = payload.get("langsmith_project") or langsmith_utils.default_project("capstone-financial-report-qa-api")
    with langsmith_utils.request_tracing(enabled=True, project_name=project, tags=["capstone", "streamlit"],
                                         metadata={"client": "streamlit", "request_id": request_id,
                                                   "conversation_id": payload.get("conversation_id")}):
        return langsmith_utils.run_traced(
            f"streamlit.{route}", lambda question: _post_json(
                api_base, path, payload, {"X-Request-ID": request_id, **langsmith_utils.outgoing_trace_headers()}),
            payload.get("query"), run_type="chain")


def _post_json(api_base: str, path: str, payload: dict[str, Any],
               extra_headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **api_headers(), **(extra_headers or {})},
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


_FUNDS: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def served_funds(api_base: str, ttl_seconds: float = 300) -> list[dict[str, Any]]:
    """Funds from GET /mutual_fund/funds, cached for a few minutes ([] when the API has no fund data)."""
    cached = _FUNDS.get(api_base)
    if cached and cached[0] > time.monotonic():
        return cached[1]
    rows = get_json(api_base, "/mutual_fund/funds").get("funds") or []
    if rows:                                             # a failed call (busy, rate-limited) is retried next time
        _FUNDS[api_base] = (time.monotonic() + ttl_seconds, rows)
    return rows


def fund_name(fund: dict[str, Any]) -> str:
    """'Vanguard 500 Index Fund' from the filing's 'VANGUARD 500 INDEX FUND'."""
    return " ".join(w if w.isupper() and len(w) <= 3 and w not in {"THE", "AND", "FUND"} else w.capitalize()
                    for w in str(fund.get("series_name") or "").split()) or "the fund"


def fund_summary(funds: list[dict[str, Any]]) -> str | None:
    """'4 funds (Vanguard, N-PORT as of 2026-06-30)' for the coverage line (None without fund data)."""
    if not funds:
        return None
    families = sorted({f.get("family") for f in funds if f.get("family")})
    dates = sorted({x.get("period_end") for f in funds for x in f.get("filings") or [] if x.get("period_end")})
    detail = ", ".join(part for part in (", ".join(families), f"N-PORT as of {dates[-1]}" if dates else "") if part)
    return f"{len(funds)} mutual fund{'s' * (len(funds) != 1)}" + (f" ({detail})" if detail else "")


def report_profiles() -> list[dict[str, Any]]:
    """Companies and years the deterministic report pipeline has a profile for (configs/financial_reports)."""
    rows = []
    for path in sorted((ROOT / "configs" / "financial_reports").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("ticker") and data.get("year"):
            rows.append({"ticker": data["ticker"], "company": data.get("company") or data["ticker"], "year": data["year"]})
    return rows


def render_capabilities(funds_available: bool) -> None:
    """The capability cards under the heading."""
    cards = [c for c in CAPABILITIES if c[0] == "core" or funds_available]
    per_row = 4
    for start in range(0, len(cards), per_row):
        columns = st.columns(per_row)
        for column, (_, icon, title, text) in zip(columns, cards[start:start + per_row]):
            with column.container(border=True):
                st.markdown(f"**{icon} {title}**")
                st.caption(text)


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
    "report_handoff": "The router read this as a full-report request: confirm it in the chat to run the report pipeline.",
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


def sector_example(companies: list[dict[str, Any]]) -> dict[str, str] | None:
    """A sector trend over the latest years most of the sector's companies have filed (None when too thin)."""
    by_sector: dict[str, list[dict[str, Any]]] = {}
    for row in companies:
        if row.get("sector"):
            by_sector.setdefault(row["sector"], []).append(row)
    for sector, rows in sorted(by_sector.items(), key=lambda item: (item[0] == "Banking", -len(item[1]), item[0])):
        counts: dict[int, int] = {}
        for row in rows:
            for year in row.get("years") or []:
                counts[year] = counts.get(year, 0) + 1
        years = sorted(y for y, n in counts.items() if n >= max(2, (len(rows) + 1) // 2))[-3:]
        if len(rows) >= 2 and len(years) >= 2:
            measure = "net income" if sector == "Banking" else "revenue"
            phrase = " ".join(w if w.isupper() else w.lower() for w in sector.split())   # "technology and IT"
            return {"group": "Sectors", "capability": "Sector trend",
                    "question": f"How did {phrase} {measure} trend from {years[0]} to {years[-1]}?",
                    "look_for": "Table per year: sector total, median change, the companies verified in every year"}
    return None


def example_questions(companies: list[dict[str, Any]], funds: list[dict[str, Any]] | None = None,
                      reports: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    """One working example per capability, built from what the served index (and fund data) actually holds."""
    if not companies:
        return [{"capability": "Cited answer", "question": DEFAULT_QUERY,
                 "look_for": "Citations to the filing pages"}]
    by_years = sorted(companies, key=lambda r: (-len(r.get("years") or []), r["ticker"]))
    first = by_years[0]
    first_name, latest = short_name(first["ticker"], companies), max(first.get("years") or [0])
    examples = [{"group": "Companies", "capability": "Cited answer from one filing",
                 "question": f"What was {first_name}'s net income in fiscal {latest}?",
                 "look_for": "Citations to the filing pages; critic support check"}]
    years = sorted(first.get("years") or [])
    if len(years) > 1:
        examples.append({"group": "Companies", "capability": "Growth with a checked calculation",
                         "question": f"How much did {first_name}'s revenue grow from {years[-2]} to {years[-1]}?",
                         "look_for": "Agent run: one search per year, then the calculator"})
    if len(years) > 2:
        examples.append({"group": "Companies", "capability": "Multi-year trend",
                         "question": f"How did {first_name}'s revenue trend from {years[0]} to {years[-1]}?",
                         "look_for": "A figure per year from its own filing, year-on-year change and CAGR"})
    peers = [r for r in companies if r["ticker"] != first["ticker"] and latest in (r.get("years") or [])]
    if peers:
        peer = short_name(peers[0]["ticker"], companies)
        examples.append({"group": "Companies", "capability": "Comparison across companies",
                         "question": f"Compare {first_name} and {peer} revenue in {latest}",
                         "look_for": "Agent run: one filtered search per company"})
        examples.append({"group": "Companies", "capability": "Follow-up questions",
                         "question": f"Compare {first_name} and {peer} revenue",
                         "look_for": "It asks which year, with options to click"})
    examples.append({"group": "Companies", "capability": "Risks and explanations",
                     "question": f"What are the main risk factors {first_name} describes?",
                     "look_for": "Answer from the risk-factor pages, with citations"})
    sector = sector_example(companies)
    if sector:
        examples.append(sector)
    if funds:
        fund = fund_name(sorted(funds, key=lambda f: str(f.get("series_name")))[0])
        examples += [
            {"group": "Mutual funds", "capability": "Fund holdings",
             "question": f"What are the top 10 holdings of the {fund}?",
             "look_for": "Holdings table from the latest N-PORT filing, with weights"},
            {"group": "Mutual funds", "capability": "Who holds a company",
             "question": f"Which funds hold the most {first_name}?",
             "look_for": "Every loaded fund that holds it, by weight"},
            {"group": "Mutual funds", "capability": "Look-through growth",
             "question": f"How did revenue of the {fund}'s top 10 holdings grow?",
             "look_for": "Each holding's growth from its own annual report, weighted by the fund's position"},
        ]
    for report in (reports or [])[:1]:
        examples.append({"group": "Reports", "capability": "Full financial report",
                         "question": f"Generate a full financial report for {short_name(report['ticker'], [report])} "
                                     f"{report['year']}",
                         "look_for": "Hands off to the report pipeline: confirm, then download the PDF"})
    covered = {row["ticker"] for row in companies}
    missing = next((t for t in NOT_COVERED_CANDIDATES if t not in covered), None)
    if missing:
        examples.append({"group": "Safety", "capability": "Honest \"not available\"",
                         "question": f"What was {short_name(missing, [])}'s net income in fiscal {latest}?",
                         "look_for": "Says the filings aren't indexed; gives no figure"})
    examples.append({"group": "Safety", "capability": "Guardrails",
                     "question": f"Should I buy {first_name} stock?",
                     "look_for": "Refuses investment advice"})
    return examples


def render_guide(companies: list[dict[str, Any]], expanded: bool, funds: list[dict[str, Any]] | None = None) -> None:
    """'What can I ask?' with clickable examples grouped by topic, plus how to read an answer."""
    with st.expander("What can I ask?", expanded=expanded):
        st.caption("Click an example to send it, or type your own question below.")
        examples = example_questions(companies, funds, report_profiles() if companies else [])
        group = None
        for index, example in enumerate(examples):
            if example.get("group") and example["group"] != group:
                group = example["group"]
                st.markdown(f"##### {group}")
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
    elif status == "report_handoff":
        st.info(status)
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

    render_trend(result.get("trend") or {})
    render_sector_trend(result.get("sector_trend") or {})
    render_mutual_fund(result.get("mutual_fund") or {})

    agent_run = result.get("agent_run")
    if agent_run:
        with st.expander(f"Agent run: {agent_run.get('steps')}/{agent_run.get('max_steps')} steps", expanded=False):
            st.markdown(" → ".join(f"`{name.removesuffix('_step')}`" for name in agent_run.get("path", [])))
            route = result.get("route") or {}
            if route:
                st.caption(f"Router: **{route.get('route')}** → {route.get('flow')} (decided by {route.get('decided_by')}; "
                           f"LLM said {route.get('llm_route')!r}, rules said {route.get('rule_route')!r})")
                if route.get("adjustments"):
                    st.caption("Router adjustments: " + "; ".join(route["adjustments"]))
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


def render_trend(trend: dict[str, Any]) -> None:
    """The trend agent's verified series: a table of every year and a chart of the verified ones."""
    points = trend.get("points") or []
    if not points:
        return
    title = f"Trend: {trend.get('company_name') or trend.get('company')} {trend.get('metric')}"
    if trend.get("cagr") is not None:
        title += f" · CAGR {trend['cagr']:.2f}%"
    with st.expander(title, expanded=True):
        changes = {c["to"]: c for c in trend.get("changes") or [] if c.get("status") == "ok"}
        st.dataframe([{"fiscal year": f"FY{p['year']}", "figure": p.get("value_text"), "unit": p.get("unit"),
                       "change vs prior": None if p["year"] not in changes else f"{changes[p['year']]['pct_change']:+.2f}%"
                       if changes[p["year"]].get("pct_change") is not None else None,
                       "status": p.get("status"), "page": p.get("page"), "chunk": p.get("chunk_id")} for p in points],
                     width="stretch")
        chart = trend.get("chart") or []
        if len(chart) >= 2:
            st.line_chart([{"fiscal year": f"FY{row['year']}", "value": row["value"]} for row in chart],
                          x="fiscal year", y="value")
        if trend.get("missing"):
            st.caption("Not verified: " + ", ".join(trend["missing"]) + ". No figure is shown for these years.")
        st.caption("Each figure was copied by the model and kept only if that year's filing states it; "
                   "changes and CAGR are computed in code.")


def pct(value: float | None) -> str | None:
    return None if value is None else f"{value:+.2f}%"


def render_sector_trend(table: dict[str, Any]) -> None:
    """The sector trend agent's table: sector total, changes and breadth per year, then each company's figures."""
    rows = table.get("rows") or []
    if not rows:
        return
    title = f"Sector trend: {table.get('sector')} {table.get('metric')}"
    if table.get("cagr_total") is not None:
        title += f" · CAGR {table['cagr_total']:.2f}%"
    with st.expander(title, expanded=True):
        st.caption(f"Panel: {', '.join(table.get('panel_names') or []) or 'none'} "
                   f"({len(table.get('panel') or [])} of {len(table.get('planned') or [])} companies read; "
                   f"{table.get('sector_companies')} indexed in the sector). Totals in USD millions.")
        st.dataframe([{"fiscal year": f"FY{r['year']}",
                       "sector total (USD m)": None if r["total"] is None else round(r["total"], 1),
                       "total change": pct(r["total_pct_change"]),
                       "median company change": pct(r["median_company_pct_change"]),
                       "companies growing": None if r["growing"] is None else f"{r['growing']} of {r['companies']}"}
                      for r in rows], width="stretch")
        chart = table.get("chart") or []
        if len(chart) >= 2:
            st.line_chart([{"fiscal year": f"FY{c['year']}", "sector total": c["total"]} for c in chart],
                          x="fiscal year", y="sector total")
        st.dataframe([{"company": f"{c['company']} ({c['ticker']})", "in panel": c["in_panel"],
                       **{f"FY{p['year']}": p.get("value_text") if p.get("status") == "ok" else p.get("status")
                          for p in c.get("points") or []},
                       "CAGR": pct(c.get("cagr"))} for c in table.get("companies") or []], width="stretch")
        if table.get("excluded"):
            st.caption("Left out: " + "; ".join(f"{e['company']} ({e['reason']})" for e in table["excluded"]))
        st.caption("Each figure is read from that company's filing for that year (statement row, else the model) and "
                   "kept only if the filing states it; totals, changes and CAGR are computed in code over companies "
                   "verified in every year.")


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


def served_backend(health: dict[str, Any]) -> str | None:
    """The financial retrieval backend the API reports in /health (None for an older API)."""
    return (health.get("runtime") or {}).get("financial_backend")


def uses_corpus_index(backend: str | None) -> bool:
    """True when the API answers from a sector-corpus index (Qdrant or Chroma). That index keeps the embedding
    model it was built with and picks its own candidates, so the embedding, hash, candidate and source-type
    settings have no effect and are not shown."""
    return backend in INDEX_BACKENDS


def sidebar() -> dict[str, Any]:
    st.sidebar.title("Filters and settings")
    api_base = DEFAULT_API_BASE                                   # API_BASE_URL (deployment setting)
    health = get_json(api_base, "/health")
    if health.get("status") != "ok":                             # say nothing while it works
        st.sidebar.warning("The copilot service is not reachable right now. Please try again shortly.")
    corpus_index = uses_corpus_index(served_backend(health))

    settings: dict[str, Any] = {"api_base": api_base, "ticker": "", "year": None, "sectors": [], **DEMO_DEFAULTS}
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
    demo = st.sidebar.expander("Demo settings", expanded=SHOW_ADVANCED)
    settings["technique"] = demo.selectbox(
        "Technique", ["router", "agentic_graph", "trend_graph", "sector_trend_graph", "fund_graph", "baseline_rag", "structured_few_shot", "decomposition",
                      "self_consistency"],
        help="router (default) sends each question to the Q&A agent, the trend agent, the sector trend agent or "
             "the report handoff; agentic_graph is the Q&A agent alone; trend_graph is the multi-year trend agent; "
             "sector_trend_graph is the sector trend agent; fund_graph is the mutual fund agent (needs loaded "
             "N-PORT data; the router uses it for fund questions); the others are "
             "single-prompt baselines for comparison.")

    settings.update({
        "llm_provider": demo.selectbox("LLM provider", LLM_PROVIDERS,
                                       index=option_index(LLM_PROVIDERS, DEFAULT_LLM_PROVIDER)),
        "retrieval_mode": demo.selectbox("Retrieval mode", ["hybrid", "dense", "bm25"], index=0,
                                         help="hybrid = dense + BM25 fused by RRF (recommended); dense or bm25 = one retriever only"),
        "top_k": demo.slider("Evidence chunks (top k)", 1, 20, 5),
    })
    if settings["technique"] in BASELINE_TECHNIQUES:             # single-prompt baselines only
        settings["max_context_chars"] = demo.slider("Max context chars", 1000, 64000, 12000, step=1000)
    if settings["technique"] == "self_consistency":
        settings["self_consistency_samples"] = demo.slider("Self-consistency samples", 1, 5, 3)
    if not corpus_index:                                         # older default Chroma index: these still apply
        settings.update({
            "embedding_provider": demo.selectbox("Embedding provider", EMBEDDING_PROVIDERS,
                                                 index=option_index(EMBEDDING_PROVIDERS, DEFAULT_EMBEDDING_PROVIDER)),
            "hash_dimensions": demo.selectbox("Hash dimensions", [384, 768, 1024, 1536], index=0),
            "dense_top_k": demo.slider("Dense candidates", 1, 100, 30),
            "bm25_top_k": demo.slider("BM25 candidates", 1, 100, 30),
            "source_type": demo.text_input("Source type filter", "").strip(),
        })
    settings.update({
        "enable_guardrails": demo.toggle("Guardrails", True),
        "enable_human_review": demo.toggle("Human review routing", True),
        "enable_langsmith": demo.toggle("LangSmith tracing", False),
    })
    if settings["enable_langsmith"]:
        settings["langsmith_project"] = demo.text_input("LangSmith project", "")

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
    funds = served_funds(settings["api_base"]) if companies else []
    messages = st.session_state["messages"]
    if not messages:                                        # the overview, until the conversation starts
        render_capabilities(funds_available=bool(funds))
    coverage = coverage_summary(companies)
    fund_text = fund_summary(funds)
    st.caption((f"Coverage: **{coverage}** of SEC annual reports" + (f" · **{fund_text}**" if fund_text else "") + "."
                if coverage else "Coverage: unknown (the API is not reachable)."))
    if settings.get("llm_provider") == "mock":
        st.caption(":orange[Test mode: answers come from the mock LLM, so their wording is placeholder text. "
                   "Retrieval, citations, guardrails and the agent steps are real.]")

    render_guide(companies, expanded=not messages, funds=funds)
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

    prompt = st.chat_input("Ask about a company, a sector trend, a mutual fund or a full report...") or st.session_state.pop("queued_prompt", None)
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
