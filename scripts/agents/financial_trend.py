"""Multi-year trend agent: the plan, the per-year searches and the verified series behind a trend answer.

The trend flow (``financial_prompting_agent.financial_graph``, technique ``trend_graph`` or the router's ``trend``
route) answers "how did X's <measure> change over the years" in four steps:

1. ``trend_plan``: one company, one measure and the fiscal years to cover. Asked years are kept only if the index
   holds them; with no years, every indexed year (at most ``MAX_TREND_YEARS``) is used. A plan that can't make a
   trend (no company, no measure, fewer than two years) is marked ``feasible=False`` and the router answers the
   question as an ordinary lookup instead.
2. One filtered search per fiscal year (``year_query``), so each year's figure comes from that year's filing.
3. Per year, the LLM copies the figure from the evidence (Series JSON); ``validate_point`` keeps it only if the cited
   chunk belongs to that year's search **and** the chunk's text contains the number. Nothing is estimated.
4. ``summarize`` computes the year-over-year changes and the CAGR in code (never by the LLM) and records each one
   as a calculation, so the critic's figure check accepts them and the answer agent only has to describe them.

Years whose figure isn't verified are listed in ``missing``; the graph retries them once with a broadened query.
"""
from __future__ import annotations

import re
from typing import Any

from scripts.agents import financial_tools as tools

MAX_TREND_YEARS = 5
SERIES_CONTEXT_CHARS = 6000          # evidence per extraction call: one year's hits, inside the 8k-token window
TREND_TOOLS = ("coverage_lookup", "text_search", "series_extract", "trend_calculator")
SCALE_WORDS = ("thousand", "million", "billion", "trillion")
_FIGURE = re.compile(r"\(?-?\$?\s*\d[\d,]*(?:\.\d+)?\)?")


def trend_plan(query: str, ticker: str | None, year: int | None, clarifier, max_years: int = MAX_TREND_YEARS) -> dict:
    """Company, measure, years and one search target per year. ``feasible`` says whether a trend can be built."""
    from scripts.agents.financial_clarifier import name
    companies = [ticker.upper()] if ticker else clarifier.companies_in(query)
    metric = clarifier.metric_in(query)
    asked = clarifier.years_in(query)
    directory = clarifier.directory or {}
    company = companies[0] if len(companies) == 1 else None
    problems: list[str] = []
    if len(companies) != 1:
        problems.append("a trend needs exactly one company" if companies else "no company named")
    if not metric:
        problems.append("no measure named (for example revenue or net income)")
    available = directory.get(company, []) if company else []
    if company and directory and company not in directory:
        problems.append(f"{name(company)} is not in the index")
    if asked:
        years = [y for y in asked if not directory or y in available]
        skipped = [y for y in asked if y not in years]
    else:
        pool = [y for y in available if not year or y <= year]
        years, skipped = pool[-max_years:], []
    years = years[-max_years:]
    if company and len(years) < 2 and not any("not in the index" in p for p in problems):
        problems.append("fewer than two fiscal years are indexed for it" if not asked
                        else "the question covers fewer than two indexed fiscal years")
    return {
        "intent": "trend",
        "company": company,
        "company_name": name(company) if company else None,
        "metric": metric,
        "years": years,
        "skipped_years": skipped,
        "available_years": available,
        "tools": list(TREND_TOOLS),
        "targets": [{"ticker": company, "year": y} for y in years] if company else [],
        "skipped_targets": [{"ticker": company, "year": y, "reason": "year_not_in_index"} for y in skipped],
        "feasible": not problems,
        "problems": problems,
        "planner": "rules",
        "adjustments": [],
    }


def year_query(plan: dict, year: int, metric_phrases: list[str], broadened: bool = False) -> str:
    """The search for one fiscal year; a retry adds the measure's synonyms and statement names."""
    query = f"{plan.get('company_name') or plan.get('company')} {plan.get('metric')} fiscal year {year}"
    if broadened:
        extra = [p for p in dict.fromkeys(metric_phrases) if p != plan.get("metric")][:4]
        query += " " + " ".join([*extra, "consolidated statements", "annual report"])
    return query


def parse_figure(text: str) -> float | None:
    """'$391,035' -> 391035.0, '(1,234)' -> -1234.0; anything else -> None."""
    match = _FIGURE.search(str(text or ""))
    if not match:
        return None
    raw = match.group(0)
    negative = raw.strip().startswith("(") and raw.strip().endswith(")") or "-" in raw
    digits = re.sub(r"[^\d.]", "", raw)
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if negative else value


def validate_point(raw: dict | None, hits: list[dict[str, Any]], year: int) -> dict[str, Any]:
    """Keep the LLM's figure only if its chunk is in this year's evidence and states the number."""
    raw = raw if isinstance(raw, dict) else {}
    value_text = str(raw.get("value") if raw.get("value") is not None else "").strip()
    point: dict[str, Any] = {"year": year, "value": None, "value_text": value_text or None,
                             "unit": str(raw.get("unit") or "").strip(), "chunk_id": raw.get("chunk_id"),
                             "quote": str(raw.get("quote") or "")[:300]}
    if not value_text or value_text.lower() in {"null", "none", "n/a", "not found", "unknown"}:
        return {**point, "status": "not_found"}
    value = parse_figure(value_text)
    if value is None:
        return {**point, "status": "unparseable"}
    hit = next((h for h in hits if str(h.get("chunk_id")) == str(raw.get("chunk_id"))), None)
    if hit is None:
        return {**point, "status": "wrong_chunk"}
    if abs(value) not in tools.evidence_numbers([hit]):
        return {**point, "status": "not_in_evidence"}
    return {**point, "value": value, "page": hit.get("page"), "status": "ok"}


def scale_of(unit: str) -> str | None:
    lower = (unit or "").lower()
    return next((word for word in SCALE_WORDS if word in lower), None)


def _fmt(value: float) -> str:
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def summarize(plan: dict, points: list[dict[str, Any]]) -> dict[str, Any]:
    """Year-over-year changes and CAGR over the verified points, computed in code and recorded as calculations."""
    ok = sorted((p for p in points if p.get("status") == "ok"), key=lambda p: p["year"])
    changes, calculations = [], []
    for a, b in zip(ok, ok[1:]):
        if scale_of(a["unit"]) and scale_of(b["unit"]) and scale_of(a["unit"]) != scale_of(b["unit"]):
            changes.append({"from": a["year"], "to": b["year"], "status": "unit_mismatch"})
            continue
        diff = b["value"] - a["value"]
        pct = diff / abs(a["value"]) * 100 if a["value"] else None
        calculations.append({"expression": f"{_fmt(b['value'])} - {_fmt(a['value'])}", "value": round(diff, 4),
                             "grounded": True, "kind": "change", "from": a["year"], "to": b["year"]})
        if pct is not None:
            calculations.append({"expression": f"({_fmt(b['value'])} - {_fmt(a['value'])}) / {_fmt(abs(a['value']))} * 100",
                                 "value": round(pct, 4), "grounded": True, "kind": "pct_change",
                                 "from": a["year"], "to": b["year"]})
        changes.append({"from": a["year"], "to": b["year"], "change": round(diff, 4),
                        "pct_change": None if pct is None else round(pct, 4), "status": "ok"})
    cagr = None
    scales = {scale_of(p["unit"]) for p in ok} - {None}
    if len(ok) >= 2 and len(scales) <= 1 and ok[0]["value"] > 0 and ok[-1]["value"] > 0 and ok[-1]["year"] > ok[0]["year"]:
        n = ok[-1]["year"] - ok[0]["year"]
        if n >= 2:
            cagr = round(((ok[-1]["value"] / ok[0]["value"]) ** (1 / n) - 1) * 100, 4)
            calculations.append({"expression": f"(({_fmt(ok[-1]['value'])} / {_fmt(ok[0]['value'])}) ** (1 / {n}) - 1) * 100",
                                 "value": cagr, "grounded": True, "kind": "cagr",
                                 "from": ok[0]["year"], "to": ok[-1]["year"]})
    by_year = {p["year"]: p for p in points}
    ordered = [by_year[y] for y in plan.get("years", []) if y in by_year]
    return {
        "company": plan.get("company"),
        "company_name": plan.get("company_name"),
        "metric": plan.get("metric"),
        "years": plan.get("years", []),
        "points": ordered,
        "verified_points": len(ok),
        "changes": changes,
        "cagr": cagr,
        "missing": [tools.target_label({"ticker": plan.get("company"), "year": p["year"]})
                    for p in ordered if p.get("status") != "ok"],
        "chart": [{"year": p["year"], "value": p["value"], "unit": p["unit"]} for p in ok],
        "calculations": calculations,
    }


def series_text(trend: dict) -> str:
    rows = []
    for p in trend.get("points", []):
        if p.get("status") == "ok":
            rows.append(f"FY{p['year']}: {p['value_text']} {p['unit']}".rstrip() + f" [{p['chunk_id']}]")
        else:
            rows.append(f"FY{p['year']}: not verified ({p.get('status')})")
    return "\n".join(rows) or "none"


def changes_text(trend: dict) -> str:
    rows = []
    for c in trend.get("changes", []):
        if c.get("status") != "ok":
            rows.append(f"FY{c['from']} to FY{c['to']}: not computed ({c.get('status')})")
            continue
        pct = "" if c.get("pct_change") is None else f", {c['pct_change']:+.2f}%"
        rows.append(f"FY{c['from']} to FY{c['to']}: {c['change']:+,.2f}{pct}")
    if trend.get("cagr") is not None:
        first, last = trend["chart"][0]["year"], trend["chart"][-1]["year"]
        rows.append(f"CAGR FY{first} to FY{last}: {trend['cagr']:.2f}%")
    return "\n".join(rows) or "none (fewer than two verified years)"
