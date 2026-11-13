"""Sector trend agent: how one measure moved across one sector's companies, fiscal year by fiscal year.

The sector flow (``financial_prompting_agent.financial_graph``, technique ``sector_trend_graph`` or the router's
``sector_trend`` route) answers "how did healthcare revenue trend from 2022 to 2025?" in five steps:

1. ``sector_plan``: one sector, one measure and the fiscal years. The companies are the sector's indexed filers
   that have a filing for every one of those years (at most ``RAG_SECTOR_MAX_COMPANIES``); the others are listed
   with the reason they were left out. Banks are refused measures their income statements don't have.
2. One filtered search per company x year, so each figure comes from that company's own filing for that year.
3. Per company-year, code first: ``label_match`` reads the statement row ("Total net revenues  56,334  54,318") in
   the column whose header is that fiscal year. Without one, a broadened, deeper search and the row again; only then
   does the LLM copy the figure, and ``on_metric_line`` keeps it only if a line of its chunk prints it next to the
   measure's name. Either way ``financial_trend.validate_point`` keeps a figure only if the cited chunk belongs to
   that company-year's search and states the number. Chunks the injection guard flags are dropped, not trusted.
4. ``aggregate`` builds a fixed panel: the companies verified in every year, so a change in the total is a change
   in performance, not in who was counted. Per year it computes the sector total (in millions, for companies whose
   unit scale is known), its year-over-year change, the median company change and how many companies grew; plus
   the total's CAGR. All of it is computed in code and recorded as calculations for the critic's figure check.
   A company change beyond ±``RAG_SECTOR_OUTLIER_PCT`` (default 50%) and a panel under half the companies read
   send the answer to human review.
5. The answer agent describes the table; the critic and governance run as in every other flow.
"""
from __future__ import annotations

import os
import re
import statistics
from typing import Any

from scripts.agents import financial_trend as trend
from scripts.common.companies import BANKING, NAMES, SECTOR_NAMES, SECTOR_OF, normalize_ticker

SECTOR_TOOLS = ("coverage_lookup", "text_search", "series_extract", "sector_calculator")
DEFAULT_MAX_COMPANIES = 8
MIN_PANEL = 2                 # a sector trend needs at least two companies verified in every year

# Phrases that name a sector, matched longest first ("technology and it" before "technology").
SECTOR_ALIASES: dict[str, tuple[str, ...]] = {
    "Banking": ("banking sector", "banking industry", "banking", "banks"),
    "Insurance": ("insurance sector", "insurance industry", "insurance", "insurers"),
    "Financial services": ("financial services",),
    "Oil and gas": ("oil and gas", "oil & gas", "oil companies"),
    "Energy and utilities": ("energy and utilities", "utilities", "utility sector", "utility companies"),
    "Healthcare": ("healthcare", "health care", "pharmaceutical", "pharmaceuticals", "pharma"),
    "Technology and IT": ("technology and it", "information technology", "technology sector", "tech sector",
                          "tech companies", "technology companies", "technology"),
    "Retail": ("retail sector", "retailers", "retail"),
    "Telecom and media": ("telecom and media", "telecommunications", "telecom", "media companies", "media sector"),
    "Industrials": ("industrials", "industrial sector", "industrial companies"),
    "Consumer goods": ("consumer goods", "consumer staples", "consumer products"),
    "Materials and chemicals": ("materials and chemicals", "chemicals", "chemical companies", "materials sector",
                                "materials"),
    "Real estate": ("real estate", "reits", "reit"),
}
assert set(SECTOR_ALIASES) == set(SECTOR_NAMES)

# Statement rows read without the LLM: the label must start the row and be followed by the figures.
STATEMENT_LABELS: dict[str, tuple[str, ...]] = {
    "revenue": ("total net revenues", "total net revenue", "total revenues", "total revenue", "net revenues",
                "net revenue", "total net sales", "net sales", "revenues", "revenue"),
    "net income": ("net income attributable to common shareholders", "net income attributable to common stockholders",
                   "net income", "net earnings"),
    "net interest income": ("net interest income",),
    "operating income": ("operating income", "income from operations"),
    "gross profit": ("gross profit",),
    "total assets": ("total assets",),
    "total liabilities": ("total liabilities",),
}
_ROW_FIGURE = re.compile(r"\(?-?\$?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?-?\$?\s?\d+\.\d+\)?")
_YEAR = re.compile(r"(?<!\d)(20[0-4]\d)(?!\d)")
_SCALE = re.compile(r"\bin (thousands|millions|billions)\b", re.I)
_TO_MILLIONS = {"thousand": 0.001, "million": 1.0, "billion": 1000.0, "trillion": 1_000_000.0}


def max_companies() -> int:
    try:
        return max(MIN_PANEL, int(os.getenv("RAG_SECTOR_MAX_COMPANIES", DEFAULT_MAX_COMPANIES)))
    except ValueError:
        return DEFAULT_MAX_COMPANIES


# ----------------------------------------------------------------------------- parsing

def sectors_in(text: str) -> list[str]:
    """Sectors the text names, in the order it names them; a longer phrase hides a shorter one inside it."""
    lower = (text or "").lower()
    found = []                                                     # (start, end, sector)
    for sector, phrases in SECTOR_ALIASES.items():
        for phrase in sorted(phrases, key=len, reverse=True):
            match = re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", lower)
            if match:
                found.append((match.start(), match.end(), sector))
                break
    found = [f for f in found if not any(o[0] <= f[0] and f[1] <= o[1] and (o[1] - o[0]) > (f[1] - f[0])
                                         for o in found)]
    ordered: list[str] = []
    for _, _, sector in sorted(found):
        if sector not in ordered:
            ordered.append(sector)
    return ordered


def sector_from_filter(value: str | None) -> list[str]:
    """The request's sector filter ('healthcare', 'Oil and gas', 'oil_and_gas', comma-separated) as sector names."""
    from scripts.common.sector_corpus import sector_slug
    by_slug = {sector_slug(name): name for name in SECTOR_NAMES}
    names = []
    for part in str(value or "").split(","):
        name = by_slug.get(sector_slug(part.strip())) if part.strip() else None
        if name and name not in names:
            names.append(name)
    return names


def members(sector: str, directory: dict[str, list[int]]) -> list[str]:
    """Indexed companies of a sector, alphabetical."""
    return sorted(t for t in directory if SECTOR_OF.get(normalize_ticker(t)) == sector)


# ----------------------------------------------------------------------------- plan

def sector_plan(query: str, clarifier, *, sector_filter: str | None = None, max_years: int = trend.MAX_TREND_YEARS,
                limit: int | None = None) -> dict[str, Any]:
    """Sector, measure, years, the companies to read and one search target per company-year."""
    from scripts.agents.financial_clarifier import NOT_REPORTED_BY_BANKS, BANK_MEASURES
    limit = limit or max_companies()
    named = sectors_in(query) or sector_from_filter(sector_filter)
    sector = named[0] if len(named) == 1 else None
    metric = clarifier.metric_in(query)
    asked = clarifier.years_in(query)
    directory = clarifier.directory or {}
    problems: list[str] = []
    if not named:
        problems.append("no sector named")
    elif len(named) > 1:
        problems.append(f"one sector at a time ({', '.join(named)} were named)")
    if not metric:
        problems.append("no measure named (for example revenue or net income)")
    elif sector == BANKING and metric in NOT_REPORTED_BY_BANKS:
        problems.append(f"banks don't report {metric}; ask for {', '.join(m.lower() for m in BANK_MEASURES)}")
    companies = members(sector, directory) if sector else []
    if sector and not companies:
        problems.append(f"no {sector} company is in the index")
    indexed_years = sorted({y for t in companies for y in directory.get(t, [])})
    if asked:
        years = [y for y in asked if y in indexed_years]
        skipped_years = [y for y in asked if y not in years]
    else:
        years, skipped_years = indexed_years[-max_years:], []
    years = years[-max_years:]
    if sector and companies and len(years) < 2:
        problems.append("fewer than two of the asked fiscal years are indexed for this sector" if asked
                        else "fewer than two fiscal years are indexed for this sector")
    eligible = [t for t in companies if all(y in directory.get(t, []) for y in years)]
    chosen = eligible[:limit]
    excluded = [{"ticker": t, "company": NAMES.get(t, t), "reason": "no filing indexed for "
                 + ", ".join(f"FY{y}" for y in years if y not in directory.get(t, []))}
                for t in companies if t not in eligible]
    excluded += [{"ticker": t, "company": NAMES.get(t, t), "reason": f"over the {limit}-company limit"}
                 for t in eligible[limit:]]
    if sector and companies and len(years) >= 2 and len(chosen) < MIN_PANEL:
        problems.append(f"fewer than {MIN_PANEL} {sector} companies have a filing for every year "
                        + "–".join(f"FY{y}" for y in (years[0], years[-1])))
    return {
        "intent": "sector_trend",
        "sector": sector,
        "sectors_named": named,
        "metric": metric,
        "years": years,
        "skipped_years": skipped_years,
        "sector_companies": companies,
        "companies": chosen,
        "excluded": excluded,
        "tools": list(SECTOR_TOOLS),
        "targets": [{"ticker": t, "year": y} for t in chosen for y in years],
        "skipped_targets": [{"ticker": None, "year": y, "reason": "year_not_in_index"} for y in skipped_years],
        "feasible": not problems,
        "problems": problems,
        "planner": "rules",
        "adjustments": [],
    }


def not_available_message(plan: dict) -> str:
    sector = plan.get("sector") or " and ".join(plan.get("sectors_named") or []) or "the sector"
    return (f"I can't build a sector trend for {sector}: " + "; ".join(plan.get("problems") or ["no plan"]) + ".")


def company_query(plan: dict, ticker: str, year: int, phrases: list[str], broadened: bool = False) -> str:
    company_plan = {"company": ticker, "company_name": NAMES.get(ticker, ticker), "metric": plan.get("metric")}
    return trend.year_query(company_plan, year, phrases, broadened=broadened)


# ----------------------------------------------------------------------------- extraction without the LLM

def _row_cells(line: str) -> tuple[str, str]:
    """('total net revenues', '$ 56,334 $ 54,318') for a statement row, in plain text or a Markdown table."""
    text = line.replace("|", " ").replace("—", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower(), text


def label_match(hits: list[dict[str, Any]], metric: str | None, year: int) -> dict[str, Any] | None:
    """The figure for ``year`` from a statement row labelled with the measure, read in the column of the nearest
    header above it that lists the fiscal years. None when no row and header line up exactly."""
    labels = STATEMENT_LABELS.get(metric or "")
    if not labels:
        return None
    for hit in hits:
        text = str(hit.get("text") or "")
        header: list[int] = []
        scale = _SCALE.search(text)
        for line in text.splitlines():
            lower, plain = _row_cells(line)
            years = list(dict.fromkeys(int(y) for y in _YEAR.findall(lower)))
            if len(years) >= 2 and not _ROW_FIGURE.search(_YEAR.sub(" ", lower)):
                header = years                              # "2024 2023 2022" or "| | 2024 | 2023 |"
                continue
            if not header or year not in header:
                continue
            label = next((l for l in labels if lower.startswith(l)), None)
            if not label:
                continue
            rest = plain[len(label):]
            if not re.match(r"^\s*[\$(:\-]?\s*\(?\d", rest):
                continue                                    # "Net revenues decreased ..." is prose, not a row
            figures = _ROW_FIGURE.findall(rest)
            if len(figures) != len(header):
                continue
            value = figures[header.index(year)].strip()
            return {"value": value.replace("$", "").strip(), "unit": scale.group(1).lower() if scale else "",
                    "chunk_id": hit.get("chunk_id"), "quote": plain[:160]}
    return None


def on_metric_line(point: dict[str, Any], hits: list[dict[str, Any]], phrases: list[str]) -> bool:
    """An LLM-copied figure counts only if a line of its chunk prints it next to a name of the measure: a number
    that is merely somewhere in the chunk (a page number, a segment, another year) is not the measure."""
    hit = next((h for h in hits if str(h.get("chunk_id")) == str(point.get("chunk_id"))), None)
    value = str(point.get("value_text") or "").replace("$", "").strip()
    if hit is None or not value:
        return False
    names = [p.lower() for p in phrases]
    for line in str(hit.get("text") or "").splitlines():
        lower = line.lower()
        if value in line and any(re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", lower) for n in names):
            return True
    return False


def outlier_pct() -> float:
    try:
        return float(os.getenv("RAG_SECTOR_OUTLIER_PCT", "50"))
    except ValueError:
        return 50.0


# ----------------------------------------------------------------------------- aggregation

def to_millions(value: float | None, unit: str) -> float | None:
    scale = trend.scale_of(unit)
    return None if value is None or scale is None else value * _TO_MILLIONS[scale]


def _fmt(value: float) -> str:
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def aggregate(plan: dict, series: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Fixed panel, sector totals, changes and CAGR, all in code; ``series`` maps ticker -> trend.summarize()."""
    years = plan.get("years") or []
    panel, companies, calculations = [], [], []
    for ticker in plan.get("companies") or []:
        summary = series.get(ticker) or {"points": [], "changes": [], "cagr": None, "calculations": []}
        ok = {p["year"]: p for p in summary["points"] if p.get("status") == "ok"}
        scales = {trend.scale_of(p.get("unit") or "") for p in ok.values()}
        complete = all(y in ok for y in years) and len(scales) == 1
        in_total = complete and None not in scales
        if complete:
            panel.append(ticker)
            calculations.extend(summary.get("calculations") or [])
        companies.append({
            "ticker": ticker, "company": NAMES.get(ticker, ticker), "in_panel": complete, "in_total": in_total,
            "cagr": summary.get("cagr") if complete else None,
            "points": [{k: p.get(k) for k in ("year", "value", "value_text", "unit", "chunk_id", "page", "status",
                                              "method")} for p in summary["points"]],
            "changes": summary.get("changes") or [],
        })
    by_ticker = {c["ticker"]: c for c in companies}
    totals_from = [t for t in panel if by_ticker[t]["in_total"]]
    rows = []
    previous_total = None
    for year in years:
        values = []
        for ticker in totals_from:
            point = next(p for p in by_ticker[ticker]["points"] if p["year"] == year)
            values.append(to_millions(point["value"], point["unit"]))
        total = round(sum(values), 4) if values else None
        if total is not None and len(values) >= 2:
            calculations.append({"expression": " + ".join(_fmt(v) for v in values), "value": total, "grounded": True,
                                 "kind": "sector_total", "year": year, "unit": "millions"})
        total_pct = None
        if total is not None and previous_total:
            total_pct = round((total - previous_total) / abs(previous_total) * 100, 4)
            calculations.append({"expression": f"({_fmt(total)} - {_fmt(previous_total)}) / {_fmt(abs(previous_total))} * 100",
                                 "value": total_pct, "grounded": True, "kind": "sector_total_pct_change",
                                 "from": year - 1, "to": year})
        changes = [c["pct_change"] for t in panel for c in by_ticker[t]["changes"]
                   if c.get("to") == year and c.get("status") == "ok" and c.get("pct_change") is not None]
        median = round(statistics.median(changes), 4) if changes else None
        if median is not None and len(changes) >= 2:
            calculations.append({"expression": "median(" + ", ".join(_fmt(c) for c in changes) + ")", "value": median,
                                 "grounded": True, "kind": "sector_median_pct_change", "year": year})
        rows.append({"year": year, "total": total, "total_pct_change": total_pct, "median_company_pct_change": median,
                     "growing": sum(1 for c in changes if c > 0) if changes else None,
                     "companies": len(changes) if changes else len(panel)})
        previous_total = total if total is not None else previous_total
    cagr_total = None
    first, last = (rows[0], rows[-1]) if rows else ({}, {})
    if first.get("total") and last.get("total") and first["total"] > 0 and last["total"] > 0 and len(years) >= 2:
        n = years[-1] - years[0]
        if n >= 2:
            cagr_total = round(((last["total"] / first["total"]) ** (1 / n) - 1) * 100, 4)
            calculations.append({"expression": f"(({_fmt(last['total'])} / {_fmt(first['total'])}) ** (1 / {n}) - 1) * 100",
                                 "value": cagr_total, "grounded": True, "kind": "sector_total_cagr",
                                 "from": years[0], "to": years[-1]})
    cagrs = [by_ticker[t]["cagr"] for t in panel if by_ticker[t]["cagr"] is not None]
    median_cagr = round(statistics.median(cagrs), 4) if cagrs else None
    if median_cagr is not None and len(cagrs) >= 2:
        calculations.append({"expression": "median(" + ", ".join(_fmt(c) for c in cagrs) + ")", "value": median_cagr,
                             "grounded": True, "kind": "sector_median_cagr"})
    planned = plan.get("companies") or []
    missing = [f"{c['ticker']} FY{p['year']}" for c in companies for p in c["points"] if p.get("status") != "ok"]
    limit = outlier_pct()
    unusual = [{"ticker": t, "from": c["from"], "to": c["to"], "pct_change": c["pct_change"]}
               for t in panel for c in by_ticker[t]["changes"]
               if c.get("status") == "ok" and c.get("pct_change") is not None and abs(c["pct_change"]) > limit]
    return {
        "sector": plan.get("sector"),
        "metric": plan.get("metric"),
        "years": years,
        "unit": "millions",
        "panel": panel,
        "panel_names": [NAMES.get(t, t) for t in panel],
        "total_from": totals_from,
        "planned": planned,
        "sector_companies": len(plan.get("sector_companies") or []),
        "coverage_share": round(len(panel) / len(planned), 4) if planned else 0.0,
        "excluded": [*plan.get("excluded", []),
                     *[{"ticker": c["ticker"], "company": c["company"], "reason": "figure not verified for "
                        + ", ".join(f"FY{p['year']}" for p in c["points"] if p.get("status") != "ok")
                        or "unit scale differs between years"} for c in companies if not c["in_panel"]]],
        "rows": rows,
        "cagr_total": cagr_total,
        "median_company_cagr": median_cagr,
        "companies": companies,
        "missing": missing,
        "unusual_changes": unusual,            # company changes beyond ±RAG_SECTOR_OUTLIER_PCT: sent to human review
        "chart": [{"year": r["year"], "total": r["total"]} for r in rows if r["total"] is not None],
        "calculations": calculations,
    }


# ----------------------------------------------------------------------------- prompt text

def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.2f}%"


def table_text(sector_trend: dict) -> str:
    rows = ["Fiscal year | Sector total (USD millions) | Total change | Median company change | Companies growing"]
    for r in sector_trend.get("rows", []):
        total = "n/a" if r["total"] is None else f"{r['total']:,.2f}"
        growing = "n/a" if r["growing"] is None else f"{r['growing']} of {r['companies']}"
        rows.append(f"FY{r['year']} | {total} | {_pct(r['total_pct_change'])} | "
                    f"{_pct(r['median_company_pct_change'])} | {growing}")
    if sector_trend.get("cagr_total") is not None:
        rows.append(f"CAGR of the sector total FY{sector_trend['years'][0]}–FY{sector_trend['years'][-1]}: "
                    f"{sector_trend['cagr_total']:.2f}%")
    if sector_trend.get("median_company_cagr") is not None:
        rows.append(f"Median company CAGR: {sector_trend['median_company_cagr']:.2f}%")
    return "\n".join(rows)


def companies_text(sector_trend: dict) -> str:
    rows = []
    for company in sector_trend.get("companies", []):
        if not company["in_panel"]:
            continue
        figures = "; ".join(f"FY{p['year']}: {p['value_text']} {p['unit']} [{p['chunk_id']}]".replace("  ", " ")
                            for p in company["points"] if p.get("status") == "ok")
        rows.append(f"{company['company']} ({company['ticker']}): {figures}")
    return "\n".join(rows) or "none"


def excluded_text(sector_trend: dict) -> str:
    return "\n".join(f"{e['company']} ({e['ticker']}): {e['reason']}" for e in sector_trend.get("excluded", [])) or "none"
