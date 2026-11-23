"""Shareholder-returns agent: which company returned the most cash to its shareholders in a fiscal year.

"Which bank returned the most to shareholders in 2024?" or "Compare what Apple, Microsoft and Alphabet returned to
shareholders in fiscal 2024" (technique ``shareholder_returns_graph`` or the router's ``shareholder_returns`` route):

1. ``returns_plan``: the companies (the ones named, or the indexed companies of the sector named or filtered on), one
   fiscal year (the one asked, else the year most of those companies have a filing for) and two measures per
   company: cash dividends paid and common stock repurchased, both from the statement of cash flows. Companies
   without a filing for that year are listed with the reason, never guessed.
2. One filtered search per company x measure, in parallel.
3. Per company-measure, code first: ``label_match`` reads the cash-flow row ("Dividends paid  (4,100)  (3,900)") in
   the column of that fiscal year. Without one, a broadened, deeper search and the row again; only then does the LLM
   copy the figure, kept only if its chunk prints it on a line naming the measure. Chunks the injection guard flags
   are dropped.
4. ``rank``: per company dividends + buybacks in USD millions, the ranking and each company's share of the total,
   all computed in code and recorded as calculations. A company is ranked only when both figures are verified.
5. ``answer_text`` writes the answer from that table; the critic checks every figure in code; governance sends a
   ranking covering under half the companies read to human review.

The answer is what the filings report as cash paid in the fiscal year. It is not a recommendation, and the input
guard still refuses advice ("which should I buy?") before this agent runs.
"""
from __future__ import annotations

import os
import re
from typing import Any

from scripts.agents import financial_sector_trend as sector_agent
from scripts.agents import financial_trend as trend
from scripts.common.companies import NAMES, normalize_ticker

RETURNS_TOOLS = ("coverage_lookup", "text_search", "cash_flow_extract", "returns_calculator")
DEFAULT_MAX_COMPANIES = 10
MEASURES = ("dividends paid", "share repurchases")

# Statement-of-cash-flows rows, longest first; collected from the 2024 filings of 40 companies across the sectors.
CASH_FLOW_LABELS: dict[str, tuple[str, ...]] = {
    "dividends paid": (
        "common stock cash dividends paid", "cash dividends paid to shareholders", "dividends paid to shareholders",
        "dividends paid to common shareholders", "cash dividends paid on common stock", "common cash dividends paid",
        "payments for cash dividends", "dividends paid on common stock", "cash dividends paid",
        "dividends on common stock", "common stock dividends paid", "common stock dividends", "payments of dividends",
        "dividends paid to stockholders", "dividends to shareholders", "dividends to stockholders",
        "cash dividends - common stock", "dividends paid", "dividend payments", "cash dividends", "dividends"),
    "share repurchases": (
        "payments for repurchases of common stock", "repurchases of common stock", "repurchase of common stock",
        "repurchases of common shares", "common stock repurchased", "common stock repurchases",
        "common share repurchases", "purchases of common stock", "purchases of treasury stock",
        "purchases of treasury shares", "purchases of stock for treasury", "treasury stock repurchased",
        "treasury stock purchased", "treasury stock purchases", "treasury stock acquired", "repurchases of stock",
        "repurchase of company common stock", "repurchases of company common stock",
        "net sales (purchases) of treasury shares",              # Chevron prints buybacks net of shares issued
        "repurchase of stock", "repurchases of shares", "repurchase of shares", "share repurchases",
        "repurchased"),          # Wells Fargo: "Repurchased" under "Common stock:" in its financing activities
}
# Words the LLM-copied figure's line must contain (``sector_agent.on_metric_line``).
PHRASES: dict[str, list[str]] = {
    "dividends paid": ["dividends paid", "dividends", "dividend"],
    "share repurchases": ["repurchased", "repurchases", "repurchase", "treasury", "buyback", "buybacks"],
}
SEARCH_WORDS = {"dividends paid": "statement of cash flows cash dividends paid to shareholders",
                "share repurchases": "statement of cash flows repurchases of common stock treasury stock purchased"}

RETURNS_SIGNAL = re.compile(
    r"\b(return(?:s|ed|ing)?|distribut(?:e|es|ed|ing|ion|ions)|gave back|paid out)\b[^.?!]{0,40}?"
    r"\b(?:to|for)\s+(?:its\s+|their\s+)?(?:share|stock)holders\b"
    r"|\b(?:shareholder|stockholder|capital)\s+(?:returns?|distributions?|payouts?)\b"
    r"|\bdividends?\s+(?:and|plus|\+|&)\s+(?:share\s+)?(?:buybacks?|repurchases?)\b"
    r"|\b(?:buybacks?|repurchases?)\s+(?:and|plus|\+|&)\s+dividends?\b", re.I)


# "Which bank / insurer / oil company returned the most?": the singular names a sector here (the shared aliases stay
# plural, so "Bank of America" is never read as the banking sector).
SINGULAR_SECTORS = (
    (re.compile(r"\bbank\b(?!\s+of\b)", re.I), "Banking"),
    (re.compile(r"\binsurer\b|\binsurance compan(?:y|ies)\b", re.I), "Insurance"),
    (re.compile(r"\boil(?: and gas| & gas)? (?:company|producer|major)\b", re.I), "Oil and gas"),
    (re.compile(r"\butility\b|\butilities compan(?:y|ies)\b", re.I), "Energy and utilities"),
    (re.compile(r"\bretailer\b", re.I), "Retail"),
    (re.compile(r"\b(?:pharma(?:ceutical)?|healthcare|health care) company\b", re.I), "Healthcare"),
    (re.compile(r"\btech(?:nology)? company\b", re.I), "Technology and IT"),
    (re.compile(r"\btelecom(?:munications)? company\b", re.I), "Telecom and media"),
    (re.compile(r"\breit\b|\breal estate company\b", re.I), "Real estate"),
)


def sectors_named(query: str) -> list[str]:
    named = sector_agent.sectors_in(query)
    if named:
        return named
    return list(dict.fromkeys(sector for pattern, sector in SINGULAR_SECTORS if pattern.search(query or "")))


# Total shareholder return is the stock's price change plus dividends (a performance graph), not cash paid out.
STOCK_RETURN = re.compile(r"\btotal\s+(?:shareholder\s+|stockholder\s+)?returns?\b|\bTSR\b|\bstock(?:'s)?\s+(?:price\s+)?"
                          r"(?:return|performance)\b|\breturns?\s+on\s+(?:the\s+)?(?:stock|shares|investment)\b", re.I)


def returns_signal(query: str) -> bool:
    """The question asks what was returned to shareholders (dividends plus buybacks), not the stock's total return."""
    return bool(RETURNS_SIGNAL.search(query or "")) and not STOCK_RETURN.search(query or "")


def max_companies() -> int:
    try:
        return max(1, int(os.getenv("RAG_RETURNS_MAX_COMPANIES", DEFAULT_MAX_COMPANIES)))
    except ValueError:
        return DEFAULT_MAX_COMPANIES


# ----------------------------------------------------------------------------- plan

def returns_plan(query: str, clarifier, *, ticker: str | None = None, sector_filter: str | None = None,
                 year: int | None = None, limit: int | None = None) -> dict[str, Any]:
    """Companies, one fiscal year and one search target per company x measure."""
    limit = limit or max_companies()
    directory = clarifier.directory or {}
    named = [normalize_ticker(t) for t in clarifier.companies_in(query)]
    if not named and ticker:
        named = [normalize_ticker(ticker)]
    sectors = [] if named else (sectors_named(query) or sector_agent.sector_from_filter(sector_filter))
    sector = sectors[0] if len(sectors) == 1 else None
    problems: list[str] = []
    if named:
        candidates, scope = named, "companies"
    elif sector:
        candidates, scope = sector_agent.members(sector, directory), "sector"
    else:
        candidates, scope = [], None
        problems.append(f"one sector at a time ({', '.join(sectors)} were named)" if len(sectors) > 1
                        else "no company or sector named")
    unknown = [t for t in candidates if t not in directory]
    covered = [t for t in candidates if t in directory]
    asked = clarifier.years_in(query) or ([int(year)] if year else [])
    adjustments = []
    if len(asked) > 1:
        adjustments.append(f"one fiscal year at a time: FY{asked[-1]} used")
    if asked:
        fiscal_year = asked[-1]
    else:
        counts: dict[int, int] = {}
        for t in covered:
            for y in directory.get(t, []):
                counts[y] = counts.get(y, 0) + 1
        fiscal_year = max(counts, key=lambda y: (counts[y], y)) if counts else None
    eligible = [t for t in covered if fiscal_year in directory.get(t, [])]
    chosen = eligible[:limit]
    excluded = [{"ticker": t, "company": NAMES.get(t, t), "reason": "no filings indexed"} for t in unknown]
    excluded += [{"ticker": t, "company": NAMES.get(t, t), "reason": f"no FY{fiscal_year} filing indexed"}
                 for t in covered if t not in eligible]
    excluded += [{"ticker": t, "company": NAMES.get(t, t), "reason": f"over the {limit}-company limit"}
                 for t in eligible[limit:]]
    if scope and not chosen:
        problems.append(f"no {'named company' if scope == 'companies' else sector + ' company'} has a "
                        f"FY{fiscal_year} filing indexed" if fiscal_year else "none of them is in the index")
    return {
        "intent": "shareholder_returns",
        "scope": scope,
        "sector": sector,
        "sectors_named": sectors,
        "year": fiscal_year,
        "measures": list(MEASURES),
        "companies": chosen,
        "excluded": excluded,
        "tools": list(RETURNS_TOOLS),
        "targets": [{"ticker": t, "year": fiscal_year, "measure": m} for t in chosen for m in MEASURES],
        "feasible": not problems,
        "problems": problems,
        "planner": "rules",
        "adjustments": adjustments,
    }


def not_available_message(plan: dict) -> str:
    return "I can't compare shareholder returns: " + "; ".join(plan.get("problems") or ["no plan"]) + "."


def target_label(target: dict) -> str:
    return f"{target['ticker']} FY{target['year']} {target['measure']}"


def search_query(ticker: str, year: int, measure: str, broadened: bool = False) -> str:
    query = f"{NAMES.get(ticker, ticker)} {SEARCH_WORDS[measure]} fiscal year {year}"
    if broadened:
        query += " " + " ".join(PHRASES[measure][:4]) + " financing activities"
    return query


CASH_AMOUNT = re.compile(r"^\(?-?\$?\s?\d{1,3}(?:,\d{3})*\)?$")      # whole USD amounts: "(13,104)", "553"
CASH_FLOW_CONTEXT = re.compile(r"financing activities|cash flows?|net cash (?:used|provided)", re.I)
# Rows that share the words but are not cash paid to common holders this year.
EXCLUDED_ROWS = re.compile(r"preferred|per (?:common )?share|noncontrolling|minority|authori[sz]|remain", re.I)
_FIGURE_LINE = re.compile(r"^\$?\s*\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?\s*\$?$")
_YEAR_LINE = re.compile(r"^(?:(?:fiscal\s+)?20[0-4]\d\s*)+$", re.I)


def join_vertical(text: str) -> str:
    """Statement rows laid out one value per line (OCR and some text layers: "Repurchases of stock / (59,296) /
    (61,504) / (62,222)", years "2022 / 2023 / 2024") joined back into one line each, so the row reader sees them."""
    out: list[str] = []
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if out and stripped and _FIGURE_LINE.match(stripped) and not _YEAR_LINE.match(stripped) and out[-1].strip():
            out[-1] = f"{out[-1].rstrip()} {stripped}"
        elif out and stripped and _YEAR_LINE.match(stripped) and _YEAR_LINE.match(out[-1].strip() or "x"):
            out[-1] = f"{out[-1].rstrip()} {stripped}"
        elif stripped == "$" and out:
            continue
        else:
            out.append(line)
    return "\n".join(out)


_ZERO_CELL = re.compile(r"(?<=\s)(?:0|—|–)(?=\s|$)")


def zero_cells(text: str) -> str:
    """A statement row's "0" or "—" cell read as 0.0, so "Dividend payments 0 0 (7,363)" keeps its three columns."""
    return "\n".join(_ZERO_CELL.sub("0.0", line) if sector_agent._ROW_FIGURE.search(line) else line
                     for line in text.splitlines())


def _header_line(text: str) -> str | None:
    for line in text.splitlines():
        lower, _ = sector_agent._row_cells(line)
        years = sector_agent._YEAR.findall(lower)
        if len(set(years)) >= 2 and not sector_agent._ROW_FIGURE.search(sector_agent._YEAR.sub(" ", lower)):
            return line
    return None


def readable_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hits with vertical rows joined, and a chunk without a year header given its page's header (a statement split
    across chunks keeps "2024 2023 2022" in the first one only). The text changes only for the row reader."""
    joined = [{**hit, "text": zero_cells(join_vertical(hit.get("text") or ""))} for hit in hits]
    headers: dict[tuple, str] = {}
    for hit in joined:
        line = _header_line(hit["text"])
        if line:
            headers.setdefault((hit.get("doc_id"), hit.get("page")), line)
    out = []
    for hit in joined:
        header = headers.get((hit.get("doc_id"), hit.get("page")))
        if header and not _header_line(hit["text"]):
            hit = {**hit, "text": f"{header}\n{hit['text']}"}
        if not sector_agent.scale_of_text(hit["text"]):
            # The page's scale: printed on another chunk of the page, or recorded at ingest ("In millions").
            page_text = " ".join(h["text"] for h in joined
                                 if (h.get("doc_id"), h.get("page")) == (hit.get("doc_id"), hit.get("page")))
            scale = sector_agent.scale_of_text(page_text) or sector_agent.scale_of_text(f"({hit.get('scale') or ''})")
            if scale:
                hit = {**hit, "text": f"(in {scale})\n{hit['text']}"}
        out.append(hit)
    return out


def row_line(hit: dict[str, Any], value_text: str) -> str | None:
    """The joined line of the chunk that prints the figure, or None."""
    value = str(value_text or "").replace("$", "").strip()
    for line in join_vertical(hit.get("text") or "").splitlines():
        if value and value in line:
            return line
    return None


def cash_figure_ok(point: dict[str, Any], hits: list[dict[str, Any]], measure: str) -> bool:
    """A figure counts as the measure's cash paid only if its chunk is a cash-flow statement (or says it is about
    financing cash flows), its line names the measure and is not a preferred, per-share or authorization row, and
    it is a whole amount (a share count such as "91.7" or a per-share "1.94" is not)."""
    hit = next((h for h in hits if str(h.get("chunk_id")) == str(point.get("chunk_id"))), None)
    value = str(point.get("value_text") or "").replace("$", "").strip()
    if hit is None or not value or not CASH_FLOW_CONTEXT.search(hit.get("text") or ""):
        return False
    if not CASH_AMOUNT.match(value) and "billion" not in str(point.get("unit") or "").lower():
        return False
    line = (row_line(hit, value) or "").lower()
    return bool(line) and not EXCLUDED_ROWS.search(line) and any(
        re.search(rf"(?<![a-z]){re.escape(p)}(?![a-z])", line) for p in PHRASES[measure])


_CELL = re.compile(r"^\(?-?\d[\d,]*(?:\.\d+)?\)?$")
_DASHES = {"—", "–", "-", "--"}
_NORMAL_LABELS = {measure: {" ".join(re.findall(r"[a-z]+", label)) for label in labels}
                  for measure, labels in CASH_FLOW_LABELS.items()}


def split_row(line: str) -> tuple[str, list[str]] | None:
    """('repurchase of common stock', ['(2,432)', '(5,054)', '(6,035)']) for a statement row: the row name as plain
    words (a damaged text layer's "Repurchase) of common stock$" reads the same) and its cells, every token after
    the name being a figure, a dash (zero) or "$". None when the line is not a row of figures."""
    text = sector_agent._DOT_LEADER.sub(" ", line.replace("|", " "))
    text = re.sub(r"(?<=[A-Za-z])\s+[-–—]\s+(?=[A-Za-z])", " ", text)          # "Cash dividends - common stock"
    tokens = text.replace("$", " ").split()
    start = next((i for i, token in enumerate(tokens) if _CELL.match(token) or token in _DASHES), None)
    if not start:
        return None
    cells = tokens[start:]
    if not all(_CELL.match(token) or token in _DASHES for token in cells):
        return None
    name = " ".join(re.findall(r"[a-z]+", " ".join(tokens[:start]).lower()))
    return name, ["0" if cell in _DASHES else cell for cell in cells]


def scale_of(text: str) -> str:
    """'millions' from "(in millions)", "(MILLIONS OF DOLLARS)", "(Dollars in Millions)" or "($ millions)"; else ''."""
    return sector_agent.scale_of_text(text)


def _rows(hit: dict[str, Any], measure: str, year: int) -> list[dict[str, Any]]:
    """Every row of the chunk named like the measure whose cells line up with the nearest year header above it
    (one cell per year), read in the column of ``year``. Unlike ``sector_agent.label_match`` a small amount such as
    "(173)" is a cell, because every token after the row name must be a figure."""
    scale = scale_of(hit["text"])
    header: list[int] = []
    rows = []
    segment, last_segment = False, -sector_agent.SEGMENT_LINES - 1
    for index, line in enumerate(hit["text"].splitlines()):
        lower = line.lower()
        if sector_agent.SEGMENT_TABLE.search(lower) and not sector_agent._ROW_FIGURE.search(lower):
            last_segment = index
        years = list(dict.fromkeys(int(y) for y in sector_agent._YEAR.findall(lower)))
        if len(years) >= 2 and not sector_agent._ROW_FIGURE.search(sector_agent._YEAR.sub(" ", lower)):
            header = years
            segment = index - last_segment <= sector_agent.SEGMENT_LINES       # a segment's table, not the company's
            continue
        split = split_row(line)
        if not split or not header or year not in header or segment:
            continue
        name, cells = split
        words = name.split()
        # The row is the measure's when its name is a label, give or take a footnote letter ("dividends paid (a)").
        if not any(name == label or (name.startswith(label + " ") and all(len(w) == 1 for w in words[len(label.split()):]))
                   for label in _NORMAL_LABELS[measure]):
            continue
        if len(cells) != len(header):
            continue
        rows.append({"value": cells[header.index(year)], "unit": scale,
                     "chunk_id": hit.get("chunk_id"), "quote": " ".join(line.split())[:160]})
    return rows


def read_row(hits: list[dict[str, Any]], measure: str, year: int) -> dict[str, Any] | None:
    """The cash-flow row of ``measure`` for ``year``, read in code (None when no row and year header line up). Only
    a row that passes ``cash_figure_ok`` counts: a "Treasury stock repurchased (91.7)" share-count row, an equity
    statement's "dividends" or a preferred-dividends row is skipped. A statement listing the same row twice (Wells
    Fargo: "Cash dividends paid" under preferred stock, then under common stock) gives the larger amount: the common
    stock's."""
    readable = readable_hits(hits)
    # Consolidated statements first; a parent-company-only cash flow statement (same dividends, usually) last.
    ordered = ([h for h in readable if not sector_agent.parent_only(h["text"])]
               + [h for h in readable if sector_agent.parent_only(h["text"])])
    for hit in ordered:
        rows = [row for row in _rows(hit, measure, year)
                if cash_figure_ok({"chunk_id": row["chunk_id"], "value_text": row["value"], "unit": row["unit"]},
                                  [hit], measure)]
        if rows:
            return max(rows, key=lambda row: abs(trend.parse_figure(row["value"]) or 0))
    return None


# ----------------------------------------------------------------------------- ranking, in code

def _fmt(value: float) -> str:
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def rank(plan: dict, points: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """Per company dividends + buybacks in USD millions, the ranking and the shares of the total. ``points`` maps
    (ticker, measure) -> a verified point (``trend.validate_point``) with its method."""
    year = plan.get("year")
    companies, calculations = [], []
    for ticker in plan.get("companies") or []:
        figures, reasons = {}, []
        for measure in MEASURES:
            point = points.get((ticker, measure)) or {"status": "not_found"}
            millions = sector_agent.to_millions(abs(point["value"]), point.get("unit") or "") \
                if point.get("status") == "ok" else None
            if point.get("status") != "ok":
                reasons.append(f"{measure} not verified")
            elif millions is None:
                reasons.append(f"{measure} unit scale not stated")
            figures[measure] = {k: point.get(k) for k in ("value", "value_text", "unit", "chunk_id", "page", "status",
                                                           "method")} | {"millions": millions}
        dividends = figures["dividends paid"]["millions"]
        buybacks = figures["share repurchases"]["millions"]
        total = round(dividends + buybacks, 4) if not reasons else None
        if total is not None:
            calculations.append({"expression": f"{_fmt(dividends)} + {_fmt(buybacks)}", "value": total,
                                 "grounded": True, "kind": "shareholder_return", "ticker": ticker, "year": year,
                                 "unit": "millions"})
            for measure in MEASURES:                       # a figure reported in thousands, shown in millions
                if figures[measure]["millions"] != abs(figures[measure]["value"]):
                    calculations.append({"expression": f"{_fmt(abs(figures[measure]['value']))} / 1000",
                                         "value": figures[measure]["millions"], "grounded": True,
                                         "kind": "to_millions", "ticker": ticker})
        companies.append({"ticker": ticker, "company": NAMES.get(ticker, ticker), "year": year,
                          "dividends": figures["dividends paid"], "buybacks": figures["share repurchases"],
                          "total": total, "ranked": total is not None, "reason": "; ".join(reasons) or None})
    ranked = sorted((c for c in companies if c["ranked"]), key=lambda c: c["total"], reverse=True)
    panel_total = round(sum(c["total"] for c in ranked), 4) if ranked else None
    if len(ranked) >= 2:
        calculations.append({"expression": " + ".join(_fmt(c["total"]) for c in ranked), "value": panel_total,
                             "grounded": True, "kind": "returns_total", "year": year, "unit": "millions"})
    for position, company in enumerate(ranked, 1):
        company["rank"] = position
        if panel_total and len(ranked) >= 2:
            share = round(company["total"] / panel_total * 100, 4)
            company["share_pct"] = share
            calculations.append({"expression": f"{_fmt(company['total'])} / {_fmt(panel_total)} * 100",
                                 "value": share, "grounded": True, "kind": "returns_share", "ticker": company["ticker"]})
        dividends = company["dividends"]["millions"]
        if company["total"]:
            company["dividend_pct"] = round(dividends / company["total"] * 100, 4)
            calculations.append({"expression": f"{_fmt(dividends)} / {_fmt(company['total'])} * 100",
                                 "value": company["dividend_pct"], "grounded": True, "kind": "dividend_share",
                                 "ticker": company["ticker"]})
    planned = plan.get("companies") or []
    return {
        "scope": plan.get("scope"),
        "sector": plan.get("sector"),
        "year": year,
        "unit": "millions",
        "planned": planned,
        "ranking": [c["ticker"] for c in ranked],
        "total": panel_total,
        "coverage_share": round(len(ranked) / len(planned), 4) if planned else 0.0,
        "companies": companies,
        "excluded": [*plan.get("excluded", []),
                     *[{"ticker": c["ticker"], "company": c["company"], "reason": c["reason"]}
                       for c in companies if not c["ranked"]]],
        "chart": [{"ticker": c["ticker"], "dividends": c["dividends"]["millions"],
                   "buybacks": c["buybacks"]["millions"], "total": c["total"]} for c in ranked],
        "calculations": calculations,
    }


# ----------------------------------------------------------------------------- the answer, written in code

def _money(value: float) -> str:
    return f"${value:,.0f} million" if float(value).is_integer() else f"${value:,.1f} million"


def _figure(point: dict) -> str:
    unit = f" {point['unit']}" if point.get("unit") else ""
    value = str(point.get("value_text") or "").strip().strip("()").replace("$", "").strip()   # "(13,104)": cash out
    return f"{value}{unit} [{point['chunk_id']}]"


def answer_text(table: dict) -> str:
    """The answer, written from the ranking: every figure is a cash-flow figure cited by its chunk, or a total or
    share the returns calculator computed. No LLM writes it; the critic checks its figures in code."""
    year = table.get("year")
    ranked = [c for c in table.get("companies", []) if c.get("ranked")]
    ranked.sort(key=lambda c: c["rank"])
    where = f"the {table['sector']} companies read" if table.get("scope") == "sector" else "the companies asked about"
    lines = []
    if ranked:
        top = ranked[0]
        if len(ranked) >= 2:
            lines.append(f"Of {where} for FY{year}, {top['company']} ({top['ticker']}) returned the most cash to "
                         f"shareholders: {_money(top['total'])} in dividends and share repurchases"
                         + (f", {top['share_pct']:.1f}% of the {_money(table['total'])} the {len(ranked)} ranked "
                            f"companies returned together." if top.get("share_pct") is not None else "."))
        else:
            lines.append(f"In FY{year}, {top['company']} ({top['ticker']}) returned {_money(top['total'])} to "
                         f"shareholders in dividends and share repurchases.")
        lines += ["", "Ranking (cash paid in the fiscal year, from each statement of cash flows):"]
        for company in ranked:
            line = (f"{company['rank']}. {company['company']} ({company['ticker']}): {_money(company['total'])} = "
                    f"dividends {_figure(company['dividends'])} + repurchases {_figure(company['buybacks'])}")
            if company.get("dividend_pct") is not None:
                line += f"; dividends were {company['dividend_pct']:.1f}% of it"
            lines.append(line + ".")
    excluded = table.get("excluded") or []
    if excluded:
        lines += ["", "Not ranked: " + "; ".join(f"{e['company']} ({e['ticker']}): {e['reason']}"
                                                 for e in excluded) + "."]
    from scripts.common.companies import BANKING, SECTOR_OF
    banks = any(SECTOR_OF.get(c["ticker"]) == BANKING for c in ranked)
    lines += ["", "These are reported cash payments, not a recommendation"
              + ("; banks' dividends paid can include preferred dividends." if banks else ".")]
    return "\n".join(lines)
