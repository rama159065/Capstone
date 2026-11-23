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
    "revenue": ("total net revenues", "total net revenue", "total operating revenues", "total net operating revenues",
                "total revenues", "total revenue", "total revenues and other income", "net operating revenues", "net revenues", "net revenue",
                "total net sales", "net sales", "operating revenues", "revenues", "revenue"),
    "net income": ("net income attributable to common shareholders", "net income attributable to common stockholders",
                   "net income", "net earnings"),
    "net interest income": ("net interest income",),
    "operating income": ("operating income", "operating income (loss)", "operating earnings", "operating profit",
                         "income from operations", "earnings from operations", "income (loss) from operations"),
    "gross profit": ("gross profit",),
    "total assets": ("total assets",),
    "total liabilities": ("total liabilities",),
    "shareholders' equity": ("total stockholders' equity", "total shareholders' equity", "total shareowners' equity",
                             "total company stockholders' equity", "total company shareholders' equity"),
    "operating cash flow": ("net cash provided by operating activities", "net cash provided by (used in) operating activities",
                            "net cash provided by (used for) operating activities", "cash generated by operating activities",
                            "net cash flows provided by operating activities", "net cash from operating activities",
                            "net cash flows from operating activities"),
}
# Margins are two rows of one statement; the division is done in code.
MARGINS = {"operating margin": "operating income", "net margin": "net income", "gross margin": "gross profit"}
# Rows naming the measure with a qualifier the plain labels don't cover, seen in the banks' statements on the
# 509-filing index: "Net income (loss)" (Truist), "Net income attributable to U.S. Bancorp", "Net income applicable to
# shareholders of The Bank of New York Mellon Corporation". The figures must follow directly. Left out on purpose:
# rows for common shareholders (after preferred dividends) and "Net income before (attribution of) noncontrolling
# interests" (includes the minority share). The other banks report net income attributable to the company, and Citi
# prints all three, so only its "Net income attributable to Citigroup" row is read.
QUALIFIED_LABELS: dict[str, re.Pattern] = {
    "net income": re.compile(
        r"^(?:[a-z.&]+\W{1,3}s )?net (?:income|earnings)(?: \(loss\))?"                # "Citigroup's net income"
        r"(?: (?:attributable|applicable) to (?!noncontrolling)(?![a-z.,&'’ ]*\bcommon\b)[a-z.,&'’ ]+?)?"
        r"(?=\s*[\$(:\-]?\s*\(?\d)"),
}
_CELL = re.compile(r"^\(?-?\d[\d,]*(?:\.\d+)?\)?$")
_PER_SHARE = re.compile(r"^\(?-?\d{1,3}\.\d{2}\)?$")
_YEAR_CELL = re.compile(r"^(?P<label>[^|]+?)\s*\|\s*(?P<year>20[0-4]\d)\s*\|\s*\$?\s*(?P<value>\(?[\d,]+(?:\.\d+)?\)?)$")
_DOT_LEADER = re.compile(r"(?:\s*\.){3,}")     # "Net income . . . . . . 2,741" (M&T)
_ROW_FIGURE = re.compile(r"\(?-?\$?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\(?-?\$?\s?\d+\.\d+\)?")
_YEAR = re.compile(r"(?<!\d)(20[0-4]\d)(?!\d)")
_SCALE = re.compile(r"\bin\s+(thousands|millions|billions)\b", re.I)     # "(in\nmillions)" wraps across lines
_SCALE_WORDS = re.compile(
    r"\b(thousands|millions|billions)\s+of\s+(?:u\.s\.\s+)?dollars\b|\(\s*(?:\$\s*)?(thousands|millions|billions)\s*[,)]"
    r"|\bdollars\s+in\s+(thousands|millions|billions)\b", re.I)
_PRO_FORMA = re.compile(r"\bpro\s+forma\b", re.I)
PRO_FORMA_LINES = 8          # a "pro forma" mention this close above a year header makes its table pro forma
# Tables that print the statement's row names but are not the consolidated statement (live check, Nov 2026):
# Bank of America's "Total assets 559,733" is Note 24's Parent Company-only balance sheet (consolidated 3,261,519);
# Walmart's "Operating income 25,158" is the Walmart U.S. segment table (consolidated 29,825).
PARENT_ONLY = re.compile(r"\bparent\s+company(?:[- ]only)?\s+(?:information|financial|condensed|balance|statements?)\b"
                         r"|\bparent[- ]company[- ]only\b|\bparent[- ]only\b|\bcondensed\s+(?:consolidating\s+)?"
                         r"(?:statements?|balance\s+sheets?)\b|\bregistrant\b", re.I)
# Capital One's "Table 8.1: Domestic Card Business Results" prints "Net interest income 20,881" (company: 31,208).
SEGMENT_TABLE = re.compile(r"\bsegments?\b|\bbusiness\s+results\b", re.I)
SEGMENT_LINES = 3            # a line naming a segment this close above a year header makes its table a segment's
CHANGE_COLUMN = re.compile(r"%\s*change|\bchange\b|\bvariance\b", re.I)


_TITLE_LINE = re.compile(r"\bconsolidated\s+(?:statements?\s+of|balance\s+sheets?)", re.I)
_NOT_A_TITLE = re.compile(r"\b(?:see|related|accompanying|notes?\s+to|in\s+the|our|on\s+the|of\s+the|these|report|audited?)\b|\.{3}",
                          re.I)
_BALANCE_TITLE = re.compile(r"balance\s+sheets?|financial\s+(?:position|condition)", re.I)
_CASH_TITLE = re.compile(r"statements?\s+of\s+(?:consolidated\s+)?cash\s+flows?", re.I)
_INCOME_TITLE = re.compile(r"statements?\s+of\s+(?:consolidated\s+)?(?:net\s+)?(?:comprehensive\s+)?(?:income|operations|earnings)",
                           re.I)


def statement_title(text: str) -> str | None:
    """The line that titles a consolidated statement ("CONSOLIDATED BALANCE SHEETS"), not a sentence, an auditor's
    report or a note that mentions one; None when the chunk has none."""
    for line in str(text or "").splitlines():
        line = line.split("|")[0].strip(" #*\t")
        if len(line) <= 110 and len(line.split()) <= 14 and _TITLE_LINE.search(line) and not _NOT_A_TITLE.search(line):
            return line
    return None


# A utility's filing holds its subsidiaries' consolidated statements too ("DUKE ENERGY OHIO, INC. CONSOLIDATED BALANCE
# SHEETS"): same titles and rows, another company.
_STATE_WORDS = re.compile(
    r"\b(alabama|arizona|arkansas|california|carolinas?|colorado|connecticut|florida|georgia|illinois|indiana|iowa|kansas|"
    r"kentucky|louisiana|maryland|michigan|minnesota|mississippi|missouri|nevada|new\s+(?:york|jersey|mexico)|ohio|oklahoma|"
    r"oregon|pennsylvania|piedmont|progress|tennessee|texas|utah|virginia|wisconsin)\b", re.I)
_ENTITY = re.compile(r"\b(?:inc|llc|l\.l\.c|corp(?:oration)?|company|co|l\.?p)\b\.?", re.I)


def subsidiary_statement(hit: dict[str, Any]) -> bool:
    """The chunk opens with another registrant of the same filing (Duke Energy Ohio, Inc. in Duke Energy's report)."""
    first = NAMES.get(str(hit.get("ticker") or "").upper(), "").split()
    if not first:
        return False
    for line in str(hit.get("text") or "").splitlines()[:6]:
        if (line.lower().lstrip(" #*").startswith(first[0].lower()) and len(line.split()) <= 8 and _ENTITY.search(line)
                and _STATE_WORDS.search(line) and not _STATE_WORDS.search(NAMES.get(str(hit.get("ticker")).upper(), ""))):
            return True
    return False


# What else the measure's statement prints: a table with the row and none of these is a note, not the statement
# (Welltower's "Total assets 1,524,609" is what an acquisition brought).
_COMPANIONS = {"total assets": re.compile(r"total\s+(?:current\s+assets|liabilities)|goodwill|stockholders|shareholders|receivable", re.I),
               "revenue": re.compile(r"cost\s+of|operating\s+(?:income|expenses|profit)|income\s+before|earnings\s+before|net\s+income|"
                                     r"net\s+earnings|expenses?\b|selling", re.I)}


def statement_pages(hits: list[dict[str, Any]], metric: str | None, limit: int = 2) -> list[tuple[str, int]]:
    """(doc_id, page) of the hits titled as the statement that prints the measure, best-ranked first."""
    wanted = (_BALANCE_TITLE if metric in BALANCE_METRICS else _CASH_TITLE if metric in CASH_METRICS else _INCOME_TITLE)
    pages: list[tuple[str, int]] = []
    for hit in hits:
        title, doc_id = statement_title(hit.get("text")), (hit.get("metadata") or {}).get("doc_id")
        if (title and wanted.search(title) and doc_id and hit.get("page") is not None
                and not parent_only(hit.get("text")) and not subsidiary_statement(hit)):
            key = (str(doc_id), int(hit["page"]))
            # The statement of income is the page before a separate statement of comprehensive income.
            before = ([(key[0], key[1] - 1)] if re.search(r"comprehensive", title, re.I)
                      and not re.search(r"(?:income|operations|earnings)\s+and\s+(?:other\s+)?comprehensive", title, re.I)
                      and key[1] > 1 and wanted is _INCOME_TITLE else [])
            pages += [k for k in (key, *before) if k not in pages]
    return pages[:limit + 1]


def parent_only(text: str) -> bool:
    """The chunk (or its page context) is a parent-company-only statement."""
    return bool(PARENT_ONLY.search(text or ""))
# The statement that prints each measure: named in the search, because the statement's own title ranks its page first.
# "Coca-Cola revenue fiscal year 2023" found the statement of comprehensive income, not the statement of income.
STATEMENT_NAMES = {"revenue": "consolidated statements of income operations",
                   "net income": "consolidated statements of income",
                   "net interest income": "consolidated statements of income total interest income expense",
                   "operating income": "consolidated statements of income operations",
                   "gross profit": "consolidated statements of income operations",
                   "total assets": "consolidated balance sheets statements of financial position",
                   "total liabilities": "consolidated balance sheets statements of financial position",
                   "shareholders' equity": "consolidated balance sheets total liabilities and equity",
                   "operating cash flow": "consolidated statements of cash flows net cash provided by operating activities"}
# The retry's search: the rows such a statement prints, not the company's words ("PepsiCo revenue" never found the page
# its own filter already narrows to). On the 45 figures the first two searches missed, it found 9 more statements.
STATEMENT_ROWS = {
    "income": "consolidated statements of income years ended in millions except per share revenue cost of sales operating "
              "income income before taxes income taxes net income earnings per share",
    "balance": "consolidated balance sheets in millions cash and cash equivalents total current assets goodwill total "
               "assets accounts payable total liabilities shareholders equity",
    "cash": "consolidated statements of cash flows net cash provided by operating activities financing activities dividends "
            "paid repurchases of common stock treasury stock net cash used in financing activities"}
BALANCE_METRICS = ("total assets", "total liabilities", "shareholders' equity")
CASH_METRICS = ("dividends paid", "share repurchases", "operating cash flow")
PRIOR_CONFLICT = 0.15        # a filing's prior-year column this far from that year's own figure: not the same measure
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


def rows_query(metric: str | None) -> str | None:
    """The second retry's query: the statement's own rows (the filter already names the company and year)."""
    if metric in CASH_METRICS:
        return STATEMENT_ROWS["cash"]
    if metric not in STATEMENT_LABELS:
        return None
    return STATEMENT_ROWS["balance" if metric in BALANCE_METRICS else "income"]


def company_query(plan: dict, ticker: str, year: int, phrases: list[str], broadened: bool = False) -> str:
    company_plan = {"company": ticker, "company_name": NAMES.get(ticker, ticker), "metric": plan.get("metric")}
    return trend.year_query(company_plan, year, phrases, broadened=broadened)


# ----------------------------------------------------------------------------- extraction without the LLM

def _row_cells(line: str) -> tuple[str, str]:
    """('total net revenues', '$ 56,334 $ 54,318') for a statement row, in plain text or a Markdown table."""
    text = _DOT_LEADER.sub(" ", line.replace("|", " ").replace("—", " ").replace("’", "'")).strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower(), text


def labels_for(metric: str | None, query: str = "") -> tuple[str, ...] | None:
    """The row names of the measure. A question that says "net sales" wants that row, not "Total revenue" (Costco
    prints both: net sales 269,912 and total revenue 275,235)."""
    labels = STATEMENT_LABELS.get(metric or "")
    if labels and metric == "revenue" and re.search(r"\bnet\s+sales\b", query or "", re.I):
        return ("total net sales", "net sales")
    return labels


def scale_of_text(text: str) -> str:
    """'millions' from "(in millions)", "(MILLIONS OF DOLLARS)", "(MILLIONS, EXCEPT PER SHARE DATA)", "(Dollars in
    Millions)" or "($ millions)"; else ''."""
    match = _SCALE.search(text or "") or _SCALE_WORDS.search(text or "")
    return next((g for g in match.groups() if g), "").lower() if match else ""


def _row_rank(metric: str | None, label: str) -> int:
    """Which of a statement's matching rows is the measure. "Total revenues" when it also prints "Net sales"; net
    income attributable to the company (what XBRL calls NetIncomeLoss) before "Net income", which can include the
    minority share (Marathon Petroleum 2025: 5,878 and 4,047), and that before the row after preferred dividends."""
    if metric == "revenue":
        return 0 if label.startswith("total") else 1
    if metric == "net income":
        return 2 if "common" in label else 0 if re.search(r"attributable to|applicable to|s net ", label) else 1
    return 0


def _label_rows(hit: dict[str, Any], metric: str | None, year: int, labels: tuple[str, ...]) -> list[dict[str, Any]]:
    """Every row of the chunk labelled with the measure whose figures line up with the year header above it."""
    text = str(hit.get("text") or "")
    if parent_only(text) or subsidiary_statement(hit):
        return []
    companions = _COMPANIONS.get(metric or "")
    if companions and not hit.get("statement_page") and not statement_title(text) and not companions.search(text):
        return []
    scale = scale_of_text(text) or scale_of_text(f"({hit.get('scale') or ''})")
    header: list[int] = []
    pro_forma, last_pro_forma = False, -PRO_FORMA_LINES - 1
    segment, last_segment = False, -SEGMENT_LINES - 1
    rows = []
    spelled: dict[str, dict[int, str]] = {}                  # "Total revenues | 2022 | 43,166": a year per line
    for index, line in enumerate(text.splitlines()):
        cell = _YEAR_CELL.match(line.strip())
        if cell and cell["label"].strip().lower() in labels and not _PER_SHARE.match(cell["value"].strip()):
            spelled.setdefault(cell["label"].strip().lower(), {}).setdefault(int(cell["year"]), cell["value"].strip())
            continue
        lower, plain = _row_cells(line)
        if _PRO_FORMA.search(lower):
            last_pro_forma = index
        if SEGMENT_TABLE.search(lower) and not _ROW_FIGURE.search(lower):
            last_segment = index
        printed = [int(y) for y in _YEAR.findall(lower)]
        years = list(dict.fromkeys(printed))
        if len(years) >= 2 and not _ROW_FIGURE.search(_YEAR.sub(" ", lower)):
            header = years                              # "2024 2023 2022" or "| | 2024 | 2023 |"
            # A pro forma table ("as if the acquisition had occurred on ...") is not the reported figure.
            pro_forma = index - last_pro_forma <= PRO_FORMA_LINES
            # A segment's table, or an MD&A comparison ("2024 2023 % Change", "2024 2023 2022 2023 2022"): statements
            # have no change column.
            segment = (index - last_segment <= SEGMENT_LINES or bool(CHANGE_COLUMN.search(lower))
                       or len(printed) > len(years) + 1)
            continue
        if not header or year not in header or pro_forma or segment:
            continue
        label = max((l for l in labels if lower.startswith(l)), key=len, default=None)
        qualified = QUALIFIED_LABELS.get(metric or "")
        if qualified and (not label or not re.match(r"^\s*[\$(:\-]?\s*\(?\d", plain[len(label):])):
            found = qualified.match(lower)
            label = found.group(0) if found else label
        if not label:
            continue
        rest = plain[len(label):]
        if not re.match(r"^\s*[\$(:\-]?\s*\(?\d", rest):
            continue                                    # "Net revenues decreased ..." is prose, not a row
        figures = [f.replace("$", "").strip() for f in _ROW_FIGURE.findall(rest)]
        if len(figures) != len(header):                     # a small amount ("3,694 1,900 401") is a cell too
            cells = rest.replace("$", " ").split()
            if len(cells) == len(header) and all(_CELL.match(c) for c in cells):
                figures = cells
        if len(figures) != len(header) or (metric in STATEMENT_LABELS and all(_PER_SHARE.match(f) for f in figures)):
            continue                                    # Citi's "Net income $ 7.04 $ 10.21 $ 4.74" is per share
        if not trend.parse_figure(figures[header.index(year)]):
            continue                                    # a nil cell is not the company's revenue or assets
        rows.append({"value": figures[header.index(year)], "unit": scale, "chunk_id": hit.get("chunk_id"),
                     "quote": plain[:160], "columns": dict(zip(header, figures)),
                     "consolidated": bool(hit.get("statement_page") or statement_title(text)),
                     "rank": _row_rank(metric, label)})
    if not (_PRO_FORMA.search(text) or SEGMENT_TABLE.search(text)):
        rows += [{"value": years[year], "unit": scale, "chunk_id": hit.get("chunk_id"),
                  "quote": f"{label} | {year} | {years[year]}", "columns": years,
                  "consolidated": bool(hit.get("statement_page") or statement_title(text)),
                  "rank": _row_rank(metric, label)} for label, years in spelled.items() if year in years]
    return rows


def label_match(hits: list[dict[str, Any]], metric: str | None, year: int, labels: tuple[str, ...] | None = None,
                exclude: frozenset | set = frozenset()) -> dict[str, Any] | None:
    """The figure for ``year`` from a statement row labelled with the measure, read in the column of the nearest
    header above it that lists the fiscal years. None when no row and header line up exactly. ``labels`` overrides
    the measure's ``STATEMENT_LABELS``; ``exclude`` skips chunks whose figure was rejected. ``columns`` holds the row's
    other years, for ``prior_conflicts``."""
    labels = labels or STATEMENT_LABELS.get(metric or "")
    if not labels:
        return None
    from scripts.agents.financial_shareholder_returns import readable_hits
    usable = [h for h in hits if str(h.get("chunk_id")) not in exclude]
    # As printed first; then with rows laid out one value per line joined (OCR) and a page's header carried over.
    for candidates in (usable, readable_hits(usable)):
        # The consolidated statement first: a segment table or MD&A summary ranked above it prints the same row names.
        for group in (True, False):
            rows = [row for hit in candidates
                    if bool(hit.get("statement_page") or statement_title(hit.get("text"))) is group
                    for row in _label_rows(hit, metric, year, labels)]
            if rows:
                best = min(rows, key=lambda row: row["rank"])            # the first of the best-named rows
                return {k: v for k, v in best.items() if k != "rank"}
    return None


def prior_conflicts(points: dict[int, dict[str, Any]]) -> set[int]:
    """Fiscal years whose figure another filing's column contradicts by over ``PRIOR_CONFLICT``: not the same measure.

    Every annual report prints the years before it, so FY2025's row states FY2024's figure too. Capital One's FY2024
    "net interest income 20,881" was a segment's row; the FY2025 filing's column says 31,208. A figure that is not from
    a titled consolidated statement is doubted first; otherwise the one contradicted most often (Cheniere's FY2023 "Net
    income" included the minority share), or all of them when that doesn't decide (Duke Energy's subsidiaries each
    print a consolidated statement in the same filing; a spin-off restates earlier years, and a series mixing both is
    not one measure)."""
    doubted: dict[int, int] = {}
    for year, point in points.items():
        for other, text in (point.get("columns") or {}).items():
            theirs = points.get(int(other))
            if int(other) == year or not theirs or point.get("status") != "ok" or theirs.get("status") != "ok":
                continue
            a, b = trend.parse_figure(text), theirs.get("value")
            if a is None or not b or trend.scale_of(point.get("unit") or "") != trend.scale_of(theirs.get("unit") or ""):
                continue
            if abs(a - b) / abs(b) > PRIOR_CONFLICT:
                for y in (year, int(other)):
                    doubted[y] = doubted.get(y, 0) + 1
    if not doubted:
        return set()
    untitled = {y: n for y, n in doubted.items() if not points[y].get("consolidated")}
    doubted = untitled or doubted
    most = max(doubted.values())
    return {y for y, n in doubted.items() if n == most}


def reconcile_units(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A filing's row states other years in its own unit: when FY2023's row prints FY2022's exact figure, both are in one
    unit. A missing or misread unit ("thousands" of shares on ConocoPhillips' statement page) takes the neighbours'."""
    ok = [p for p in points if p.get("status") == "ok"]
    out = []
    for point in points:
        unit = trend.scale_of(point.get("unit") or "")
        votes = []
        for other in ok:
            if point.get("status") != "ok" or other is point:
                continue
            for mine, theirs in ((point, other), (other, point)):
                text = (mine.get("columns") or {}).get(theirs["year"])
                if text and trend.parse_figure(text) == theirs.get("value"):
                    votes.append(trend.scale_of(other.get("unit") or ""))
                    break
        votes = [v for v in votes if v]
        agreed = max(set(votes), key=votes.count) if votes else None
        if agreed and agreed != unit and (unit is None or votes.count(agreed) > votes.count(unit)
                                          and votes.count(agreed) >= 2):
            point = {**point, "unit": f"{agreed}s"}
        out.append(point)
    return out


def row_position_ok(point: dict[str, Any], hits: list[dict[str, Any]], year: int) -> bool:
    """An LLM-copied figure from a row of figures must sit in that year's column: under the header above it when the
    columns line up, else first (the filing's own year comes first). EOG's "Revenues 22,486 310 - 22,796" is a table by
    country, and 22,796 is not the company's 2022 revenue. A figure in a sentence passes."""
    hit = next((h for h in hits if str(h.get("chunk_id")) == str(point.get("chunk_id"))), None)
    value = str(point.get("value_text") or "").replace("$", "").strip()
    if hit is None or not value:
        return False
    header: list[int] = []
    for line in str(hit.get("text") or "").splitlines():
        lower, plain = _row_cells(line)
        years = list(dict.fromkeys(int(y) for y in _YEAR.findall(lower)))
        if len(years) >= 2 and not _ROW_FIGURE.search(_YEAR.sub(" ", lower)):
            header = years
            continue
        if value not in line:
            continue
        figures = [f.replace("$", "").strip() for f in _ROW_FIGURE.findall(plain)]
        if len(figures) < 2 or value not in figures or len(plain.split()) - len(figures) > 12:
            return True                                     # one figure, or prose
        if len(figures) == len(header) and year in header:
            return figures[header.index(year)] == value
        return figures[0] == value
    return True


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


def other_companys_row(point: dict[str, Any], hits: list[dict[str, Any]]) -> bool:
    """The LLM copied the figure from a parent-company-only statement or a segment's table: not the company's figure."""
    hit = next((h for h in hits if str(h.get("chunk_id")) == str(point.get("chunk_id"))), None)
    value = str(point.get("value_text") or "").replace("$", "").strip()
    if hit is None or not value:
        return False
    text = str(hit.get("text") or "")
    if parent_only(text):
        return True
    lines = text.splitlines()
    at = next((i for i, line in enumerate(lines) if value in line), None)
    if at is None:
        return False
    header = next((i for i in range(at, -1, -1) if len(set(_YEAR.findall(lines[i]))) >= 2), at)
    return any(SEGMENT_TABLE.search(line) and not _ROW_FIGURE.search(line) for line in lines[max(0, header - SEGMENT_LINES):header])


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
                                              "method", "reported_in")} for p in summary["points"]],
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
                     *[{"ticker": c["ticker"], "company": c["company"], "reason": ("figure not verified for "
                        + ", ".join(f"FY{p['year']}" for p in c["points"] if p.get("status") != "ok"))
                        if any(p.get("status") != "ok" for p in c["points"]) else "unit scale differs between years"} for c in companies if not c["in_panel"]]],
        "rows": rows,
        "cagr_total": cagr_total,
        "median_company_cagr": median_cagr,
        "companies": companies,
        "missing": missing,
        "unusual_changes": unusual,            # company changes beyond ±RAG_SECTOR_OUTLIER_PCT: sent to human review
        "chart": [{"year": r["year"], "total": r["total"]} for r in rows if r["total"] is not None],
        "calculations": calculations,
    }


# ----------------------------------------------------------------------------- the answer, written in code

def _money(value: float) -> str:
    return f"${value:,.0f} million" if float(value).is_integer() else f"${value:,.1f} million"


def answer_text(sector_trend: dict) -> str:
    """The sector answer, written from the table: every figure is either a company's verified figure (cited by its
    chunk) or a total, change or CAGR the sector calculator computed. No LLM writes it, so there is no claim for the
    support judge to check and nothing to repair; the critic checks the figures in code (``numeric_grounding``).
    On the hosted Qwen3-8B the LLM-written answer and its judge took ~55 s and the judge's reply on 24 figures was cut
    off at the 1,024-token limit, so the answer was withheld."""
    rows = [r for r in sector_trend.get("rows", []) if r.get("total") is not None]
    years = sector_trend.get("years") or []
    sector, metric = sector_trend.get("sector") or "The sector", sector_trend.get("metric") or "the measure"
    names = ", ".join(f"{n} ({t})" for n, t in zip(sector_trend.get("panel_names", []), sector_trend.get("panel", [])))
    lines = []
    panel, in_total = sector_trend.get("panel") or [], sector_trend.get("total_from")
    absent = [NAMES.get(t, t) for t in panel if in_total is not None and t not in in_total]
    left_out = f"without {', '.join(absent)}, whose unit could not be read" if absent else ""
    if rows and years:
        first, last = rows[0], rows[-1]
        head = (f"{sector} {metric}, FY{years[0]}–FY{years[-1]}, for the {len(sector_trend.get('panel') or [])} "
                f"companies whose figure is verified in every year ({names}): the panel total"
                + (f" ({left_out})" if left_out else "") + " went from "
                f"{_money(first['total'])} in FY{first['year']} to {_money(last['total'])} in FY{last['year']}")
        if sector_trend.get("cagr_total") is not None:
            head += f", a compound annual growth rate of {sector_trend['cagr_total']:.2f}%"
        lines.append(head + ".")
        lines.append("")
        for row in rows:
            line = f"- FY{row['year']}: {_money(row['total'])}"
            if row.get("total_pct_change") is not None:
                line += f" ({row['total_pct_change']:+.2f}% on FY{row['year'] - 1})"
            if row.get("median_company_pct_change") is not None:
                line += f"; median company change {row['median_company_pct_change']:+.2f}%"
            if row.get("growing") is not None:
                line += f"; {row['growing']} of {row['companies']} companies grew"
            lines.append(line + ".")
        if sector_trend.get("median_company_cagr") is not None:
            lines.append(f"- Median company CAGR: {sector_trend['median_company_cagr']:.2f}%.")
    lines += ["", "Company figures (as reported, USD millions):"]
    for company in sector_trend.get("companies", []):
        if not company.get("in_panel"):
            continue
        figures = "; ".join(f"FY{p['year']} {p['value_text']}"
                            + (f" (as printed in the FY{p['reported_in']} report)" if p.get("reported_in") else "")
                            + f" [{p['chunk_id']}]"
                            for p in company.get("points", []) if p.get("status") == "ok")
        lines.append(f"- {company['company']} ({company['ticker']}): {figures}.")
    excluded = sector_trend.get("excluded") or []
    if excluded:
        lines += ["", "Not in the panel: " + "; ".join(f"{e['company']} ({e['ticker']}): {e['reason']}"
                                                       for e in excluded) + "."]
    return "\n".join(lines)
