"""Dividend-income agent: what the dividends on the shares a user holds came to, from the companies' own filings.

"I hold 100 shares of Coca-Cola and 50 shares of JPMorgan. How much dividend income was that in 2024?" (technique
``dividend_income_graph`` or the router's ``dividend_income`` route):

1. ``income_plan``: the holdings (share count -> company, at most ``MAX_HOLDINGS``) and each company's fiscal year
   (the one asked, else its latest indexed filing). A holding without a share count, or a company without a filing
   for that year, is reported with the reason, never guessed.
2. One filtered search per holding for the dividends declared per common share, in parallel.
3. Per holding, code first: ``read_dps`` reads the per-share row ("Cash dividends declared per common share  $ 1.94
   $ 1.84") in the column of that fiscal year. Without one, a broadened, deeper search and the row again; only then
   does the LLM copy the figure, kept only if its chunk prints it on a line naming dividends per share.
4. The action: ``income`` multiplies each verified dividend per share by the user's share count and adds the
   holdings up, in code. The share count is the user's input and is recorded as such (``source: user``); the
   dividend per share must come from a cited chunk.
5. ``answer_text`` writes the answer from that table; the critic checks every figure in code.

Dividends declared per share for the company's fiscal year, before tax. What a holder actually received depends on
when the shares were held (record dates); the answer says so. It is not advice on whether to hold or buy.
"""
from __future__ import annotations

import re
from typing import Any

from scripts.agents import financial_sector_trend as sector_agent
from scripts.common.companies import NAMES, normalize_ticker

INCOME_TOOLS = ("coverage_lookup", "text_search", "per_share_extract", "dividend_calculator")
MAX_HOLDINGS = 5
MAX_SHARES = 100_000_000
MEASURE = "dividends per share"

DPS_LABELS: tuple[str, ...] = (
    "cash dividends declared per share of common stock", "dividends declared per share of common stock",
    "cash dividends declared per common share", "cash dividends declared per share", "dividends declared per common share",
    "dividends declared per share", "cash dividends paid per common share", "cash dividends paid per share",
    "dividends paid per common share", "dividends paid per share", "cash dividends per common share",
    "cash dividends per share", "dividends per common share", "dividends per share")
PHRASES = ["dividends declared per", "dividends paid per", "dividends per", "dividend per", "per common share",
           "per share"]
DIVIDEND_WORD = re.compile(r"\bdividends?\b|\bpayouts?\b", re.I)
# A share count: "100 shares", "1,500 Apple shares", "250 shares of JPM" (a year is never a share count).
SHARE_COUNT = re.compile(r"(?<![\w.$])(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s+(?:[A-Za-z&.'’\-]+\s+){0,3}?(?:shares?|stocks?)\b",
                         re.I)


def holdings_signal(query: str) -> bool:
    """The question gives a share count and asks about dividends: a dividend-income calculation."""
    text = query or ""
    return bool(DIVIDEND_WORD.search(text)) and any(not _is_year(m.group(1)) for m in SHARE_COUNT.finditer(text))


def _is_year(raw: str) -> bool:
    return "," not in raw and 1990 <= int(raw) <= 2049


def parse_holdings(query: str, clarifier, ticker: str | None = None) -> tuple[list[dict], list[str]]:
    """(shares, ticker) pairs: each share count goes with the company named between it and the next count (or, for
    "Apple: 100 shares", just before it). ``ticker`` (the request's filter) fills a count with no company."""
    counts = [m for m in SHARE_COUNT.finditer(query or "") if not _is_year(m.group(1))]
    holdings, problems = [], []
    for index, match in enumerate(counts):
        end = counts[index + 1].start() if index + 1 < len(counts) else len(query)
        start = counts[index - 1].end() if index else 0
        after = clarifier.companies_in(query[match.start():end])
        before = clarifier.companies_in(query[start:match.start()]) if not after else []
        found = after or before[-1:] or ([normalize_ticker(ticker)] if ticker else [])
        shares = float(match.group(1).replace(",", ""))
        if not found:
            problems.append(f"no company named for the {match.group(1)} shares")
            continue
        if not 0 < shares <= MAX_SHARES:
            problems.append(f"{match.group(1)} is not a share count this calculator takes (1 to {MAX_SHARES:,})")
            continue
        holdings.append({"ticker": normalize_ticker(found[0]), "shares": shares, "shares_text": match.group(1)})
    merged: dict[str, dict] = {}
    for holding in holdings:                       # the same company named twice: the counts add up
        if holding["ticker"] in merged:
            merged[holding["ticker"]]["shares"] += holding["shares"]
            merged[holding["ticker"]]["shares_text"] = f"{merged[holding['ticker']]['shares']:,.0f}"
        else:
            merged[holding["ticker"]] = dict(holding)
    return list(merged.values()), problems


# ----------------------------------------------------------------------------- plan

def income_plan(query: str, clarifier, *, ticker: str | None = None, year: int | None = None) -> dict[str, Any]:
    """Holdings, each company's fiscal year and one search target per holding."""
    directory = clarifier.directory or {}
    holdings, problems = parse_holdings(query, clarifier, ticker)
    asked = clarifier.years_in(query) or ([int(year)] if year else [])
    adjustments = []
    if len(asked) > 1:
        adjustments.append(f"one fiscal year at a time: FY{asked[-1]} used")
    if not holdings and not problems:
        problems.append("no holding given (for example: 100 shares of Coca-Cola)")
    if len(holdings) > MAX_HOLDINGS:
        adjustments.append(f"at most {MAX_HOLDINGS} holdings: "
                           + ", ".join(h["ticker"] for h in holdings[MAX_HOLDINGS:]) + " left out")
        holdings = holdings[:MAX_HOLDINGS]
    targets, excluded = [], []
    for holding in holdings:
        t = holding["ticker"]
        available = directory.get(t, [])
        fiscal_year = asked[-1] if asked else (available[-1] if available else None)
        holding.update(company=NAMES.get(t, t), year=fiscal_year)
        if not available:
            excluded.append({"ticker": t, "company": NAMES.get(t, t), "reason": "no filings indexed"})
        elif fiscal_year not in available:
            excluded.append({"ticker": t, "company": NAMES.get(t, t), "reason": f"no FY{fiscal_year} filing indexed "
                             f"(indexed: {', '.join(f'FY{y}' for y in available)})"})
        else:
            targets.append({"ticker": t, "year": fiscal_year, "measure": MEASURE, "shares": holding["shares"]})
    if holdings and not targets:
        problems.append("none of the holdings has a filing for the year asked")
    return {
        "intent": "dividend_income",
        "holdings": holdings,
        "excluded": excluded,
        "tools": list(INCOME_TOOLS),
        "targets": targets,
        "feasible": not problems,
        "problems": problems,
        "planner": "rules",
        "adjustments": adjustments,
    }


def not_available_message(plan: dict) -> str:
    return "I can't calculate the dividend income: " + "; ".join(plan.get("problems") or ["no plan"]) + "."


def target_label(target: dict) -> str:
    return f"{target['ticker']} FY{target['year']} {MEASURE}"


def search_query(ticker: str, year: int, broadened: bool = False) -> str:
    query = f"{NAMES.get(ticker, ticker)} cash dividends declared per common share fiscal year {year}"
    if broadened:
        query += " dividends paid per share selected financial data stockholders' equity"
    return query


PER_SHARE = re.compile(r"^\(?\$?\s?\d{1,3}\.\d{2,4}\)?$")             # "1.94", "$ 0.25": a per-share amount


def read_dps(hits: list[dict[str, Any]], year: int) -> dict[str, Any] | None:
    """The dividends-per-share row for ``year``, read in code; only a per-share amount (two to four decimals)."""
    from scripts.agents.financial_shareholder_returns import readable_hits
    for hit in readable_hits(hits):                  # vertical rows joined, the page's year header carried over
        row = sector_agent.label_match([hit], MEASURE, year, labels=DPS_LABELS)
        if row and PER_SHARE.match(row["value"].strip()):
            return {**row, "unit": "USD per share"}
    return None


# ----------------------------------------------------------------------------- the action, in code

def _fmt(value: float) -> str:
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def income(plan: dict, points: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """shares x dividend per share for each holding with a verified figure, and the total; recorded as calculations
    (the share count marked as the user's input)."""
    rows, calculations = [], []
    for target in plan.get("targets") or []:
        t, shares = target["ticker"], target["shares"]
        point = points.get(t) or {"status": "not_found"}
        row = {"ticker": t, "company": NAMES.get(t, t), "year": target["year"], "shares": shares,
               "dps": {k: point.get(k) for k in ("value", "value_text", "unit", "chunk_id", "page", "status", "method")},
               "income": None}
        if point.get("status") == "ok" and point.get("value") is not None:
            dps = abs(point["value"])
            amount = round(shares * dps, 4)
            row["income"] = amount
            calculations.append({"expression": f"{_fmt(shares)}", "value": shares, "grounded": True,
                                 "kind": "user_input", "source": "user", "ticker": t})
            calculations.append({"expression": f"{_fmt(shares)} * {_fmt(dps)}", "value": amount, "grounded": True,
                                 "kind": "dividend_income", "ticker": t, "year": target["year"],
                                 "operands": {"shares": "user", "dividend_per_share": point.get("chunk_id")}})
        rows.append(row)
    done = [r for r in rows if r["income"] is not None]
    total = round(sum(r["income"] for r in done), 4) if done else None
    if len(done) >= 2:
        calculations.append({"expression": " + ".join(_fmt(r["income"]) for r in done), "value": total,
                             "grounded": True, "kind": "dividend_income_total"})
    return {
        "holdings": rows,
        "total": total,
        "calculated": len(done),
        "excluded": [*plan.get("excluded", []),
                     *[{"ticker": r["ticker"], "company": r["company"],
                        "reason": f"dividends per share for FY{r['year']} not verified"} for r in rows if r["income"] is None]],
        "calculations": calculations,
    }


def _usd(value: float) -> str:
    return f"${value:,.2f}"


def answer_text(table: dict) -> str:
    """The answer, written from the calculation: each dividend per share is cited by its chunk; each amount is the
    user's share count times that figure. No LLM writes it; the critic checks its figures in code."""
    done = [r for r in table.get("holdings", []) if r.get("income") is not None]
    lines = []
    if done:
        if len(done) >= 2:
            lines.append(f"Your {len(done)} holdings come to {_usd(table['total'])} in dividends, at the dividends "
                         "declared per share in each company's fiscal year:")
        for row in done:
            dps = row["dps"]
            lines.append(f"- {row['company']} ({row['ticker']}), FY{row['year']}: {_fmt(row['shares'])} shares x "
                         f"${dps['value_text'].replace('$', '').strip()} declared per share [{dps['chunk_id']}] = "
                         f"{_usd(row['income'])}.")
    excluded = table.get("excluded") or []
    if excluded:
        lines += ["", "Not calculated: " + "; ".join(f"{e['company']} ({e['ticker']}): {e['reason']}"
                                                     for e in excluded) + "."]
    lines += ["", "Dividends declared per common share for the company's fiscal year, before tax. What you actually "
                  "received depends on when you held the shares (the record dates); it describes past payments, not a "
                  "recommendation."]
    return "\n".join(lines)
