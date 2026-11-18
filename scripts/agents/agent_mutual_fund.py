"""Fund agent (design section 6): questions about a mutual fund's N-PORT holdings, linked to the holdings' annual reports.

The router's ``fund`` route runs: route (fund plan) -> fund_lookup (fixed SQL rows as evidence) -> fund_company (only
when company figures or text are needed) -> answer -> critic -> governance: 8 steps, 10 with one repair. Retries happen
inside a node, as in the sector trend agent.

What a plan can ask (``kind``):

| kind | example | data |
|---|---|---|
| top_holdings | What are the top holdings of the Vanguard 500 Index Fund? | N-PORT rows |
| sector_mix | How much of the fund is in technology? | N-PORT rows mapped to corpus sectors |
| exposure | How much NVIDIA does the Vanguard 500 fund hold? | N-PORT rows |
| holders | Which funds hold the most NVIDIA? | N-PORT rows across loaded funds |
| facts / flows / returns | net assets, monthly sales and redemptions, monthly total returns | N-PORT rows |
| holdings_growth | How did revenue of the fund's top 5 holdings grow? | N-PORT weights + each company's annual report |
| holdings_text | What risks do the fund's three largest holdings report? | N-PORT weights + filtered search per company |

Every number is a row of the database or a calculation recorded by code (weighted growth); company figures are read
from the annual report each company filed on or before the holdings date (no look-ahead), statement row in code first,
the LLM last, and cached in ``company_figures``. Without loaded fund data there is no fund route at all.
"""
from __future__ import annotations

import os
import re
from typing import Any

from scripts.agents.tools_mutual_fund import (FundStore, alias_words, coverage_row, exposure_row, facts_row, flow_row,
                                              holder_rows, holding_rows, pct, report_for, return_rows, sector_rows)
from scripts.common.companies import NAMES

FUND_WORDS = re.compile(r"\b(mutual funds?|index funds?|funds?|etfs?)\b", re.I)
HOLDING_WORDS = re.compile(r"\b(hold|holds|holding|holdings|held|portfolio|positions?|exposure|allocation|weights?|"
                           r"net assets|invest(s|ed|ing)?|owns?|top \d+|largest)\b", re.I)
HOLDERS = re.compile(r"\b(which|what|who|list)\b.{0,30}\bfunds?\b.{0,40}\b(hold|holds|own|owns|invest)", re.I)
GROWTH = re.compile(r"\b(grow|grew|growth|trend|change|changed|increase|increased|decline|declined|rise|rose)\b", re.I)
TEXT = re.compile(r"\b(risks?|strategy|strategies|outlook|priorities|say|said|describe|explain|challenges?)\b", re.I)
SECTORS = re.compile(r"\b(sectors?|industr(y|ies)|allocation|breakdown|diversifi\w*|technology|healthcare|banking)\b", re.I)
RETURNS = re.compile(r"\b(returns?|performance|performed|perform)\b", re.I)
FLOWS = re.compile(r"\b(flows?|inflows?|outflows?|redemptions?|subscriptions?|sales)\b", re.I)
FACTS = re.compile(r"\b(net assets|total assets|size|how (big|large)|aum|assets under management|liabilities)\b", re.I)
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "fifteen": 15, "twenty": 20}
TOP_N = re.compile(r"\b(?:top|largest|biggest)\s+(\d{1,2}|" + "|".join(NUMBER_WORDS) + r")\b|\b(\d{1,2}|"
                   + "|".join(NUMBER_WORDS) + r")\s+(?:largest|biggest|top)\b", re.I)
DEFAULT_N = {"top_holdings": 10, "holdings_growth": 5, "holdings_text": 3, "holders": 10}
MIN_COVERAGE = 0.5              # human review when the companies used carry under half the weight asked about
# Advice wording in a fund question. "hold" / "holdings" describe the data here, so they don't count as advice.
ADVICE = re.compile(r"\b(buy|sell|should (i|we)|recommend\w*|price target|undervalued|overvalued|best fund|"
                    r"better fund|worth (buying|investing)|invest in)\b", re.I)


def review_triggers(query: str, triggers: list[str]) -> list[str]:
    """The governance triggers that apply to a fund answer: the investment-language trigger only for advice wording."""
    if ADVICE.search(query or ""):
        return list(triggers)
    return [t for t in triggers if t != "investment_or_decision_support_language"]


def max_companies() -> int:
    return max(1, int(os.getenv("RAG_FUND_MAX_COMPANIES", "10")))


def default_store() -> FundStore:
    return FundStore()


def fund_signal(query: str, store: FundStore | None = None) -> bool:
    """The question is about a fund: it names a loaded fund, or a fund word plus a holdings word. False whenever no
    fund data is loaded, so the other routes are unchanged without it."""
    store = store or default_store()
    if not store.available():
        return False
    if store.resolve(query)["matches"]:
        return True
    return bool(FUND_WORDS.search(query) and (HOLDING_WORDS.search(query) or HOLDERS.search(query)))


def top_n(query: str, kind: str) -> int:
    match = TOP_N.search(query)
    if match:
        raw = (match.group(1) or match.group(2)).lower()
        n = NUMBER_WORDS.get(raw) or int(raw)
    else:
        n = DEFAULT_N.get(kind, 10)
    limit = max_companies() if kind in ("holdings_growth", "holdings_text") else 50
    return max(1, min(n, limit))


def kind_of(query: str, fund_named: bool, companies: list[str], metric: str | None) -> str:
    if not fund_named and companies and (HOLDERS.search(query) or FUND_WORDS.search(query)):
        return "holders"
    if metric and GROWTH.search(query):
        return "holdings_growth"
    if TEXT.search(query) and re.search(r"\b(holdings?|positions?|companies)\b", query, re.I):
        return "holdings_text"
    if companies:
        return "exposure"
    if RETURNS.search(query):
        return "returns"
    if FLOWS.search(query):
        return "flows"
    if SECTORS.search(query):
        return "sector_mix"
    if FACTS.search(query):
        return "facts"
    return "top_holdings"


def fund_plan(query: str, clarifier, store: FundStore | None = None, ticker: str | None = None) -> dict[str, Any]:
    """What to look up, for which fund and period. Infeasible plans say why (no data, fund not loaded, which fund)."""
    store = store or default_store()
    plan: dict[str, Any] = {"intent": "fund", "tools": ["fund_search", "fund_lookup"], "feasible": False,
                            "problems": [], "query": query}
    if not store.available():
        plan["problems"].append("no mutual fund data is loaded (run scripts.rag.nport_mutual_fund load)")
        return plan
    if ADVICE.search(query):
        plan.update(advice=True, problems=["it asks for investment advice. This copilot states facts from SEC filings "
                                           "only; ask about a fund's holdings, sector mix, exposure, flows or returns"])
        return plan
    slots = clarifier.parse(query)
    resolved = store.resolve(query)
    funds = resolved["matches"]
    if not funds:
        found = search_cards(query)
        funds = [f for f in store.funds() if f["series_id"] in found[:1]]
        if funds:
            plan["fund_search"] = {"method": "fund_cards", "series_id": found[0]}
    fund = funds[0] if len(funds) == 1 else None
    names = " ".join(f["series_name"].lower() for f in funds)
    companies = [c for c in (slots.companies or ([ticker.upper()] if ticker else []))
                 if not any(a and a in names for a in alias_words(c))]           # 'JPMorgan' in a JPMorgan fund's name
    metric = clarifier.metric_in(query)
    kind = kind_of(query, bool(funds), companies, metric)
    plan.update(kind=kind, companies=companies, metric=metric, years=slots.years)
    if kind == "holders":
        plan.update(feasible=True, ticker=companies[0], n=top_n(query, kind), tools=["fund_lookup"])
        return plan
    if len(funds) > 1:
        plan["problems"].append("the question matches several loaded funds: "
                                + "; ".join(f["series_name"].title() for f in funds[:8]) + ". Name one")
        return plan
    if not fund:
        family = resolved["family_only"]
        plan["problems"].append(
            ("which " + family[0]["family"] + " fund? Loaded: " + "; ".join(f["series_name"].title() for f in family[:8]))
            if family else "the question names no loaded fund. Loaded funds: "
            + "; ".join(f["series_name"].title() for f in store.funds()[:8]))
        return plan
    year = slots.years[-1] if slots.years else None
    filing = store.filing(fund["series_id"], year) or (None if year else store.filing(fund["series_id"]))
    if not filing:
        periods = ", ".join(f["period_end"] for f in fund["filings"])
        plan["problems"].append(f"no {fund['series_name'].title()} filing with holdings dated in {year} is loaded "
                                f"(loaded: {periods})")
        return plan
    needs_companies = kind in ("holdings_growth", "holdings_text")
    if kind == "holdings_growth" and not metric:
        plan["problems"].append("name the measure to compare across the holdings (revenue, net income, ...)")
        return plan
    plan.update(feasible=True, series_id=fund["series_id"], fund=fund["series_name"].title(), filing=filing,
                accession=filing["accession"], period_end=filing["period_end"], n=top_n(query, kind),
                needs_companies=needs_companies,
                tools=["fund_search", "fund_lookup"] + (["company_figures", "text_search"] if needs_companies else []))
    return plan


CARD_MIN_SCORE, CARD_MIN_MARGIN = 0.6, 0.03


def search_cards(query: str) -> list[str]:
    """The nearest fund card's series ID, only when it clearly beats the next card. Index funds share most holdings,
    so their cards score close together (4 Vanguard index funds: top-two gaps of 0.005 to 0.03): a close call is
    not a match, and the plan then asks which fund. Empty when no cards are built or the embedder can't start."""
    try:
        from scripts.rag import cards_mutual_fund as cards
        hits = cards.search(query, top_k=2)
    except Exception:  # noqa: BLE001 - search is optional: names already resolved most questions
        return []
    if not hits or hits[0].get("score", 0) < CARD_MIN_SCORE:
        return []
    if len(hits) > 1 and hits[0]["score"] - hits[1].get("score", 0) < CARD_MIN_MARGIN:
        return []
    return [hits[0]["series_id"]]


def not_available_message(plan: dict[str, Any]) -> str:
    return "I can't answer this fund question: " + "; ".join(plan.get("problems") or ["no plan"]) + "."


# ---------------------------------------------------------------------------- fund_lookup
def lookup(plan: dict[str, Any], store: FundStore | None = None) -> dict[str, Any]:
    """Run the plan's fixed queries. Returns evidence rows, the table shown in the UI and the tool-call log."""
    store = store or default_store()
    kind = plan["kind"]
    table: dict[str, Any] = {"kind": kind, "metric": plan.get("metric"), "rows": [], "companies": [], "excluded": []}
    rows: list[dict[str, Any]] = []
    if kind == "holders":
        holders = store.holders(plan["ticker"], plan["n"])
        rows = holder_rows(plan["ticker"], holders, store)
        table.update(ticker=plan["ticker"], company=NAMES.get(plan["ticker"], plan["ticker"]),
                     rows=[{"fund": h["series_name"].title(), "holdings as of": h["period_end"],
                            "% of net assets": round(h["pct_of_net_assets"], 4),
                            "value (USD m)": round((h["value_usd"] or 0) / 1e6, 1)} for h in holders])
        return {"evidence": rows, "table": table, "calls": [{"tool": "fund_lookup", "query": "holders",
                                                              "ticker": plan["ticker"], "rows": len(rows)}]}
    filing, accession = plan["filing"], plan["accession"]
    info = store.coverage(accession)
    table.update(fund=plan["fund"], series_id=plan["series_id"], accession=accession, period_end=plan["period_end"],
                 filing_date=filing.get("filing_date"), net_assets=filing.get("net_assets"), coverage=info)
    rows = [facts_row(filing), coverage_row(filing, info)]
    queries = ["facts", "coverage"]
    if kind in ("top_holdings", "holdings_growth", "holdings_text"):
        holdings = store.top_holdings(accession, plan["n"])
        rows += holding_rows(filing, holdings)
        queries.append(f"top_holdings({plan['n']})")
        table["rows"] = [{"rank": h["rank"], "holding": h["issuer_name"], "ticker": h.get("ticker") or "-",
                          "% of net assets": round(h["pct_of_net_assets"], 4),
                          "value (USD m)": round((h["value_usd"] or 0) / 1e6, 1), "match": h.get("method") or "-"}
                         for h in holdings]
        table["holdings"] = holdings
    if kind == "sector_mix":
        mix = store.sector_mix(accession)
        rows += sector_rows(filing, mix)
        queries.append("sector_mix")
        table["sector_mix"] = [{"sector": r["sector"], "% of net assets": round(r["pct_of_net_assets"], 4),
                                "companies": r["companies"]} for r in mix]
    if kind == "exposure":
        for ticker in plan["companies"][:max_companies()]:
            found = store.exposure(accession, ticker)
            rows.append(exposure_row(filing, ticker, found))
            table["rows"].append({"company": f"{NAMES.get(ticker, ticker)} ({ticker})",
                                  "% of net assets": round(found["pct_of_net_assets"], 4) if found else 0.0,
                                  "value (USD m)": round((found["value_usd"] or 0) / 1e6, 1) if found else 0.0})
        queries.append("exposure")
    if kind in ("flows", "facts"):
        row = flow_row(filing, store.flows(accession))
        rows += [row] if row else []
        queries.append("flows")
    if kind == "returns":
        rows += return_rows(filing, store.returns(accession))
        queries.append("monthly_returns")
    return {"evidence": rows, "table": table,
            "calls": [{"tool": "fund_lookup", "query": q, "accession": accession} for q in queries]}


# ---------------------------------------------------------------------------- linking holdings to annual reports
def company_targets(table: dict[str, Any], directory: dict[str, list[int]]) -> list[dict[str, Any]]:
    """For each top holding with a corpus ticker: the annual report filed by the holdings date and indexed.
    Holdings without one are listed in ``table['excluded']`` with the reason."""
    targets = []
    for h in table.get("holdings") or []:
        ticker = h.get("ticker")
        if not ticker:
            table["excluded"].append({"company": h["issuer_name"], "weight": h["pct_of_net_assets"],
                                      "reason": "not in the corpus"})
            continue
        report = report_for(ticker, table["period_end"], directory.get(ticker) or [])     # indexed filings only
        if not report:
            table["excluded"].append({"company": f"{NAMES.get(ticker, ticker)} ({ticker})", "weight": h["pct_of_net_assets"],
                                      "reason": f"no indexed annual report filed by {table['period_end']}"})
            continue
        targets.append({"ticker": ticker, "year": report["year"], "weight": h["pct_of_net_assets"], "report": report})
    return targets


def growth(table: dict[str, Any], figures: dict[str, dict[int, dict[str, Any]]]) -> list[dict[str, Any]]:
    """Each company's growth from its two verified figures and the weighted average over the covered weight,
    computed in code and recorded as calculations."""
    calculations, covered, weighted_sum = [], 0.0, 0.0
    asked = sum(h["pct_of_net_assets"] or 0 for h in table.get("holdings") or [])
    for company in table["companies"]:
        points = figures.get(company["ticker"], {})
        year = company["year"]
        now, before = points.get(year) or {}, points.get(year - 1) or {}
        company["points"] = [before or {"year": year - 1, "status": "missing"}, now or {"year": year, "status": "missing"}]
        same = (now.get("status") == before.get("status") == "ok" and now.get("value") == before.get("value")
                and now.get("chunk_id") == before.get("chunk_id") and "llm" in (now.get("method"), before.get("method")))
        if same:
            # The model copied one figure for both years from one chunk: a column it could not tell apart, not 0%.
            company["growth_pct"] = None
            table["excluded"].append({"company": f"{NAMES.get(company['ticker'])} ({company['ticker']})",
                                      "weight": company["weight"],
                                      "reason": f"the same figure was read for FY{year - 1} and FY{year}"})
            continue
        if now.get("status") == "ok" and before.get("status") == "ok" and before.get("value"):
            change = (now["value"] - before["value"]) / abs(before["value"]) * 100
            company["growth_pct"] = round(change, 4)
            calculations.append({"expression": f"({now['value']:g} - {before['value']:g}) / {abs(before['value']):g} * 100",
                                 "value": round(change, 4), "grounded": True, "label": f"{company['ticker']} growth"})
            covered += company["weight"]
            weighted_sum += company["weight"] * change
        else:
            company["growth_pct"] = None
            table["excluded"].append({"company": f"{NAMES.get(company['ticker'])} ({company['ticker']})",
                                      "weight": company["weight"],
                                      "reason": f"{table.get('metric')} not verified for FY{year - 1} and FY{year}"})
    used = [c for c in table["companies"] if c.get("growth_pct") is not None]
    table["weighted_growth_pct"] = round(weighted_sum / covered, 4) if covered else None
    if used:
        calculations.append({"expression": "(" + " + ".join(f"{c['weight']:g} * {c['growth_pct']:g}" for c in used)
                             + ") / (" + " + ".join(f"{c['weight']:g}" for c in used) + ")",
                             "value": table["weighted_growth_pct"], "grounded": True, "label": "weighted growth"})
        calculations.append({"expression": " + ".join(f"{c['weight']:g}" for c in used), "value": round(covered, 4),
                             "grounded": True, "label": "covered weight"})
    table.update(covered_weight=round(covered, 4), asked_weight=round(asked, 4),
                 coverage_share=round(covered / asked, 4) if asked else 0.0)
    return calculations


def low_coverage(table: dict[str, Any]) -> bool:
    return table.get("kind") in ("holdings_growth", "holdings_text") and table.get("coverage_share", 1) < MIN_COVERAGE


# ---------------------------------------------------------------------------- prompt text
def table_text(table: dict[str, Any]) -> str:
    lines = []
    if table.get("kind") == "holdings_growth":
        for c in table.get("companies") or []:
            points = " ; ".join(f"FY{p['year']} {p.get('value_text') or p.get('status')} [{p.get('chunk_id') or '-'}]"
                                for p in c.get("points") or [])
            lines.append(f"{NAMES.get(c['ticker'])} ({c['ticker']}): weight {pct(c['weight'])}; {table.get('metric')} "
                         f"{points}; growth {pct(c.get('growth_pct'))}")
        lines.append(f"Weighted growth over the covered holdings: {pct(table.get('weighted_growth_pct'))} "
                     f"(covered weight {pct(table.get('covered_weight'))} of the {pct(table.get('asked_weight'))} asked "
                     "about)")
    elif table.get("kind") == "holdings_text":
        for c in table.get("companies") or []:
            lines.append(f"{NAMES.get(c['ticker'])} ({c['ticker']}): weight {pct(c['weight'])}; annual report FY{c['year']}")
    for e in table.get("excluded") or []:
        lines.append(f"Not used: {e['company']} (weight {pct(e['weight'])}): {e['reason']}")
    return "\n".join(lines) or "none"
