"""Router (supervisor) of the financial agent: which flow answers a question.

The router LLM proposes a route (Route JSON); ``decide`` validates it against deterministic signals from the
clarifier's parser, the same way ``financial_tools.normalize_plan`` validates the planner's tools:

* ``lookup`` and ``comparison`` -> the Q&A agent (``agentic_graph``: one filtered search per company x year);
* ``trend`` -> the multi-year trend agent (``financial_trend``), only when a trend can be built (one company,
  one measure, two or more indexed years); otherwise the question is answered as a lookup or comparison;
* ``sector_trend`` -> the sector trend agent (``financial_sector_trend``): one measure across one sector's
  companies, year by year. Chosen when the question names a sector (or the request filters on one) and no company;
  a plan that can't be built is answered as ``not_available`` with the reason, never as a guess;
* ``report`` -> a handoff to the deterministic report pipeline, which runs only after the user confirms;
* ``out_of_scope`` -> refused, but only when the question names no company, measure or known intent.

Explicit signals win: a question the parser reads as a report, comparison or trend goes there even if the LLM said
otherwise, and an LLM "report" or "out_of_scope" without those signals is downgraded. Every override is listed in
``adjustments`` and ``decided_by`` says whether the LLM, the rules or the technique chose the route.
"""
from __future__ import annotations

from typing import Any

ROUTES = ("lookup", "comparison", "trend", "sector_trend", "report", "out_of_scope")
FLOW_OF = {"lookup": "agentic_graph", "comparison": "agentic_graph", "trend": "trend_graph",
           "sector_trend": "sector_trend_graph", "report": "report_handoff", "out_of_scope": "refuse"}
ALIASES = {"qa": "lookup", "figure": "lookup", "fact": "lookup", "explain": "lookup", "explanation": "lookup",
           "question": "lookup", "annual_report_qa": "lookup", "compare": "comparison", "trend_analysis": "trend",
           "growth": "trend", "time_series": "trend", "sector": "sector_trend", "sector_analysis": "sector_trend",
           "industry_trend": "sector_trend", "full_report": "report", "report_generation": "report",
           "refuse": "out_of_scope", "off_topic": "out_of_scope", "unsupported": "out_of_scope"}
RULE_ROUTE = {"report": "report", "comparison": "comparison", "trend": "trend", "sector_trend": "sector_trend",
              "figure": "lookup", "explain": "lookup"}
EXPLICIT = ("report", "comparison", "trend", "sector_trend")      # the parser's reading wins over the LLM's
OUT_OF_SCOPE_ANSWER = ("This copilot answers questions about the companies' annual reports. The question names no "
                       "company, measure or report topic, so it was not routed to an agent.")


def normalize_route(value: Any) -> str | None:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    text = ALIASES.get(text, text)
    return text if text in ROUTES else None


def rule_route(query: str, clarifier, ticker: str | None = None, sector: str | None = None) -> tuple[str, dict[str, Any]]:
    """The route the question's own words imply, and the signals behind it. ``sector`` is the request's sector
    filter: a trend question naming no company and no sector becomes that sector's trend."""
    from scripts.agents.financial_sector_trend import sector_from_filter
    slots = clarifier.parse(query)
    companies = [ticker.upper()] if ticker and not slots.companies else slots.companies
    intent = clarifier.intent_in(query, companies, slots.years) if companies != slots.companies else slots.intent
    sectors = slots.sectors or ([] if companies else sector_from_filter(sector))
    if intent == "trend" and not companies and sectors:
        intent = "sector_trend"
    signals = {"intent": intent, "companies": companies, "metric": slots.metric, "years": slots.years,
               "sectors": sectors}
    return RULE_ROUTE.get(intent, "lookup"), signals


def decide(raw: dict | None, query: str, clarifier, trend_plan: dict, *, ticker: str | None = None,
           forced: str | None = None, sector_plan: dict | None = None, sector: str | None = None) -> dict[str, Any]:
    rule, signals = rule_route(query, clarifier, ticker, sector)
    raw_route = raw.get("route") if isinstance(raw, dict) else None
    llm = normalize_route(raw_route)
    adjustments: list[str] = []
    if forced:
        route, decided_by = forced, "technique"
    elif llm is None:
        route, decided_by = rule, "rules"
        if raw is not None:
            adjustments.append(f"router LLM gave no valid route ({raw_route!r}): rules chose {rule!r}")
    elif rule in EXPLICIT and llm != rule:
        route, decided_by = rule, "rules"
        adjustments.append(f"rules overrode {llm!r}: the question reads as a {rule} (intent {signals['intent']!r})")
    elif llm == "report":
        route, decided_by = "lookup", "rules"
        adjustments.append("downgraded 'report' to 'lookup': a report needs an explicit request (generate/full report)")
    elif llm == "out_of_scope" and (signals["companies"] or signals["metric"] or signals["intent"]):
        route, decided_by = "lookup", "rules"
        adjustments.append("downgraded 'out_of_scope' to 'lookup': the question names a company, measure or topic")
    else:
        route, decided_by = llm, "llm"
    if route == "sector_trend" and decided_by == "llm" and not (sector_plan or {}).get("feasible"):
        # The LLM saw a sector question the parser didn't: without a buildable plan, the rules decide.
        adjustments.append(f"sector trend not possible ({'; '.join((sector_plan or {}).get('problems') or [])}): "
                           f"rules chose {rule!r}")
        route, decided_by = rule, "rules"
    if route == "trend" and not trend_plan.get("feasible"):
        fallback = "comparison" if len(signals["companies"]) >= 2 else "lookup"
        adjustments.append(f"trend not possible ({'; '.join(trend_plan.get('problems') or [])}): "
                           f"answered as a {fallback}")
        route = fallback
    return {"route": route, "flow": FLOW_OF[route], "decided_by": decided_by, "llm_route": llm,
            "rule_route": rule, "reason": str((raw or {}).get("reason") or "")[:300] if isinstance(raw, dict) else "",
            "signals": signals, "adjustments": adjustments}


def report_handoff(query: str, clarifier, *, ticker: str | None = None, year: int | None = None) -> dict[str, Any]:
    """What the report pipeline would need, and the message telling the user how to run it."""
    from scripts.agents.financial_clarifier import name
    slots = clarifier.parse(query)
    company = ticker.upper() if ticker else next(iter(slots.companies), None)
    available = (clarifier.directory or {}).get(company, []) if company else []
    fiscal_year = year or (slots.years[-1] if slots.years else (available[-1] if available else None))
    profile = clarifier.profile_for(company, fiscal_year) if company and fiscal_year else None
    target = f"{name(company)} FY{fiscal_year}" if company and fiscal_year else (name(company) if company else None)
    message = ("This asks for a full financial report. Reports come from the deterministic report pipeline "
               "(numbers parsed from the filing, reconciliations checked) and run only after you confirm in the chat")
    message += f": ask \"Generate a full financial report for {target}\" in the conversation." if target else "."
    if company and not profile:
        reports = clarifier.available_reports()
        message += (" No report profile exists for it yet" + (f"; profiles exist for {', '.join(reports)}." if reports
                                                               else ".") +
                    " You can still ask for specific figures, trends or explanations.")
    return {"action": "report", "ticker": company, "year": fiscal_year, "profile_available": bool(profile),
            "message": message}
