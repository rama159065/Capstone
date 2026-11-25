"""The "Thought for N s" dropdown at the top of each answer (like Databricks Genie's thinking steps): the agent's steps as
a timeline, a flowchart of the path it took, the sources, the guardrail and grounding checks, and the raw response.

Everything comes from the API response: ``agent_run`` (path, tool calls, calculations), the checks and the evidence.
The response has no time per step, so the dropdown shows the total time measured in the UI, the server's workflow time
(``trace``) and each search's own time. The render functions import Streamlit when they run (as ``ui_mutual_fund``
does), so tests can swap the module.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

from scripts.common.companies import NAMES as COMPANY_NAMES, fiscal_label

# (icon, title) of every node of financial_graph() in scripts/agents/financial_prompting_agent.py, plus the clarifier
# before the graph and the API's grounding policy after it.
STEP_INFO: dict[str, tuple[str, str]] = {
    "clarifier": (":material/help_center:", "Understood the question"),
    "security_step": (":material/shield:", "Input guardrails"),
    "plain_text_step": (":material/fact_check:", "Intent check"),
    "route_step": (":material/alt_route:", "Router"),
    "plan_step": (":material/checklist:", "Planner"),
    "retrieve_step": (":material/search:", "Searched the filings"),
    "evidence_guard_step": (":material/policy:", "Screened the evidence"),
    "trend_fetch_step": (":material/search:", "Searched each year's filing"),
    "trend_series_step": (":material/table_rows:", "Read the statement rows"),
    "sector_fetch_step": (":material/travel_explore:", "Searched each company's filings"),
    "sector_series_step": (":material/table_chart:", "Built the sector table"),
    "fund_lookup_step": (":material/account_balance:", "Looked up the fund's holdings (N-PORT)"),
    "fund_company_step": (":material/domain:", "Read the holdings' annual reports"),
    "action_fetch_step": (":material/search:", "Searched the cash-flow statements"),
    "action_series_step": (":material/calculate:", "Read the figures and computed the table"),
    "answer_step": (":material/edit_note:", "Wrote the answer"),
    "calculator_step": (":material/calculate:", "Ran the calculator"),
    "critic_step": (":material/rule:", "Critic checked citations and figures"),
    "governance_step": (":material/verified_user:", "Governance and human review"),
    "grounding": (":material/gpp_good:", "API grounding policy"),
}
ROUTE_NAMES = {
    "lookup": "Q&A agent (lookup)", "comparison": "Q&A agent (comparison)", "trend": "trend agent",
    "sector_trend": "sector trend agent", "fund": "mutual fund agent", "shareholder_returns": "shareholder-returns agent",
    "dividend_income": "dividend-income agent", "report": "report handoff", "out_of_scope": "refusal (out of scope)",
}
# The last, empty timeline step: what happened to the answer.
FINAL_STEPS = {
    "answered": (":material/task_alt:", "Answer released"),
    "evidence_only": (":material/format_quote:", "Cited excerpts shown instead of the draft"),
    "pending_human_review": (":material/rate_review:", "Sent for human review"),
    "not_available": (":material/search_off:", "Not in the filings, so no figure is given"),
    "insufficient_evidence": (":material/search_off:", "Not enough evidence to answer"),
    "not_grounded": (":material/gpp_maybe:", "Answer withheld"),
    "refused": (":material/block:", "Refused"),
    "report_handoff": (":material/description:", "Handed to the report pipeline"),
    "step_limit": (":material/timer_off:", "Stopped at the step limit"),
    "tool_error": (":material/error:", "A service failed or timed out"),
}
STATE_COLORS = {"pass": "green", "warn": "orange", "fail": "red", "off": "gray"}
STATE_ICONS = {"pass": ":material/check_circle:", "warn": ":material/warning:", "fail": ":material/cancel:",
               "off": ":material/remove_circle:"}

# Tool calls are logged in run order, so each step's calls are the run of tools that step can log.
SEARCH_TOOLS = frozenset({"text_search", "visual_search", "text_search+visual_search", "statement_search"})
READ_TOOLS = frozenset({"series_extract", "cash_flow_extract", "per_share_extract", "prior_year_check",
                        "prior_year_column", "company_figures"})
NODE_TOOLS: dict[str, frozenset[str]] = {
    "retrieve_step": SEARCH_TOOLS | {"coverage_lookup"},
    "trend_fetch_step": SEARCH_TOOLS,
    "sector_fetch_step": SEARCH_TOOLS,
    "action_fetch_step": SEARCH_TOOLS,
    "trend_series_step": READ_TOOLS | {"trend_calculator"},
    "sector_series_step": READ_TOOLS | SEARCH_TOOLS | {"sector_calculator"},
    "action_series_step": READ_TOOLS | SEARCH_TOOLS | {"returns_calculator", "dividend_calculator"},
    "fund_lookup_step": frozenset({"fund_lookup"}),
    "fund_company_step": READ_TOOLS | SEARCH_TOOLS | {"fund_calculator"},
    "calculator_step": frozenset({"calculator"}),
}
# The router logs its coverage check only for the routes that plan their own targets.
ROUTE_TOOLS = {"fund": frozenset({"fund_search"}), "trend": frozenset({"coverage_lookup"}),
               "sector_trend": frozenset({"coverage_lookup"}), "shareholder_returns": frozenset({"coverage_lookup"}),
               "dividend_income": frozenset({"coverage_lookup"})}
KNOWN_TOOLS = frozenset().union(*NODE_TOOLS.values(), *ROUTE_TOOLS.values())
FETCH_STEPS = ("retrieve_step", "trend_fetch_step", "sector_fetch_step", "action_fetch_step")


def plural(count: int, word: str, many: str | None = None) -> str:
    return f"{count} {word if count == 1 else many or word + 's'}"


def assign_tool_calls(path: list[str], calls: list[dict[str, Any]], route: str | None = None
                      ) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """Each path step's tool calls, and the calls no step could claim. A tool this UI doesn't know stays with the step
    that is logging calls when it appears."""
    groups, index = [], 0
    for node in path:
        allowed = ROUTE_TOOLS.get(route or "", frozenset()) if node == "route_step" else NODE_TOOLS.get(node, frozenset())
        taken = []
        while allowed and index < len(calls) and (calls[index].get("tool") in allowed
                                                  or calls[index].get("tool") not in KNOWN_TOOLS):
            taken.append(calls[index])
            index += 1
        groups.append(taken)
    return groups, calls[index:]


def workflow_seconds(result: dict[str, Any]) -> float | None:
    """The agent workflow's time on the server (the ``financial_workflow`` call in ``trace``)."""
    for event in result.get("trace") or []:
        if isinstance(event, dict) and event.get("node") == "financial_workflow" and event.get("elapsed_ms") is not None:
            return float(event["elapsed_ms"]) / 1000
    return None


def thinking_label(result: dict[str, Any], elapsed: float | None = None) -> str:
    """'Thought for 12.4 s · 8 steps · 4 searches · critic: supported'."""
    run = result.get("agent_run") or {}
    seconds = elapsed if elapsed is not None else workflow_seconds(result)
    parts = [f"Thought for {seconds:.1f} s" if seconds is not None else "How this answer was made"]
    if run.get("steps"):
        parts.append(plural(int(run["steps"]), "step"))
    searches = sum(1 for call in run.get("tool_calls") or [] if call.get("tool") in SEARCH_TOOLS)
    if searches:
        parts.append(plural(searches, "search", "searches"))
    verdict = (result.get("support_check") or {}).get("verdict")
    if verdict:
        parts.append(f"critic: {verdict.replace('_', ' ')}")
    return " · ".join(parts)


# ----------------------------------------------------------------------------- timeline steps

def timeline_steps(result: dict[str, Any], turn: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The run as timeline steps: icon, title, one-line summary, state (pass/warn/fail/off), detail lines, a table
    and the step's tool calls."""
    run = result.get("agent_run") or {}
    path = [str(node) for node in run.get("path") or []]
    groups, other = assign_tool_calls(path, list(run.get("tool_calls") or []),
                                      (result.get("route") or {}).get("route"))
    steps = []
    if turn and turn.get("resolved_query"):
        steps.append(clarifier_step(turn))
    seen: Counter[str] = Counter()
    for node, calls in zip(path, groups):
        seen[node] += 1
        steps.append(describe(node, seen[node], path.count(node), calls, result, path))
    if other:
        steps.append(step("other", "Other tool calls", plural(len(other), "call"), "off", calls=other,
                          icon=":material/build:"))
    if result.get("grounding"):
        steps.append(grounding_step(result))
    return steps


def step(node: str, title: str, summary: str = "", state: str = "pass", *, lines: list[str] | None = None,
         table: list[dict[str, Any]] | None = None, calls: list[dict[str, Any]] | None = None,
         icon: str | None = None) -> dict[str, Any]:
    return {"node": node, "icon": icon or STEP_INFO.get(node, (":material/radio_button_checked:", ""))[0],
            "title": title, "summary": summary, "state": state, "lines": lines or [], "table": table or [],
            "calls": calls or []}


def clarifier_step(turn: dict[str, Any]) -> dict[str, Any]:
    resolved, original = turn.get("resolved_query"), turn.get("original")
    lines = [f"Resolved question: *{resolved}*"]
    lines += [f"Assumption: {note}" for note in turn.get("assumptions") or []]
    if turn.get("clarification_turns"):
        lines.append(f"Clarifying questions asked so far: {turn['clarification_turns']}")
    summary = "filled in what was missing" if original and original != resolved else "company, year and measure are clear"
    return step("clarifier", STEP_INFO["clarifier"][1], summary, lines=lines)


def describe(node: str, k: int, n: int, calls: list[dict[str, Any]], result: dict[str, Any],
             path: list[str]) -> dict[str, Any]:
    """One path step, the k-th of n times the run passed through this node."""
    title = STEP_INFO.get(node, ("", node.removesuffix("_step").replace("_", " ").capitalize()))[1]
    describer = DESCRIBERS.get(node)
    if describer is None:
        return step(node, title, calls=calls)
    return describer(node, title, k, n, calls, result, path)


def _security(node, title, k, n, calls, result, path):
    security = result.get("security") or {}
    policy = security.get("policy") or {}
    labels = ", ".join(security.get("risk_labels") or [])
    if not security:
        return step(node, title, "guardrails turned off", "off")
    if security.get("decision") == "block":
        return step(node, title, f"blocked ({labels})", "fail")
    if policy.get("refuse"):
        return step(node, title, "refused by policy: " + ", ".join(policy.get("labels") or []), "fail",
                    lines=[policy["message"]] if policy.get("message") else [])
    if security.get("decision") == "review":
        return step(node, title, f"passed, flagged for review ({labels})", "warn")
    return step(node, title, "passed: no injection, code, secrets or personal data")


def _plain_text(node, title, k, n, calls, result, path):
    plain = result.get("plain_text")
    if not plain:
        return step(node, title, "skipped (guardrails turned off)", "off")
    lines = [f"Reason: {plain['reason']}"] if plain.get("reason") else []
    if not plain.get("allowed"):
        return step(node, title, "refused: not a plain financial-report question", "fail", lines=lines)
    return step(node, title, f"financial-report question, risk {plain.get('risk_level', 'low')}", lines=lines)


def _route(node, title, k, n, calls, result, path):
    route, plan = result.get("route") or {}, result.get("plan") or {}
    name = route.get("route")
    if route.get("decided_by") == "technique":
        lines = [f"The technique `{result.get('technique')}` chose this agent."]
    else:
        lines = [f"Decided by the **{route.get('decided_by', '?')}**: the LLM said `{route.get('llm_route')}`, "
                 f"the rules said `{route.get('rule_route')}`."]
    lines += [f"Adjustment: {note}" for note in route.get("adjustments") or []]
    if plan.get("tools"):
        lines.append("Plan: " + ", ".join(f"`{tool}`" for tool in plan["tools"]))
    lines += [f"Rule adjustment: {note}" for note in plan.get("adjustments") or []]
    state = "fail" if name == "out_of_scope" else "pass"
    return step(node, title, f"chose the {ROUTE_NAMES.get(name, name or 'Q&A agent')}", state, lines=lines, calls=calls)


def _plan(node, title, k, n, calls, result, path):
    plan = result.get("plan") or {}
    targets = [" ".join(str(v) for v in (t.get("ticker"), t.get("year")) if v) for t in plan.get("targets") or []
               if isinstance(t, dict)]
    lines = [f"Targets: {', '.join(t for t in targets if t) or 'all filings'}"]
    lines += [f"Rule adjustment: {note}" for note in plan.get("adjustments") or []]
    tools = ", ".join(f"`{tool}`" for tool in plan.get("tools") or []) or "none"
    return step(node, title, f"tools {tools} ({plan.get('planner', '?')} plan)", lines=lines, calls=calls)


def _fetch(node, title, k, n, calls, result, path):
    searches = [call for call in calls if call.get("tool") in SEARCH_TOOLS]
    new = sum(int(call.get("new") or 0) for call in searches)
    targets = list(dict.fromkeys(str(call["target"]) for call in searches if call.get("target")))
    shown = ", ".join(targets[:4]) + (f" +{len(targets) - 4} more" if len(targets) > 4 else "")
    summary = f"{plural(len(searches), 'search', 'searches')}, {plural(new, 'new passage')}"
    lines = []
    for call in calls:
        if call.get("tool") == "coverage_lookup":
            skipped = ", ".join(" ".join(str(v) for v in (s.get("ticker"), s.get("year")) if v) + f" ({s.get('reason')})"
                                for s in call.get("skipped") or [] if isinstance(s, dict))
            lines.append("Coverage check: " + (f"not in the index: {skipped}" if skipped else "every target is indexed"))
    if k > 1:
        title = f"Searched again (attempt {k})"
        lines.append("A broadened query, twice as deep, for the targets still missing.")
    return step(node, title, summary + (f" · {shown}" if shown else ""), lines=lines, calls=calls)


def _evidence_guard(node, title, k, n, calls, result, path):
    guard = result.get("evidence_guard") or {}
    check = (result.get("agent_run") or {}).get("evidence_check") or {}
    if not guard.get("safe", True):
        flagged = guard.get("flagged_chunks") or []
        return step(node, title, f"flagged {plural(len(flagged), 'passage')}: " + ", ".join(guard.get("risk_labels") or []),
                    "fail")
    if k < n:
        return step(node, title, "a company or year had no evidence, so the agent searched again", "warn")
    lines = [f"Still missing: {', '.join(check['missing'])}"] if check.get("missing") else []
    return step(node, title, "no instructions hidden in the retrieved text", lines=lines)


def _read(node, title, k, n, calls, result, path):
    reads = [call for call in calls if call.get("tool") in READ_TOOLS - {"prior_year_check", "prior_year_column"}]
    methods = Counter(call.get("method") for call in reads)
    lines = []
    if reads:
        lines.append(f"Read in code from the statement row: {methods.get('label', 0)} · copied by the model and "
                     f"checked: {methods.get('llm', 0)}")
    conflicts = [call for call in calls if call.get("tool") == "prior_year_check"]
    if conflicts:
        lines.append("Dropped because another filing contradicts it: " + ", ".join(str(c.get("target")) for c in conflicts))
    later = [call for call in calls if call.get("tool") == "prior_year_column"]
    if later:
        lines.append("Read from the next filing's column: " + ", ".join(str(c.get("target")) for c in later))
    retries = [call for call in calls if call.get("tool") in SEARCH_TOOLS]
    if retries:
        lines.append(f"Broadened searches for figures still missing: {len(retries)}")
    summary, state = TABLE_SUMMARIES[node](result, k < n)
    return step(node, title, summary, state, lines=lines, calls=calls)


def _trend_summary(result, retried):
    trend = result.get("trend") or {}
    points = trend.get("points") or []
    verified = sum(1 for p in points if p.get("status") == "ok")
    if retried:
        return "some years had no verified figure, so the agent searched again", "warn"
    cagr = f", CAGR {trend['cagr']:.2f}%" if trend.get("cagr") is not None else ""
    return f"{verified} of {plural(len(points), 'year')} verified{cagr}", "pass" if points and verified == len(points) else "warn"


def _sector_summary(result, retried):
    table = result.get("sector_trend") or {}
    panel = table.get("panel") or []
    cagr = f", total CAGR {table['cagr_total']:.2f}%" if table.get("cagr_total") is not None else ""
    return (f"{plural(len(panel), 'company', 'companies')} verified in every year{cagr}",
            "pass" if panel and not table.get("excluded") else "warn")


def _action_summary(result, retried):
    returns, income = result.get("shareholder_returns") or {}, result.get("dividend_income") or {}
    if returns:
        ranked = sum(1 for c in returns.get("companies") or [] if c.get("ranked"))
        return f"ranked {plural(ranked, 'company', 'companies')} by dividends plus buybacks", "pass" if ranked else "warn"
    if income:
        total = f"${income['total']:,.2f}" if income.get("total") is not None else "no total"
        return f"dividend income {total}", "pass" if income.get("calculated") else "warn"
    return "no figure could be verified", "warn"


def _fund_company_summary(result, retried):
    table = result.get("mutual_fund") or {}
    companies = table.get("companies") or []
    growth = f", weighted growth {table['weighted_growth_pct']:.2f}%" if table.get("weighted_growth_pct") is not None else ""
    return f"{plural(len(companies), 'holding')} linked to their reports{growth}", "pass"


TABLE_SUMMARIES = {"trend_series_step": _trend_summary, "sector_series_step": _sector_summary,
                   "action_series_step": _action_summary, "fund_company_step": _fund_company_summary}


def _fund_lookup(node, title, k, n, calls, result, path):
    table = result.get("mutual_fund") or {}
    rows = max([int(call.get("rows") or 0) for call in calls] or [0])
    as_of = f" as of {table['period_end']}" if table.get("period_end") else ""
    fund = table.get("fund") or table.get("company") or "the loaded funds"
    return step(node, title, f"{plural(rows, 'row')} for {fund}{as_of}", "pass" if rows else "warn", calls=calls)


def _answer(node, title, k, n, calls, result, path):
    if k > 1:
        return step(node, "Repaired the answer", "rewrote it with the critic's feedback", "warn")
    if (result.get("support_check") or {}).get("method") == "code_table":
        return step(node, title, "written in code from the verified table (no model wording)")
    return step(node, title, "drafted from the cited passages")


def _calculator(node, title, k, n, calls, result, path):
    expressions = sum(int(call.get("expressions") or 0) for call in calls)
    grounded = all(call.get("grounded") for call in calls)
    lines = [f"`{c.get('expression')}` = {c.get('value')}" + ("" if c.get("grounded") else " (operand not in the evidence)")
             for c in (result.get("agent_run") or {}).get("calculations") or [] if c.get("expression")][:10]
    return step(node, title, f"{plural(expressions, 'expression')}, operands "
                + ("all found in the evidence" if grounded else "not all found in the evidence"),
                "pass" if grounded else "warn", lines=lines, calls=calls)


def _critic(node, title, k, n, calls, result, path):
    support, check = result.get("support_check") or {}, result.get("citation_check") or {}
    if k < n:
        return step(node, title, "found a problem and asked for a repair", "warn")
    if not support and not check.get("cited_chunk_ids"):
        return step(node, title, "nothing to check (the run stopped earlier)", "off")
    claims = [c for c in support.get("claims") or [] if isinstance(c, dict)]
    numeric = support.get("numeric") or {}
    checked, ungrounded = int(numeric.get("checked") or 0), numeric.get("ungrounded") or []
    verdict = support.get("verdict") or "not run"
    parts = [f"citations {'valid' if check.get('valid') else 'not valid'}", verdict.replace("_", " ")]
    if claims:
        parts.append(f"{sum(1 for c in claims if c.get('supported') is True)}/{len(claims)} claims")
    if checked:
        parts.append(f"{checked - len(ungrounded)}/{checked} figures in the cited pages")
    lines = []
    if support.get("method") == "code_table":
        lines.append("The answer was written in code, so its figures were checked in code and no LLM judge was needed.")
    if ungrounded:
        lines.append("Figures not found in any cited passage: " + ", ".join(map(str, ungrounded)))
    if numeric.get("unverified"):
        lines.append("Figures only a page-reading model read: " + ", ".join(map(str, numeric["unverified"])))
    if check.get("missing_chunk_ids"):
        lines.append("Cited passages that were not retrieved: " + ", ".join(check["missing_chunk_ids"]))
    lines += [f"Not supported: {note}" for note in support.get("unsupported_claims") or []][:5]
    table = [{"claim": c.get("claim"), "supported": c.get("supported") is True,
              "reason": c.get("reason") or c.get("evidence") or ""} for c in claims]
    state = "pass" if verdict == "supported" and check.get("valid") else "warn" if verdict == "partially_supported" else "fail"
    return step(node, title, ", ".join(parts), state, lines=lines, table=table)


def _governance(node, title, k, n, calls, result, path):
    review, output = result.get("human_review") or {}, result.get("output_filter") or {}
    lines = []
    if output:
        lines.append(f"Output filter: **{output.get('decision')}**"
                     + (f" ({', '.join(output['risk_labels'])})" if output.get("risk_labels") else ""))
    if review.get("queue_id"):
        lines.append(f"Review queue item: `{review['queue_id']}`")
    if output.get("decision") == "block":
        return step(node, title, "the output filter blocked the answer", "fail", lines=lines)
    if review.get("required"):
        return step(node, title, "flagged for human review: " + ", ".join(review.get("triggers") or []), "warn",
                    lines=lines)
    if not output and review.get("status") == "not_required":
        return step(node, title, "skipped: no draft answer to review", "off", lines=lines)
    if review.get("triggers"):
        return step(node, title, f"would need review ({', '.join(review['triggers'])}), but human review is off",
                    "warn", lines=lines)
    return step(node, title, "output filter passed, no human review needed", lines=lines)


def grounding_step(result: dict[str, Any]) -> dict[str, Any]:
    grounding = result.get("grounding") or {}
    status, method = grounding.get("status"), grounding.get("method")
    lines = [grounding["reason"]] if grounding.get("reason") else []
    if method:
        lines.append({"llm_support_check": "The critic's support check verified every claim and figure.",
                      "source_excerpts": "The generated answer couldn't be verified, so cited excerpts are shown.",
                      "mock": "Test mode: the mock LLM's judge stands in for the critic."}.get(method, f"Method: {method}"))
    if status == "grounded":
        return step("grounding", STEP_INFO["grounding"][1], "citations match the retrieved passages", lines=lines)
    if status == "not_grounded":
        return step("grounding", STEP_INFO["grounding"][1], "answer withheld", "fail", lines=lines)
    return step("grounding", STEP_INFO["grounding"][1], "nothing to ground", "off", lines=lines)


DESCRIBERS = {
    "security_step": _security, "plain_text_step": _plain_text, "route_step": _route, "plan_step": _plan,
    **{name: _fetch for name in FETCH_STEPS}, "evidence_guard_step": _evidence_guard,
    "trend_series_step": _read, "sector_series_step": _read, "action_series_step": _read, "fund_company_step": _read,
    "fund_lookup_step": _fund_lookup, "answer_step": _answer, "calculator_step": _calculator, "critic_step": _critic,
    "governance_step": _governance,
}


# ----------------------------------------------------------------------------- flowchart

NODE_LABELS = {
    "START": "Question", "END": "Answer", "security_step": "Input guardrails", "plain_text_step": "Intent check",
    "route_step": "Router", "plan_step": "Planner", "retrieve_step": "Search filings",
    "evidence_guard_step": "Evidence guard", "trend_fetch_step": "Trend: search each year",
    "trend_series_step": "Trend: read rows", "sector_fetch_step": "Sector: search companies",
    "sector_series_step": "Sector: build table", "fund_lookup_step": "Fund: N-PORT lookup",
    "fund_company_step": "Fund: holdings' reports", "action_fetch_step": "Action: search cash flows",
    "action_series_step": "Action: compute table", "answer_step": "Write answer", "calculator_step": "Calculator",
    "critic_step": "Critic", "governance_step": "Governance",
}
# The edges financial_graph() adds; "stop" edges are drawn only when the run took them.
COMMON_EDGES = [
    ("START", "security_step", ""), ("security_step", "plain_text_step", ""), ("security_step", "critic_step", "stop"),
    ("plain_text_step", "critic_step", "stop"), ("retrieve_step", "evidence_guard_step", ""),
    ("evidence_guard_step", "retrieve_step", "retry"), ("evidence_guard_step", "answer_step", ""),
    ("evidence_guard_step", "critic_step", "stop"), ("answer_step", "calculator_step", "CALC"),
    ("answer_step", "critic_step", ""), ("calculator_step", "critic_step", ""), ("critic_step", "answer_step", "repair"),
    ("critic_step", "governance_step", ""), ("governance_step", "END", ""),
]
AGENTIC_EDGES = [("plain_text_step", "plan_step", ""), ("plan_step", "retrieve_step", "")]
ROUTER_EDGES = [
    ("plain_text_step", "route_step", ""), ("route_step", "critic_step", "stop"),
    ("route_step", "retrieve_step", "lookup / compare"),
    ("route_step", "trend_fetch_step", "trend"), ("trend_fetch_step", "trend_series_step", ""),
    ("trend_series_step", "trend_fetch_step", "retry"), ("trend_series_step", "answer_step", ""),
    ("trend_series_step", "critic_step", "stop"),
    ("route_step", "sector_fetch_step", "sector"), ("sector_fetch_step", "sector_series_step", ""),
    ("sector_series_step", "answer_step", ""), ("sector_series_step", "critic_step", "stop"),
    ("route_step", "fund_lookup_step", "fund"), ("fund_lookup_step", "fund_company_step", ""),
    ("fund_lookup_step", "answer_step", ""), ("fund_lookup_step", "critic_step", "stop"),
    ("fund_company_step", "answer_step", ""),
    ("route_step", "action_fetch_step", "returns / dividends"), ("action_fetch_step", "action_series_step", ""),
    ("action_series_step", "answer_step", ""), ("action_series_step", "critic_step", "stop"),
]


def node_id(node: str) -> str:
    return "n_" + node.removesuffix("_step").lower()


def agent_flowchart(path: list[str], technique: str | None = None) -> str:
    """Mermaid source of the agent graph with the path the run took highlighted (×2: a loop taken twice)."""
    agentic = technique == "agentic_graph" or "plan_step" in path
    edges = [*COMMON_EDGES, *(AGENTIC_EDGES if agentic else ROUTER_EDGES)]
    walk = ["START", *path, "END"] if path else []
    taken = Counter(zip(walk, walk[1:]))
    shown = [(a, b, label) for a, b, label in edges if label != "stop" or (a, b) in taken]
    nodes = list(dict.fromkeys(node for a, b, _ in shown for node in (a, b)))
    lines = ["flowchart TD"]
    for node in nodes:
        text = NODE_LABELS.get(node, node)
        lines.append(f'    {node_id(node)}(["{text}"])' if node in ("START", "END") else f'    {node_id(node)}["{text}"]')
    highlighted = []
    for index, (a, b, label) in enumerate(shown):
        count = taken.get((a, b), 0)
        text = " ".join(part for part in (label, f"×{count}" if count > 1 else "") if part)
        arrow = "-->" if count else "-.->"
        lines.append(f"    {node_id(a)} {arrow}{f'|{text}|' if text else ''} {node_id(b)}")
        if count:
            highlighted.append(str(index))
    visited = {"START", "END", *path} if path else set()
    lines.append("    classDef taken fill:#0F7C80,stroke:#0B5E61,color:#FFFFFF")
    lines.append("    classDef idle fill:transparent,stroke:#94A3B8,color:#94A3B8,stroke-dasharray:4 3")
    for name, members in (("taken", [n for n in nodes if n in visited]), ("idle", [n for n in nodes if n not in visited])):
        if members:
            lines.append(f"    class {','.join(node_id(n) for n in members)} {name}")
    if highlighted:
        lines.append(f"    linkStyle {','.join(highlighted)} stroke:#0FA3A3,stroke-width:3px")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- citations

# The citation IDs extract_citations() in scripts/agents/financial_prompting_agent.py recognises.
CHUNK_ID = re.compile(r"(?:annual_report|sec_facts|nport)_[A-Za-z0-9_.=-]+")
CITATION = re.compile(rf"\[\s*({CHUNK_ID.pattern}(?:\s*[,;]\s*{CHUNK_ID.pattern})*)\s*\]|{CHUNK_ID.pattern}")
FILING_PAGE = re.compile(r"_([A-Z][A-Z0-9.\-]{0,9})_((?:19|20)\d{2})_p(\d{2,5})")


def chunk_ids(text: str) -> list[str]:
    """The chunk IDs in a text, in order, each once (a sentence's full stop is not part of the ID)."""
    return list(dict.fromkeys(match.rstrip(".") for match in CHUNK_ID.findall(text or "")))


def cite_numbers(answer: str) -> tuple[str, list[str]]:
    """The answer with each chunk ID shown as a numbered badge, numbered in order of first appearance, and the cited
    IDs in that order: "37,676 [annual_report_..._JPM_2022_p0180_vlm0]" -> "37,676 :blue-badge[1]"."""
    order: dict[str, int] = {}

    def badge(chunk_id: str) -> str:
        return f":blue-badge[{order.setdefault(chunk_id, len(order) + 1)}]"

    def replace(match: re.Match) -> str:
        if match.group(1) is not None:
            return " ".join(badge(chunk_id) for chunk_id in chunk_ids(match.group(1)))
        raw = match.group(0)
        chunk_id = raw.rstrip(".")
        return badge(chunk_id) + raw[len(chunk_id):]

    return CITATION.sub(replace, answer or ""), list(order)


def source_label(chunk_id: str, row: dict[str, Any] | None = None) -> str:
    """'JPMorgan Chase · FY2022 annual report · p. 180' for a chunk ID (the fiscal year as the company names it)."""
    row = row or {}
    if chunk_id.startswith("nport_"):
        return "SEC N-PORT fund card" if chunk_id.endswith("_card") else "SEC N-PORT fund holdings (one holding row)"
    if chunk_id.startswith("sec_facts_"):
        ticker = chunk_id.split("_")[2] if chunk_id.count("_") >= 2 else ""
        return f"{COMPANY_NAMES.get(ticker, ticker)} · SEC company facts (XBRL)"
    match = FILING_PAGE.search(chunk_id)
    ticker = str(row.get("ticker") or (match.group(1) if match else "")).upper()
    filed = int(match.group(2)) if match else row.get("year")
    page = int(match.group(3)) if match else row.get("page")
    if not ticker:
        return chunk_id
    parts = [COMPANY_NAMES.get(ticker, ticker)]
    if filed:
        parts.append(f"FY{fiscal_label(ticker, int(filed))} annual report")
    if page:
        parts.append(f"p. {int(page)}")
    return " · ".join(parts)


def evidence_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in result.get("evidence") or result.get("evidence_preview") or [] if isinstance(row, dict)]


# ----------------------------------------------------------------------------- checks

def check_cards(result: dict[str, Any]) -> list[dict[str, Any]]:
    """One card per guardrail and grounding check: title, icon, state (pass/warn/fail/off), verdict and lines."""
    cards = []

    def card(title, icon, state, verdict, *lines):
        cards.append({"title": title, "icon": icon, "state": state, "verdict": verdict,
                      "lines": [line for line in lines if line]})

    security = result.get("security") or {}
    policy = security.get("policy") or {}
    labels = ", ".join(security.get("risk_labels") or [])
    if not security:
        card("Input guardrails", ":material/shield:", "off", "not run")
    elif security.get("decision") == "block":
        card("Input guardrails", ":material/shield:", "fail", "blocked", labels)
    elif policy.get("refuse"):
        card("Input guardrails", ":material/shield:", "fail", "refused by policy", ", ".join(policy.get("labels") or []))
    elif security.get("decision") == "review":
        card("Input guardrails", ":material/shield:", "warn", "review", labels)
    else:
        card("Input guardrails", ":material/shield:", "pass", "passed", "No injection, code, secrets or personal data.")

    plain = result.get("plain_text")
    if not plain:
        card("Intent check", ":material/fact_check:", "off", "not run")
    else:
        card("Intent check", ":material/fact_check:", "pass" if plain.get("allowed") else "fail",
             "financial question" if plain.get("allowed") else "refused",
             f"Risk level: {plain.get('risk_level', '?')}", plain.get("reason"))

    guard = result.get("evidence_guard") or {}
    if not guard:
        card("Evidence guard", ":material/policy:", "off", "not run")
    elif not guard.get("safe", True):
        card("Evidence guard", ":material/policy:", "fail", "flagged",
             f"{plural(len(guard.get('flagged_chunks') or []), 'passage')}: {', '.join(guard.get('risk_labels') or [])}")
    elif guard.get("dropped"):
        card("Evidence guard", ":material/policy:", "warn", "dropped flagged passages",
             f"{plural(len(guard['dropped']), 'passage')} never shown to the model.")
    else:
        card("Evidence guard", ":material/policy:", "pass", "safe", "No instructions hidden in the retrieved text.")

    support = result.get("support_check") or {}
    if not support:
        card("Critic support check", ":material/rule:", "off", "not run")
    else:
        verdict = support.get("verdict") or "unverified"
        claims = [c for c in support.get("claims") or [] if isinstance(c, dict)]
        numeric = support.get("numeric") or {}
        checked = int(numeric.get("checked") or 0)
        card("Critic support check", ":material/rule:",
             {"supported": "pass", "partially_supported": "warn"}.get(verdict, "fail"), verdict.replace("_", " "),
             f"Claims supported: {sum(1 for c in claims if c.get('supported') is True)}/{len(claims)}" if claims else "",
             f"Figures in the cited pages: {checked - len(numeric.get('ungrounded') or [])}/{checked}" if checked else "",
             "Mock judge (test mode)." if (result.get("grounding") or {}).get("method") == "mock" else "")

    check = result.get("citation_check") or {}
    if not check:
        card("Citation check", ":material/link:", "off", "not run")
    else:
        cited, missing = check.get("cited_chunk_ids") or [], check.get("missing_chunk_ids") or []
        card("Citation check", ":material/link:", "pass" if check.get("valid") else "fail",
             "valid" if check.get("valid") else "not valid", f"{plural(len(cited), 'passage')} cited.",
             f"Not retrieved: {', '.join(missing)}" if missing else "")

    grounding = result.get("grounding") or {}
    status = grounding.get("status")
    card("Grounding policy", ":material/gpp_good:",
         {"grounded": "pass", "not_grounded": "fail"}.get(status or "", "off"), (status or "not applied").replace("_", " "),
         grounding.get("reason"))

    output = result.get("output_filter") or {}
    if not output:
        card("Output filter", ":material/filter_alt:", "off", "not run")
    else:
        decision = output.get("decision")
        card("Output filter", ":material/filter_alt:", {"allow": "pass", "review": "warn"}.get(decision, "fail"),
             str(decision), ", ".join(output.get("risk_labels") or []))

    review = result.get("human_review") or {}
    if not review:
        card("Human review", ":material/rate_review:", "off", "not run")
    elif review.get("required"):
        card("Human review", ":material/rate_review:", "warn", "pending review", ", ".join(review.get("triggers") or []),
             f"Queue item: {review['queue_id']}" if review.get("queue_id") else "")
    else:
        card("Human review", ":material/rate_review:", "pass", "not needed")

    company = result.get("company_check")
    if company:
        card("Company scope", ":material/domain_verification:", "pass" if company.get("valid") else "fail",
             "matches the question" if company.get("valid") else "cites another company",
             ", ".join(company.get("foreign_chunk_ids") or []))
    coverage = result.get("coverage")
    if coverage:
        card("Coverage", ":material/search_off:", "warn", "not in the filings", coverage.get("message"))
    langsmith = result.get("langsmith") or {}
    card("Tracing", ":material/timeline:", "pass" if langsmith.get("enabled") else "off",
         "LangSmith" if langsmith.get("enabled") else "LangSmith off",
         f"Project: {langsmith['project']}" if langsmith.get("project") and langsmith.get("enabled") else "",
         f"Trace ID: {result['trace_id']}" if result.get("trace_id") else "")
    return cards


# ----------------------------------------------------------------------------- rendering

def call_rows(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool calls as table rows; lists and dicts become JSON text."""
    return [{key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
             for key, value in call.items()} for call in calls]


def colored(text: str, state: str) -> str:
    return {"warn": f":orange[{text}]", "fail": f":red[{text}]", "off": f":gray[{text}]"}.get(state, text)


def render_thinking(result: dict[str, Any], *, turn: dict[str, Any] | None = None, elapsed: float | None = None,
                    cited: list[str] | tuple[str, ...] = ()) -> None:
    """The dropdown's body: Steps, Agent graph, Sources, Checks and Raw tabs."""
    import streamlit as st                    # at call time: the app's streamlit (tests swap the module)
    rows = evidence_rows(result)
    count = len(cited) or len(rows)
    steps_tab, graph_tab, sources_tab, checks_tab, raw_tab = st.tabs([
        ":material/timeline: Steps", ":material/account_tree: Agent graph",
        f":material/library_books: Sources ({count})", ":material/verified_user: Checks", ":material/data_object: Raw"])
    with steps_tab:
        render_steps(result, turn, elapsed)
    with graph_tab:
        render_graph(result)
    with sources_tab:
        render_sources(result, list(cited), rows)
    with checks_tab:
        render_checks(result)
    with raw_tab:
        st.caption("The API response, as received.")
        st.code(json.dumps(turn or result, ensure_ascii=False, indent=2), language="json")


def render_steps(result: dict[str, Any], turn: dict[str, Any] | None, elapsed: float | None) -> None:
    import streamlit as st
    run = result.get("agent_run") or {}
    facts = []
    if elapsed is not None:
        facts.append(f"total {elapsed:.1f} s")
    server = workflow_seconds(result)
    if server is not None:
        facts.append(f"agent workflow {server:.1f} s on the server")
    if run.get("steps"):
        facts.append(f"{run['steps']} of {run.get('max_steps', '?')} graph steps used")
    if result.get("technique"):
        facts.append(f"technique `{result['technique']}`")
    if facts:
        st.caption(" · ".join(facts))
    if not run.get("path"):
        st.info("This run has no agent steps: a single-prompt baseline, a stop before the graph, or a service error.",
                icon=":material/info:")
    for item in timeline_steps(result, turn):
        label = f"**{item['title']}**" + (f" · {colored(item['summary'], item['state'])}" if item["summary"] else "")
        with st.expander(label, type="step", icon=item["icon"]):
            for line in item["lines"]:
                st.markdown(line)
            if item["table"]:
                st.dataframe(item["table"], hide_index=True, width="stretch")
            if item["calls"]:
                st.dataframe(call_rows(item["calls"]), hide_index=True, width="stretch")
            if not (item["lines"] or item["table"] or item["calls"]):
                st.caption("Nothing more to show for this step.")     # a step with no content would end the timeline
    icon, label = FINAL_STEPS.get(str(result.get("final_status")), (":material/flag:", "Done"))
    st.expander(f"**{label}**", type="step", icon=icon)


def render_graph(result: dict[str, Any]) -> None:
    import streamlit as st
    path = (result.get("agent_run") or {}).get("path") or []
    if not path:
        st.caption("No agent graph ran for this answer.")
        return
    st.mermaid_chart(agent_flowchart(path, result.get("technique")),
                     alt="The financial agent's LangGraph, with the path this answer took highlighted")
    st.caption("Filled boxes and bold arrows: the path this answer took. Dashed: branches it didn't take. "
               "×2: a loop taken twice (a retry or a repair).")


def render_sources(result: dict[str, Any], cited: list[str], rows: list[dict[str, Any]]) -> None:
    import streamlit as st
    by_id = {str(row.get("chunk_id")): row for row in rows}
    if cited:
        st.markdown("**Cited in the answer**")
        for number, chunk_id in enumerate(cited, start=1):
            row = by_id.get(chunk_id) or {}
            st.markdown(f":blue-badge[{number}] **{source_label(chunk_id, row)}**")
            st.caption(f"`{chunk_id}`" + (f" · {str(row['preview'])[:300]}" if row.get("preview") else ""))
    others = [row for row in rows if str(row.get("chunk_id")) not in set(cited)]
    if others:
        st.markdown("**Also retrieved**" if cited else "**Retrieved passages**")
        for row in others:
            chunk_id = str(row.get("chunk_id"))
            st.markdown(f"- {source_label(chunk_id, row)}")
            st.caption(f"`{chunk_id}`" + (f" · {str(row['preview'])[:300]}" if row.get("preview") else ""))
    shown = set(cited) | set(by_id)
    rest = [str(c) for c in result.get("evidence_chunk_ids") or [] if c and str(c) not in shown]
    if rest:
        st.caption(f"{plural(len(rest), 'more passage')} retrieved: " + ", ".join(f"`{c}`" for c in rest[:12])
                   + (" …" if len(rest) > 12 else ""))
    if not (cited or rows or rest):
        st.caption("No passages were retrieved for this answer.")


def render_checks(result: dict[str, Any]) -> None:
    import streamlit as st
    if result.get("explanation"):
        st.caption(result["explanation"])
    cards = check_cards(result)
    columns = st.columns(3)
    for index, item in enumerate(cards):
        with columns[index % 3].container(border=True):
            st.markdown(f"{item['icon']} **{item['title']}**")
            st.badge(item["verdict"], color=STATE_COLORS[item["state"]], icon=STATE_ICONS[item["state"]])
            for line in item["lines"]:
                st.caption(line)
