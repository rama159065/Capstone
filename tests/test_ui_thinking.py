"""The "Thought for N s" dropdown (scripts/ui/ui_thinking.py): numbered citations, source labels, the timeline steps
and the agent flowchart, on hand-made runs and on a mock router run through the API. No network, models or index."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import scripts.agents.financial_prompting_agent as agent
from scripts.agents import financial_clarifier as fc
from scripts.common.companies import NAMES
from scripts.ui import ui_thinking as ut

A = "annual_report_edgar_ars_JPM_2022_p0180_vlm0"
B = "annual_report_edgar_ars_JPM_2023_p0012_text1"


# ----------------------------------------------------------------------------- citations

def test_citations_become_badges_numbered_in_order_of_first_use():
    text, cited = ut.cite_numbers(f"Net income was 37,676 [{A}] and 49,552 [{B}; {A}]. See {B}.")
    assert text == "Net income was 37,676 :blue-badge[1] and 49,552 :blue-badge[2] :blue-badge[1]. See :blue-badge[2]."
    assert cited == [A, B]
    assert ut.cite_numbers("No citations here.") == ("No citations here.", [])


def test_citation_ids_are_the_ones_the_agent_validates():
    text = f"Revenue [{A}, sec_facts_AAPL_us-gaap_Revenues] and a fund row nport_000123456_17. Also {B}."
    assert ut.chunk_ids(text) == [A, "sec_facts_AAPL_us-gaap_Revenues", "nport_000123456_17", B]
    assert set(ut.chunk_ids(text)) == set(agent.extract_citations(text))


def test_source_labels_name_the_company_fiscal_year_and_page():
    assert ut.source_label(A) == "JPMorgan Chase · FY2022 annual report · p. 180"
    # Home Depot names its fiscal years by the year they start: the 2026 filing is fiscal 2025.
    assert ut.source_label("annual_report_edgar_ars_HD_2026_p0045_text0") == "Home Depot · FY2025 annual report · p. 45"
    assert ut.source_label("sec_facts_AAPL_us-gaap_Revenues") == f"{NAMES['AAPL']} · SEC company facts (XBRL)"
    assert ut.source_label("nport_000123456_17").startswith("SEC N-PORT fund holdings")
    assert ut.source_label("nport_000123456_card") == "SEC N-PORT fund card"
    row = {"ticker": "cost", "year": 2024, "page": 7}
    assert ut.source_label("chunk-7", row) == f"{NAMES['COST']} · FY2024 annual report · p. 7"
    assert ut.source_label("chunk-7") == "chunk-7"


# ----------------------------------------------------------------------------- tool calls and timeline steps

RETRY_PATH = ["security_step", "plain_text_step", "route_step", "trend_fetch_step", "trend_series_step",
              "trend_fetch_step", "trend_series_step", "answer_step", "critic_step", "governance_step"]
RETRY_CALLS = [
    {"tool": "coverage_lookup"},
    {"tool": "text_search", "target": "COST FY2023", "new": 3}, {"tool": "text_search", "target": "COST FY2024", "new": 2},
    {"tool": "series_extract", "method": "label"}, {"tool": "series_extract", "method": "llm"},
    {"tool": "trend_calculator"},
    {"tool": "text_search", "attempt": 2, "target": "COST FY2023", "new": 1},
    {"tool": "statement_search", "attempt": 2, "target": "COST FY2023", "new": 1},
    {"tool": "series_extract", "method": "label"}, {"tool": "trend_calculator"}, {"tool": "new_tool"},
]
RETRY_RUN = {
    "final_status": "answered", "technique": "router",
    "route": {"route": "trend", "decided_by": "rules", "llm_route": "lookup", "rule_route": "trend"},
    "agent_run": {"path": RETRY_PATH, "tool_calls": RETRY_CALLS, "steps": 10, "max_steps": 10},
    "trend": {"points": [{"year": 2023, "status": "ok"}, {"year": 2024, "status": "ok"}], "cagr": 5.0},
    "security": {"decision": "allow", "risk_labels": []}, "plain_text": {"allowed": True, "risk_level": "low"},
    "support_check": {"verdict": "supported", "method": "code_table", "numeric": {"checked": 4, "ungrounded": []}},
    "citation_check": {"valid": True, "cited_chunk_ids": [A]},
    "output_filter": {"decision": "allow"}, "human_review": {"required": False, "status": "auto_approved"},
    "grounding": {"status": "grounded"},
}


def test_tool_calls_go_to_the_step_that_made_them():
    groups, other = ut.assign_tool_calls(RETRY_PATH, RETRY_CALLS, "trend")
    assert [len(group) for group in groups] == [0, 0, 1, 2, 3, 2, 3, 0, 0, 0]
    assert groups[6][-1]["tool"] == "new_tool" and other == []      # an unknown tool stays with the step logging calls
    qa, rest = ut.assign_tool_calls(["route_step", "retrieve_step"], RETRY_CALLS[:3], "lookup")
    assert qa == [[], RETRY_CALLS[:3]] and rest == []                # Q&A logs its coverage check in the search step
    assert ut.assign_tool_calls(["answer_step"], [{"tool": "calculator"}]) == ([[]], [{"tool": "calculator"}])


def test_the_timeline_names_each_step_and_marks_the_retry():
    steps = ut.timeline_steps(RETRY_RUN)
    assert [s["title"] for s in steps] == [
        "Input guardrails", "Intent check", "Router", "Searched each year's filing", "Read the statement rows",
        "Searched again (attempt 2)", "Read the statement rows", "Wrote the answer",
        "Critic checked citations and figures", "Governance and human review", "API grounding policy"]
    assert [s["state"] for s in steps] == ["pass"] * 4 + ["warn"] + ["pass"] * 6
    assert steps[2]["summary"] == "chose the trend agent"
    assert steps[3]["summary"] == "2 searches, 5 new passages · COST FY2023, COST FY2024"
    assert steps[4]["lines"][0] == "Read in code from the statement row: 1 · copied by the model and checked: 1"
    assert steps[6]["summary"] == "2 of 2 years verified, CAGR 5.00%"
    assert steps[7]["summary"].startswith("written in code")
    assert steps[8]["summary"] == "citations valid, supported, 4/4 figures in the cited pages"
    clarified = ut.timeline_steps(RETRY_RUN, {"resolved_query": "Costco revenue 2023-2024", "original": "Costco revenue"})
    assert clarified[0]["title"] == "Understood the question" and clarified[0]["summary"] == "filled in what was missing"


def test_a_refusal_shows_the_failed_guard_and_skipped_checks():
    refused = {"final_status": "refused", "agent_run": {"path": ["security_step", "critic_step", "governance_step"]},
               "security": {"decision": "block", "risk_labels": ["prompt_injection"]},
               "human_review": {"required": False, "status": "not_required", "triggers": []}}
    assert [(s["state"], s["summary"]) for s in ut.timeline_steps(refused)] == [
        ("fail", "blocked (prompt_injection)"), ("off", "nothing to check (the run stopped earlier)"),
        ("off", "skipped: no draft answer to review")]
    cards = {card["title"]: card for card in ut.check_cards(refused)}
    assert cards["Input guardrails"]["state"] == "fail" and cards["Critic support check"]["state"] == "off"


def test_the_dropdown_label_sums_up_the_run():
    assert ut.thinking_label(RETRY_RUN, 12.44) == "Thought for 12.4 s · 10 steps · 4 searches · critic: supported"
    assert ut.thinking_label({"trace": [{"node": "financial_workflow", "elapsed_ms": 2500}]}) == "Thought for 2.5 s"
    assert ut.thinking_label({}) == "How this answer was made"


# ----------------------------------------------------------------------------- flowchart

def test_the_flowchart_highlights_the_path_and_counts_loops():
    chart = ut.agent_flowchart(RETRY_PATH, "router")
    lines = chart.splitlines()
    assert lines[0] == "flowchart TD"
    assert "    n_route -->|trend| n_trend_fetch" in lines
    assert "    n_trend_fetch -->|×2| n_trend_series" in lines                  # the retry ran the fetch twice
    assert "    n_trend_series -->|retry| n_trend_fetch" in lines
    assert "    n_route -.->|lookup / compare| n_retrieve" in lines             # a branch not taken
    assert not any("stop" in line for line in lines) and "n_plan" not in chart  # stop edges only when taken
    taken = next(line for line in lines if line.startswith("    class ") and line.endswith(" taken"))
    assert "n_trend_series" in taken and "n_retrieve" not in taken
    assert re.search(r"^    linkStyle [\d,]+ ", chart, re.M)

    agentic = ut.agent_flowchart(["security_step", "plain_text_step", "plan_step", "retrieve_step",
                                  "evidence_guard_step", "answer_step", "critic_step", "governance_step"],
                                 "agentic_graph")
    assert "n_plan" in agentic and "n_route" not in agentic and "n_trend_fetch" not in agentic
    stopped = ut.agent_flowchart(["security_step", "critic_step", "governance_step"], "router")
    assert "    n_security -->|stop| n_critic" in stopped.splitlines()


def test_steps_and_flowchart_cover_every_node_and_edge_of_the_graph():
    source = Path(agent.__file__).read_text(encoding="utf-8")
    nodes = set(re.findall(r'"([a-z_]+_step)"', source))
    assert len(nodes) >= 18 and nodes <= set(ut.STEP_INFO) and nodes <= set(ut.NODE_LABELS)
    edges = set(re.findall(r'add_edge\("(\w+)", "(\w+)"\)', source))
    for start, targets in re.findall(r'add_conditional_edges\("(\w+)", \w+,\s*\{([^}]*)\}', source):
        edges |= {(start, target) for target in re.findall(r':\s*"(\w+)"', targets)}
    drawn = {(a, b) for a, b, _ in [*ut.COMMON_EDGES, *ut.AGENTIC_EDGES, *ut.ROUTER_EDGES]}
    assert len(edges) > 20 and edges <= drawn, edges - drawn


# ----------------------------------------------------------------------------- a mock router run through the API

DIRECTORY = {"COST": [2022, 2023, 2024, 2025], "WMT": [2025]}
REVENUE = {2021: "195,929", 2022: "226,954", 2023: "242,290", 2024: "254,453", 2025: "275,235"}


def row_hit(ticker, year, page=41):
    """That year's income statement: the row the trend agent reads in code."""
    chunk_id = f"annual_report_edgar_ars_{ticker}_{year}_p00{page}_text0"
    text = (f"CONSOLIDATED STATEMENTS OF INCOME\n(in millions)\n{year} {year - 1}\n"
            f"Net sales 1,000 900\nTotal revenue {REVENUE[year]} {REVENUE[year - 1]}")
    return {"chunk_id": chunk_id, "ticker": ticker, "year": year, "page": page, "source_type": "annual_report_pdf",
            "text": text, "preview": "revenue", "metadata": {"chunk_id": chunk_id}}


@pytest.fixture
def mock_trend_run(monkeypatch):
    monkeypatch.setattr(fc, "default_clarifier", lambda: fc.Clarifier(directory=DIRECTORY))
    for name in ("RAG_FIN_VISUAL", "RAG_BACKEND", "RAG_SUPPORT_CHECK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")
    monkeypatch.setattr(agent, "retrieve_evidence", lambda run_config: [row_hit(run_config.ticker, run_config.year)])
    import scripts.api.api_app as api_app
    response = TestClient(api_app.app).post("/chat/financial_report_qa", json={
        "query": "How did Costco revenue change over the years?", "technique": "router", "llm_provider": "mock"})
    assert response.status_code == 200
    return response.json()


def test_a_mock_router_run_gives_every_node_a_step_and_a_box(mock_trend_run):
    result = mock_trend_run
    run = result["agent_run"]
    path = run["path"]
    groups, other = ut.assign_tool_calls(path, run["tool_calls"], result["route"]["route"])
    assert other == []
    assert [call["tool"] for call in groups[path.index("trend_fetch_step")]] == ["text_search"] * 4
    assert [call["tool"] for call in groups[path.index("trend_series_step")]].count("series_extract") == 4

    steps = ut.timeline_steps(result)
    assert [s["node"] for s in steps] == [*path, "grounding"]
    assert all(s["state"] == "pass" for s in steps), [(s["node"], s["state"], s["summary"]) for s in steps]
    assert ut.thinking_label(result, 1.0) == f"Thought for 1.0 s · {run['steps']} steps · 4 searches · critic: supported"
    assert ut.workflow_seconds(result) is not None                    # the server's time, from the response's trace

    text, cited = ut.cite_numbers(result["answer"])
    assert cited and not ut.chunk_ids(text) and set(cited) <= set(result["evidence_chunk_ids"])
    chart = ut.agent_flowchart(path, result["technique"])
    assert all(ut.node_id(node) in chart for node in path)
    cards = {card["title"]: card["state"] for card in ut.check_cards(result)}
    assert cards["Critic support check"] == "pass" and cards["Citation check"] == "pass"
