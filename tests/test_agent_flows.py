"""The router (supervisor) and the multi-year trend agent. No network, models or index."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import scripts.agents.financial_prompting_agent as agent
from scripts.agents import financial_clarifier as fc
from scripts.agents import financial_router as router
from scripts.agents import financial_trend as trend

DIRECTORY = {"COST": [2022, 2023, 2024, 2025], "WMT": [2025], "HD": [2025]}
REVENUE = {2022: "226,954", 2023: "242,290", 2024: "254,453", 2025: "275,235"}


def config(query, ticker=None, year=None, top_k=4):
    return agent.AgentConfig(query=query, ticker=ticker, year=year, source_type=None, top_k=top_k, dense_top_k=4,
                             bm25_top_k=4, retrieval_mode="hybrid", embedding_provider="hashing", hash_dimensions=16,
                             max_context_chars=6000, self_consistency_samples=1)


def hit(ticker, year, text=None, page=40):
    chunk_id = f"annual_report_edgar_ars_{ticker}_{year}_p00{page}_text0"
    return {"chunk_id": chunk_id, "ticker": ticker, "year": year, "page": page, "source_type": "annual_report_pdf",
            "text": text or f"Total revenue was {REVENUE.get(year, '100,000')} million in fiscal {year}.",
            "preview": "revenue", "metadata": {"chunk_id": chunk_id}}


def row_hit(ticker, year, page=41):
    """That year's income statement: the row the trend agent reads in code."""
    return hit(ticker, year, page=page, text=(
        "CONSOLIDATED STATEMENTS OF INCOME\n(in millions)\n" + f"{year} {year - 1}\n"
        f"Net sales 1,000 900\nTotal revenue {REVENUE[year]} {REVENUE.get(year - 1, '195,929')}"))


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(fc, "default_clarifier", lambda: fc.Clarifier(directory=DIRECTORY))
    monkeypatch.delenv("RAG_FIN_VISUAL", raising=False)
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    monkeypatch.delenv("RAG_SUPPORT_CHECK", raising=False)
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")


def searches(monkeypatch, respond=lambda c, n: [hit(c.ticker, c.year)]):
    calls = []

    def fake(run_config):
        calls.append((run_config.ticker, run_config.year, run_config.query, run_config.top_k))
        return respond(run_config, len(calls))

    monkeypatch.setattr(agent, "retrieve_evidence", fake)
    return calls


class ScriptedLLM(agent.MockLLM):
    """The mock LLM, with chosen replies for the router and the series extractor."""

    def __init__(self, route=None, series=None, answer=None):
        self.route, self.series, self.answer = route, series, answer

    def invoke(self, messages):
        joined = "\n".join(getattr(m, "content", str(m)) for m in messages)
        if self.route is not None and "Route JSON" in joined:
            return agent.MockResponse(json.dumps(self.route))
        if self.series is not None and "Series JSON" in joined:
            reply = self.series(joined)
            if reply is not None:
                return agent.MockResponse(json.dumps(reply))
        if self.answer is not None and "trend agent" in joined:
            return agent.MockResponse(self.answer(joined))
        return super().invoke(messages)


# ----------------------------------------------------------------------------- router rules

def clarifier():
    return fc.Clarifier(directory=DIRECTORY)


def plan_for(query, ticker=None):
    return trend.trend_plan(query, ticker, None, clarifier())


@pytest.mark.parametrize("llm_route, query, expected, decided_by", [
    ("lookup", "How did Costco revenue change over the years?", "trend", "rules"),
    ("lookup", "Compare Costco and Walmart revenue in 2025", "comparison", "rules"),
    ("lookup", "Generate a full financial report for Costco 2024", "report", "rules"),
    ("report", "What was Costco revenue in 2024?", "lookup", "rules"),
    ("out_of_scope", "What was Costco revenue in 2024?", "lookup", "rules"),
    ("out_of_scope", "Tell me a fun fact about penguins", "out_of_scope", "llm"),
    ("trend", "How has Costco's revenue developed?", "trend", "llm"),       # wording the rules don't know
    ("banana", "What was Costco revenue in 2024?", "lookup", "rules"),
    ("trend", "What was Costco revenue in 2024?", "lookup", "llm"),         # one year: no trend possible
])
def test_router_validates_the_llm_route_against_the_question(llm_route, query, expected, decided_by):
    decision = router.decide({"route": llm_route}, query, clarifier(), plan_for(query))
    assert decision["route"] == expected and decision["decided_by"] == decided_by
    assert decision["flow"] == router.FLOW_OF[expected]
    if decision["route"] != llm_route:
        assert decision["adjustments"]


def test_trend_route_falls_back_when_no_trend_can_be_built():
    query = "How did Walmart revenue change over the years?"                 # WMT has one indexed year
    decision = router.decide({"route": "trend"}, query, clarifier(), plan_for(query))
    assert decision["route"] == "lookup" and "fewer than two" in decision["adjustments"][-1]


# ----------------------------------------------------------------------------- trend plan and series

def test_trend_plan_uses_indexed_years_and_skips_the_rest():
    every = plan_for("How did Costco revenue change over time?")
    assert every["feasible"] and every["years"] == [2022, 2023, 2024, 2025] and every["metric"] == "revenue"
    assert [t["year"] for t in every["targets"]] == [2022, 2023, 2024, 2025]
    asked = plan_for("Costco revenue from 2019 to 2024")
    assert asked["years"] == [2022, 2023, 2024] and asked["skipped_years"] == [2019, 2020, 2021]
    capped = trend.trend_plan("Costco revenue trend", None, None, clarifier(), max_years=2)
    assert capped["years"] == [2024, 2025]
    assert not plan_for("Costco and Walmart revenue trend")["feasible"]
    assert not plan_for("How did Costco change over time?")["feasible"]           # no measure


def test_a_series_point_must_be_stated_by_that_years_evidence():
    hits = [hit("COST", 2024)]
    chunk = hits[0]["chunk_id"]
    ok = trend.validate_point({"value": "$254,453", "unit": "millions", "chunk_id": chunk}, hits, 2024)
    assert ok["status"] == "ok" and ok["value"] == 254453.0 and ok["page"] == 40
    assert trend.validate_point({"value": "254,999", "chunk_id": chunk}, hits, 2024)["status"] == "not_in_evidence"
    assert trend.validate_point({"value": "254,453", "chunk_id": "other"}, hits, 2024)["status"] == "wrong_chunk"
    assert trend.validate_point({"value": None}, hits, 2024)["status"] == "not_found"
    assert trend.validate_point("garbage", hits, 2024)["status"] == "not_found"
    assert trend.parse_figure("(1,234)") == -1234.0


def test_changes_and_cagr_are_computed_in_code():
    plan = plan_for("Costco revenue trend")
    points = [{"year": y, "value": float(REVENUE[y].replace(",", "")), "value_text": REVENUE[y], "unit": "millions",
               "chunk_id": f"c{y}", "status": "ok"} for y in (2022, 2023, 2024)]
    points.append({"year": 2025, "value": None, "status": "not_found", "unit": ""})
    summary = trend.summarize(plan, points)
    assert summary["verified_points"] == 3 and summary["missing"] == ["COST 2025"]
    assert summary["changes"][0]["pct_change"] == pytest.approx(6.7573, abs=1e-3)
    assert summary["cagr"] == pytest.approx(((254453 / 226954) ** 0.5 - 1) * 100, abs=1e-3)
    assert {c["kind"] for c in summary["calculations"]} == {"change", "pct_change", "cagr"}
    assert all(c["grounded"] for c in summary["calculations"])
    mixed = trend.summarize(plan, [{**points[0]}, {**points[1], "unit": "billions"}])
    assert mixed["changes"][0]["status"] == "unit_mismatch" and mixed["cagr"] is None


# ----------------------------------------------------------------------------- graphs

def test_router_sends_a_trend_question_to_the_trend_agent(monkeypatch):
    calls = searches(monkeypatch, lambda c, n: [row_hit(c.ticker, c.year)])
    result = agent.router_graph_prompt(agent.MockLLM(), config("How did Costco revenue change over the years?"))
    assert result["technique"] == "router" and result["route"]["route"] == "trend"
    assert [(t, y) for t, y, _, _ in calls] == [("COST", 2022), ("COST", 2023), ("COST", 2024), ("COST", 2025)]
    assert result["agent_run"]["path"] == ["security_step", "plain_text_step", "route_step", "trend_fetch_step",
                                           "trend_series_step", "answer_step", "critic_step", "governance_step"]
    series = result["trend"]
    assert [p["status"] for p in series["points"]] == ["ok"] * 4 and series["cagr"] == pytest.approx(6.6404, abs=1e-3)
    assert [c["tool"] for c in result["agent_run"]["tool_calls"]].count("series_extract") == 4
    assert result["final_status"] == "answered" and result["support_check"]["verdict"] == "supported"


def test_router_keeps_lookups_and_comparisons_on_the_qa_agent(monkeypatch):
    searches(monkeypatch)
    result = agent.router_graph_prompt(agent.MockLLM(), config("Compare Costco and Walmart revenue in 2025"))
    assert result["route"]["route"] == "comparison" and "route_step" in result["agent_run"]["path"]
    assert "plan_step" not in result["agent_run"]["path"]                     # the route step planned it
    assert {h["target"] for h in result["evidence"]} == {"COST 2025", "WMT 2025"}
    assert result["agent_run"]["steps"] <= 10


def test_trend_retries_a_year_whose_figure_was_not_verified(monkeypatch):
    # FY2023's first search finds no statement row; the broadened one does. No LLM reads a figure.
    calls = searches(monkeypatch, lambda c, n: [hit(c.ticker, c.year, "Costco opened warehouses.")]
                     if c.year == 2023 and "annual report" not in c.query else [row_hit(c.ticker, c.year)])
    result = agent.trend_graph_prompt(ScriptedLLM(series=lambda joined: {"value": None}),
                                      config("Costco revenue trend 2022-2024"))
    run_info = result["agent_run"]
    assert run_info["retrieval_attempts"] == 2 and run_info["path"].count("trend_fetch_step") == 2
    retry = calls[-1]
    assert retry[:2] == ("COST", 2023) and calls[-2][:2] == ("COST", 2023)       # other words, then the statement's rows
    assert "annual report" in calls[-2][2] and calls[-2][3] == 2 * calls[0][3] and "cost of sales" in retry[2]
    assert [p["status"] for p in result["trend"]["points"]] == ["ok", "ok", "ok"]
    assert run_info["steps"] <= 10


def test_trend_budget_skips_the_retry_and_names_the_missing_year(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "8")
    searches(monkeypatch)
    llm = ScriptedLLM(series=lambda joined: {"value": None} if "Fiscal year: 2023" in joined else None)
    result = agent.trend_graph_prompt(llm, config("Costco revenue trend 2022-2024"))
    assert result["final_status"] != "step_limit" and result["agent_run"]["steps"] <= 8
    assert result["trend"]["missing"] == ["COST 2023"] and result["trend"]["verified_points"] == 2


def test_trend_answer_is_written_in_code_with_every_change_and_the_cagr(monkeypatch):
    searches(monkeypatch, lambda c, n: [row_hit(c.ticker, c.year)])
    llm = ScriptedLLM(answer=lambda joined: pytest.fail("no LLM writes a trend answer"))
    result = agent.trend_graph_prompt(llm, config("Costco revenue trend 2022-2024"))
    for text in ("226,954", "242,290", "254,453", "+6.76%", "+5.02%", "5.89%"):
        assert text in result["answer"]
    support = result["support_check"]
    assert support["method"] == "code_table" and support["verdict"] == "supported" and support["numeric"]["grounded"]


@pytest.mark.parametrize("text", ["from fiscal 2023 to fiscal 2026", "FY2023-FY2026", "between fiscal 2023 and 2026",
                                  "2023 through fiscal year 2026", "FY2023 to FY2026"])
def test_year_spans_include_the_years_between(text):
    assert fc.Clarifier.years_in(f"How did Walmart revenue trend {text}?") == [2023, 2024, 2025, 2026]


def test_a_trend_of_a_segment_is_not_answered_with_the_company_total(monkeypatch):
    searches(monkeypatch)
    result = agent.router_graph_prompt(agent.MockLLM(), config("How did Costco membership fee revenue trend from 2022 to 2024?"))
    assert result["route"]["route"] != "trend" and "membership fee" in str(result["route"]["adjustments"])


def test_a_year_another_filing_contradicts_is_not_used():
    from scripts.agents import financial_sector_trend as sector
    points = {2023: {"year": 2023, "status": "ok", "value": 29241.0, "unit": "millions", "consolidated": True,
                     "columns": {2023: "29,241", 2022: "27,114"}},
              2024: {"year": 2024, "status": "ok", "value": 20881.0, "unit": "millions", "consolidated": False,
                     "columns": {2024: "20,881", 2023: "18,610"}}}
    assert sector.prior_conflicts(points) == {2024}                    # Capital One's Domestic Card row
    points[2024] = {**points[2024], "value": 31208.0, "consolidated": True, "columns": {2024: "31,208", 2023: "29,241"}}
    assert sector.prior_conflicts(points) == set()


def test_trend_graph_answers_a_single_year_question_as_a_lookup(monkeypatch):
    searches(monkeypatch)
    result = agent.trend_graph_prompt(agent.MockLLM(), config("What was Costco revenue in 2024?"))
    assert result["route"]["route"] == "lookup" and result["route"]["decided_by"] == "technique"
    assert "retrieve_step" in result["agent_run"]["path"] and not result.get("trend")


def test_report_request_is_handed_off_without_searching(monkeypatch):
    calls = searches(monkeypatch)
    result = agent.run_technique("router", "mock", config("Generate a full financial report for Costco 2024"))
    assert calls == [] and result["final_status"] == "report_handoff"
    assert result["handoff"]["ticker"] == "COST" and result["handoff"]["year"] == 2024
    assert "conversation" in result["answer"]
    assert result["agent_run"]["path"][-2:] == ["critic_step", "governance_step"]


def test_out_of_scope_is_refused_only_when_the_question_names_nothing(monkeypatch):
    calls = searches(monkeypatch)
    llm = ScriptedLLM(route={"route": "out_of_scope"})
    refused = agent.router_graph_prompt(llm, config("What is the main financial risk of owning a dog?"))
    assert refused["final_status"] == "refused" and calls == []
    kept = agent.router_graph_prompt(llm, config("What was Costco net income in 2025?"))
    assert kept["route"]["route"] == "lookup" and kept["final_status"] != "refused"


# ----------------------------------------------------------------------------- API

def test_api_runs_the_router_and_returns_route_and_trend(monkeypatch):
    import scripts.api.api_app as api_app
    searches(monkeypatch)
    client = TestClient(api_app.app)
    response = client.post("/chat/financial_report_qa", json={
        "query": "How did Costco revenue change over the years?", "technique": "router", "llm_provider": "mock"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["technique"] == "router" and payload["route"]["route"] == "trend"
    assert [row["year"] for row in payload["trend"]["chart"]] == [2022, 2023, 2024, 2025]
    assert payload["grounding"]["status"] == "grounded"

    handoff = client.post("/chat/financial_report_qa", json={
        "query": "Generate a full financial report for Costco 2024", "technique": "router", "llm_provider": "mock"}).json()
    assert handoff["final_status"] == "report_handoff" and handoff["grounding"]["status"] == "not_applicable"
    assert handoff["handoff"]["ticker"] == "COST" and handoff["answer"]
