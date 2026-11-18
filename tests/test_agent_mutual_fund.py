"""Fund agent: the router's fund route, fixed SQL rows as cited evidence, holdings linked to their annual reports
(weighted growth in code), and no change to any other route when no fund data is loaded. No network or models."""
from __future__ import annotations

import sqlite3

import pytest

import scripts.agents.financial_prompting_agent as agent
from scripts.agents import agent_mutual_fund as fund_agent
from scripts.agents import financial_clarifier as fc
from scripts.agents import financial_router as router
from scripts.agents import tools_mutual_fund as fund_tools
from tests.data_mutual_fund import load_funds

DIRECTORY = {"NVDA": [2024, 2025], "GOOGL": [2024, 2025], "MSFT": [2024, 2025], "ABBV": [2024, 2025],
             "AMGN": [2024, 2025], "AAPL": [2024, 2025]}
# Annual reports filed by the holdings date (2026-06-30); the FY2026 rows were filed after it and must not be used.
FILED = {t: [("2025-02-20", 2024, f"edgar_ars_{t}_2024"), ("2026-02-20", 2025, f"edgar_ars_{t}_2025"),
             ("2027-02-20", 2026, f"edgar_ars_{t}_2026")] for t in DIRECTORY}
REVENUE = {"NVDA": ("130,497", "60,922"), "GOOGL": ("350,018", "307,394"), "MSFT": ("245,122", "211,915"),
           "ABBV": ("56,334", "54,318")}
TOP5 = "How did revenue of the Vanguard 500 Index Fund's top 5 holdings grow?"


def config(query, ticker=None):
    return agent.AgentConfig(query=query, ticker=ticker, year=None, source_type=None, top_k=4, dense_top_k=4,
                             bm25_top_k=4, retrieval_mode="hybrid", embedding_provider="hashing", hash_dimensions=16,
                             max_context_chars=8000, self_consistency_samples=1)


def statement(ticker, year):
    now, before = REVENUE[ticker]
    return ("Consolidated Statements of Income (in millions)\n"
            f"Years ended {year} {year - 1}\n"
            f"Total revenues $ {now} $ {before}\n"
            "Cost of revenues 17,452 16,123")


@pytest.fixture
def funds(tmp_path, monkeypatch):
    """Fund data loaded, the directory and manifests above, and a search that returns each company's statement."""
    folder = tmp_path / "mf"
    load_funds(folder)
    monkeypatch.setenv("RAG_MUTUAL_FUND_DIR", str(folder))
    monkeypatch.setattr(fc, "default_clarifier", lambda: fc.Clarifier(directory=DIRECTORY))
    monkeypatch.setattr(fund_tools, "manifest_filings", lambda: FILED)
    for key in ("RAG_FIN_VISUAL", "RAG_BACKEND", "RAG_SUPPORT_CHECK", "RAG_FUND_MAX_COMPANIES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")
    searches = []

    def fake(run_config):
        searches.append((run_config.ticker, run_config.year, run_config.query))
        if run_config.ticker not in REVENUE:
            return []
        chunk_id = f"annual_report_edgar_ars_{run_config.ticker}_{run_config.year}_p0050_vlm0"
        return [{"chunk_id": chunk_id, "ticker": run_config.ticker, "year": run_config.year, "page": 50,
                 "source_type": "annual_report_pdf", "text": statement(run_config.ticker, run_config.year),
                 "preview": "revenue", "metadata": {"chunk_id": chunk_id}}]

    monkeypatch.setattr(agent, "retrieve_evidence", fake)
    return folder, searches


def run(query, technique="router"):
    return agent.financial_graph(agent.MockLLM(), config(query), technique)


# ----------------------------------------------------------------------------- no fund data: nothing changes
def test_without_fund_data_there_is_no_fund_route(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_MUTUAL_FUND_DIR", str(tmp_path / "empty"))
    monkeypatch.setattr(fc, "default_clarifier", lambda: fc.Clarifier(directory=DIRECTORY))
    clarifier = fc.Clarifier(directory=DIRECTORY)
    for query in ("What are the top holdings of the Vanguard 500 Index Fund?", "Which funds hold the most NVIDIA?",
                  "What was NVIDIA's revenue in 2025?"):
        assert router.rule_route(query, clarifier)[0] != "fund"
        assert router.decide({"route": "fund"}, query, clarifier, {"feasible": False})["route"] != "fund"
    result = run("What are the top holdings of the Vanguard 500 Index Fund?", "fund_graph")
    assert result["final_status"] == "not_available" and "no mutual fund data is loaded" in result["answer"]


# ----------------------------------------------------------------------------- holdings questions (SQL rows)
def test_top_holdings_answer_cites_nport_rows(funds):
    result = run("What are the top holdings of the Vanguard 500 Index Fund?")
    assert result["route"]["route"] == "fund" and result["route"]["decided_by"] == "rules"
    assert result["agent_run"]["path"] == ["security_step", "plain_text_step", "route_step", "fund_lookup_step",
                                           "answer_step", "critic_step", "governance_step"]
    table = result["mutual_fund"]
    assert table["kind"] == "top_holdings" and table["period_end"] == "2026-06-30"
    assert [r["ticker"] for r in table["rows"][:4]] == ["NVDA", "GOOGL", "MSFT", "-"]
    assert table["rows"][1]["% of net assets"] == round(3.25 + 2.59, 4)
    assert all(h["source_type"] == "nport" for h in result["evidence"])
    assert result["citation_check"]["valid"] and all(c.startswith("nport_") for c in result["citation_check"]["cited_chunk_ids"])
    assert result["support_check"]["verdict"] == "supported"
    facts = next(h for h in result["evidence"] if h["chunk_id"].endswith("_fund"))
    assert "net assets USD 1,000,000.0 million" in facts["text"] and "filed 2026-08-27" in facts["text"]


def test_which_funds_hold_a_company(funds):
    result = run("Which funds hold the most NVIDIA?")
    table = result["mutual_fund"]
    assert table["kind"] == "holders" and table["ticker"] == "NVDA"
    assert [(r["fund"], r["% of net assets"]) for r in table["rows"]] == [("Vanguard 500 Index Fund", 7.5),
                                                                        ("Vanguard Total Stock Market Index Fund", 6.0)]


def test_exposure_and_sector_mix(funds):
    exposure = run("How much Microsoft does the Vanguard Total Stock Market Index Fund hold?")["mutual_fund"]
    assert exposure["kind"] == "exposure" and exposure["rows"] == [
        {"company": "Microsoft (MSFT)", "% of net assets": 4.0, "value (USD m)": 80000.0}]
    mix = run("What is the sector allocation of the Vanguard 500 Index Fund?")["mutual_fund"]
    assert mix["kind"] == "sector_mix" and mix["sector_mix"][-1]["sector"] == "Not in corpus"


def test_family_only_asks_which_fund(funds):
    result = run("What does the Vanguard fund hold?")
    assert result["final_status"] == "not_available"
    assert "which Vanguard fund" in result["answer"] and "Vanguard 500 Index Fund" in result["answer"]


# ----------------------------------------------------------------------------- linked to the annual reports
def test_weighted_growth_of_top_holdings_from_their_annual_reports(funds):
    folder, searches = funds
    result = run(TOP5)
    path = result["agent_run"]["path"]
    assert path[3:5] == ["fund_lookup_step", "fund_company_step"] and len(path) <= 10
    table = result["mutual_fund"]
    assert table["kind"] == "holdings_growth" and table["metric"] == "revenue"
    assert {c["ticker"]: c["year"] for c in table["companies"]} == {"NVDA": 2025, "GOOGL": 2025, "MSFT": 2025, "ABBV": 2025}
    assert {(t, y) for t, y, _ in searches} == {("NVDA", 2025), ("GOOGL", 2025), ("MSFT", 2025), ("ABBV", 2025)}  # no look-ahead
    assert [e["company"] for e in table["excluded"]] == ["Acme Widgets Ltd"]
    growth = {c["ticker"]: c["growth_pct"] for c in table["companies"]}
    expected = {t: (float(a.replace(",", "")) - float(b.replace(",", ""))) / float(b.replace(",", "")) * 100
                for t, (a, b) in REVENUE.items()}
    assert growth == pytest.approx(expected, abs=1e-3)
    weights = {"NVDA": 7.5, "GOOGL": 5.84, "MSFT": 4.3, "ABBV": 0.8}
    weighted = sum(weights[t] * expected[t] for t in weights) / sum(weights.values())
    assert table["weighted_growth_pct"] == pytest.approx(weighted, abs=1e-3)
    assert table["coverage_share"] == pytest.approx(sum(weights.values()) / (sum(weights.values()) + 1.0), abs=1e-4)
    assert any(c.get("label") == "weighted growth" for c in result["agent_run"]["calculations"])
    assert "low_fund_coverage" not in (result["human_review"] or {}).get("triggers", [])
    with sqlite3.connect(folder / "nport_mutual_fund.sqlite3") as db:            # verified figures are cached
        cached = db.execute("SELECT ticker, fiscal_year, value, method FROM company_figures ORDER BY 1, 2").fetchall()
    assert ("NVDA", 2024, 60922.0, "label") in cached and ("NVDA", 2025, 130497.0, "label") in cached
    again = run(TOP5)
    assert any(c.get("tool") == "company_figures" and c.get("cached") for c in again["agent_run"]["tool_calls"])
    assert again["mutual_fund"]["weighted_growth_pct"] == table["weighted_growth_pct"]


def test_low_coverage_goes_to_human_review(funds, monkeypatch):
    monkeypatch.setattr(fund_tools, "manifest_filings", lambda: {"ABBV": FILED["ABBV"]})
    monkeypatch.setattr(fc, "default_clarifier", lambda: fc.Clarifier(directory={"ABBV": [2024, 2025]}))
    result = run(TOP5)
    assert result["mutual_fund"]["coverage_share"] < fund_agent.MIN_COVERAGE
    assert "low_fund_coverage" in result["human_review"]["triggers"]


# ----------------------------------------------------------------------------- around the graph
def test_clarifier_passes_fund_questions_straight_to_the_agent(funds):
    clarifier = fc.Clarifier(directory=DIRECTORY)
    outcome = clarifier.step(fc.Conversation("c1"), "What are the top holdings of the Vanguard 500 Index Fund?")
    assert outcome["status"] == "passthrough" and outcome["route_hint"] == "fund"
    normal = clarifier.step(fc.Conversation("c2"), "What was the revenue?")
    assert normal["status"] == "needs_clarification"


def test_run_technique_keeps_a_fund_answer(funds):
    result = agent.run_technique("router", "mock", config("What are the top holdings of the Vanguard 500 Index Fund?"))
    assert result["final_status"] not in {"not_available", "insufficient_evidence", "not_grounded"}
    assert result["mutual_fund"]["kind"] == "top_holdings" and result["evidence"]
    summary = agent.summarize_result(result)
    assert summary["mutual_fund"]["fund"] == "Vanguard 500 Index Fund"


def test_plan_ignores_a_company_that_is_part_of_the_fund_name(funds):
    plan = fund_agent.fund_plan("What are the top holdings of the Vanguard 500 Index Fund?", fc.Clarifier(directory=DIRECTORY))
    assert plan["feasible"] and plan["kind"] == "top_holdings" and plan["n"] == 10
    assert fund_agent.top_n("top three holdings risks", "holdings_text") == 3


# ----------------------------------------------------------------------------- governance
def test_advice_is_refused_and_holdings_wording_is_not_advice(funds):
    advice = run("Should I buy the Vanguard 500 Index Fund?")
    assert advice["final_status"] == "refused" and "investment advice" in advice["answer"]
    assert advice["agent_run"]["path"][-2:] == ["critic_step", "governance_step"]
    holders = run("Which funds hold the most NVIDIA?")
    assert "investment_or_decision_support_language" not in (holders["human_review"] or {}).get("triggers", [])


def test_one_llm_figure_for_both_years_is_not_zero_growth():
    table = {"kind": "holdings_growth", "metric": "revenue", "excluded": [],
             "holdings": [{"pct_of_net_assets": 3.0}, {"pct_of_net_assets": 2.0}],
             "companies": [{"ticker": "JPM", "year": 2025, "weight": 3.0}, {"ticker": "ABBV", "year": 2025, "weight": 2.0}]}
    same = {"status": "ok", "value": 65816.0, "chunk_id": "c1", "method": "llm"}
    figures = {"JPM": {2024: {**same, "year": 2024}, 2025: {**same, "year": 2025}},
               "ABBV": {2024: {"status": "ok", "value": 100.0, "chunk_id": "a", "method": "label", "year": 2024},
                        2025: {"status": "ok", "value": 110.0, "chunk_id": "a", "method": "label", "year": 2025}}}
    fund_agent.growth(table, figures)
    assert table["companies"][0]["growth_pct"] is None and "same figure" in table["excluded"][0]["reason"]
    assert table["weighted_growth_pct"] == pytest.approx(10.0) and table["covered_weight"] == 2.0
