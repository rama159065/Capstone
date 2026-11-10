"""Dynamic routing of the financial agent graph: planner-chosen tools, per-company searches, the retrieval
retry, the calculator, the repair loop and the step budget. No network, models or index."""
from __future__ import annotations

import pytest

import scripts.agents.financial_prompting_agent as agent
from scripts.agents import financial_clarifier as fc
from scripts.agents import financial_tools as tools

DIRECTORY = {"COST": [2024, 2025], "WMT": [2025], "HD": [2025]}


def config(query, ticker=None, year=None, top_k=4):
    return agent.AgentConfig(query=query, ticker=ticker, year=year, source_type=None, top_k=top_k, dense_top_k=4,
                             bm25_top_k=4, retrieval_mode="hybrid", embedding_provider="hashing", hash_dimensions=16,
                             max_context_chars=6000, self_consistency_samples=1)


def hit(ticker, year, n, text=None):
    chunk_id = f"annual_report_edgar_ars_{ticker}_{year}_p00{n:02d}_text0"
    return {"chunk_id": chunk_id, "ticker": ticker, "year": year, "page": n, "source_type": "annual_report_pdf",
            "text": text or f"{ticker} total revenue was 254,453 million in fiscal {year}.", "preview": "revenue",
            "metadata": {"chunk_id": chunk_id}}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(fc, "default_clarifier", lambda: fc.Clarifier(directory=DIRECTORY))
    monkeypatch.delenv("RAG_FIN_VISUAL", raising=False)
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")


def searches(monkeypatch, respond):
    calls = []

    def fake(run_config):
        calls.append((run_config.ticker, run_config.year, run_config.query, run_config.top_k))
        return respond(run_config, len(calls))

    monkeypatch.setattr(agent, "retrieve_evidence", fake)
    return calls


def run(cfg):
    return agent.agentic_graph_prompt(agent.MockLLM(), cfg)


# ----------------------------------------------------------------------------- plan rules

def test_plan_keeps_known_tools_and_adds_what_the_question_needs():
    clarifier = fc.Clarifier(directory=DIRECTORY)
    plan = tools.normalize_plan({"tools": ["hybrid_financial_search", "web_browse", "chart_lookup"]},
                                "How did Costco and Walmart revenue grow in 2025?", clarifier=clarifier)
    assert plan["tools"] == ["coverage_lookup", "text_search", "calculator"]       # visual dropped: no ColPali served
    assert plan["targets"] == [{"ticker": "COST", "year": 2025}, {"ticker": "WMT", "year": 2025}]
    assert plan["planner"] == "llm" and plan["llm_tools"] == ["hybrid_financial_search", "web_browse", "chart_lookup"]
    assert any("web_browse" in a for a in plan["adjustments"]) and any("visual_search" in a for a in plan["adjustments"])


def test_plan_falls_back_and_visual_auto_follows_the_question(monkeypatch):
    clarifier = fc.Clarifier(directory=DIRECTORY)
    plain = tools.normalize_plan(None, "Summarise the risk factors", clarifier=clarifier)
    assert plain["tools"] == ["text_search"] and plain["planner"] == "fallback"
    monkeypatch.setenv("RAG_BACKEND", "financial_qdrant")
    monkeypatch.setenv("RAG_FIN_VISUAL", "auto")
    assert "visual_search" in tools.normalize_plan({}, "Show the revenue chart", clarifier=clarifier)["tools"]
    assert "visual_search" not in tools.normalize_plan({}, "Summarise the risk factors", clarifier=clarifier)["tools"]
    monkeypatch.setenv("RAG_FIN_VISUAL", "1")
    assert "visual_search" in tools.normalize_plan({}, "Summarise the risk factors", clarifier=clarifier)["tools"]


def test_request_ticker_and_year_win_over_the_text():
    clarifier = fc.Clarifier(directory=DIRECTORY)
    plan = tools.normalize_plan({}, "Compare Costco and Walmart revenue", ticker="HD", year=2025, clarifier=clarifier)
    assert plan["targets"] == [{"ticker": "HD", "year": 2025}]


# ----------------------------------------------------------------------------- calculator

def test_calculator_marks_operands_missing_from_the_evidence():
    evidence = [hit("COST", 2025, 1, "Net sales 254,453 and 237,710")]
    answer, records = tools.run_calculations("Up CALC[(254,453 - 237,710) / 237,710 * 100]% and CALC[5 * 999]",
                                             evidence)
    assert answer.startswith("Up 7.04%")
    assert records[0]["grounded"] and records[0]["value"] == pytest.approx(7.0434, abs=1e-3)
    assert not records[1]["grounded"] and records[1]["ungrounded_operands"] == [5.0, 999.0]


@pytest.mark.parametrize("expression", ["__import__('os').system('x')", "2 ** 999999", "abs(-1)", "a + 1"])
def test_calculator_rejects_anything_but_arithmetic(expression):
    with pytest.raises((ValueError, SyntaxError)):
        tools.safe_calculate(expression)


# ----------------------------------------------------------------------------- graph routing

def test_comparison_runs_one_filtered_search_per_company(monkeypatch):
    calls = searches(monkeypatch, lambda c, n: [hit(c.ticker, c.year, n)])
    result = run(config("Compare Costco and Walmart revenue in 2025"))
    assert [(t, y) for t, y, _, _ in calls] == [("COST", 2025), ("WMT", 2025)]
    run_info = result["agent_run"]
    assert [c["tool"] for c in run_info["tool_calls"]][:3] == ["coverage_lookup", "text_search", "text_search"]
    assert {h["target"] for h in result["evidence"]} == {"COST 2025", "WMT 2025"}
    assert run_info["retrieval_attempts"] == 1 and run_info["steps"] <= 10


def test_missing_company_triggers_one_broadened_retry(monkeypatch):
    calls = searches(monkeypatch, lambda c, n: [] if (c.ticker == "WMT" and n == 2) else [hit(c.ticker, c.year, n)])
    result = run(config("What was Costco and Walmart net income in 2025?"))
    run_info = result["agent_run"]
    assert run_info["retrieval_attempts"] == 2 and run_info["path"].count("retrieve_step") == 2
    retry = calls[-1]
    assert retry[0] == "WMT" and "consolidated statements" in retry[2] and retry[3] == 2 * calls[0][3]
    assert {h["target"] for h in result["evidence"]} == {"COST 2025", "WMT 2025"}
    assert run_info["evidence_check"]["missing"] == []


def test_budget_skips_the_retry_instead_of_failing(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "8")
    searches(monkeypatch, lambda c, n: [] if c.ticker == "WMT" else [hit(c.ticker, c.year, n)])
    result = run(config("What was Costco and Walmart net income in 2025?"))
    assert result["final_status"] != "step_limit"
    assert result["agent_run"]["retrieval_attempts"] == 1 and result["agent_run"]["steps"] <= 8
    assert result["agent_run"]["evidence_check"]["missing"] == ["WMT 2025"]


def test_company_outside_the_index_is_never_searched(monkeypatch):
    calls = searches(monkeypatch, lambda c, n: [hit(c.ticker, c.year, n)])
    result = run(config("What was Apple net income in 2024?"))
    assert calls == []
    assert result["final_status"] == "not_available"
    assert result["agent_run"]["tool_calls"][0]["skipped"][0]["reason"] == "company_not_in_index"


def test_growth_question_routes_through_the_calculator(monkeypatch):
    searches(monkeypatch, lambda c, n: [hit("COST", 2025, 1, "Net sales were 254,453 in 2025 and 237,710 in 2024.")])
    result = run(config("How much did Costco revenue grow in 2025?"))
    run_info = result["agent_run"]
    assert "calculator_step" in run_info["path"]
    assert run_info["calculations"][0]["grounded"] and "CALC[" not in result["answer"]
    assert "-6.58%" in result["answer"]            # mock: (2nd figure - 1st) / 1st, computed by the calculator


class RepairingLLM(agent.MockLLM):
    """Cites a chunk that wasn't retrieved, then follows the critic's feedback."""

    def invoke(self, messages):
        joined = "\n".join(getattr(m, "content", str(m)) for m in messages)
        if "Answer with citations." in joined and "Critic feedback" not in joined:
            return agent.MockResponse("Revenue rose [annual_report_FAKE_p0001_text0].")
        return super().invoke(messages)


def test_invalid_citation_is_repaired_once_with_the_critics_feedback(monkeypatch):
    searches(monkeypatch, lambda c, n: [hit("COST", 2025, 1)])
    result = agent.agentic_graph_prompt(RepairingLLM(), config("What was Costco revenue in 2025?"))
    assert result["agent_run"]["repairs"] == 1
    assert result["citation_check"]["valid"]
    assert result["agent_run"]["path"].count("critic_step") == 2


def test_default_cap_never_hits_the_recursion_limit(monkeypatch):
    searches(monkeypatch, lambda c, n: [])
    for query in ("What was Costco and Walmart net income in 2025?", "How much did Costco revenue grow in 2025?"):
        result = run(config(query))
        assert result["final_status"] == "insufficient_evidence" and result["agent_run"]["steps"] <= 10
