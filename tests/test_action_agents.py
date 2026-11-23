"""Shareholder-returns and dividend-income agents: routing, plans, figures read in code, the action, the graph."""
from __future__ import annotations

import pytest

import scripts.agents.financial_prompting_agent as agent
from scripts.agents import financial_clarifier as fc
from scripts.agents import financial_dividend_income as income
from scripts.agents import financial_router as router
from scripts.agents import financial_shareholder_returns as returns

DIRECTORY = {"JPM": [2023, 2024], "BAC": [2023, 2024], "WFC": [2024], "MTB": [2023], "KO": [2023, 2024],
             "MSFT": [2024]}
CASH_FLOW = {  # dividends paid, repurchases (USD millions, as the statements print them)
    "JPM": ("(14,783)", "(18,830)"),
    "BAC": ("(9,503)", "(13,104)"),
    "WFC": ("(5,133)", "(19,448)"),
}
RETURNS_QUERY = "Which bank returned the most to shareholders in 2024?"
INCOME_QUERY = "I hold 100 shares of Coca-Cola and 50 shares of JPMorgan. How much dividend income was that in 2024?"


def config(query, ticker=None, year=None, sector_filter=None):
    return agent.AgentConfig(query=query, ticker=ticker, year=year, source_type=None, top_k=4, dense_top_k=4,
                             bm25_top_k=4, retrieval_mode="hybrid", embedding_provider="hashing", hash_dimensions=16,
                             max_context_chars=6000, self_consistency_samples=1, sector=sector_filter)


def cash_flow(ticker):
    dividends, buybacks = CASH_FLOW[ticker]
    return ("Consolidated Statements of Cash Flows (in millions)\nYear ended December 31 2024 2023 2022\n"
            "Financing activities\n"
            f"Dividends paid {dividends} (13,000) (12,000)\n"
            f"Treasury stock repurchased {buybacks} (9,000) (8,000)\n"
            "Net cash used in financing activities (40,000) (30,000) (20,000)")


def per_share(ticker):
    dps = {"KO": "1.94", "JPM": "4.80"}[ticker]
    return ("Consolidated Statements of Shareowners' Equity\nYear ended December 31 2024 2023 2022\n"
            f"Cash dividends declared per common share $ {dps} $ 1.84 $ 1.76")


def hit(ticker, year, text, page=60):
    chunk_id = f"annual_report_edgar_ars_{ticker}_{year}_p00{page}_vlm0"
    return {"chunk_id": chunk_id, "ticker": ticker, "year": year, "page": page, "doc_id": f"edgar_ars_{ticker}_{year}",
            "source_type": "annual_report_pdf", "text": text, "preview": "", "metadata": {"chunk_id": chunk_id}}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(fc, "default_clarifier", lambda: fc.Clarifier(directory=DIRECTORY))
    for key in ("RAG_FIN_VISUAL", "RAG_BACKEND", "RAG_SUPPORT_CHECK", "RAG_RETURNS_MAX_COMPANIES"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")


def searches(monkeypatch, respond):
    calls = []

    def fake(run_config):
        calls.append((run_config.ticker, run_config.year, run_config.query, run_config.top_k))
        return respond(run_config, len(calls))

    monkeypatch.setattr(agent, "retrieve_evidence", fake)
    return calls


def clarifier():
    return fc.Clarifier(directory=DIRECTORY)


# ----------------------------------------------------------------------------- routing

@pytest.mark.parametrize("query, route", [
    (RETURNS_QUERY, "shareholder_returns"),
    ("Compare what JPMorgan and Bank of America returned to shareholders in 2024", "shareholder_returns"),
    ("Which bank paid the most in dividends and buybacks in 2024?", "shareholder_returns"),
    (INCOME_QUERY, "dividend_income"),
    ("How much dividend would 250 Microsoft shares have paid in fiscal 2024?", "dividend_income"),
    ("What dividends did JPMorgan pay in 2024?", "lookup"),                     # a figure, not a ranking
    ("What was JPMorgan's net income in 2024?", "lookup"),
    ("How many shares did JPMorgan repurchase in 2024?", "lookup"),             # no share count of the user's
    ("What was the total return to shareholders of JPMorgan's stock in 2024?", "lookup"),   # stock return, not cash
    ("How much did JPMorgan return to shareholders in 2024?", "shareholder_returns"),
])
def test_rules_route_only_questions_that_ask_for_these_agents(query, route):
    assert router.rule_route(query, clarifier())[0] == route


def test_an_llm_route_to_an_action_agent_needs_the_questions_words():
    plan = {"feasible": False, "problems": []}
    decision = router.decide({"route": "dividend_income"}, "What was JPMorgan's net income in 2024?", clarifier(), plan)
    assert decision["route"] == "lookup" and decision["decided_by"] == "rules"


# ----------------------------------------------------------------------------- plans

def test_returns_plan_reads_a_sectors_companies_for_one_year_and_says_who_was_left_out():
    plan = returns.returns_plan(RETURNS_QUERY, clarifier())
    assert plan["feasible"] and plan["sector"] == "Banking" and plan["year"] == 2024
    assert set(plan["companies"]) <= {"JPM", "BAC", "WFC"} and "JPM" in plan["companies"]
    assert {"ticker": "MTB", "company": "M&T Bank", "reason": "no FY2024 filing indexed"} in plan["excluded"]
    assert len(plan["targets"]) == 2 * len(plan["companies"])                     # dividends and repurchases each


def test_returns_plan_without_a_company_or_sector_is_not_available():
    plan = returns.returns_plan("Who returned the most to shareholders in 2024?", clarifier())
    assert not plan["feasible"] and "no company or sector named" in returns.not_available_message(plan)


def test_holdings_pair_each_share_count_with_its_company():
    holdings, problems = income.parse_holdings(INCOME_QUERY, clarifier())
    assert [(h["ticker"], h["shares"]) for h in holdings] == [("KO", 100.0), ("JPM", 50.0)] and not problems
    holdings, _ = income.parse_holdings("Coca-Cola: 1,500 shares. Dividends in 2024?", clarifier())
    assert [(h["ticker"], h["shares"]) for h in holdings] == [("KO", 1500.0)]
    holdings, problems = income.parse_holdings("Dividends on 40 shares in 2024?", clarifier())
    assert not holdings and problems == ["no company named for the 40 shares"]
    assert income.parse_holdings("Dividends on 40 shares in 2024?", clarifier(), ticker="KO")[0][0]["ticker"] == "KO"


def test_income_plan_says_which_holdings_have_no_filing_for_the_year():
    plan = income.income_plan("I own 10 shares of M&T Bank and 20 shares of Coca-Cola. Dividends in 2024?", clarifier())
    assert [t["ticker"] for t in plan["targets"]] == ["KO"]
    assert plan["excluded"][0]["ticker"] == "MTB" and "no FY2024 filing" in plan["excluded"][0]["reason"]


# ----------------------------------------------------------------------------- figures read in code

def test_cash_flow_rows_are_read_in_code_and_share_counts_or_preferred_rows_are_not():
    rows = returns.read_row([hit("JPM", 2024, cash_flow("JPM"))], "share repurchases", 2024)
    assert rows["value"] == "(18,830)" and rows["unit"] == "millions"
    counts = ("Shares (in millions)\nYear ended December 31 2024 2023 2022\nFinancing activities\n"
              "Treasury stock repurchased (91.7) (151.0) (40.4)")
    assert returns.read_row([hit("JPM", 2024, counts)], "share repurchases", 2024) is None
    preferred = ("Statement of cash flows (in millions)\n2024 2023 2022\nPreferred stock:\n"
                 "Cash dividends paid (1,099) (1,141) (1,115)\nCommon stock:\nRepurchased (19,448) (11,851) (6,033)\n"
                 "Cash dividends paid (5,133) (4,789) (4,178)\nNet cash used by financing activities (1) (2) (3)")
    assert returns.read_row([hit("WFC", 2024, preferred)], "dividends paid", 2024)["value"] == "(5,133)"
    assert returns.read_row([hit("WFC", 2024, preferred)], "share repurchases", 2024)["value"] == "(19,448)"


def test_vertical_rows_small_amounts_and_a_damaged_text_layer_are_still_read():
    vertical = ("Consolidated Statements of Cash Flows\n(in millions)\n2022\n2023\n2024\nFinancing activities\n"
                "Repurchases of stock\n(59,296)\n(61,504)\n(62,222)\nDividend payments\n0\n0\n(7,363)")
    assert returns.read_row([hit("MSFT", 2024, vertical)], "share repurchases", 2024)["value"] == "(62,222)"
    assert returns.read_row([hit("MSFT", 2024, vertical)], "dividends paid", 2024)["value"] == "(7,363)"
    small = ("Cash flows (Dollars in Millions)\n2024 2023 2022\nFinancing activities\n"
             "Repurchase of common stock (173) (62) (69)")
    assert returns.read_row([hit("BAC", 2024, small)], "share repurchases", 2024)["value"] == "(173)"
    damaged = ("Cash flows from financing activities (Dollars in Millions)\n2024*-*. 2023*-*< 2022*-**\n"
               "Dividends@ to shareholders (11,823) (11,770) (11,682)\nRepurchase) of common stock$ (2,432) (5,054) (6,035)")
    assert returns.read_row([hit("KO", 2024, damaged)], "dividends paid", 2024)["value"] == "(11,823)"
    assert returns.read_row([hit("KO", 2024, damaged)], "share repurchases", 2024)["value"] == "(2,432)"


def test_a_chunk_without_the_header_takes_its_pages_header():
    head = hit("BAC", 2024, "Consolidated Statement of Cash Flows (in millions)\nYear ended December 31 2024 2023 2022\n"
                            "Net income 27,132 26,515 27,528", page=70)
    rest = {**hit("BAC", 2024, "Financing activities\nCash dividends paid (9,503) (8,800) (8,600)", page=70),
            "chunk_id": "annual_report_edgar_ars_BAC_2024_p0070_vlm1"}
    row = returns.read_row([rest, head], "dividends paid", 2024)
    assert row["value"] == "(9,503)" and row["chunk_id"].endswith("vlm1") and row["unit"] == "millions"


def test_chevrons_cash_flow_rows_are_read():
    chevron = hit("CVX", 2024, "Consolidated Statement of Cash Flows\n(in millions)\n2024 2023 2022\n"
                  "Cash dividends - common stock (11,801) (11,336) (10,968)\n"
                  "Net sales (purchases) of treasury shares (15,044) (14,678) (5,417)")
    assert returns.read_row([chevron], "dividends paid", 2024)["value"] == "(11,801)"
    assert returns.read_row([chevron], "share repurchases", 2024)["value"] == "(15,044)"


def test_dividends_per_share_are_read_only_as_per_share_amounts():
    assert income.read_dps([hit("KO", 2024, per_share("KO"))], 2024)["value"] == "1.94"
    total = "Statement (in millions)\n2024 2023 2022\nCash dividends per share 8,359 7,952 7,616"
    assert income.read_dps([hit("KO", 2024, total)], 2024) is None


# ----------------------------------------------------------------------------- the action, in code

def test_rank_adds_dividends_and_buybacks_and_ranks_in_code():
    plan = returns.returns_plan(RETURNS_QUERY, clarifier())
    points = {}
    for ticker in plan["companies"]:
        for measure, text in zip(returns.MEASURES, CASH_FLOW[ticker]):
            points[(ticker, measure)] = {"value": -float(text.strip("()").replace(",", "")), "value_text": text,
                                         "unit": "millions", "chunk_id": f"{ticker}-cf", "status": "ok"}
    table = returns.rank(plan, points)
    assert table["ranking"] == ["JPM", "WFC", "BAC"]                    # 33,613 > 24,581 > 22,607
    top = next(c for c in table["companies"] if c["ticker"] == "JPM")
    assert top["total"] == 14783 + 18830 and top["rank"] == 1
    assert {c["kind"] for c in table["calculations"]} >= {"shareholder_return", "returns_total", "returns_share"}


def test_income_multiplies_the_users_shares_by_the_verified_dividend_per_share():
    plan = income.income_plan(INCOME_QUERY, clarifier())
    points = {"KO": {"value": 1.94, "value_text": "1.94", "chunk_id": "ko", "status": "ok"},
              "JPM": {"value": None, "status": "not_found"}}
    table = income.income(plan, points)
    assert table["holdings"][0]["income"] == 194.0 and table["total"] == 194.0 and table["calculated"] == 1
    assert {"kind": "user_input", "source": "user"}.items() <= table["calculations"][0].items()
    assert table["excluded"][-1]["ticker"] == "JPM"


# ----------------------------------------------------------------------------- the graph

def test_returns_graph_reads_every_company_and_answers_from_the_ranking(monkeypatch):
    calls = searches(monkeypatch, lambda c, n: [hit(c.ticker, c.year, cash_flow(c.ticker))])
    result = agent.financial_graph(agent.MockLLM(), config(RETURNS_QUERY), "router")
    run = result["agent_run"]
    assert result["route"]["route"] == "shareholder_returns" and result["final_status"] == "answered"
    assert run["path"] == ["security_step", "plain_text_step", "route_step", "action_fetch_step",
                           "action_series_step", "answer_step", "critic_step", "governance_step"]
    assert {(t, y) for t, y, *_ in calls} == {(t, 2024) for t in ("JPM", "BAC", "WFC")}
    table = result["shareholder_returns"]
    assert table["ranking"] == ["JPM", "WFC", "BAC"]
    assert all(c["dividends"]["method"] == "label" for c in table["companies"])
    assert result["answer"].startswith("Of the Banking companies read for FY2024, JPMorgan Chase (JPM) returned the most")
    assert result["support_check"]["verdict"] == "supported" and result["support_check"]["method"] == "code_table"
    assert result["human_review"]["required"] is False


def test_income_graph_calculates_from_the_filings_per_share_figures(monkeypatch):
    searches(monkeypatch, lambda c, n: [hit(c.ticker, c.year, per_share(c.ticker))])
    result = agent.financial_graph(agent.MockLLM(), config(INCOME_QUERY), "router")
    assert result["route"]["route"] == "dividend_income" and result["final_status"] == "answered"
    table = result["dividend_income"]
    assert [h["income"] for h in table["holdings"]] == [194.0, 240.0] and table["total"] == 434.0
    assert "$434.00" in result["answer"] and "100 shares x $1.94" in result["answer"]
    assert result["support_check"]["numeric"]["grounded"] is True
    assert result["human_review"]["required"] is False                    # "I hold" is not a request for advice


def test_a_missing_figure_gets_a_broadened_search_and_the_stop_message_is_kept(monkeypatch):
    calls = searches(monkeypatch, lambda c, n: [hit(c.ticker, c.year, "No figures here.")])
    result = agent.run_technique("dividend_income_graph", "mock", config(INCOME_QUERY))
    assert result["final_status"] == "insufficient_evidence" and result["answer"].startswith("No dividend income")
    assert len(calls) == 4 and all(top_k == 6 for *_, top_k in calls[2:])   # one broadened, deeper retry each


def test_advice_is_refused_before_the_dividend_agent_runs(monkeypatch):
    searches(monkeypatch, lambda c, n: [])
    result = agent.financial_graph(agent.MockLLM(), config("Should I buy 100 shares of Coca-Cola for the dividend?"),
                                   "router")
    assert result["final_status"] == "refused" and result["agent_run"]["path"][0] == "security_step"


def test_clarifier_passes_these_questions_through(monkeypatch):
    conversation = fc.Conversation("c1")
    outcome = clarifier().step(conversation, RETURNS_QUERY)
    assert outcome["status"] == "passthrough" and outcome["route_hint"] == "shareholder_returns"


def test_api_returns_the_ranking_and_the_longer_deadline(monkeypatch):
    from fastapi.testclient import TestClient
    from scripts.api import api_app
    searches(monkeypatch, lambda c, n: [hit(c.ticker, c.year, cash_flow(c.ticker))])
    body = TestClient(api_app.app).post("/chat/financial_report_qa", json={
        "query": RETURNS_QUERY, "technique": "shareholder_returns_graph", "llm_provider": "mock"}).json()
    assert body["shareholder_returns"]["ranking"] == ["JPM", "WFC", "BAC"]
    monkeypatch.setenv("RAG_REQUEST_TIMEOUT_SECONDS", "60")
    assert agent.request_deadline("router", config(RETURNS_QUERY)) == 300


def test_a_report_the_llm_and_the_rules_both_name_is_not_downgraded():
    # Live check (Nov 2026): hosted Qwen3-8B said "report" for "Generate a full financial report for ...", the rules
    # agreed, and decide() still downgraded it to a lookup, so no report request ever reached the report pipeline.
    query = "Generate a full financial report for JPMorgan FY2024"
    decision = router.decide({"route": "report"}, query, clarifier(), {"feasible": False, "problems": []})
    assert decision["route"] == "report" and not decision["adjustments"]
    lookup = router.decide({"route": "report"}, "What was JPMorgan's net income in 2024?", clarifier(),
                           {"feasible": False, "problems": []})
    assert lookup["route"] == "lookup"                                      # an LLM "report" alone is still downgraded
