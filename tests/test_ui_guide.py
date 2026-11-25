"""Streamlit feature guide: coverage line, examples built from the served index, status legend."""
from __future__ import annotations

from pathlib import Path

from scripts.ui import api_streamlit_app as ui

RETAIL = [
    {"ticker": "AMZN", "company": "Amazon.com, Inc.", "sector": "Retail", "years": [2025]},
    {"ticker": "COST", "company": "Costco Wholesale Corporation", "sector": "Retail", "years": [2024, 2025]},
    {"ticker": "WMT", "company": "Walmart Inc.", "sector": "Retail", "years": [2025]},
]


def test_coverage_summary_counts_filings_companies_sectors_and_years():
    assert ui.coverage_summary(RETAIL) == "4 filings · 3 companies · Retail · FY2024–2025"
    assert ui.coverage_summary([RETAIL[0]]) == "1 filing · 1 company · Retail · FY2025"
    assert ui.coverage_summary([]) is None
    many = [{"ticker": f"T{i}", "sector": f"S{i}", "years": [2024]} for i in range(5)]
    assert "5 sectors" in ui.coverage_summary(many)


def test_examples_use_only_companies_and_years_the_index_holds():
    examples = {e["capability"]: e["question"] for e in ui.example_questions(RETAIL)}
    assert examples["Cited answer from one filing"] == "What was Costco's net income in fiscal 2025?"
    assert examples["Growth with a checked calculation"] == "How much did Costco's revenue grow from 2024 to 2025?"
    assert examples["Comparison across companies"] == "Compare Costco and Amazon revenue in 2025"
    assert examples["Follow-up questions"] == "Compare Costco and Amazon revenue"
    assert examples["Honest \"not available\""] == "What was Apple's net income in fiscal 2025?"
    assert examples["Guardrails"] == "Should I buy Costco stock?"


def test_examples_drop_what_the_coverage_cannot_show():
    one = ui.example_questions([{"ticker": "AAPL", "company": "Apple Inc.", "sector": "Technology", "years": [2024]}])
    capabilities = [e["capability"] for e in one]
    assert "Growth with a checked calculation" not in capabilities          # one year only
    assert "Comparison across companies" not in capabilities                # one company only
    assert next(e for e in one if "not available" in e["capability"])["question"].startswith("What was Microsoft")
    fallback = ui.example_questions([])
    assert fallback == [{"capability": "Cited answer", "question": ui.DEFAULT_QUERY,
                         "look_for": "Citations to the filing pages"}]


FUNDS = [{"series_name": "VANGUARD VALUE INDEX FUND", "family": "Vanguard", "filings": [{"period_end": "2026-06-30"}]},
         {"series_name": "VANGUARD 500 INDEX FUND", "family": "Vanguard", "filings": [{"period_end": "2026-06-30"}]}]
SECTORS = [
    {"ticker": "ABBV", "company": "AbbVie Inc.", "sector": "Healthcare", "years": [2022, 2023, 2024, 2025]},
    {"ticker": "AMGN", "company": "Amgen Inc.", "sector": "Healthcare", "years": [2023, 2024, 2025]},
    {"ticker": "JPM", "company": "JPMorgan Chase & Co.", "sector": "Banking", "years": [2022, 2023, 2024, 2025]},
    {"ticker": "BAC", "company": "Bank of America Corporation", "sector": "Banking", "years": [2022, 2023, 2024, 2025]},
    {"ticker": "BAC2", "company": "Bank Two", "sector": "Banking", "years": [2024, 2025]},
]


def test_examples_cover_trends_sectors_funds_and_reports():
    examples = ui.example_questions(SECTORS, FUNDS, [{"ticker": "AAPL", "company": "Apple Inc.", "year": 2024}])
    by = {e["capability"]: e for e in examples}
    assert by["Multi-year trend"]["question"] == "How did AbbVie's revenue trend from 2022 to 2025?"
    assert by["Sector trend"]["question"] == "How did healthcare revenue trend from 2023 to 2025?"   # not banks first
    assert by["Fund holdings"]["question"] == "What are the top 10 holdings of the Vanguard 500 Index Fund?"
    assert by["Look-through growth"]["question"] == "How did revenue of the Vanguard 500 Index Fund's top 10 holdings grow?"
    assert by["Full financial report"]["question"] == "Generate a full financial report for Apple 2024"
    groups = [e["group"] for e in examples]
    assert groups == sorted(groups, key=["Companies", "Sectors", "Mutual funds", "Reports", "Safety"].index)
    without = {e["capability"] for e in ui.example_questions(SECTORS)}
    assert not without & {"Fund holdings", "Look-through growth", "Full financial report"}
    banks = [r for r in SECTORS if r["sector"] == "Banking"]
    assert ui.sector_example(banks)["question"] == "How did banking net income trend from 2023 to 2025?"  # latest 3


def test_fund_coverage_and_names():
    assert ui.fund_name(FUNDS[1]) == "Vanguard 500 Index Fund"
    assert ui.fund_name({"series_name": "SPDR S&P 500 ETF TRUST"}) == "Spdr S&P 500 ETF Trust"
    assert ui.fund_summary(FUNDS) == "2 mutual funds (Vanguard, N-PORT as of 2026-06-30)"
    assert ui.fund_summary([]) is None
    assert {r["ticker"] for r in ui.report_profiles()} >= {"AAPL"}


def test_status_legend_covers_every_answer_status():
    assert {"answered", "evidence_only", "pending_human_review", "not_available", "insufficient_evidence",
            "not_grounded", "refused"} <= set(ui.STATUS_HELP)


def real_streamlit(monkeypatch):
    """Other test files install a bare streamlit stub; AppTest needs the real package (imported once only)."""
    import sys
    if hasattr(sys.modules.get("streamlit"), "__path__"):
        return
    if getattr(real_streamlit, "module", None):
        monkeypatch.setitem(sys.modules, "streamlit", real_streamlit.module)
        return
    monkeypatch.delitem(sys.modules, "streamlit", raising=False)
    import streamlit
    real_streamlit.module = streamlit


def test_guide_renders_and_an_example_click_sends_it(monkeypatch):
    real_streamlit(monkeypatch)
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")        # nothing listens: fallback example
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "mock")
    app = AppTest.from_file(str(Path(ui.__file__).resolve()), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "Financial Report Copilot"
    assert not [s for s in app.selectbox if s.label == "Domain"]            # financial only: no domain picker
    assert [e.label for e in app.expander][:2] == ["What can I ask?", "How to read an answer"]
    assert any("Test mode" in c.value for c in app.caption)
    button = next(b for b in app.button if b.label == ui.DEFAULT_QUERY)
    button.click().run()
    assert not app.exception
    assert any(ui.DEFAULT_QUERY in m.markdown[0].value for m in app.chat_message
               if m.markdown)


def test_demo_settings_hide_what_the_served_index_ignores(monkeypatch):
    real_streamlit(monkeypatch)
    from streamlit.testing.v1 import AppTest
    assert ui.served_backend({"runtime": {"financial_backend": "financial_vlm"}}) == "financial_vlm"
    assert ui.uses_corpus_index("financial_vlm") and ui.uses_corpus_index("financial_qdrant")
    assert not ui.uses_corpus_index("chroma") and not ui.uses_corpus_index(None)

    def labels(backend):
        monkeypatch.setenv("TEST_UI_BACKEND", backend)
        app = AppTest.from_function(sidebar_script, default_timeout=60).run()
        assert not app.exception
        return {w.label for w in [*app.selectbox, *app.slider, *app.text_input, *app.toggle]}

    served = labels("financial_vlm")
    assert {"Technique", "LLM provider", "Retrieval mode", "Evidence chunks (top k)", "Guardrails"} <= served
    assert not served & {"FastAPI base URL", "Embedding provider", "Hash dimensions", "Dense candidates",
                         "BM25 candidates", "Source type filter", "Self-consistency samples", "LangSmith project"}
    assert {"Embedding provider", "Dense candidates"} <= labels("chroma")      # older default index: still used
    assert ui.build_payload("q", {**ui.DEMO_DEFAULTS, "llm_provider": "mock", "top_k": 5, "technique": "router",
                                  "retrieval_mode": "hybrid", "ticker": "", "year": None, "sectors": [],
                                  "enable_guardrails": True, "enable_human_review": True,
                                  "enable_langsmith": False})["hash_dimensions"] == 384


def sidebar_script():
    """AppTest script: the sidebar against an API that reports the backend in TEST_UI_BACKEND."""
    import importlib.util
    import os
    from scripts.common.paths import ROOT
    spec = importlib.util.spec_from_file_location("ui_under_test", ROOT / "scripts" / "ui" / "api_streamlit_app.py")
    ui = importlib.util.module_from_spec(spec)           # a fresh copy bound to the real streamlit
    spec.loader.exec_module(ui)
    health = {"status": "ok", "runtime": {"financial_backend": os.environ["TEST_UI_BACKEND"]}}
    ui.get_json = lambda base, path: health if path == "/health" else {}
    ui.sidebar()


def test_sector_trend_table_renders(monkeypatch):
    real_streamlit(monkeypatch)
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_function(sector_table_script, default_timeout=60).run()
    assert not app.exception
    assert app.expander[0].label == "Sector trend: Healthcare revenue · CAGR 5.08%"
    assert len(app.dataframe) == 2 and any("Left out: Abbott" in c.value for c in app.caption)


def sector_table_script():
    """AppTest script: the sector table of a two-company healthcare panel."""
    import importlib.util
    from scripts.common.paths import ROOT
    spec = importlib.util.spec_from_file_location("ui_sector", ROOT / "scripts" / "ui" / "api_streamlit_app.py")
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    point = lambda y, v: {"year": y, "value_text": v, "status": "ok"}
    ui.render_sector_trend({
        "sector": "Healthcare", "metric": "revenue", "cagr_total": 5.0838, "panel": ["ABBV", "AMGN"],
        "panel_names": ["AbbVie", "Amgen"], "planned": ["ABBV", "ABT", "AMGN"], "sector_companies": 6,
        "rows": [{"year": 2022, "total": 84377.0, "total_pct_change": None, "median_company_pct_change": None,
                  "growing": None, "companies": 2},
                 {"year": 2023, "total": 82508.0, "total_pct_change": -2.22, "median_company_pct_change": 0.33,
                  "growing": 1, "companies": 2}],
        "chart": [{"year": 2022, "total": 84377.0}, {"year": 2023, "total": 82508.0}],
        "companies": [{"ticker": "ABBV", "company": "AbbVie", "in_panel": True, "cagr": 1.2,
                       "points": [point(2022, "58,054"), point(2023, "54,318")]}],
        "excluded": [{"ticker": "ABT", "company": "Abbott", "reason": "figure not verified for FY2023"}]})


def test_mutual_fund_table_renders(monkeypatch):
    real_streamlit(monkeypatch)
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_function(fund_table_script, default_timeout=60).run()
    assert not app.exception
    assert app.expander[0].label == "Holdings' growth: Vanguard 500 Index Fund · holdings as of 2026-06-30"
    assert len(app.dataframe) == 2
    assert any("Weighted growth 40.00%" in c.value for c in app.caption)
    assert any("Not used: Acme Widgets Ltd (not in the corpus)" in c.value for c in app.caption)


def fund_table_script():
    """AppTest script: a fund answer's table (top holdings linked to their annual reports)."""
    import importlib.util
    from scripts.common.paths import ROOT
    spec = importlib.util.spec_from_file_location("ui_fund", ROOT / "scripts" / "ui" / "api_streamlit_app.py")
    ui = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ui)
    point = lambda y, v: {"year": y, "value_text": v, "status": "ok"}
    ui.render_mutual_fund({
        "kind": "holdings_growth", "fund": "Vanguard 500 Index Fund", "period_end": "2026-06-30",
        "accession": "0000036405-26-000473", "filing_date": "2026-08-27", "net_assets": 1e12,
        "coverage": {"corpus_pct_of_net_assets": 86.8, "corpus_companies": 240},
        "rows": [{"rank": 1, "holding": "NVIDIA Corp", "ticker": "NVDA", "% of net assets": 7.5}],
        "companies": [{"ticker": "NVDA", "year": 2025, "weight": 7.5, "growth_pct": 40.0,
                       "points": [point(2024, "100"), point(2025, "140")]}],
        "weighted_growth_pct": 40.0, "covered_weight": 7.5, "asked_weight": 8.5,
        "excluded": [{"company": "Acme Widgets Ltd", "weight": 1.0, "reason": "not in the corpus"}]})


def test_dollar_amounts_are_not_read_as_maths():
    from scripts.ui import api_streamlit_app as ui
    assert ui.plain_dollars("was $275,235 million and $680,985 million") == "was \\$275,235 million and \\$680,985 million"
    assert ui.plain_dollars("already \\$5") == "already \\$5" and ui.plain_dollars(None) == ""


CITED = ["annual_report_edgar_ars_COST_2024_p0041_text0", "annual_report_edgar_ars_COST_2025_p0041_text0"]
TREND_ANSWER = {
    "domain": "financial_report_qa", "final_status": "answered", "technique": "router",
    "query": "How did Costco revenue trend from 2024 to 2025?",
    "answer": f"Costco's total revenue rose from $254,453 million in fiscal 2024 [{CITED[0]}] to $275,235 million "
              f"in fiscal 2025 [{CITED[1]}], up 8.17%.",
    "route": {"route": "trend", "decided_by": "rules", "llm_route": "lookup", "rule_route": "trend"},
    "agent_run": {"path": ["security_step", "plain_text_step", "route_step", "trend_fetch_step", "trend_series_step",
                           "answer_step", "critic_step", "governance_step"], "steps": 8, "max_steps": 10,
                  "tool_calls": [{"tool": "coverage_lookup", "searchable": ["COST FY2024", "COST FY2025"]},
                                 {"tool": "text_search", "target": "COST FY2024", "hits": 1, "new": 1, "ms": 120},
                                 {"tool": "text_search", "target": "COST FY2025", "hits": 1, "new": 1, "ms": 95},
                                 {"tool": "series_extract", "target": "COST FY2024", "method": "label"},
                                 {"tool": "series_extract", "target": "COST FY2025", "method": "label"},
                                 {"tool": "trend_calculator", "verified_years": 2, "cagr": 8.17}]},
    "trend": {"company": "COST", "company_name": "Costco", "metric": "revenue", "cagr": 8.17, "missing": [],
              "points": [{"year": 2024, "value": 254453.0, "value_text": "254,453", "unit": "millions", "status": "ok",
                          "page": 41, "chunk_id": CITED[0]},
                         {"year": 2025, "value": 275235.0, "value_text": "275,235", "unit": "millions", "status": "ok",
                          "page": 41, "chunk_id": CITED[1]}],
              "changes": [{"from": 2024, "to": 2025, "pct_change": 8.17, "status": "ok"}],
              "chart": [{"year": 2024, "value": 254453.0}, {"year": 2025, "value": 275235.0}]},
    "security": {"decision": "allow", "risk_labels": []}, "plain_text": {"allowed": True, "risk_level": "low"},
    "evidence_guard": {"safe": True}, "output_filter": {"decision": "allow", "risk_labels": []},
    "human_review": {"required": False, "status": "auto_approved", "triggers": []},
    "support_check": {"verdict": "supported", "method": "code_table", "numeric": {"checked": 3, "ungrounded": []}},
    "citation_check": {"valid": True, "cited_chunk_ids": CITED, "missing_chunk_ids": []},
    "grounding": {"status": "grounded", "evidence_count": 2, "cited_chunk_ids": CITED, "reason": "Citations matched."},
    "evidence_chunk_ids": CITED,
    "evidence_preview": [{"chunk_id": chunk, "ticker": "COST", "year": year, "page": 41, "preview": "Total revenue"}
                         for chunk, year in zip(CITED, (2024, 2025))],
    "trace": [{"node": "financial_workflow", "status": "ok", "elapsed_ms": 2900}],
}


def test_an_answer_has_one_thinking_dropdown_badges_and_numbered_sources(monkeypatch):
    real_streamlit(monkeypatch)
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("API_BASE_URL", "http://127.0.0.1:9")        # nothing listens; the answer is in the history
    monkeypatch.setenv("DEFAULT_LLM_PROVIDER", "mock")
    app = AppTest.from_file(str(Path(ui.__file__).resolve()), default_timeout=60)
    app.session_state["messages"] = [
        {"role": "user", "content": TREND_ANSWER["query"]},
        {"role": "assistant", "content": TREND_ANSWER["answer"], "result": TREND_ANSWER, "elapsed": 3.2}]
    app.run()
    assert not app.exception
    labels = [e.label for e in app.main.expander]
    assert [label for label in labels if label.startswith("Thought for")] == [
        "Thought for 3.2 s · 8 steps · 2 searches · critic: supported"]
    assert not [label for label in labels                            # the five old panels are one dropdown now
                if label.startswith(("Agent run", "Citations", "Evidence", "Guardrails", "Raw API"))]
    assert "Trend: Costco revenue · CAGR 8.17%" in labels and "What can I ask?" not in labels
    assert "What can I ask?" in [e.label for e in app.sidebar.expander]     # the guide moves aside once chatting
    assert [t.label for t in app.tabs] == [":material/timeline: Steps", ":material/account_tree: Agent graph",
                                          ":material/library_books: Sources (2)", ":material/verified_user: Checks",
                                          ":material/data_object: Raw"]
    answer = next(m.value for m in app.markdown if "Costco's total revenue" in m.value)
    assert ":blue-badge[1]" in answer and ":blue-badge[2]" in answer and "annual_report_" not in answer
    assert ":green-badge[:material/check_circle: Answered]" in [m.value for m in app.markdown]
    source = f"{ui.COMPANY_NAMES['COST']} · FY2024 annual report · p. 41"
    assert any(source in c.value for c in app.caption)


def test_every_material_icon_the_ui_uses_exists(monkeypatch):
    real_streamlit(monkeypatch)
    import re
    from streamlit.string_util import validate_icon_or_emoji
    from scripts.ui import ui_thinking
    icons = {icon for module in (ui, ui_thinking)
             for icon in re.findall(r":material/[a-z0-9_]+:", Path(module.__file__).read_text(encoding="utf-8"))}
    assert len(icons) > 30
    for icon in sorted(icons):
        validate_icon_or_emoji(icon)                    # raises for a name Streamlit doesn't know
