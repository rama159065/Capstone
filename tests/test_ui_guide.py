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


def test_status_legend_covers_every_answer_status():
    assert {"answered", "evidence_only", "pending_human_review", "not_available", "insufficient_evidence",
            "not_grounded", "refused"} <= set(ui.STATUS_HELP)


def test_guide_renders_and_an_example_click_sends_it(monkeypatch):
    import sys
    if not hasattr(sys.modules.get("streamlit"), "__path__"):     # other test files install a bare stub module
        monkeypatch.delitem(sys.modules, "streamlit", raising=False)
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
