import json

import pytest

from scripts.evaluation import evaluate_agent_api as ev


def item(category, answerability="answerable", **extra):
    return {"id": f"T-{category}", "category": category, "answerability": answerability,
            "question": f"question about {category}", "gold_answer": "", **extra}


def response(status, answer="", **extra):
    return {"final_status": status, "answer": answer, "security": {"decision": "allow", "risk_labels": []},
            "agent_run": {"steps": 8, "repairs": 0, "path": []}, **extra}


@pytest.mark.parametrize("text, ok", [
    ("Revenue was $61,160 million [c1].", True),
    ("Revenue was $61.16 billion.", True),
    ("Revenue was $61.2 billion.", True),              # rounding within 0.5%
    ("Net revenues | 61,160 | 56,334", True),          # table figure in millions, no scale word
    ("Revenue was $62.5 billion.", False),
    ("Revenue grew 61.16%.", False),                    # a percentage is not an amount
])
def test_amounts_match_at_any_stated_or_implied_scale(text, ok):
    assert ev.has_amount(text, 61_160_000_000) is ok


def test_percent_and_per_share_values():
    assert ev.has_percent("Revenue rose 8.6% year on year", 8.6)
    assert ev.has_percent("an increase of 8.57 percent", 8.6)
    assert ev.has_percent("revenue fell (3.2)%", -3.2)
    assert not ev.has_percent("revenue was $8.6 billion", 8.6)
    assert ev.has_value("Diluted EPS was $6.08", 6.08) and not ev.has_value("Diluted EPS was $6.18", 6.08)


def test_correct_uses_the_gold_value_or_the_gold_answer():
    single = item("single_figure", gold_value={"value": 61160000000, "unit": "USD"})
    assert ev.correct(single, "AbbVie's revenue was $61,160 million.") == (True, "amount")
    chart = item("chart_value", gold_value={"values": ["113.27", "54.03"]})
    assert ev.correct(chart, "NVIDIA $113.27 billion vs peer median $54.03 billion")[0]
    assert not ev.correct(chart, "NVIDIA $113.27 billion")[0]
    comparison = item("comparison", gold_answer="JPMorgan Chase: JPMorgan Chase $57,048 million vs "
                                                "Bank of America $30,509 million.")
    assert ev.correct(comparison, "JPMorgan ($57.0 billion) beat Bank of America ($30.5 billion).")[0]
    assert not ev.correct(comparison, "JPMorgan earned $57,048 million.")[0]
    employees = item("text_fact", gold_answer='Approximately 57,000 employees. Quote: "Employees AbbVie ..."')
    assert ev.correct(employees, "AbbVie had about 57,000 employees [c3].") == (True, "figures_in_gold_answer")
    auditor = item("text_fact", gold_answer="Ernst & Young LLP (dei:AuditorName in the FY2025 10-K).")
    assert ev.correct(auditor, "The auditor was Ernst & Young LLP.") == (True, "key_words")
    assert not ev.correct(auditor, "The auditor was PricewaterhouseCoopers LLP.")[0]


def test_expected_behaviour_decides_pass_or_fail():
    refuse = item("refusal", "refuse")
    assert ev.score(refuse, response("refused"), 200)["passed"]
    assert not ev.score(refuse, response("answered", "Buy it."), 200)["passed"]
    missing = item("not_disclosed", "not_answerable_from_corpus")
    assert ev.score(missing, response("not_available"), 200)["passed"]
    assert not ev.score(missing, response("answered", "Exxon earned $33.7 billion."), 200)["passed"]
    single = item("single_figure", gold_value={"value": 5, "unit": "USD/shares"})
    assert ev.score(single, response("answered", "EPS was $5.00"), 200)["passed"]
    assert ev.score(single, response("pending_human_review", "EPS was $5.00"), 200)["passed"]
    assert not ev.score(single, response("insufficient_evidence", "No answer found."), 200)["passed"]


def test_withheld_answers_record_whether_the_gold_was_in_the_excerpts():
    single = item("single_figure", gold_value={"value": 61160000000, "unit": "USD"})
    row = ev.score(single, response("evidence_only", "Unverified generation withheld. ... 61,160 [c1]"), 200)
    assert not row["passed"] and row["correct"] is None and row["gold_in_excerpts"] is True


def test_run_scores_every_item_resumes_and_summarizes(tmp_path):
    items = [item("single_figure", gold_value={"value": 2_000_000, "unit": "USD"}) | {"id": "A"},
             item("refusal", "refuse") | {"id": "B"},
             item("not_disclosed", "not_answerable_from_corpus") | {"id": "C"},
             item("computed_metric", gold_value={"value": 10.0, "unit": "percent"}) | {"id": "D"}]
    replies = {
        "A": response("answered", "Net income was $2.0 million [c1].", support_check={"verdict": "supported"},
                      grounding={"status": "grounded", "semantic_answer_verified": True},
                      agent_run={"steps": 9, "repairs": 1, "path": []}),
        "B": response("refused", security={"decision": "block", "risk_labels": ["investment_advice"]}),
        "C": response("answered", "It earned $5 billion."),
        "D": response("evidence_only", "Unverified generation withheld. growth of 10.0% [c2]"),
    }
    for i in items:
        i["question"] = f"q-{i['id']}"
    asked = []

    def post(body):
        asked.append(body)
        return 200, replies[body["query"][2:]]

    rows = ev.run(items, post, tmp_path, technique="agentic_graph", llm_provider="mock")
    assert [r["id"] for r in rows] == ["A", "B", "C", "D"] and len(asked) == 4
    assert asked[0]["technique"] == "agentic_graph" and asked[0]["llm_provider"] == "mock"
    assert "ticker" not in asked[0] and "year" not in asked[0]          # asked as a user would type it

    again = ev.run(items, post, tmp_path, technique="agentic_graph", llm_provider="mock")
    assert len(asked) == 4 and len(again) == 4                          # nothing re-asked

    summary = ev.write_summary(again, tmp_path, {"run_id": "t", "technique": "agentic_graph", "llm_provider": "mock"})
    assert summary["passed"] == 2 and summary["guardrails"]["refusal_recall"] == 1.0
    assert summary["guardrails"]["security_blocks"] == 1 and summary["guardrails"]["answered_unanswerable"] == 1
    assert summary["answers"]["answered"] == 1 and summary["answers"]["accuracy_on_answerable"] == 0.5
    assert summary["critic"]["verified_precision"] == 1.0 and summary["critic"]["questions_repaired"] == 1
    assert summary["critic"]["withheld_by_critic"] == 1 and summary["critic"]["withheld_gold_in_excerpts"] == 1
    md = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| C | not_disclosed | answered |" in md and json.loads((tmp_path / "summary.json").read_text())["passed"] == 2


def test_http_errors_are_rows_not_crashes(tmp_path):
    rows = ev.run([item("refusal", "refuse")], lambda body: (599, {"detail": "URLError: refused"}), tmp_path,
                  technique="router", llm_provider="hosted")
    assert rows[0]["final_status"] == "http_error" and not rows[0]["passed"] and "URLError" in rows[0]["error"]


def test_frozen_eval_set_has_a_check_for_every_item():
    items = ev.load_items(ev.EVAL_FILE)
    assert len(items) == 86
    for i in items:
        if i["answerability"] == "answerable":
            _, check = ev.correct(i, "")
            assert check != "no_gold", i["id"]
