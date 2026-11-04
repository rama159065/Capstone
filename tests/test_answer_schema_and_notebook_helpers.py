import pytest

from scripts.agents.answer_schema import FinancialAnswer, ToolCall, validate_answer
from scripts.common import notebook_helpers as nb

EVIDENCE = [{"chunk_id": "c1", "doc_id": "d", "page": 3, "text": "Total net revenue (in millions) 2025 34,639 2024 25,785"}]


def test_valid_answer_passes_schema_and_grounding():
    parsed = {"answer": "Revenue was $34,639 million [c1].", "answerability": "answerable", "value": 34639,
              "unit": "USD millions", "period": "fiscal 2025", "citations": ["c1"]}
    model, errors = validate_answer(parsed, EVIDENCE)
    assert isinstance(model, FinancialAnswer) and model.value == "34639" and errors == []


@pytest.mark.parametrize("parsed,error", [
    (None, "no_json"),
    ({"answer": "x", "answerability": "maybe"}, "schema: answerability"),
    ({"answer": "Revenue was 34,639.", "answerability": "answerable", "value": "34,639"}, "schema: model"),
    ({"answer": "n/a", "answerability": "refuse", "value": "5"}, "schema: model"),
])
def test_schema_failures_are_reported(parsed, error):
    model, errors = validate_answer(parsed, EVIDENCE)
    assert model is None and errors[0].startswith(error)


def test_grounding_failures_are_reported():
    parsed = {"answer": "Revenue was $40,000 million [c9].", "answerability": "answerable", "value": "40,000",
              "citations": ["c9"]}
    _, errors = validate_answer(parsed, EVIDENCE)
    assert errors == ["citation_not_in_evidence", "value_not_in_evidence"]


def test_tool_call_needs_complete_arguments():
    assert ToolCall(tool="none").tool == "none"
    with pytest.raises(ValueError):
        ToolCall(tool="get_xbrl_fact", ticker="AMD", metric="revenue")
    with pytest.raises(ValueError):
        ToolCall(tool="get_xbrl_fact", ticker="amd lower", metric="revenue", fiscal_year=2025)


def test_metric_aliases_and_eval_items_and_labels():
    assert nb.metric_label("diluted EPS") == "diluted earnings per share"
    assert nb.metric_label("net income") == "net income attributable to the company"
    assert nb.metric_label("weather") is None
    items = nb.eval_items(["AMD_2025"], general=False)
    assert items and all("edgar_ars_AMD_2025" in {s.get("doc_id") for s in i["sources"]} for i in items)
    assert {i["category"] for i in nb.eval_items(["AMD_2025"])} >= {"refusal", "not_disclosed"}
    hits = nb.label_hits([{"chunk_id": "annual_report_x_p0001_text0", "text": "t", "page": 1, "metadata": {"doc_id": "x"}}])
    assert hits == [{"chunk_id": "c1", "source_chunk_id": "annual_report_x_p0001_text0", "doc_id": "x", "page": 1, "text": "t"}]
    assert nb.pick_filings(["AMD_2025"])[0]["ticker"] == "AMD"
    with pytest.raises(KeyError):
        nb.pick_filings(["ZZZ_1999"])


def test_template_run_with_mock_client():
    reply = nb.run_template(nb.make_llm("mock")[0], "chain_of_thought", "What was revenue?", EVIDENCE)
    assert reply["parsed"]["citations"] == ["c1"] and reply["strategy"] == "chain_of_thought"
