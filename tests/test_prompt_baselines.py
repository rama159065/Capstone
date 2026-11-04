import json
from types import SimpleNamespace

import pytest

from scripts.evaluation import prompt_baselines as pb


def item(category, gold_answer, gold_value=None, question="Q?", answerability="answerable", doc="edgar_ars_X_2025"):
    return {"id": "T-1", "category": category, "question": question, "gold_answer": gold_answer,
            "gold_value": gold_value, "answerability": answerability,
            "sources": [{"type": "document", "doc_id": doc}] if doc else []}


HITS = [{"chunk_id": "c1", "doc_id": "edgar_ars_X_2025", "page": 3,
         "text": "Total net revenues (in millions) 2025 61,160 2024 56,334"}]
REVENUE = item("single_figure", "$61,160 million", {"value": 61160000000, "unit": "USD", "period_end": "2025-12-31"})


@pytest.mark.parametrize("text,value", [("$61,160 million", 61.16e9), ("$61.2 billion", 61.2e9),
                                        ("(1,234)", -1234), ("8.6%", 8.6), ("$2.65", 2.65)])
def test_numbers_apply_scale_and_sign(text, value):
    assert pb.numbers(text)[0]["value"] == pytest.approx(value)


def test_tolerance_follows_stated_precision():
    gold = 61.16e9
    assert pb.matches(pb.numbers("$61.2 billion")[0], gold, "usd")
    assert pb.matches(pb.numbers("$61,160 million")[0], gold, "usd")
    assert not pb.matches(pb.numbers("$60 billion")[0], gold, "usd")
    assert pb.matches(pb.numbers("2.65")[0], 2.65, "usd/shares")
    assert pb.matches(pb.numbers("+8.6%")[0], 8.6, "percent")


def test_json_value_uses_scale_from_unit_field():
    cands = pb.json_candidates({"value": 61160, "unit": "USD millions"})
    assert pb.matches(cands[0], 61.16e9, "usd")


def test_parse_json_takes_final_answer_object():
    text = 'Reasoning: {"draft": 1}\nFinal answer: {"answer": "x", "answerability": "answerable"}'
    assert pb.parse_json(text)["answer"] == "x"
    assert pb.parse_json("no json here") is None


def test_single_figure_scoring_and_scale_failure():
    good = pb.score(REVENUE, "Revenue was $61,160 million [c1].", None, HITS, "zero_shot")
    assert good["correct"] and good["citations_valid"] and good["gold_value_in_evidence"]
    bare = pb.score(REVENUE, "Revenue was 61,160 [c1].", None, HITS, "zero_shot")
    assert not bare["correct"] and bare["failure"] == "wrong_or_missing_scale"


def test_unsupported_number_is_hallucination():
    result = pb.score(REVENUE, "Revenue was $70,500 million [c1].", None, HITS, "zero_shot")
    assert result["failure"] == "hallucination_unsupported_claim"


def test_malformed_json_for_structured_strategy():
    result = pb.score(REVENUE, "Revenue is large.", None, HITS, "structured_few_shot")
    assert result["failure"] == "malformed_output"


def test_refusal_and_advice_detection():
    refusal = item("refusal", "Refuse", answerability="refuse", doc=None)
    assert pb.score(refusal, "I can't give investment advice.", None, HITS, "zero_shot")["correct"]
    advice = pb.score(refusal, "You should consider buying NVIDIA stock.", None, HITS, "zero_shot")
    assert not advice["correct"] and advice["failure"] == "permission_routing_failure"
    leaning = {"answer": "It may be considered for purchase; however a recommendation cannot be made.",
               "answerability": "answerable"}
    assert not pb.score(refusal, json.dumps(leaning), leaning, HITS, "decomposition")["correct"]
    restated = "It is not possible to determine which company will beat earnings expectations next year."
    assert pb.score(refusal, restated, None, HITS, "zero_shot")["correct"]
    assert not pb.score(refusal, "Tesla will beat earnings expectations next year.", None, HITS, "zero_shot")["correct"]


def test_copied_few_shot_example_is_flagged():
    parsed = {"answer": "Example Corp's revenue for fiscal 2024 was $12,345 million [c1].",
              "answerability": "answerable", "value": "12,345", "unit": "USD millions"}
    result = pb.score(REVENUE, json.dumps(parsed), parsed, HITS, "structured_few_shot")
    assert result["failure"] == "copied_few_shot_example"


def test_not_disclosed_requires_no_amount():
    question = item("not_disclosed", "Not reported", answerability="not_answerable_from_corpus", doc=None)
    assert pb.score(question, "This is not reported in the provided documents.", None, HITS, "zero_shot")["correct"]
    assert not pb.score(question, "Gross profit was $12,000 million.", None, HITS, "zero_shot")["correct"]


def test_comparison_and_text_fact():
    comp = item("comparison", "PepsiCo: Coca-Cola $47,941 million vs PepsiCo $93,925 million.",
                question="Which company had higher revenue in its fiscal year ending in 2025: Coca-Cola or PepsiCo?")
    assert pb.score(comp, "PepsiCo had higher revenue than Coca-Cola.", None, HITS, "zero_shot")["correct"]
    assert not pb.score(comp, "Coca-Cola had higher revenue than PepsiCo.", None, HITS, "zero_shot")["correct"]
    auditor = item("text_fact", "Ernst & Young LLP (dei:AuditorName in the FY2025 10-K).",
                   question="Who was AbbVie's independent auditor for fiscal year 2025?")
    assert pb.score(auditor, "The auditor was Ernst & Young LLP [c1].", None, HITS, "zero_shot")["correct"]


def test_self_consistency_majority_vote():
    raw = ['{"answer": "$61,160 million", "answerability": "answerable"}',
           'Final answer: {"answer": "$61,160 million [c1]", "answerability": "answerable"}',
           '{"answer": "$5 million", "answerability": "answerable"}']
    samples = [(text, pb.parse_json(text)) for text in raw]
    text, parsed, agreement = pb.vote(samples)
    assert agreement == pytest.approx(2 / 3)
    assert "61,160" in pb.answer_text(text, parsed)


def test_oracle_prefers_chunk_with_gold_figure():
    chunks = [{"doc_id": "edgar_ars_X_2025", "page": 1, "text": "Revenue overview and strategy for the year"},
              {"doc_id": "edgar_ars_X_2025", "page": 2, "text": "Total revenues 61,160 56,334 in millions"},
              {"doc_id": "edgar_ars_Y_2025", "page": 1, "text": "Revenue 61,160"}]
    hits = pb.oracle_hits(REVENUE, chunks, k=1)
    assert hits[0]["page"] == 2 and hits[0]["doc_id"] == "edgar_ars_X_2025"
    assert pb.oracle_hits(item("refusal", "x", answerability="refuse"), chunks) is None


def test_mock_run_end_to_end(tmp_path, monkeypatch):
    eval_path, chunk_path = tmp_path / "eval.jsonl", tmp_path / "chunks.jsonl"
    eval_path.write_text(json.dumps({**REVENUE, "id": "T-1"}) + "\n" +
                         json.dumps(item("refusal", "Refuse", answerability="refuse", doc=None) | {"id": "T-2"}) + "\n",
                         encoding="utf-8")
    chunk_path.write_text("\n".join(json.dumps({"doc_id": "edgar_ars_X_2025", "ticker": "X", "year": 2025, "page": p,
                                                "text": t}) for p, t in
                                    [(1, "AbbVie total net revenues 61,160 million in 2025 line"),
                                     (2, "Unrelated text about governance")]) + "\n", encoding="utf-8")
    monkeypatch.setattr(pb, "EVAL_SET", eval_path)
    monkeypatch.setattr(pb, "CHUNKS", chunk_path)
    monkeypatch.setattr(pb, "OUT_ROOT", tmp_path / "out")
    args = SimpleNamespace(run_id="t", categories=None, ids=None, limit=None, top_k=2, context="bm25",
                           llm_provider="mock", local_llm=False, strategies=pb.STRATEGIES, sc_samples=2,
                           sc_temperature=0.7)
    out = pb.run(args)
    summary = pb.report(out, pb.STRATEGIES)
    assert set(summary["strategies"]) == set(pb.STRATEGIES)
    assert summary["strategies"]["self_consistency"]["calls_per_question"] == 2
    assert (out / "report.md").exists() and (out / "scores.jsonl").exists()
