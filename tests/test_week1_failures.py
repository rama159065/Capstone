"""Week 1 single-prompt failures: the policy refusal, the capstone-retrieval context and the failure replay."""
from __future__ import annotations

import json

import pytest

import scripts.agents.financial_prompting_agent as agent
from scripts.evaluation import failure_replay as fr
from scripts.evaluation import prompt_baselines as pb

def eval_items():
    from scripts.common.paths import ROOT
    return pb.read_jsonl(ROOT / pb.EVAL_SET)


def test_every_advice_forecast_and_privacy_item_is_refused_by_rule():
    for item in eval_items():
        policy = agent.policy_refusal(item["question"])
        guard = agent.deterministic_input_guard(item["question"])
        if item["category"] == "refusal":
            assert policy["refuse"] or guard["decision"] == "block", item["id"]
        elif item["answerability"] == "answerable":
            assert not policy["refuse"], (item["id"], policy["labels"])


@pytest.mark.parametrize("question", [
    "What does Apple's annual report say about its outlook for fiscal 2026?",
    "What risk factors did NVIDIA report about export controls?",
    "What was Microsoft's revenue forecast accuracy discussion in the MD&A?",
    "How much did JPMorgan pay in dividends in 2024?",
])
def test_report_questions_about_outlook_are_not_refused(question):
    assert not agent.policy_refusal(question)["refuse"]


def test_agentic_graph_refuses_advice_in_its_first_step(monkeypatch):
    calls = []
    monkeypatch.setattr(agent, "retrieve_evidence", lambda config: calls.append("retrieve") or [])
    config = agent.AgentConfig(query="Should I buy NVIDIA stock right now?", ticker=None, year=None, source_type=None,
                               top_k=5, dense_top_k=20, bm25_top_k=20, retrieval_mode="hybrid",
                               embedding_provider="hashing", hash_dimensions=64, max_context_chars=4000,
                               self_consistency_samples=1)
    result = agent.run_technique("agentic_graph", "mock", config)
    assert result["final_status"] == "refused"
    assert "investment advice" in result["answer"]
    assert calls == []
    assert result["agent_run"]["path"][:1] == ["security_step"]


def test_rag_context_interleaves_targets_and_keeps_doc_ids(monkeypatch):
    def fake(config):
        return [{"chunk_id": f"annual_report_edgar_ars_{config.ticker}_2025_p{n:04d}_vlm0", "ticker": config.ticker,
                 "year": 2025, "page": n, "text": f"{config.ticker} page {n}", "score": 1.0 / n} for n in range(1, 6)]
    monkeypatch.setattr(agent, "answer_evidence", fake)
    item = {"question": "Which had higher net income in fiscal 2025: Apple or Microsoft?"}
    hits = pb.rag_hits(item, 5)
    assert [h["chunk_id"] for h in hits] == ["c1", "c2", "c3", "c4", "c5"]
    assert [h["ticker"] for h in hits] == ["AAPL", "MSFT", "AAPL", "MSFT", "AAPL"]
    assert hits[1]["doc_id"] == "edgar_ars_MSFT_2025"


def write_run(root, run_id, answer, correct=False, failure="hallucination_unsupported_claim"):
    out = root / run_id
    out.mkdir(parents=True)
    hits = [{"chunk_id": "c1", "doc_id": "edgar_ars_ABBV_2025", "ticker": "ABBV", "page": 1,
             "text": "Net revenues (in millions) 2025 $61,160 2024 $56,334"}]
    (out / "retrieval.jsonl").write_text(json.dumps({"id": "FIN-001", "hits": hits}) + "\n", encoding="utf-8")
    row = {"id": "FIN-001", "category": "single_figure", "strategy": "zero_shot", "correct": correct,
           "failure": failure, "answer": answer}
    (out / "scores.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return out


def test_replay_withholds_a_figure_that_is_not_in_the_cited_chunk(tmp_path, monkeypatch):
    from scripts.common.paths import ROOT
    monkeypatch.setattr(pb, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(pb, "EVAL_SET", ROOT / pb.EVAL_SET)
    write_run(tmp_path, "r1", "AbbVie's revenue for fiscal 2025 was $63,512 million [c1].")
    rows = fr.replay_run("r1")
    assert rows[0]["caught_by"]["numeric_grounding"].endswith("63,512")
    assert "policy" not in rows[0]["caught_by"]


def test_retrieval_measures_and_compare(tmp_path, monkeypatch):
    from scripts.common.paths import ROOT
    monkeypatch.setattr(pb, "OUT_ROOT", tmp_path)
    monkeypatch.setattr(pb, "EVAL_SET", ROOT / pb.EVAL_SET)
    write_run(tmp_path, "a", "x", failure="retrieval_miss")
    write_run(tmp_path, "b", "AbbVie's revenue for fiscal 2025 was $61,160 million [c1].", correct=True, failure=None)
    measures = pb.retrieval_measures(tmp_path / "a")
    assert measures["gold_doc_in_top5"] == 1 and measures["gold_value_in_evidence"] == 1
    text = pb.compare(["a", "b"], focus="zero_shot")
    assert "fixed (1): FIN-001" in text
    assert (tmp_path / "comparison.md").exists()


@pytest.mark.parametrize("answer,grounded", [
    ("As of December 31, 2025, revenue was $61,160 million [c1].", True),
    ("For the fiscal year ended 2025-12-31 revenue was $61.2 billion (page 49) [c1].", True),
    ("Revenue was $61,160 million, up 31% [c1].", False),                 # a growth rate needs the calculator
    ("Revenue was $63,512 million on May 3 [c1].", False),
])
def test_numeric_grounding_ignores_dates_and_pages_but_not_figures(answer, grounded):
    from scripts.agents.financial_tools import numeric_grounding
    evidence = [{"chunk_id": "c1", "text": "Total net revenues (in millions) 2025 61,160 2024 56,334"}]
    assert numeric_grounding(answer, evidence, [])["grounded"] is grounded


def test_review_terms_match_whole_words_only():
    from scripts.guardrails.checks import has_term
    assert not has_term("NVIDIA's total stockholders' equity was $157,293 million", agent.INVESTMENT_REVIEW_TERMS)
    assert not has_term("Selling, general and administrative expenses", agent.INVESTMENT_REVIEW_TERMS)
    assert has_term("Should we sell or hold?", agent.INVESTMENT_REVIEW_TERMS)
    assert agent.output_content_filter("Total stockholders' equity was $157,293 million [c1].")["decision"] == "allow"
