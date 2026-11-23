"""Hybrid retrieval (Chroma readings + ColPali pages), same-page expansion and the critic's unverified-figure flag."""
from __future__ import annotations

import json

import pytest

import scripts.agents.financial_prompting_agent as agent
from scripts.agents import financial_clarifier as fc
from scripts.agents import financial_tools as tools
from scripts.rag import chunk_schema
from scripts.rag import financial_hybrid as fh
from scripts.rag import financial_ingest_chroma_db as chroma_db

DOC = "edgar_ars_CVX_2022"


def payload(page, index, content_type="text", text="", unverified=()):
    return {"chunk_id": f"annual_report_{DOC}_p{page:04d}_vlm{index}", "doc_id": DOC, "ticker": "CVX", "fiscal_year": 2022,
            "year": 2022, "page": page, "chunk_index": index, "content_type": content_type, "text": text,
            "reader": "qwen3VL" if content_type != "page_description" else "qwen3VL",
            "unverified_in_chunk": list(unverified), "unverified_numbers": list(unverified)}


def hit(p, paths=("vlm_text",), rank=1):
    return chunk_schema.evidence_hit(p, rrf_score=1 / (60 + rank), paths=list(paths), ranks={"dense": rank})


class FakeStore:
    def __init__(self, pages):
        self.pages, self.asked = pages, []

    def page_payloads(self, doc_id, page):
        self.asked.append((doc_id, page))
        return [dict(p) for p in self.pages.get((doc_id, page), [])]


DESCRIPTION = payload(9, 0, "page_description", "Bar chart on page 9: Chevron net crude oil production, 2018-2022.")
VALUES = payload(9, 1, "table", "| Year | Net crude oil production (MBD) |\n|---|---|\n| 2022 | 1,440 |\n| 2021 | 1,523 |")
NOTE = payload(9, 2, "text", "Production figures include the company's share of affiliates.")


# ----------------------------------------------------------------------------- same-page expansion (Chroma)

def test_a_description_brings_its_pages_values():
    store = FakeStore({(DOC, 9): [DESCRIPTION, VALUES, NOTE]})
    other = hit(payload(40, 0, "text", "Upstream earnings were 30,000 million."), rank=2)
    out = chroma_db.with_page_values(store, [hit(DESCRIPTION), other])
    assert [h["chunk_id"] for h in out] == [DESCRIPTION["chunk_id"], VALUES["chunk_id"], NOTE["chunk_id"], other["chunk_id"]]
    added = out[1]
    assert added["retrieval_paths"] == ["vlm_text", "same_page"] and added["retrieval_ranks"]["same_page"] == 1
    assert added["rrf_score"] == out[0]["rrf_score"]                      # ranked with its description


def test_no_expansion_when_the_page_already_has_a_values_chunk_or_for_plain_hits():
    store = FakeStore({(DOC, 9): [DESCRIPTION, VALUES, NOTE]})
    out = chroma_db.with_page_values(store, [hit(DESCRIPTION), hit(VALUES, rank=2)])
    assert len(out) == 2 and store.asked == []
    assert chroma_db.with_page_values(store, [hit(VALUES)]) == [hit(VALUES)] and store.asked == []


def test_expansion_is_capped(monkeypatch):
    many = [payload(9, i, "text", f"row {i}") for i in range(1, 8)]
    out = chroma_db.with_page_values(FakeStore({(DOC, 9): [DESCRIPTION, *many]}), [hit(DESCRIPTION)], limit=2)
    assert len(out) == 3


def test_chroma_store_reads_one_page_in_order(tmp_path):
    store = chroma_db.ChromaStore(tmp_path)
    records = [(p["chunk_id"], p["text"], {k: v for k, v in p.items() if k != "text"}) for p in (NOTE, DESCRIPTION, VALUES)]
    store.collection.upsert(ids=[r[0] for r in records], documents=[r[1] for r in records],
                            metadatas=[chroma_db.ChromaStore.metadata(r[2]) for r in records],
                            embeddings=[[1.0, 0.0, 0.0]] * 3)
    assert [p["chunk_index"] for p in store.page_payloads(DOC, 9)] == [0, 1, 2]
    assert store.page_payloads(DOC, 10) == []


# ----------------------------------------------------------------------------- hybrid backend

def test_colpali_page_brings_chromas_reading_of_that_page(monkeypatch):
    store = FakeStore({(DOC, 9): [DESCRIPTION, VALUES]})
    text_hits = [hit(payload(40, 0, "text", "Chevron reported net production in its upstream review."), rank=1)]
    monkeypatch.setattr(chroma_db, "retrieve_for_agent", lambda *a, **k: text_hits)
    monkeypatch.setattr(chroma_db, "ChromaStore", lambda root: store)
    monkeypatch.setattr(fh, "visual_pages", lambda *a, **k: [{"doc_id": DOC, "page": 9, "rank": 1,
                                                               "payload": payload(9, 0, "chart", "pipeline 2 text")}])
    monkeypatch.setenv("RAG_FIN_VISUAL", "auto")
    out = fh.retrieve_for_agent("Chevron net crude oil production 2022 chart", top_k=3, ticker="CVX", year=2022, visual=True)
    ids = [h["chunk_id"] for h in out]
    assert VALUES["chunk_id"] in ids and DESCRIPTION["chunk_id"] in ids
    assert ids.index(VALUES["chunk_id"]) < ids.index(DESCRIPTION["chunk_id"])          # values before the description
    values = next(h for h in out if h["chunk_id"] == VALUES["chunk_id"])
    assert "visual" in values["retrieval_paths"] and values["retrieval_ranks"]["colpali"] == 1
    assert "1,440" in values["text"]


def test_a_page_chroma_never_read_falls_back_to_pipeline_2_text(monkeypatch):
    monkeypatch.setattr(chroma_db, "retrieve_for_agent", lambda *a, **k: [])
    monkeypatch.setattr(chroma_db, "ChromaStore", lambda root: FakeStore({}))
    page2 = payload(9, 0, "chart", "Net crude oil production 1,440")
    page2["chunk_id"] = f"annual_report_{DOC}_p0009_v0"
    monkeypatch.setattr(fh, "visual_pages", lambda *a, **k: [{"doc_id": DOC, "page": 9, "rank": 1, "payload": page2}])
    out = fh.retrieve_for_agent("crude production chart", top_k=3, visual=True)
    assert [h["chunk_id"] for h in out] == [page2["chunk_id"]] and out[0]["retrieval_paths"] == ["visual"]


def test_visual_search_only_when_asked_or_always(monkeypatch):
    calls = []
    monkeypatch.setattr(chroma_db, "retrieve_for_agent", lambda *a, **k: ["text"])
    monkeypatch.setattr(fh, "visual_pages", lambda *a, **k: calls.append(1) or [])
    monkeypatch.delenv("RAG_FIN_VISUAL", raising=False)
    assert fh.visual_setting() == "auto"
    assert fh.retrieve_for_agent("q", visual=None) == ["text"] and calls == []         # auto: the planner decides
    assert fh.retrieve_for_agent("q", visual=True) == ["text"] and calls == [1]
    monkeypatch.setenv("RAG_FIN_VISUAL", "0")
    assert fh.retrieve_for_agent("q", visual=True) == ["text"] and calls == [1]        # turned off
    monkeypatch.setenv("RAG_FIN_VISUAL", "1")
    assert fh.retrieve_for_agent("q", visual=None) == ["text"] and calls == [1, 1]


def test_fusion_adds_both_scores_for_a_chunk_found_both_ways():
    both = hit(VALUES, rank=2)
    text_only = hit(payload(40, 0, "text", "x"), rank=1)
    pages = [{"doc_id": DOC, "page": 9, "rank": 1}]
    out = fh.fuse([text_only, both], pages, {(DOC, 9): [dict(both, retrieval_paths=["visual", "vlm_text"])]}, top_k=2)
    assert out[0]["chunk_id"] == VALUES["chunk_id"] and out[0]["retrieval_ranks"]["colpali"] == 1
    assert out[0]["rrf_score"] == pytest.approx(1 / 62 + 1 / 61, abs=1e-6)


def test_a_colpali_only_page_does_not_push_text_hits_down():
    text = [hit(payload(40 + i, 0, "text", f"text hit {i}"), rank=i + 1) for i in range(3)]
    wrong_page = [hit(payload(25, i, "text", f"other chart {i}")) for i in range(3)]
    pages = [{"doc_id": DOC, "page": 25, "rank": 1}]
    out = fh.fuse(text, pages, {(DOC, 25): wrong_page}, top_k=3)
    assert [h["chunk_id"] for h in out[:3]] == [h["chunk_id"] for h in text]          # text order kept
    assert [h["chunk_id"] for h in out[3:]] == [h["chunk_id"] for h in wrong_page]    # ColPali pages follow


def test_planner_enables_visual_search_for_chart_questions_with_the_hybrid(monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "financial_hybrid")
    monkeypatch.delenv("RAG_FIN_VISUAL", raising=False)
    plan = tools.normalize_plan({"tools": ["text_search"]}, "What does the production chart show for 2022?")
    assert "visual_search" in plan["tools"]
    assert "visual_search" not in tools.normalize_plan({"tools": ["text_search"]}, "What was revenue in 2022?")["tools"]
    monkeypatch.setenv("RAG_FIN_VISUAL", "0")
    assert "visual_search" not in tools.normalize_plan({"tools": ["visual_search"]}, "chart")["tools"]


def test_agent_dispatches_to_the_hybrid(monkeypatch):
    seen = {}
    monkeypatch.setenv("RAG_BACKEND", "financial_hybrid")
    monkeypatch.setattr(fh, "retrieve_for_agent", lambda *a, **k: seen.update(args=a, kwargs=k) or [{"chunk_id": "hit"}])
    config = agent.AgentConfig(query="q", ticker="CVX", year=2022, source_type=None, top_k=4, dense_top_k=4, bm25_top_k=4,
                               retrieval_mode="hybrid", embedding_provider="hashing", hash_dimensions=16,
                               max_context_chars=6000, self_consistency_samples=1, visual=True)
    assert agent.retrieve_evidence(config) == [{"chunk_id": "hit"}]
    assert seen["kwargs"]["visual"] is True and seen["kwargs"]["ticker"] == "CVX"


# ----------------------------------------------------------------------------- critic: figures only a VLM read

def test_figure_only_a_vlm_read_is_grounded_but_unverified():
    flagged = {"chunk_id": "a", "text": "| 2022 | 1,440 |", "unverified_numbers": ["1,440"]}
    result = tools.numeric_grounding("Net crude oil production was 1,440 thousand barrels per day [a].", [flagged], [])
    assert result["grounded"] and result["unverified"] == ["1,440"]
    confirmed = {"chunk_id": "b", "text": "Net crude oil production 1,440 MBD", "unverified_numbers": []}
    assert tools.numeric_grounding("It was 1,440 [a][b].", [flagged, confirmed], [])["unverified"] == []
    assert tools.numeric_grounding("It was 9,999 [a].", [flagged], [])["ungrounded"] == ["9,999"]


CHUNK = f"annual_report_{DOC}_p0009_vlm1"


class ScriptedLLM(agent.MockLLM):
    def __init__(self, answer):
        self.answer = answer

    def invoke(self, messages):
        joined = "\n".join(getattr(m, "content", str(m)) for m in messages)
        if "Support check JSON" in joined:
            return agent.MockResponse(json.dumps({"claims": [{"claim": "production was 1,440", "chunk_ids": [CHUNK],
                                                              "supported": True, "reason": "stated"}],
                                                  "verdict": "supported"}))
        if "Answer with citations." in joined:
            return agent.MockResponse(self.answer)
        return super().invoke(messages)


@pytest.mark.parametrize("unverified, review", [(["1,440"], True), ([], False)])
def test_an_unverified_vlm_figure_sends_the_answer_to_human_review(monkeypatch, unverified, review):
    monkeypatch.setattr(fc, "default_clarifier", lambda: fc.Clarifier(directory={"CVX": [2022]}))
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    monkeypatch.delenv("RAG_SUPPORT_CHECK", raising=False)
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")
    evidence = {"chunk_id": CHUNK, "ticker": "CVX", "year": 2022, "page": 9, "source_type": "annual_report_pdf",
                "text": "| Year | Net crude oil production |\n| 2022 | 1,440 |", "preview": "1,440",
                "unverified_numbers": unverified, "metadata": {"chunk_id": CHUNK}}
    monkeypatch.setattr(agent, "retrieve_evidence", lambda run_config: [evidence])
    config = agent.AgentConfig(query="What was Chevron's net crude oil production in its 2022 annual report?",
                               ticker=None, year=None,
                               source_type=None, top_k=4, dense_top_k=4, bm25_top_k=4, retrieval_mode="hybrid",
                               embedding_provider="hashing", hash_dimensions=16, max_context_chars=6000,
                               self_consistency_samples=1)
    result = agent.agentic_graph_prompt(ScriptedLLM(f"Net crude oil production was 1,440 MBD in 2022 [{CHUNK}]."), config)
    assert result["support_check"]["verdict"] == "supported"
    assert result["support_check"]["numeric"]["unverified"] == unverified
    assert ("unverified_vlm_figure" in result["human_review"]["triggers"]) is review
    assert result["final_status"] == ("pending_human_review" if review else "answered")
