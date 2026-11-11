"""Both vector stores retrieve hybrid (dense + BM25, fused by RRF) and the agent's prompt shows how each chunk matched."""
import hashlib
import json

import pytest

from scripts.rag import financial_ingest_qdrant_db as fi
from scripts.rag import hybrid
from scripts.rag import financial_ingest_chroma_db as vi
from tests.test_financial_ingest_qdrant_db import args_for as qdrant_args, make_pdf
from tests.test_financial_ingest_chroma_db import FakeVLM, args_for as chroma_args, corpus  # noqa: F401  (Chroma corpus fixture)


@pytest.fixture
def qdrant_corpus(tmp_path):
    (tmp_path / "pdfs").mkdir()
    pdf = tmp_path / "pdfs" / "EDGAR_ARS_TEST_2025.pdf"
    make_pdf(pdf)
    row = {"document_id": "edgar_ars_TEST_2025", "company": "Test Corp", "ticker": "TEST", "cik": "1", "year": 2025,
           "title": "Test Corp annual report 2025", "source_url": "https://example.invalid/x.pdf",
           "local_path": "EDGAR_ARS_TEST_2025.pdf", "content_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def test_rrf_fuse_adds_reciprocal_ranks_and_keeps_each_retrievers_rank():
    fused = hybrid.rrf_fuse({"dense": ["a", "b", "c"], "bm25": ["c", "a"]})
    assert [e["id"] for e in fused] == ["a", "c", "b"]
    assert fused[0]["ranks"] == {"dense": 1, "bm25": 2}
    assert fused[0]["rrf"] == pytest.approx(1 / 61 + 1 / 62)
    assert hybrid.rrf_fuse({"dense": ["a", "b"]}, limit=1) == [{"id": "a", "rrf": pytest.approx(1 / 61), "ranks": {"dense": 1}}]
    assert hybrid.matched_by({"dense": 3, "bm25": 1}) == "bm25 #1, dense #3"
    with pytest.raises(ValueError, match="retrieval mode"):
        hybrid.check_mode("sparse")


def test_qdrant_text_search_is_hybrid_and_modes_isolate_each_retriever(tmp_path, qdrant_corpus, monkeypatch):
    client = fi.connect(None, tmp_path / "root")
    run = fi.Ingest(qdrant_args(tmp_path, qdrant_corpus, stages="text"), client=client)
    run.run()
    both = fi.search_text(client, run.embedder, "total net sales 391,035", top_k=3)
    assert both[0].ranks.keys() == {"dense", "bm25"}                     # found by both retrievers
    assert all(set(h.ranks) == {"dense"} for h in fi.search_text(client, run.embedder, "net sales", 3, mode="dense"))
    assert all(set(h.ranks) == {"bm25"} for h in fi.search_text(client, run.embedder, "net sales", 3, mode="bm25"))
    monkeypatch.setattr(fi, "_serving", {"client": client, "embedder": run.embedder})
    hit = fi.retrieve_for_agent("total net sales 391,035", top_k=1, ticker="TEST")[0]
    assert set(hit["retrieval_ranks"]) == {"dense", "bm25"} and hit["retrieval_paths"] == ["text"]


def test_chroma_search_is_hybrid_and_modes_isolate_each_retriever(tmp_path, corpus):
    run = vi.VlmIngest(chroma_args(tmp_path, corpus, "chroma"), vlm=FakeVLM())
    run.run()
    root = tmp_path / "chroma"
    top = vi.retrieve("Downstream segment revenue 391,053", 1, root=root, embedder=run.embedder)[0]
    assert set(top["retrieval_ranks"]) == {"dense", "bm25"}
    dense = vi.retrieve("segment revenue", 3, root=root, embedder=run.embedder, mode="dense")
    keyword = vi.retrieve("segment revenue", 3, root=root, embedder=run.embedder, mode="bm25")
    assert dense and all(set(h["retrieval_ranks"]) == {"dense"} for h in dense)
    assert keyword and all(set(h["retrieval_ranks"]) == {"bm25"} for h in keyword)


def test_agent_sends_one_combined_prompt_with_the_hybrid_evidence(monkeypatch):
    from scripts.agents import financial_prompting_agent as agent
    hits = [{"chunk_id": "c1", "text": "Net sales were $391,035 million.", "ticker": "TEST", "year": 2025, "page": 1,
             "retrieval_ranks": {"dense": 2, "bm25": 1}, "rerank_score": .03},
            {"chunk_id": "c2", "text": "Membership fees rose 10%.", "ticker": "TEST", "year": 2025, "page": 3,
             "retrieval_ranks": {"bm25": 4}, "rerank_score": .01}]
    context = agent.evidence_to_context(hits, 4000)
    assert "matched_by: bm25 #1, dense #2" in context and "matched_by: bm25 #4" in context
    assert context.index("chunk_id: c1") < context.index("chunk_id: c2")

    seen = []

    class Recorder:
        def invoke(self, messages):
            seen.append(messages)
            return agent.MockResponse("Net sales were $391,035 million [c1].")

    monkeypatch.setattr(agent, "retrieve_evidence", lambda config: hits)
    config = agent.AgentConfig(query="What were net sales?", ticker="TEST", year=2025, source_type=None, top_k=2,
                               dense_top_k=5, bm25_top_k=5, retrieval_mode="hybrid", embedding_provider="hashing",
                               hash_dimensions=64, max_context_chars=4000, self_consistency_samples=1,
                               enable_guardrails=False)
    agent.baseline_prompt(Recorder(), config, hits)
    prompt = "\n".join(str(m.content) for m in seen[0])
    assert "What were net sales?" in prompt and "chunk_id: c1" in prompt and "chunk_id: c2" in prompt
