from __future__ import annotations

from pathlib import Path

import pytest

import scripts.index_financial_report_pdfs as indexer


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0] for text in texts]


class FakeCollection:
    def query(self, **kwargs):
        self.kwargs = kwargs
        return {
            "ids": [["dense-a", "dense-b"]],
            "documents": [["Apple revenue risk", "Microsoft cloud revenue"]],
            "metadatas": [[
                {"chunk_id": "dense-a", "ticker": "AAPL", "year": 2024, "source_type": "annual_report_pdf"},
                {"chunk_id": "dense-b", "ticker": "MSFT", "year": 2024, "source_type": "annual_report_pdf"},
            ]],
            "distances": [[0.1, 0.9]],
        }


def test_normalize_chunk_filter_and_table_helpers() -> None:
    assert indexer.normalize_text("A\t  B\n\n\nC\x00") == "A B\n\nC"
    assert indexer.image_extension("foo.jpeg") == "jpg"
    assert indexer.image_extension("unknown") == "png"
    assert indexer.page_context("short\nRevenue increased 20 percent in 2024\nAnother useful line") == (
        "Revenue increased 20 percent in 2024"
    )
    assert indexer.build_query_filter("annual_report_pdf", "aapl", 2024) == {
        "$and": [{"source_type": "annual_report_pdf"}, {"ticker": "AAPL"}, {"year": 2024}]
    }

    markdown = indexer.table_to_markdown([["Metric", "Value"], ["Revenue", "100|200"]])
    assert "| Metric | Value |" in markdown
    assert "100/200" in markdown


def test_build_chunks_adds_page_metadata() -> None:
    report = indexer.PdfReport(
        document_id="annual_report_TEST_2024",
        title="Test report",
        ticker="TST",
        company="Test Co",
        year=2024,
        source_url="https://example.test/report.pdf",
        local_path=Path("report.pdf"),
        content_sha256="abc",
    )

    chunks = indexer.build_chunks(report, [(2, "Revenue risk. " * 80)], chunk_size=180, chunk_overlap=20)

    assert chunks
    assert chunks[0].chunk_id.startswith("annual_report_TEST_2024_p0002_c")
    assert chunks[0].metadata["ticker"] == "TST"
    assert chunks[0].metadata["page"] == 2


def test_bm25_search_filters_and_scores() -> None:
    rows = [
        {
            "chunk_id": "a",
            "text": "Apple net sales and product demand",
            "metadata": {"ticker": "AAPL", "year": 2024, "source_type": "annual_report_pdf"},
        },
        {
            "chunk_id": "b",
            "text": "Microsoft cloud revenue",
            "metadata": {"ticker": "MSFT", "year": 2024, "source_type": "annual_report_pdf"},
        },
    ]

    hits = indexer.bm25_search(rows, "Apple net sales", top_k=2, ticker="AAPL")

    assert [hit["chunk_id"] for hit in hits] == ["a"]
    assert isinstance(hits[0]["bm25_score"], float)


def test_dense_and_hybrid_retrieval_with_fake_collection() -> None:
    rows = [
        {
            "chunk_id": "dense-a",
            "text": "Apple revenue risk",
            "metadata": {"ticker": "AAPL", "year": 2024, "source_type": "annual_report_pdf"},
        },
        {
            "chunk_id": "lexical-only",
            "text": "Apple net sales holiday demand",
            "metadata": {"ticker": "AAPL", "year": 2024, "source_type": "annual_report_pdf"},
        },
    ]
    collection = FakeCollection()

    dense_hits = indexer.retrieve_top_k(collection, FakeEmbedder(), "Apple revenue", 2, ticker="AAPL")
    hybrid_hits = indexer.retrieve_hybrid_top_k(
        collection,
        FakeEmbedder(),
        rows,
        "Apple net sales",
        top_k=2,
        dense_top_k=2,
        bm25_top_k=2,
        rrf_k=60,
        reranker="simple",
        ticker="AAPL",
    )

    assert dense_hits[0]["chunk_id"] == "dense-a"
    assert collection.kwargs["where"] == {"ticker": "AAPL"}
    assert {hit["chunk_id"] for hit in hybrid_hits} >= {"dense-a"}


def test_embedder_env_validation(monkeypatch) -> None:
    monkeypatch.delenv("EMBEDDING_API_BASE", raising=False)
    monkeypatch.delenv("EMBEDDING_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)

    assert isinstance(indexer.get_embedder("hashing", 8), indexer.HashingEmbeddings)
    with pytest.raises(RuntimeError, match="Missing hosted embedding"):
        indexer.get_embedder("hosted", 8)
