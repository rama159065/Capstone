"""Fund cards: written by code, embedded with bge-small-en-v1.5 (hashing stands in offline) into the Chroma collection
fund_cards_mutual_fund, in its own folder. The annual-report index is never opened."""
from __future__ import annotations

import sqlite3

import pytest

from scripts.rag import cards_mutual_fund as cards
from scripts.rag import financial_ingest_qdrant_db as fi
from tests.data_mutual_fund import ACC_500, SERIES_TOTAL, load_funds


class FakeEmbedder(fi.Embedder):
    def __init__(self, signature="local:BAAI/bge-small-en-v1.5:384"):
        self.provider, self.batch, self.dim, self.signature = "hashing", 64, 384, signature


def test_card_text_is_written_from_the_tables(tmp_path):
    db = load_funds(tmp_path)
    with sqlite3.connect(db) as conn:
        text, meta = cards.card_text(conn, ACC_500)
    assert text.startswith("Vanguard 500 Index Fund (Vanguard Index Funds; family Vanguard), Form N-PORT holdings as of 2026-06-30")
    assert "Net assets $1,000.0 billion" in text and "NVIDIA Corp 7.50%" in text
    assert "Alphabet Inc-Cl A 5.84%" in text                                     # share classes summed
    assert meta["chunk_id"] == "nport_000003640526000473_card" and meta["doc_type"] == "fund_card"


def test_build_and_search_in_its_own_collection(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_HOME", str(tmp_path / "rag_home"))
    db = load_funds(tmp_path / "mf")
    summary = cards.build(db, embedder=FakeEmbedder())
    assert summary["collection"] == "fund_cards_mutual_fund" and summary["cards"] == summary["count"] == 2
    assert summary["chroma"].endswith("chroma_mutual_fund")
    assert summary["embedding"] == "local:BAAI/bge-small-en-v1.5:384"
    hits = cards.search("Vanguard Total Stock Market Index Fund", 1, db, embedder=FakeEmbedder())
    assert hits[0]["series_id"] == SERIES_TOTAL
    assert not (tmp_path / "rag_home" / "indexes" / "chroma_db").exists()      # the annual-report index is untouched
    again = cards.build(db, embedder=FakeEmbedder("hosted:bge-small-en-v1.5:384"))   # same model via the shared API
    assert again["count"] == 2 and again["embedding"] == "local:BAAI/bge-small-en-v1.5:384"
    with pytest.raises(ValueError, match="rebuild with --reset"):
        cards.build(db, embedder=FakeEmbedder("local:sentence-transformers/all-MiniLM-L6-v2:384"))
    assert cards.build(db, embedder=FakeEmbedder("local:sentence-transformers/all-MiniLM-L6-v2:384"), reset=True)["count"] == 2


def test_default_model_is_bge_small(monkeypatch):
    made = []
    monkeypatch.setattr(fi, "Embedder", lambda provider="local", model=None, batch=64: made.append((provider, model)))
    monkeypatch.delenv("RAG_EMBEDDING_PROVIDER", raising=False)
    cards.make_embedder()
    monkeypatch.setenv("RAG_EMBEDDING_PROVIDER", "hosted")
    cards.make_embedder()
    assert made == [("local", "BAAI/bge-small-en-v1.5"), ("hosted", None)]
    assert cards.build_parser().parse_args(["build"]).embedding_model == "BAAI/bge-small-en-v1.5"
