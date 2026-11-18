"""Fund cards: one short text per fund and period, written by code from the N-PORT tables and embedded with
bge-small-en-v1.5 into the Chroma collection ``fund_cards_mutual_fund`` (design section 5.2).

Cards let the agent find a fund from the way users name it ("Vanguard's S&P 500 fund"); the numbers an answer uses
are always read from the SQL tables (``tools_mutual_fund``), never from a card.

The collection lives in its own folder, ``<RAG_HOME>/indexes/mutual_fund/chroma_mutual_fund``, not in the annual-report
index (``chroma_db``): that index, its chunk schema and its tests stay untouched, and building cards never opens it.

Embedding: ``BAAI/bge-small-en-v1.5`` run locally (default) or through the shared embedding API
(``--embedding-provider hosted`` or ``RAG_EMBEDDING_PROVIDER=hosted``; the same vectors, see ``docs/embedding_key.md``).
The signature is recorded in the database's ``settings`` table (``card_embedding``); another model is refused.

    python -m scripts.rag.cards_mutual_fund build [--reset]
    python -m scripts.rag.cards_mutual_fund search "Vanguard S&P 500 index fund"
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

from scripts.common.companies import NAMES
from scripts.rag import nport_mutual_fund as nport

COLLECTION = "fund_cards_mutual_fund"
CHROMA_DIR = "chroma_mutual_fund"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
SETTING = "card_embedding"


def billions(value: float | None) -> str:
    return f"${(value or 0) / 1e9:,.1f} billion"


def card_text(db: sqlite3.Connection, accession: str) -> tuple[str, dict[str, Any]]:
    """The card of one filing: fund, period, size, top 10 equity holdings, sector mix and corpus coverage."""
    row = db.execute("SELECT f.series_id, f.series_name, f.registrant, f.family, g.period_end, g.net_assets, "
                     "g.holdings_count FROM filings g JOIN funds f USING (series_id) WHERE g.accession = ?",
                     (accession,)).fetchone()
    if row is None:
        raise KeyError(accession)
    series_id, series_name, registrant, family, period_end, net_assets, holdings_count = row
    top = db.execute("SELECT issuer_name, SUM(pct_of_net_assets) p FROM holdings WHERE accession = ? AND asset_cat = 'EC' "
                     "AND COALESCE(payoff, 'Long') = 'Long' GROUP BY COALESCE(NULLIF(substr(cusip, 1, 6), '000000'), "
                     "issuer_name) ORDER BY p DESC LIMIT 10", (accession,)).fetchall()
    mix = nport.sector_mix(db, accession)
    info = nport.coverage(db, accession)
    text = (f"{series_name.title()} ({registrant.title()}; family {family}), Form N-PORT holdings as of {period_end} "
            f"(accession {accession}). Net assets {billions(net_assets)}; {holdings_count} holdings. "
            "Top 10 equity holdings: " + ", ".join(f"{name} {pct:.2f}%" for name, pct in top) + ". "
            "Sector mix by corpus sector: " + ", ".join(f"{r['sector']} {r['pct_of_net_assets']:.1f}%" for r in mix[:8])
            + f". {info['corpus_pct_of_net_assets']:.1f}% of net assets is in {info['corpus_companies']} companies "
              "with annual reports in the corpus.")
    metadata = {"series_id": series_id, "series_name": series_name, "family": family, "period_end": period_end,
                "accession": accession, "chunk_id": f"nport_{nport.compact(accession)}_card", "doc_type": "fund_card",
                "net_assets": float(net_assets or 0)}
    return text, metadata


def make_embedder(provider: str | None = None, model: str | None = None):
    """bge-small-en-v1.5: local (default), or the shared API with ``hosted``; ``hashing`` for offline tests."""
    from scripts.rag import financial_ingest_qdrant_db as fi
    provider = (provider or os.getenv("RAG_EMBEDDING_PROVIDER") or "local").strip().lower()
    if provider == "hosted":
        return fi.Embedder("hosted")
    return fi.Embedder(provider, model or DEFAULT_MODEL)


def recorded(db: sqlite3.Connection) -> str | None:
    row = db.execute("SELECT value FROM settings WHERE key = ?", (SETTING,)).fetchone()
    return row[0] if row else None


def serving_embedder(db: sqlite3.Connection):
    """The embedder the cards were built with; bge-small may run locally or hosted (``RAG_EMBEDDING_PROVIDER``)."""
    from scripts.rag import financial_ingest_qdrant_db as fi
    signature = recorded(db)
    if not signature:
        return None
    if signature.startswith("hashing"):
        return fi.Embedder("hashing")
    portable = fi.portable_signature(signature)
    if portable != signature:
        return fi.equivalent_embedder(signature.partition(":")[0], portable.rsplit(":", 1)[0])
    provider, _, rest = signature.partition(":")
    return fi.Embedder(provider, rest.rpartition(":")[0])


class CardStore:
    def __init__(self, root: Path | str):
        import chromadb
        self.client = chromadb.PersistentClient(path=str(Path(root) / CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    def reset(self) -> None:
        self.client.delete_collection(COLLECTION)
        self.collection = self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def build(db_path: Path | str | None = None, root: Path | str | None = None, embedder=None, reset: bool = False,
          provider: str | None = None, model: str | None = None) -> dict[str, Any]:
    """Write (or rewrite) the card of every filing in the database."""
    from scripts.rag import financial_ingest_qdrant_db as fi
    db_path = Path(db_path or nport.default_db())
    root = Path(root or db_path.parent)
    db = nport.connect(db_path)
    embedder = embedder or make_embedder(provider, model)
    store = CardStore(root)
    signature = recorded(db)
    if reset:
        store.reset()
        db.execute("DELETE FROM settings WHERE key = ?", (SETTING,))
        signature = None
    if signature and fi.portable_signature(signature) != embedder.portable:
        raise ValueError(f"cards were embedded with {signature}, not {embedder.signature}; rebuild with --reset")
    accessions = [a for (a,) in db.execute("SELECT accession FROM filings ORDER BY accession")]
    cards = [card_text(db, a) for a in accessions]
    if cards:
        vectors = embedder([text for text, _ in cards])
        store.collection.upsert(ids=[m["chunk_id"] for _, m in cards], embeddings=vectors,
                                documents=[text for text, _ in cards], metadatas=[m for _, m in cards])
    if not signature:
        db.execute("INSERT OR REPLACE INTO settings VALUES (?, ?)", (SETTING, embedder.signature))
    db.commit()
    db.close()
    return {"db": str(db_path), "chroma": str(root / CHROMA_DIR), "collection": COLLECTION, "cards": len(cards),
            "embedding": signature or embedder.signature, "count": store.collection.count()}


def search(query: str, top_k: int = 3, db_path: Path | str | None = None, root: Path | str | None = None,
           embedder=None) -> list[dict[str, Any]]:
    """Cards nearest the query: [{chunk_id, series_id, series_name, period_end, accession, text, score}]."""
    db_path = Path(db_path or nport.default_db())
    root = Path(root or db_path.parent)
    if not (root / CHROMA_DIR).exists() or not db_path.exists():
        return []
    if embedder is None:
        with sqlite3.connect(db_path) as db:
            embedder = serving_embedder(db)
    if embedder is None:
        return []
    store = CardStore(root)
    if not store.collection.count():
        return []
    result = store.collection.query(query_embeddings=embedder([query]), n_results=min(top_k, store.collection.count()),
                                    include=["documents", "metadatas", "distances"])
    return [{**meta, "text": text, "score": round(1 - distance, 4)} for text, meta, distance in
            zip(result["documents"][0], result["metadatas"][0], result["distances"][0])]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("build", help="write and embed one card per fund filing")
    one.add_argument("--db")
    one.add_argument("--root", help="folder holding chroma_mutual_fund/ (default: the database's folder)")
    one.add_argument("--embedding-provider", default=None, help="local (default) | hosted | hashing")
    one.add_argument("--embedding-model", default=DEFAULT_MODEL)
    one.add_argument("--reset", action="store_true", help="drop the collection first (needed to change model)")
    two = sub.add_parser("search", help="nearest fund cards to a query")
    two.add_argument("query")
    two.add_argument("--db")
    two.add_argument("--root")
    two.add_argument("--top-k", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        out = build(args.db, args.root, reset=args.reset, provider=args.embedding_provider, model=args.embedding_model)
    else:
        out = [{k: v for k, v in hit.items() if k != "text"} | {"text": hit["text"][:300]}
               for hit in search(args.query, args.top_k, args.db, args.root)]
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
