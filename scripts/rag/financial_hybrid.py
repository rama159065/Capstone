"""Hybrid financial retrieval (``RAG_BACKEND=financial_hybrid``): Chroma's page readings plus ColPali page finding.

The two ingestion pipelines are good at different halves of a chart or table question:

* Pipeline 1 (Chroma, ``RAG_VLM_DIR``) *reads* visual pages: OCR, Qwen3-VL or a paid model turn tables and charts into
  Markdown with the values written out, and chart pages get a ``page_description`` chunk. But it *finds* pages by text
  alone (bge-small dense, 512 tokens, + BM25), so a values chunk whose wording doesn't match the question, or whose
  rows sit past the embedder's 512 tokens, can rank out of the top k.
* Pipeline 2 (Qdrant, ``RAG_INGEST_DIR``) *finds* visual pages from their pixels: ColPali multivectors, mean-pooled
  rows and columns for the first stage, exact MaxSim over the full on-disk vectors. But it hands the agent the page's
  text layer or OCR text, so a value that exists only as pixels never reaches the answer.

Here each does what it is good at. Chroma's own search runs first (with its same-page expansion: a description
brings its page's values). When the agent's plan includes ``visual_search`` (``RAG_FIN_VISUAL``: ``auto`` by default
here, ``1`` for every question, ``0`` to turn it off), ColPali finds the best pages for the question, and for each
page Chroma supplies that page's reading chunks, so the values come from the VLM reading of the page ColPali found.
A page Chroma never read falls back to pipeline 2's text of that page. Both lists are fused by reciprocal rank.

All hits keep the shared evidence format (``chunk_schema.evidence_hit``) and the agents' citation convention, so the
critic, the citation checks and the API work unchanged. ``retrieval_paths`` says how each hit was found
(``vlm_text``, ``same_page``, ``visual``).

ColPali runs on CPU here unless CUDA is available; ``RAG_FIN_COLPALI_DTYPE`` (default ``bfloat16`` on CPU, about
6 GB of RAM) picks its precision.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from scripts.rag import chunk_schema
from scripts.rag import financial_ingest_chroma_db as chroma_db

VISUAL_PAGES = 3          # ColPali pages used per search
RRF_K = 60


def visual_setting() -> str:
    """'always', 'auto' (the planner or chart words decide; the hybrid default) or 'off'."""
    value = os.getenv("RAG_FIN_VISUAL", "").strip().lower()
    return {"1": "always", "true": "always", "auto": "auto", "": "auto"}.get(value, "off")


def _qdrant():
    """The pipeline 2 client and its recorded text embedder, shared with ``financial_ingest_qdrant_db``'s own serving
    state: embedded Qdrant allows one client per folder and process, and ColPali should load once."""
    from scripts.rag import financial_ingest_qdrant_db as qdrant_db
    with qdrant_db._serving_lock:
        if "client" not in qdrant_db._serving:
            root = Path(os.getenv("RAG_INGEST_DIR", qdrant_db.DEFAULT_ROOT))
            client = qdrant_db.connect(os.getenv("QDRANT_URL"), root)
            qdrant_db.check_legacy(client)
            qdrant_db._serving.update(client=client, embedder=qdrant_db.recorded_embedder(argparse.Namespace(
                root=root, embedding_provider="hashing", embedding_model=None)))
        return {"client": qdrant_db._serving["client"], "embedder": qdrant_db._serving["embedder"], "module": qdrant_db}


def colpali():
    from scripts.rag import financial_ingest_qdrant_db as qdrant_db
    return qdrant_db.serving_colpali()


def visual_pages(query, ticker=None, year=None, sector=None, limit=VISUAL_PAGES) -> list[dict[str, Any]]:
    """The pages ColPali ranks best for the query: [{doc_id, page, rank, payload}], one entry per page."""
    served = _qdrant()
    qdrant_db, client = served["module"], served["client"]
    if not client.collection_exists(qdrant_db.VISUAL):
        return []
    flt = qdrant_db.query_filter(ticker, year, sector)
    pages, seen = [], set()
    for point in qdrant_db.search_visual(client, colpali(), served["embedder"], query, max(limit * 3, 10), flt):
        key = (point.payload.get("doc_id"), point.payload.get("page"))
        if key in seen or None in key:
            continue
        seen.add(key)
        pages.append({"doc_id": key[0], "page": key[1], "rank": len(pages) + 1, "payload": point.payload})
        if len(pages) >= limit:
            break
    return pages


def page_evidence(store, found: dict[str, Any], limit=chroma_db.SAME_PAGE_LIMIT) -> list[dict[str, Any]]:
    """Evidence for one ColPali page: Chroma's reading chunks of it (values first, description last), or pipeline 2's
    own text of the page when Chroma has none."""
    ranks = {"colpali": found["rank"]}
    payloads = store.page_payloads(found["doc_id"], found["page"]) if store is not None else []
    payloads.sort(key=lambda p: p.get("content_type") == "page_description")
    if payloads:
        return [chunk_schema.evidence_hit(p, rrf_score=0.0, paths=["visual", "vlm_text"], ranks=ranks)
                for p in payloads[:limit]]
    return [chunk_schema.evidence_hit(found["payload"], rrf_score=0.0, paths=["visual"], ranks=ranks)]


def fuse(text_hits: list[dict[str, Any]], pages: list[dict[str, Any]], page_hits: dict[tuple, list[dict[str, Any]]],
         top_k: int) -> list[dict[str, Any]]:
    """Reciprocal-rank fusion of Chroma's list and ColPali's pages.

    ColPali adds pages; it does not displace text hits. A chunk found both ways adds both scores (so it moves up); a
    chunk only ColPali found is ranked as if it came after Chroma's top k (rank ``top_k + page rank + its position on
    the page``), so a wrong ColPali page can't push a correct text hit out of the answer's context. At most ``top_k``
    text hits plus each ColPali page's chunks are kept.
    On the Chevron 2022 test, CPU ColPali ranked other chart pages above page 35 for three of four chart questions;
    with equal weights their chunks pushed Chroma's correct value chunks from rank 2-5 down to 8-10.
    """
    scored: dict[str, dict[str, Any]] = {}
    for position, hit in enumerate(text_hits[:max(top_k, 1) * 2], 1):
        entry = scored.setdefault(hit["chunk_id"], {"hit": dict(hit), "score": 0.0})
        entry["score"] += 1 / (RRF_K + position)
    text_ids = set(scored)
    for found in pages:
        for position, hit in enumerate(page_hits.get((found["doc_id"], found["page"]), [])):
            entry = scored.setdefault(hit["chunk_id"], {"hit": dict(hit), "score": 0.0})
            rank = found["rank"] if hit["chunk_id"] in text_ids else top_k + found["rank"] + position
            entry["score"] += 1 / (RRF_K + rank)
            known = entry["hit"]
            known["retrieval_paths"] = sorted(set(known.get("retrieval_paths") or []) | {"visual"})
            known["retrieval_ranks"] = {**(known.get("retrieval_ranks") or {}), "colpali": found["rank"]}
    ordered = sorted(scored.values(), key=lambda e: e["score"], reverse=True)
    keep = top_k + sum(len(page_hits.get((p["doc_id"], p["page"]), [])) for p in pages)
    hits = []
    for entry in ordered[:keep]:
        hit = entry["hit"]
        hit.update(rrf_score=round(entry["score"], 6), rerank_score=round(entry["score"], 6), score=round(entry["score"], 6))
        hits.append(hit)
    return hits


def retrieve_for_agent(query, top_k=5, ticker=None, year=None, sector=None, visual=None, mode="hybrid"):
    """Evidence for the financial agents from both indexes (see the module docstring)."""
    text_hits = chroma_db.retrieve_for_agent(query, top_k, ticker=ticker, year=year, sector=sector, mode=mode)
    setting = visual_setting()
    use_visual = setting == "always" if visual is None else (bool(visual) and setting != "off")
    if not use_visual:
        return text_hits
    pages = visual_pages(query, ticker, year, sector)
    if not pages:
        return text_hits
    store = chroma_db.ChromaStore(Path(os.getenv("RAG_VLM_DIR", chroma_db.DEFAULT_ROOT)))
    page_hits = {(p["doc_id"], p["page"]): page_evidence(store, p) for p in pages}
    return fuse(text_hits, pages, page_hits, top_k)
