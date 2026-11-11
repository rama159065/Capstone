"""Hybrid retrieval shared by the Chroma and Qdrant backends: dense + BM25 rankings fused by RRF.

Each backend runs its two searches (Qdrant: the ``dense`` and ``bm25`` sparse vectors of
``annual_reports_text``; Chroma: the collection's embeddings and the FTS5 BM25 sidecar), then calls
``rrf_fuse`` here, so both fuse the same way and every hit records which retriever found it and at what
rank (``retrieval_ranks``, e.g. ``{"dense": 3, "bm25": 1}``). The agent puts that in the evidence block
it sends to the LLM.

Modes: ``hybrid`` (default, both), ``dense`` or ``bm25`` (one retriever, for ablations).
"""
from __future__ import annotations

RRF_K = 60
MODES = ("hybrid", "dense", "bm25")


def check_mode(mode: str | None) -> str:
    mode = mode or "hybrid"
    if mode not in MODES:
        raise ValueError(f"retrieval mode must be one of {MODES}, not {mode!r}")
    return mode


def uses(mode: str | None, retriever: str) -> bool:
    """Does this mode run the given retriever ("dense" or "bm25")?"""
    mode = check_mode(mode)
    return mode == "hybrid" or mode == retriever


def rrf_fuse(rankings: dict[str, list], k: int = RRF_K, limit: int | None = None) -> list[dict]:
    """{"dense": [id, ...], "bm25": [id, ...]} -> [{"id", "rrf", "ranks"}] best first.

    Reciprocal rank fusion: score = sum over retrievers of 1 / (k + rank). Ties keep the first-seen order.
    """
    fused: dict = {}
    for name, ids in rankings.items():
        for rank, item in enumerate(ids, 1):
            entry = fused.setdefault(item, {"id": item, "rrf": 0.0, "ranks": {}})
            if name not in entry["ranks"]:
                entry["ranks"][name] = rank
                entry["rrf"] += 1 / (k + rank)
    ordered = sorted(fused.values(), key=lambda e: e["rrf"], reverse=True)
    return ordered[:limit] if limit else ordered


def matched_by(ranks: dict | None) -> str:
    """{"dense": 3, "bm25": 1} -> "dense #3, bm25 #1" for the evidence block."""
    return ", ".join(f"{name} #{rank}" for name, rank in sorted((ranks or {}).items())) or "n/a"
