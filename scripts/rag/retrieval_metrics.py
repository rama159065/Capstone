"""Shared retrieval metrics; nDCG uses ViDoRe/trec_eval linear relevance gains."""

import math
from collections.abc import Mapping, Sequence


def retrieval_metrics(retrieved_ids: Sequence[str], qrels: Mapping[str, float] | None,
                      k_values=(1, 5, 10)) -> dict:
    """Score an ordered ranking against complete chunk/page relevance judgments.

    Duplicate IDs occupy a rank but receive gain only once. Missing judgments
    are not inferred from retrieval scores. Empty judgments score zero.
    """
    keys = [f"recall@{k}" for k in k_values] + ["ndcg@5"]
    if qrels is None:
        return dict.fromkeys(keys)
    grades = {str(key): float(value) for key, value in qrels.items()}
    if any(not math.isfinite(value) or value < 0 for value in grades.values()):
        raise ValueError("Relevance grades must be finite and nonnegative")
    relevant = {key for key, value in grades.items() if value > 0}
    ranking = [str(key) for key in retrieved_ids]
    result = {}
    for k in k_values:
        if k < 1:
            raise ValueError("Metric cutoffs must be positive")
        result[f"recall@{k}"] = len(set(ranking[:k]) & relevant) / len(relevant) if relevant else 0.0
    seen = set()
    dcg = 0.0
    for rank, key in enumerate(ranking[:5], 1):
        if key not in seen:
            dcg += grades.get(key, 0.0) / math.log2(rank + 1)
        seen.add(key)
    ideal = sum(value / math.log2(rank + 1) for rank, value in
                enumerate(sorted(grades.values(), reverse=True)[:5], 1))
    result["ndcg@5"] = dcg / ideal if ideal else 0.0
    return result


def response_metrics(evidence, qrels=None, relevant_chunk_ids=None, top_k=5):
    """Evaluate the evidence actually returned by any RAG workflow."""
    if qrels is None and relevant_chunk_ids is not None:
        qrels = dict.fromkeys(relevant_chunk_ids, 1)
    ids = [str(hit["chunk_id"]) for hit in evidence if hit.get("chunk_id") is not None]
    scores = retrieval_metrics(ids, qrels)
    # A caller asking for fewer results has not measured the larger cutoffs.
    for key, cutoff in (("recall@1", 1), ("recall@5", 5), ("recall@10", 10), ("ndcg@5", 5)):
        if top_k < cutoff:
            scores[key] = None
    return {**scores, "status": "evaluated" if qrels is not None else "missing_ground_truth",
            "retrieved_count": len(ids), "requested_top_k": top_k}
