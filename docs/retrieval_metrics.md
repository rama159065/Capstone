
All RAG API retrieval and chat responses include `retrieval_metrics` with
`recall@1`, `recall@5`, `recall@10`, and `ndcg@5`. Supply complete ground-truth
judgments in the request as `qrels` (chunk/page ID to nonnegative relevance
grade), or `relevant_chunk_ids` for binary relevance. Graded `qrels` take
precedence. For example:

```json
{"query": "What is the leave policy?", "top_k": 10,
 "qrels": {"relevant-chunk-id": 2, "another-relevant-id": 1}}
```

Recall is the fraction of all positively judged IDs retrieved at the cutoff.
nDCG@5 uses linear relevance gains discounted by log2(rank + 1), matching
ViDoRe's pytrec_eval `ndcg_cut_5` convention. Scores are in [0, 1]. These are
metrics on your supplied judgments, not a claim of running the ViDoRe benchmark.
Duplicates receive credit once. Explicit empty judgments score zero; missing
judgments produce null scores with `status: missing_ground_truth`. Cutoffs
above the requested `top_k` are null, so use `top_k: 10` to measure all four.
The ranking evaluated is the evidence returned by the workflow, including any
filtering it performs; no extra generation or retrieval is triggered for scoring.

`python scripts/evaluation/evaluate_financial_rag.py` also reports these metrics per query
and as macro averages, and accepts `qrels` in evaluation rows in addition to the
existing `relevant_chunk_ids` format. Its previous hit-rate recall calculation
has been corrected to fractional recall. The shared Python scorer is available
as `scripts.rag.retrieval_metrics.retrieval_metrics` for other offline
retrieval experiments.
