# ColPali pooling (1030 → 38 vectors per page) for the Qdrant RAG on a Colab GPU

Status: **analysis + test harness ready; GPU results pending.** Sections 10–12 add Qdrant's published benchmark (checked against their notebooks), the top-20 speed for Colab, and the store → fetch → LLM flow with tested code. Figures marked *measured* were measured in this repo. Figures marked *estimate* need to be confirmed on Colab with [`scripts/evaluation/colab_colpali_pooling_eval.py`](../scripts/evaluation/colab_colpali_pooling_eval.py).

---

## 1. Short answer

| Question | Answer |
|---|---|
| Does pooling help? | **Yes, if it is used as a first stage and the full vectors rerank the results.** Pooled vectors are 27× smaller, can be HNSW-indexed and are about 27× cheaper to score. Above all, they give the visual path its own way to *find* pages. Today, visual candidates come only from a text prefetch. |
| Does it lose information? | **Yes, at the first stage.** Row pooling blurs left–right position, column pooling blurs top–bottom, and both dilute small details such as a single number or a legend colour. Reranking with the full 1030 vectors restores precision for the pages that reach the rerank. The only loss that remains is recall: a page that the prefetch misses can't be reranked. |
| Mean or max? | **Mean.** In Qdrant's notebook, mean pooling kept **NDCG@20 0.952 / Recall@20 0.917** of full ColPali's top 20; max pooling kept only **0.759 / 0.656** (section 10). |
| Top-20 retrieval speed? | **About 0.1–0.25 s per query on Colab with the index warm (*estimate*), versus ~0.7–2 s for an exact full scan.** The search arithmetic itself is measured (19 ms to rerank 200 pages; 454–729 ms for a full scan of 5,880–8,674 pages, 2 CPU threads). Qdrant measured 0.31 s versus 3.75 s per query (12×) on 21k pages over a network (sections 10–11). |
| How are full vectors stored and used? | **On disk (memory-mapped, float16, no index) in Qdrant's storage on the Colab local disk.** Qdrant reads only the ~200 candidates it reranks, and returns point IDs and payload. **Vectors never go to the LLM.** It gets the page text, tables and 1–2 page PNGs (section 12). |
| Replace the full vectors with pooled ones? | **Not for the 100-document corpus.** The full vectors cost 1.2–2.3 GB in float16, which fits easily. At 2,000 documents they cost 23–46 GB, so pooled-only, or full vectors for a subset of pages, becomes a real trade-off (section 7). |
| Is it feasible on Colab? | **Yes.** Pooling takes milliseconds on CPU. The GPU cost is the ColPali encoding itself, which you pay with or without pooling. One T4 session can encode the visual pages of the 100 EDGAR PDFs (section 6). |
| Does it affect the answer the LLM gives? | **No.** Vectors only rank pages. The answer agent receives the payload (`text`, `table_markdown`, OCR text) and the page PNG, and pooling doesn't change either. |

---

## 2. What "1030 → 38" means (measured on ColPali v1.3 in this repo)

ColPali v1.3 resizes every page to 448 × 448 pixels and cuts it into a **32 × 32 grid of 14-pixel patches**. Running the local model on an NVIDIA ARS page gave:

| Token group | Count | Notes |
|---|---|---|
| Image patches | **1,024** | Row-major 32 × 32 grid, first in the sequence |
| Text/special tokens | **6** | `<bos> Describe the image . \n`, the same on every page |
| **Total vectors per page** | **1,030 × 128-d** | What `ColPaliEncoder.embed_image` stores today |

Pooling variants (all keep the 6 special tokens unpooled, then L2-normalise):

| Variant | Vectors/page | KB/page (float16) | What is averaged together |
|---|---|---|---|
| Full (current) | 1,030 | 258 | Nothing |
| **Mean rows** | **38** | 9.5 | The 32 patches across each horizontal strip (1/32 of page height) |
| **Mean columns** | **38** | 9.5 | The 32 patches down each vertical strip |
| Rows + columns | 70 | 17.5 | Both of the above, as one multivector |
| Max rows | 38 | 9.5 | Element-wise max across each row |
| Mean 2 × 2 blocks | 262 | 65.5 | Each 2 × 2 patch block |
| Hierarchical, factor 3 | ~347 | ~87 | Patches that look alike, wherever they are on the page |
| Global mean | 7 | 1.8 | The whole page |

Because image patches come first in the sequence and the specials last, pooled vectors **can be computed from the vectors already stored in Qdrant**. Adding pooling to an existing index does not need the GPU again.

---

## 3. How the current code retrieves visual pages

From [multimodal_rag.py](../scripts/rag/multimodal_rag.py):

| Step | Today | Consequence |
|---|---|---|
| Collection `visual_pages` | `colpali`: 1030 × 128 float32, MaxSim, **`hnsw m=0`**, `on_disk`; `summary`: dense text vector | No graph is built for ColPali. It can only be scored exhaustively or as a rerank. |
| Candidate generation | `Prefetch(query=text_embedding, using="summary", limit=max(30, top_k))` | **Visual candidates come from page text only.** A chart page with little text, or text the OCR misread, may never reach the ColPali rerank. |
| Rerank | Full MaxSim on the ≤30 candidates | Precise and cheap, but only over pages the text already found |
| Encoder | `colpali_retrieval.py` docstring: "never pooled vectors" | Deliberate. Changing it means a new signature and a new `RAG_LOCAL_DIR` |
| Visual page rule | `tables or images or drawings > 4 or no text or OCR` | Marks **59%** of pages in the annual-report PDFs, not the 30% the sizing docs assumed (*measured*, 3,236 pages) |

So in this repo, pooling's main benefit is **not speed**: the rerank already runs over only 30 pages. The benefit is **a visual first stage**, so ColPali can find pages the text search misses.

---

## 4. Advantages, in numbers

### Storage and RAM per page (computed)

| | Full float32 (today) | Full float16 | Mean rows + cols, float16 | Mean rows only, float16 |
|---|---|---|---|---|
| KB/page | 515 | 258 | 19.0 | 9.5 |
| Ratio to today | 1× | 2× smaller | 27× smaller | 54× smaller |

### For the 100 EDGAR PDFs (14,701 pages, *measured*) and the 2,000-document scale

| Visual share | Pages (100 docs) | Full f32 | Full f16 | Pooled rows+cols f16 | Pages (2,000 docs) | Full f16 | Pooled rows+cols f16 |
|---|---|---|---|---|---|---|---|
| 30% (sizing docs) | 4,410 | 2.33 GB | 1.16 GB | 0.09 GB | 88,206 | 23.3 GB | 1.7 GB |
| 40% (rule with drawings > 40) | 5,880 | 3.10 GB | 1.55 GB | 0.11 GB | 117,608 | 31.0 GB | 2.3 GB |
| 59% (current rule) | 8,674 | 4.57 GB | 2.29 GB | 0.17 GB | 173,472 | 45.7 GB | 3.4 GB |
| 100% | 14,701 | 7.75 GB | 3.88 GB | 0.29 GB | 294,020 | 77.5 GB | 5.7 GB |

The 2,000-document column assumes the same 147 pages per document as the 100 EDGAR PDFs.

If the full vectors are kept for reranking, **pooling adds ~7% storage, it does not remove any**. The saving comes from moving the full vectors to disk and keeping only the pooled ones in RAM.

### Compute

| Operation | Full | 38 pooled | Ratio |
|---|---|---|---|
| MaxSim per page per query (20 query tokens) | 2.64 M multiply-adds | 0.10 M | 27× |
| One HNSW build comparison (page vs page) | 1,060,900 dot products | 1,444 | **735×** |
| Exhaustive scan of 5,880 pages | ~15.5 G multiply-adds | ~0.57 G | 27× |

The 735× figure explains why the repo uses `m=0` for full vectors: an HNSW graph over full ColPali vectors is impractical, and over pooled ones it is cheap. Qdrant reports **~13× faster retrieval** and indexing **an order of magnitude faster** with this two-stage setup ([Qdrant blog](https://qdrant.tech/blog/colpali-qdrant-optimization/), [tutorial](https://qdrant.tech/documentation/tutorials/pdf-retrieval-at-scale/)).

---

## 5. Information loss, by content type

Pooling discards **position within the page**. How much that matters depends on what the page contains.

| Content | What row (mean) pooling loses | What column pooling loses | Recovered by full rerank? | Risk |
|---|---|---|---|---|
| **Financial tables** (income statement, segment tables) | **Which column a value is in.** A strip holding "Net sales 391,035 383,285 394,328" becomes one vector, so the year columns can't be told apart. | Which row a value is in: all of the "2024" column becomes one vector. | Yes, for the ranking of candidates. Row and column pooling complement each other, so use both in the prefetch. | Medium. A query about a specific cell may rank lower in the first stage. |
| **Bar / column charts** | Bars run vertically, across many rows, so each row mixes every bar at that height. Bar-to-label alignment is blurred. | Each column keeps roughly one bar from base to top, but loses the height scale on the left axis. | Yes, for pages that reach the rerank | Medium |
| **Line charts / trends** | Rises and falls from left to right are averaged away | Values at a given x-position stay together; slope is partly kept | Partly. ColPali itself sees charts at only 448 px | Medium–high for "which year peaked" questions |
| **Pie / donut charts, legends** | Colour-to-label mapping is diluted: a small legend swatch is 1 of 32 patches in its row | Same | Yes | Medium |
| **Images / photos** (products, people, infographics) | Little: images are large regions, and their meaning survives averaging | Little | n/a | Low |
| **Small numbers and footnotes** | A single distinctive patch counts for 1/32 of the pooled vector (mean) | Same | Yes | Medium. Mean dilutes it; max keeps it but adds noise. |
| **Text-heavy pages** | Low: the text search finds these pages anyway | Low | n/a | Low |

### Loss that pooling does *not* cause
- **448-pixel resolution.** A US Letter page (612 × 792 pt) is resized to 448 × 448, so 9-pt text is about 5 pixels high. Small table figures are hard for ColPali to read even with no pooling. This limit is set before pooling happens, and it is why the payload's `table_markdown` and OCR text, not ColPali, must supply the numbers.
- **Answer content.** The LLM never sees vectors. It reads the payload text and table markdown, and optionally the page PNG through the vision model. Pooling changes which pages are retrieved, not what is said about them.

### Mean versus max
- **Mean** keeps the overall meaning of a row and is what Qdrant evaluated. It reports **NDCG@20 = 0.952 and Recall@20 = 0.917** relative to full ColPali on its dataset ([blog](https://qdrant.tech/blog/colpali-qdrant-optimization/)). That is a different corpus, so our figure has to be measured.
- **Max** keeps rare, strong features such as a bold number, but it mixes dimensions from different patches. The result is a vector the model never produced, and that adds noise to similarity scores. Qdrant's notebook confirms this: max pooling kept NDCG@20 0.759 / Recall@20 0.656 against 0.952 / 0.917 for mean, and Qdrant called max "not viable". The test script still includes `max_rows`, to check on our chart pages.

---

## 6. Feasibility on a Colab GPU

| Item | Value | Basis |
|---|---|---|
| Model | `vidore/colpali-v1.3-hf`, ~3 B parameters, 5.5 GB on disk | *Measured* (local copy in `indexes/models/`) |
| GPU precision | T4: **float16** (no fast bfloat16). L4 / A100: bfloat16. | Script picks it automatically and stops if the output contains NaN or Inf values |
| VRAM | ~6 GB weights + ~1–3 GB activations at batch 4 → fits a 16 GB T4 | *Estimate* |
| Encoding speed | CPU here: **43 s/page** (float32, 16 threads), 118–158 s/page (bfloat16). T4: roughly 0.5–1.5 pages/s | CPU *measured*; T4 *estimate*, the script measures it |
| Pooling cost | Mean/max: < 1 ms/page; hierarchical: ~0.1–0.3 s/page (CPU) | *Estimate*, the script measures it |
| Visual pages of 100 PDFs (5,880–8,674) at 1 page/s | **1.6–2.4 h** | *Estimate*. Fits in one session; save the vectors per batch |
| All 14,701 pages at 1 page/s | ~4.1 h | *Estimate* |
| Qdrant | Server binary on Colab (not embedded local mode); full vectors `on_disk`, pooled in RAM | Same setup as `docs/qdrant_colab_sizing/` |

Pooling adds essentially nothing to the Colab run: the GPU time is spent on encoding, which happens with or without pooling.

---

## 7. Recommended design for writing to RAG

### Collection `visual_pages` (new schema → new `RAG_LOCAL_DIR`, re-ingest)

| Named vector | Type | Index | Storage | Purpose |
|---|---|---|---|---|
| `colpali` | 1030 × 128, MaxSim, **float16** | `hnsw m=0` | `on_disk` | Rerank only (precision) |
| `mean_rows` | 38 × 128, MaxSim, float16 | HNSW | RAM | Visual first stage |
| `mean_cols` | 38 × 128, MaxSim, float16 | HNSW | RAM | Visual first stage; covers what rows miss in tables |
| `summary` | dense text vector | HNSW | RAM | Text first stage (as today) |

The payload is unchanged: `doc_id`, `page`, `image_path`, `text`, `table_markdown`, `source_url`, `ocr_artifact_uri`. Add `pooling: "mean_rows,mean_cols"`.

### Query flow

```text
query ─┬─ prefetch mean_rows (ColPali query vectors, limit 100) ─┐
       ├─ prefetch mean_cols (ColPali query vectors, limit 100) ─┼─► rerank: full `colpali` MaxSim → top-k pages
       └─ prefetch summary   (text embedding,       limit 100) ─┘        │
                                                                         ▼
                                     payload text + table_markdown (+ page PNG) → answer agent
```

Qdrant evaluates nested prefetches in one request, so this is a single `query_points` call. A page is reranked if *any* of the three stages finds it.

### Code changes (to be assessed later; not made yet)

| File | Change |
|---|---|
| `rag/colpali_retrieval.py` | Return the image-token mask (or pooled vectors) with `embed_image`; add the pooling settings to the configuration hash so the existing signature guard forces a new index; update the "never pooled" docstring |
| `rag/multimodal_rag.py` `ensure_visual_collection` | Add `mean_rows` and `mean_cols` named vectors; switch `colpali` to `Datatype.FLOAT16`; extend the schema check |
| `multimodal_rag.py` `index_visual_page` | Upsert all four vectors |
| `multimodal_rag.py` `retrieve_visual` (and the fused path in `retrieve_local`) | Replace the single text prefetch with the three prefetches above |
| `backfill_visual` | Compute pooled vectors from the stored `colpali` vectors, with no GPU needed (section 2) |
| `multimodal_rag.py:329` visual rule | Raise `drawings > 4` (59% of pages) to about `> 40` (~40%), or the threshold the test supports |
| `tests/test_colpali_qdrant.py` | Fake encoder returns 1030 vectors; assert pooled vectors exist, prefetch runs, and the signature changes with the pooling settings |

### 100 versus 2,000 documents

| Corpus | Recommendation |
|---|---|
| **100 EDGAR PDFs** (implemented domain) | Keep full float16 vectors on disk (1.2–2.3 GB) and add pooled first stages (≤0.17 GB). No measurable quality loss is expected because the full vectors decide the final ranking. |
| **2,000 documents** | Full float16 vectors take 23–46 GB, more than the 15 GB of free Drive space. Options: (a) keep full vectors only on the Colab disk and rebuild them from cached `.npz` batches; (b) **pooled-only** (1.7–3.4 GB) with text + `table_markdown` reranking, accepting the measured recall loss; (c) Qdrant's binary or scalar quantization on the full vectors. Decide after the measurement. |

---

## 8. The Colab measurement

[`scripts/evaluation/colab_colpali_pooling_eval.py`](../scripts/evaluation/colab_colpali_pooling_eval.py) is standalone: copy it to Colab along with `data/raw/financial_report_qa/edgar/pdf_manifest.jsonl`. It downloads the PDFs from SEC using the manifest URLs, renders the pages at the ingest scale (1.5×), encodes them once on the GPU, and derives all 8 variants from the same vectors.

```bash
!pip -q install "transformers>=4.57,<5" "qdrant-client>=1.12,<2" pymupdf sentence-transformers scikit-learn
!mkdir -p /content/qdrant && cd /content/qdrant && \
  curl -sL https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz
!cd /content/qdrant && nohup ./qdrant > qdrant.log 2>&1 &
!python colab_colpali_pooling_eval.py --manifest pdf_manifest.jsonl --docs 10 \
    --user-agent "capstone/0.1 you@example.com" --qdrant-url http://127.0.0.1:6333 \
    --queries-csv visual_queries.csv        # optional, recommended
```

| Output | Meaning |
|---|---|
| `R@1`, `R@5`, `MRR`, per content type | Each variant used alone (would pooled-only be good enough?) |
| `keeps_full_top5@100` | Share of full MaxSim's top 5 that a 100-page prefetch keeps (loss caused by the first stage) |
| `rerank_recall@5_from_{20,50,100,200}` | Pooled prefetch → full rerank: the design in section 7 |
| `vs_full_top20@200_recall`, `_ndcg` | **Qdrant's own measurement on our pages:** prefetch 200 → rerank → top 20, compared with exact full ColPali's top 20 (graded relevance 20…1). Directly comparable with 0.917 / 0.952 |
| `text_summary (current prefetch)` | Today's text-only prefetch, for comparison |
| `fused rows+cols+text` | The recommended three-way prefetch |
| `qdrant` | Real upsert time, first-query and p50/p95 latency for **top 20**: exact full scan, pooled only, current text prefetch, Qdrant's rows-200 prefetch, and the recommended three-way prefetch, with recall@20 against the exact scan |
| `encoding` | GPU name, precision and pages/second |

**About the queries.** The automatic queries are distinctive headings from each page, which favours text matching. Add 40–60 hand-written queries describing charts and tables in a CSV (`query,doc_id,page`), for example "bar chart of data center revenue by fiscal year" for a chart page. These test what only the visual path can find.

### Decision rules

| Accept | If |
|---|---|
| Pooled prefetch + full rerank (section 7) | `rerank_recall@5_from_100` is within 0.01 of full `R@5` overall **and** on chart and table queries |
| Adding pooled stages to the text prefetch | Fused prefetch recall exceeds `text_summary` on chart/image queries |
| Mean over max | Unless `max_rows` beats `mean_rows` by > 0.02 on chart queries |
| Pooled-only (no full vectors, 2,000-document case) | `R@5` of `rows_cols` within 0.03 of full on chart + table queries; otherwise keep full vectors or quantize them |

Verified locally: all 8 variants give the expected shapes (1030 / 38 / 38 / 38 / 70 / 262 / 347 / 7) with unit-norm vectors. MaxSim scores match a direct numpy calculation. The Qdrant collection, with four named vectors, float16, `m=0` and nested prefetch, runs end to end in Qdrant's in-memory mode on synthetic data. Page rendering, classification and query generation also run on the real EDGAR PDFs. No real-model GPU results yet.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| float16 overflow on T4 produces NaN vectors | Script stops at the first non-finite output; rerun on L4/A100 (bfloat16) or float32 (~12 GB VRAM) |
| Automatic queries flatter text retrieval | Add hand-written visual queries; report per content type |
| 10-document sample is not the full corpus | Use `--docs 25+` once the first results look stable; prefetch limits scale with corpus size |
| Pooling settings drift between ingest and query | Pooling settings join the ColPali signature; the existing guard refuses mismatched indexes |
| Visual rule too broad (59%) inflates GPU time and storage | Set the threshold from the measured `visual_rule_share` and per-type results |

## 10. Qdrant's published benchmark (checked against their notebooks)

Source: the blog post and the two notebooks in [qdrant/demo-colpali-optimized](https://github.com/qdrant/demo-colpali-optimized). The code and outputs below were read from the notebook files themselves. The blog has no latency figures beyond "13×".

### Their setup

| Item | Value |
|---|---|
| Pages | **20,000+** (about 21k points): ViDoRe + UFO + DocVQA, deduplicated |
| Model | `vidore/colpali-v1.3` (`colpali_engine`), bfloat16, Colab **A100** |
| Qdrant | Qdrant Cloud ("a larger, more robust cluster" than free tier), **4 shards**, queried from Colab over the internet |
| Storage | **Everything in RAM** (`on_disk=False`, `memmap_threshold=0`) "to test retrieval speed without disk caching" |
| Named vectors | `initial` (1030 × 128, MaxSim, binary quantization), `mean_pooling` (38 × 128, binary quantization), `max_pooling` (38 × 128) |
| Pooling code | `torch.mean(emb[:, :1024, :].reshape((B, 32, 32, 128)), dim=2)` + the 6 special tokens → 38 vectors (rows only; columns listed as future work) |
| Upload | batch of 16 pages per `upsert(models.Batch(...))`; **500 pages in 4 min 35 s = 1.81 pages/s** including encoding on the A100 and upload |
| Queries | 1,000 UFO `specific_detail_query` queries, 8 per `query_batch_points` call |
| Ground truth | Exact full ColPali top 20 (`SearchParams(exact=True)` on `initial`). Quality = agreement with it, **not** human relevance judgements |

### Their results (top 20)

| Method | Time for 1,000 queries | Per query | Speed-up | NDCG@20 | Recall@20 |
|---|---|---|---|---|---|
| Exact full ColPali (baseline) | 1 h 02 min 28 s | **3.75 s** | 1× | 1.000 | 1.000 |
| Max pooling, prefetch 200 → full rerank | 5 min 11 s | 0.31 s | 12.0× | 0.759 | 0.656 |
| **Mean pooling, prefetch 200 → full rerank** | 5 min 06 s | **0.31 s** | **12.2×** | **0.952** | **0.917** |
| Mean pooling, query `<pad>` tokens removed | 4 min 55 s | 0.30 s | 12.7× | 0.946 | 0.907 |
| Mean pooling, binary-quantized prefetch | 4 min 46 s | 0.29 s | 13.1× | 0.913 | 0.856 |

How to read these:
- **Network time is included.** About 0.3 s per query includes the round trip from Colab to Qdrant Cloud, so a Qdrant server on the same Colab VM removes most of it.
- **More candidates were reranked than it looks.** Each of the 4 shards prefetched 200 pages, so up to 800 were reranked before the top 20 were picked.
- **Removing `<pad>` tokens from queries is free.** It gave −0.006 NDCG and was slightly faster. Our encoder already does this by dropping masked tokens.
- **Binary quantization of the prefetch costs quality** (−0.04 NDCG, −0.06 recall) for about 6% more speed. Not worth it at our scale.

---

## 11. Top-20 retrieval speed for our setup on Colab

### The arithmetic, measured on this machine (MaxSim in float32, 2 CPU threads, 20 query tokens)

| Operation | 5,880 pages (40% visual) | 8,674 pages (59% visual) | 21,000 pages (Qdrant's size) |
|---|---|---|---|
| Exact full scan, 1030 vectors/page | **454 ms** | **729 ms** | 1,710 ms |
| Exact scan of pooled vectors, 38/page (HNSW is faster still) | 16 ms | 30 ms | 74 ms |
| Rerank 100 candidates with full vectors | 7.9 ms | 7.9 ms | 7.9 ms |
| Rerank 200 candidates with full vectors | **19 ms** | **19 ms** | 19 ms |

The rerank cost depends only on the number of candidates, not on corpus size. That's why the two-stage design stays fast as the corpus grows, while a full scan grows linearly: 1.7 s at 21k pages here, against Qdrant's 3.75 s with network and batching included. Colab's 2 vCPUs are probably 1.5–2× slower than 2 threads on this machine (*estimate*).

### End-to-end top 20 on Colab: Qdrant server on the same VM, T4, index warm (*estimate*)

| Step | Time |
|---|---|
| ColPali query encoding on the GPU (~20 tokens) | 30–80 ms |
| Text query embedding (all-MiniLM-L6-v2) | 5–15 ms |
| Prefetch: `mean_rows` + `mean_cols` HNSW (RAM), `summary` HNSW, 200 each | 10–40 ms |
| Full ColPali rerank of the merged candidates (200–600 unique pages) | 30–110 ms |
| Return 20 points with payload (no vectors) | < 10 ms |
| **Total, warm** | **~0.1–0.25 s** |
| Exact full scan instead (no pooling) | ~0.7–1.5 s, and several seconds when cold (it reads every full vector from disk) |
| First query after a restart (cold disk) | + ~0.2–0.5 s while the ~52 MB of candidate vectors is read (200 × 258 KB) |

To keep the three-way rerank near Qdrant's candidate count, use **100 per prefetch** (≤300 unique pages) unless the Colab eval shows that 200 finds more. The script reports both.

---

## 12. Storing full vectors on disk, fetching them, and what the LLM receives (Colab)

### Who reads what

| Stage | Reads | From | Goes to |
|---|---|---|---|
| Prefetch | `mean_rows`, `mean_cols`, `summary` vectors | RAM (HNSW) | Qdrant only |
| Rerank | Full `colpali` vectors of the **candidates only** | **Disk, memory-mapped** (OS page cache once warm) | Qdrant only |
| Result | Point IDs, scores, **payload** | Payload storage (`on_disk_payload=True`) | Application |
| Answer | `text`, `table_markdown`, OCR text, `source_url`, page; PNG for the top 1–2 pages | Payload + `/content/rag/pages/*.png` | **LLM / vision model** |

So "fetching full vectors from disk" happens **inside Qdrant** during the rerank. The application asks for `with_vectors=False` and never handles 1030 × 128 matrices. The LLM can't use vectors at all: it reads text and images.

### Colab layout for 100 documents (40% visual = 5,880 pages; 59% in brackets)

| Path | Contents | Size |
|---|---|---|
| `/content/rag/qdrant/storage/` (local disk, **not** Drive) | `visual_pages`: full float16 on disk | 1.55 GB (2.29) |
| | pooled rows + cols in RAM, with HNSW | 0.11 GB (0.17) |
| | `summary` vectors + payload text (~4 KB/page) | ~0.03 GB |
| | `docs_text` (all 14,701 pages: dense + sparse + payload) | ~0.1–0.2 GB |
| `/content/rag/pages/<doc_id>/page_NNNN.png` | Page PNGs at 1.5× (*measured* 376 KB/page) | 2.2 GB (3.3) |
| `/content/rag/metadata.sqlite3` | Document and source records, settings signatures | < 10 MB |
| **Drive (persistence)** | Qdrant snapshots + PNG tar + `metadata.sqlite3` | **~4–6 GB of the 15 GB free tier** |

RAM at query time is about 1.6–2.5 GB. That covers pooled vectors, HNSW graphs and the page cache holding the full vectors, out of the ~12.7 GB available. GPU memory is about 6–7 GB for the ColPali query encoder. A local Qwen3-VL-2B answer model (~4–5 GB) also fits on a 16 GB T4.

Don't put Qdrant storage on the Drive mount. Drive is a network filesystem, which is slow for memory-mapped random reads, and it can corrupt storage when a session disconnects. Keep storage on the local disk and copy snapshots to Drive.

### Session lifecycle on Colab

| When | Do |
|---|---|
| Ingest (per batch of PDFs) | render visual pages → ColPali on GPU (batch 4–16) → pool → `upsert` 4 named vectors + payload → write PNGs → `create_snapshot` + copy to Drive every N batches |
| Session start | copy snapshot + PNG tar from Drive to `/content/rag` → start the Qdrant server → `recover_snapshot` → warm up with ~20 queries (or read the vector files once) |
| Query | encode the query (ColPali + MiniLM) → `query_points` with prefetch + rerank → top 20 → fuse with `docs_text` hits → top 5 pages to the answer agent |
| Session end | snapshot → Drive |

Keep the same root path (`/content/rag`) in every session. The repo stores `image_path` as an absolute path ([multimodal_rag.py:369](../scripts/rag/multimodal_rag.py#L369)).

### Code: adapted from Qdrant's notebooks, tested here against an on-disk local Qdrant with synthetic vectors

```python
import torch
from qdrant_client import QdrantClient, models

COLLECTION, GRID, IMAGE_TOKENS = "visual_pages", 32, 1024


def create_collection(client: QdrantClient, text_dim: int = 384) -> None:
    multi = models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
    client.create_collection(
        COLLECTION,
        on_disk_payload=True,
        vectors_config={
            # full 1030 x 128: rerank only -> no HNSW graph (m=0), memory-mapped from disk, float16
            "colpali": models.VectorParams(size=128, distance=models.Distance.COSINE, on_disk=True,
                                           datatype=models.Datatype.FLOAT16, multivector_config=multi,
                                           hnsw_config=models.HnswConfigDiff(m=0)),
            # 38 x 128 pooled: first stage -> HNSW, kept in RAM
            "mean_rows": models.VectorParams(size=128, distance=models.Distance.COSINE, on_disk=False,
                                             datatype=models.Datatype.FLOAT16, multivector_config=multi),
            "mean_cols": models.VectorParams(size=128, distance=models.Distance.COSINE, on_disk=False,
                                             datatype=models.Datatype.FLOAT16, multivector_config=multi),
            # existing text summary vector (all-MiniLM-L6-v2 = 384-d)
            "summary": models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
        },
    )


def pool_batch(embeddings: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """embeddings: (B, 1030, 128) = 1024 image patches then 6 special tokens (Qdrant notebook I)."""
    b = embeddings.shape[0]
    grid = embeddings[:, :IMAGE_TOKENS, :].reshape(b, GRID, GRID, 128)
    special = embeddings[:, IMAGE_TOKENS:, :]
    rows = torch.cat((grid.mean(dim=2), special), dim=1)      # (B, 38, 128): mean across each row
    cols = torch.cat((grid.mean(dim=1), special), dim=1)      # (B, 38, 128): mean down each column
    return rows, cols


def upsert_batch(client, ids, embeddings, summaries, payloads) -> None:
    rows, cols = pool_batch(embeddings)
    client.upsert(COLLECTION, wait=True, points=models.Batch(
        ids=ids, payloads=payloads,
        vectors={"colpali": embeddings.cpu().float().numpy().tolist(),
                 "mean_rows": rows.cpu().float().numpy().tolist(),
                 "mean_cols": cols.cpu().float().numpy().tolist(),
                 "summary": summaries}))


def search_top20(client, query_multivector, query_text_vector, query_filter=None,
                 prefetch_limit=200, top_k=20):
    """Pooled HNSW prefetch (rows, cols) + text prefetch -> exact full ColPali MaxSim rerank -> top 20."""
    prefetch = [
        models.Prefetch(query=query_multivector, using="mean_rows", limit=prefetch_limit, filter=query_filter),
        models.Prefetch(query=query_multivector, using="mean_cols", limit=prefetch_limit, filter=query_filter),
        models.Prefetch(query=query_text_vector, using="summary", limit=prefetch_limit, filter=query_filter),
    ]
    return client.query_points(COLLECTION, prefetch=prefetch, query=query_multivector, using="colpali",
                               query_filter=query_filter, limit=top_k,
                               with_payload=True, with_vectors=False).points   # vectors never leave Qdrant


def llm_context(points, max_pages=5, max_images=2):
    """What the answer model receives: payload text/tables + a few page PNGs, never vectors."""
    pages = []
    for rank, p in enumerate(points[:max_pages], 1):
        pages.append({"rank": rank, "score": p.score, "doc_id": p.payload["doc_id"], "page": p.payload["page"],
                      "source_url": p.payload.get("source_url"), "text": p.payload.get("text", ""),
                      "table_markdown": p.payload.get("table_markdown", []),
                      "image_path": p.payload.get("image_path") if rank <= max_images else None})
    return pages
```

Checked locally against `QdrantClient(path=...)` with 120 synthetic pages:
- The schema reads back as intended: `colpali` on disk, float16, `m=0`; pooled vectors in RAM, float16.
- `search_top20` returns 20 points with the target page at rank 1 and `vector=None`.
- A `doc_id` filter limits results to that document.
- `llm_context` returns payload only.

In the repo these functions belong in `LocalRAG` (section 7). Use `SearchParams(exact=True)` only for the ground-truth measurement.

### What the LLM gets per question

| | Pages | Tokens (≈4 characters each; *measured* ~4,000 characters per EDGAR page) |
|---|---|---|
| Text + tables of the top 5 pages | 5 | ~5,000 (p90 page ~1,750) |
| Page PNGs for chart/table questions | 1–2 | ~1,000 each for Qwen3-VL at 1.5× render (*estimate*) |
| All 20 retrieved pages | 20 | ~20,000 + images. **Too much for a 2B vision model**; use the 20 for fusion and citations, send only 5 |

Feasibility verdict:
- **100 documents:** feasible on free Colab. A T4 encodes the visual pages in about 1.6–2.4 h. The index (~1.8 GB) plus PNGs (~2.2–3.3 GB) fit on the local disk and in the Drive snapshot. Top-20 retrieval should take about 0.1–0.25 s warm.
- **2,000 documents:** needs the choice in section 7. Full vectors take 23–46 GB, more than Drive's free 15 GB.

## Sources

- Qdrant, *Optimizing ColPali for Retrieval at Scale* (mean pooling, 1030 → 38 vectors, NDCG@20 0.952, Recall@20 0.917, ~13× faster): [https://qdrant.tech/blog/colpali-qdrant-optimization/](https://qdrant.tech/blog/colpali-qdrant-optimization/)
- Qdrant tutorial, *Multivector Document Retrieval with ColPali/ColQwen* (row/column mean pooling, HNSW on pooled vectors, rerank with originals): [https://qdrant.tech/documentation/tutorials/pdf-retrieval-at-scale/](https://qdrant.tech/documentation/tutorials/pdf-retrieval-at-scale/)
- Qdrant, *Multivector representations for reranking*: [https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/](https://qdrant.tech/documentation/tutorials-search-engineering/using-multivector-representations/)
- Qdrant demo code and outputs, notebooks *ColPali as a reranker I / II* (collection config, pooling, batch upload, prefetch queries, timings): [https://github.com/qdrant/demo-colpali-optimized](https://github.com/qdrant/demo-colpali-optimized)
