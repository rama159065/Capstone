# Qdrant on Google Colab — sizing for `competitive_research`

**Domain:** Competitive-research analyst. **Corpus assumed:** Market reports, competitor investor decks, analyst notes, product brochures. **Workload:** 2,000 PDFs × 50–100 pages; **two Qdrant indexes** — (1) `docs_text`: every page is OCR'd and the OCR text is chunked and embedded as text vectors (dense + BM25); (2) `visual_pages`: the ~30% of pages with charts, images or tables are rendered and embedded with ColPali multivectors; all agents run against Qdrant via `RAG_BACKEND=qdrant`.

> **Bottom line:** the Colab **free** account is **not enough** for the full 2,000-document corpus of this domain — persistent storage, ingestion compute time and session limits all fall short (details in “Is the free account enough?”). It is enough for a **pilot of about 583 documents** and for running the agents/API once an index exists.

## Agents that use this index

Shared 9-node domain graph with competitive-research intent terms. The chat LLM is the hosted ≤8B endpoint (Track B), so agents add only ~1 GB RAM on the Colab side (FastAPI + query embedder + reranker). Retrieval tools: `hybrid_retriever` (dense + BM25 + RRF + rerank) and, for visual questions, `/multimodal/competitive_research/visual/retrieve` (ColPali MaxSim).

## Assumptions

| Parameter | Low | Typical | High | Basis |
|---|---|---|---|---|
| Pages per document | 50 | 70 | 100 | User range 50–100 |
| Total pages | 100,000 | 140,000 | 200,000 | × 2,000 docs |
| Words per page (OCR text) | 350 | 350 | 350 | Corpus type estimate |
| Text index source | OCR text | OCR text | OCR text | All pages OCR'd; OCR text is what gets chunked and embedded (not appended to native text) |
| Visual (ColPali) page share | 30% | 30% | 30% | Design choice: ~30% of pages. Typical visual content here: Slide-style pages with market-share charts, product images and comparison tables dominate. |
| Text chunks per page | 1 | 1 | 1 | ceil(words / 350) — code's per-page window |
| Embedding | 384-d | 384-d | 384-d | all-MiniLM-L6-v2 (code default); see dimension table for hosted endpoints |

## What the current code stores (verified in `scripts/domain_common/multimodal_rag.py`)

| Item | Current implementation | Sizing impact |
|---|---|---|
| Text collection `docs_text` | Per page, 400-word chunks with 350-word stride; named vectors `dense` (384-d, all-MiniLM-L6-v2, cosine, `on_disk=True`) + sparse `bm25` (IDF modifier); `on_disk_payload=True` | One point per chunk; payload repeats the whole page as `grounding_text` in every chunk |
| Visual collection `visual_pages` | Only pages flagged visual (tables, images, >4 drawings, empty text, or OCR applied); vectors `summary` (384-d) + `colpali` multivector 128-d, MaxSim, `hnsw m=0`, float32 | ~1,030 × 128 × 4 B ≈ **0.53 MB per visual page** — dominates storage |
| Query path | Hybrid RRF over dense+bm25 (prefetch 30 each); visual: prefetch 30 by `summary`, then ColPali MaxSim rerank | ColPali vectors only read for 30 pages/query → can stay on disk |
| Files outside Qdrant | Copy of source PDF in `objects/<domain>/<doc_id>/`, PNG per visual page (1.5× render), OCR JSON artifact per page, SQLite sidecar | Counted in disk totals below |
| Qdrant mode | Embedded local mode (`QdrantClient(path=...)`) unless `QDRANT_URL` is set | **Local mode is pure Python, holds data in process RAM and warns above 20,000 points** — not usable at this scale; run the Qdrant server binary |
| Text source | `content` = native text + table markdown + caption + OCR text appended | Your design indexes OCR text only → change `ingest()` to chunk `ocr.text` (keeps 1× text, not 2×) |
| Forced OCR on all pages | `visual = ... or ocr.mode != "none"` | **If OCR runs on every page, every page becomes a ColPali page** — decouple visual detection from OCR (see worst-case row) |

## Point counts

| Collection | Low | Typical | High |
|---|---|---|---|
| `docs_text` points (chunks) | 100,000 | 140,000 | 200,000 |
| `visual_pages` points (ColPali pages) | 30,000 | 42,000 | 60,000 |
| ColPali patch vectors (≈1,030/page) | 30,900,000 | 43,260,000 | 61,800,000 |

## Storage required

| Component | Low | Typical | High |
|---|---|---|---|
| `docs_text` (dense + sparse + HNSW + payload) | 1.2 GB | 1.7 GB | 2.5 GB |
| `visual_pages` as coded (float32 multivectors) | 18.5 GB | 25.9 GB | 36.9 GB |
| `visual_pages` optimized (float16 + token pooling ×3) | 3.3 GB | 4.6 GB | 6.6 GB |
| **Qdrant total — as coded** | **19.7 GB** | **27.6 GB** | **39.4 GB** |
| **Qdrant total — optimized** | **4.4 GB** | **6.1 GB** | **8.7 GB** |
| Source PDFs (one copy) | 9.0 GB | 12.6 GB | 18.0 GB |
| Page PNGs (visual pages, 1.5× render) | 7.5 GB | 10.5 GB | 15.0 GB |
| OCR JSON artifacts (every page) | 0.7 GB | 0.9 GB | 1.3 GB |
| **Disk working set — as coded** (Qdrant ×1.5 optimizer headroom, PDF in raw + `objects/`) | **55.7 GB** | **78.0 GB** | **111.5 GB** |
| **Disk working set — optimized** | **23.7 GB** | **33.2 GB** | **47.4 GB** |
| Must persist between sessions (Qdrant snapshot + PDFs) | 13.4 GB | 18.7 GB | 26.7 GB |

Add once per Colab VM (shared by all domains): **~10 GB** for models and extra packages — ColPali v1.3 HF weights (~6 GB), Qwen3-VL-2B Q4 + projector (~1.9 GB), MiniLM embedder and reranker (~0.2 GB), RapidOCR, llama.cpp, bitsandbytes/transformers wheels.

> **Worst case to avoid:** if forced OCR makes every page “visual” (current `ocr.mode != "none"` rule), this domain has 140,000 ColPali pages → Qdrant ≈ **88.1 GB** as coded. Keep visual detection based on tables/images/drawings only.

## RAM required

| Process / phase | Typical RAM | Notes |
|---|---|---|
| Qdrant server — serving, as coded (dense + BM25 index hot) | 1.8 GB | Dense vectors + sparse inverted index in page cache, payload indexes, 1 GB cache |
| Qdrant server — serving, optimized | 0.9 GB | int8 dense copy in RAM, sparse index + ColPali + payload on disk (mmap) |
| Ingestion worker (ColPali 4-bit on T4 + RapidOCR + MiniLM + PyMuPDF) | ~8.5 GB system RAM + ~3 GB VRAM | ColPali checkpoint load peaks at ~6 GB system RAM; fp16 needs ~6.5 GB VRAM |
| Ingestion with Qwen3-VL-2B captions (optional) | +2.5 GB → ~11.9 GB total | Captions are optional; ColPali already covers visual retrieval |
| Agents + FastAPI + Streamlit (LLM hosted) | ~1.0–1.5 GB | Query embedder, cross-encoder reranker, LangGraph state |
| ColPali query encoder for visual retrieval | ~3 GB VRAM (4-bit) or ~12 GB CPU RAM (fp32) | Use the T4; CPU fp32 does not fit free Colab alongside Qdrant |
| Financial report generation | ~0.5 GB | Deterministic; PyMuPDF + matplotlib |
| **Peak, all in one free VM** | **~9.4 GB** (no captions) | Free Colab has ~12.7 GB → run ingest phases separately (text+OCR, then ColPali), never all models at once |

## Embedding dimension sensitivity (typical scenario)

Track B provides a hosted embeddings endpoint; if you use it instead of MiniLM, the dense part grows:

| Dense dimension | `docs_text` size | Qdrant optimized total | Serving RAM (optimized) |
|---|---|---|---|
| 384 | 1.7 GB | 6.1 GB | 0.9 GB |
| 768 | 2.0 GB | 6.4 GB | 1.0 GB |
| 1024 | 2.1 GB | 6.6 GB | 1.0 GB |
| 1536 | 2.5 GB | 7.0 GB | 1.1 GB |

## Ingestion compute time (typical scenario)

| Stage | Free Colab CPU (2 vCPU) | Free Colab T4 GPU |
|---|---|---|
| PyMuPDF text + `find_tables()` + render | 11.7 h | 11.7 h |
| RapidOCR on every page | 77.8 h | 15.6 h (onnxruntime-gpu) |
| ColPali page embeddings | not practical | 7.0 h |
| MiniLM chunk embeddings | 1.0 h | 0.0 h |
| **Total (no captions)** | **97.4 h** (ColPali on GPU) | **34.3 h** |
| Optional Qwen3-VL captions | not practical | 93.3 h |

Rates assumed: 0.3 s/page extraction, 2.0 s/page OCR on CPU (0.4 s on GPU), 0.6 s/page ColPali on T4, 40 chunks/s MiniLM on CPU (800/s on GPU), 8 s/page Qwen3-VL captions. Measure on 50 pages and rescale.

## Is the free account enough?

| Free resource | Needed (typical, optimized) | Verdict |
|---|---|---|
| Colab free system RAM (~12.7 GB) | 9.4 GB peak during ingestion (ColPali + OCR + embedder + Qdrant) | ✅ fits (tight; never load Qwen3-VL or a CPU ColPali query encoder at the same time) |
| Colab free VM disk (~78 GB GPU runtime, ephemeral) | 33.2 GB working set (optimized) | ✅ fits |
| Google Drive free (15 GB, persistent) | 18.7 GB to keep (Qdrant snapshot + PDFs) | ❌ exceeds |
| Qdrant Cloud free (1 GB RAM / 4 GB disk) | 6.1 GB optimized index (as coded is larger) | ❌ exceeds |
| Colab free compute (≤12 h sessions, T4 not guaranteed) | 34.3 h GPU-path ingestion | ❌ needs ~4 sessions of 10 h; checkpointing required |

- **Verdict: No for the full 2,000 documents.** Free Colab can host the Qdrant server and the agents, but the VM disk is wiped at session end, free Drive (15 GB) cannot hold the 18.7 GB to persist, and ~34.3 h of GPU ingestion exceeds a 12-hour session with no guaranteed T4.
- **Free tier is enough for a pilot:** ~583 documents (limited by a 10-hour session; storage alone allows ~1,282, time alone ~583). Use it to confirm the 30% visual ratio and measure recall@k before scaling.
- **Qdrant Cloud free (1 GB RAM / 4 GB disk)** fits only a text-only pilot index; the ColPali collection exceeds it at any realistic scale.
- **Minimum to run the full corpus:** Colab Pro/Pro+ (more RAM/disk, background execution) or a paid VM, plus persistent storage of at least the “must persist” figure × 1.5 (Google One 100 GB/2 TB, a GCS bucket, or a paid Qdrant Cloud cluster with ≥ 4 GB RAM / ≥ 10 GB disk).

## Analysis for 1,000 documents per domain

Same design (OCR text of every page → `docs_text`, ~30% visual pages → `visual_pages` with ColPali), half the corpus. Track B allows 500–2,000 documents, so 1,000 is a valid corpus size. Storage and time scale linearly with pages; Qdrant process RAM and model RAM are mostly fixed.

| Metric (typical, 70 pages/doc) | 1,000 docs | 2,000 docs |
|---|---|---|
| Total pages | 70,000 | 140,000 |
| `docs_text` points | 70,000 | 140,000 |
| `visual_pages` points | 21,000 | 42,000 |
| `docs_text` size | 0.9 GB | 1.7 GB |
| `visual_pages` as coded (float32) | 12.9 GB | 25.9 GB |
| `visual_pages` optimized | 2.3 GB | 4.6 GB |
| **Qdrant total — as coded** | **13.8 GB** | **27.6 GB** |
| **Qdrant total — optimized** | **3.1 GB** | **6.1 GB** |
| Disk working set — as coded | 39.0 GB | 78.0 GB |
| Disk working set — optimized | 16.6 GB | 33.2 GB |
| Must persist (optimized) | 9.4 GB | 18.7 GB |
| Qdrant serving RAM — as coded | 1.6 GB | 1.8 GB |
| Qdrant serving RAM — optimized | 0.9 GB | 0.9 GB |
| Peak ingestion RAM (no captions) | ~9.4 GB | ~9.4 GB |
| GPU-path ingestion time | 17.1 h | 34.3 h |

| 1,000 docs: by pages per document | 50 pages | 70 pages | 100 pages |
|---|---|---|---|
| Qdrant — as coded | 9.9 GB | 13.8 GB | 19.7 GB |
| Qdrant — optimized | 2.2 GB | 3.1 GB | 4.4 GB |
| Disk working set — optimized | 11.9 GB | 16.6 GB | 23.7 GB |
| Must persist — optimized | 6.7 GB | 9.4 GB | 13.4 GB |
| GPU ingestion | 12.2 h | 17.1 h | 24.5 h |

| Free resource (1,000 docs, typical, optimized) | Needed | Verdict |
|---|---|---|
| Colab RAM ~12.7 GB | ~9.4 GB peak | ✅ fits if ingestion phases run separately |
| Colab VM disk ~78 GB (ephemeral) | 16.6 GB (as coded 39.0 GB) | ✅ optimized; as coded fits |
| Google Drive 15 GB (persistent) | 9.4 GB | ✅ tight; Drive is shared with Gmail/Photos |
| Qdrant Cloud free 1 GB RAM / 4 GB disk | 3.1 GB index, ~0.9 GB RAM | ⚠️ disk fits barely, RAM is at the 1 GB limit; no room for growth |
| Ingestion time (≤12 h sessions) | 17.1 h | ❌ ~2 resumable sessions of 10 h |

> **1,000-document verdict:** possible on the free account, but only with the optimized settings, resumable batch ingestion over ~2 sessions, and 9.4 GB of the 15 GB free Drive kept for this index. Not practical as coded (20.1 GB Qdrant + PDFs exceed Drive).

## If this is the one domain implemented

Only this domain is written to Qdrant, so the numbers below are the whole RAG footprint — the all-six totals in the overview do not apply. Provision for the figures here plus ~10 GB for models.

| Provision (typical, optimized) | 1,000 docs | 2,000 docs |
|---|---|---|
| Qdrant storage (`docs_text` + `visual_pages`) | 3.1 GB | 6.1 GB |
| Qdrant storage — as coded | 13.8 GB | 27.6 GB |
| Colab disk working set (+ ~10 GB models) | 16.6 GB | 33.2 GB |
| Persistent storage (snapshot + PDFs, ×1.5 margin) | 14.0 GB | 28.1 GB |
| RAM — Qdrant serving | 0.9 GB | 0.9 GB |
| RAM — peak during ingestion | ~9.4 GB | ~9.4 GB |
| RAM — agents + API while serving (LLM hosted) | ~2.4 GB (+3 GB VRAM for ColPali queries) | ~2.4 GB (+3 GB VRAM for ColPali queries) |
| GPU ingestion time / 10-h sessions | 17.1 h / 2 | 34.3 h / 4 |

Steps to write this domain to the RAG index (Colab, Qdrant server, `RAG_BACKEND=qdrant`):

```bash
# 1. Start Qdrant server and point the code at it (see "Recommended Colab setup")
export QDRANT_URL=http://127.0.0.1:6333 RAG_BACKEND=qdrant RAG_LOCAL_DIR=/content/rag_competitive_research
# 2. Put ONLY this domain's PDFs under <input>/competitive_research/  (ingest loops over every domain folder present)
# 3. OCR text -> docs_text and ColPali (~30% visual pages) -> visual_pages, resumable
#    (do NOT use `run_local_multimodal ingest` on Colab: it launches the Windows llama-server.exe captioner)
python - <<'PY'   # OCR + text + ColPali in ONE pass; no Qwen3-VL captions
from pathlib import Path
import json
from scripts.colpali_index import configure_colpali
configure_colpali()               # MiniLM + ColPali v1.3 (4-bit) from indexes/models
from scripts.domain_common.multimodal_rag import LocalRAG
rag = LocalRAG()
try:
    for pdf in sorted(Path("/content/pdfs/competitive_research").glob("*.pdf")):
        print(json.dumps(rag.ingest("competitive_research", pdf)), flush=True)
finally:
    rag.close()
PY
# 4. Check the visual index
python -m scripts.colpali_index status
# 5. Snapshot both collections to Drive/GCS after every batch
curl -X POST http://127.0.0.1:6333/collections/docs_text/snapshots
curl -X POST http://127.0.0.1:6333/collections/visual_pages/snapshots
# 6. Serve agents on the index
uvicorn scripts.api_app:app --host 127.0.0.1 --port 8000   # /chat/competitive_research, /multimodal/competitive_research/...
```

- Before step 3 apply the code changes listed above: OCR-text-only chunking, visual flag independent of OCR, float16 ColPali, sparse index on disk.
- Validate on 50 documents first: check real pages/doc, visual share vs 30%, and recall@k; then rescale these numbers.

## Alternative approach: local download + batch upload to Colab

**Approach A (above):** all PDFs live on Colab/Drive for the whole run. **Approach B:** PDFs are downloaded and kept on the local PC; a batch is uploaded to Colab, processed (OCR text → `docs_text`, ColPali → `visual_pages`) and written to Qdrant; then the batch's PDFs, PNGs and OCR JSON are deleted on Colab and the next batch is pushed. Qdrant (and its SQLite sidecar) are the only things that accumulate.

| Approach B (typical, optimized) | 1,000 docs | 2,000 docs |
|---|---|---|
| Batch size (≈2 h of GPU processing) | 110 docs (0.7 GB upload) | 110 docs (0.7 GB upload) |
| Number of batches | 10 | 19 |
| Local PC storage — PDFs | 6.3 GB | 12.6 GB |
| Local PC storage — page PNGs (optional, for VLM answers/citations) | 5.2 GB | 10.5 GB |
| **Colab disk peak — Approach A** (all files + Qdrant + models) | **26.6 GB** | **43.2 GB** |
| **Colab disk peak — Approach B** (Qdrant ×1.5 + 2 batches + models) | **17.2 GB** | **21.8 GB** |
| Persistent (Drive) — Approach A (snapshot + all PDFs) | 9.4 GB ✅ | 18.7 GB ❌ |
| **Persistent (Drive) — Approach B (snapshot + SQLite only)** | **3.1 GB** ✅ | **6.1 GB** ✅ |
| Persistent — B, keeping page PNGs too | 8.3 GB ✅ | 16.6 GB ❌ |
| Drive during snapshot rotation (old + new) | 6.1 GB ✅ | 12.2 GB ✅ |
| RAM (Qdrant serving / ingestion peak) | 0.9 GB / ~9.4 GB (unchanged) | 0.9 GB / ~9.4 GB (unchanged) |
| GPU ingestion time / 10-h sessions | 17.1 h / 2 (unchanged) | 34.3 h / 4 (unchanged) |
| Upload time for all PDFs at 10 / 50 / 100 Mbps | 1.4 h / 0.3 h / 0.1 h | 2.8 h / 0.6 h / 0.3 h |
| Snapshot restore at each new session | ~3 min | ~3 min |

### What changes vs Approach A

- **Storage is no longer the blocker:** Drive only needs the Qdrant snapshot (3.1 GB at 1,000 / 6.1 GB at 2,000 docs) instead of snapshot + PDFs; Colab disk stays far below ~78 GB.
- **RAM and GPU time do not change** — the same pages are processed; batching only makes each run fit inside a session and resumable.
- **OCR and ColPali must run in the same ingest pass.** `ocr_index` / `colpali_index backfill` re-read `source.pdf` and page PNGs from `objects/`, which are deleted after each batch — keep `RAG_OCR_ENABLED=1` and call `LocalRAG.ingest()` with ColPali configured (`configure_colpali()`), as in the loop below.
- **Persist `metadata.sqlite3` with every snapshot.** It holds the embedding/ColPali signatures and completed `doc_id`s; restoring Qdrant without it makes `LocalRAG` refuse to start ("Existing index has no embedding signature") or re-ingest everything.
- **Payload file paths become dangling** (`source_uri`, `image_path`, `page_image_uri`, `ocr_artifact_uri` point at deleted Colab files). Text answers and citations still work from the payload text; VLM answers skip missing images with a `visual_evidence_omitted` warning. Store the public PDF URL from `pdf_manifest.jsonl` as `source_url` (needs a small change: `ingest()` only keeps `ticker`/`year` metadata), or keep PNGs on Drive if visual answers are required.
- **Upload path:** sync batches through Google Drive for desktop (or `gsutil` to a bucket) and copy to `/content` in Colab; browser upload (`files.upload`) is too slow for multi-GB batches. Upload the next batch while the current one is processing.
- **Financial report generation** (`generate_financial_report.py`) needs the original PDF with matching `source_sha256` — run it on the local PC or upload just that PDF.

### Batch loop (Colab)

```bash
# once per session: start Qdrant server, restore latest snapshot + metadata.sqlite3 from Drive
export QDRANT_URL=http://127.0.0.1:6333 RAG_BACKEND=qdrant RAG_OCR_ENABLED=1 RAG_LOCAL_DIR=/content/rag_competitive_research
cp /content/drive/MyDrive/rag_competitive_research/metadata.sqlite3 $RAG_LOCAL_DIR/
curl -X POST "http://127.0.0.1:6333/collections/docs_text/snapshots/upload?priority=snapshot" -F snapshot=@/content/drive/MyDrive/rag_competitive_research/docs_text.snapshot
curl -X POST "http://127.0.0.1:6333/collections/visual_pages/snapshots/upload?priority=snapshot" -F snapshot=@/content/drive/MyDrive/rag_competitive_research/visual_pages.snapshot

# per batch (~110 docs)
cp -r /content/drive/MyDrive/upload/batch_007 /content/in/competitive_research            # PDFs synced from the local PC
python - <<'PY'   # OCR + text + ColPali in ONE pass; no Qwen3-VL captions
from pathlib import Path
import json
from scripts.colpali_index import configure_colpali
configure_colpali()               # MiniLM + ColPali v1.3 (4-bit) from indexes/models
from scripts.domain_common.multimodal_rag import LocalRAG
rag = LocalRAG()
try:
    for pdf in sorted(Path("/content/in/competitive_research").glob("*.pdf")):
        print(json.dumps(rag.ingest("competitive_research", pdf)), flush=True)
finally:
    rag.close()
PY
rm -rf /content/in/competitive_research $RAG_LOCAL_DIR/objects/competitive_research/*                     # delete PDFs, PNGs, OCR JSON (to KEEP PNGs see the next section)
curl -X POST http://127.0.0.1:6333/collections/docs_text/snapshots        # snapshot, then copy to Drive
curl -X POST http://127.0.0.1:6333/collections/visual_pages/snapshots
cp $RAG_LOCAL_DIR/metadata.sqlite3 /content/drive/MyDrive/rag_competitive_research/
```

> **Approach B verdict on the free account:** storage fits for this domain at both 1,000 and 2,000 documents (Drive 3.1 GB / 6.1 GB). The remaining limit is compute: 2 sessions at 1,000 docs, 4 at 2,000 docs, with a T4 that is not guaranteed — feasible but slow. 

## Recommended: Qdrant + page images (30% visual pages) kept on Colab/Drive

Same batch flow as Approach B, but each batch's rendered page PNGs (only the ~30% visual pages) are kept next to Qdrant instead of deleted. Source PDFs and OCR JSON are still deleted. This keeps VLM answers over charts/tables (`safe_answer` attaches up to 3 page images) and page-preview citations working, without the PDFs.

| Qdrant + page images (typical, optimized) | 1,000 docs | 2,000 docs |
|---|---|---|
| Page images kept (files) | 21,000 | 42,000 |
| Qdrant (`docs_text` + `visual_pages`) | 3.1 GB | 6.1 GB |
| Page PNGs (1.5× render, ~250 KB each) | 5.2 GB | 10.5 GB |
| Page images as WebP/JPEG q≈85 (~100 KB each, optional) | 2.1 GB | 4.2 GB |
| **Drive persistent — Qdrant + PNG** | **8.3 GB** ✅ | **16.6 GB** ❌ |
| **Drive persistent — Qdrant + WebP** | **5.2 GB** ✅ | **10.3 GB** ✅ |
| Colab disk while serving (Qdrant + PNG + models) | 18.3 GB | 26.6 GB |
| Colab disk peak during ingestion (Qdrant ×1.5 + PNG + 2 batches + models) | 22.5 GB | 32.3 GB |
| RAM | unchanged: Qdrant 0.9 GB; images read from disk, ≤3 per answer | unchanged: Qdrant 0.9 GB; images read from disk, ≤3 per answer |
| Per-batch image archive (tar to Drive) | 0.6 GB | 0.6 GB |
| Session restore (snapshot + images from Drive) | ~5 min | ~8 min |

- **Free Drive verdict:** 1,000 docs fits with PNG (8.3 GB), fits with WebP (5.2 GB); 2,000 docs does not fit with PNG (16.6 GB), fits with WebP (10.3 GB). Colab VM disk is fine in every case. Free Drive's 15 GB is shared with Gmail/Photos; a Google One 100 GB plan removes the limit.
- **Keep paths stable:** payload `image_path` / `page_image_uri` are absolute paths under `RAG_LOCAL_DIR/objects/<domain>/<doc_id>/page_NNNN.png`. Always use the same `RAG_LOCAL_DIR=/content/rag_competitive_research` and restore images to the same place, or the answer step drops them (`visual_evidence_omitted`).
- **Store images as one tar per batch on Drive**, not tens of thousands of single files — Drive is slow with many small files. Extract all tars at session start.
- **WebP/JPEG is optional:** convert after ColPali indexing (ColPali vectors are already stored), then update `image_path`/`page_image_uri` with Qdrant `set_payload` per `doc_id`. Check that chart labels stay readable for the VLM.
- **Also keep a copy on the local PC** (the master store with the PDFs); Drive holds the working copy for Colab.

```bash
# per batch: `touch /content/batch_start` BEFORE the ingest step, then after ingest:
# delete PDFs + OCR JSON, KEEP page PNGs
find $RAG_LOCAL_DIR/objects/competitive_research \( -name source.pdf -o -name 'page_*_ocr.json' \) -delete
rm -rf /content/in/competitive_research
cd $RAG_LOCAL_DIR && find objects/competitive_research -mindepth 1 -maxdepth 1 -newer /content/batch_start -print0 \
  | tar -cf /content/drive/MyDrive/rag_competitive_research/images/batch_007.tar --null -T -   # only this batch's doc_id folders
# snapshot docs_text + visual_pages and copy metadata.sqlite3 to Drive (as in the batch loop above)

# session start: restore images to the SAME path before serving
mkdir -p $RAG_LOCAL_DIR && for t in /content/drive/MyDrive/rag_competitive_research/images/*.tar; do tar -xf "$t" -C $RAG_LOCAL_DIR; done
```

## Recommended Colab setup

```python
# Colab cell: run Qdrant as a real server (NOT embedded local mode)
!mkdir -p /content/qdrant && cd /content/qdrant && \
  curl -sL https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz
!cd /content/qdrant && nohup ./qdrant > qdrant.log 2>&1 &
import os
os.environ["QDRANT_URL"] = "http://127.0.0.1:6333"   # LocalRAG switches to server mode
os.environ["RAG_BACKEND"] = "qdrant"
os.environ["RAG_LOCAL_DIR"] = "/content/rag_competitive_research"  # PNGs, OCR JSON, sqlite sidecar
# Persist: snapshot each collection to Drive at the end of every session
# curl -X POST http://127.0.0.1:6333/collections/docs_text/snapshots
```

```python
# Recommended collection settings (changes to LocalRAG.__init__ / ensure_visual_collection)
models.VectorParams(size=384, distance=models.Distance.COSINE, on_disk=True,
    quantization_config=models.ScalarQuantization(scalar=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8, always_ram=True)))                       # dense: int8 copy in RAM
models.SparseVectorParams(modifier=models.Modifier.IDF,
    index=models.SparseIndexParams(on_disk=True))                             # BM25 index on disk
models.VectorParams(size=128, distance=models.Distance.COSINE, on_disk=True,
    datatype=models.Datatype.FLOAT16,                                         # halves ColPali storage
    multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
    hnsw_config=models.HnswConfigDiff(m=0))                                   # keep: rerank-only, no graph
client.create_payload_index("docs_text", "domain", models.PayloadSchemaType.KEYWORD)
client.create_payload_index("docs_text", "doc_id", models.PayloadSchemaType.KEYWORD)
```

- Ingest in **resumable batches** (the SQLite `documents` table already skips `complete` docs); snapshot Qdrant to Drive/GCS after every batch.
- Ingest with `LocalRAG.ingest()` after `configure_colpali()` (OCR + MiniLM + ColPali per page). Do not use `run_local_multimodal ingest` on Colab — it starts the Windows llama.cpp captioner. If RAM is tight, ingest text first with ColPali off and run `colpali_index backfill` while the files are still on disk.
- Delete raw PDFs after ingestion (or skip the `objects/` copy) to avoid storing every source twice.
- Token pooling (factor 3) is not in the current code; it needs `colpali-engine` or a pooling step before `upsert` and a recall@k re-check. Float16 is a one-line `datatype` change.
- Slide decks often exceed 30% visual pages; check the real ratio on a sample — float16 + token pooling matters most here.

## Formulas used

- text bytes = chunks × (dim×4 + 150 HNSW + sparse≈2×8×0.55×min(400,words) + payload[chunk text + full page `grounding_text` + 1.5 KB metadata]) × 1.15
- visual bytes = visual pages × (ColPali 1,030×128×4 B [opt: 344×128×2 B] + summary dim×4 + 150 + payload[2× page text + 1.5 KB]) × 1.15
- PDF ≈ pages × (30 KB + visual share × 200 KB); PNG ≈ 250 KB/visual page; OCR JSON ≈ 2× page text + 2 KB per page
- Disk working set = Qdrant × 1.5 (segment optimizer needs temporary space) + PDFs + PNGs + OCR JSON
- Free-tier limits: Colab ~12.7 GB RAM, ~78 GB (GPU) / ~107 GB (CPU) ephemeral disk, ≤12 h sessions, T4 not guaranteed; Drive 15 GB; Qdrant Cloud free 1 GB RAM / 4 GB disk. Check current values before relying on them.

Overview of all six domains: [README.md](README.md) / [index.html](index.html).
