# Annual-report RAG: structure for the 1,004-filing corpus

How the plan in [annual_report_corpus_plan.html](annual_report_corpus_plan.html) (1,004 Form ARS PDFs, 281 companies, 13 sectors, 11.3 GB) becomes a searchable index that the financial agent and API answer from. One notebook runs it end to end: [notebooks/colab_rag_pipelines.ipynb](../notebooks/colab_rag_pipelines.ipynb). All its logic is in [scripts/rag/colab_pipeline.py](../scripts/rag/colab_pipeline.py).

```
corpus_plan.jsonl ──download──▶ Drive: annual_reports/<sector>/ ──stage──▶ local disk ──ingest──▶ index ──serve──▶ API / agent / UI
   (1,004 rows)                  PDFs + one manifest per sector        (per sector)      (2 pipelines)       (ticker, year, sector filters)
```

## 1. Stages

| Stage | Code | What it does | Resumable by |
|---|---|---|---|
| Plan | `data/raw/financial_report_qa/annual_reports/corpus_plan.jsonl` | 1,004 rows: sector, ticker, year, EDGAR URL, bytes. Every company is in `scripts/common/companies.py` | — |
| Download | `scripts/downloads/download_annual_report_corpus.py` (`cp.download`) | Fetches each PDF from EDGAR at ≤8 requests/s with a contact e-mail, checks the `%PDF-` header and size, and records SHA-256 and pages in the sector manifest | skipping PDFs already listed with the right size |
| Stage | `cp.stage` | Copies one sector's PDFs and manifest from Drive to the VM's local disk before ingesting | skipping files with the same size |
| Ingest | `cp.ingest` → `financial_ingest_qdrant_db.Ingest` or `financial_ingest_chroma_db.VlmIngest` | One sector at a time; models loaded once for all sectors | per-filing state in SQLite; finished filings are skipped |
| Back up | snapshot (Qdrant) or folder copy (Chroma) to Drive | After each sector (`snapshot_every=0`) | `cp.restore` at the start of the next session |
| Check | `cp.status`, `cp.coverage` | Points per sector and failures; every planned filing as planned → downloaded → ingested / pending / failed | — |
| Serve | `cp.serve` | FastAPI (one worker) + Streamlit on localhost, with `cp.serving_env(s)` pointing them at this index | — |

The plan is complete when `cp.coverage(s)["total"]["ingested"] == 1004`.

## 2. Two pipelines, same corpus

Choose one with `PIPELINE` in the notebook. Both read the same sector manifests, and every chunk carries the same filing payload (`doc_id` = `edgar_ars_<TICKER>_<YEAR>`, `ticker`, `fiscal_year`, `sector`, `sector_slug`, `page`, citable `chunk_id`).

| | `colpali_qdrant` ([financial_ingest_qdrant_db.py](../scripts/rag/financial_ingest_qdrant_db.py)) | `vlm_chroma` ([financial_ingest_chroma_db.py](../scripts/rag/financial_ingest_chroma_db.py)) |
|---|---|---|
| Table, chart and image pages | ColPali v1.3 page vectors in Qdrant `annual_reports_visual` | Qwen2.5-VL transcribes them to Markdown text |
| Unreadable text pages | RapidOCR text into `annual_reports_text` (all of them: `ocr_max_pages_per_filing=0`) | the VLM transcribes them too (`vlm_pages="visual"`) |
| All text | Qdrant `annual_reports_text`: dense (MiniLM, 384-d, RAM, HNSW) + number-aware BM25 sparse; 1,200-char windows, 200 overlap, tables split with their header | Chroma `annual_reports_vlm`: text embeddings of text-layer chunks and VLM Markdown |
| Visual vectors | `mean_rows` + `mean_cols` (38 × 128 each, float16, RAM, HNSW) for the first stage; full `colpali` (1030 × 128, float16, on disk, no HNSW) for the MaxSim rerank | none: everything is text |
| Retrieval | dense + sparse fused with Qdrant RRF; with `RAG_FIN_VISUAL=1` also pooled prefetch → full MaxSim | dense search in Chroma |
| Agent backend | `RAG_BACKEND=financial_qdrant` | `RAG_BACKEND=financial_vlm` |

**Unreadable pages** are found by two checks (`financial_ingest_qdrant_db.analyse_page`): garbled English (fewer than 5% stopwords among 40+ words, or `�` characters) and font-encoding garbage (`non_ascii_garbage`: more than 30% non-ASCII among 40+ letters and symbols, e.g. Costco 2024's `èçé ´±½¿¬`). On the 100 local PDFs the second check adds 155 pages, all garbage. Their text layer is never indexed: it is replaced by OCR (pipeline 1) or a VLM transcription (pipeline 2).

**Collection names** are this corpus's own (`annual_reports_text`, `annual_reports_visual`), so `LocalRAG` (`scripts/rag/multimodal_rag.py`, `docs_text` / `visual_pages`) can share the Qdrant server. Snapshots from before the rename (`docs_text-*.snapshot`) are restored into the new names; an old live index stops with a message.

Page selection is shared (`financial_ingest_qdrant_db.visual_targets`): `tables_charts_images` (pages classified table, chart or image), `tables_charts`, `rule`/`visual` (those plus unreadable pages) or `all`.

## 3. Where everything lives

**Google Drive** (`MyDrive/capstone_rag/`, durable; paid storage):

```
capstone_code.zip                     code (python -m scripts.common.colab_bundle)
annual_reports/
    corpus_plan.jsonl                 copied from the repo
    pdf_manifest.jsonl                all sectors (local_path = <sector>/<file>.pdf)
    download_summary.json
    banking/pdf_manifest.jsonl        this sector (local_path = <file>.pdf)
    banking/EDGAR_ARS_JPM_2025.pdf
    ... 13 sector folders
snapshots/colpali_qdrant/             annual_reports_text-*.snapshot, annual_reports_visual-*.snapshot, ingest.sqlite3, pages/
indexes/vlm_chroma/                   chroma/, vlm_ingest.sqlite3, pages/
vlm_pages/                            VLM transcriptions, one JSONL per filing (append-only)
```

**Colab local disk** (`/content/rag/`, fast, lost at disconnect):

```
annual_reports/<sector>/              the staged PDFs being ingested
colpali_qdrant/                       ingest.sqlite3, pages/ (page analysis cache), qdrant_storage/, snapshots/
vlm_chroma/                           chroma/, vlm_ingest.sqlite3, pages/
bin/qdrant, qdrant.log, logs/         Qdrant server binary; API and Streamlit logs
```

Smoke runs (`SMOKE_TEST = True`) write to `<pipeline>_smoke` folders and embedded Qdrant, and never touch the real index.

**On your PC** (the `local` runtime profile): `indexes/financial_qdrant` (ColPali/Qdrant, embedded) and `indexes/financial_vlm` (Chroma). Copy a Colab index there, or ingest locally with `python -m scripts.rag.colab_pipeline ingest --qdrant-url ""`.

## 4. Serving

Deployment (Docker Compose, a VM, HTTPS, settings and costs): [deployment.md](deployment.md).

[scripts/common/runtime_profile.py](../scripts/common/runtime_profile.py) fills the defaults; anything you set yourself wins:

| | local | colab |
|---|---|---|
| `RAG_INGEST_DIR` | `indexes/financial_qdrant` | `/content/rag/colpali_qdrant` |
| `RAG_VLM_DIR` | `indexes/financial_vlm` | `/content/rag/vlm_chroma` |
| `QDRANT_URL` | unset (embedded) | `http://127.0.0.1:6333` |
| `RAG_BACKEND` | `financial_qdrant` if its `ingest.sqlite3` exists, else `financial_vlm` if `vlm_ingest.sqlite3` exists, else the older Chroma index | same |

The Streamlit sidebar has a **Financial sectors** picker (the 13 sectors). `/chat/financial_report_qa`, `/conversation/financial_report_qa` and `/retrieve/financial_report_qa` accept `ticker`, `year` and `sector` (`"banking"`, `"Oil and gas"`, or several, comma-separated). The sector filter applies to both backends above. The older Chroma and `LocalRAG` indexes have no sector field and ignore it.

## 5. Commands

```bash
python -m scripts.rag.colab_pipeline run --pipeline colpali_qdrant --max-per-sector 0          # everything, all 13 sectors
python -m scripts.rag.colab_pipeline coverage --pipeline colpali_qdrant                          # planned -> ingested by sector
python -m scripts.rag.colab_pipeline status --pipeline vlm_chroma
python -m scripts.rag.colab_pipeline query --pipeline colpali_qdrant "net interest income" --sector banking
# locally, embedded Qdrant and offline vectors:
python -m scripts.rag.colab_pipeline run --pipeline colpali_qdrant --qdrant-url "" --embedding-provider hashing --sectors banking --max-per-sector 2
```

## 6. Sizes (page mix measured on the 100 local filings; scaled to ~147,600 pages)

| Item | Size |
|---|---|
| PDFs | 11.3 GB (Drive, and staged per sector on local disk) |
| Page mix (measured: 14,701 pages) | text or blank 68.9%, table 10.4%, chart or image 7.7%, unreadable 13.1% |
| ColPali full vectors | ~264 KB per selected page: **~7 GB** for table, chart and image pages (18.0%, ~26,600 pages) |
| Pooled vectors (RAM) | ~19 KB per selected page: **~0.5 GB** |
| Snapshot rotation | the new snapshot is written before the old one is deleted: up to 2× on Drive and a local copy on the VM |

## 7. Known limits

- **VLM at full scale:** one page at a time with `transformers` is too slow for the ~46,000 pages `vlm_pages="visual"` selects (tables, charts, images and unreadable pages); serve Qwen2.5-VL with vLLM and use `vlm="openai"`.
- **Not implemented** from [financial_rag_final_design.md](financial_rag_final_design.md): the 10-K HTML/iXBRL text source and the facts DB. Numbers come from the ARS PDFs only.
