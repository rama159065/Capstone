# Annual-report RAG: structure for the 1,004-filing corpus

Updated 11 November 2026. How the plan in [annual_report_corpus_plan.html](annual_report_corpus_plan.html) (1,004 Form ARS PDFs, 281 companies, 13 sectors, 11.3 GB; or the 509-filing `min500` preset) becomes a searchable index that the financial agent and the API answer from.

There are **two pipelines over the same corpus**:

- **Pipeline 1, Chroma (text index):** every page becomes text, so the text-only Track B model (Qwen3-8B) can read all of it. This is the main index. Code: [financial_ingest_chroma_db.py](../scripts/rag/financial_ingest_chroma_db.py). Notebook: [colab_ingest_chroma_db.ipynb](../notebooks/colab_ingest_chroma_db.ipynb).
- **Pipeline 2, Qdrant (text + ColPali index):** text and OCR chunks plus ColPali page vectors that find table, chart and image pages from their pixels. Ready for a vision chat model. Code: [financial_ingest_qdrant_db.py](../scripts/rag/financial_ingest_qdrant_db.py). Notebook: [colab_ingest_qdrant_db.ipynb](../notebooks/colab_ingest_qdrant_db.ipynb).

Both are driven by [scripts/rag/colab_pipeline.py](../scripts/rag/colab_pipeline.py) (pipeline names `vlm_chroma` and `colpali_qdrant`). The older [colab_rag_pipelines.ipynb](../notebooks/colab_rag_pipelines.ipynb) runs either one behind a `PIPELINE` switch.

```
corpus_plan.jsonl ──download──▶ Drive: annual_reports/<sector>/ ──stage──▶ local disk ──analyse pages──┬──▶ Pipeline 1 ──▶ Chroma text index ──┐
   (1,004 rows)                  PDFs + one manifest per sector        (per sector)    (+ page audit)   └──▶ Pipeline 2 ──▶ Qdrant text + ColPali ┴──▶ API / agent / UI
                                                                                                                                              (ticker, year, sector filters)
```

## 1. Stages

| Stage | Code | What it does | Resumable by |
|---|---|---|---|
| Plan | `data/raw/financial_report_qa/annual_reports/corpus_plan.jsonl` | 1,004 rows: sector, ticker, year, EDGAR URL, bytes. Every company is in `scripts/common/companies.py` | — |
| Download | `scripts/downloads/download_annual_report_corpus.py` (`cp.download`) | Fetches each PDF from EDGAR at ≤8 requests/s with a contact e-mail, checks the `%PDF-` header and size, and records SHA-256 and pages in the sector manifest | skipping PDFs already listed with the right size |
| Page audit (optional) | `scripts/rag/page_audit.py` (`cp.page_audit`, [colab_page_audit.ipynb](../notebooks/colab_page_audit.ipynb)) | One CSV per PDF (`pageNumber,parserUsage`): which reader each page needs. CPU only, about 0.22 s per page per core | filings that already have a CSV are skipped |
| Stage | `cp.stage` | Copies one sector's PDFs and manifest from Drive to the VM's local disk before ingesting | skipping files with the same size |
| Ingest | `cp.ingest` → **Pipeline 1** `financial_ingest_chroma_db.VlmIngest` or **Pipeline 2** `financial_ingest_qdrant_db.Ingest` | One sector at a time; models loaded once for all sectors | per-filing state in SQLite; finished filings are skipped |
| Back up | Chroma folder copy (Pipeline 1) or Qdrant snapshot (Pipeline 2) to Drive | After each sector | `cp.restore` at the start of the next session |
| Check | `cp.status`, `cp.coverage` | Chunks or points per sector and failures; every planned filing as planned → downloaded → ingested / pending / failed | — |
| Serve | `cp.serve` | FastAPI (one worker) + Streamlit on localhost, with `cp.serving_env(s)` pointing them at this index | — |

The plan is complete when `cp.coverage(s)["total"]["ingested"]` equals the preset's count (509 for `min500`, 1,004 for `full`).

## 2. Two pipelines, same corpus

Both read the same sector manifests, run the same page analysis (`financial_ingest_qdrant_db.analyse_page`) and store **one chunk schema** ([chunk_schema.py](../scripts/rag/chunk_schema.py)): the filing (`doc_id` = `edgar_ars_<TICKER>_<YEAR>`, `ticker`, `fiscal_year`, `sector`, `sector_slug`…), the page (`page`, `section`, `scale`, `page_type`, `page_route`), the chunk (citable `chunk_id`, `content_type`, `text`, `table_markdown`) and its provenance (`pipeline`, `text_source`, `reader`, `reader_model`, `escalation`, `number_check`, `unverified_numbers`).

| | **Pipeline 1: Chroma** (`vlm_chroma`) | **Pipeline 2: Qdrant** (`colpali_qdrant`) |
|---|---|---|
| Purpose | Every page as text, for the text-only Qwen3-8B; the main index | Text plus page-image search; ready for a vision chat model |
| Plain text pages | PDF text layer | PDF text layer |
| Tables with a text layer | Text layer + native tables (`find_tables`) as Markdown | Text layer + native tables as Markdown, and ColPali page vectors |
| Garbled text, glyphs drawn as outlines, scans | RapidOCR | RapidOCR (all of them: `ocr_max_pages_per_filing=0`) |
| Charts without printed values, infographics | **Qwen3-VL** (default 4B; 2B or 8B), figures checked against the text layer | ColPali page vectors; text from the text layer or OCR |
| Tables, charts or pictures with an unusable text layer (`paidVL`) | OCR → Qwen3-VL, every figure checked against the OCR text → a **paid vision model** only when more than 10% are unconfirmed | ColPali page vectors; text from OCR |
| Model failure | A failed or empty Qwen3-VL or paid reading falls back to OCR; the filing still completes | — |
| Blank pages, photos and logos | Skipped (`none`) | Not indexed when they have no text |
| Finding chart and image pages | A Qwen3-VL **search description** chunk per chart or image page (kind, title, measures, series, years; no values) | ColPali page vectors (MaxSim) |
| Stored in | Chroma `annual_reports_vlm`: MiniLM vectors (384-d) + a BM25 file `chroma/bm25.sqlite3` (SQLite FTS5, number-aware terms) | `annual_reports_text`: dense (MiniLM, RAM, HNSW) + BM25 sparse (IDF); `annual_reports_visual`: `mean_rows` + `mean_cols` (38 × 128 each, fp16, RAM, HNSW), full `colpali` (1,030 × 128, fp16, on disk, no HNSW), `summary` |
| Chunking | 1,200-character windows, 200 overlap; tables split up to 3,000 characters with their title and header | the same |
| Retrieval | Dense + BM25, fused by RRF (`scripts/rag/hybrid.py`) | Dense + BM25, fused by RRF; with `RAG_FIN_VISUAL=1` also pooled ColPali prefetch → full MaxSim rerank |
| Rerank | Cross-encoder when `RAG_RERANKER_MODEL` is set | the same |
| Agent backend | `RAG_BACKEND=financial_vlm` (`RAG_VLM_DIR`) | `RAG_BACKEND=financial_qdrant` (`RAG_INGEST_DIR`) |

**Readers per page in Pipeline 1** (`--vlm routed`, the default) follow the page audit rules (`page_audit.route`):

```
page analysis ─┬─ none ──────────────────────────────▶ skipped
               ├─ pdfParser ─────────────────────────▶ text layer
               ├─ pdfParser+colpali ─────────────────▶ text layer + native tables (Markdown)
               ├─ ocr ───────────────────────────────▶ RapidOCR
               ├─ qwen3VL ───────────────────────────▶ Qwen3-VL ── failed or empty ──▶ OCR
               └─ paidVL ─▶ OCR ─▶ Qwen3-VL ─▶ figures confirmed by OCR?
                                                 ├─ ≤ 10% unconfirmed ─▶ keep Qwen3-VL
                                                 └─ > 10% ─▶ paid model (if configured) ─▶ keep the reading with fewer unconfirmed figures
```

**Finding chart pages:** chart and image pages routed `pdfParser+colpali`, `qwen3VL` or `paidVL` also get a **Qwen3-VL search description** (`DESCRIBE_PROMPT`): the kind of visual, its title, what it measures, the series and the years, with no values. It is stored as an extra `page_description` chunk (e.g. *"Chart on page 2 (page description): Bar chart titled Revenue by segment…"*), searched by dense and BM25 like any other chunk. Pages routed to OCR (outlined text, scans) and skipped pages are not described; a failed description is skipped; `--no-describe-visuals` turns it off. Figures in a description that the page doesn't print are flagged in `unverified_numbers`. Tested on CPU with Qwen3-VL-2B (llama.cpp): about 54 s per page, e.g. Walmart 2025 p2 → *"an infographic titled 'A message from our CEO'… revenues, operating income, … eCommerce sales, advertising, membership income"*.

Every reading is cached on Drive per page and reader (`vlm_pages/`), so a rerun never pays twice, and changing the paid model or the threshold reuses the OCR and Qwen3-VL readings. On 20 EDGAR filings (3,316 pages) the page audit routed 79.9% of pages to `pdfParser`, 8.6% to `pdfParser+colpali`, 8.6% to `ocr`, 0.8% to `qwen3VL` and 2.2% to `paidVL`.

**Unreadable pages** are found by two checks (`analyse_page`): garbled English (fewer than 5% stopwords among 40+ words, or `�` characters) and font-encoding garbage (`non_ascii_garbage`: more than 30% non-ASCII among 40+ letters and symbols, e.g. Costco 2024's `èçé ´±½¿¬`). Pages whose text is drawn as vector outlines (Costco 2025: about 3,000 shapes per page, no text layer) also go to OCR. A garbled text layer is never indexed.

**Collection names** are this corpus's own (`annual_reports_text`, `annual_reports_visual`, `annual_reports_vlm`), so `LocalRAG` (`scripts/rag/multimodal_rag.py`, `docs_text` / `visual_pages`) can share a Qdrant server. Snapshots from before the rename (`docs_text-*.snapshot`) are restored into the new names.

## 3. Where everything lives

**Google Drive** (`MyDrive/capstone_rag/`, durable; paid storage):

```
capstone_code.zip                     code (python -m scripts.common.colab_bundle)
annual_reports/
    corpus_plan.jsonl                 copied from the repo
    pdf_manifest.jsonl                all sectors (local_path = <sector>/<file>.pdf)
    banking/pdf_manifest.jsonl        this sector (local_path = <file>.pdf)
    banking/EDGAR_ARS_JPM_2025.pdf
    ... 13 sector folders
indexes/vlm_chroma/                   Pipeline 1: chroma/ (+ bm25.sqlite3), vlm_ingest.sqlite3, pages/
vlm_pages/                            Pipeline 1 readings (OCR, Qwen3-VL, paid), one JSONL per filing (append-only)
snapshots/colpali_qdrant/             Pipeline 2: annual_reports_text-*.snapshot, annual_reports_visual-*.snapshot, ingest.sqlite3, pages/
page_audit/<sector>/<pdf>.csv         page audit (pageNumber,parserUsage) + _summary.csv / _summary.json
```

**Colab local disk** (`/content/rag/`, fast, lost at disconnect):

```
annual_reports/<sector>/              the staged PDFs being ingested
vlm_chroma/                           Pipeline 1: chroma/, vlm_ingest.sqlite3, pages/
colpali_qdrant/                       Pipeline 2: ingest.sqlite3, pages/, qdrant_storage/, snapshots/
bin/qdrant, qdrant.log, logs/         Qdrant server binary; API and Streamlit logs
```

Smoke runs (`SMOKE_TEST = True`) write to `<pipeline>_smoke` folders (and embedded Qdrant) and never touch the real index.

**On your PC** (the `local` runtime profile): `indexes/financial_vlm` (Pipeline 1, Chroma) and `indexes/financial_qdrant` (Pipeline 2, embedded Qdrant). The folder names still differ between environments; [deployment.md](deployment.md) §7 analyses this and proposes one layout.

## 4. Serving

Deployment (Docker Compose, a VM, HTTPS, paths, settings and costs): [deployment.md](deployment.md).

[scripts/common/runtime_profile.py](../scripts/common/runtime_profile.py) fills the defaults; anything you set yourself wins:

| | local | colab |
|---|---|---|
| `RAG_VLM_DIR` (Pipeline 1) | `indexes/financial_vlm` | `/content/rag/vlm_chroma` |
| `RAG_INGEST_DIR` (Pipeline 2) | `indexes/financial_qdrant` | `/content/rag/colpali_qdrant` |
| `QDRANT_URL` | unset (embedded) | `http://127.0.0.1:6333` |
| `RAG_BACKEND` | `financial_qdrant` if its `ingest.sqlite3` exists, else `financial_vlm` if `vlm_ingest.sqlite3` exists, else the older Chroma index | same |

When both indexes exist, the automatic choice still prefers Pipeline 2. To serve Pipeline 1, set `RAG_BACKEND=financial_vlm` (the Chroma notebook's `cp.serve` does this).

The Streamlit sidebar has a **Sectors** picker (the 13 sectors) and a **Retrieval mode** (`hybrid`, or `dense` / `bm25` for comparisons). `/chat/financial_report_qa`, `/conversation/financial_report_qa` and `/retrieve/financial_report_qa` accept `ticker`, `year` and `sector`; the filters apply to both pipelines. Each evidence chunk sent to the LLM shows how it matched (`matched_by: bm25 #1, dense #8`).

## 5. Commands

```bash
# Pipeline 1: Chroma text index (routed readers; the paid model needs RAG_PAID_VISION_API_BASE / _MODEL / _API_KEY)
python -m scripts.rag.colab_pipeline run --pipeline vlm_chroma --corpus-preset min500 --max-per-sector 0
python -m scripts.rag.financial_ingest_chroma_db ingest --manifest data/raw/financial_report_qa/annual_reports --sectors banking --vlm routed
python -m scripts.rag.financial_ingest_chroma_db query "net interest income" --sector banking

# Pipeline 2: Qdrant text + ColPali index
python -m scripts.rag.colab_pipeline run --pipeline colpali_qdrant --corpus-preset min500 --max-per-sector 0
python -m scripts.rag.colab_pipeline query --pipeline colpali_qdrant "net interest income" --sector banking

# Both
python -m scripts.rag.colab_pipeline coverage --pipeline vlm_chroma
python -m scripts.rag.page_audit --manifest data/raw/financial_report_qa/annual_reports --corpus-preset min500
# locally, embedded Qdrant and offline vectors:
python -m scripts.rag.colab_pipeline run --pipeline colpali_qdrant --qdrant-url "" --embedding-provider hashing --sectors banking --max-per-sector 2
```

## 6. Sizes (page mix measured on the 100 local filings; scaled to ~147,600 pages)

| Item | Size |
|---|---|
| PDFs | 11.3 GB full, 5.4 GB min500 (Drive, and staged per sector on local disk) |
| Page mix (measured: 14,701 pages) | text or blank 68.9%, table 10.4%, chart or image 7.7%, unreadable 13.1% |
| Pipeline 1 model pages | about 3% of pages need Qwen3-VL or the paid model (page audit, 20 filings); about 10% need OCR |
| Pipeline 2 ColPali full vectors | ~264 KB per selected page: **~7 GB** for table, chart and image pages (18.0%, ~26,600 pages) |
| Pipeline 2 pooled vectors (RAM) | ~19 KB per selected page: **~0.5 GB** |
| Snapshot rotation | the new snapshot is written before the old one is deleted: up to 2× on Drive and a local copy on the VM |

## 7. Known limits

- **Qwen3-VL and the paid model haven't run on a GPU yet;** routing, OCR, cross-checks and fallbacks are tested with real OCR and stand-in models. The first Colab run is the real test. One page at a time with `transformers` is slow at full scale; serve Qwen3-VL with vLLM and use `--qwen-reader openai`.
- **Mixed pages:** a text page with a small embedded chart is read through its text layer only, so values that exist only as pixels are missed.
- **Not implemented** from [financial_rag_final_design.md](financial_rag_final_design.md): the 10-K HTML/iXBRL text source and the facts DB. Numbers come from the ARS PDFs only.
