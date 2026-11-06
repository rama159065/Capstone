# PDF ingestion into Qdrant on Colab

Date: 2026-11-04. This implements the PDF part of [financial_rag_final_design.md](financial_rag_final_design.md), sections 5 and 7.

- Code: [scripts/rag/financial_ingest.py](../scripts/rag/financial_ingest.py)
- Colab session: [notebooks/colab_financial_ingest.ipynb](../notebooks/colab_financial_ingest.ipynb)
- Tests: [tests/test_financial_ingest.py](../tests/test_financial_ingest.py)

**Not covered here:** the 10-K HTML / XBRL facts DB (store A) is a separate step.

---

## 0. Getting the data (EDGAR → Drive)

[scripts/downloads/download_edgar_corpus.py](../scripts/downloads/download_edgar_corpus.py) runs as notebook cell 6. For each of the 100 filings in `pdf_manifest.jsonl` it downloads:

| Folder | Content | 100 filings (**measured**) |
|---|---|---|
| `pdfs/` | ARS PDF (SHA-256 checked; a mismatching file is deleted) | 976 MB |
| `10k/` | the 10-K for the same fiscal year, filed on or before the ARS (amendments ignored): primary iXBRL HTML | 420 MB |
| `proxy/` | the first DEF 14A filed from 7 days before to 400 days after the 10-K | 255 MB |
| `companyfacts/` | XBRL companyfacts JSON, once per company | 122 MB |
| `submissions/` | submissions JSON, once per company | 26 MB |

**Wells Fargo:** its main 10-K document holds only 2–42 tagged numbers; the statements are in a sibling iXBRL file. Whenever a main document has fewer than 500 tagged numbers, its sibling iXBRL documents are downloaded too, which brings WFC to 7,153–8,081.

**Output:** `corpus_manifest.jsonl` has one row per `filing_id` with:
- the ARS, 10-K and proxy accession numbers and URLs;
- local paths and SHA-256;
- iXBRL fact counts;
- an `errors` field.

**Measured run:** 100/100 filings got a 10-K and a proxy with no errors, in 6.5 minutes on a home connection. No 10-K or proxy was matched twice.
- Tesla's proxy and ARS came 230–263 days after the 10-K. That is genuine: its 2025 and 2026 annual meetings were late.
- One proxy (ADBE FY2022) has no iXBRL tags; it predates pay-versus-performance tagging.

```bash
python -m scripts.downloads.download_edgar_corpus --output-dir $DRIVE/edgar --user-agent "name email@domain"
python -m scripts.downloads.download_edgar_corpus --only 10k,companyfacts --tickers NVDA,CVX   # subsets
```

Requests are spaced 0.15 s apart, under SEC's limit of 10 per second, and retried only on HTTP 429 or 503. Rerunning skips files that already exist.

## 1. What it writes

| Collection · vector | Content | Type / index | Storage |
|---|---|---|---|
| `docs_text` · `dense` | text window (1,200 chars, 200 overlap) or table part, prefixed with company, year, section, page | cosine, HNSW m=16, ef_construct=100 | RAM |
| `docs_text` · `bm25` | **number-aware** sparse terms: `$391,035` → `$391,035`, `391,035`, `391035`; `2.1%` → `2.1%`, `2.1`; `fy2024` | sparse, IDF modifier | default |
| `visual_pages` · `colpali` | full ColPali v1.3 page multivector (1030 × 128) | MaxSim, float16, **m=0 (no graph)** | **disk (mmap)** |
| `visual_pages` · `mean_rows` / `mean_cols` | 32 row means / 32 column means + the 6 special tokens (38 × 128) | MaxSim, float16, HNSW | RAM |
| `visual_pages` · `summary` | text vector of the page's readable text | cosine, HNSW | RAM |

### Payload on every point

| Group | Fields |
|---|---|
| Identity | `doc_id`, `filing_id` (`TICKER_FYyyyy`, shared later with the 10-K and its facts), `doc_kind` = `ARS` |
| Company and period | `ticker`, `company`, `cik`, `fiscal_year` (= `year`) |
| Provenance | `accession_number`, `source_url`, `content_sha256` |
| Location and type | `page`, `section` (`front`, `item_7`, `item_8`…), `content_type`, `page_type` |
| Content | `scale` (e.g. "in millions, except per share amounts"), `text_source` (`native_pdf` or `ocr`), `text`, `table_markdown` |

**Payload indexes:**
- `ticker` (keyword, `is_tenant=true`);
- `fiscal_year`, `year`, `page` (integer);
- `doc_id`, `filing_id`, `section`, `content_type`, `page_type`, `doc_kind`, `domain` (keyword).

**Payload storage:** on disk (`on_disk_payload=true`).

Numbers are stored only as payload strings. Pooling and float16 change which pages are found, never a digit.

## 2. How a filing is processed

1. **Verify.** Find the PDF in `--pdf-dir`, or download it from EDGAR with `--download-missing`. Check its SHA-256 against the manifest. A mismatch fails that filing and the run continues.
2. **Analyse each page (CPU, parallel workers, cached in `<root>/pages/`).** For each page:
   - **Text:** native text, plus tables as markdown with their header rows.
   - **Scale:** the "in millions…" phrase.
   - **Section:** taken from the `Item N` headings. A page listing four or more items is treated as a table of contents.
   - **Page type:** `table`, `chart` (more than 60 vector shapes), `image` (more than 8% of the page), `outlined_text`, `blank` or `text`.
   - **Garbled check:** fewer than 5% common English stopwords among at least 40 words, or replacement characters.
3. **OCR (RapidOCR, `--ocr garbled`).** Pages that are garbled or have outlined text are OCR'd, front section first.
   - The default limit is `--ocr-max-pages-per-filing 20`; `0` means all such pages.
   - Their broken native text is **never** indexed. A page that is not OCR'd is still covered by ColPali.
4. **`text` stage → `docs_text`.**
   - The filing's old points are deleted first, so a retry can't leave stale chunks.
   - Prose windows and table parts are embedded in batches of 128.
5. **`visual` stage → `visual_pages`.** This covers table, chart, image, outlined-text and garbled pages (`--visual-pages all` takes every non-blank page).
   - Each page is rendered at 1.5×, encoded by ColPali in batches of `--colpali-batch`, and pooled into rows and columns.
   - Full and pooled vectors and the summary go into one point.
   - Page images are not stored; they are rendered on demand.
6. **State.** Each `(filing, stage)` is marked `complete` in `<root>/ingest.sqlite3`, together with a signature of its settings: embedding model, ColPali model, chunking, OCR settings, page limit.
   - A rerun skips finished work.
   - Changing a setting redoes only the affected stage.
   - A different embedding or ColPali model against an existing index stops with an error. Use a new `--root` or `--reset`.

**After each batch of `--batch-size` filings (server mode):**
- HNSW building is paused during upload (`indexing_threshold=0`) and then restored.
- The run waits for the collections to turn green.
- Each collection is snapshotted and streamed to `--snapshot-dir` (Drive), and the SHA-256 is checked against Qdrant's checksum.
- The local snapshot is deleted. Only the newest `--keep-snapshots` are kept on Drive.
- `ingest.sqlite3` and the page cache are copied next to the snapshots.

## 3. Commands

```bash
# Colab, after starting the Qdrant server (notebook cell 4)
python -m scripts.rag.financial_ingest ingest --qdrant-url http://127.0.0.1:6333 \
    --pdf-dir $DRIVE/edgar/pdfs --user-agent "name email@domain" \
    --snapshot-dir $DRIVE/snapshots --stages text,visual
python -m scripts.rag.financial_ingest status
python -m scripts.rag.financial_ingest query "total revenue 130,497" --ticker NVDA          # text path
python -m scripts.rag.financial_ingest query "revenue by market chart" --ticker NVDA --visual
python -m scripts.rag.financial_ingest restore --snapshot-dir $DRIVE/snapshots              # next session

# Local smoke run (embedded Qdrant, hashing vectors, no GPU)
python -m scripts.rag.financial_ingest ingest --root tmp/ingest_smoke --embedding-provider hashing \
    --stages text --limit-filings 3 --ocr none --reset
```

**Main options:**

| Option | Values |
|---|---|
| `--embedding-provider` | `local` (MiniLM, 384-d, GPU if present; default), `hosted` (`EMBEDDING_API_BASE/KEY/MODEL`), `hashing` (offline tests) |
| `--colpali-dtype` | `auto`: bf16 on L4/A100, fp16 on T4, fp32 on CPU |
| `--tickers`, `--limit-filings`, `--max-pages-per-pdf` | subsets for trials |
| `--force` | redo finished filings |

**Retrieval** (`search_text`, `search_visual`, reusable by the API):
- **Text:** dense and sparse prefetch of 50 each, fused with Qdrant's native RRF.
- **Visual:** `mean_rows`, `mean_cols` and `summary` prefetch of 200 each, then exact MaxSim over the full `colpali` vectors. Vectors are never returned to the client.

## 4. Colab plan

| Session | Runtime | Stages | Time for 100 filings |
|---|---|---|---|
| 1 | CPU (High-RAM) | `text` | ~1 h at ~0.5 s/page/core with 2 workers (**measured** locally under load), plus OCR at ~4–18 s per garbled page on CPU, up to 20 per filing |
| 2 | GPU (L4, or T4) | `visual` | ~5,900 pages at ~0.5–1.5 pages/s on a T4 → ~1–3.5 h (**estimate**; confirm with the pooling evaluation) |

**Visual share (measured):** 24–27% of pages for clean filings (NVDA FY2025/FY2026) and 95% for a fully garbled one (NVDA FY2024). The corpus average is about 40%.

**Session 2** restores session 1's snapshot and page cache, so the GPU only encodes.

**After a disconnect:** rerun notebook cells 1–5 and the ingest cell.

**Storage limits** (from the design, section 6):
- Qdrant ~1.9 GB at 100 filings and ~19 GB at 1,000.
- Rotating snapshots needs the same space again on Drive.

## 5. Verified so far

| Check | Result |
|---|---|
| Offline tests | `tests/test_financial_ingest.py`: 7 passed. They cover number terms, windows, table split, pooling (shape, special tokens, row/column orientation), page classification, end-to-end text + visual ingest with a fake ColPali, number search (`391035` finds `391,035`), visual prefetch → rerank with a ticker/year filter, resume, model-mismatch guard, and a SHA-256 failure being recorded |
| Real filings, text stage (embedded Qdrant, hashing vectors) | NVDA FY2024–FY2026: 556 pages → 1,553 points. Sections found (`item_1` … `item_16`). FY2024 detected as 155/187 garbled pages and never indexed as text |
| OCR | 2 garbled NVDA FY2024 pages OCR'd, front section first, ~4 s/page CPU |
| Server-only paths (snapshot, restore, bulk indexing toggle, gRPC) | Calls checked against qdrant-client 1.20. **Not yet run against a Qdrant server**: running the Qdrant binary on the dev PC was blocked, so the first Colab session is the test |
| ColPali on GPU | **Not yet run.** Uses the same encoding code as the pooling evaluation script |

## 6. Known limits

- **Restore:** it copies the snapshot into the server's snapshots folder before recovering, because servers may refuse `file://` paths outside it. This is the first thing to check in session 2.
- **Sections:** a "contents" page mid-report keeps the previous section, and `Item 15` holds the statements for filers such as Nvidia.
- **Garbled pages beyond the OCR limit:** they have no text points; their numbers come from the 10-K HTML (store A).
- **Embedded mode** (no `--qdrant-url`) ignores payload indexes and on-disk settings. Use it for smoke runs only.

## 7. Running the stack locally or on Colab (runtime profile)

The same code runs on the PC and on Colab. [scripts/common/runtime_profile.py](../scripts/common/runtime_profile.py) picks the profile and fills in only the settings you haven't set yourself:
- **Choosing the profile:** `RAG_PROFILE=local|colab`, or Colab is detected automatically.
- **When it runs:** FastAPI applies the profile at startup, and `GET /health` reports the result under `runtime`. The Streamlit sidebar shows it next to "Backend connected".

| | **local** (your PC) | **colab** |
|---|---|---|
| Index folder (`RAG_INGEST_DIR`) | `indexes/financial_qdrant` | `/content/rag` (restored from Drive) |
| Qdrant | Embedded in the index folder (no server). `QDRANT_URL` points to a server if you run one | Server at `http://127.0.0.1:6333`, started by notebook cell 4 |
| Financial retrieval (`RAG_BACKEND`) | `financial_qdrant` once `ingest.sqlite3` exists there, otherwise the existing Chroma index | same rule |
| Text-model device | `cpu` | `cuda` on a GPU runtime |
| Streamlit → API | `http://127.0.0.1:8000` | `http://127.0.0.1:8000`, same VM; you open Streamlit through the Colab port proxy |

**How financial retrieval works with `financial_qdrant`:** the agents call `retrieve_for_agent`.
- **Hybrid search:** dense and number-aware sparse search, fused with Qdrant RRF, filtered by ticker and fiscal year.
- **Rerank:** by RRF plus term overlap. Set `RAG_RERANKER_MODEL` (for example `indexes/models/ms-marco-MiniLM-L-6-v2`) to rerank text hits with the cross-encoder instead. If the model fails, the overlap order stays and hits carry the warning `reranker_unavailable`.
- **Relevance floor:** with the cross-encoder on, the agent gives `insufficient_evidence` if the best text score is below `RAG_MIN_RERANK_SCORE` (default `-4`). `/retrieve` still returns the ranked hits.
- **Visual pages:** ColPali visual pages are added when `RAG_FIN_VISUAL=1`.
- **Citable IDs:** chunk IDs follow the agents' citation convention (`annual_report_<doc>_p<page>_<part>`), so citation checks and `apply_grounding_policy` work unchanged.
- **Other domains:** the non-financial domains keep their Chroma pipeline.

**Local commands:**

```powershell
# 1. Build or refresh the local index (embedded Qdrant; stop the API first, since embedded Qdrant allows one process)
.venv\Scripts\python.exe -m scripts.rag.financial_ingest ingest --root indexes\financial_qdrant --stages text
# 2. Serve
uvicorn scripts.api.api_app:app --host 127.0.0.1 --port 8000
streamlit run scripts\ui\api_streamlit_app.py
```

**On Colab:** run notebook cells 1–5, then ingest (or restore), then the **Serve** cells. They install the dependencies from `pyproject.toml`, start uvicorn (one worker) and Streamlit on localhost, and print the proxy URL. The project isn't an installable package, so `pip install -e .` fails; the notebook installs the dependency list instead.

**Verified locally** (mock LLM, hashing vectors, NVIDIA smoke index):
- `/health` reported `financial_backend=financial_qdrant`;
- the conversation asked for the year, then answered from Qdrant evidence filtered to NVIDIA FY2025;
- the answer's citations passed grounding (`grounded`).

**Not yet run on Colab:** the serving cells and the Colab port proxy.
