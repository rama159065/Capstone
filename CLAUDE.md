# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Track B capstone: a citation-grounded **Agentic RAG** "Financial Report Copilot" over SEC annual reports (`financial_report_qa` only; the five other Track B domains were removed on 10 Nov 2026 and live in git history). It uses LangGraph-guarded agents, hybrid retrieval, human-in-the-loop (HITL) routing, a FastAPI backend and a Streamlit UI. It also has a local multimodal path (Qdrant, OCR, ColPali, Qwen3-VL). Python 3.12 on Windows (PowerShell). Dependencies are managed with `uv` (`uv sync`); the venv lives at `.venv/`. The README has the full command reference.

## Commands

Run all commands from the repo root. Imports are always absolute (`from scripts.rag.multimodal_rag import ...`). `scripts/` itself is a namespace package with no `__init__.py`. Code is grouped by layer:
- `api/`
- `ui/`
- `agents/`
- `guardrails/`
- `prompts/`
- `rag/`
- `observability/`
- `downloads/`
- `evaluation/`
- `reports/`
- `common/`

Each folder has a docstring-only `__init__.py`.

Use `scripts.common.paths.ROOT` for the repo root; don't count `Path(__file__).parents`. Older entry points add the repo root to `sys.path` themselves, so `python scripts\<folder>\x.py` works. Newer modules must be run as `python -m scripts.<folder>.<module>`.

```powershell
# Tests (no network, models or downloaded data needed; ~45s)
.venv\Scripts\python.exe -m pytest tests
.venv\Scripts\python.exe -m pytest tests/test_api_app.py::test_name     # single test
# If the multimodal/OCR/ColPali tests hit FileNotFoundError/WinError 3, the Windows
# 260-char path limit is the cause (content-addressed object paths are deep). Use a short basetemp:
.venv\Scripts\python.exe -m pytest tests --basetemp tmp/pt

# Offline smoke run: hashing embeddings + mock LLM
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hashing --limit-pdfs 5 --max-pages-per-pdf 10 --reset
python scripts\agents\financial_prompting_agent.py --llm-provider mock --technique agentic_graph --query "..." --ticker AAPL

# API + UI
uvicorn scripts.api.api_app:app --host 127.0.0.1 --port 8000
streamlit run scripts\ui\api_streamlit_app.py

# Local multimodal profile (Qdrant + local models; prepare downloads models and llama.cpp into indexes/)
python -m scripts.rag.run_local_multimodal prepare|ingest|evaluate|serve|query [--colpali]
python -m scripts.rag.ocr_index [--domain D] [--force]
python -m scripts.rag.colpali_index backfill|query|status

# Financial report generation (deterministic, numbers come from the PDF)
python -m scripts.reports.generate_financial_report --profile configs/financial_reports/apple_2024.json
```

No linter or formatter is configured.

## Architecture

### Agents
- **Financial** (`scripts/agents/financial_prompting_agent.py` + `scripts/rag/index_financial_report_pdfs.py`) is a richer, standalone implementation. It has several prompting `--technique`s (baseline, structured few-shot, decomposition, self-consistency, `agentic_graph`, `all`), and its graph includes a critic→answer **repair loop**. It loads the indexer module dynamically (`load_indexer_module`).
  - Three LangGraph flows share one builder (`financial_graph`) with the same guards, critic and governance: `agentic_graph` (Q&A), `trend_graph` (multi-year trend: `agents/financial_trend.py`, one search per fiscal year, the LLM copies each figure and code keeps it only if that year's chunk states it, changes and CAGR computed in code) and `router` (supervisor: `agents/financial_router.py` validates the router LLM's route against the parser; lookup/comparison → Q&A, trend → trend agent, report → `report_handoff`, out_of_scope → refused). The route step also plans, so routed runs cost no extra step. Streamlit defaults to `router`; the API default stays `agentic_graph`.
  - `agentic_graph` routes per request: the planner LLM proposes tools and `agents/financial_tools.normalize_plan` validates them (`coverage_lookup`, `text_search` always, `visual_search` only with `RAG_FIN_VISUAL=1|auto`, `calculator` for growth/ratios) and splits comparisons into one filtered search per company × year. Missing targets get one broadened retrieval retry, `CALC[...]` in the answer goes to the calculator (operands must appear in the evidence), invalid citations get one repair. Loops run only if the rest of the run fits `AGENT_MAX_STEPS`. Results carry `agent_run` (path, tool calls, calculations). Full flow: `docs/agent_flows.md`.
- The former shared domain agent (`rag_agent.py`), `prompts/domain.py` and the `DOMAIN_*` guard patterns were removed on 11 Nov 2026 (git history). The `/multimodal/financial_report_qa/chat` route guards input with the financial agent's `deterministic_input_guard`.
- **Guardrails** live in `scripts/guardrails/`: `patterns.py` holds the regex sets and `checks.py` the input, evidence and output checks.
  - The financial agent keeps its public guard function names (`deterministic_input_guard`, ...) as thin wrappers over these checks, bound to the `FINANCIAL_*` pattern sets.
- **Prompt templates** live in `scripts/prompts/financial.py`.
  - The mock LLM routes on phrases in these prompts, so prompt text changes can change test behaviour.
  - The Week 1 baseline prompts stay in `evaluation/prompt_baselines.py`.
- **LLM and embedding clients** are still separate copies in the agent, the indexers and `prompt_baselines`. They differ in timeouts, error wrapping and User-Agent, so merging them is a behaviour change. The hosted chat clients share `common/hosted_llm.py` (`LLM_CA_CERT`, one request at a time, 429/503 back-off 2/4/8 s, Qwen3 `enable_thinking=false` and `<think>` stripping).

Financial graph: `security → plain_text → plan → retrieve ⇄ evidence_guard → answer → calculator? → critic ⟲ → governance`. A guard that fails short-circuits to the critic (not END), so the governance/HITL output is always produced. Results are auditable JSON with `security`, `evidence_guard`, `output_filter`, `support_check`, `human_review` and `agent_run`. Output-filter decisions are `allow`/`review`/`block`.

### Single domain
The API (`ALL_DOMAINS = ("financial_report_qa",)`), the Streamlit UI (`DOMAIN`) and `downloads/download_rag_corpora.py` serve `financial_report_qa` only; other domain names get 404 (`/multimodal/{domain}`) or no route (`/chat/{domain}` and `/retrieve/{domain}` were removed). `rag/multimodal_rag.DOMAINS` is `("financial_report_qa",)` too; other names raise `Unknown domain`.

### Retrieval backends
- **Default:** Chroma (cosine HNSW) at `indexes/chroma/financial_report_qa_pdfs`, plus BM25 over `data/processed/financial_report_qa/pdf_chunks.jsonl`, fused with RRF (k=60) and then a lexical-overlap rerank. Corpora are listed in `data/raw/<domain>/pdf_manifest.jsonl`.
- **`RAG_BACKEND=qdrant`**: `retrieve_hybrid` and the financial `retrieve_evidence` both delegate to `multimodal_rag.retrieve_local`, which wraps a process-wide `LocalRAG` singleton (`get_rag`). `LocalRAG` uses embedded Qdrant at `$RAG_LOCAL_DIR/qdrant` (or a server if `QDRANT_URL` is set), with collections `docs_text` (dense + IDF-sparse "bm25") and `visual_pages` (ColPali multivectors, MaxSim). A SQLite sidecar stores the embedding-model signature. **Changing `RAG_EMBEDDING_MODEL` against an existing index raises an error; use a new `RAG_LOCAL_DIR` and reingest.** Embedded Qdrant allows only a single process, so the multimodal `serve` runs uvicorn with `workers=1`.
- Selective OCR (`selective_ocr.py`, RapidOCR) adds text from scanned pages and embedded images into `docs_text`.
- **Sector corpus:** `downloads/download_annual_report_corpus.py` fetches the 1,004-PDF plan (`data/raw/financial_report_qa/annual_reports/corpus_plan.jsonl`) into `annual_reports/<sector_slug>/`, with one manifest per sector. `common/sector_corpus.py` reads a manifest file or a corpus folder (`local_path` is relative to the manifest's folder). Both financial indexers take `--manifest <folder> --sectors ...` and store `sector`/`sector_slug` on every chunk. Sectors are a filter field in one collection, not separate indexes.
- **Corpus presets:** `full` (1,004 filings) or `min500` (509: the 41 largest filings per sector plus every filing of the eval-set companies; Track B's 500-document minimum). Set with `--preset` (downloader), `--corpus-preset` (both ingesters, `colab_pipeline`) or `CORPUS` (notebook). `download_annual_report_corpus.preset_rows` defines them. Ingestion skips downloaded filings outside the preset, and `cp.coverage` counts against it.
- **Two visual-page pipelines over the same manifests.** `rag/financial_ingest_qdrant_db.py` (pipeline 2) writes ColPali pooled and full multivectors to Qdrant `annual_reports_visual` and text to `annual_reports_text` (its own names, so `LocalRAG`'s `docs_text`/`visual_pages` can share a server; `restore` maps old-named snapshots). Pipeline 2 is **Qdrant only**. `rag/financial_ingest_chroma_db.py` (pipeline 1) is **Chroma only** (`indexes/financial_vlm/chroma`, collection `annual_reports_vlm`): a page reader (`--vlm ocr` RapidOCR + native tables, `transformers` local Qwen3-VL (default 4B), `openai` any OpenAI-compatible endpoint incl. paid APIs with 429/503 back-off, or `mock`) turns visual pages into Markdown, and retrieval fuses Chroma's dense hits with a BM25 sidecar (`chroma/bm25.sqlite3`, FTS5 over `financial_ingest_qdrant_db.terms`) by RRF. Transcriptions are cached per page; figures the text layer doesn't print go into `unverified_numbers`. `rag/colab_pipeline.py` (`Settings` → `prepare/restore/download/ingest/status/query`, notebook `notebooks/colab_rag_pipelines.ipynb`) runs either pipeline sector by sector. It copies each sector's PDFs from Drive to local disk first, and snapshots once per sector (`snapshot_every`). The live index is on local disk; the PDFs, VLM cache and index copies are on Drive.
- **One chunk schema for both indexes:** `rag/chunk_schema.py` (`FIELDS`: filing, page, chunk and provenance fields; `chunk_payload`, `normalize`, `evidence_hit`). Qdrant `annual_reports_text` (and the visual payloads) and Chroma `annual_reports_vlm` store the same fields and both backends return the same evidence keys (`EVIDENCE_KEYS`); Chroma only encodes lists as JSON and drops `None`. `tests/test_chunk_schema.py` pins it, so a new field goes into `chunk_schema` for both.
- **Routed reading (`--vlm routed`, the default):** pipeline 1 picks a reader per page with `page_audit.route` (none skipped, text layer, native tables, OCR, Qwen3-VL). `paidVL` pages run the fallback: OCR the page, Qwen3-VL transcribes, `cross_check` confirms each figure against the OCR text; above `--paid-threshold` (0.1) unconfirmed, the paid model (`RAG_PAID_VISION_API_BASE/_MODEL/_API_KEY`) reads it and the reading with fewer unconfirmed figures is kept. Chart and image pages also get a Qwen3-VL search description (`DESCRIBE_PROMPT`, a `page_description` chunk, no values; `--no-describe-visuals`). Readings are cached per reader; chunks carry `reader`, `page_route`, `escalation`, `unverified_numbers`.
- **Page audit:** `rag/page_audit.py` (`cp.page_audit`, `colab_pipeline audit`) routes every downloaded page to `pdfParser` / `pdfParser+colpali` / `ocr` / `qwen3VL` / `paidVL` from `analyse_pdf` signals and writes `page_audit/<sector_slug>/<pdf stem>.csv` (`pageNumber,parserUsage`) plus `_summary.csv/json`. About 0.22 s per page per CPU core.
- **Serving the sector corpus:** `docs/annual_report_rag_structure.md` is the end-to-end layout. `cp.coverage` reports planned → downloaded → ingested per sector; `cp.serve` starts FastAPI + Streamlit with `cp.serving_env`. The agent backends are `RAG_BACKEND=financial_qdrant` (`RAG_INGEST_DIR`) and `financial_vlm` (`RAG_VLM_DIR`, `financial_ingest_chroma_db.retrieve_for_agent`). The runtime profile picks whichever index exists, in the same `RAG_HOME` layout that `colab_pipeline` writes. Financial requests and `AgentConfig` take an optional `sector`.
- **Colab code zip:** `python -m scripts.common.colab_bundle` writes `output/colab/capstone_code.zip` (git-tracked plus untracked-not-ignored code and small data). The setup cell of each notebook that imports `scripts` (02–05, `colab_financial_ingest`, `colab_rag_pipelines`) unpacks it from `MyDrive/capstone_rag/` and falls back to a git clone.

### Providers (chosen by flag or env var)
- Embeddings: `hashing` (deterministic, offline; for tests and smoke runs) or `hosted` (`EMBEDDING_API_BASE/KEY/MODEL`, OpenAI-compatible).
- Chat LLM: `mock` or `hosted` (`LLM_API_BASE/KEY/MODEL`).
- Multimodal answering (`safe_answer.answer`): `extractive` | `ollama` (`RAG_VISION_MODEL`, `OLLAMA_HOST`) | `transformers` (`RAG_TRANSFORMERS_MODEL`) | `llamacpp` (`RAG_VISION_API_BASE`). `run_local_multimodal.configure()` / `colpali_index.configure_colpali()` set the local defaults with `os.environ.setdefault`, including `HF_HUB_OFFLINE=1`.

### Reliability layer (used by all model and retrieval calls)
- `common/resilience.py` `Calls(timeout)` is a per-request deadline. Each `.run(name, fn, key=..., retry=...)` call runs on a shared 4-slot thread pool. When no slot is free it fails fast (`capacity_exceeded`); it does not queue. Calls with an idempotency key are deduplicated and cached. A retry **requires** an idempotency key and only happens on explicit HTTP 429/503; ambiguous timeouts are never retried. All failures surface as `CallFailure(<reason>)`. Timeouts come from `RAG_HTTP_TIMEOUT_SECONDS` (default 45) and `RAG_REQUEST_TIMEOUT_SECONDS` (default 60).
- `observability/trace_log.py`: ContextVar-based JSONL traces (`TraceLog`, `emit`) that are propagated into worker threads through `copy_context`. `emit` does nothing when no trace is active.
- `agents/safe_answer.py` is fail-closed. It splits multi-part queries (at most 8), drops unusable hits, and never treats generated prose as verified fact.
- `api_app.apply_grounding_policy` withholds the generated answer when citations don't validate against the retrieved chunk IDs (consistency, not entailment). Entailment is the financial critic's **support check** (`critic_step`): every figure must appear in a cited chunk (`financial_tools.numeric_grounding`) and an LLM judge must support every claim. With a hosted LLM, `run_technique` keeps the generated answer only when that verdict is `supported` (`grounding.semantic_answer_verified`); otherwise it returns cited excerpts (`evidence_only`). `RAG_SUPPORT_CHECK=0` turns it off.
- Financial rerank: `financial_ingest_qdrant_db.retrieve_for_agent` uses the cross-encoder when `RAG_RERANKER_MODEL` is set; `guardrails.checks.relevance_floor` then stops answers when the best text score is below `RAG_MIN_RERANK_SCORE`. Agents call `answer_evidence`; `/retrieve` calls `retrieve_evidence` and has no floor.
- Financial coverage: `financial_clarifier.question_scope` decides whether the filings can answer (company, single-figure year, measures banks don't report). `run_technique` uses it to return `not_available` with no figure, and `guardrails.checks.company_scope_check` withholds answers citing a company the question didn't ask about. Company names, aliases and sectors for all 289 tickers live in `common/companies.py` (`COMPANIES`, plus `AMBIGUOUS_TICKERS` for word-like tickers such as AR, ICE and NOW). The clarifier, the manifest reader and both indexers use it, and manifests without a `sector` get it from the ticker. A company added to the corpus needs an entry there; tests check it against the manifests, the sector plan and the frozen evaluation set.

### API (`scripts/api/api_app.py`)
Routes: `/health`, `/domains`, `/companies`, `/chat/financial_report_qa`, `/conversation/financial_report_qa`, `/retrieve/financial_report_qa`, and `/multimodal/financial_report_qa/{retrieve,chat,visual/retrieve}`. It has an in-memory per-IP rate limiter (`RATE_LIMIT_PER_MINUTE`, default 60, per process), `X-Request-ID` correlation, and access logs that never record bodies or prompts. LangSmith tracing is optional (`LANGSMITH_API_KEY` plus `enable_langsmith` or `LANGSMITH_TRACING=true`), and runs degrade gracefully without the key. See `docs/fastapi_api.md`.

Deployment settings are all environment variables (`.env.example`):
- `API_AUTH_TOKEN` requires `X-API-Key` or a Bearer token on every route except `/health` and `/docs`; it is empty (open) locally.
- `CORS_ALLOW_ORIGINS`.
- `DEFAULT_LLM_PROVIDER` / `DEFAULT_EMBEDDING_PROVIDER` set the request defaults (`mock`/`hashing` unless set).
- `ANSWER_CACHE_SECONDS` reuses answers to identical requests (`AnswerCache`, off by default; failures are never cached).
- `CONVERSATION_DB` stores clarifier state in SQLite (`ConversationStore(path=...)`) so it survives restarts; otherwise it is in memory.
- `AGENT_MAX_STEPS` (default 10, `common/agent_limits.py`) caps LangGraph node executions. Both graphs are invoked with `recursion_limit = steps + 1`, and hitting the cap returns `final_status="step_limit"` with no answer.

`GET /companies` lists tickers, names, sectors and covered years from the clarifier's directory. That directory is the corpus manifests (plus `RAG_CORPUS_DIR`) **and** the filings whose `text` stage is complete in the served index's `ingest.sqlite3` / `vlm_ingest.sqlite3`, so a server holding only a restored index still knows its coverage. Streamlit builds its ticker/year pickers from it.

### Deployment (`Dockerfile`, `docker-compose.yml`, `deploy/`)
- **One CPU-only image for both services:** torch comes from the CPU wheel index; `deploy/requirements_from_pyproject.py` drops torch and JupyterLab from the pyproject list. Compose runs it as `api` (uvicorn, 1 worker, `--proxy-headers`) and `ui` (Streamlit), next to `qdrant`, with an optional `https` profile (Caddy + basic auth, `deploy/Caddyfile`).
- **Profiles and storage:** `RAG_PROFILE=local|colab|server|gcp`. Colab is detected; otherwise the default is local.
  - Every path comes from `common/storage.py`: one layout under `RAG_HOME`, with `indexes/qdrant_db` = `RAG_INGEST_DIR`, `indexes/chroma_db` = `RAG_VLM_DIR`, `snapshots/qdrant_db`, `cache/vlm_pages`, `state/conversations.sqlite3`, `models/` and `secrets/llm_ca.crt`.
  - Homes: local `indexes/rag_home`; colab `/content/rag` (durable copy `MyDrive/capstone_rag`); server/gcp `/app/rag_home` (host folder `RAG_HOME_DIR`; gcp durable copy `gs://$RAG_GCS_BUCKET`).
  - Older folder names (`colpali_qdrant`, `financial_qdrant`, `vlm_chroma`, `financial_vlm`) are still found.
  - `colab_pipeline` writes the same layout, so a local ingest is what local serving finds.
  - The reranker and the CA cert are set when their files exist. Explicit env vars always win.
  - `tests/conftest.py` points `RAG_HOME` at a temp folder.
- **Restoring the Colab index:** `docker compose run --rm --user root api sh deploy/restore_index.sh` restores the Qdrant snapshots and copies the Chroma folder, from `RAG_HOME/snapshots` or `./snapshots`. On GCP, `sh deploy/sync_from_gcs.sh` first pulls both from the bucket (the Colab side is `cp.export_to_gcs`). The `qdrant_snapshots` volume is mounted at `/qdrant/snapshots` in both the api and qdrant containers, because `restore` hands Qdrant a `file://` path.
- **Docs:** `docs/deployment.md` (and `.html`) covers the options, steps and costs.

### Financial report generation (`scripts/reports/generate_financial_report.py`)
This is a deterministic pipeline driven by a JSON profile in `configs/financial_reports/`. The source PDF must match the profile's `source_sha256`. Statement rows are parsed by exact label (an ambiguous or missing row is a hard failure; values are never guessed). The arithmetic uses `Decimal` and reconciliations must pass. Charts and tables share one dataset, and the run writes PDF/MD/charts/manifest/trace to `output/pdf/<TICKER>_<YEAR>/<trace_id>/` with atomic writes. An LLM is optional and never supplies the numbers. See `docs/financial_report_generation.md`.

## Data and generated files
`data/processed/`, `indexes/` (Chroma, Qdrant, downloaded models, llama.cpp runtime), `output/`, `eval/` and `tmp/` are gitignored. `data/raw/` is partly committed (corpus JSON); `data/samples/multimodal*` holds the small sample PDFs used by the multimodal evaluation. Design and validation notes for each subsystem are in `docs/`.
