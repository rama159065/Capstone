# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Track B capstone: a citation-grounded **Agentic RAG** "Operations Copilot" over public PDF corpora in six domains (`financial_report_qa`, `procurement_vendor`, `travel_expense`, `competitive_research`, `incident_postmortem`, `hr_policy`). It uses LangGraph-guarded agents, hybrid retrieval, human-in-the-loop (HITL) routing, a FastAPI backend and Streamlit UIs. It also has a local multimodal path (Qdrant, OCR, ColPali, Qwen3-VL). Python 3.12 on Windows (PowerShell). Dependencies are managed with `uv` (`uv sync`); the venv lives at `.venv/`. The README has a full command reference for each domain.

## Commands

Run all commands from the repo root. Imports are always absolute (`from scripts.rag.multimodal_rag import ...`). `scripts/` itself is a namespace package with no `__init__.py`. Code is grouped by layer:
- `api/`
- `ui/`
- `agents/`
- `rag/`
- `observability/`
- `downloads/`
- `evaluation/`
- `reports/`
- `common/`
- `domains/<domain>/`

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
python scripts\domains\<domain>\index.py --embedding-provider hashing --max-pages-per-pdf 5 --reset
python scripts\domains\<domain>\agent.py --query "..." [--html-out output\x.html] [--enable-langsmith]

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

### Two agent implementations
- **Financial** (`scripts/agents/financial_prompting_agent.py` + `scripts/rag/index_financial_report_pdfs.py`) is a richer, standalone implementation. It has several prompting `--technique`s (baseline, structured few-shot, decomposition, self-consistency, `agentic_graph`, `all`), and its graph includes a critic→answer **repair loop**. It loads the indexer module dynamically (`load_indexer_module`).
- **Other five domains** share `scripts/agents/rag_agent.py`. Each domain folder holds only a `DomainConfig` (`domain_config.py`: intent terms, HITL review terms, source types, default query) and three ~10-line wrappers: `index.py` → `index_cli`, `agent.py` → `agent_cli`, and `html_app.py` → `streamlit_app`. Domain behavior belongs in `DomainConfig`, not in these wrappers.
- Both implementations have their own copies of the guardrail regexes and LLM/embedding clients. A guardrail change usually needs to be made in **both** files.

Shared graph (`run_domain_agent`): `security_guard → plain_text_classifier → planner → hybrid_retriever → evidence_guard → answer_agent → output_filter → citation_critic → governance`. A guard that fails short-circuits to `citation_critic` (not END), so the governance/HITL output is always produced. Results are auditable JSON with `security`, `evidence_guard`, `output_filter` and `human_review` fields. Output-filter decisions are `allow`/`review`/`block`.

### Adding or renaming a domain
Domain names are hardcoded in several places: `api/api_app.py` (`NON_FINANCIAL_DOMAINS`), `ui/api_streamlit_app.py`, `rag/multimodal_rag.py` (`DOMAINS`), `downloads/download_domain_pdfs.py` and `downloads/download_rag_corpora.py`. The tests (`test_domain_entrypoints.py`, `test_api_app.py`, and others) check this wiring.

### Retrieval backends
- **Default:** Chroma (cosine HNSW) at `indexes/chroma/<domain>...`, plus BM25 over `data/processed/<domain>/pdf_chunks.jsonl`, fused with RRF (k=60) and then a lexical-overlap rerank. Corpora are listed in `data/raw/<domain>/pdf_manifest.jsonl`.
- **`RAG_BACKEND=qdrant`**: `retrieve_hybrid` and the financial `retrieve_evidence` both delegate to `multimodal_rag.retrieve_local`, which wraps a process-wide `LocalRAG` singleton (`get_rag`). `LocalRAG` uses embedded Qdrant at `$RAG_LOCAL_DIR/qdrant` (or a server if `QDRANT_URL` is set), with collections `docs_text` (dense + IDF-sparse "bm25") and `visual_pages` (ColPali multivectors, MaxSim). A SQLite sidecar stores the embedding-model signature. **Changing `RAG_EMBEDDING_MODEL` against an existing index raises an error; use a new `RAG_LOCAL_DIR` and reingest.** Embedded Qdrant allows only a single process, so the multimodal `serve` runs uvicorn with `workers=1`.
- Selective OCR (`selective_ocr.py`, RapidOCR) adds text from scanned pages and embedded images into `docs_text`.

### Providers (chosen by flag or env var)
- Embeddings: `hashing` (deterministic, offline; for tests and smoke runs) or `hosted` (`EMBEDDING_API_BASE/KEY/MODEL`, OpenAI-compatible).
- Chat LLM: `mock` or `hosted` (`LLM_API_BASE/KEY/MODEL`).
- Multimodal answering (`safe_answer.answer`): `extractive` | `ollama` (`RAG_VISION_MODEL`, `OLLAMA_HOST`) | `transformers` (`RAG_TRANSFORMERS_MODEL`) | `llamacpp` (`RAG_VISION_API_BASE`). `run_local_multimodal.configure()` / `colpali_index.configure_colpali()` set the local defaults with `os.environ.setdefault`, including `HF_HUB_OFFLINE=1`.

### Reliability layer (used by all model and retrieval calls)
- `common/resilience.py` `Calls(timeout)` is a per-request deadline. Each `.run(name, fn, key=..., retry=...)` call runs on a shared 4-slot thread pool. When no slot is free it fails fast (`capacity_exceeded`); it does not queue. Calls with an idempotency key are deduplicated and cached. A retry **requires** an idempotency key and only happens on explicit HTTP 429/503; ambiguous timeouts are never retried. All failures surface as `CallFailure(<reason>)`. Timeouts come from `RAG_HTTP_TIMEOUT_SECONDS` (default 45) and `RAG_REQUEST_TIMEOUT_SECONDS` (default 60).
- `observability/trace_log.py`: ContextVar-based JSONL traces (`TraceLog`, `emit`) that are propagated into worker threads through `copy_context`. `emit` does nothing when no trace is active.
- `agents/safe_answer.py` is fail-closed. It splits multi-part queries (at most 8), drops unusable hits, and never treats generated prose as verified fact.
- `api_app.apply_grounding_policy` withholds the generated answer when citations don't validate against the retrieved chunk IDs. This is consistency checking, not entailment.

### API (`scripts/api/api_app.py`)
Routes: `/health`, `/domains`, `/chat/financial_report_qa`, `/chat/{domain}`, `/retrieve/...`, and `/multimodal/{domain}/{retrieve,chat,visual/retrieve}`. It has an in-memory per-IP rate limiter (60 req/min, per process), `X-Request-ID` correlation, and access logs that never record bodies or prompts. LangSmith tracing is optional (`LANGSMITH_API_KEY` plus `enable_langsmith` or `LANGSMITH_TRACING=true`), and runs degrade gracefully without the key. See `docs/fastapi_api.md`.

### Financial report generation (`scripts/reports/generate_financial_report.py`)
This is a deterministic pipeline driven by a JSON profile in `configs/financial_reports/`. The source PDF must match the profile's `source_sha256`. Statement rows are parsed by exact label (an ambiguous or missing row is a hard failure; values are never guessed). The arithmetic uses `Decimal` and reconciliations must pass. Charts and tables share one dataset, and the run writes PDF/MD/charts/manifest/trace to `output/pdf/<TICKER>_<YEAR>/<trace_id>/` with atomic writes. An LLM is optional and never supplies the numbers. See `docs/financial_report_generation.md`.

## Data and generated files
`data/processed/`, `indexes/` (Chroma, Qdrant, downloaded models, llama.cpp runtime), `output/`, `eval/` and `tmp/` are gitignored. `data/raw/` is partly committed (corpus JSON); `data/samples/multimodal*` holds the small sample PDFs used by the multimodal evaluation. Design and validation notes for each subsystem are in `docs/`.
