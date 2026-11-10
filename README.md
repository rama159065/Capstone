# Financial Report Copilot

Agentic RAG system for the Track B capstone (API-side, hosted <=8B inference APIs): a **financial report copilot over SEC annual reports**.

It answers questions about company filings with every figure cited to its page, compares companies, checks growth with a calculator, and refuses advice or anything the filings don't cover. It uses hybrid retrieval, LangGraph agents with guardrails and a support-checking critic, human-in-the-loop routing, a FastAPI backend, a Streamlit UI and retrieval evaluation. The agent flow is described in [docs/agent_flows.md](docs/agent_flows.md).

## Why This Is Agentic RAG

This is **Agentic RAG** because retrieval is only one step inside a stateful agent workflow. A plain RAG chain usually does `query -> retrieve -> answer`; this project adds LangGraph nodes that validate the request, plan retrieval, call hybrid search tools, inspect retrieved evidence, draft an answer, filter unsafe output, validate citations, and route risky cases to human review.

The implemented agent loop has these decision-making stages:

```text
security -> plain_text -> planner (tools) -> retrieve <-> evidence_guard (retry) -> answer -> calculator -> critic (repair) -> governance
```

That makes the system agentic in the capstone sense: it uses tools, maintains structured state, applies bounded decisions, checks its own evidence/citations, records observability metadata, and can stop or escalate instead of always answering.

## Scope

The product is **financial report question answering** only (`financial_report_qa`): SEC annual reports (Form ARS) from 281 companies in 13 sectors, served as `full` (1,004 filings) or `min500` (509).

The five other Track B domains (procurement, travel and expense, competitive research, incident postmortem, HR policy) were removed on 10 November 2026; they remain in git history. `scripts/agents/rag_agent.py` stays for now because the local multimodal routes use its input guard.

## Implementation Status

| Area | Status |
|---|---|
| Financial PDF RAG | Implemented with PDF verification, chunking, Chroma, BM25, RRF, reranking, optional table summaries, optional chart-summary chunks, and top-k query mode. |
| Agent orchestration | Implemented with LangGraph nodes for security, plain-text validation, planning, hybrid retrieval, evidence guard, answer generation, output filtering, citation criticism, and governance/HITL. |
| Guardrails and HITL | Implemented for input filtering, evidence prompt-injection filtering, output content filtering, citation validation, and human-review routing. |
| API backend and UI | Implemented with FastAPI backend endpoints plus Streamlit frontends, including an API-backed Streamlit app for the final architecture. |
| Observability | Optional LangSmith tracing is implemented for agents and financial retrieval evaluation. |
| Evaluation | Financial retrieval recall@k evaluation is implemented separately from answer generation. |

## Repository Layout

```text
docs/                         Design notes and capstone documentation
scripts/
  api/            FastAPI backend (api_app.py): financial chat, conversation, retrieval, companies
  ui/             Streamlit front ends: API-backed Financial Report Copilot (api_streamlit_app.py), standalone financial app
  agents/         Financial agent graph and tools, clarifier, safe answer; rag_agent.py (shared guard, used by multimodal)
  guardrails/     Shared input / evidence / output safety checks and their regex pattern sets
  prompts/        Prompt templates for the financial and domain agents
  rag/            Ingestion, indexing and retrieval: Chroma/BM25 indexer, Qdrant ingest (financial_ingest.py),
                  multimodal RAG, ColPali, selective OCR, retrieval metrics
  observability/  JSONL traces, LangSmith helpers, report-trace monitor
  downloads/      Corpus and model downloaders (EDGAR ARS/10-K, SEC company facts, models)
  evaluation/     Eval set builder, prompt baselines, retrieval/OCR/ColPali evaluations, validation-data generators
  reports/        Deterministic financial report generation
  common/         Repo paths, runtime profile (local/colab/server), resilience, local models, company directory
notebooks/                    Colab ingestion and serving notebook
tests/                        Unit tests
```

Generated data is intentionally ignored by git:

```text
data/raw/
data/processed/
indexes/
output/
eval/
```

## Setup

Python 3.12 is recommended.

Using `uv`:

```powershell
uv sync
```

Or with plain `pip`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

Optional OCR support for scanned pages, tables, or chart images requires Tesseract installed on the operating system. The current implementation can run without OCR for text-based PDFs.

## Quick Start

Run a small local smoke test with deterministic hashing embeddings:

```powershell
python scripts\downloads\download_financial_report_pdfs.py --target-pdfs 5
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hashing --limit-pdfs 5 --max-pages-per-pdf 10 --reset
python scripts\agents\financial_prompting_agent.py --llm-provider mock --technique agentic_graph --query "What does Apple say can affect quarterly net sales?" --ticker AAPL
```

Start the final API-side demo stack:

```powershell
uvicorn scripts.api.api_app:app --host 127.0.0.1 --port 8000
streamlit run scripts\ui\api_streamlit_app.py
```

## Data Download

Download financial annual report PDFs:

```powershell
python scripts\downloads\download_financial_report_pdfs.py --target-pdfs 100
```

Optional raw SEC XBRL company-facts records (JSON) for corpus analysis:

```powershell
python scripts\downloads\download_rag_corpora.py --domains all --target-docs 500 --overwrite --sec-user-agent "capstone-rag/0.1 your_email@example.com"
```

PDF manifests are written to:

```text
data/raw/<domain>/pdf_manifest.jsonl
```

### Sector annual-report corpus (1,004 EDGAR ARS PDFs, 13 sectors)

The corpus planned in `docs/annual_report_corpus_plan.html`. The plan ships as `data/raw/financial_report_qa/annual_reports/corpus_plan.jsonl`. The downloader writes one folder and one manifest per sector, plus a combined manifest; every row carries `sector` / `sector_slug` / `sic`:

```text
data/raw/financial_report_qa/annual_reports/
  pdf_manifest.jsonl               all sectors (local_path = banking/EDGAR_ARS_JPM_2025.pdf)
  banking/pdf_manifest.jsonl       banking only (local_path = EDGAR_ARS_JPM_2025.pdf)
  banking/EDGAR_ARS_JPM_2025.pdf
  oil_and_gas/...
```

```powershell
python -m scripts.downloads.download_annual_report_corpus --dry-run                       # per-sector counts and GB, no network
python -m scripts.downloads.download_annual_report_corpus --user-agent "capstone-research you@example.com" --sectors banking,oil_and_gas --max-per-sector 20
# Index by sector: pass the corpus folder as the manifest; every chunk gets sector metadata.
python -m scripts.rag.financial_ingest ingest --manifest data\raw\financial_report_qa\annual_reports --sectors banking
python -m scripts.rag.financial_ingest query "net interest income" --sector banking
python scripts\rag\index_financial_report_pdfs.py --manifest data\raw\financial_report_qa\annual_reports --sectors banking --filter-sector banking --query "net interest income"
```

- **Reruns:** downloads are resumable. PDFs already in `data/raw/financial_report_qa/edgar/pdfs` are hard-linked or copied instead of fetched.
- **One collection for all sectors:** sectors are a payload-indexed field, not separate collections.
- **Older indexes:** `financial_ingest tag-sectors --manifest <corpus>` adds the field to points ingested before it existed.
- **On Colab:** see the "sector corpus" cells in `notebooks/colab_financial_ingest.ipynb`. Use `--output-dir` to put the corpus on Drive.

### Two ingestion pipelines for visual pages

Both pipelines read the same manifests (a file or the sector corpus folder, with `--sectors`), and both put `sector`, `ticker`, `fiscal_year` and a citable `chunk_id` on every point.

| | Pipeline 1: ColPali, `scripts.rag.financial_ingest` | Pipeline 2: VLM, `scripts.rag.vlm_ingest` |
|---|---|---|
| Visual pages become | ColPali multivectors: mean-pooled rows/columns in RAM (first stage), full 1030×128 on disk (MaxSim rerank) | Markdown written by a VLM (tables as Markdown tables, charts as labelled values), then chunked and embedded |
| Pages | `--visual-pages rule` (tables, charts, pictures, unreadable pages), `tables_charts` or `all` | `--vlm-pages visual`, `tables_charts` or `all`; other pages keep their text layer |
| Store | Qdrant `annual_reports_visual` (+ `annual_reports_text` for text) | Chroma (default) or Qdrant `vlm_text` (`--store qdrant`) |
| Model | `vidore/colpali-v1.3-hf` | `--vlm transformers` (Qwen2.5-VL-3B, 4-bit on GPU), `--vlm openai` (llama.cpp / vLLM / Ollama endpoint) or `mock` |
| Numbers | stay as exact strings in the payload, never in vectors | figures the text layer doesn't print are flagged as `unverified_numbers` |

```powershell
python -m scripts.rag.financial_ingest ingest --manifest data\raw\financial_report_qa\annual_reports --sectors banking --stages visual --visual-pages tables_charts
python -m scripts.rag.vlm_ingest ingest --manifest data\raw\financial_report_qa\annual_reports --sectors banking --vlm transformers
python -m scripts.rag.vlm_ingest query "net interest income" --sector banking
```

**On Colab:** first run `python -m scripts.common.colab_bundle` and upload `output/colab/capstone_code.zip` to `MyDrive/capstone_rag/`; every notebook that imports `scripts` unpacks it (see `notebooks/README.md`). Then open `notebooks/colab_rag_pipelines.ipynb`, choose `PIPELINE` (`colpali_qdrant` or `vlm_chroma`), the sectors and `SMOKE_TEST`, then Run all. Its defaults are the full plan: all 13 sectors (`SECTORS = ""`), every planned PDF (`MAX_PER_SECTOR = 0`), and ColPali on every page. It calls `scripts/rag/colab_pipeline.py`, which you can also run from a terminal:
- **Live index:** on the VM's local disk. Each sector's PDFs are copied there from Drive before ingesting (`stage_pdfs`).
- **Drive:** the PDFs by sector, the VLM transcriptions, and a copy of the index (a Qdrant snapshot or the Chroma folder after each sector; `snapshot_every=N` snapshots every N batches of 10 filings instead).
- **Next session:** the index is restored from Drive, and finished filings are skipped.

```powershell
python -m scripts.rag.colab_pipeline run --pipeline vlm_chroma --sectors banking --vlm openai --qdrant-url ""   # local
python -m scripts.rag.colab_pipeline status --pipeline colpali_qdrant --qdrant-url ""
```

## Financial RAG Index

Fast local smoke-test mode with deterministic hashing embeddings:

```powershell
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hashing --reset
```

Hosted embedding mode for final capstone runs:

```powershell
$env:EMBEDDING_API_BASE="https://your-hosted-endpoint/v1"
$env:EMBEDDING_API_KEY="your-key"
$env:EMBEDDING_MODEL="your-embedding-model"
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hosted --reset
```

The financial indexer writes:

- chunks: `data/processed/financial_report_qa/pdf_chunks.jsonl`
- chart images: `data/processed/financial_report_qa/chart_images/`
- Chroma index: `indexes/chroma/financial_report_qa_pdfs`

## Retrieval Strategy

The RAG pipeline uses:

- PDF text extraction with page-level metadata
- Recursive character chunking with overlap: financial defaults to `chunk_size=2000`, `chunk_overlap=250` (the Qdrant sector-corpus ingest uses 1,200 / 200)
- ChromaDB vector index using cosine HNSW
- BM25 lexical retrieval over `pdf_chunks.jsonl`
- Reciprocal rank fusion of dense and BM25 results
- Lightweight lexical reranking
- Citation-grounded answers using chunk IDs

For 500-2,000 PDFs, HNSW in ChromaDB is the recommended default. IVF/PQ or IVF-PQ is mainly useful when the vector count becomes much larger or memory pressure becomes the main bottleneck.

## Run Agents

Financial agent, offline mock mode:

```powershell
python scripts\agents\financial_prompting_agent.py --llm-provider mock --technique all --query "What does Apple say can affect quarterly net sales?" --ticker AAPL
```

Financial agent with a hosted OpenAI-compatible <=8B chat API:

```powershell
$env:LLM_API_BASE="https://your-hosted-endpoint/v1"
$env:LLM_API_KEY="your-key"
$env:LLM_MODEL="your-chat-model"
python scripts\agents\financial_prompting_agent.py --llm-provider hosted --technique agentic_graph --query "What are revenue risks?" --ticker AAPL
```

## Streamlit Chat UIs

API-backed Streamlit UI (the Financial Report Copilot) for the final architecture:

```powershell
uvicorn scripts.api.api_app:app --host 127.0.0.1 --port 8000
streamlit run scripts\ui\api_streamlit_app.py
```

Financial UI:

```powershell
streamlit run scripts\ui\financial_streamlit_app.py
```

## FastAPI Backend

Run the backend:

```powershell
uvicorn scripts.api.api_app:app --host 127.0.0.1 --port 8000
```

Interactive docs are available at `http://127.0.0.1:8000/docs` (Swagger UI) and `http://127.0.0.1:8000/redoc` (ReDoc). The raw OpenAPI schema is at `http://127.0.0.1:8000/openapi.json`. See [docs/fastapi_api.md](docs/fastapi_api.md) for request fields, response shapes, error codes, examples, rate limits, and tracing setup. Streaming is not implemented; see the [proposed FastAPI streaming design](docs/fastapi_streaming_design.md).

Useful endpoints:

```text
GET  /health
GET  /domains
GET  /companies
POST /conversation/financial_report_qa
POST /chat/financial_report_qa
POST /retrieve/financial_report_qa
```

Example request:

```powershell
$body = @{
  query = "What was Costco's net income in fiscal 2025?"
  llm_provider = "mock"
  embedding_provider = "hashing"
  enable_guardrails = $true
  enable_human_review = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat/financial_report_qa" -ContentType "application/json" -Body $body
```

The API allows 60 requests per minute per client IP across all routes. Excess requests return `429 Too Many Requests` with a `Retry-After` header. The limiter is in-memory and per application process; use a shared rate-limit store such as Redis when running multiple workers or replicas. Client IPs come from the direct connection, so configure trusted proxy handling at the deployment boundary if the API is behind a reverse proxy.

Enable LangSmith for API calls by setting `LANGSMITH_API_KEY` and either passing `enable_langsmith=true` in the request body or setting `LANGSMITH_TRACING=true` in the environment.

API access logs include request IDs, route, status, and duration without recording request bodies or prompts. Each response returns `X-Request-ID`; when LangSmith tracing is active, the ID is attached as trace metadata. See [docs/fastapi_api.md](docs/fastapi_api.md#request-logging-and-trace-correlation).

Chat endpoints return generated answers only when citations validate against retrieved evidence; otherwise the answer is withheld and the response explains the grounding failure. The agent's critic also checks support: every figure must appear in a cited chunk and an LLM judge must support every claim; with a hosted LLM an unverified answer is replaced by cited excerpts (see [docs/agent_flows.md](docs/agent_flows.md)). Financial chat also returns `not_available` when the filings can't answer the question; see [docs/fastapi_api.md](docs/fastapi_api.md#when-the-filings-cant-answer).

## Deployment

The API, UI and Qdrant ship as one Docker Compose stack: CPU only, using the hosted LLM. It runs the same on a laptop or on any cloud VM, and optionally adds HTTPS with a login.

```bash
cp .env.example .env                                   # LLM endpoint, API_AUTH_TOKEN, ...
docker compose up -d qdrant
docker compose run --rm --user root api sh deploy/restore_index.sh   # Colab snapshots in ./snapshots
docker compose up -d --build                           # UI http://127.0.0.1:8501, API http://127.0.0.1:8000/docs
docker compose --profile https up -d                   # on a VM with DOMAIN set: https://$DOMAIN
```

[docs/deployment.md](docs/deployment.md) ([HTML](docs/deployment.html)) covers:
- the deployment options (laptop, Colab, a VM, Qdrant Cloud);
- every setting;
- costs: building the index, closed-model VLM transcription, serving, and the LLM per 1,000 questions.

Capstone build plan: [docs/capstone_implementation_plan.md](docs/capstone_implementation_plan.md). Feature specifications: [docs/features_to_implement.md](docs/features_to_implement.md) ([HTML](docs/features_to_implement.html)).

## End-to-end commands

Use hashing embeddings and mock LLM mode for local smoke tests. For final capstone runs, replace `--embedding-provider hashing` with `--embedding-provider hosted` after setting the embedding endpoint variables, and use `--llm-provider hosted` after setting the chat endpoint variables.

### Financial Report QA

```powershell
python scripts\downloads\download_financial_report_pdfs.py --target-pdfs 100
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hashing --reset
python scripts\agents\financial_prompting_agent.py --llm-provider mock --technique agentic_graph --query "What does Apple say can affect quarterly net sales?" --ticker AAPL
streamlit run scripts\ui\financial_streamlit_app.py
```

Optional top-k retrieval check:

```powershell
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hashing --query-only --retrieval-mode hybrid --query "What does Apple report about net sales and revenue?" --top-k 5 --filter-ticker AAPL
```

## Guardrails, Governance, and HITL

The agent flow includes:

- LangGraph orchestration for security, validation, planning, retrieval, evidence checks, answer generation, output filtering, citation criticism, and governance
- Input guard for prompt injection, code/tool payloads, and secrets
- LLM-backed plain-text/domain-intent classifier
- Retrieval evidence guard against injected instructions inside retrieved chunks
- Output content filter for generated prompt-injection echoes, executable payloads, secrets, sensitive data, and domain-sensitive recommendations
- Citation validation against retrieved chunk IDs
- Financial agent only: a plain "not available" reply, with no figure, for a company or year without filings or a measure a bank doesn't report; answers citing a company the question didn't ask about are withheld
- Human review routing for high-impact decisions, unsafe evidence, low confidence, or citation failure
- HTML/JSON audit artifacts for review

See [docs/agent_flows.md](docs/agent_flows.md) and `docs/financial_report_agent_design.md` for flow diagrams and governance details.

Content filtering is layered:

```text
user input -> input filter -> plain-text classifier -> retrieval -> evidence filter -> answer -> output filter -> citation/HITL
```

The filters return auditable JSON fields such as `security`, `evidence_guard`, `output_filter`, and `human_review`. Output filter decisions are `allow`, `review`, or `block`; review-level content adds `output_content_filter_review` to HITL triggers.

## LangSmith Logging and Monitoring

LangSmith tracing is optional and can be enabled for agent runs and retrieval evaluation.

Set your LangSmith credentials:

```powershell
$env:LANGSMITH_API_KEY="your-langsmith-key"
$env:LANGSMITH_PROJECT="capstone-financial-report-qa"
```

Run the financial agent with LangSmith tracing:

```powershell
python scripts\agents\financial_prompting_agent.py --enable-langsmith --langsmith-project capstone-financial-report-qa --llm-provider mock --technique agentic_graph --query "What does Apple say can affect quarterly net sales?" --ticker AAPL
```

Run retrieval evaluation with LangSmith tracing:

```powershell
python scripts\evaluation\evaluate_financial_rag.py --enable-langsmith --langsmith-project capstone-financial-rag-eval --embedding-provider hashing
```

Each trace includes tags and metadata for domain, retrieval mode, LLM provider, embedding provider, top-k settings, guardrail status, and HITL status. If `--enable-langsmith` is used without `LANGSMITH_API_KEY`, the run still completes locally and reports LangSmith as disabled in the JSON output.

## Retrieval Evaluation

Generate a silver retrieval set and measure recall separately from answer quality:

```powershell
python scripts\evaluation\evaluate_financial_rag.py --embedding-provider hashing --generate-silver
```

Run evaluation:

```powershell
python scripts\evaluation\evaluate_financial_rag.py --embedding-provider hashing
```

The evaluation reports recall@k and related retrieval metrics without calling the LLM.

## Final Capstone Run Checklist

Before a final demo or report run:

- Download the corpus (`min500` or `full`) and ingest it (see [docs/deployment.md](docs/deployment.md)).
- Rebuild indexes with hosted embeddings instead of hashing embeddings.
- Use hosted chat mode for the final Agentic RAG answer path through FastAPI.
- Run `uvicorn scripts.api.api_app:app --host 127.0.0.1 --port 8000` and `streamlit run scripts\ui\api_streamlit_app.py` for the final API-backed demo.
- Run retrieval recall@k evaluation separately from answer-quality evaluation.
- Capture LangSmith traces for at least representative successful, refused, and HITL-routed runs.
- Keep generated data, indexes, HTML reports, and eval artifacts out of git.

## Tests

Run the full unit test suite:

```powershell
python -m pytest tests
```

Current coverage includes:

- financial PDF downloader helpers
- financial indexing helpers
- hybrid retrieval and evaluation helpers
- financial prompting agent guardrails, content filters, LangGraph workflows, and HITL routing
- agent routing, tools, retrieval retry, calculator, support-checking critic and step budget
- the shared guard code in `rag_agent.py` (tested with `tests/sample_domain.py`)
- LangSmith optional tracing helpers
- the Streamlit feature guide and the financial-only API surface

## Notes

- Hashing embeddings are for local smoke tests and deterministic CI-style checks.
- Hosted embeddings and hosted chat APIs should be used for final capstone inference runs.
- ChromaDB is the default vector database; local Qdrant is available for multimodal PDF RAG.
- Downloaded PDFs, processed chunks, Chroma indexes, outputs, and eval artifacts are generated files and are excluded from git.

## Local Qdrant multimodal RAG

See [the local setup guide](docs/local_multimodal_rag.md) for PDF ingestion, hybrid retrieval, image-aware local answers, and six included sample PDFs covering all domains. Set `RAG_BACKEND=qdrant` to use Qdrant from the existing app.

The complete CPU profile uses learned embeddings, a cross-encoder reranker, and a local Qwen vision model. Run `.venv/Scripts/python.exe -m scripts.rag.run_local_multimodal evaluate` to validate text, tables, and an image-only chart; run the same module with `serve` to start the prepared local API and model service.

For ColPali page multivectors and Qdrant MaxSim retrieval, see [ColPali implementation and Qdrant storage](docs/colpali_qdrant.md). Backfill the existing index with `python -m scripts.rag.colpali_index backfill`, then start the API with `python -m scripts.rag.run_local_multimodal serve --colpali`.

Selective OCR now adds scanned-page and embedded-image text to the same text-vector collection. Use `python -m scripts.rag.ocr_index` to backfill existing PDFs; see [OCR implementation and validation](docs/selective_ocr.md).
