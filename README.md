# Track B Multi-Agent Operations Copilot

Agentic RAG system for the Track B capstone: **API-side Multi-Agent Operations Copilot built around hosted <=8B inference APIs**.

The project implements citation-grounded Agentic RAG over public PDF corpora, hybrid retrieval, LangGraph-orchestrated guarded workflows, human-in-the-loop routing, Streamlit chat UIs, and retrieval evaluation.

## Why This Is Agentic RAG

This is **Agentic RAG** because retrieval is only one step inside a stateful agent workflow. A plain RAG chain usually does `query -> retrieve -> answer`; this project adds LangGraph nodes that validate the request, plan retrieval, call hybrid search tools, inspect retrieved evidence, draft an answer, filter unsafe output, validate citations, and route risky cases to human review.

The implemented agent loop has these decision-making stages:

```text
security_guard -> plain_text_classifier -> planner -> hybrid_retriever -> evidence_guard -> answer_agent -> output_filter -> citation_critic -> governance
```

That makes the system agentic in the capstone sense: it uses tools, maintains structured state, applies bounded decisions, checks its own evidence/citations, records observability metadata, and can stop or escalate instead of always answering.

## Domains

Implemented domains:

- Financial report question answering
- Procurement and vendor-review assistant
- Travel-and-expense policy agent
- Competitive-research analyst
- Incident postmortem drafter
- HR policy navigator

The financial domain has a richer dedicated implementation. The remaining domains share a common LangGraph RAG and guarded-agent runtime in `scripts/domain_common/rag_agent.py`.

## Implementation Status

| Area | Status |
|---|---|
| Financial PDF RAG | Implemented with PDF verification, chunking, Chroma, BM25, RRF, reranking, optional table summaries, optional chart-summary chunks, and top-k query mode. |
| Non-financial domain RAG | Implemented for procurement, travel/expense, competitive research, incident postmortem, and HR policy using the shared `scripts/domain_common/rag_agent.py` runtime. |
| Agent orchestration | Implemented with LangGraph nodes for security, plain-text validation, planning, hybrid retrieval, evidence guard, answer generation, output filtering, citation criticism, and governance/HITL. |
| Guardrails and HITL | Implemented for input filtering, evidence prompt-injection filtering, output content filtering, citation validation, and human-review routing. |
| API backend and UI | Implemented with FastAPI backend endpoints plus Streamlit frontends, including an API-backed Streamlit app for the final architecture. |
| Observability | Optional LangSmith tracing is implemented for agents and financial retrieval evaluation. |
| Evaluation | Financial retrieval recall@k evaluation is implemented separately from answer generation. |

## Repository Layout

```text
docs/                         Design notes and capstone documentation
scripts/
  domain_common/              Shared LangGraph PDF RAG, guardrails, HITL, HTML rendering
  procurement_vendor/         Domain config, index CLI, agent CLI, Streamlit app
  travel_expense/             Domain config, index CLI, agent CLI, Streamlit app
  competitive_research/       Domain config, index CLI, agent CLI, Streamlit app
  incident_postmortem/        Domain config, index CLI, agent CLI, Streamlit app
  hr_policy/                  Domain config, index CLI, agent CLI, Streamlit app
  download_domain_pdfs.py     Downloads PDFs for non-financial domains
  api_app.py                  FastAPI backend for all domains
  api_streamlit_app.py        Streamlit frontend that calls the FastAPI backend
  download_financial_report_pdfs.py
  index_financial_report_pdfs.py
  financial_prompting_agent.py
  financial_streamlit_app.py
  evaluate_financial_rag.py
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
python scripts\download_financial_report_pdfs.py --target-pdfs 5
python scripts\index_financial_report_pdfs.py --embedding-provider hashing --limit-pdfs 5 --max-pages-per-pdf 10 --reset
python scripts\financial_prompting_agent.py --llm-provider mock --technique agentic_graph --query "What does Apple say can affect quarterly net sales?" --ticker AAPL
```

For a non-financial domain smoke test:

```powershell
python scripts\download_domain_pdfs.py --domains travel_expense
python scripts\travel_expense\index.py --embedding-provider hashing --max-pages-per-pdf 5 --reset
python scripts\travel_expense\agent.py --query "What documentation is required for travel expenses?"
```

Start the final API-side demo stack:

```powershell
uvicorn scripts.api_app:app --host 127.0.0.1 --port 8000
streamlit run scripts\api_streamlit_app.py
```

## Data Download

Download financial annual report PDFs:

```powershell
python scripts\download_financial_report_pdfs.py --target-pdfs 100
```

Download the public PDF corpora for the other Track B domains:

```powershell
python scripts\download_domain_pdfs.py
```

Optional raw JSON/API corpora for the recommended domains can also be downloaded for corpus analysis and future retrieval expansion:

```powershell
python scripts\download_rag_corpora.py --domains all --target-docs 500 --overwrite --sec-user-agent "capstone-rag/0.1 your_email@example.com"
```

PDF manifests are written to:

```text
data/raw/<domain>/pdf_manifest.jsonl
```

## Financial RAG Index

Fast local smoke-test mode with deterministic hashing embeddings:

```powershell
python scripts\index_financial_report_pdfs.py --embedding-provider hashing --reset
```

Hosted embedding mode for final capstone runs:

```powershell
$env:EMBEDDING_API_BASE="https://your-hosted-endpoint/v1"
$env:EMBEDDING_API_KEY="your-key"
$env:EMBEDDING_MODEL="your-embedding-model"
python scripts\index_financial_report_pdfs.py --embedding-provider hosted --reset
```

The financial indexer writes:

- chunks: `data/processed/financial_report_qa/pdf_chunks.jsonl`
- chart images: `data/processed/financial_report_qa/chart_images/`
- Chroma index: `indexes/chroma/financial_report_qa_pdfs`

## Other Domain RAG Indexes

Each non-financial domain has the same command pattern.

```powershell
python scripts\procurement_vendor\index.py --embedding-provider hashing --reset
python scripts\travel_expense\index.py --embedding-provider hashing --reset
python scripts\competitive_research\index.py --embedding-provider hashing --reset
python scripts\incident_postmortem\index.py --embedding-provider hashing --reset
python scripts\hr_policy\index.py --embedding-provider hashing --reset
```

For quick smoke tests, limit pages:

```powershell
python scripts\hr_policy\index.py --embedding-provider hashing --max-pages-per-pdf 5 --reset
```

## Retrieval Strategy

The RAG pipeline uses:

- PDF text extraction with page-level metadata
- Recursive character chunking with overlap: financial defaults to `chunk_size=2000`, `chunk_overlap=250`; shared domain defaults are `chunk_size=1800`, `chunk_overlap=220`
- ChromaDB vector index using cosine HNSW
- BM25 lexical retrieval over `pdf_chunks.jsonl`
- Reciprocal rank fusion of dense and BM25 results
- Lightweight lexical reranking
- Citation-grounded answers using chunk IDs

For 500-2,000 PDFs, HNSW in ChromaDB is the recommended default. IVF/PQ or IVF-PQ is mainly useful when the vector count becomes much larger or memory pressure becomes the main bottleneck.

The current shared non-financial runtime indexes PDF chunks from `data/raw/<domain>/pdf_manifest.jsonl`. The JSON/API corpora downloaded by `download_rag_corpora.py` are retained as raw data and can be merged into the same Chroma/BM25 pattern later.

## Run Agents

Financial agent, offline mock mode:

```powershell
python scripts\financial_prompting_agent.py --llm-provider mock --technique all --query "What does Apple say can affect quarterly net sales?" --ticker AAPL
```

Financial agent with a hosted OpenAI-compatible <=8B chat API:

```powershell
$env:LLM_API_BASE="https://your-hosted-endpoint/v1"
$env:LLM_API_KEY="your-key"
$env:LLM_MODEL="your-chat-model"
python scripts\financial_prompting_agent.py --llm-provider hosted --technique agentic_graph --query "What are revenue risks?" --ticker AAPL
```

Other domain agents:

```powershell
python scripts\procurement_vendor\agent.py --query "What evidence describes vendor evaluation criteria?"
python scripts\travel_expense\agent.py --query "What documentation is required for travel expenses?"
python scripts\competitive_research\agent.py --query "Compare competitor risk language about market demand."
python scripts\incident_postmortem\agent.py --query "What should be included in an incident postmortem?"
python scripts\hr_policy\agent.py --query "What does the FMLA guide say about requesting leave?"
```

To write an HTML report:

```powershell
python scripts\hr_policy\agent.py --query "What does the FMLA guide say about requesting leave?" --html-out output\hr_policy_agent.html
```

## Streamlit Chat UIs

API-backed Streamlit UI for the final architecture:

```powershell
uvicorn scripts.api_app:app --host 127.0.0.1 --port 8000
streamlit run scripts\api_streamlit_app.py
```

Financial UI:

```powershell
streamlit run scripts\financial_streamlit_app.py
```

Domain-specific UIs:

```powershell
streamlit run scripts\procurement_vendor\html_app.py
streamlit run scripts\travel_expense\html_app.py
streamlit run scripts\competitive_research\html_app.py
streamlit run scripts\incident_postmortem\html_app.py
streamlit run scripts\hr_policy\html_app.py
```

## FastAPI Backend

Run the backend:

```powershell
uvicorn scripts.api_app:app --host 127.0.0.1 --port 8000
```

Interactive docs are available at `http://127.0.0.1:8000/docs` (Swagger UI) and `http://127.0.0.1:8000/redoc` (ReDoc). The raw OpenAPI schema is at `http://127.0.0.1:8000/openapi.json`. See [docs/fastapi_api.md](docs/fastapi_api.md) for request fields, response shapes, error codes, examples, rate limits, and tracing setup. Streaming is not implemented; see the [proposed FastAPI streaming design](docs/fastapi_streaming_design.md).

Useful endpoints:

```text
GET  /health
GET  /domains
POST /chat/financial_report_qa
POST /chat/{domain}
POST /retrieve/financial_report_qa
POST /retrieve/{domain}
```

Example request:

```powershell
$body = @{
  query = "What documentation is required for travel expenses?"
  llm_provider = "mock"
  embedding_provider = "hashing"
  enable_guardrails = $true
  enable_human_review = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/chat/travel_expense" -ContentType "application/json" -Body $body
```

The API allows 60 requests per minute per client IP across all routes. Excess requests return `429 Too Many Requests` with a `Retry-After` header. The limiter is in-memory and per application process; use a shared rate-limit store such as Redis when running multiple workers or replicas. Client IPs come from the direct connection, so configure trusted proxy handling at the deployment boundary if the API is behind a reverse proxy.

Enable LangSmith for API calls by setting `LANGSMITH_API_KEY` and either passing `enable_langsmith=true` in the request body or setting `LANGSMITH_TRACING=true` in the environment.

API access logs include request IDs, route, status, and duration without recording request bodies or prompts. Each response returns `X-Request-ID`; when LangSmith tracing is active, the ID is attached as trace metadata. See [docs/fastapi_api.md](docs/fastapi_api.md#request-logging-and-trace-correlation).

Chat endpoints return generated answers only when citations validate against retrieved evidence; otherwise the answer is withheld and the response explains the grounding failure. This verifies evidence/citation consistency, not semantic entailment or factual correctness.

## Execute By Domain

Use hashing embeddings and mock LLM mode for local smoke tests. For final capstone runs, replace `--embedding-provider hashing` with `--embedding-provider hosted` after setting the embedding endpoint variables, and use `--llm-provider hosted` after setting the chat endpoint variables.

### Financial Report QA

```powershell
python scripts\download_financial_report_pdfs.py --target-pdfs 100
python scripts\index_financial_report_pdfs.py --embedding-provider hashing --reset
python scripts\financial_prompting_agent.py --llm-provider mock --technique agentic_graph --query "What does Apple say can affect quarterly net sales?" --ticker AAPL
streamlit run scripts\financial_streamlit_app.py
```

Optional top-k retrieval check:

```powershell
python scripts\index_financial_report_pdfs.py --embedding-provider hashing --query-only --retrieval-mode hybrid --query "What does Apple report about net sales and revenue?" --top-k 5 --filter-ticker AAPL
```

### Procurement And Vendor Review

```powershell
python scripts\download_domain_pdfs.py --domains procurement_vendor
python scripts\procurement_vendor\index.py --embedding-provider hashing --reset
python scripts\procurement_vendor\agent.py --query "What evidence describes vendor evaluation criteria?" --html-out output\procurement_vendor_agent.html
streamlit run scripts\procurement_vendor\html_app.py
```

### Travel And Expense Policy

```powershell
python scripts\download_domain_pdfs.py --domains travel_expense
python scripts\travel_expense\index.py --embedding-provider hashing --reset
python scripts\travel_expense\agent.py --query "What documentation is required for travel expenses?" --html-out output\travel_expense_agent.html
streamlit run scripts\travel_expense\html_app.py
```

### Competitive Research

```powershell
python scripts\download_domain_pdfs.py --domains competitive_research
python scripts\competitive_research\index.py --embedding-provider hashing --reset
python scripts\competitive_research\agent.py --query "Compare competitor risk language about market demand." --html-out output\competitive_research_agent.html
streamlit run scripts\competitive_research\html_app.py
```

### Incident Postmortem

```powershell
python scripts\download_domain_pdfs.py --domains incident_postmortem
python scripts\incident_postmortem\index.py --embedding-provider hashing --reset
python scripts\incident_postmortem\agent.py --query "What should be included in an incident postmortem?" --html-out output\incident_postmortem_agent.html
streamlit run scripts\incident_postmortem\html_app.py
```

### HR Policy Navigator

```powershell
python scripts\download_domain_pdfs.py --domains hr_policy
python scripts\hr_policy\index.py --embedding-provider hashing --reset
python scripts\hr_policy\agent.py --query "What does the FMLA guide say about requesting leave?" --html-out output\hr_policy_agent.html
streamlit run scripts\hr_policy\html_app.py
```

## Guardrails, Governance, and HITL

The agent flow includes:

- LangGraph orchestration for security, validation, planning, retrieval, evidence checks, answer generation, output filtering, citation criticism, and governance
- Input guard for prompt injection, code/tool payloads, and secrets
- LLM-backed plain-text/domain-intent classifier
- Retrieval evidence guard against injected instructions inside retrieved chunks
- Output content filter for generated prompt-injection echoes, executable payloads, secrets, sensitive data, and domain-sensitive recommendations
- Citation validation against retrieved chunk IDs
- Human review routing for high-impact decisions, unsafe evidence, low confidence, or citation failure
- HTML/JSON audit artifacts for review

See `docs/financial_report_agent_design.md` and the domain design files under `docs/` for flow diagrams and governance details.

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
python scripts\financial_prompting_agent.py --enable-langsmith --langsmith-project capstone-financial-report-qa --llm-provider mock --technique agentic_graph --query "What does Apple say can affect quarterly net sales?" --ticker AAPL
```

Run retrieval evaluation with LangSmith tracing:

```powershell
python scripts\evaluate_financial_rag.py --enable-langsmith --langsmith-project capstone-financial-rag-eval --embedding-provider hashing
```

Run any generic domain agent with LangSmith tracing:

```powershell
python scripts\hr_policy\agent.py --enable-langsmith --langsmith-project capstone-hr-policy --query "What does the FMLA employee guide say about requesting leave?"
```

Each trace includes tags and metadata for domain, retrieval mode, LLM provider, embedding provider, top-k settings, guardrail status, and HITL status. If `--enable-langsmith` is used without `LANGSMITH_API_KEY`, the run still completes locally and reports LangSmith as disabled in the JSON output.

## Retrieval Evaluation

Generate a silver retrieval set and measure recall separately from answer quality:

```powershell
python scripts\evaluate_financial_rag.py --embedding-provider hashing --generate-silver
```

Run evaluation:

```powershell
python scripts\evaluate_financial_rag.py --embedding-provider hashing
```

The evaluation reports recall@k and related retrieval metrics without calling the LLM.

## Final Capstone Run Checklist

Before a final demo or report run:

- Download enough PDFs to reach the required corpus size for the selected domain.
- Rebuild indexes with hosted embeddings instead of hashing embeddings.
- Use hosted chat mode for the final Agentic RAG answer path through FastAPI.
- Run `uvicorn scripts.api_app:app --host 127.0.0.1 --port 8000` and `streamlit run scripts\api_streamlit_app.py` for the final API-backed demo.
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
- generic domain PDF downloader helpers
- financial indexing helpers
- hybrid retrieval and evaluation helpers
- financial prompting agent guardrails, content filters, LangGraph workflows, and HITL routing
- shared domain LangGraph RAG orchestration, guardrails, citation checks, output filters, and HITL routing
- LangSmith optional tracing helpers
- all domain configs and entrypoint wiring

## Notes

- Hashing embeddings are for local smoke tests and deterministic CI-style checks.
- Hosted embeddings and hosted chat APIs should be used for final capstone inference runs.
- ChromaDB is the default vector database; local Qdrant is available for multimodal PDF RAG.
- Downloaded PDFs, processed chunks, Chroma indexes, outputs, and eval artifacts are generated files and are excluded from git.

## Local Qdrant multimodal RAG

See [the local setup guide](docs/local_multimodal_rag.md) for PDF ingestion, hybrid retrieval, image-aware local answers, and six included sample PDFs covering all domains. Set `RAG_BACKEND=qdrant` to use Qdrant from the existing app.

The complete CPU profile uses learned embeddings, a cross-encoder reranker, and a local Qwen vision model. Run `.venv/Scripts/python.exe -m scripts.run_local_multimodal evaluate` to validate text, tables, and an image-only chart; run the same module with `serve` to start the prepared local API and model service.

For ColPali page multivectors and Qdrant MaxSim retrieval, see [ColPali implementation and Qdrant storage](docs/colpali_qdrant.md). Backfill the existing index with `python -m scripts.colpali_index backfill`, then start the API with `python -m scripts.run_local_multimodal serve --colpali`.

Selective OCR now adds scanned-page and embedded-image text to the same text-vector collection. Use `python -m scripts.ocr_index` to backfill existing PDFs; see [OCR implementation and validation](docs/selective_ocr.md).
