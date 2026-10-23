# Track B Multi-Agent Operations Copilot

Agentic RAG system for the Track B capstone: **API-side Multi-Agent Operations Copilot built around hosted <=8B inference APIs**.

The project implements citation-grounded RAG over public PDF corpora, hybrid retrieval, guarded agent workflows, human-in-the-loop routing, Streamlit chat UIs, and retrieval evaluation.

## Domains

Implemented domains:

- Financial report question answering
- Procurement and vendor-review assistant
- Travel-and-expense policy agent
- Competitive-research analyst
- Incident postmortem drafter
- HR policy navigator

The financial domain has a richer dedicated implementation. The remaining domains share a common generic RAG and guarded-agent runtime in `scripts/domain_common/rag_agent.py`.

## Repository Layout

```text
docs/                         Design notes and capstone documentation
scripts/
  domain_common/              Shared PDF RAG, guardrails, HITL, HTML rendering
  procurement_vendor/         Domain config, index CLI, agent CLI, Streamlit app
  travel_expense/             Domain config, index CLI, agent CLI, Streamlit app
  competitive_research/       Domain config, index CLI, agent CLI, Streamlit app
  incident_postmortem/        Domain config, index CLI, agent CLI, Streamlit app
  hr_policy/                  Domain config, index CLI, agent CLI, Streamlit app
  download_domain_pdfs.py     Downloads PDFs for non-financial domains
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

## Data Download

Download financial annual report PDFs:

```powershell
python scripts\download_financial_report_pdfs.py --target-pdfs 100
```

Download the public PDF corpora for the other Track B domains:

```powershell
python scripts\download_domain_pdfs.py
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
- Recursive character chunking with overlap
- ChromaDB vector index using cosine HNSW
- BM25 lexical retrieval over `pdf_chunks.jsonl`
- Reciprocal rank fusion of dense and BM25 results
- Lightweight lexical reranking
- Citation-grounded answers using chunk IDs

For 500-2,000 PDFs, HNSW in ChromaDB is the recommended default. IVF/PQ or IVF-PQ is mainly useful when the vector count becomes much larger or memory pressure becomes the main bottleneck.

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

## Guardrails, Governance, and HITL

The agent flow includes:

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
- financial prompting agent guardrails and workflows
- shared domain RAG guardrails, citation checks, and HITL routing
- all domain configs and entrypoint wiring

## Notes

- Hashing embeddings are for local smoke tests and deterministic CI-style checks.
- Hosted embeddings and hosted chat APIs should be used for final capstone inference runs.
- ChromaDB is the project vector database.
- Downloaded PDFs, processed chunks, Chroma indexes, outputs, and eval artifacts are generated files and are excluded from git.
