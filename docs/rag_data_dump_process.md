# Track B RAG Data Dump Process

This project follows Track B: API-side multi-agent operations copilot. The
corpus is for retrieval, citations, evaluation, and tool grounding. It is not a
training set and should not be used for fine-tuning.

## Why The Built System Is Agentic RAG

The downloaded corpora support RAG, but the implemented project is **Agentic RAG** because the indexed evidence is consumed by governed agents rather than by a single retrieve-and-answer prompt. The financial agent and the shared non-financial domain agents use LangGraph to coordinate input security, plain-text/domain validation, planning, hybrid retrieval, evidence filtering, answer generation, output filtering, citation validation, and HITL governance.

In practice, the data dump feeds tools such as dense retrieval, BM25 retrieval, table/chart lookup, and citation validation. The agent decides when to use those tools, when evidence is insufficient, when output should be blocked, and when a human reviewer is required.

## Folder layout

Run:

```powershell
python scripts\downloads\download_rag_corpora.py --domains all --target-docs 500 --overwrite --sec-user-agent "capstone-rag/0.1 your_email@example.com"
```

The downloader writes:

```text
data/raw/
  procurement_vendor/
    documents/*.json
    manifest.jsonl
    source_files/
  travel_expense/
    documents/*.json
    manifest.jsonl
    source_files/
  competitive_research/
    documents/*.json
    manifest.jsonl
    source_files/
  incident_postmortem/
    documents/*.json
    manifest.jsonl
    source_files/
  financial_report_qa/
    documents/*.json
    manifest.jsonl
    source_files/
  hr_policy/
    documents/*.json
    manifest.jsonl
    source_files/
  download_summary.json
```

Financial report PDFs are stored as a separate raw layer:

```text
data/raw/financial_report_qa/
  pdfs/*.pdf
  pdf_manifest.jsonl
```

Each file in `documents/` is a raw retrieval document with:

- `document_id`
- `title`
- `text`
- `source_name`
- `source_url`
- `metadata`
- `downloaded_at`
- `content_sha256`

Keep this raw folder immutable after download. Later steps should write cleaned
documents, chunks, indexes, and evaluation artifacts into separate versioned
folders such as `data/processed/`, `data/chunks/`, and `indexes/`.

## Domain source map

| Domain | Source family | Why it fits RAG |
|---|---|---|
| `procurement_vendor` | World Bank procurement notices and contract awards, with eCFR Title 48 fallback | Vendor names, award descriptions, procurement methods, dates, countries, and FAR policy sections support vendor-review and procurement QA. |
| `travel_expense` | eCFR Title 41, especially Federal Travel Regulation sections | Public travel policy corpus with section-level legal citations and stable source URLs. |
| `competitive_research` | SEC EDGAR company submissions metadata for major public companies | Company, industry, filing, and disclosure metadata support competitor monitoring and research routing. |
| `incident_postmortem` | Public Statuspage incident histories plus NTSB aviation incident data fallback | Incident timelines, impact, updates, status changes, and root-cause style records support postmortem drafting patterns. |
| `financial_report_qa` | SEC EDGAR XBRL companyfacts | Numeric financial facts, concepts, forms, filing dates, fiscal periods, and company metadata support cited financial QA. |
| `hr_policy` | eCFR Title 29 Labor | Public labor and employment policy sections support HR policy navigation with section-grounded answers. |

## Required next steps

1. Inventory and EDA
   - Count documents by domain, source, type, and date.
   - Report parse failures, empty documents, duplicate hashes, very short and very long documents.
   - Confirm every document has source URL, title, source name, and content hash.

2. Clean and normalize
   - Write cleaned output outside `data/raw/`.
   - Normalize whitespace and HTML.
   - Deduplicate exact hashes and near-duplicates.
   - Preserve legal section numbers, table labels, dates, units, record IDs, company names, and filing accessions.

3. Chunking experiments
   - Strategy A: structure-aware chunks by source record, section, heading, paragraph, and table row groups.
   - Strategy B: recursive token chunks around 400-700 tokens with 50-100 token overlap.
   - Store `chunk_id`, `document_id`, `domain`, `source_url`, heading/section, record/page range, `content_sha256`, parser version, chunker version, and corpus version.

4. Hybrid retrieval
   - Build a dense index in Chroma from the same chunk IDs.
   - Build a BM25 index over the same authorized chunks.
   - Retrieve 20-50 candidates from each retriever.
   - Fuse using reciprocal rank fusion.
   - Deduplicate by document/section.

5. Reranking
   - Add a cross-encoder reranker such as `BAAI/bge-reranker-base` or `cross-encoder/ms-marco-MiniLM-L-6-v2` if local compute permits.
   - If not, use the hosted endpoint as a budgeted reranker with a strict schema.
   - Pass only the best 3-6 cited chunks to generation.

6. Evaluation
   - Freeze 60-100 gold queries before tuning prompts or agent behavior.
   - For each query, store gold answer, answerability, relevant `document_id` or `chunk_id`, domain, allowed tools, and refusal expectation.
   - Measure retrieval independently with recall@k, MRR/nDCG, and latency.
   - Measure end-to-end answer quality separately with correctness, citation support, abstention, schema validity, tool-call success, and latency.
   - For smoke testing before the human-authored gold set is ready, generate a clearly labeled silver retrieval set from indexed chunks and do not report it as final capstone quality.

7. Citation-grounded answers
   - Answers must cite retrieved chunk IDs and source URLs.
   - Reject or abstain when the answer is not supported by retrieved evidence.
   - Validate that every citation ID returned to the user exists in the retrieved evidence set.

## Financial PDF RAG stack

Use this stack for the annual-report PDF corpus:

| Layer | Choice | Notes |
|---|---|---|
| PDF reader | `pypdf` | Already installed. Use it for page text extraction and page-aware citations. |
| Chunking | `RecursiveCharacterTextSplitter` | First pass uses 2,000 characters with 250 overlap, page by page, so citations retain page metadata. |
| Vector DB | ChromaDB | Local persistent DB suitable for the 500-2,000 document capstone prototype. |
| Final embeddings | Hosted `<=8B` track embedding endpoint | Use `--embedding-provider hosted` with `EMBEDDING_API_BASE`, `EMBEDDING_API_KEY`, and `EMBEDDING_MODEL`. |
| Local smoke-test embeddings | Hashing embeddings | Deterministic fallback only for validating the pipeline without network/model download. Replace with hosted embeddings for evaluation. |
| Chart handling | `chart_summary` chunks | Extract large embedded PDF images, save them to disk, and index text summaries with source PDF, page, ticker, year, and image path. This is not image embedding; it is a simple retrieval hook for later visual review. |
| Table handling | `table_summary` chunks | Use `pdfplumber` to extract selectable PDF tables into markdown-like chunks with page, ticker, year, row count, and column count metadata. |
| OCR handling | Optional local Tesseract | Use `--enable-ocr` to OCR extracted chart images when Tesseract is installed. If Tesseract is unavailable, chart chunks are still created with `ocr_status=unavailable`. |

Build the local smoke-test Chroma index:

```powershell
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hashing --reset --max-pages-per-pdf 20
```

Build the local smoke-test index with chart/figure retrieval hooks:

```powershell
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hashing --reset --max-pages-per-pdf 20 --include-chart-summaries
```

Build the local smoke-test index with tables, chart/figure hooks, and optional OCR status:

```powershell
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hashing --reset --max-pages-per-pdf 20 --include-tables --include-chart-summaries --enable-ocr
```

For final table coverage, index the full PDFs or raise `--max-pages-per-pdf`; the first 20 pages mostly cover early filing sections and may miss financial statement tables later in the report.

Retrieve top-k chunks from the existing Chroma index with dense retrieval:

```powershell
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hashing --query-only --retrieval-mode dense --query "What does Apple report about net sales and revenue?" --top-k 5 --filter-ticker AAPL
```

Retrieve top-k chunks with hybrid retrieval: BM25 + Chroma dense retrieval, reciprocal rank fusion, and simple reranking:

```powershell
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hashing --query-only --retrieval-mode hybrid --query "What does Apple report about net sales and revenue?" --top-k 5 --dense-top-k 30 --bm25-top-k 30 --filter-ticker AAPL
```

Retrieve only chart/figure summary chunks:

```powershell
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hashing --query-only --retrieval-mode hybrid --query "Find annual report chart or figure image for Microsoft" --top-k 3 --filter-source-type chart_summary --filter-ticker MSFT
```

Retrieve only table summary chunks:

```powershell
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hashing --query-only --retrieval-mode hybrid --query "Apple table net sales products services" --top-k 5 --filter-source-type table_summary --filter-ticker AAPL
```

Run a 100-query silver retrieval evaluation with current hashing embeddings:

```powershell
python scripts\evaluation\evaluate_financial_rag.py --embedding-provider hashing --retrieval-mode hybrid --generate-silver --num-silver-queries 100 --dense-top-k 30 --bm25-top-k 30
```

This writes:

```text
eval/financial_report_qa/retrieval_eval_silver.json
eval/financial_report_qa/retrieval_eval_results.json
```

Build the final hosted-embedding Chroma index:

```powershell
$env:EMBEDDING_API_BASE="https://your-hosted-endpoint/v1"
$env:EMBEDDING_API_KEY="your_api_key"
$env:EMBEDDING_MODEL="your_embedding_model"
python scripts\rag\index_financial_report_pdfs.py --embedding-provider hosted --reset --include-tables --include-chart-summaries --enable-ocr
```

The indexer writes:

```text
data/processed/financial_report_qa/pdf_chunks.jsonl
data/processed/financial_report_qa/pdf_verification.jsonl
data/processed/financial_report_qa/chart_images/
indexes/chroma/financial_report_qa_pdfs/
indexes/chroma/financial_report_qa_pdfs/index_summary.json
```

## Practical commands

Download only policy-heavy domains:

```powershell
python scripts\downloads\download_rag_corpora.py --domains travel_expense hr_policy --target-docs 500 --overwrite
```

Download only SEC-heavy domains:

```powershell
python scripts\downloads\download_rag_corpora.py --domains competitive_research financial_report_qa --target-docs 500 --overwrite --sec-user-agent "capstone-rag/0.1 your_email@example.com"
```

Download financial annual report PDFs:

```powershell
python scripts\downloads\download_financial_report_pdfs.py --target-pdfs 50 --years 2023 2022 2021 2020 2019 --timeout 5 --user-agent "capstone-rag-pdf/0.1 your_email@example.com"
```

Download public PDF corpora for the remaining Track B domains:

```powershell
python scripts\downloads\download_domain_pdfs.py --domains procurement_vendor travel_expense competitive_research incident_postmortem hr_policy
```

This writes:

```text
data/raw/<domain>/pdfs/*.pdf
data/raw/<domain>/pdf_manifest.jsonl
data/raw/domain_pdf_download_summary.json
```

Index a smoke RAG corpus for each non-financial domain:

```powershell
python scripts\domains\procurement_vendor\index.py --embedding-provider hashing --max-pages-per-pdf 5 --reset
python scripts\domains\travel_expense\index.py --embedding-provider hashing --max-pages-per-pdf 5 --reset
python scripts\domains\competitive_research\index.py --embedding-provider hashing --max-pages-per-pdf 5 --reset
python scripts\domains\incident_postmortem\index.py --embedding-provider hashing --max-pages-per-pdf 5 --reset
python scripts\domains\hr_policy\index.py --embedding-provider hashing --max-pages-per-pdf 5 --reset
```

Run a guarded domain agent and write an HTML result:

```powershell
python scripts\domains\procurement_vendor\agent.py --query "What evidence describes vendor evaluation criteria?" --html-out output\procurement_vendor_agent.html
```

The non-financial domain agents use the shared LangGraph runtime in `scripts/agents/rag_agent.py`. The graph runs:

```text
security_guard -> plain_text_classifier -> planner -> hybrid_retriever -> evidence_guard -> answer_agent -> output_filter -> citation_critic -> governance
```

The JSON result includes `orchestration.engine=langgraph`, guardrail decisions, citation validation, output content filtering, HITL routing, and summarized evidence.

Run a domain agent with optional LangSmith tracing:

```powershell
$env:LANGSMITH_API_KEY="your_langsmith_key"
python scripts\domains\travel_expense\agent.py --enable-langsmith --langsmith-project capstone-travel-expense --query "What documentation is required for travel expenses?"
```

Launch a domain HTML/Streamlit chat UI:

```powershell
streamlit run scripts\domains\procurement_vendor\html_app.py
```

Download a smoke-test corpus:

```powershell
python scripts\downloads\download_rag_corpora.py --domains all --target-docs 25 --overwrite --sec-user-agent "capstone-rag/0.1 your_email@example.com"
```

## Current implementation notes

- Financial report QA uses a dedicated LangGraph `agentic_graph` plus baseline, structured few-shot, decomposition, and self-consistency prompting modes.
- The remaining domains use the shared LangGraph domain runtime.
- Current shared domain retrieval indexes PDF chunks from `data/raw/<domain>/pdf_manifest.jsonl`; the 500-document JSON corpora are downloaded and available as raw corpora, but they are not yet merged into the shared domain Chroma/BM25 retrieval path.
- Content filtering is layered at input, retrieved evidence, and generated output.
- `output_filter.decision=block` prevents release; `output_filter.decision=review` adds `output_content_filter_review` to HITL triggers when human review is enabled.
- LangSmith tracing is optional and safe to enable without breaking local smoke tests. Without `LANGSMITH_API_KEY`, the run completes and reports tracing as disabled.
- Unit tests currently cover downloader helpers, indexing helpers, retrieval/evaluation helpers, financial prompting workflows, shared LangGraph domain orchestration, content filters, HITL routing, LangSmith helpers, and domain entrypoints.
