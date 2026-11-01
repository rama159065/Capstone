# Financial-report generation and monitoring

## Purpose and delivered example

Generate a detailed, source-reconciled financial report for a company using an
explicit annual-report profile. The Apple example covers the fiscal year ended
September 28, 2024, with FY2022-FY2024 income/cash-flow comparisons and FY2023-FY2024
balance sheets. FY2024 is the most recent Apple filing in the local source folder
used for this example; it is not asserted to be the latest public filing.

Outputs include an eight-page PDF, a companion Markdown report, five charts in
both PNG and SVG formats, extracted evidence, calculated metrics, a manifest,
rendered page previews and a persistent JSONL trace.

Completed Apple example from this session:

- Trace ID: `eb92c9fd-c252-4562-9799-d3115a3c59ca`.
- Logical thread ID: `f2681e5b-3f72-4d90-b07a-3dd3f406e275`.
- Directory: `output/pdf/AAPL_2024/eb92c9fd-c252-4562-9799-d3115a3c59ca/`.
- Manifest status: `completed`; all eight rendered pages were inspected.
- Trace: 86 persisted events, including 43 numerical row extractions, five
  statement retrievals, nine narrative anchors, seven agent stages and five charts.
- Seventeen arithmetic reconciliations passed. This run used no LLM; the skipped
  stage is explicit in the log. LLM/retrieval trace propagation is regression-tested.

## Reliability commitment

No implementation can truthfully promise zero failures or eliminate every
extraction, model, filesystem or service error. This workflow instead provides
these concrete safeguards:

1. Financial statement numbers come from the source PDF, not an LLM.
2. The source must match the profile's SHA-256 checksum.
3. Ambiguous statements, missing rows and incomplete numeric columns stop the
   numerical report; values are not guessed or borrowed from a different row.
4. Seventeen arithmetic reconciliations must pass for the Apple example.
5. Calculations use Python Decimal; display rounding happens after calculation.
6. Tables and charts use the same extracted dataset.
7. Every run has a unique trace ID and a terminal manifest state; failures are
   visible and retain their logs. A failed run is not a completed report.
8. Rendering and text checks precede external visual inspection. Generated
   reports remain `generated_pending_visual_review` until inspection is recorded.
9. Optional RAG/LLM failures do not overwrite or invalidate the reconciled
   deterministic numeric report. They produce a separate, labeled supplement.
10. Logging failures are not silently ignored: if persistent logging cannot be
    written, generation raises rather than claiming fully observable success.

These checks establish internal source consistency, not an independent audit or
proof of every narrative's relevance. The profile's narrative summaries require
human review against the source passages. Matching an anchor alone is not an
entailment check. Forecasting, investment advice and live market valuation are
outside this report's scope.

## Run the Apple example

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m scripts.generate_financial_report `
  --profile configs/financial_reports/apple_2024.json
```

The final line prints the manifest path. The generator uses local PDFs and needs
no model server, API key or network access for the standard report. Required
packages are `pymupdf` and `reportlab`, already used by this repository. Charts are
drawn with ReportLab and rasterized with PyMuPDF; matplotlib and Poppler are not
required by this implementation.

PDF authoring in this session follows the PDF skill's artifact-start marker and
render/inspection workflow. The generator renders all pages automatically; a
person or agent must inspect those images before recording visual approval.

To associate follow-up reports with the same logical conversation/work item:

```powershell
.\.venv\Scripts\python.exe -m scripts.generate_financial_report `
  --profile configs/financial_reports/apple_2024.json `
  --thread-id <existing-thread-uuid>
```

`thread_id` is a logical identifier, not an operating-system thread number. A new
run receives a new `trace_id`, even when the logical thread is reused.

## Optional RAG evidence supplement

```powershell
.\.venv\Scripts\python.exe -m scripts.generate_financial_report `
  --profile configs/financial_reports/apple_2024.json --provider llamacpp
```

This requires a previously ingested financial corpus with matching `ticker` and
`year`, and the appropriate local model configuration. The supplement is saved
as `rag_supplement.json`, separate from the PDF's validated financial tables.
Provider choices are `extractive`, `ollama`, `llamacpp` and `transformers`.

The standard run emits `llm.skipped`: it does not pretend that an LLM was called.
The optional mode logs retrieval filters, ranked sources, model identity, prompt
hash/size, selected source IDs, image count, response hash/size, validation outcome,
retries, cache joins, errors and duration. It does not log API credentials.
Raw prompt/response text is not copied into JSONL. The separate evidence and
supplement artifacts preserve the source/result content needed for review.

Optional RAG uses the existing bounded-call and fail-closed quotation safeguards.
Unverified synthesized answers and unsupported image-only interpretations are
withheld. Neither the report generator nor a successful quotation match claims
to prove every part of a complex question correct.

## Agent stages

These are named orchestration stages, not separate autonomous LLM agents.

| Stage | Work performed | Successful evidence |
|---|---|---|
| `source_validation_agent` | Locate source and verify its checksum | Source path, bytes, SHA-256 |
| `financial_retrieval_agent` | Locate each statement and extract configured rows; validate narrative anchors | Row labels, values, fiscal years, PDF pages, quote hashes and evidence JSON |
| `reconciliation_and_analysis_agent` | Verify statement identities, totals and compute ratios | Named check results and Decimal metrics |
| `rag_supplement_agent` | Skip or retrieve optional risk/liquidity evidence and call configured model | Explicit skipped event or nested retrieval/model events |
| `chart_agent` | Build charts from validated values | PNG/SVG paths, bytes and hashes |
| `report_writer_agent` | Assemble the Markdown and PDF | Final PDF path, bytes and hash |
| `artifact_validation_agent` | Reopen PDF, reject blank/missing pages, render previews | Page count, artifact validation and pending visual-review state |

## Report contents

1. Executive financial review: revenue, gross profit, operating income, GAAP net
   income, diluted EPS and comparative earnings chart.
2. Profitability: cost base, R&D, SG&A, taxes, gross/operating/net margins and
   effective tax rate, with a three-year margin chart.
3. Product and Services mix: three-year category revenues, shares, growth rates
   and a labeled revenue chart.
4. Geographic exposure: reportable-segment revenue, shares, growth rates and chart.
5. Balance sheet: cash, securities, assets, liabilities, equity, debt, current
   ratio, working capital and debt/equity.
6. Cash flow and capital allocation: operating/investing/financing cash, capex,
   dividends, cash buybacks, defined free cash flow and a cash-generation chart.
7. Financial watchpoints and sourced accounting/risk context, including Apple's
   FY2024 State Aid tax charge and the 53-week FY2023 comparison.
8. Methodology, units, formulas, source provenance and thread/trace identifiers.

Unrestricted liquidity is not equated with balance-sheet cash plus marketable
securities. Repurchase authorizations are not equated with cash buybacks. No
adjusted earnings substitute is silently introduced for reported GAAP income.

## Reconciliations and definitions

For each of three income years:

- Revenue - cost of sales = gross profit.
- Gross profit - operating expenses = operating income.
- Pretax income - income-tax provision = net income.
- Product-category revenues sum to consolidated revenue.
- Geographic revenues sum to consolidated revenue.

For each of two balance-sheet dates: assets = liabilities + shareholders' equity.
This produces 17 checks. All must pass.

| Calculated measure | Formula |
|---|---|
| Growth | `(current / prior - 1) * 100` |
| Gross, operating or net margin | Relevant profit / revenue * 100 |
| Effective tax rate | Tax provision / pretax income * 100 |
| Current ratio | Current assets / current liabilities |
| Working capital | Current assets - current liabilities |
| Debt carrying value | Commercial paper + current term debt + non-current term debt |
| Debt / equity | Defined debt carrying value / total equity; leases excluded |
| Cash plus securities | Cash + current and non-current marketable securities |
| Free cash flow | Operating cash flow - absolute cash capital expenditure |
| Cash conversion | Operating cash flow / GAAP net income |

Zero denominators cause failure instead of infinite or invented ratios. Free
cash flow is explicitly labeled a report-defined non-GAAP measure.

## Files and states

```text
output/pdf/AAPL_2024/<trace_id>/
  aapl_fy2024_financial_report.pdf
  aapl_fy2024_financial_report.md
  evidence.json
  analysis.json
  manifest.json
  rag_supplement.json                 # only when requested
  charts/{earnings,margins,products,geography,cashflow}.{png,svg}
  preview/page_01.png ... page_08.png
  logs/<trace_id>.jsonl
```

The manifest lists artifact paths, byte sizes, hashes, company/year, logical
thread ID, trace ID, log path, PDF page count and passed-check count. Each run has
its own directory. Files are written through temporary siblings and atomic
replacement. This is atomic per file, not a distributed multi-file transaction;
the manifest indicates whether the entire run is usable.

States are `started`, `failed`, `generated_pending_visual_review`, and `completed`.
Never publish a run marked `failed` or assume that the mere presence of a PDF
means generation succeeded.

After inspecting **every** preview page:

```powershell
.\.venv\Scripts\python.exe -m scripts.generate_financial_report `
  --record-visual-review <manifest-path>
```

This records an already performed review. It does not perform a visual inspection
itself. It rechecks artifact hashes and the preview count, appends review and
completion events, then marks the manifest `completed`. Run it only after the
generator exits; logs use a single local writer per active run.

## Trace schema and monitoring

Every JSONL event has these common fields:

| Field | Meaning |
|---|---|
| `timestamp_utc` | UTC ISO timestamp; report generation also records IST time |
| `sequence` | Increasing local event number |
| `thread_id` | Logical work/conversation UUID across related reports |
| `trace_id` | One report-run UUID shared by nested calls and output files |
| `span_id` | Stage/call UUID; a `parent_span_id` links child starts |
| `worker_thread_id` | Physical Python thread ID, distinct from logical thread |
| `event` | Structured event name |

Stage events include `span.started`, `span.completed`, and `span.failed`. Call
events include `call.started`, `call.attempt`, `call.retry`, `call.joined` and
`call.completed`. Call completion includes `node`, `status`, `attempts`,
`elapsed_ms` and an error type when present. Status distinguishes successful,
cached, errored and timed-out calls. The idempotency key is hashed in the log.
Context variables are explicitly copied into worker threads; constructing a
nested `Calls` object inherits the active report trace instead of inventing an
unrelated root ID.

Financial events include `retrieval.row`, `retrieval.statement`,
`retrieval.narrative`, `validation.reconciliation` and `analysis.metrics`.
Artifact events include `chart.generated`, `report.file_written`,
`artifact.validated`, `artifact.visual_review_passed` and `report.completed`.
Failures include `span.failed`, `report.failed` and `rag.supplement_failed`.

```powershell
.\.venv\Scripts\python.exe -m scripts.monitor_report_trace --trace-id <run-uuid>
.\.venv\Scripts\python.exe -m scripts.monitor_report_trace --thread-id <logical-thread-uuid>
Get-Content <jsonl-log-path> -Wait
```

The monitor returns statuses, event counts, agent durations, call results,
failures and final artifact metadata. No credentials or logging service are
needed. Existing LangSmith tracing remains optional. Numeric token counts and
model costs are not fabricated when a provider does not return usage data.

## Timeouts, retries and remaining limits

Optional RAG has a 60-second request budget. Existing `RAG_HTTP_TIMEOUT_SECONDS`
controls model HTTP limits. Only explicit transient HTTP 429/503 rejection is
automatically retried once with a stable idempotency key; ambiguous timeouts are
not automatically replayed. Identical in-flight calls join one future and
successful results have a bounded five-minute process-local cache.

Native PDF extraction/rendering and ReportLab numeric-report stages are local
synchronous work, not remotely cancellable agent calls. They do not currently
have a hard-kill process watchdog. Python worker deadlines stop waiting but
cannot forcibly terminate already-running native inference. Filesystem failure,
process termination and disk-full conditions remain possible. Production hard
isolation requires a job worker/process supervisor and durable job store.

Log writes flush and fsync for local durability. JSONL is a local single-writer
trace, not a cross-process transactional event database. The manifest and final
completion events cannot be committed as one filesystem transaction; after an
abrupt crash, compare both and artifact hashes. There is no cross-worker or
cross-restart exactly-once guarantee. A provider may ignore an HTTP idempotency
header. Logs/evidence contain company financial data and should receive suitable
access controls before use with private financial records.

## Configure another company

Copy `configs/financial_reports/apple_2024.json`, then supply the company name,
ticker, fiscal year/end date, currency, source path/link and SHA-256 checksum.
The current standardized layout requires amounts in millions, three descending
consecutive income/cash-flow years and two balance-sheet dates.

Each group defines a unique source-page heading, expected value count, and row
selectors. A selector can be a label or an object with `label` and zero-based
`occurrence` for repeated labels such as current/non-current debt. Keep the
standard internal metric keys; update source labels to match the company's PDF.
Update product and geographic categories to match that filing. Review the
negative-outflow convention before using a different statement layout.

Narrative notes require a reviewed summary, one-based PDF page and anchor text.
Source-specific notes must be replaced rather than copied from Apple. The
generator is a profile-driven annual-statement adapter, not a universal parser
that can infer arbitrary companies, units or PDF layouts without configuration.
Image-only annual reports need a reviewed extraction adapter before this numeric
pipeline can accept them; it does not fabricate numbers from chart appearance.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_financial_report_generation.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

Tests cover real Apple source values, row boundaries, signed outflows, failed
reconciliation, checksum failures, persistent failure logs, root trace propagation
through nested worker calls, PDF contents, chart count and trace monitoring.
Rendered page inspection remains a separate required quality gate.
