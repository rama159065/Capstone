# Annual reports: what data they contain, and what this project delivers

Date: 2026-11-04. This is a short overview for the capstone team and reviewers. Figures marked **measured** were measured on the project corpus; **target** marks an acceptance goal, not a result. Details are in [financial_rag_final_design.md](financial_rag_final_design.md).

---

## 1. The documents

| Document | Format | What it is | Role in this project |
|---|---|---|---|
| **Annual report to shareholders (Form ARS)** | PDF, on SEC EDGAR since 2023 | The "glossy" report: CEO letter, highlights, charts and photos, usually followed by the full 10-K | Main corpus: page images, charts, narrative |
| **Form 10-K** | HTML with inline XBRL tags | The official annual filing; every financial figure is machine-tagged | Clean text and exact numbers |
| **Proxy statement (DEF 14A)** | HTML with XBRL (`ecd` taxonomy) | Governance, executive pay, pay-versus-performance and shareholder return | Exact figures for performance charts |
| **SEC companyfacts API** | JSON | All tagged figures per company across years | Ground truth and cross-checks |
| AnnualReports.com PDFs | PDF (third-party mirror) | Older years and companies without an ARS on EDGAR (e.g. Apple) | Apple, and the existing report profile |

**Corpus downloaded (measured):** 100 official ARS PDFs from **27 companies**, fiscal years 2022–2026. That is **14,701 pages** (147 per document on average, 76–562), 0.98 GB. 30 AnnualReports.com PDFs are kept for Apple and three others.

---

## 2. What is inside an annual report

| Part | Typical content | Data types |
|---|---|---|
| Front section (letter, highlights) | Strategy, year in review, headline KPIs | Narrative text, highlight charts, infographics, photos |
| Item 1 – Business | Products, segments, customers, employees | Narrative, some tables |
| Item 1A – Risk factors | Risks to the business | Long narrative |
| Item 5 – Market for equity | Share repurchases, dividends, **stock performance graph** | Tables, line chart (total shareholder return) |
| Item 7 – Management's discussion (MD&A) | Why results changed, segment results, liquidity, outlook | Narrative + KPI tables |
| Item 7A – Market risk | Interest-rate, currency, commodity exposure | Tables, narrative |
| Item 8 – Financial statements | Income statement, balance sheet, cash flow, equity | **Structured tables of exact figures** |
| Notes to the statements | Revenue by segment and region, debt, tax, leases, stock compensation | Detailed tables + narrative |
| Governance and shareholder information | Directors, officers, auditor, contacts | Text, photos |

### Measured composition

| Property | Value |
|---|---|
| Text per page | ~4,000 characters (~1,000 tokens) on average |
| Pages containing a table, chart or image | ~29–40%, depending on the detection threshold (59% under the current, too-loose rule) |
| Chart numbers that are real PDF text (exact) | Most chart and table labels: 117 vector pages + 57 images with a text layer, in 25 documents |
| Pixel-only images that contain numbers | 21 of 84 image regions (~0.8 per document); the rest are photos and diagrams |
| **Pages with a garbled text layer** (broken font encoding) | **20% of text pages, in 49 of 100 documents** |
| Tagged XBRL figures per 10-K | 1,200–3,200 (e.g. Chevron 3,146) |
| Official figures findable in the PDF text | 89% overall: 100% for clean PDFs, 0% for garbled ones (Chevron) |
| Same figures in the 10-K HTML | Present, including all of Chevron's |

**What this means:** the PDF is best for **seeing** the report (layout, charts, photos), and the 10-K HTML/XBRL is best for **reading exact numbers**. The project uses both.

---

## 3. How each kind of data is captured

| Data type | How it is captured | Where it is stored |
|---|---|---|
| Financial figures | XBRL facts from the 10-K (and proxy) | Facts database: exact values with period, unit and source |
| Narrative text | 10-K sections; ARS front pages (OCR where the text layer is garbled) | Qdrant `docs_text`: dense + number-aware keyword search |
| Tables | 10-K HTML tables with row/column headers and units | Qdrant `docs_text` + facts database |
| Charts and infographics | ColPali page embeddings (full vectors + row/column pooled) | Qdrant `visual_pages` |
| Numbers printed in charts | PDF text labels; validated OCR; reviewer approval (tiers T1–T6) | Facts database (`chart_value`) |
| Photos and diagrams | ColPali embeddings; page rendered on demand | Qdrant `visual_pages` + PDF |
| Provenance | Accession number, URL, SHA-256, page, bounding box | Manifest + payload of every record |

---

## 4. What the project delivers

### 4.1 Capabilities

| Outcome | What the user gets |
|---|---|
| **Financial Q&A (chat)** | Questions in plain English, e.g. "What was Chevron's net income in FY2025?", "Why did Nvidia's data-center revenue grow?", "Show the shareholder-return chart". Answers cite the filing, page and source, and follow-up questions keep their context |
| **Complete company report (job)** | "Generate a complete financial report for Apple" produces a PDF/Markdown report with financial statements, ratios, trends, charts, segment analysis, risks and commentary. **Every number traces to a filing figure**, with 17 arithmetic reconciliations in the existing pipeline |
| **Honest handling of missing data** | The report says "not disclosed (searched: …)", or shows a derived value with its formula, or uses a figure from another year's filing, clearly labelled. It never guesses |
| **Multimodal search** | Finds pages by meaning, keywords, exact figures, or how a chart looks |
| **Human-in-the-loop** | Approval steps before publishing a report; a review screen for uncertain chart readings; checks for financial-advice language and citations |
| **Resumable, auditable runs** | Checkpoints after every step: resume after a failure, rerun one section, keep versioned outputs, full trace |
| **Interfaces** | FastAPI backend + Streamlit UI; demo on a Colab runtime or an always-on VM |

### 4.2 Quality and speed

| Measure | Value |
|---|---|
| Exact-figure accuracy on XBRL-based questions | **target ≥ 98%**; 100% when the facts tool returns the figure |
| Chart values used automatically | **target ≥ 99.5%** precision |
| Visual search quality vs full ColPali | Recall@20 0.917 / NDCG@20 0.952 (published by Qdrant); **target ≥ 0.90** on our pages |
| Invented numbers | **target 0**: every number must match a fact, a cited passage or a stated formula |
| Q&A response time | ~4–12 s (text), ~7–15 s (visual, CPU server); retrieval itself ~0.1–0.3 s |
| Report generation | ~2–7 min of machine time per company, plus human approvals |
| Scale | Designed for 500–1,000 documents: Qdrant ~9–19 GB, ~1–2 GB of it in RAM |
| Cost | ~$10–15 one-off for Colab compute; $2/month for Drive; serving ~$0.05–0.39 per demo on Colab, or ~$132/month for an always-on VM |

### 4.3 Project artefacts so far

| Artefact | Status |
|---|---|
| 100 EDGAR ARS PDFs + manifest ([download_edgar_ars_pdfs.py](../scripts/downloads/download_edgar_ars_pdfs.py)) | Done (PDFs git-ignored, manifest kept) |
| Design documents: [final design](financial_rag_final_design.md), [implementation recommendations](implementation_recommendations.md), [ColPali pooling analysis](colpali_pooling_analysis.md), [agentic report design](financial_report_agentic_ai_recommendations.md), [summary](financial_copilot_summary.md) | Done |
| Colab evaluation script for pooling and retrieval speed ([colab_colpali_pooling_eval.py](../scripts/evaluation/colab_colpali_pooling_eval.py)) | Ready; GPU results pending |
| EDGAR corpus download: 100 ARS PDFs + matching 10-K iXBRL HTML + DEF 14A proxies + companyfacts/submissions, linked by `filing_id` ([download_edgar_corpus.py](../scripts/downloads/download_edgar_corpus.py)) | Done: 100/100 filings, 1.8 GB, 6.5 min |
| PDF → Qdrant ingestion (`docs_text` + pooled/full `visual_pages`), resumable with Drive snapshots ([financial_ingest_qdrant_db.py](../scripts/rag/financial_ingest_qdrant_db.py), [Colab notebook](../notebooks/colab_financial_ingest.ipynb), [guide](financial_ingestion_colab.md)) | Done; tested offline and on real filings (text stage). First Colab server/GPU run pending |
| Deterministic Apple FY2024 report generator (existing) | Working |
| 10-K/XBRL parser, facts DB, chart-value extractor, Qdrant schema changes, agent flows with checkpoints, report API | To implement ([order of work](implementation_recommendations.md)) |

---

## 5. Limits

- **The LLM is the provided hosted ≤8B model** (Track B rule). Report prose is kept short and structured, and all numbers are filled in by code.
- **Values that no document prints** (for example unlabelled chart segments) are described only as trends; they are never estimated.
- **ARS PDFs on EDGAR start in 2023,** and some companies (e.g. Apple) file none. Older years come from 10-K HTML or the AnnualReports.com mirror.
- **Colab is used for ingest and demos.** An always-on service needs a VM.
- **Estimates in the design** (T4 encoding speed, end-to-end latency) are confirmed only once the Colab evaluation runs.
