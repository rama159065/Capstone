# Notebooks

Five step-by-step notebooks build the financial-report copilot up layer by layer. Each one imports the repo's code (`scripts/...`) instead of copying it. Each runs on **Google Colab** with a few filings, or locally from the repo root. A sixth notebook, `colab_financial_ingest.ipynb`, ingests the full corpus and serves the API + UI.

| # | Notebook | What it tests | Filings | Runtime |
|---|---|---|---|---|
| 01 | [01_chat_templates_pydantic.ipynb](01_chat_templates_pydantic.ipynb) | Zero-shot, few-shot and chain-of-thought templates; LLM-chosen **tool call** (Pydantic `ToolCall`) to **SEC XBRL** as the external source; answers validated as Pydantic `FinancialAnswer`, plus citation and number grounding | none (XBRL API) | CPU |
| 02 | [02_text_rag_chat_templates.ipynb](02_text_rag_chat_templates.ipynb) | Text-only RAG (Qdrant `docs_text`, hybrid + rerank) + templates. Scenarios: filtered, unfiltered, `k=2`, a garbled filing | AMD FY2025, AbbVie FY2025, NVIDIA FY2026, Chevron FY2022 | CPU or GPU |
| 03 | [03_ocr_rag.ipynb](03_ocr_rag.ipynb) | Full-page OCR of garbled statements checked against XBRL; image-region OCR of a chart (2 passes); QA with no OCR, front-first OCR and full OCR | Chevron FY2022 | GPU recommended |
| 04 | [04_visual_rag_colpali.ipynb](04_visual_rag_colpali.ipynb) | Charts and tables as **ColPali image multivectors** in Qdrant (full vectors, exact MaxSim); chart questions, descriptive and exact-number queries; optional vision model | NVIDIA FY2026, Chevron FY2022 | **GPU** |
| 05 | [05_final_qdrant_colpali_rag.ipynb](05_final_qdrant_colpali_rag.ipynb) | Final design: text + OCR + ColPali **mean pooling** (rows/columns) with full-vector rerank. Pooled-only vs prefetch → rerank vs exhaustive; hybrid evidence; **clarifier → LangGraph agent** on the eval set | the four above | **GPU** |

## Running on Colab

1. **Open a notebook:** File → Open notebook → GitHub. Use repo `rama159065/Capstone`, branch `finance_report_qa_agent`.
2. **Choose a runtime:** set it to **GPU** for notebooks 03–05.
3. **Add Colab secrets** (🔑 panel; turn on notebook access):

   | Secret | When needed |
   |---|---|
   | `GITHUB_TOKEN` | while the repo is private (read-only "Contents" permission is enough) |
   | `SEC_USER_AGENT` | always: `your-name your.email@domain`. SEC requires a contact address |
   | `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL` | for the provided Track B endpoint. Without them, notebooks 01–04 use a small Hugging Face stand-in on the GPU and notebook 05's agent uses the mock |

4. **Run all cells.** Cell 0 clones the branch, installs what that notebook needs, and loads the secrets. PDFs are downloaded from EDGAR and SHA-256-checked.

**Settings** (set them before the config cell with `os.environ[...] = ...`):

| Variable | Values |
|---|---|
| `NB_LLM` | `auto` (default), `hosted`, `transformers`, `mock` |
| `NB_EMBED` | `local` (MiniLM, the default) or `hashing`; `NB_EMBED_MODEL` sets the model |
| `NB_COLPALI` | `auto` (real on a GPU), `real`, or `fake` (plumbing only) |
| `NB_QUESTIONS`, `NB_MAX_PAGES`, `NB_MAX_VISUAL_PAGES` | smaller runs |
| `NB_OCR_PAGES`, `NB_OCR_LIMIT`, `NB_OCR_SCENARIOS`, `NB_OCR_CUDA=1` | OCR scope; GPU OCR needs `onnxruntime-gpu` |
| `NB_VLM=1` | notebook 04's optional vision-model reading |

**Running locally:** from the repo root use the project venv, which needs a Jupyter kernel installed. The notebooks contain no shell `!` commands, so every cell is plain Python.

## Local verification (CPU)

All five notebooks were executed end to end with:
- the mock LLM;
- MiniLM embeddings from `indexes/models`;
- real OCR;
- the fake ColPali encoder, plus a separate real-ColPali check.

These runs show that the pipeline works. Model-quality numbers need the hosted endpoint.

| Notebook | Measured |
|---|---|
| 01 | The XBRL tool returned exact figures for AMD, Chevron and AbbVie. Pydantic flagged unusable tool calls (`tool_call_error`, with a slot-parser fallback) and the mock's ungrounded values and citations |
| 02 | The right filing was retrieved for **100%** of filtered questions (78% unfiltered). But the gold figure was in the top-5 evidence for only **17%** (11% at `k=2`): question-style queries prefer narrative pages over statement tables, which is why the design puts a facts tool first. Chevron FY2022: 0% |
| 03 | Garbled text: 0 of 6 statement figures. OCR of 12 pages: 5 of 6 at 4.5 s/page. Chart OCR: 4 of 4 values plus the false "12,000". Gold value in evidence: 20% → 40% → 50% (none → front-first 20 → all). Chart values are never indexed at ingest (a gap) |
| 04 | 127 visual pages encoded into a multivector collection; queries and thumbnails work (fake rankings) |
| 05 | Full ingest (4 filings, text + visual). Pooled-only search ~23 ms vs ~220–330 ms with full MaxSim (plumbing timing). Clarifier → agent; 17 of 32 answered with valid citations. The runs found three clarifier bugs (now fixed), an input-guard gap (fixed) and a narrow classifier fallback (documented) |
| Real ColPali (CPU, bfloat16) | Each page gives 1030 vectors = 1024 patches + 6 special tokens; pooled to 38 × 128. NVIDIA p60, Chevron p35 and AMD p65 were ranked first by both full MaxSim and pooled rows. CPU speed is ~73 s/page, so use a GPU |
