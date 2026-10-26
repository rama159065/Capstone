# Local multimodal RAG

For the consolidated architecture, implementation, schemas, API contract, configuration, and verified results, see [Multimodal RAG: Design, Implementation, and Validation](multimodal_rag_design_and_implementation.md).

ColPali page multivectors can now be written to the same Qdrant database. See [ColPali implementation and Qdrant storage](colpali_qdrant.md) for checkpoint preparation, backfill, MaxSim queries, and `serve --colpali`.

Selective CPU OCR is enabled for scanned pages and uncovered image regions. OCR text uses the same `docs_text` embeddings as native text. Run `python -m scripts.ocr_index` to enrich the existing index without recomputing ColPali matrices; see [selective OCR documentation](selective_ocr.md).

This implements section 7 (the lighter alternative) of `qdrant_multimodal_rag_architecture.md` for all six application domains. Original PDFs and rendered visual pages live on the filesystem; SQLite records ingestion status and hashes. One shared Qdrant `docs_text` collection stores domain-filtered dense and sparse vectors. The Query API fuses dense and BM25-style sparse candidates with RRF and expands the matching page's context.

## Complete local CPU profile

The complete profile uses learned MiniLM text embeddings, BM25-style sparse retrieval, RRF fusion, an MS MARCO cross-encoder reranker, Qwen3-VL-2B-Instruct Q4_K_M, and a Q8 image projector served by llama.cpp. Visual pages are captioned by the actual VLM during ingestion and attached as images at answer time. Text-only pages avoid visual inference during ingestion. All question answering uses the local model.

```powershell
# CPU model runtime; install once in this project's virtual environment:
uv pip install --python .venv/Scripts/python.exe torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe "transformers>=4.57,<5" "sentence-transformers>=3,<6" "accelerate>=1,<2"
# Downloads model weights and the pinned Windows CPU runtime into indexes/:
.venv/Scripts/python.exe -m scripts.run_local_multimodal prepare
# Ingests three independent PDFs, then runs actual inference through FastAPI:
.venv/Scripts/python.exe -m scripts.run_local_multimodal evaluate
# Starts both the local model service and application API:
.venv/Scripts/python.exe -m scripts.run_local_multimodal serve
```

The validation fixtures contain operations text, a regional-budget table, and an **image-only** revenue chart with no PDF text layer. Nine checks cover text lookup, table lookup and subtraction, chart trend/category/value, and an unsupported question. Checks cover source retrieval, answer values, citations, and actual image attachment. Results and generated answers are saved to `output/multimodal_validation/results.json`, `output/multimodal_validation/report.md`, and `docs/multimodal_validation_results.md`. Model outputs are not mocked. Downloads require internet once; the prepared profile runs offline on CPU. The first model load and image encoding can take time on a busy machine.

`serve` uses `llamacpp` as the default multimodal answer provider. Both services bind only to localhost; the launcher starts its own model service on port 8081 and stops it when the command exits. Set `RAG_VISION_API_BASE` to another localhost port if needed. Do not run two launchers simultaneously against the same embedded Qdrant directory.

```powershell
$body = @{query="What is the North regional budget in million USD?"; top_k=1} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/multimodal/financial_report_qa/chat -Method Post -ContentType application/json -Body $body
$body = @{query="Which quarter has the highest revenue in the chart?"; top_k=1} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/multimodal/financial_report_qa/chat -Method Post -ContentType application/json -Body $body
```

The complete profile uses `indexes/multimodal_full`, separate from the hashing demo's `indexes/multimodal`. Embedding signatures prevent accidental mixing of incompatible models. To ingest the original six domain samples with the full model pipeline, run `python -m scripts.run_local_multimodal ingest`; use `--input` to select another domain-folder corpus. `python -m scripts.run_local_multimodal query --domain financial_report_qa --query "What is the annual review code?"` starts the model service, answers once, then shuts it down.

An alternative direct-Python Transformers vision backend is available with `RAG_TRANSFORMERS_MODEL` and `provider="transformers"`. The tested complete profile uses Qwen via llama.cpp; the smaller SmolVLM model was inadequate for the validation chart and is not the recommended profile.

## Run locally (PowerShell)

```powershell
uv pip install --python .venv/Scripts/python.exe -e .
# Six synthetic four-page PDFs are included. Regenerate them if needed:
.venv/Scripts/python.exe -m scripts.generate_multimodal_samples
.venv/Scripts/python.exe -m scripts.domain_common.multimodal_rag ingest
.venv/Scripts/python.exe -m scripts.domain_common.multimodal_rag query --domain hr_policy --query "What does the training table show?"
.venv/Scripts/python.exe -m uvicorn scripts.api_app:app --host 127.0.0.1 --port 8000
```

No Docker, keys, or model downloads are required for the extractive demo. Default storage is `indexes/multimodal`, overridable with `RAG_LOCAL_DIR`. Qdrant embedded storage supports one process; stop the API before running ingestion and use one API worker. Set `QDRANT_URL=http://localhost:6333` to use a separately running Qdrant server instead. All processes must share the same file storage directory.

```powershell
$body = @{query="What does the training table show?"; provider="extractive"} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/multimodal/hr_policy/chat -Method Post -ContentType application/json -Body $body
Invoke-RestMethod http://127.0.0.1:8000/multimodal/hr_policy/retrieve -Method Post -ContentType application/json -Body $body
```

Replace `hr_policy` with `financial_report_qa`, `procurement_vendor`, `travel_expense`, `competitive_research`, or `incident_postmortem`. API responses include document/page citations, content type, table Markdown, original PDF URI, and rendered page image paths. The new API reads files already ingested locally; it does not accept arbitrary server filesystem paths.

## Local vision model

Install and start Ollama, then install a vision-capable model that fits your machine. Set its exact installed name:

```powershell
$env:RAG_VISION_MODEL="YOUR_INSTALLED_VISION_MODEL"
# Optional: $env:OLLAMA_HOST="http://127.0.0.1:11434"
.venv/Scripts/python.exe -m scripts.domain_common.multimodal_rag ingest --force
.venv/Scripts/python.exe -m scripts.domain_common.multimodal_rag query --domain hr_policy --provider ollama --query "Describe the review process diagram"
```

Set the same environment variable before starting the API and use `provider="ollama"`. Ingestion asks the model to caption visual pages, including chart values and trends; answering sends retrieved text plus up to three unique page images to the model. Model/service errors are surfaced rather than silently replaced with invented captions. Without a vision model, retrieval still uses native text, extracted tables, and selective OCR text; extractive mode returns labelled evidence excerpts. OCR can read scanned text and chart labels, while visual relationships and image reasoning still require the vision path.

## Existing app integration

Set `$env:RAG_BACKEND="qdrant"` before starting the existing API, agents, or Streamlit apps to route their existing retrieval functions through this store for every domain. Existing guarded agent chat stays on its configured text LLM. Use `/multimodal/{domain}/chat` for image-aware answers. Without the variable, existing Chroma behavior remains the default. Existing embedding-provider and retrieval-mode request settings apply to Chroma; the Qdrant local demo uses its fixed 384-dimensional hashing embedding and hybrid RRF retrieval.

## Your PDFs and incremental updates

Organize PDFs as `<input>/<domain>/*.pdf`, then run:

```powershell
.venv/Scripts/python.exe -m scripts.domain_common.multimodal_rag ingest --input C:/path/to/pdfs
```

Use `--domain` to ingest a single domain and `--force` to recaption/reindex unchanged files. Unchanged hashes are skipped; changed files at the same path replace their earlier vectors after successful ingestion. Failures record a retryable status and clear partial vectors. Domain and hash are part of deterministic point IDs. Python callers may pass `source_type`, `ticker`, and `year` to `LocalRAG.ingest`; filters match those metadata values exactly. Samples use `source_type=sample_pdf` and no ticker/year.

## Scope and limitations

The samples contain text, a ruled table, labelled vector bar/line charts, and a raster process diagram. All data is fictional. PyMuPDF routes pages using tables, raster images, and drawing counts; this heuristic can miss sparse vector diagrams or classify decorative pages as visual. Image-only pages are always rendered.

The lightweight demo uses hashing embeddings; the complete profile uses learned embeddings and cross-encoder reranking. Sparse vectors use BM25 saturation with a fixed reference document length plus Qdrant corpus IDF. This implementation follows the architecture's summary-based visual retrieval alternative; ColPali multivectors are available when enabled, while S3/MinIO and production-scale workers remain outside this local small-corpus profile. Scanned/image-only content is interpreted by the VLM. Captions are generated from actual images, not synthetic sidecars. Page expansion is capped at 100 chunks/16,000 characters. The application renders source identifiers for locally generated answers; identifier validation does not establish factual entailment. The inference evaluation independently checks expected answer values.

Validation: `.venv/Scripts/python.exe -m pytest tests/test_multimodal_rag.py tests/test_api_app.py`.
