# Local multimodal RAG

This implements section 7 (the lighter alternative) of `qdrant_multimodal_rag_architecture.md` for all six application domains. Original PDFs and rendered visual pages live on the filesystem; SQLite records ingestion status and hashes. One shared Qdrant `docs_text` collection stores domain-filtered dense and sparse vectors. The Query API fuses dense and BM25-style sparse candidates with RRF and expands the matching page's context.

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

Set the same environment variable before starting the API and use `provider="ollama"`. Ingestion asks the model to caption visual pages, including chart values and trends; answering sends retrieved text plus up to three unique page images to the model. Model/service errors are surfaced rather than silently replaced with invented captions. Without this model, retrieval uses extracted text and tables; extractive mode returns labelled evidence excerpts. It cannot interpret image-only charts or scanned text.

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

Hashing embeddings are a reproducible offline baseline, not learned semantic embeddings. Sparse vectors use BM25 saturation with a fixed reference document length plus Qdrant corpus IDF. This implementation uses RRF ranking, with no cross-encoder or ColPali. It intentionally does not implement the optional `visual_pages` multivector tier, S3/MinIO, OCR, or production-scale ingestion workers. Vision captions require the configured local model; they are not precomputed from synthetic sidecars. Page expansion is capped at 100 chunks/16,000 characters. Citation validation checks identifiers, not factual entailment. Review generated answers against their evidence.

Validation: `.venv/Scripts/python.exe -m pytest tests/test_multimodal_rag.py tests/test_api_app.py`.
