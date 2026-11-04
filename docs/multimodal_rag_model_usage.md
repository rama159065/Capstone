# Models used in the local multimodal RAG

This inventory describes the implemented local pipeline, based on the model configuration and source code. The same models serve all six domains; document metadata and retrieval filters provide domain separation. The complete profile enables learned text embeddings, reranking, and Qwen generation. ColPali is enabled separately with `--colpali`; selective OCR is enabled by default.

## Model usage and purpose

| Model / component | Type | Where it is used | Inputs → outputs | Reason for usage | Configuration / status |
|---|---|---|---|---|---|
| `Qwen/Qwen3-VL-2B-Instruct-GGUF` — `Qwen3VL-2B-Instruct-Q4_K_M.gguf` | Generative vision-language model (VLM) | Visual-page captioning during ingestion; answer generation after retrieval | Page images and prompts, or retrieved text with selected page images → captions or answers | Interprets charts, tables, and images and generates answers grounded in retrieved evidence. The quantized 2B checkpoint supports the local CPU profile. The same model handles text questions and visual questions. | Complete profile, served locally through llama.cpp. `RAG_VISION_API_BASE` sets the endpoint; `RAG_VISION_MODEL_ID` records the model identity. |
| `mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf` | Qwen vision projector companion weights | Image processing within the Qwen llama.cpp service | Image representations → representations usable by the language model | Enables the selected Qwen runtime to consume page images. Loaded with the matching Qwen checkpoint. | Launched automatically by `vision_server()` with `--mmproj`; not an independent answer model. |
| `vidore/colpali-v1.3-hf` | Visual retrieval model | Indexing rendered visual pages and encoding text queries for visual search | Page pixels → multiple 128-dimensional page vectors; query text → multiple 128-dimensional query vectors | Retrieves pages using visual content and layout, including tables and charts. Qdrant MaxSim compares query vectors against page vectors; ColPali supplies retrieval evidence rather than generated answers. | Optional `--colpali` / `RAG_COLPALI_MODEL`. Local checkpoint under `indexes/models/colpali-v1.3-hf`. Stored in `visual_pages.colpali`. |
| `sentence-transformers/all-MiniLM-L6-v2` | Text embedding encoder | Text-chunk indexing, query embedding, and visual-summary prefetch | Native PDF text, extracted tables, OCR text, captions, or queries → normalized 384-dimensional vectors | Provides semantic retrieval across wording differences. OCR and table text use the same encoder and collection as native text. Summary vectors provide a candidate set for ColPali search. | Complete profile: `RAG_EMBEDDING_MODEL`. Stored in `docs_text.dense` and `visual_pages.summary` when visual points exist. |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Text relevance reranker | After text/visual candidate fusion, before answer generation | Query–candidate text pairs → relevance scores | Reorders retrieved evidence using joint query/document evaluation. It evaluates candidate text, including OCR and captions, rather than page pixels. | Complete profile: `RAG_RERANKER_MODEL`. Scores are computed at query time, not stored as embedding vectors. |
| RapidOCR bundled detection/recognition ONNX models | OCR models; not a generative LLM or VLM | Selectively on scanned pages, image regions lacking usable text, and supported outlined-text cases | Selected page/region image → recognized text, confidence, and source polygons | Makes image-only words, identifiers, and chart labels searchable through text embeddings. Native pages with adequate text avoid unnecessary OCR. OCR supplies text; visual reasoning remains with the VLM. | Default enabled: `RAG_OCR_ENABLED=1`; confidence and rendering controlled by `RAG_OCR_MIN_CONFIDENCE` and `RAG_OCR_SCALE`. Runs locally through ONNX Runtime. |

There is no separate text-only answer LLM in the complete profile: Qwen3-VL generates both text-only and image-assisted answers.

## Supporting retrieval components

These components participate in the pipeline but are not LLMs or VLMs.

| Component | Usage | Reason / behavior |
|---|---|---|
| BM25-style sparse vectors with Qdrant IDF | `docs_text.bm25`, built from the same text chunks as dense vectors | Complements semantic retrieval with lexical matching, useful for OCR codes, names, and numbers. |
| Reciprocal rank fusion (RRF) | Combines dense/sparse results and merges visual candidates when enabled | Combines ranked retrieval paths before optional cross-encoder reranking. |
| Qdrant MaxSim | Scores ColPali multivectors in `visual_pages` | Matches query-token vectors to page vectors within the summary-prefetched candidate set. |
| Hashing embeddings (`hashing-384-v1`) | Lightweight extractive demo when `RAG_EMBEDDING_MODEL` is unset | Provides deterministic 384-dimensional vectors without learned-model downloads. This is the demo fallback; the complete profile uses MiniLM. |
| PyMuPDF | Native PDF text/table extraction and page rendering | Preserves directly extractable text and tables, and produces images for OCR, ColPali, and Qwen. |

## How the models work together

1. **Ingestion:** PyMuPDF extracts native text and tables. Selective OCR adds otherwise inaccessible text. When configured, Qwen captions visual pages. MiniLM embeds the combined text into `docs_text`, alongside sparse vectors. ColPali separately encodes visual pages into `visual_pages` when enabled.
2. **Retrieval:** MiniLM dense search and sparse search find text evidence. With ColPali enabled, summary prefetch and MaxSim add visual evidence. Fusion combines candidates, and the configured cross-encoder reranks their text.
3. **Answering:** Qwen receives retrieved text and, when available, selected page images to generate an answer. The application renders source references. Extractive demo mode returns evidence excerpts without calling a generative model.
4. **OCR backfill:** Newly recognized text updates text embeddings and existing visual summary vectors. Existing ColPali page matrices are preserved because the underlying page image has not changed.

## Other supported backends

| Backend | Model selection | Implementation status |
|---|---|---|
| Direct Transformers VLM | `RAG_TRANSFORMERS_MODEL` with `provider="transformers"` | Alternative configurable backend; the validated complete profile uses Qwen via llama.cpp. The earlier SmolVLM experiment was inadequate for the validation chart, as recorded in the local guide. |
| Ollama vision backend | An installed vision-capable model selected by `RAG_VISION_MODEL`, with `provider="ollama"` | Alternative backend; the repository does not prescribe a single checkpoint for it. |

## Implementation references

- [Complete-profile model configuration and launcher](../scripts/rag/run_local_multimodal.py)
- [Model loading and generation backends](../scripts/common/local_models.py)
- [Retrieval, ingestion, and answer orchestration](../scripts/rag/multimodal_rag.py)
- [ColPali encoder](../scripts/rag/colpali_retrieval.py)
- [Selective OCR](../scripts/rag/selective_ocr.py)
- [ColPali design and validation](colpali_qdrant.md)
- [OCR design and validation](selective_ocr.md)
- [Complete local profile guide](local_multimodal_rag.md)

This is an inventory of the repository configuration, not a claim that every model is running simultaneously. Models are loaded according to the selected profile, enabled features, and operation.
