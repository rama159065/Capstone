# Complex PDF query audit — 2026-10-30

Historical baseline: the findings below describe the pre-fix implementation.
Safeguards implemented on 2026-10-31 and remaining limitations are documented in
[rag_reliability.md](rag_reliability.md). The diagnostic tests have since been
updated into regression checks for the corrected behavior.

## Conclusion

The RAG supports collecting evidence across PDF pages, and its multimodal answer
path can send several rendered pages to a vision model. It does not currently
ensure a complete or correct answer to a compound question across many pages or
charts. Existing single-source validation results do not establish that capability.

## Verification performed

Commands executed with the workspace virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_multimodal_rag.py tests/test_colpali_qdrant.py tests/test_local_models.py tests/test_selective_ocr.py -q
# 45 passed in 50.52s
.\.venv\Scripts\python.exe -m pytest tests/test_complex_query_audit.py -q
# 5 passed in 1.41s
```

The new tests document current behavior; passing them confirms the limitations,
not successful complex reasoning. No production code was changed.

| Check | Observed result | Scope |
|---|---|---|
| Revenue, costs and target margin on three PDF pages | All three pages and values appear in evidence excerpts | Real generated PDF, ingestion and Qdrant retrieval; extractive answer only |
| Four retrieved visual pages | Only three page images attached | Mocked retrieval and model boundary |
| Long first page followed by another page | Later-page fact absent from local model prompt, but its source ID still appended to answer | Mocked retrieval and llama.cpp response |
| Invented value with a valid retrieved source ID | Citation valid; final status answered | Mocked model response; demonstrates guard limitation |
| Compound question with several subquestions | One retrieval call with the full query | Instrumented retrieval boundary |

There was no new live vision inference or end-to-end answer-accuracy benchmark in
this audit. Chart reading, arithmetic and full subquestion completion therefore
remain unverified for the user's complex-query scenario.

## Findings

1. `LocalRAG.answer` retrieves once for the entire question. Default `top_k=5`
   limits chunks, not distinct pages. Multiple chunks from a single page can crowd
   out evidence for another subquestion. It does not retry missing subquestions.
2. Only the first three unique retrieved image paths are attached. Several charts
   on one page share one rendered image; small labels may require crop or zoom
   support, which this answer path does not provide.
3. llama.cpp and Transformers receive at most 8,000 evidence characters. Each
   retrieved page context is separately capped at 16,000 characters. Truncation
   can remove necessary evidence without explicitly reporting the omission.
4. For llama.cpp and Transformers, all retrieved source IDs are automatically
   appended to non-abstaining answers, including sources whose text was truncated.
   The citation check only checks ID membership; it does not validate claims,
   arithmetic, contradictions or coverage of every subquestion.
5. Extractive mode returns excerpts, not synthesized reasoning or chart analysis.
   The generic domain-agent generation path uses text evidence; use the explicit
   multimodal chat path with a vision provider when image pixels are required.
6. Visual retrieval is conditional on configuration. Without ColPali, chart
   retrieval relies on extracted text, OCR or ingestion-time vision captions.
   ColPali visual retrieval itself first shortlists pages using summary embeddings,
   which can exclude a relevant chart if its summary is poor.
7. Local generation budgets are short: Transformers defaults to 192 new tokens,
   and llama.cpp requests 256. Long multipart answers can be incomplete.

## Recommended implementation work

- Decompose compound questions; retrieve evidence for each part, merge and
  diversify pages, and explicitly track unanswered parts.
- Use budget-aware context construction with omission metadata and page-specific
  image/source mapping. Process visual evidence in batches when it exceeds the
  model's image budget, then synthesize grounded intermediate results.
- Require per-claim citations and verify evidence support and subquestion coverage
  before returning `answered`. Separate citation-ID validity from factual support.
- Use explicit, testable calculations for numeric comparisons and preserve units.
- Benchmark live inference on held-out PDFs with cross-page facts, four or more
  charts, text-plus-chart arithmetic, distracting pages, contradictions and missing
  evidence. Score retrieval recall, each subanswer, numerical accuracy and citation
  support separately before claiming reliable complex-query handling.
