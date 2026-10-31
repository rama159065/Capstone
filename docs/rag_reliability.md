# RAG reliability safeguards

Implemented on 2026-10-31. These safeguards deliberately favor abstention and
source excerpts over unverified generated answers.

## Behavior

| Requirement | Implementation |
|---|---|
| Nothing useful returned | Empty, malformed, placeholder-only and low-reranker-score results do not trigger generation. Missing evidence produces an explicit no-answer response. |
| Grounded model calls | Whole evidence records are budgeted before prompting. Model output must be structured selections with exact source IDs and verbatim quotations. Invalid output is withheld. Model-authored captions help retrieval but cannot verify facts. |
| Tool errors | Failures become sanitized errors or explicit warnings. Optional visual retrieval/reranking can fall back to text/RRF. Caption failure preserves native-text/OCR ingestion. |
| Idempotent retries | Calls eligible for retry require a key. Identical in-flight calls join one future; completed successful calls are cached for five minutes, up to 128 entries. Prompt, evidence, provider and image content affect the key. HTTP model requests carry a stable Idempotency-Key header. |
| Timeouts | Overall request deadlines, bounded graph-node waits, HTTP timeouts, lock wait limits, and four bounded worker slots prevent unbounded caller waits and queues. |
| Tracing/evaluation | Responses expose a trace ID and node status, duration and attempt count. Existing optional LangSmith integration wraps tool/model calls. Offline fault injection and live inference scoring remain separate. |
| Graceful degradation | Available excerpts remain accessible after model failure, with warnings and evidence-only/partial status. Excerpts are never promoted to a verified synthesized answer. |

Multimodal questions separated by `?` or `;` are retrieved independently, up to
eight parts, and merged by chunk ID and page. This is deterministic splitting,
not a semantic decomposition model. A compound question with conjunctions may
still need a better decomposition stage.

## API

POST `/multimodal/financial_report_qa/chat`:

```json
{
  "query": "What is revenue? What are costs?",
  "provider": "llamacpp",
  "top_k": 5,
  "idempotency_key": "report-question-001",
  "timeout_seconds": 60
}
```

New response statuses include `evidence_found`, `evidence_only`,
`partial_evidence`, `insufficient_evidence` and `tool_error`. Clients must not
treat `citation_check.valid` as an answer-correctness score. Its scope is source
membership. `grounding.semantic_answer_verified` remains false: quotation
matching does not prove relevance, completeness, OCR accuracy or arithmetic.

Generic domain agents enforce bounded nodes and verify hosted answer selections.
Legacy financial prompting experiments return labeled source excerpts for hosted
outputs, withholding their unverified free-form generation. Mock providers remain
test fixtures and are explicitly labeled in grounding metadata.

The new ingestion pipeline version separates native/table/OCR grounding text
from generated captions. Re-run the usual ingestion command for existing PDFs
to populate that field; pipeline version changes trigger reingestion. Old
captioned pages without separate grounding text abstain rather than treating a
caption as proof. Ingestion is not automatically started by this code change.

## Configuration

- `RAG_REQUEST_TIMEOUT_SECONDS`: generic/financial workflow deadline, default 60.
- `RAG_HTTP_TIMEOUT_SECONDS`: model/remote Qdrant timeout, default 45.
- Multimodal API `timeout_seconds`: request budget, greater than zero and at most
  600 seconds. Default 60.
- Existing `RAG_MIN_RERANK_SCORE` controls rejection when reranking succeeds.
- Existing LangSmith configuration remains opt-in and needs credentials. Local
  trace fields do not contain raw prompts, keys, service errors or document text.
  Enabling LangSmith may transmit tool inputs/outputs under its normal behavior.

Only explicit HTTP 429/503 rejection is automatically retried, once, respecting a
bounded Retry-After delay and the remaining deadline. Ambiguous timeouts and
connection failures are not retried automatically. Invalid model output is not
retried as if it were a transport error.

## Evaluation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m scripts.evaluate_rag_resilience
```

The second command writes `output/resilience_evaluation/summary.json` and
`junit.xml`. It exercises real cross-page PDF ingestion/retrieval plus mocked
failure injection, duplicate requests, timeouts, retry classification, malformed
results, withheld hallucinations and caption provenance. It does not measure
live-model answer accuracy.

Validation completed: 158 tests passed in the full suite; the dedicated offline
resilience evaluation passed all 25 cases. The final financial-workflow adjustment
was additionally checked with its 13-test module.

The existing `scripts.evaluate_multimodal_local` live evaluation now requires the
appropriate result status as well as expected answer content. A fallback excerpt
containing the expected number no longer counts as an accurate generated answer.
No new live-model benchmark was run for this implementation. Earlier validation
reports describe the earlier generation behavior, not this stricter contract.

## Limits

- Deduplication is process-local and expires. Cross-worker/restart durability
  requires a shared persistent store. Sending a header does not guarantee that
  a third-party model server implements idempotency.
- Python threads cannot forcibly stop an already running native inference or
  blocking library call. The caller stops waiting at its deadline; the bounded
  slot remains occupied until that work exits. There is no timeout retry that
  silently starts a duplicate. Hard cancellation requires process isolation.
- A maximum of three page images and 24,000 evidence characters are admitted in
  multimodal generation. Omissions are explicit warnings; automatic image
  batching, crop/zoom, chart-value verification and deterministic calculations
  are not implemented. Image-only claims without verifiable extracted text
  abstain. This trades answer coverage for stronger source grounding.
- Accurate extraction can still contain irrelevant evidence. Source quotations
  are presented as excerpts; there is no claim of eliminating every form of
  hallucination or proving an entire multipart answer correct.
