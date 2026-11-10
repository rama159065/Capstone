# FastAPI Streaming Design for Guarded Agentic RAG

## 1. Purpose and status

This document defines a proposed end-to-end design for streaming responses from the FastAPI Agentic RAG service. It covers the request lifecycle, event contract, model and graph integration, security, grounding, guardrails, human review, failure handling, observability, deployment, and rollout.

**This is a design, not an implementation.** The current chat routes return a completed JSON response. The current hosted LLM clients use blocking HTTP calls, LangGraph workflows use `.invoke()`, and the current human-review routing produces review information but does not provide a durable interactive approval workflow. Those are implementation prerequisites described below.

## 2. Goals and non-goals

### Goals

- Give clients timely, observable progress through a long-running retrieval and agent workflow.
- Preserve the current policy that an answer is not released unless it passes evidence, citation, and output checks.
- Keep refusal and human-review outcomes explicit.
- Correlate client events, API logs, and LangSmith traces with one request ID.
- Provide bounded resource use, cancellation, and predictable error semantics.

### Non-goals for the first streaming release

- Do not stream unvalidated answer tokens or hidden model reasoning.
- Do not use an open WebSocket when one-way server events are sufficient.
- Do not convert bulk evaluation, corpus indexing, or large batch inference into long-lived chat streams.
- Do not treat a citation-ID match as proof that every claim is entailed by its cited source.
- Do not claim a human approval has occurred merely because a review request was queued.

## 3. Recommended interaction model

Use **Server-Sent Events (SSE)** for interactive progress and a final validated result. SSE is a one-way HTTP stream from server to client, works with standard HTTP infrastructure, and allows the server to emit typed events over a single request.

Retain the existing JSON endpoints for compatibility:

- `POST /chat/financial_report_qa` — current request/response behavior.
- `POST /chat/financial_report_qa` — current financial request/response behavior.

Add separate opt-in streaming routes:

- `POST /chat/financial_report_qa/stream`
- `POST /chat/financial_report_qa/stream`

The request body should reuse the existing Pydantic chat models. A streaming request should not silently change the response shape of the existing routes.

Use **SSE progress events first**, followed by a final answer event only after the full workflow has finished validation. Do not begin token-by-token answer streaming in the first release. If future UX requirements require token streaming, first design a safe buffering or claim-validation strategy; tokens already delivered cannot be recalled if a later guardrail blocks the completed answer.

### When to use a job API instead

Use a separate asynchronous job model for bulk inference, long evaluations, corpus indexing, or work that should survive client disconnection:

1. `POST /jobs` returns `202 Accepted` and a job ID.
2. `GET /jobs/{job_id}` returns durable state.
3. `GET /jobs/{job_id}/events` streams job progress, or clients poll the status endpoint.
4. `POST /jobs/{job_id}/cancel` requests cancellation.

Interactive chat should not become a background job unless there is a concrete need for restart recovery, long-lived review, or reconnect-after-disconnect.

## 4. Current-state facts and required changes

The current service has the right logical stages to expose as progress:

- deterministic security checks and plain-text/domain classification;
- planner and retrieval;
- evidence prompt-injection screening;
- answer generation;
- output filtering and citation validation;
- governance/HITL routing;
- final response grounding policy at the API boundary.

However, the current model clients and graph calls are synchronous. Merely declaring a route `async` or wrapping a synchronous generator in `StreamingResponse` does not make blocking inference non-blocking.

Before production streaming:

1. Add asynchronous chat and embedding clients using an async HTTP client, explicit connection/read timeouts, response-size limits, and cancellation propagation.
2. Add async graph execution and event consumption using LangGraph streaming APIs (`astream` or `astream_events`) where supported.
3. Preserve the existing sync routes and CLI behavior, or migrate them separately after compatibility tests.
4. Ensure tracing context and request IDs propagate across async tasks and graph callbacks.
5. Introduce durable storage/checkpointing before promising review persistence, reconnect/replay, or human approval workflows.

As a temporary bridge, the synchronous graph could be run in a worker thread while emitting only progress that is known from application-owned stage callbacks. This is less efficient, has limited cancellation semantics, and must have a bounded thread pool. It is not the target architecture.

## 5. End-to-end architecture

```text
Client
  | POST JSON + Accept: text/event-stream
  v
FastAPI streaming route
  | authenticate/authorize, validate body, rate-limit, assign request_id
  | emit SSE headers and connected event
  v
Agent orchestration service
  |-- input security and intent checks
  |-- planner
  |-- retriever (dense + BM25 / domain-specific filters)
  |-- retrieved-evidence safety guard
  |-- answer model (answer stays server-side)
  |-- output safety guard
  |-- citation/evidence consistency validation
  |-- semantic support verifier (recommended before strong grounding claims)
  |-- governance and HITL decision
  v
Finalize response policy
  | grounded answer, refusal, or withheld/review result
  | emit final SSE event; close stream
  v
Client renders final result and citations
```

LangSmith is an optional observability path alongside orchestration; it is not a decision authority and must not be required for request success.

### Detailed sequence

1. **Connection and validation**
   - Client sends JSON and `Accept: text/event-stream`.
   - Validate the request with Pydantic before starting a stream.
   - Authenticate and authorize the caller (if enabled); enforce body size, per-user/IP rate limits, concurrency quotas, and request deadline.
   - Generate or validate the request UUID. Return it as `X-Request-ID` and include it in every event and trace.
   - If route, authentication, authorization, rate limit, or request validation fails before streaming starts, return the normal HTTP error response.

2. **Guard input**
   - Run deterministic input checks before sending the query to a model.
   - If blocked, emit a refusal outcome without emitting the query or matched sensitive text.
   - Run domain-intent classification when enabled. Treat classifier failure as an explicit failure or conservative refusal according to configured policy; do not silently assume the query is safe.

3. **Plan and retrieve**
   - Emit stage-only progress events such as `planning` and `retrieving`.
   - Execute bounded planner output through an allowlisted tool set; never execute arbitrary model-proposed tools or code.
   - Retrieve top-k evidence with domain/source filters.
   - Emit an evidence summary containing counts and safe source metadata, not full chunk text by default.

4. **Inspect evidence**
   - Run the evidence prompt-injection guard on every retrieved item before it is supplied to generation.
   - If unsafe evidence is detected, refuse or route to review according to policy; do not continue to answer generation with the flagged content.
   - Apply context-size, source allowlist, and metadata validation limits.

5. **Generate privately**
   - Generate the answer on the server. Do not send answer tokens yet.
   - Keep model output in bounded memory and apply a generation timeout / token limit.
   - Treat model output and tool output as untrusted input.

6. **Validate before release**
   - Apply output-content filtering.
   - Validate citation IDs against the actual retrieved chunk IDs. Require at least one cited ID where the answer asserts evidence-based facts.
   - If available, run claim-level support checks against the cited passages and reject unsupported claims.
   - Apply the API grounding policy. If validation fails, clear the answer before constructing any client event.
   - Run governance/HITL routing. If review is required, return a review-pending outcome rather than presenting the draft as an approved answer.

7. **Finalize**
   - Emit exactly one terminal event (`completed`, `refused`, `review_required`, `not_grounded`, `error`, or `cancelled`).
   - The final event contains the final answer only if release policy passed.
   - Close the stream. A disconnected client does not imply the workflow completed successfully.

## 6. SSE protocol and event contract

Use `Content-Type: text/event-stream; charset=utf-8`, `Cache-Control: no-cache, no-transform`, and `X-Accel-Buffering: no` where the deployment proxy supports it. Keep-alive comments can be emitted periodically, for example `: keep-alive\n\n`; these are transport heartbeats and are not application events.

Each application event uses standard SSE framing:

```text
id: 12345678-1234-5678-1234-567812345678:7
event: progress
data: {"request_id":"12345678-1234-5678-1234-567812345678","sequence":7,"stage":"retrieving","status":"started"}

```

Every JSON `data` payload should include:

| Field | Meaning |
|---|---|
| `request_id` | Correlates response, logs, and LangSmith trace. |
| `sequence` | Monotonically increasing event number for this request. |
| `timestamp` | UTC event time. |
| `stage` | Current workflow stage, when applicable. |
| `status` | Stage/event status, when applicable. |

### Event types

| Event | When sent | Safe payload |
|---|---|---|
| `connected` | Stream established | Request ID, protocol version, supported event features. |
| `progress` | A workflow stage starts or ends | Stage name, status, elapsed time, safe count/summary values. |
| `evidence_summary` | Retrieval completed | Number of selected chunks and source IDs/metadata permitted for the user. Avoid raw chunk text by default. |
| `guardrail` | Guard result is available | Check name, `allow`/`review`/`block`, sanitized reason codes. Never echo matched secrets or full prompt text. |
| `human_review_required` | Governance requests review | Review ID, trigger codes, status, next action such as polling a review resource. No unapproved draft answer. |
| `completed` | Validated answer can be released | Final answer, citations/evidence references, grounding result, explanation, final status. |
| `refused` | A safety/policy check blocks the request | Safe refusal reason and applicable reason codes. No hidden prompt or blocked content. |
| `not_grounded` | Answer fails evidence/citation policy | No answer text; evidence/citation validation summary and reason. |
| `error` | Failure after HTTP headers were sent | Stable public error code/message, request ID, retryability indicator where known. No stack trace/provider response. |
| `cancelled` | Workflow cancellation is confirmed | Request ID and cancellation stage. |

Terminal events must be mutually exclusive, sent no more than once, and followed by stream close. Clients must treat EOF without a terminal event as an incomplete request.

### Example successful stream

```text
event: connected
data: {"request_id":"...","sequence":0,"protocol_version":"1"}

event: progress
data: {"request_id":"...","sequence":1,"stage":"input_guard","status":"passed"}

event: progress
data: {"request_id":"...","sequence":2,"stage":"retrieving","status":"completed"}

event: evidence_summary
data: {"request_id":"...","sequence":3,"evidence_count":3,"chunk_ids":["chunk_a","chunk_b","chunk_c"]}

event: progress
data: {"request_id":"...","sequence":4,"stage":"answer_validation","status":"started"}

event: completed
data: {"request_id":"...","sequence":5,"final_status":"answered","answer":"...","grounding":{"status":"grounded","cited_chunk_ids":["chunk_a"]},"citation_check":{"valid":true,"cited_chunk_ids":["chunk_a"]},"evidence":[{"chunk_id":"chunk_a","title":"...","source_url":"..."}]}

```

The event example is illustrative. The exact fields must reuse or deliberately version the existing Pydantic response contracts.

## 7. Security, privacy, and abuse controls

### Request and transport security

- Require TLS outside local development.
- Validate allowed origins and CORS policy; do not treat CORS as authentication.
- Add authentication and per-tenant authorization before exposing sensitive corpora or review resources.
- Enforce request-body limits, query-length limits, timeouts, max concurrent streams per user/IP, and an overall generation deadline.
- Rate-limit stream creation and total concurrent streams. A minute-based request quota alone does not prevent long-lived connection exhaustion.
- Trust forwarded client-IP headers only when set by a configured trusted reverse proxy.
- Disable intermediary buffering and caching for SSE; never cache personalized answers.

### Prompt, evidence, model, and output security

- Treat user input, retrieved documents, model completions, and tool arguments as untrusted.
- Keep the planner on a strict tool allowlist and validate every tool argument in application code.
- Screen user input before model calls; screen retrieved evidence before it is placed into model context.
- Do not allow instructions in evidence to alter system/developer policies.
- Screen the completed answer before any answer-bearing event is emitted.
- Keep API keys and provider credentials in server-side secret storage. Never place them in event payloads, logs, trace metadata, or error messages.
- Do not expose full prompts, retrieved chunk text, chain-of-thought, raw provider errors, stack traces, or sensitive model internals in progress events.
- Avoid logging query text. If limited diagnostic text is ever required, obtain explicit privacy review and apply redaction, access control, and retention limits.

### Grounding requirements

The minimum release gate is:

1. At least one usable retrieved evidence item exists.
2. Citation validation succeeds.
3. Every cited chunk ID exists in the retrieved evidence set for this request.
4. Output filtering passes.
5. Governance policy permits the response to be released to the caller.

This minimum gate verifies **citation/evidence consistency**, not semantic entailment. A model can cite a real chunk that does not support its claim. For stronger grounding, use claim-level verification: split the draft into factual claims, map each claim to one or more passages, check support/contradiction, and reject or qualify unsupported claims. Treat verifier outputs as fallible; critical domains may require conservative review rather than automatic release.

The API must not stream answer tokens before these gates. Progress events may expose stage names, result counts, and permitted evidence references, but not a provisional answer.

## 8. Guardrails and outcome policy

| Condition | Stream outcome | Answer release |
|---|---|---|
| Input blocked | `refused` | No |
| Out-of-domain or unsafe classifier decision | `refused` or configured review outcome | No |
| No usable evidence | `not_grounded` or `review_required` | No |
| Retrieved content flagged as prompt injection | `refused` or `review_required` | No |
| Output filter blocks content | `refused` | No |
| Citation IDs missing/invalid/not in evidence | `not_grounded` | No |
| Citation IDs valid, all checks pass, no review required | `completed` | Yes |
| HITL policy requires approval | `human_review_required` | No unapproved draft |
| Provider/retriever/internal failure | `error` | No |
| Client cancellation confirmed | `cancelled` | No partial answer |

Guardrails should produce reason codes and sanitized explanations, not echo unsafe input or content.

## 9. Human-in-the-loop design

The current workflow can signal that human review is required and include triggers/payload data. That alone is not a durable HITL process. Production HITL needs:

- a durable review record with an opaque review ID, tenant/user ownership, request ID, trigger codes, evidence references, draft version, and timestamps;
- encrypted and access-controlled storage for any draft or sensitive evidence;
- explicit reviewer authentication and authorization;
- reviewer actions such as approve, reject, or request revision, each audited with actor and time;
- a state machine such as `pending -> approved | rejected | revision_requested | expired`;
- retention/deletion policy and protection from duplicate actions;
- a signed/versioned reference to the exact answer and evidence being reviewed.

When review is required, the SSE stream sends `human_review_required` and terminates. The client receives a review ID and can use authenticated review/status endpoints. Approval should produce a separately auditable result resource or notification. Do not keep a single SSE connection open indefinitely waiting for a human.

If review persistence/checkpointing does not exist yet, the stream must say that review is pending but must not imply the user can later resume or approve it through a nonexistent route.

## 10. Async orchestration and provider integration

Target path:

- async FastAPI endpoint returns `StreamingResponse`;
- asynchronous provider clients stream or await model responses with bounded timeouts;
- async retriever calls avoid blocking the event loop, or use a bounded executor for CPU/local operations;
- LangGraph async execution reports application stage updates through callbacks or graph events;
- cancellation of the HTTP task propagates to provider requests and graph execution where supported;
- semaphores limit in-flight provider calls and streams.

Do not expose raw LangGraph internal event objects as the public API protocol. Translate them into a stable, versioned application event schema. Internal graph nodes and model vendors may change without breaking clients.

If a provider only supports complete-response calls, progress streaming is still useful: send stage events while awaiting the model, then validate and emit the single completed response. Never claim token streaming in that mode.

## 11. Errors, retries, and cancellation

### Before streaming starts

Use normal HTTP status codes and the existing JSON error schema:

- `400` for malformed stream negotiation or unsupported protocol option;
- `401` / `403` for authentication/authorization failures when added;
- `404` for unsupported domain/route;
- `413` for request body too large;
- `422` for Pydantic validation failure;
- `429` for quota exceeded (include `Retry-After`);
- `503` when provider capacity/circuit state prevents starting work;
- `504` for a request deadline exceeded before the stream is established, if the app can still respond normally.

Document any newly used codes in OpenAPI and the API guide.

### After streaming starts

HTTP status and headers are already committed; do not try to replace them. Emit an `error` terminal event with a stable public code, a safe message, the request ID, and whether retry is appropriate. Keep detailed exception/provider information in protected server logs only.

Retries should be bounded and apply only to idempotent provider operations or operations known not to have side effects. Use backoff and respect provider `Retry-After`. Avoid transparent retries that can multiply cost or duplicate HITL records.

On disconnect, cancel the work if the request is an interactive, non-durable chat. If the request has already become a durable job/review task, detach it according to job policy instead. The client can retry a new chat, but must not assume an incomplete stream is resumable unless event persistence and replay are implemented.

## 12. Observability and LangSmith

- Generate one request ID; echo it in `X-Request-ID`, every event, application logs, and LangSmith metadata.
- Log route, domain, stage, status, duration, event count, evidence count, provider/model identifier (non-secret), final outcome, and failure category.
- Never log prompts, answer text, API keys, authorization headers, raw retrieved chunks, or unredacted provider error bodies.
- Create one top-level LangSmith trace per chat request and child spans/runs for classifier, planner, retriever, answer generation, validators, and governance.
- Include request ID, domain, chosen technique, retrieval mode, provider identifier, limits, and guardrail outcome as trace metadata.
- Add tags for `streaming`, domain, route, and outcome. Do not place user content in tags/metadata.
- Trace failure must not fail the API request. Report tracing disabled/failure through safe observability status/logging.
- Measure time to first event, time per stage, time to final event, tokens/provider usage if available, cancellations, grounding failure rate, review rate, and stream errors.
- Propagate trace context into async tasks explicitly and test concurrent requests for cross-request context leakage.

## 13. Deployment and operations

- Use an ASGI server and reverse proxy configured to allow streaming, disable response buffering/compression that delays events, and set idle/read timeouts longer than heartbeat intervals.
- Add heartbeats to keep intermediaries from closing idle streams, but do not count them as progress.
- Set maximum stream duration and maximum concurrent streams. Apply per-user and per-IP quotas.
- In multi-worker or replicated deployments, use a shared rate-limit/quota store. Current in-memory limiting is process-local and not sufficient to coordinate replicas.
- Ensure graceful shutdown stops accepting new streams and gives active requests a bounded drain period before cancellation.
- Provide dashboards/alerts for provider timeout rates, 429s, 5xxs, cancellations, latency percentiles, and grounding/HITL outcomes.
- Document client requirements: SSE parser, terminal event handling, reconnect behavior, and response closure.

## 14. Compatibility and versioning

- Keep current JSON endpoints unchanged.
- Add separate `/stream` paths; use the same request models wherever practical.
- Version the SSE payload contract independently (for example `protocol_version: "1"` in `connected`).
- Additive event fields are allowed. Renaming/removing event types or changing terminal semantics requires a new protocol version.
- Include `Accept: text/event-stream`; return `406 Not Acceptable` if a client requests an unsupported representation.
- Set `Cache-Control: no-cache, no-transform` and `X-Request-ID` for successful and failed streams.
- Ensure CORS exposes `X-Request-ID` and supports the browser client's `Accept`/authorization headers if browser access is needed.

## 15. Testing strategy

### Unit tests

- Event serialization and sequence monotonicity.
- Terminal-event uniqueness and payload schema validation.
- Citation IDs must be a subset of retrieved evidence IDs.
- Empty retrieval, no citations, invalid citation, output-filter block, and refusal all prevent answer release.
- HITL-required results never include the unapproved answer.
- Error sanitization does not expose prompt text, secrets, provider body, or stack trace.
- Cancellation and timeout map to the correct terminal outcome.

### API/integration tests

- TestClient consumes SSE frames from all streaming routes.
- Assert event order: `connected`, progress/evidence/guardrail events, then exactly one terminal event.
- Verify request ID is consistent across response header, events, logs, and mocked LangSmith metadata.
- Test pre-stream validation and rate limiting as HTTP JSON errors.
- Test post-stream failure as an `error` SSE event, not a second HTTP response.
- Disconnect a client during retrieval and generation; verify cancellation propagation and resource cleanup.
- Test multiple concurrent streams to detect context contamination and event interleaving.
- Test hosted-provider timeout, malformed response, throttling, and provider failure without exposing details.
- Test through the actual reverse proxy in deployment because TestClient does not validate proxy buffering or idle timeout behavior.

### Acceptance criteria

1. No model-generated answer token is sent before all release checks pass.
2. Every successful answer event has a non-empty evidence set and valid citations pointing to those evidence IDs.
3. All refusal, grounding-failure, and HITL paths omit the answer.
4. Every established stream ends with exactly one terminal event or is recorded as disconnected/incomplete.
5. Requests can be correlated across response, logs, and LangSmith without recording user content.
6. Concurrency, deadline, and cancellation limits are observed under load.

## 16. Phased rollout

### Phase 0 — contract and safety tests

Agree on event payloads, document them in OpenAPI/docs, and add tests around the current grounding and guardrail outcomes. No streaming route yet.

### Phase 1 — progress-only SSE

Add separate stream routes with Pydantic request validation, request IDs, rate/concurrency limits, heartbeats, progress events, and one final validated JSON-shaped event. Keep generation non-streamed from the model if needed, but run blocking calls off the event loop with bounded capacity.

### Phase 2 — async model and graph execution

Introduce async provider clients and async graph execution. Propagate cancellation, deadlines, trace context, and structured stage events. Run latency/concurrency tests against hosted providers.

### Phase 3 — durable HITL and reconnectable jobs

Add durable review/job storage, authenticated status/actions, idempotency, checkpointing, and persisted event replay only if product requirements need resumption after disconnect.

### Phase 4 — optional token streaming

Only consider token streaming after claim-level validation, output moderation, cancellation, and client UX policies are established. Prefer buffering each sentence/claim for validation before release. If whole-answer validation remains necessary, retain progress streaming and send the answer once validated.

## 17. Decisions recommended for this project

1. Keep existing non-streaming JSON routes.
2. Add SSE routes as an opt-in progress interface.
3. Stream workflow stages and safe evidence references, not answer tokens, until grounding and moderation can be enforced before each released segment.
4. Return one final answer only after safety, citation/evidence consistency, output filtering, and governance complete.
5. Treat citation checking as a necessary but not sufficient grounding guarantee; add claim-level support verification for stronger guarantees.
6. Use a terminal `human_review_required` result and a durable review resource before offering approval/resume UX.
7. Use a separate job API for batch inference, evaluation, and long-running tasks.

