# FastAPI API Reference

The FastAPI backend exposes domain discovery, guarded chat, and evidence retrieval. Request validation, response schemas, and endpoint descriptions are also published in the interactive API documentation.

Streaming is not currently implemented. The proposed SSE architecture, event contract, security controls, grounding gates, and HITL workflow are described in the [FastAPI streaming design](fastapi_streaming_design.md).

## Start the API

From the repository root, start the server:

```powershell
uvicorn scripts.api.api_app:app --host 127.0.0.1 --port 8000
```

Open the interactive Swagger UI at <http://127.0.0.1:8000/docs>, ReDoc at <http://127.0.0.1:8000/redoc>, or the raw OpenAPI schema at <http://127.0.0.1:8000/openapi.json>.

## Routes

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Return service status and domain identifiers. |
| `GET` | `/domains` | List domain identifiers, display names, descriptions, and default queries. |
| `POST` | `/chat/{domain}` | Answer a question using a non-financial domain. |
| `POST` | `/chat/financial_report_qa` | Answer a financial-report question using the selected financial technique. |
| `POST` | `/conversation/financial_report_qa` | One chat turn: clarify a missing company, year or measure first, then run the agents or the report pipeline. Used by the Streamlit chat. |
| `POST` | `/retrieve/{domain}` | Return ranked evidence from a non-financial domain without generating an answer. |
| `POST` | `/retrieve/financial_report_qa` | Return ranked financial-report evidence without generating an answer. |

The financial routes have dedicated request schemas and should be used for the `financial_report_qa` domain. The supported non-financial domain IDs are listed by `GET /domains`.

## Chat requests

Chat requests are JSON objects. `query` is required; all other fields are optional.

| Field | Type | Default | Constraints / purpose |
|---|---|---|---|
| `query` | string | — | Required, 1–4000 characters. |
| `llm_provider` | string | `mock` | `mock` or `hosted`. Hosted mode requires the configured chat endpoint environment variables. |
| `embedding_provider` | string | `hashing` | `hashing` or `hosted`. |
| `hash_dimensions` | integer | `384` | 32–4096; used by hashing embeddings. |
| `top_k` | integer | `5` | 1–20; maximum evidence passed to the answer workflow. |
| `dense_top_k` | integer | `30` | 1–100 dense retrieval candidates. |
| `bm25_top_k` | integer | `30` | 1–100 lexical retrieval candidates. |
| `source_type` | string or null | `null` | Optional source category filter. |
| `max_context_chars` | integer | `12000` | 1000–64000; non-financial and financial chat context limit. |
| `enable_guardrails` | boolean | `true` | Enable input, evidence, and output safety checks. |
| `enable_human_review` | boolean | `true` | Enable governance routing for human review. |
| `enable_langsmith` | boolean | `false` | Request LangSmith tracing (credentials must be configured). |
| `langsmith_project` | string or null | `null` | Optional project override. |

Financial chat additionally accepts:

| Field | Type | Default | Constraints / purpose |
|---|---|---|---|
| `technique` | string | `agentic_graph` | One of `baseline_rag`, `structured_few_shot`, `decomposition`, `self_consistency`, `agentic_graph`. |
| `retrieval_mode` | string | `hybrid` | `dense` or `hybrid`. |
| `ticker` | string or null | `null` | Optional company ticker filter. |
| `year` | integer or null | `null` | Optional fiscal/reporting year filter. |
| `self_consistency_samples` | integer | `3` | 1–5 answer samples for the self-consistency technique. |

Example non-financial chat request:

```powershell
$body = @{
  query = "What documentation is required for travel expenses?"
  llm_provider = "mock"
  embedding_provider = "hashing"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/chat/travel_expense" `
  -ContentType "application/json" `
  -Body $body
```

Example financial chat request:

```json
{
  "query": "What does Apple say about revenue?",
  "ticker": "AAPL",
  "year": 2024,
  "technique": "agentic_graph",
  "llm_provider": "mock"
}
```

## Conversation requests (clarify, then run)

`POST /conversation/financial_report_qa` accepts every financial chat field plus:

| Field | Type | Default | Purpose |
|---|---|---|---|
| `conversation_id` | string or null | `null` | Returned by the first turn; send it back with each follow-up |
| `clarify` | boolean | `true` | `false` skips clarification and behaves like `/chat/financial_report_qa` |

[scripts/agents/financial_clarifier.py](../scripts/agents/financial_clarifier.py) reads each turn into slots: intent, companies, fiscal years and measure. It checks them against the companies and years the corpus actually has, from both PDF manifests. It is deterministic and makes no LLM call.

| Intent | Needs before agents run | Defaults it states instead of asking |
|---|---|---|
| figure ("What was Apple's revenue?") | company, year, measure | — |
| comparison | two or more companies, a year all of them have, measure | — |
| trend | company, measure | all years in the corpus |
| explain ("Why did Chevron's earnings fall?") | company | latest year |
| report ("Generate a financial report for Apple") | company, year, then a **Yes/No confirmation** | — |
| high-level ("Tell me about Microsoft") | the user chooses: key figures, explain, trend or full report | — |

**Response `status`:**

| Status | Meaning |
|---|---|
| `needs_clarification` | `message` and `questions[]`, each with `slot`, `question` and `options`. No agent has run |
| `answered` | The agents ran on `resolved_query` with the `ticker` and `year` filters. `result` is the normal financial chat response, and `assumptions` lists any defaults used |
| `report_generated` | A confirmed report ran through the deterministic pipeline. `report_manifest` holds the artifacts; the report is pending visual review |
| `report_unavailable` | No report profile exists yet for that company and year (only Apple FY2024 so far) |
| `report_failed` | The report pipeline raised an error; the details are in the report trace |
| `cancelled` | The user answered No to the report confirmation |

**Rules:**
- **Already complete:** a complete question runs unchanged, with no extra turn.
- **Unknown companies and years:** companies and years outside the corpus are named ("FY2027 is not in the filings I have"), and the user is offered the years that are available.
- **Follow-ups keep context:** "and for 2023?" or "what about Visa?" reuse the measure and company. Any other sentence after an answer starts a new request.
- **Escape hatches:** `continue` answers the question as asked; `start over` resets. After 3 clarification turns the question is answered as asked.
- **Guarded messages:** advice, prediction and prompt-injection messages are never clarified. They go straight to the guarded agent, which refuses them.
- **State:** conversations are kept in memory per API process with a 30-minute TTL, like the rate limiter. Run one worker, or use a shared store when scaling out.

```json
{"query": "What was Apple's revenue?"}
→ {"status": "needs_clarification", "conversation_id": "…", "message": "Which fiscal year for Apple?",
   "questions": [{"slot": "year", "options": ["FY2024", "FY2023", "…", "Latest"]}]}
{"query": "FY2023", "conversation_id": "…"}
→ {"status": "answered", "resolved_query": "What was Apple's revenue for FY2023?", "ticker": "AAPL", "year": 2023, "result": {…}}
```

## Retrieval requests

Retrieval endpoints accept a JSON body with a required `query` (1–4000 characters) and optional `embedding_provider` (`hashing` or `hosted`), `hash_dimensions` (32–4096), `top_k` (1–20), `dense_top_k` and `bm25_top_k` (1–100), `source_type`, `enable_langsmith`, and `langsmith_project`.

Financial retrieval also accepts `retrieval_mode` (`dense` or `hybrid`), `ticker`, and `year`. Non-financial domains use hybrid retrieval.

Successful retrieval responses contain `domain`, `query`, `langsmith`, `count`, and an `evidence` array. Evidence fields may include `chunk_id`, `title`, `source_type`, `ticker`, `year`, `page`, `source_url`, `image_path`, `rank`, `score`, and `preview`; which fields are present depends on the domain and source.

## Responses

Chat responses include the domain and final status, and may include the answer, an `explanation`, `grounding` status, citation checks, evidence previews, guardrail decisions, human-review information, orchestration metadata, elapsed time, and LangSmith tracing status. The API only returns a generated answer when there is retrieved evidence and citation validation reports cited chunk IDs that belong to that evidence. Otherwise, it withholds the generated answer (`answer: null`, `final_status: "not_grounded"`) and explains the reason. A guardrail refusal is marked `not_applicable` for answer grounding because no generated answer is being returned.

The `grounding` object reports `status` (`grounded`, `not_grounded`, or `not_applicable`), evidence count, cited chunk IDs, and a reason. This is a citation/evidence consistency check, not a guarantee that every factual claim is entailed by its cited passage or is factually correct. The explanation does not expose hidden chain-of-thought.

`GET /health` returns `status`, `service`, and `domains`. `GET /domains` returns a `domains` array with each domain's identifier, display name, description, and default query; non-financial domains also include their supported `source_types`.

## Errors and rate limits

Errors use a consistent JSON shape:

```json
{
  "error": "validation_error",
  "message": "Request validation failed.",
  "details": [
    {
      "field": "body.query",
      "message": "String should have at least 1 character",
      "type": "string_too_short"
    }
  ],
  "request_id": "12345678-1234-5678-1234-567812345678"
}
```

| Status | Meaning |
|---|---|
| `404` | Unknown domain includes a link to `GET /domains`; unknown routes include a link to the API docs. |
| `405` | HTTP method is not supported for the endpoint; the response includes the `Allow` header. |
| `422` | Request body or parameter validation failed; `details` identifies invalid fields. |
| `429` | Rate limit exceeded; response includes the configured limit and a `Retry-After` header. |
| `500` | Unexpected server error. Internal exception details are logged by the server and are not returned to clients. |

Error responses include the request's `request_id`, which also appears in the `X-Request-ID` response header and can be used to find the corresponding server log.

The API allows 60 requests per minute per client IP across all routes. The current limiter is in-memory and per application process. Multi-worker or replicated deployments need a shared rate-limit store to enforce a unified limit. Client identity uses the direct peer address; configure trusted proxy handling at the deployment boundary when running behind a reverse proxy.

## LangSmith tracing

Set `LANGSMITH_API_KEY` (or `LANGCHAIN_API_KEY`) and either pass `"enable_langsmith": true` or set `LANGSMITH_TRACING=true`. `langsmith_project` can override the default project name. If tracing is requested without credentials, the response reports tracing as disabled with a reason.

## Request logging and trace correlation

The API logs one completion event per request with its request ID, HTTP method, path, response status, and elapsed milliseconds. Client IP addresses, request bodies, query text, authorization data, and response content are intentionally excluded from access logs. Server-side unexpected exceptions are logged with stack traces; clients receive only the sanitized `500` error.

Every response includes an `X-Request-ID` header. A valid UUID from the incoming `X-Request-ID` header is reused; missing or invalid values are replaced with a generated UUID. When LangSmith tracing is enabled, that same ID is attached to the trace metadata so an API log event can be correlated with its LangSmith run. The API also logs whether LangSmith tracing was enabled and the selected project (never credentials).
