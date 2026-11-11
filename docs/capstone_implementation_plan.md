# Annual Report Copilot: capstone implementation plan

**Submission:** 25 November 2026 (today is 10 November: 15 days). **Track:** B, API-side (hosted ≤8B chat endpoint plus an embeddings endpoint; no GPU for serving).

**What we submit** (from `Capstone-Templates-GenAI-Agentic-AI.pdf`):
- the Git repository with the full agent stack;
- a trace dashboard (LangSmith) showing real queries;
- an evaluation report with the failure taxonomy;
- a 12-page technical report;
- a 10-minute live demo of the deployed system.

**How it is graded:**

| Component | Weight | What earns it here |
|---|---|---|
| Technical depth and rigour | 30% | Hybrid RAG, a planner → executor → critic graph, tools with Pydantic schemas, grounding checks, measured results |
| Punctuality and consistent progress | 25% | The day plan in section 6, with commits every day |
| Working deployed system | 20% | One server running FastAPI, Streamlit and Qdrant behind HTTPS |
| Report and demo | 25% | Evaluation tables, the failure taxonomy, and the "what we did to compensate for a small model" section |

Status column in this document:
- **Exists:** in the repo and tested; 387 tests pass (10 Nov).
- **Extend:** partly there.
- **New:** to build.

---

## 1. Product scope

An analyst copilot over **1,004 SEC annual reports (Form ARS)** from **281 companies in 13 sectors**, fiscal years 2022–2025 ([annual_report_corpus_plan.html](annual_report_corpus_plan.html)).

- **Every answer is grounded:** each figure carries a citation to the filing and page it came from, or a source XBRL fact.
- **Calculations are code:** they are done by a calculator tool from cited inputs, never by the LLM.
- **Refusals:** the copilot refuses investment advice and anything outside the corpus. It answers "not available" rather than guess.
- **Human review:** a person reviews answers flagged for review before they are shown as final, and approves any generated report.

### What the copilot must refuse

| Request | Behaviour |
|---|---|
| Buy, sell or hold advice, price targets | Refuse; offer factual metrics instead |
| A company, year or metric not in the corpus | `not_available`, naming what *is* covered (the clarifier does this) |
| A metric that doesn't apply to the company (e.g. gross margin for a bank) | Explain why, and suggest the right measure |
| Prompt injection, code execution, secrets | Block at the input guard |
| A figure that can't be supported by evidence | Withhold the answer (grounding policy) |

---

## 2. Feature catalogue

The full build specification for every feature is in [features_to_implement.md](features_to_implement.md) ([HTML](features_to_implement.html)): design, files, Pydantic contracts, tests and metrics.

Priorities:
- **P0:** required for the submission.
- **P1:** strong demo value; build if P0 is on track.
- **P2:** stretch.

### 2.1 Single filing (one company, one year)

| # | Feature | Example question | Priority | Status |
|---|---|---|---|---|
| F1 | Cited Q&A | "What was Costco's net income in FY2024?" | P0 | Exists |
| F2 | Clarifier for missing company or year | "What was revenue?" → "Which company and year?" | P0 | Exists |
| F3 | "Not available" / refusal | "Apple's 2019 revenue?" (not in the corpus) | P0 | Exists |
| F4 | Section summary (MD&A, risks, CEO letter) | "Summarise JPM's 2025 risk factors" | P1 | Extend: summary prompt plus a citation per bullet |
| F5 | Statement extraction | income statement, balance sheet, cash flow as tables | P1 | Extend: report generator (Apple profile) or the XBRL facts tool |
| F6 | Chart and table reading | "What does the revenue-by-region chart show?" | P2 | Extend: ColPali search exists; chart values are estimates |

### 2.2 One ticker across years

| # | Feature | Example | Priority | Status |
|---|---|---|---|---|
| F7 | Trend and growth (YoY, 3-year CAGR) | "How did NVIDIA's revenue grow 2023–2026?" | **P0** | New: facts tool plus calculator |
| F8 | Ratios with cited inputs | margins, ROE, debt/equity, FCF | **P0** | New: calculator tool |
| F9 | What changed this year | new risk factors, changed outlook | P1 | New: section comparison of year *n* and *n−1* |
| F10 | Company fact-sheet PDF | KPIs, charts and highlights in one page | P2 | Extend: `generate_financial_report` |

### 2.3 Within a sector

| # | Feature | Example | Priority | Status |
|---|---|---|---|---|
| F11 | Peer comparison table | "Compare JPM, BAC and WFC on net income 2024" | **P0** | New: decompose → parallel retrieval → table |
| F12 | Sector filter in search and UI | Streamlit sector picker | P0 | Exists |
| F13 | Sector-specific metrics | banks: NIM, CET1; insurers: combined ratio | P1 | Extend: the clarifier already knows bank exceptions |
| F14 | Ranking within a sector | "Highest operating margin in retail, 2024" | P1 | New: facts tool |

### 2.4 Across sectors

| # | Feature | Example | Priority | Status |
|---|---|---|---|---|
| F15 | Theme tracking | "Which sectors mention tariffs most in 2025?" | P2 | New: keyword counts by sector and year, from payload filters |
| F16 | Cross-sector benchmark | R&D share of revenue across sectors | P2 | New: facts tool |

### 2.5 Agentic workflows

| # | Feature | What the agent does | Priority | Status |
|---|---|---|---|---|
| F17 | Planner → executor → critic with repair | Plan, call tools, check citations and claim support, retry retrieval or repair once | P0 | **Done** (10 Nov): planner-chosen tools, retrieval retry, support-checking critic, budget-aware loops ([agent_flows.md](agent_flows.md)) |
| F18 | Comparison agent | Splits multi-company or multi-year questions; joins results; never fills gaps | **P0** | **Extend**: one filtered search per company × year is done; the joined comparison table is not |
| F19 | Research memo with human approval | Plan → tools → draft → critic → **human approves** → PDF | P1 | New: reuse the HITL queue |
| F20 | Change alerts | A new filing is compared with last year and big changes flagged | P2 | New |

**Scope for 25 November:** all P0 (F1–F3, F7, F8, F11, F12, F17, F18), then F4, F5 and F19 if time allows. P2 goes in the report as "future work".

---

## 3. Architecture

```mermaid
flowchart LR
  subgraph Offline["Offline (Colab, batch)"]
    Plan["corpus_plan.jsonl<br>1,004 ARS PDFs"] --> DL["download → Drive"]
    DL --> Ingest["page analysis · OCR · chunking<br>(financial_ingest_qdrant_db)"]
    Ingest --> QT["Qdrant annual_reports_text<br>dense + BM25 sparse"]
    Ingest -. optional .-> QV["Qdrant annual_reports_visual<br>ColPali"]
    XBRL["SEC companyfacts JSON<br>281 companies"] --> Facts["facts.sqlite<br>(new)"]
    Eval["eval_set_v1.jsonl<br>86 frozen queries"]
  end
  subgraph Online["Online (one server)"]
    UI["Streamlit"] -->|HTTP| API["FastAPI<br>rate limit · request ID · cache"]
    API --> G["LangGraph agent"]
    G --> T1["retrieve_filing_text"]
    G --> T2["lookup_fact"]
    G --> T3["calculate"]
    G --> T4["compare_companies"]
    G --> T5["company_directory"]
    T1 --> QT
    T2 --> Facts
    G -.-> LLM["Hosted ≤8B endpoint"]
    G -.-> LS["LangSmith traces"]
  end
```

### 3.1 Agent graph (LangGraph)

As built (10 Nov). The clarifier runs before the graph in `/conversation`; every branch is a code decision on the state, and a loop runs only if the rest of the run fits `AGENT_MAX_STEPS`. Full description: [agent_flows.md](agent_flows.md).

```
security → plain_text (LLM) → planner (LLM + rules) → executor ⇄ evidence_guard   (retry once: missing company/year)
                                                       │ coverage_lookup, text_search per company × year,
                                                       │ visual_search (ColPali, optional)
                                                       ▼
                         answer (LLM, CALC[...]) → calculator? → critic ──repair once──► answer
                                                                  │ citation IDs + figures in cited chunks
                                                                  │ + LLM judge per claim
                                                                  ▼
            refusals / not available / nothing found ───────► governance / human review → response
```

| Element | Implementation | Status |
|---|---|---|
| State | `AgentState` TypedDict; conversation slots (company, year, metric) | Exists |
| **Persistent state across turns** | `ConversationStore(path=CONVERSATION_DB)`: SQLite keyed by `conversation_id`, 30-minute TTL, survives API restarts | **Done** (10 Nov; verified across a restart) |
| **Step cap** | `AGENT_MAX_STEPS=10` node executions (`recursion_limit` = steps + 1) on both graphs; over the cap → `step_limit`, no answer | **Done** (10 Nov) |
| **Token budget** | Count prompt and evidence tokens; trim evidence to fit; stop when the budget is used up | **New** |
| Malformed JSON | Pydantic validation → one retry with the validation error in the prompt → fallback to extractive | Extend (`answer_schema.py`) |
| Tool failures | `Calls` timeouts and `CallFailure` reasons; the tool error goes back into the state, never crashes the run | Exists |
| Infinite loops | At most 1 retrieval retry and 1 repair; each runs only if the rest of the run fits the step cap, so runs shorten instead of hitting `step_limit` | **Done** (10 Nov) |
| Routing | Planner-chosen tools validated by rules (`financial_tools.normalize_plan`); per-request branches for retry, calculator and repair; the path and every tool call are returned as `agent_run` and shown in Streamlit | **Done** (10 Nov) |

### 3.2 Tools (Pydantic input and output schemas)

| Tool | Input model | Output | Status |
|---|---|---|---|
| `retrieve_filing_text` | `query, ticker?, year?, sector?, k` | chunks with `chunk_id`, page, score | Exists (hybrid RRF and rerank); runs as the graph's `text_search` tool, one call per company × year |
| `coverage_lookup` | targets (ticker, year) | searchable vs skipped targets, with the reason | **Done** (10 Nov): reads the served index's filing list |
| `visual_search` | as `text_search` | ColPali page hits merged into the ranking | **Done** (10 Nov): planner-chosen with `RAG_FIN_VISUAL=auto`; not run live (no local ColPali index) |
| `lookup_fact` | `ticker, concept, fiscal_year` | value, unit, period, accession (XBRL) | **New** |
| `calculate` | `operation, inputs[{value, source_id}]` | result (`Decimal`), formula, input sources | **Extend**: the graph's `calculator` evaluates `CALC[...]` (numbers and `+ - * /` only) and checks each operand is in the evidence; `Decimal` and named operations are not done |
| `compare_companies` | `tickers[], metric, years[]` | table; missing cells marked, not guessed | **New** (calls the two tools above) |
| `company_directory` | `name or ticker` | ticker, sector, years covered | Exists (`companies.py`, clarifier) |

The tools are called by the executor node from the validated plan; the Pydantic input and output models below are still to do.

**Track B note:** small models fumble tool calls. Every tool call is parsed into its Pydantic model and validated. If that fails, the model gets one retry with the error message, and then the run uses a deterministic fallback (the slot parser in the clarifier). This is the "constrained decoding, schema validation, retry" content the template asks for.

### 3.3 Grounding and anti-hallucination ("no hallucination" in measurable form)

A guarantee of zero hallucination isn't possible with any LLM. What the system guarantees is that **unsupported output is not shown as fact**, and it measures the rate.

| Layer | Rule | Status |
|---|---|---|
| Retrieval floor | No answer when the best rerank score is below `RAG_MIN_RERANK_SCORE` | Exists |
| Evidence guard | Injection or code in retrieved text → the chunk is dropped | Exists |
| Citation validity | Every cited ID must be in the retrieved set; otherwise the answer is withheld | Exists (`apply_grounding_policy`) |
| Company scope | An answer citing a company the question didn't ask about is withheld | Exists |
| **Number grounding** | Every number in the answer must appear in a cited chunk or be a calculator result, at the precision the answer prints, in any unit scale (`financial_tools.numeric_grounding`) | **Done** (10 Nov; in the critic) |
| **Claim support (entailment)** | An LLM judge checks each claim against the cited chunks; a failed figure check overrides it. Not supported → one repair with the judge's feedback → still not supported → human review (`unsupported_claims`) | **Done** (10 Nov; `RAG_SUPPORT_CHECK`) |
| **Hosted answers only when verified** | With a hosted LLM the generated answer is shown only when the support verdict is `supported` (`grounding.semantic_answer_verified`); otherwise cited excerpts (`evidence_only`) | **Done** (10 Nov) |
| Coverage check | Company, year or metric not covered → `not_available` | Exists |
| Human review | `review` decisions go to the HITL queue | Exists |

### 3.4 Guardrails and responsible AI

| Area | Implementation | Status |
|---|---|---|
| Input guard | Regex sets for injection, code and tool abuse; encoded payload detection | Exists |
| Injected instructions in the corpus | Evidence guard on retrieved chunks; **new test: plant 10 injected chunks** | Extend |
| PII and secrets | Redaction of secret and sensitive-data patterns in logs and output | Exists; add a test set |
| Advice refusal | Output filter plus a refusal prompt; 8 refusal cases in the eval set | Exists |
| Logs | Access logs never store bodies or prompts; traces redacted | Exists |
| Human-in-the-loop boundary | The copilot drafts and recommends only. Reports, memos and anything marked `review` need a person's approval | Exists; document it |
| Red-team log | 20 or more adversarial prompts with outcomes | **New** (report appendix) |

---

## 4. Evaluation plan

**Frozen set:** [data/evaluation/financial_report_qa/eval_set_v1.jsonl](../data/evaluation/financial_report_qa/eval_set_v1.jsonl), 86 queries (rubric: 60–100).

| Category | Count | Note |
|---|---|---|
| Single figure | 40 | |
| Computed metric | 10 | |
| Text fact | 10 | |
| Refusal | 8 | must refuse |
| Comparison | 6 | |
| Chart value | 6 | |
| Not disclosed | 6 | must answer "not available" |

### 4.1 Metrics

| Metric | How it is measured | Tool |
|---|---|---|
| Retrieval recall@k, MRR, nDCG (**separately from answers**) | Gold evidence pages vs retrieved chunks, k = 1, 5, 10 | `evaluate_financial_rag.py`, `retrieval_metrics.py` |
| End-to-end accuracy | Numeric match within rounding and unit; text facts by rubric | **New:** `evaluate_agent.py` |
| Task completion rate | Answered correctly, or correctly refused or marked not available | New |
| Citation correctness and coverage | The cited page contains the figure; share of claims with a citation | New |
| Schema validity rate | Pydantic parse success, first try vs after retry | New |
| Unsupported-claim (hallucination) rate | Figures not in cited evidence and not computed, plus claims the judge rejects; first answer vs after repair | Data ready (`support_check` in every response); aggregation in `evaluate_agent.py` is New |
| Verified-answer rate (hosted) | Share of answers with `semantic_answer_verified = true` vs excerpt fallback | Data ready; aggregation New |
| Average tool calls per task | From `agent_run.tool_calls` | Data ready; aggregation New |
| Cost and tokens per query; p50/p95 latency | From the trace and endpoint usage | New |

### 4.2 Comparisons for the report

| Run | Purpose |
|---|---|
| 3 single-prompt baselines: structured few-shot, chain of thought or self-consistency, decomposition | Week 1 requirement; `prompt_baselines.py`, `--technique` |
| RAG without tools | Shows what retrieval alone achieves |
| Full agent (tools + critic) | Final system |
| Ablations: no reranker; no critic; no support check (`RAG_SUPPORT_CHECK=0`); no facts tool | "What did we do to compensate for a small model?" |

### 4.3 Failure taxonomy

Each wrong answer gets one label:
- retrieval miss;
- reasoning error;
- tool-call error;
- hallucination (unsupported claim);
- permission or routing failure;
- malformed output.

The report includes the counts by label and 2–3 worked examples of each.

---

## 5. Deployment

| Item | Choice |
|---|---|
| Containers | **Done (10 Nov), not yet built on a Docker host:** `Dockerfile` (one CPU image) and `docker-compose.yml` with `qdrant`, `api` (uvicorn, 1 worker), `ui` (streamlit) and `caddy` (HTTPS profile). Steps and costs: [deployment.md](deployment.md) |
| Server | One cloud VM: 4 vCPU, 16 GB RAM, 100 GB disk, **no GPU** (about $100 a month, or ~$50 for the two weeks needed) |
| Index | Text collection restored from the Drive snapshot (`cp.restore`); ColPali collection only if the visual demo is shown |
| LLM | Track B hosted endpoint (`LLM_API_BASE/KEY/MODEL`) |
| Rate limiting | Exists: 60 requests a minute per IP |
| **Caching** | Model and retrieval calls (5-minute idempotent cache), plus the **answer cache** (`ANSWER_CACHE_SECONDS`, identical requests; failures never cached). **Done** (10 Nov) |
| Access | **Done:** `API_AUTH_TOKEN` on the API; Caddy basic auth in front of Streamlit |
| Fallback demo | Same compose stack on a laptop, plus a recorded video in case the network fails |

---

## 6. Day plan to 25 November

| Date | Work | Exit check |
|---|---|---|
| Mon 10 | Freeze scope (this doc). Start the corpus download on Colab (about 1 hour). Write the Dockerfile and compose skeleton | 1,004 PDFs on Drive; `download_summary.json` has no failures |
| Tue 11 – Wed 12 | Ingest the text collection (text + OCR, ColPali off) sector by sector on Colab; snapshot to Drive. In parallel: XBRL companyfacts for 281 companies → `facts.sqlite`; `lookup_fact` tool | `cp.coverage` = 1,004 ingested; facts tool tests pass |
| Thu 13 | `calculate` tool: `Decimal` and named operations on top of the graph calculator. (Number grounding, the claim judge, tool routing and the retrieval retry were done on 10 Nov) | Unit tests; a calculation answer shows its formula and sources |
| Fri 14 | `compare_companies` and the comparison planner (F11, F18); trend and growth (F7) | Comparison questions in the eval set answer with a table |
| Sat 15 | Token budget, Pydantic retry. (Persistent state and the 10-step cap were done on 10 Nov) | Malformed tool output is retried, then falls back |
| Sun 16 | Streamlit: comparison table, trend chart, citation panel, HITL approve button | UI walk-through |
| Mon 17 – Tue 18 | **Evaluation runs:** baselines, RAG only, full agent, ablations; recall@k; latency and cost | `eval/` results JSON and tables |
| Wed 19 | Failure taxonomy labelling; red-team (20+ prompts, 10 injected chunks, PII) | Taxonomy and red-team tables |
| Thu 20 | **Deploy** on the VM with compose and HTTPS; LangSmith project with real traces | Public URL works; trace dashboard link |
| Fri 21 – Sun 23 | 12-page report, architecture diagram, demo script, backup video | Draft reviewed |
| Mon 24 | Full dry run of the demo on the deployed URL; fix only blockers | Freeze code; tag `v1.0` |
| **Tue 25** | **Submit:** repository link, report, evaluation report, trace dashboard, demo | — |

**Smaller corpus:** the 509-filing variant (section 10) finishes indexing a day earlier and costs about $37 instead of $62.

**If the schedule slips, cut in this order:** F19 memo, then F4/F5, then the ColPali visual demo. Never cut: evaluation, the failure taxonomy, deployment.

---

## 7. Six-member role split

| Role | Owns |
|---|---|
| Domain spec and evaluation design | Scope, refusal policy, eval set, accuracy scoring |
| Prompt engineering | Baselines, structured output, small-model failure notes |
| RAG and retrieval | Corpus ingestion, hybrid search, reranker, recall@k |
| Agent orchestration and tools | LangGraph graph, the 5 tools, state, step and token limits |
| Observability and safety | LangSmith, failure taxonomy, red-team, PII tests |
| Front-end and deployment | Streamlit, Docker, VM, HTTPS, caching, demo |

---

## 8. 10-minute demo script

| Time | Scene | Shows |
|---|---|---|
| 0:00 | Problem, corpus (1,004 filings, 13 sectors), architecture slide | Scope |
| 1:00 | "What was JPMorgan's net income in 2024?" → answer with a citation; click to the page | Grounding |
| 2:00 | "What was revenue?" → clarifier asks for company and year | Multi-turn state |
| 3:00 | "Compare JPM, BAC and WFC net income 2022–2025" → table, growth, sources | Planner, tools, calculator |
| 4:30 | "Should I buy NVIDIA?" → refusal; injected-prompt attempt → blocked | Guardrails |
| 5:30 | A question outside the corpus → `not_available` | No hallucination |
| 6:00 | LangSmith trace of the comparison run: steps, tool calls, latency | Observability |
| 7:00 | Evaluation dashboard: baseline vs agent, recall@k, failure taxonomy | Rigour |
| 8:30 | HITL: a `review` answer approved in the UI | Human boundary |
| 9:30 | Limits and next steps | Honesty |

---

## 9. Risks and decisions to confirm with the mentor

Choosing between the full corpus and the 509-filing variant is also worth confirming early (section 10).


| Risk | Mitigation |
|---|---|
| Track B says "no GPU", but ColPali, OCR on GPU and VLM ingestion use a Colab GPU | Keep the **graded path GPU-free** (text + CPU OCR + hosted LLM). Present ColPali and VLM as an optional offline extension. Confirm with the mentor |
| Track B expects the **provided embeddings endpoint**; the index uses local MiniLM | Ask whether local MiniLM is acceptable. If not, build the text index with `--embedding-provider hosted` (re-ingest of the text collection only) |
| The hosted 8B model mis-formats tool calls | Pydantic validation, one retry, a deterministic slot-parser fallback; report the rates |
| Ingestion overruns on Colab | Text-only first (ColPali off); sector by sector with snapshots; use 500 filings if needed (the rubric minimum) |
| Token quota on the endpoint | Answer cache, evaluation runs once per configuration, small `k` |
| Deployment network or credentials fail during the demo | Laptop compose fallback and a recorded video |

---

## 10. Variant: 509 filings (the 500-document minimum)

Track B requires **500–2,000 documents**. Indexing about half the corpus meets that, fits a single Colab indexing session, and roughly halves storage and server cost. The pipeline stays the same, and the rest of the corpus can be added later by rerunning with `MAX_PER_SECTOR = 0`.

### 10.1 Which filings

| Selection | Filings | Companies | Size | Eval filings covered |
|---|---|---|---|---|
| Largest companies, 40 filings per sector | 490 | 138 | 5.1 GB | 38 / 42: misses Adobe 2024–25 and Salesforce 2025–26 (and is below 500) |
| Only fiscal years 2024 + 2025 | 522 | 272 | 6.4 GB | 35 / 42, and only 2 years per company, which weakens trends |
| **Chosen: 41 per sector + all Adobe and Salesforce filings** | **509** | **143** | **5.36 GB** | **42 / 42** |

**Coverage of the chosen set:**
- All 13 sectors, about 10 of the largest companies in each (5 in real estate, about 8 in materials).
- 106 of the 143 companies have all 4 years; 12 have only one filing (cut at the sector limit).
- **Eval set v1 is unchanged:** every one of the 42 filings it uses is included.

### 10.2 What changes

| | 1,004 filings | **509 filings** |
|---|---|---|
| Pages (147 per filing) | ~147,600 | **~74,800** |
| Download | 11.3 GB, 30–60 min | **5.4 GB, 15–30 min** |
| Colab indexing, text + OCR (L4) | ~5–8 h, several sessions | **~2.5–3.5 h, one session** |
| Plus ColPali image vectors | +2.5–7.5 h | **+1.3–3.8 h** |
| Image vectors on disk | ~7 GB | **~3.6 GB** |
| Pages with no usable text (for a vision model) | ~30,600 | **~15,500** |
| XBRL facts download | 281 companies | **143 companies** |
| `GET /companies` lists | 281 tickers | **143 tickers** |
| Server | 4 vCPU / 16 GB / 100 GB | **2 vCPU / 8 GB / 50 GB** |
| **Total spend** | ≈ $62 | **≈ $37** |

Times and sizes are estimates scaled from earlier measurements; costs are in [deployment.md](deployment.md) §5.7.

### 10.3 Day plan differences

| Date | Change |
|---|---|
| Mon 10 | Download about 509 PDFs instead of 1,004. Exit check: 509 on Drive, `download_summary.json` has no failures |
| **Tue 11** | Indexing (text + OCR) finishes in one session, by the evening. Exit check: 509 ingested; `GET /companies` lists 143 tickers |
| Wed 12 | **The day freed up:** the XBRL facts store (143 companies) and `lookup_fact`, a day earlier than planned. Spend the later slack on evaluation runs, or on F4/F13/F14 |
| Thu 20 | Deploy on the smaller server (2 vCPU / 8 GB) |

Every other row of the day plan (section 6) stays the same.

### 10.4 How to run it

The corpus size is one setting, **`CORPUS`**, used by download, ingestion and coverage alike:

| Setting | Filings | Companies | Size |
|---|---|---|---|
| `full` | 1,004 | 281 | 11.3 GB |
| `min500` | 509 | 143 | 5.36 GB |

**On Colab:** in [colab_rag_pipelines.ipynb](../notebooks/colab_rag_pipelines.ipynb) cell 1, set `CORPUS = "min500"`, `MAX_PER_SECTOR = 0` and `SMOKE_TEST = False`. Then run as usual.

**From the command line:**

```bash
python -m scripts.downloads.download_annual_report_corpus --preset min500 --output-dir <drive>/annual_reports
python -m scripts.rag.colab_pipeline run --pipeline colpali_qdrant --corpus-preset min500 --max-per-sector 0
python -m scripts.rag.colab_pipeline coverage --pipeline colpali_qdrant --corpus-preset min500   # planned = 509
python -m scripts.rag.financial_ingest_qdrant_db ingest --manifest <drive>/annual_reports --corpus-preset min500 ...
```

**What `min500` selects:** the 41 largest filings per sector, plus every filing of the companies the frozen eval set asks about. It is defined in `download_annual_report_corpus.preset_rows`.

**Behaviour:**
- Ingestion skips downloaded filings outside the preset; nothing is deleted. Switching to `full` later just adds the rest, and filings already ingested are skipped.
- `cp.coverage` counts against the preset, so "planned" is 509.

### 10.5 What we give up, and how the report should say it

| Limit | Effect | How to present it |
|---|---|---|
| About 10 companies per sector | Rankings (F14) and sector benchmarks (F16) cover sector leaders only | The UI and the report say "largest ~10 companies by sector" |
| 138 companies not indexed | Questions about them get `not_available`, listing what is covered | Correct behaviour: the clarifier reads coverage from the index |
| 12 companies with one filing | No trend for those companies | Alternative selection: top 10 companies × all years per sector |
| Half the planned corpus | — | "509 of 1,004 planned filings; the pipeline resumes to the full corpus with `MAX_PER_SECTOR = 0`" |
