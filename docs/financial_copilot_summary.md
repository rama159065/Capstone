# Financial copilot: summary of the agentic design

This is a short summary of the recommendations. The full design is in [financial_report_agentic_ai_recommendations.md](financial_report_agentic_ai_recommendations.md). Status: **recommendation for review**. The implementation will be assessed after the open decisions in section 6 are settled.

---

## 1. Two flows

The copilot has two flows. Both use the same guardrails, retrieval and Qdrant index (`docs_text` + `visual_pages`).

| | Flow A: report generation | Flow B: financial Q&A |
|---|---|---|
| Example | "Generate a complete financial report for Apple" | "What was Apple's gross margin in FY2024?" |
| How it runs | Background job (`POST /reports`) | Chat turn (`POST /chat/financial_report_qa` + `conversation_id`) |
| Agents | **15 top-level nodes** + a 6-node section subgraph | **10 nodes** (existing 8 + `conversation_memory` + `report_facts`) |
| New agents | `report_planner`, `source_resolver`, `profile_builder`, `narrative_research`, `numeric_consistency_critic` (+ optional `visual_qa`) | `conversation_memory`, `report_facts` (+ a `report_handoff` route) |
| LLM calls | 4 nodes (+1 vision model if `visual_qa` is on) | Classifier, planner and answer (+1 if repaired) |
| Human approval | Mandatory: approve the profile for a new company; approve publication | Only on triggers: advice language, low confidence, citation failure, security |
| Checkpoint key | `thread_id = report_id`, saved after every node (`durability="sync"`) | `thread_id = conversation_id`, saved every turn (`durability="async"`) |

How the flows connect:
- **Chat → report.** Asking for a report in chat creates a Flow A job and returns its `report_id`. The report never runs inside a chat turn.
- **Report → chat.** `report_facts` answers numeric questions from a completed report's verified `analysis.json`.

**Key rule in both flows: numbers never come from the LLM.** They come from deterministic extraction and the 17 reconciliations in `analysis.json`, or are quoted from cited evidence.

---

## 2. Time per request

| Case | Time | Basis |
|---|---|---|
| Deterministic report today (7 stages, no LLM) | **~1.6 s** | Measured on 4 Apple runs (1.56–1.67 s) |
| Flow A, Apple (profile exists): sections one at a time / 3 in parallel | **~2.5–6.5 min / ~1.5–3.5 min** of machine time | Estimate |
| Flow A, company with no profile | +~1–2 min of machine time + human profile approval (5–15 min) | Estimate |
| Flow A human approvals (visual review + publication) | +10–25 min | Estimate |
| Flow A end to end | **~12–30 min** (Apple), **~20–45 min** (new company) | Estimate |
| Flow A resume after one failed section | ~15–40 s + ~1 s for the steps after it | Estimate |
| Flow B question (prose, table or chart) | **~6–15 s** | Estimate |
| Flow B numeric question from a completed report | **~2–4 s** | Estimate |
| Tokens | ~55k per report (~75k for a new company); ~3–9k per question | Estimate |

---

## 3. Checkpoints and rerunning steps

| Operation | How | Result |
|---|---|---|
| Resume after a failure (crash, timeout, Colab disconnect) | `invoke(None, {"configurable": {"thread_id": report_id}})` → `POST /reports/{id}/resume` | **Verified:** only the failed step reruns. Completed sections are not repeated |
| Rerun from a chosen step | Pick a checkpoint from `get_state_history`, then `invoke(None, checkpoint_config)` → `POST /reports/{id}/rerun` | **Verified:** reruns from that node onward; earlier steps are not repeated |
| Rerun selected sections only | `update_state(..., {"sections_to_rerun": [...]})` → `POST /reports/{id}/rerun` | Only the listed sections run again; their old drafts are replaced |
| Continue after a human decision | `invoke(Command(resume=decision))` → `POST /reports/{id}/review` | The graph continues from the paused step |
| Audit | `get_state` / `get_state_history` → `GET /reports/{id}`, `/checkpoints` | Shows status, the next step, pending approvals and the checkpoint list |

"Verified" means the behavior was tested with a small fan-out graph on the installed LangGraph 1.2.17.

Rules that keep reruns safe:

- **Small state.** Large files (PDF, charts, `evidence.json`, `analysis.json`) are stored as paths + hashes, not inside checkpoints. Checkpoints take ~0.5–2.5 MB per report.
- **Versioned outputs.** Each rerun writes `output/pdf/<TICKER>_<YEAR>/<report_id>/v<N+1>/`, so earlier versions stay intact.
- **Retries match `resilience.py`.** Nodes retry automatically only on HTTP 429/503. Ambiguous timeouts are rerun from a checkpoint instead.
- **No duplicate charges.** LLM calls use the idempotency key `report_id:node:section:attempt`.
- **Index safety.** Resuming is refused if the Qdrant index signature has changed.
- **Colab backup.** Copy `checkpoints.sqlite` together with the Qdrant snapshot and `metadata.sqlite3`.
- **Storage.** Use `SqliteSaver` on Colab/dev (`langgraph-checkpoint-sqlite`) and `PostgresSaver` in deployment.

---

## 4. Gaps in the current code

| Gap | Impact |
|---|---|
| Neither graph has a checkpointer (`graph.compile()` with no saver) | A failure means starting over, and a single step cannot be rerun |
| Q&A has no conversation state | Does not meet Track B's "persistent state across turns" requirement |
| The report pipeline is a script, not a LangGraph graph, and has no API route | No routing, human-approval pause/resume or job status |
| Reports need a hand-written profile JSON | Only Apple FY2024 is supported |
| Narrative comes from 9 reviewed notes + one optional RAG question | Commentary is thin; drivers and risks are not researched |
| `langgraph-checkpoint-sqlite` is not installed | Must be added before checkpointing |
| Node caching needs a cache store at compile time, and the default lasts only for the process | After a Colab restart, the checkpoint (not the cache) provides resume |
| The shared call pool has 4 slots and fails fast (`capacity_exceeded`) | Section fan-out must be limited to 3 at a time |
| 60 s request budget and the brief's 10-step cap | A report must run as a job; each section subgraph has its own 10-step cap |

---

## 5. Recommended order of work

1. Add a persistent checkpointer to both graphs. Flow B immediately gets state across turns.
2. Split `generate_financial_report.generate()` into node functions without changing behavior.
3. Build the Flow A graph: planner, source resolver, narrative research, numeric critic, human-approval pauses, retry and cache settings, versioned outputs.
4. Extend Flow B with `conversation_memory`, `report_facts` and `report_handoff`.
5. Add the report job API: create, status, events, review, resume, rerun, checkpoints.
6. Add `profile_builder` with mandatory approval.
7. Optional: add `visual_qa`.
8. Replace the estimates with timings and token counts measured from traces.

---

## 6. Decisions needed

| Decision | Why it matters |
|---|---|
| Report section list and audience (internal or external) | Sets which human approvals are mandatory, and the token and time budget |
| Checkpoint retention period | Storage on Colab/Drive, and audit needs |
| Can chat start a report job directly, or must it ask for confirmation? | User experience and quota use |

---

## 7. Assessing the implementation (after the decisions)

Use the checklist in [section 14 of the full recommendations](financial_report_agentic_ai_recommendations.md). The key checks are:

- Numbers match `analysis.json` 100%.
- All citations resolve.
- Kill-and-resume reruns only unfinished sections.
- A step and selected sections can be rerun, producing a new version.
- A human approval survives a process restart.
- Resume works in a new Colab session.
- Flow B follow-ups resolve the company and year from memory.
- Asking for a report in chat returns a `report_id`.
- p50/p95 timings and token use fall within the estimates.
- All existing tests pass.
