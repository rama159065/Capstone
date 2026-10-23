# Travel-and-Expense Policy Agent Design

## Agent Idea

Build a **Travel-and-Expense Policy Agent** that answers reimbursement, per diem, mileage, travel-card, gift, and documentation questions from public travel-policy PDFs.

Example questions:

- What records are required for business travel expenses?
- What does IRS Publication 463 say about meals and car expenses?
- Which policy covers official travel charge cards?
- Is this expense reimbursable under the indexed travel guidance?

## Core Agents

| Agent | Role |
|---|---|
| Security Guardrail Agent | Blocks prompt injection, executable payloads, secrets, card numbers, or attempts to bypass reimbursement policy. |
| Plain-Text Validation Agent | Confirms the request is plain-text travel/expense-policy intent. |
| LangGraph Orchestrator | Coordinates guardrails, planning, hybrid retrieval, evidence filtering, drafting, output filtering, citation checks, and HITL routing. |
| Planner / Router | Routes to tax travel expense, federal travel regulation, JTR, travel-card, or foreign-travel policy search. |
| Retrieval Agent | Runs hybrid retrieval over indexed travel-policy PDFs. The raw JSON corpus is available separately for future extension. |
| Expense Evidence Agent | Normalizes policy source, page, section, and chunk IDs. |
| Expense Decision Drafting Agent | Produces evidence-grounded eligibility guidance and documentation requirements. |
| Citation Critic | Verifies every reimbursement claim cites retrieved chunks. |
| Human Review Agent | Routes high-value, ambiguous, or exception-based reimbursement decisions to a reviewer. |

## RAG Corpus

PDFs are downloaded under:

```text
data/raw/travel_expense/pdfs/
data/raw/travel_expense/pdf_manifest.jsonl
```

Seed PDF sources include IRS Publication 463, Joint Travel Regulations, Federal Travel Regulation PDF, and GSA foreign travel policy.

## Tools

| Tool | Purpose |
|---|---|
| `deterministic_input_guard` | Blocks unsafe input and secrets such as card numbers. |
| `llm_plain_text_classifier` | Confirms safe travel/expense-policy intent. |
| `hybrid_travel_policy_search` | BM25 + dense retrieval over travel PDFs. |
| `expense_category_lookup` | Retrieves meal, mileage, lodging, gift, car, and documentation rules. |
| `output_content_filter` | Blocks generated secrets/tool payloads and routes sensitive reimbursement or personal-data outputs to review. |
| `citation_validator` | Confirms cited evidence exists. |
| `human_review_queue` | Routes high-risk reimbursement decisions for approval. |

## Guardrails and HITL

Human review is required for:

- reimbursement approval/denial
- expense exceptions
- high-value claims
- unclear business purpose
- conflicting policy evidence
- payment-card or personal-data exposure
- output content filter review for payment, exception, card, or sensitive-data language

The agent can explain policy and draft a recommendation, but it must not approve payment.

## Evaluation

Evaluate recall@k for policy sections separately from end-to-end answer quality. Security and HITL tests should cover card numbers, fake receipts, policy bypass requests, and ambiguous reimbursement scenarios.

## Implementation Status

- Index CLI: `scripts/travel_expense/index.py`
- Agent CLI: `scripts/travel_expense/agent.py`
- Streamlit/HTML UI: `scripts/travel_expense/html_app.py`
- Shared runtime: `scripts/domain_common/rag_agent.py`
- Optional tracing: `--enable-langsmith --langsmith-project <project-name>`
- Unit tests cover config wiring, indexing helpers, guarded LangGraph orchestration, citation validation, output filtering, and HITL routing.
