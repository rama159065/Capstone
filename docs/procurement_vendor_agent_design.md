# Procurement and Vendor-Review Agent Design

## Agent Idea

Build a **Procurement and Vendor-Review Assistant** that answers procurement-policy questions, drafts vendor-review notes, compares bid-evaluation criteria, and cites public procurement regulations or guidance PDFs.

Example questions:

- What criteria should be checked before selecting a vendor?
- Which World Bank procurement guidance discusses rated criteria?
- Draft a vendor review summary from the retrieved procurement evidence.
- What does the FAR say about competition or contractor qualifications?

## Core Agents

| Agent | Role |
|---|---|
| Security Guardrail Agent | Blocks prompt injection, executable payloads, secrets, and attempts to bypass procurement policy. |
| Plain-Text Validation Agent | Confirms the message is plain text with procurement/vendor-review intent. |
| LangGraph Orchestrator | Coordinates guardrails, planning, hybrid retrieval, evidence filtering, drafting, output filtering, citation checks, and HITL routing. |
| Planner / Router | Routes to procurement regulation search, vendor evaluation search, contract-risk search, or drafting. |
| Retrieval Agent | Runs hybrid retrieval over indexed procurement PDFs. The raw JSON corpus is available separately for future extension. |
| Procurement Evidence Agent | Normalizes clauses, guidance sections, source PDFs, page numbers, and chunk IDs. |
| Vendor Review Drafting Agent | Drafts vendor review notes only from retrieved evidence. |
| Citation Critic | Verifies every policy claim cites retrieved evidence. |
| Human Review Agent | Requires analyst approval before vendor disqualification, award recommendation, or externally shared review. |

## Why This Is Agentic RAG

This assistant is **Agentic RAG** because the procurement corpus is accessed through an agent workflow, not a single retrieval prompt. The planner chooses procurement/vendor-review retrieval actions, the evidence guard treats retrieved clauses as untrusted text, the answer agent drafts only from cited evidence, the output filter catches unsafe award or rejection language, and governance can route consequential vendor decisions to HITL.

## RAG Corpus

PDFs are downloaded under:

```text
data/raw/procurement_vendor/pdfs/
data/raw/procurement_vendor/pdf_manifest.jsonl
```

Seed PDF sources include World Bank procurement regulations, Project Procurement Strategy for Development guidance, borrower guidance, procurement directives, rated-criteria bid evaluation guidance, and the Federal Acquisition Regulation.

## Tools

| Tool | Purpose |
|---|---|
| `deterministic_input_guard` | Blocks unsafe or malicious input before retrieval. |
| `llm_plain_text_classifier` | Confirms procurement/vendor-review plain-text intent. |
| `hybrid_procurement_search` | BM25 + dense retrieval over indexed procurement PDF chunks. |
| `vendor_evaluation_lookup` | Retrieves rated criteria, bid/proposal evaluation, and vendor qualification evidence. |
| `procurement_clause_lookup` | Retrieves FAR/World Bank clauses by topic. |
| `output_content_filter` | Blocks generated secrets/tool payloads and routes vendor-award or sensitive outputs to review. |
| `citation_validator` | Confirms cited chunk IDs exist in retrieved evidence. |
| `human_review_queue` | Queues high-impact vendor decisions for approval. |

## Guardrails and HITL

Human review is required for:

- vendor award or disqualification recommendation
- conflict-of-interest or fraud allegation
- externally shared vendor review
- low-confidence or conflicting policy evidence
- failed citation validation
- output content filter review for award/rejection language, sensitive data, or generated unsafe content

The assistant may draft and summarize, but it must not make binding procurement decisions.

## Evaluation

Measure retrieval recall@k separately from answer quality. Security and HITL tests should cover prompt injection, fake vendor secrets, bid-manipulation requests, and unsafe award recommendations.

## Implementation Status

- Index CLI: `scripts/domains/procurement_vendor/index.py`
- Agent CLI: `scripts/domains/procurement_vendor/agent.py`
- Streamlit/HTML UI: `scripts/domains/procurement_vendor/html_app.py`
- Shared runtime: `scripts/agents/rag_agent.py`
- Optional tracing: `--enable-langsmith --langsmith-project <project-name>`
- Unit tests cover config wiring, indexing helpers, guarded LangGraph orchestration, citation validation, output filtering, and HITL routing.
