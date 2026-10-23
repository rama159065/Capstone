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
| Planner / Router | Routes to procurement regulation search, vendor evaluation search, contract-risk search, or drafting. |
| Retrieval Agent | Runs hybrid retrieval over procurement PDFs and existing JSON/API records. |
| Procurement Evidence Agent | Normalizes clauses, guidance sections, source PDFs, page numbers, and chunk IDs. |
| Vendor Review Drafting Agent | Drafts vendor review notes only from retrieved evidence. |
| Citation Critic | Verifies every policy claim cites retrieved evidence. |
| Human Review Agent | Requires analyst approval before vendor disqualification, award recommendation, or externally shared review. |

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
| `hybrid_procurement_search` | BM25 + dense retrieval over procurement PDFs and records. |
| `vendor_evaluation_lookup` | Retrieves rated criteria, bid/proposal evaluation, and vendor qualification evidence. |
| `procurement_clause_lookup` | Retrieves FAR/World Bank clauses by topic. |
| `citation_validator` | Confirms cited chunk IDs exist in retrieved evidence. |
| `human_review_queue` | Queues high-impact vendor decisions for approval. |

## Guardrails and HITL

Human review is required for:

- vendor award or disqualification recommendation
- conflict-of-interest or fraud allegation
- externally shared vendor review
- low-confidence or conflicting policy evidence
- failed citation validation

The assistant may draft and summarize, but it must not make binding procurement decisions.

## Evaluation

Measure retrieval recall@k separately from answer quality. Add security and HITL test cases covering prompt injection, fake vendor secrets, bid-manipulation requests, and unsafe award recommendations.
