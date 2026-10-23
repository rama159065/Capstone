# Incident Postmortem Drafter Agent Design

## Agent Idea

Build an **Incident Postmortem Drafter** that uses incident-response guidance and public incident records to draft structured postmortems, timelines, contributing factors, impact summaries, and action items.

Example questions:

- Draft a postmortem outline for a service outage using NIST incident-response guidance.
- What should be included in incident recovery documentation?
- Which evidence supports containment, eradication, and recovery steps?
- Summarize lessons learned for an incident without assigning blame.

## Core Agents

| Agent | Role |
|---|---|
| Security Guardrail Agent | Blocks prompt injection, exploit instructions, secrets, credentials, and requests to hide incident facts. |
| Plain-Text Validation Agent | Confirms plain-text incident/postmortem intent. |
| Planner / Router | Routes to incident handling, recovery, forensics, timeline, or action-item drafting. |
| Retrieval Agent | Runs hybrid retrieval over incident-response PDFs and incident JSON records. |
| Incident Evidence Agent | Normalizes source guidance, public incident updates, timestamps, page numbers, and chunk IDs. |
| Postmortem Drafting Agent | Drafts timeline, impact, root-cause hypotheses, detection, mitigation, and follow-up actions. |
| Citation Critic | Verifies claims against retrieved evidence and removes blameful unsupported language. |
| Human Review Agent | Requires approval for sensitive incident reports or external publication. |

## RAG Corpus

PDFs are downloaded under:

```text
data/raw/incident_postmortem/pdfs/
data/raw/incident_postmortem/pdf_manifest.jsonl
```

Seed PDF sources include NIST SP 800-61 Rev. 3, NIST SP 800-61 Rev. 2, NIST SP 800-184, and NIST SP 800-86.

## Tools

| Tool | Purpose |
|---|---|
| `deterministic_input_guard` | Blocks unsafe input, exploit requests, and secrets. |
| `llm_plain_text_classifier` | Confirms incident/postmortem drafting intent. |
| `hybrid_incident_search` | BM25 + dense retrieval over incident guidance and records. |
| `timeline_builder` | Builds draft incident timelines from retrieved updates. |
| `action_item_drafter` | Drafts preventive and corrective actions from evidence. |
| `citation_validator` | Confirms claims cite evidence. |
| `human_review_queue` | Routes sensitive or publishable postmortems for review. |

## Guardrails and HITL

Human review is required for:

- public incident postmortems
- security-sensitive details
- exploit steps or credential exposure
- blame, legal, or customer-impact language
- unsupported root-cause claims

The agent should draft, not publish or assign blame independently.

## Evaluation

Evaluate retrieval recall@k separately from postmortem quality. Add red-team cases for credential leaks, exploit requests, hiding facts, and unsupported blame.
