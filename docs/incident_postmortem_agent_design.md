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
| LangGraph Orchestrator | Coordinates guardrails, planning, hybrid retrieval, evidence filtering, drafting, output filtering, citation checks, and HITL routing. |
| Planner / Router | Routes to incident handling, recovery, forensics, timeline, or action-item drafting. |
| Retrieval Agent | Runs hybrid retrieval over indexed incident-response PDFs. Incident JSON records are available separately for future extension. |
| Incident Evidence Agent | Normalizes source guidance, public incident updates, timestamps, page numbers, and chunk IDs. |
| Postmortem Drafting Agent | Drafts timeline, impact, root-cause hypotheses, detection, mitigation, and follow-up actions. |
| Citation Critic | Verifies claims against retrieved evidence and removes blameful unsupported language. |
| Human Review Agent | Requires approval for sensitive incident reports or external publication. |

## Why This Is Agentic RAG

This drafter is **Agentic RAG** because incident-response guidance is used by a controlled drafting agent rather than a direct answer chain. The graph validates safety, plans incident/postmortem retrieval, builds evidence-backed timelines and action items, filters exploit or credential-sensitive text, validates citations, and routes public or blame-sensitive postmortems to human review.

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
| `hybrid_incident_search` | BM25 + dense retrieval over indexed incident-guidance PDF chunks. |
| `timeline_builder` | Builds draft incident timelines from retrieved updates. |
| `action_item_drafter` | Drafts preventive and corrective actions from evidence. |
| `output_content_filter` | Blocks generated secrets/tool payloads and routes exploit, credential, or publishable outputs to review. |
| `citation_validator` | Confirms claims cite evidence. |
| `human_review_queue` | Routes sensitive or publishable postmortems for review. |

## Guardrails and HITL

Human review is required for:

- public incident postmortems
- security-sensitive details
- exploit steps or credential exposure
- blame, legal, or customer-impact language
- unsupported root-cause claims
- output content filter review for exploit, credential, publishable, or sensitive-data language

The agent should draft, not publish or assign blame independently.

## Evaluation

Evaluate retrieval recall@k separately from postmortem quality. Security and HITL tests should cover credential leaks, exploit requests, hiding facts, and unsupported blame.

## Implementation Status

- Index CLI: `scripts/domains/incident_postmortem/index.py`
- Agent CLI: `scripts/domains/incident_postmortem/agent.py`
- Streamlit/HTML UI: `scripts/domains/incident_postmortem/html_app.py`
- Shared runtime: `scripts/agents/rag_agent.py`
- Optional tracing: `--enable-langsmith --langsmith-project <project-name>`
- Unit tests cover config wiring, indexing helpers, guarded LangGraph orchestration, citation validation, output filtering, and HITL routing.
