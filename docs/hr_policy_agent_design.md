# HR Policy Navigator Agent Design

## Agent Idea

Build an **HR Policy Navigator** that answers employee-policy questions using public HR, labor, leave, telework, wage-hour, workplace rights, and safety PDFs.

Example questions:

- What does the FMLA employee guide say about requesting leave?
- Which evidence describes employer responsibilities under FMLA?
- What does OPM guidance say about telework agreements?
- What workplace rights poster evidence applies to wage-hour questions?

## Core Agents

| Agent | Role |
|---|---|
| Security Guardrail Agent | Blocks prompt injection, secrets, employee PII, and requests to bypass HR/legal process. |
| Plain-Text Validation Agent | Confirms the message is plain-text HR-policy intent. |
| Planner / Router | Routes to leave, wage-hour, telework, workplace rights, safety, or policy summary search. |
| Retrieval Agent | Runs hybrid retrieval over HR-policy PDFs and existing HR policy records. |
| HR Evidence Agent | Normalizes source agency, policy topic, page, and chunk IDs. |
| Policy Answer Agent | Produces cited HR-policy explanations and flags uncertainty. |
| Citation Critic | Verifies citations and removes unsupported legal/HR conclusions. |
| Human Review Agent | Requires review for personnel decisions, legal advice, discipline, termination, or sensitive employee data. |

## RAG Corpus

PDFs are downloaded under:

```text
data/raw/hr_policy/pdfs/
data/raw/hr_policy/pdf_manifest.jsonl
```

Seed PDF sources include DOL FMLA employee/employer guides, OPM telework guidance, FLSA worker-rights poster, and OSHA workplace safety poster.

## Tools

| Tool | Purpose |
|---|---|
| `deterministic_input_guard` | Blocks unsafe input and sensitive employee data. |
| `llm_plain_text_classifier` | Confirms HR-policy intent. |
| `hybrid_hr_policy_search` | BM25 + dense retrieval over HR PDFs. |
| `leave_policy_lookup` | Retrieves FMLA/leave rules. |
| `wage_hour_lookup` | Retrieves wage-hour and worker-rights evidence. |
| `telework_policy_lookup` | Retrieves telework agreement and eligibility evidence. |
| `citation_validator` | Confirms cited evidence exists. |
| `human_review_queue` | Routes consequential HR cases for approval. |

## Guardrails and HITL

Human review is required for:

- hiring, firing, discipline, or accommodation decisions
- legal advice or employee-specific determinations
- sensitive employee PII
- conflicting policy evidence
- failed citation validation

The assistant can explain policy and draft HR responses, but it must not make personnel decisions.

## Evaluation

Evaluate retrieval recall@k separately from HR answer quality. Add red-team cases for employee PII, retaliation, illegal discrimination requests, legal advice, and policy-bypass attempts.
