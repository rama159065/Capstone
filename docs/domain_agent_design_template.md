# Domain Agent Design Template

Use this structure for each non-financial Track B domain. Each domain follows the same core implementation pattern as the financial-report agent:

```text
security guardrail
-> plain-text LLM classifier
-> planner/router
-> hybrid RAG retrieval
-> evidence prompt-injection guard
-> answer/drafting agent
-> output content filter
-> citation critic
-> human-review router when required
-> final cited answer or refusal
```

The implementation can share the financial agent's guardrail, citation, HITL, and Streamlit patterns while swapping domain-specific tools, source filters, and review triggers.

## Content Filtering Pattern

Each domain uses layered content filtering:

- **Input filter:** detects prompt injection, executable/tool payloads, secrets, credentials, card-like numbers, SSNs, emails, and oversized/obfuscated content before retrieval.
- **Plain-text/domain classifier:** checks that the message is ordinary plain text and belongs to the selected domain.
- **Evidence filter:** treats retrieved text as untrusted and blocks retrieved prompt-injection or tool-control instructions before answer generation.
- **Output filter:** scans the generated answer for prompt-injection echo, executable content, leaked secrets, sensitive data, and domain-sensitive recommendations.
- **HITL router:** sends review-level content to a human reviewer instead of auto-releasing it.

Output filtering decisions are:

| Decision | Meaning |
|---|---|
| `allow` | No content-filter risk labels found. |
| `review` | Domain-sensitive or personal/sensitive data was detected; route to HITL if enabled. |
| `block` | Generated content includes prompt-injection echo, executable/tool payloads, or secrets; refuse before release. |
