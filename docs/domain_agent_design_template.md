# Domain Agent Design Template

Use this structure for each non-financial Track B domain. Each domain follows the same core implementation pattern as the financial-report agent:

```text
security guardrail
-> plain-text LLM classifier
-> planner/router
-> hybrid RAG retrieval
-> evidence prompt-injection guard
-> answer/drafting agent
-> citation critic
-> human-review router when required
-> final cited answer or refusal
```

The implementation can share the financial agent's guardrail, citation, HITL, and Streamlit patterns while swapping domain-specific tools, source filters, and review triggers.
