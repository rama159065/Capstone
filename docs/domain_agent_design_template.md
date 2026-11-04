# Domain Agent Design Template

Use this structure for each non-financial Track B domain. Each domain follows the same core implementation pattern as the financial-report agent and is orchestrated by LangGraph in `scripts/agents/rag_agent.py`:

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

The shared implementation compiles a LangGraph state machine with explicit nodes for `security_guard`, `plain_text_classifier`, `planner`, `hybrid_retriever`, `evidence_guard`, `answer_agent`, `output_filter`, `citation_critic`, and `governance`. Each domain swaps domain-specific config, source filters, intent terms, and review triggers.

## Why This Is Agentic RAG

These domain agents are **Agentic RAG** rather than simple RAG because retrieval is treated as a controlled tool call inside a larger reasoning and governance graph. The system does not directly answer from the top-k chunks. It first checks the request, classifies intent, plans the retrieval action, combines dense and BM25 results, inspects evidence for prompt injection, generates a cited answer, filters the generated output, validates citations, and decides whether to release, refuse, or route to HITL.

The agentic behavior is intentionally bounded: each node has a narrow responsibility, tool inputs are structured, retrieved content is treated as untrusted evidence, and governance can override the answer when confidence, safety, citation, or domain-risk conditions require review.

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
