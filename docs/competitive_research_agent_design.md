# Competitive-Research Analyst Agent Design

## Agent Idea

Build a **Competitive-Research Analyst** that compares companies using annual reports, SEC metadata, risk factors, business descriptions, and financial narrative evidence.

Example questions:

- Compare Apple and Microsoft risk language about revenue concentration.
- Which competitor discusses AI infrastructure investment?
- Summarize competitive threats mentioned by NVIDIA and Amazon.
- Draft a cited competitive brief from annual-report evidence.

## Core Agents

| Agent | Role |
|---|---|
| Security Guardrail Agent | Blocks prompt injection, market-manipulation requests, secrets, and attempts to bypass citation rules. |
| Plain-Text Validation Agent | Confirms plain-text competitive-research intent. |
| Planner / Router | Routes to company comparison, risk-factor search, product/market search, or briefing draft. |
| Retrieval Agent | Runs hybrid retrieval over competitor annual-report PDFs and SEC-derived JSON records. |
| Competitive Evidence Agent | Groups evidence by company, year, page, section, and chunk ID. |
| Comparison Agent | Produces side-by-side summaries grounded in retrieved evidence. |
| Citation Critic | Verifies citations and flags unsupported market claims. |
| Human Review Agent | Requires review before investment advice, external publication, or market-sensitive claims. |

## RAG Corpus

PDFs are downloaded under:

```text
data/raw/competitive_research/pdfs/
data/raw/competitive_research/pdf_manifest.jsonl
```

Seed PDF sources include 2024 annual reports for Apple, Microsoft, Amazon, NVIDIA, and Alphabet.

## Tools

| Tool | Purpose |
|---|---|
| `deterministic_input_guard` | Blocks unsafe or market-manipulative input. |
| `llm_plain_text_classifier` | Confirms competitive-research intent. |
| `hybrid_competitive_search` | BM25 + dense retrieval over annual-report PDFs. |
| `company_compare` | Groups and compares retrieved chunks by ticker/year. |
| `risk_factor_lookup` | Retrieves competitive, market, revenue, supply-chain, and product-risk language. |
| `output_content_filter` | Blocks generated secrets/tool payloads and routes investment-sensitive outputs to review. |
| `citation_validator` | Verifies every factual claim has evidence. |
| `human_review_queue` | Routes investment-sensitive briefs for review. |

## Guardrails and HITL

Human review is required for:

- buy/sell/hold or valuation language
- externally published competitive briefs
- market-moving claims
- unsupported competitor allegations
- conflicting annual-report evidence
- output content filter review for investment, publication, or sensitive-data language

The system may draft competitive summaries but must not give investment advice.

## Evaluation

Measure retrieval recall@k by gold relevant company/year/chunk IDs. Evaluate comparison quality, citation coverage, company grouping accuracy, and investment-advice refusals separately.
