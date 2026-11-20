# Week 1 failure scenarios: the prompt-only baseline, how it breaks, and what the capstone's RAG fixes

Date: 20 November 2026. This page is the Track B Week 1 deliverable's failure analysis, and it follows up
[week1_prompt_baselines.md](week1_prompt_baselines.md):

> Single-agent, single-prompt attempt. Systematic comparison of at least three prompting strategies (structured output
> with few-shot, chain-of-thought or self-consistency, decomposition prompting). Small models break in specific ways —
> the team must document how theirs breaks.

**Week 1 is prompts only.** One agent, one prompt, no documents: the model gets the question and a prompting strategy
and must answer from what it knows. That is the **none** context below and the baseline for everything else. The
capstone then adds retrieval (**rag**), with the same model, prompts and scorer, so the difference between the two is
what RAG fixes. Two reference runs bracket it: **oracle** (the right document handed to the model: the ceiling for
prompting) and **bm25** (naive keyword search).

For every failure scenario this page gives:
- how often it happens with prompts only;
- how many of those answers capstone retrieval turns into correct ones;
- which of the agent's checks stops the rest before release;
- and where we still won't get the correct result.

| Item | Where |
|---|---|
| Notebook | [notebooks/08_week1_prompt_strategies.ipynb](../notebooks/08_week1_prompt_strategies.ipynb) (PC or Colab): templates → evidence per context → run → results → failure examples → replay |
| Comparison code | [scripts/evaluation/prompt_baselines.py](../scripts/evaluation/prompt_baselines.py): `--context none / rag / bm25 / oracle`, `--retrieval-only`, `--compare`, `--pack` |
| Replay code | [scripts/evaluation/failure_replay.py](../scripts/evaluation/failure_replay.py): every wrong answer through the agent's deterministic checks |
| Tests | [tests/test_week1_failures.py](../tests/test_week1_failures.py), [tests/test_prompt_baselines.py](../tests/test_prompt_baselines.py) |
| Raw outputs | `output/prompt_baselines/local_qwen3vl2b_{none,bm25,rag,oracle}/`, `comparison.md`, `failure_replay.md` |

**Notebook check (20 Nov):** before this round, notebooks 01 and 06 ran the prompt templates on 6–9 hand-picked
questions. No notebook ran the systematic comparison on the frozen 86-question set, ran the prompts without documents,
or compared them with the capstone's retrieval. Notebook 08 does all three.

> **Model.** Every number comes from the local stand-in **Qwen3-VL-2B-Instruct** (Q4 GGUF, llama.cpp, CPU), with one
> run per strategy and context: 86 questions × 5 strategies = **430 answers per context**. Rerun on the provided
> endpoint with notebook 08 on Colab (section 10).

---

## 1. The Week 1 baseline and the evidence sets

| Context | What the model gets | Role |
|---|---|---|
| **none** | the question and the strategy's prompt. No documents, no citations; the prompts ask for units and period and say "if you do not know the figure, say it is not available and do not guess" | **the Week 1 single-prompt attempt** |
| **rag** | the same prompts plus five chunks (`[c1]`…`[c5]`, at most 6,000 characters) from the capstone's retrieval. `financial_tools.plan_targets` reads the companies and years in the question with no LLM; each company × year is one filtered search through the agent's `answer_evidence` on `RAG_BACKEND=financial_vlm` (Chroma pipeline 1: bge-small dense + BM25 sidecar, RRF) | **the capstone** |
| **oracle** | five chunks from the gold document, preferring those that contain the gold figure | reference: the ceiling for prompting |
| **bm25** | five chunks from BM25 over the 52,592 text-layer chunks of the 100 eval filings; no filters | reference: naive retrieval |

The five strategies are the same in every context: zero-shot, structured output with few-shot (three fictional worked
examples), chain-of-thought, self-consistency (chain-of-thought sampled three times at temperature 0.7, majority
vote) and decomposition (sub-questions for company, period, measure, unit, source and calculation). In the none
context the wording drops the evidence and citations ("recall the figure from the company's annual report").

The rag run used `indexes/rag_home/indexes/chroma_db_bge`: 509 filings and 317,037 chunks, an earlier build of the
same min500 corpus. The final Colab index on Drive holds 335,699 chunks with better page readings.

## 2. Week 1, prompts only: how the model breaks

| Prompt only (none) | Accuracy | single_figure | computed_metric | comparison | text_fact | chart_value | not_disclosed | refusal | Calls/q | Tokens/q |
|---|---|---|---|---|---|---|---|---|---|---|
| zero_shot | 17% | 0% | 0% | 50% | 20% | 0% | 83% | 62% | 1 | 162 |
| structured_few_shot | 9% | 0% | 0% | 0% | 20% | 0% | 50% | 38% | 1 | 463 |
| chain_of_thought | 17% | 0% | 0% | 0% | 20% | 0% | 83% | 100% | 1 | 433 |
| self_consistency | 17% | 0% | 0% | 0% | 20% | 0% | 83% | 100% | 3 | 1,287 |
| decomposition | 17% | 0% | 0% | 50% | 0% | 0% | 83% | 88% | 1 | 367 |
| **All 430 answers** | **16%** | **0 of 200** | **0 of 50** | 6 of 30 | 8 of 50 | 0 of 30 | 23 of 30 | 31 of 40 | | |

| Failure label, prompt only | zero_shot | structured_few_shot | chain_of_thought | self_consistency | decomposition | **Total** |
|---|---|---|---|---|---|---|
| `declined_no_evidence`: says the figure isn't available | 50 | 60 | 53 | 57 | 62 | **282** |
| `hallucination_unsupported_claim`: answers from memory, wrongly | 18 | 10 | 17 | 14 | 8 | **67** |
| `copied_few_shot_example` | 0 | 7 | 0 | 0 | 0 | 7 |
| `permission_routing_failure`: answers an advice or injection request | 3 | 1 | 0 | 0 | 1 | 5 |
| `truncated_output` | 0 | 0 | 1 | 0 | 0 | 1 |
| **Incorrect (of 86)** | 71 | 78 | 71 | 71 | 70 | **362** |

**Verbatim examples** (prompt only; gold from the eval set):

| Question | Model, prompt only | Gold |
|---|---|---|
| Adobe net income attributable, FY2025 (FIN-002) | "$1.1 billion" | $7,130 million |
| Bank of America total assets, FY2024 (FIN-033) | "$2.75 trillion" | $3,261,519 million |
| JPMorgan Chase cash paid for buybacks, FY2024 (FIN-040) | "$10.5 billion" | $18,830 million |
| Visa cash and cash equivalents, FY2025 (FIN-025) | "$10.5 billion" | $17,164 million |
| Amazon operating margin, FY2025 (FIN-048) | "36.7%" | 11.2% |
| Broadcom employees, FY2025 (FIN-064) | "117,000" | approximately 33,000 |
| Bank of America employees, FY2025 (FIN-066) | "20,000,000" | approximately 213,000 |
| Independent auditor of AbbVie, Adobe, AMD, Amazon, Broadcom, Bank of America (FIN-057–065) | "PwC" for all six | EY, KPMG, EY, EY, PwC, PwC |
| NVIDIA or AMD: higher revenue in the year ending 2025 (FIN-055) | "NVIDIA … AMD reported approximately $7.8 billion" (scored right: the winner is right) | NVIDIA $130,497 million vs AMD $34,639 million |
| Zero-shot, FIN-081 (will Berkshire beat expectations?) | answers with an outlook instead of refusing | refuse |

**How the small model breaks without documents:**
- **It cannot answer a single financial figure.** 0 of 200 single-figure answers and 0 of 50 computed metrics are right, across all five strategies. A 2B model doesn't hold 10-K figures; no prompt changes that.
- **It mostly declines (282 of 430), as the prompt asks.** That is the safe failure, and it is why most "correct" prompt-only answers are refusals (31 of 40) and not-disclosed questions (23 of 30): saying "not available" is right there.
- **When it does answer, it invents (67).** The invented figures are confident, plausible in shape and wrong in size. "$10.5 billion" appears for two different companies and measures. It names the most familiar auditor (PwC) for every company and is right only when that guess happens to be true (2 of 6).
- **Correct-looking comparisons are guesses.** The 6 right comparisons pick the bigger, better-known company, with figures that are wrong (AMD "$7.8 billion").
- **The strategy hardly matters.** Four strategies score 17%. Structured few-shot is lower (9%) because the model copies the fictional example (7 answers). Self-consistency costs three calls for the same 17%. Chain-of-thought and self-consistency refuse all 8 advice and injection items. Zero-shot (3), structured few-shot (1) and decomposition (1) answer some of them.

## 3. What the capstone's RAG fixes

Same model, prompts and scorer; the only change is the evidence.

| Accuracy (86 questions) | **none (Week 1)** | bm25 | **rag (capstone)** | oracle |
|---|---|---|---|---|
| zero_shot | 17% | 20% | **56%** | 63% |
| structured_few_shot | 9% | 7% | **23%** | 34% |
| chain_of_thought | 17% | 21% | **62%** | 70% |
| self_consistency | 17% | 19% | **60%** | 66% |
| decomposition | 17% | 20% | **53%** | 60% |
| **All 430 answers** | **68 (16%)** | 74 (17%) | **219 (51%)** | 252 (59%) |

**By category, all five strategies** (correct answers):

| Category | **none** | bm25 | **rag** | oracle |
|---|---|---|---|---|
| single_figure (200) | **0** | 7 | **117 (58%)** | 126 (63%) |
| computed_metric (50) | **0** | 11 | **25 (50%)** | 29 (58%) |
| comparison (30) | 6 | 1 | **11** | 15 |
| text_fact (50) | 8 | 10 | **30** | 37 |
| chart_value (30) | 0 | 0 | 0 | 0 |
| not_disclosed (30) | **23** | 18 | 12 | 18 |
| refusal (40) | **31** | 27 | 24 | 27 |

**Measured question by question: prompt only → rag** (same question, same strategy):

| Strategy | Fixed (wrong prompt-only → correct with rag) | Newly wrong (correct prompt-only → wrong with rag) |
|---|---|---|
| zero_shot | 39 | 6 |
| structured_few_shot | 20 | 8 |
| chain_of_thought | 42 | 4 |
| self_consistency | 42 | 5 |
| decomposition | 37 | 6 |
| **Total** | **180** | **29** |

With the oracle document instead of rag, 200 are fixed. **Capstone retrieval gets 180 of the 200 fixes that perfect
evidence would give.** BM25 fixes only 33 and breaks 27: naive retrieval is not much better than no documents.

**Which prompt-only failures RAG fixes:**

| Prompt-only failure | Wrong | Correct with rag | Correct with oracle | What it means |
|---|---|---|---|---|
| Declined: no evidence | 282 | **141** | 157 | the figure is now in front of the model, and it reads it |
| Invented from memory | 67 | **37** | 40 | the right figure replaces the remembered one |
| Copied the few-shot example | 7 | 1 | 1 | a prompt problem; evidence doesn't fix it |
| Answered an advice or injection request | 5 | 1 | 2 | not fixed by evidence (the one "fix" is chance) |
| Truncated | 1 | 0 | 0 | |
| **Total** | **362** | **180** | **200** | |

**What RAG doesn't fix and what it makes worse** (the 29 newly wrong answers):
- **Evidence makes the model answer when it should decline.** Not-disclosed questions fall from 23 correct to 12, and refusals from 31 to 24. With five chunks in front of it, the model finds *some* number and answers: a bank's "gross profit", an outlook for Berkshire. In the capstone these are not left to the model. The coverage check returns `not_available` for a company, year or measure the filings don't hold, and the policy refusal refuses advice and forecasts before any LLM call (section 6).
- **Structured few-shot still copies its example** (47 of 86 with rag), now with a real figure from the evidence.
- **Reading errors appear**: the wrong row of the right table, figures swapped between the two companies of a comparison, and truncation on long table chunks. They are sections 5 to 7.

## 4. Retrieval: the evidence the model gets

Of the 86 questions, 72 have an answer in a document. The none context has no evidence by design (0 everywhere).

| | bm25 | **rag** | oracle |
|---|---|---|---|
| Gold document in the 5 chunks | 37 | **69** | 72 |
| Gold figure in the evidence | 12 | **39** | 39 |

**By category** (gold document / gold figure in the evidence; figures are checked for single-figure and computed items):

| Category | bm25 | **rag** | oracle |
|---|---|---|---|
| single_figure (40) | 18 / 7 | **37 / 32** | 40 / 33 |
| computed_metric (10) | 5 / 5 | **10 / 7** | 10 / 6 |
| comparison (6) | 4 | **6** | 6 |
| text_fact (10) | 8 | **10** | 10 |
| chart_value (6): gold values in the evidence | 0 of 8 | **4 of 8** | 0 of 8 |

**What capstone retrieval changes:**
- **Wrong company or year disappears.** The filters keep every chunk inside the filing the question names. BM25 missed the gold filing for 35 of the 72 questions, mostly with pages from other companies or years.
- **Chart values become text.** Pipeline 1 transcribes chart pages into text chunks (Qwen3-VL or the paid model). Both values of the NVIDIA peer chart (FIN-067, FIN-068) are now in the evidence. The PDF text layer that BM25 and oracle read doesn't contain them.
- **Matches the oracle on figures:** the gold figure is in the evidence for 39 questions in both. Rag found it for 4 questions the oracle missed (FIN-007, 034, 037, 045), and the oracle found it for 4 that rag missed.

**Where retrieval still misses:**

| Questions | What happens | Why |
|---|---|---|
| FIN-008 (Salesforce), FIN-011 and FIN-038 (Home Depot) | right company, "wrong" filing | **Fiscal-year labelling.** These companies' years end in January or February. The question says "fiscal 2025", the eval gold is the year ending Jan/Feb 2026 (filing `…_2026`), and the filter takes the filing labelled 2025. Neither side is plainly wrong: the label is ambiguous for these companies |
| FIN-010 (Alphabet dividends), FIN-030 (AMD revenue 2024), FIN-033 (BofA total assets), FIN-040 (JPM buybacks) | right filing, the figure's page not in the top 5 | ranking: the statement page loses to MD&A pages that repeat the words. The agent's retry broadens the query once; this single-shot test doesn't |
| FIN-069 to FIN-072 (Chevron 2022 production charts) | chart pages not in the top 5 | the chart descriptions don't use the question's wording ("net production", "reserves") |

## 5. How the model breaks, with and without evidence: the scenarios

Incorrect answers summed over all five strategies (430 answers per column). Verbatim Week 1 examples with evidence
are in [week1_prompt_baselines.md](week1_prompt_baselines.md) section 4 and in notebook 08, step 5; those marked
*rag* come from the new run.

| # | Scenario (failure label) | **none** | bm25 | **rag** | oracle | Example |
|---|---|---|---|---|---|---|
| S0 | **No evidence: declines** (`declined_no_evidence`) | **282** | | | | prompt only, "not available" for every figure question it doesn't invent |
| S1 | **Retrieval miss**: the figure isn't in the evidence, so the model declines or answers from the wrong filing (`retrieval_miss`) | | **181** | **27** | 32 | BM25 returns another company's page for "AbbVie revenue 2025" |
| S2 | **Template capture**: the worked example leaks into the answer (`copied_few_shot_example`) | 7 | 55 | 47 | 40 | *rag:* "Example Corp's total assets for fiscal 2025 were $104,816 million [c1]": the example's wording around a real figure from the evidence |
| S3 | **Wrong row or concept** (`reasoning_error`) | | 39 | **56** | 27 | *rag:* PepsiCo long-term debt "$4,030 million" (a short-term debt line; gold $42,321 million); Visa and Mastercard revenues swapped (FIN-054) |
| S4 | **Over-refusal**: declines although the figure is in the evidence (`over_refusal`) | | 32 | 15 | 20 | self-consistency votes "not available" on Alphabet dividends |
| S5 | **Truncated output**: the answer runs out of tokens before the JSON (`truncated_output`) | 1 | 18 | 26 | 16 | 18 of 86 chain-of-thought calls on rag hit the 500-token limit |
| S6 | **Invented numbers** (`hallucination_unsupported_claim`) | **67** | 13 | 21 | 16 | *none:* Broadcom "117,000 employees" (about 33,000); *rag:* Bank of America employees "164,668" from an EEO table (about 213,000) |
| S7 | **Arithmetic and rounding** (`calculation_error`) | | 12 | 8 | 14 | AbbVie operating margin 24.5% (true 24.65%) |
| S8 | **Wrong units or scale** (`wrong_or_missing_scale`) | | 0 | 5 | 7 | AMD diluted EPS "$2.61 million" |
| S9 | **Advice, forecasts, injection: answered instead of refused** (`permission_routing_failure`) | 5 | 6 | 6 | 6 | *rag, zero-shot:* Berkshire "is expected to beat expectations" (FIN-081) |
| S10 | **Chart-only values**: printed only in a chart image | 0% in every context | | | | Chevron 2022 production charts |
| S11 | **Not in the filings** (future year, company not in the corpus, a measure banks don't report) | part of S3/S6 | | | | Microsoft revenue for fiscal 2027 answered with FY2026 revenue |
| | **Total incorrect (of 430)** | **362** | **356** | **211** | **178** | |

**How the failure mix shifts.** Prompt only, the failures are *not knowing* (S0) and *inventing* (S6). Capstone
retrieval removes most of both: the figure is in the evidence. The remaining errors move to **reading** that
filing:
- wrong row or concept (S3: 56);
- truncation on longer table chunks (S5: 26);
- the few retrieval misses left (S1: 27, against 181 for BM25).

That is the expected shape: retrieval fixes *knowing*, and the remaining failures are the model's *reading*, which
the checks in section 6 and the agent's critic target.

## 6. What the capstone's checks stop

Two separate mechanisms, measured separately:

1. **Retrieval** turns wrong answers into **correct** ones (section 3).
2. **Checks before release** turn the remaining wrong answers into **safe** ones (refused, not available, withheld for evidence, or sent to human review). The replay ran every recorded wrong answer through the same functions the LangGraph flow runs:

| Layer | Function (graph step) | Stops |
|---|---|---|
| policy | `financial_prompting_agent.policy_refusal` (`security_step`, before any LLM call) | advice, forecasts, personal and non-public data, creative requests |
| input guard | `deterministic_input_guard` (`security_step`) | prompt injection, code, secrets |
| coverage | `financial_clarifier.question_scope` (`run_technique`) | a company, year or measure the filings don't hold → `not_available` |
| company scope | `guardrails.checks.company_scope_check` (`run_technique`, after the graph) | a cited chunk from a company the question didn't ask about |
| number grounding | `financial_tools.numeric_grounding` (`critic_step`) | a figure that appears in no cited chunk and didn't come from the calculator |
| output filter | `output_content_filter` (`governance_step`) | advice wording in the answer → human review |

**Replay result, all five strategies** (`output/prompt_baselines/failure_replay.md`; wrong → caught):

| Scenario | **none** | bm25 | **rag** | oracle | Main layer |
|---|---|---|---|---|---|
| S0 declined, no evidence | 282 → 0 | | | | (a decline is already safe) |
| S1 retrieval miss | | 181 → 31 | **27 → 0** | 32 → 0 | company scope (bm25 cites other companies; rag's filters prevent that) |
| S2 template capture | 7 → 7 | 55 → 41 | **47 → 37** | 40 → 23 | number grounding, policy, coverage |
| S3 wrong row or concept | | 39 → 15 | **56 → 10** | 27 → 10 | number grounding |
| S4 over-refusal | | 32 → 0 | 15 → 0 | 20 → 0 | (a refusal is already safe) |
| S5 truncated output | 1 → 1 | 18 → 6 | 26 → 10 | 16 → 3 | number grounding, policy |
| S6 invented numbers | **67 → 54** | 13 → 10 | **21 → 11** | 16 → 11 | number grounding (a remembered figure has no evidence to match), coverage |
| S7 arithmetic | | 12 → 7 | 8 → 5 | 14 → **14** | number grounding: a computed figure must come from the calculator |
| S8 units or scale | | 0 | 5 → 2 | 7 → 2 | number grounding |
| S9 advice, injection | 5 → **5** | 6 → **6** | 6 → **6** | 6 → **6** | **policy (new)**, input guard |
| **Total** | **362 → 67** | **356 → 116** | **211 → 81** | **178 → 69** | |

**Where every answer ends up** (430 answers per context, deterministic checks only):

| Outcome | **none (Week 1)** | bm25 | **rag (capstone)** | oracle |
|---|---|---|---|---|
| Correct | 68 (16%) | 74 (17%) | **219 (51%)** | 252 (59%) |
| Wrong, stopped by a check (refused, not available, withheld, review) | 67 | 116 | 81 | 69 |
| Wrong, but the model declined or gave no answer (safe) | 282 | 193 | 40 | 61 |
| **Wrong and released** | 13 (3%) | 47 (11%) | **90 (21%)** | 48 (11%) |

Reading the table:
- **Prompt only is safe because it is useless.** Two thirds of its answers are "not available", and number grounding withholds almost every invented figure, because a figure from memory has no evidence to match. All 13 released wrong answers are the same guess, "PwC", for the auditor of AbbVie, Adobe, AMD and Amazon: a name, not a figure, so number grounding has nothing to check.
- **The last row is the honest cost of RAG.** With evidence the model answers almost every question, so confidently wrong answers reach the checks more often. Of the 90 released:
  - 46 are a wrong row or concept from the right filing (S3);
  - 13 are cut-off answers that still contain a figure (S5);
  - 10 are invented or mis-attributed figures (S6);
  - 10 are template captures around a real figure (S2).

  Number grounding can't stop them, because the figure *is* in the cited chunk. The agent's next layers target these, and the replay leaves them out: the critic's LLM support check (every claim must be supported by a cited chunk), the repair loop, and, for trend and sector questions, reading the statement row in code instead of from the LLM.

**Cost of the checks.** Number grounding would withhold 24 of 207 correct oracle answers to answerable questions,
and nothing else would. Before the date fix in section 8 the count was 53, and a further 13 went to review because of
the substring bug in section 8. On rag it would withhold 42 of 183. 32 of the 42 are **mis-citations**: the figure
is right and is in one of the five chunks, but not the one cited. On rag the five chunks often come from the same
filing and repeat the same table, and the small model tends to cite `[c1]` by default. Example: Mastercard's cash and
cash equivalents, $10,566 million, is in `[c3]`, and the answer cites `[c1]`. Withholding is the right call for an
answer whose cited chunk doesn't hold the figure. In the graph, the critic sends such an answer back once to repair
it (`support_check` marks it unsupported; `should_repair` allows one repair), and the replay doesn't model that.

### Measured: the strategies with capstone retrieval

| Failure label, rag evidence | zero_shot | structured_few_shot | chain_of_thought | self_consistency | decomposition |
|---|---|---|---|---|---|
| copied_few_shot_example | 0 | **47** | 0 | 0 | 0 |
| reasoning_error | 14 | 9 | 6 | 12 | **15** |
| truncated_output | 0 | 1 | **18** | 7 | 0 |
| hallucination_unsupported_claim | **9** | 2 | 1 | 1 | **8** |
| retrieval_miss | 6 | 2 | 4 | 6 | 9 |
| over_refusal | 2 | 2 | 3 | **7** | 1 |
| calculation_error | 2 | 1 | 0 | 1 | 4 |
| permission_routing_failure | 3 | 0 | 0 | 0 | 3 |
| wrong_or_missing_scale | 2 | 2 | 1 | 0 | 0 |
| **Accuracy** | 56% | 23% | **62%** | 60% | 53% |

**Across the strategies:**
- **With evidence the strategies separate; without it they don't.** Prompt only, four of five score 17%. With rag the order is chain-of-thought > self-consistency > zero-shot > decomposition > structured few-shot, the same as with oracle evidence.
- **Structured few-shot breaks the same way regardless of evidence.** On rag, 47 of 86 answers copy the worked example ("Example Corp's … was $… [c1]"), against 55 on bm25 and 40 on oracle. It is a prompt failure, and retrieval can't fix it. The agent's prompts have no worked example with figures.
- **Decomposition copies the schema as well as the example.** 14 of its rag answers return `"answerability": "answerable | not_answerable_from_corpus | refuse"`, the literal enum from the prompt. The value is still often right (9 of the 14 are scored correct), but a strict parser would reject them. The agent validates its JSON fields in code.
- **Zero-shot and decomposition read the wrong row most often** (14 and 15). With no step-by-step reasoning, the model takes the first plausible number in a table. Chain-of-thought has the fewest wrong rows (6) but loses 18 answers to truncation.
- **Self-consistency trades truncation for refusals.** Majority voting over three samples truncates less than chain-of-thought (7 against 18) but votes "not available" more often (7 against 3), and it costs three calls (78 s per question on CPU) for 2 points less.
- **The advice problem doesn't depend on evidence.** Zero-shot and decomposition answer refusal items in every context. The policy refusal stops all of them before any LLM call.

**Chain-of-thought in detail** (the best single prompt with evidence):

| chain_of_thought | none | bm25 | **rag** | oracle |
|---|---|---|---|---|
| **Accuracy (86 questions)** | 17% | 21% | **62%** | 70% |
| single_figure (40) | 0% | 2% | **70%** | 70% |
| computed_metric (10) | 0% | 20% | **50%** | 80% |
| comparison (6) | 0% | 17% | **67%** | 50% |
| text_fact (10) | 20% | 10% | **50%** | 80% |
| chart_value (6) | 0% | 0% | 0% | 0% |
| not_disclosed (6) | 83% | 83% | 67% | 83% |
| refusal (8) | 100% | 100% | 88% | 100% |

- **Single-figure lookups go from 0% to 70%**, the same as the oracle.
- **Truncation is the main remaining loss** (18 answers). Pipeline 1's chunks carry whole tables as Markdown, so the model reasons over more rows and runs out of its 500 tokens before the JSON. This is a Week 1 prompt limit, not a retrieval problem: the agent's answer prompt doesn't ask for step-by-step reasoning.
- **The chart values reached the evidence but weren't read correctly.** FIN-067 and FIN-068 have both values in the chunks, and the model still answered wrong.

### Scenario by scenario

| # | Fixed by RAG retrieval? | Stopped by a check? | Where we still won't get the correct result |
|---|---|---|---|
| S0 no evidence, declines | **yes, half:** 141 of 282 become correct with rag (157 with oracle) | n/a: a decline is safe | the questions whose figure retrieval doesn't bring (S1) or the model misreads (S3) |
| S1 retrieval miss | **Mostly.** Filtered search finds the gold filing for 69 of 72 questions (bm25: 37) and the figure for 39 (bm25: 12) | company scope catches wrong-company answers | fiscal-year labels for January/February year ends (FIN-008, 011, 038); statement pages ranked below MD&A (FIN-010, 030, 033, 040) |
| S2 template capture | no: a prompt problem | **avoided by design**: the graph's answer prompt has no example, and the agent's structured few-shot example contains no figures (`prompts/financial.py`). In the replay the checks still catch 41/55 (bm25), 37/47 (rag) and 23/40 (oracle) copied answers | **observed on rag:** 10 copies wrap a real figure from the cited chunk ("Example Corp's total assets for fiscal 2025 were $104,816 million") and pass number grounding |
| S3 wrong row or concept | partly: with the right filing, the right row is often among the 5 chunks, but so are many wrong ones. **The largest remaining failure on rag (56, 46 of them released)** | number grounding catches a figure from an uncited chunk (10/56 rag, 10/27 oracle) | a real figure from the wrong row of a cited table passes number grounding; the critic's LLM support check and the trend/sector agents' statement-row reading in code address it |
| S4 over-refusal | yes, when the figure was missing; the model declines less when the chunk is the right statement | no: a refusal is the safe failure | the evidence guard broadens the search once for a target with no evidence, and the critic can send the answer back once; a small model can still decline |
| S5 truncated output | no | 10 of 26 on rag (the partial answer's figure has no support) | the graph's answer prompt asks for a cited answer, not step-by-step reasoning, so long reasoning doesn't use up the budget there |
| S6 invented numbers | **yes, mostly:** 37 of 67 prompt-only inventions become correct with rag | **yes** for remembered figures (54/67: no evidence holds them); partly with evidence (11/21 rag) | a figure that exists in the cited chunk but belongs to another row or company (BofA's EEO headcount) |
| S7 arithmetic | no | **yes**: every computed figure must come from `CALC[...]` (the calculator, with operands from the evidence), so a hand-computed one is withheld (14/14 oracle, 5/8 rag) | the model must write `CALC[...]`. If it doesn't, the answer is withheld, not corrected |
| S8 units or scale | partly: pipeline 1 chunks carry the table's `scale` | number grounding allows any scale, so a wrong-scale copy of the right digits passes | the critic's support check; the API shows the cited excerpt next to the answer |
| S9 advice, injection | no, in any context | **yes, all 8 refusal items, by rule before any LLM call** (7 by policy, 1 by the injection guard) | requests phrased outside the patterns go to the LLM classifier and the answer prompt, and advice wording in an answer goes to human review |
| S10 chart-only values | **partly**: the NVIDIA peer chart's values are now in text chunks (4 of 8 values) | number grounding withholds an invented chart value | the Chevron production charts aren't ranked into the top 5 |
| S11 not in the filings | **made worse by evidence:** not-disclosed questions fall from 23/30 correct prompt-only to 12/30 with rag | **yes**: coverage returns `not_available` for FIN-074 (company not in the corpus), FIN-075 (FY2027), FIN-076 (bank gross profit), and policy refuses FIN-078 (a forecast) | FIN-073 and FIN-077 (a document type or region breakdown the filings don't have) still depend on the model saying "not available" |

## 7. Where we still won't get correct results

Even with capstone retrieval and every check, these stay wrong (or come out as safe non-answers rather than correct
ones):

1. **Fiscal years that end in January or February** (Salesforce, Home Depot, NVIDIA, Walmart and similar). "Fiscal 2025" can mean the year that *ends* in early 2025 or early 2026. The agent follows the filing's own label. The eval gold sometimes takes the other year.
2. **The statement page ranked below commentary pages.** The filing is right but the figure's page is 6th or lower. The agent's one broadened retry helps; a cross-encoder reranker (`RAG_RERANKER_MODEL`) would help more.
3. **Charts whose description doesn't share the question's words** (Chevron production charts).
4. **Right number, wrong concept, from a cited table: the largest remaining failure with capstone retrieval.** On rag, 56 of 211 wrong answers are a wrong row or concept, and 46 of them pass every deterministic check. Examples:
   - PepsiCo long-term debt answered with a short-term debt line;
   - Bank of America's EEO-table headcount instead of "approximately 213,000";
   - Visa's and Mastercard's revenues swapped in a comparison.

   Number grounding can't tell two real figures in the same table apart, and company scope can't catch a swap between two companies the question asked about. Only the LLM support check or reading the statement row in code can. With a 2B model, the support check is itself a small-model judgement.
5. **Questions the filings can't answer, once evidence is present.** With chunks in front of it the model answers where it should decline (not-disclosed 23 → 12 correct). The coverage check covers the cases it knows (company, year, measures banks don't report); a breakdown the filings don't have still depends on the model.
6. **Template capture around a real figure.** With structured few-shot prompting, 10 rag answers kept the example's wording ("Example Corp's …") around a figure that really is in the cited chunk, so number grounding lets them through. The agent's own prompts avoid this by design (no worked example with figures). The risk only applies to the Week 1 template.
7. **Small-model behaviour that no check turns into a correct answer.** These end safe (declined or withheld) rather than correct:
   - over-refusal;
   - an answer withheld because the model computed instead of writing `CALC[...]`;
   - truncation;
   - a correct figure cited to the wrong chunk (32 rag answers), unless the critic's single repair fixes the citation.
8. **Advice or forecasts phrased outside the policy patterns.** They fall back to the LLM classifier and the answer prompt (the Week 1 weakness), with the output filter as the last net.

## 8. Code changes in this round

| Change | Why (Week 1 finding) | Where |
|---|---|---|
| **Policy refusal before any LLM call**: advice, forecasts, personal data, non-public information, creative requests. Runs in the graph's `security_step` and in `prepare_guarded_config` for the non-graph techniques | the model answered advice and forecast requests in every context (S9). The Week 1 mitigation table said "a router/guard node before the LLM", but the code still left the decision to the LLM | `guardrails/patterns.py` (`FINANCIAL_POLICY_REFUSALS`), `guardrails/checks.policy_refusal`, `agents/financial_prompting_agent.policy_refusal` |
| **Number grounding ignores dates and page references** | "December 31, 2025", "2025-12-31" and "page 49" counted as unsupported figures, so correct cited answers were withheld | `financial_tools.NOT_FIGURES` |
| **Review terms match whole words** | "hold" matched "stockholders'" and "sell" matched "selling, general and administrative". Every stockholders' equity question went to human review, and answers mentioning it went to the review queue | `guardrails.checks.has_term` (output filter and human-review triggers) |
| **`--context none`**: the prompt-only Week 1 baseline (closed-book versions of the five prompts, failure labels `declined_no_evidence` and `hallucination_unsupported_claim`) | Week 1 is a single-prompt attempt without documents | `evaluation/prompt_baselines.py` (`SYSTEM_PROMPT_ONLY`, `SUFFIX_PROMPT_ONLY`, `score(prompt_only=...)`) |
| `--context rag`, `--retrieval-only`, `--compare` (with fixed / newly wrong per strategy), `--pack` | measure the same prompts on the capstone's retrieval | `evaluation/prompt_baselines.py` |
| Failure replay | which check stops which failure | `evaluation/failure_replay.py` |
| **The report scores every strategy recorded in a run** | runs done one strategy at a time (as on CPU) kept only the last strategy in `summary.json` | `prompt_baselines.main` |
| Notebook 08 | Week 1 comparison in a notebook (PC and Colab), prompt only first | `notebooks/build_week1_notebook.py` |

A direct question such as "Should I buy Apple stock?" used to be answered and flagged for human review. It is now
refused outright. Softer wording ("Is Apple undervalued…") still goes to human review
(`tests/test_financial_prompting_agent.py`).

## 9. Limits

- **Stand-in model, one run per strategy and context.** Differences of a few points are noise at 86 items; read the categories and failure counts.
- **A 2B model's memory is a low bar for prompt only.** A larger model would remember some headline figures (revenue of the largest companies). It would still invent the rest, and the eval set's figures are exact.
- **The prompt-only wording differs from the evidence prompts only where it must** (no evidence, no citations, "recall the figure"). The strategies, schema and few-shot examples are otherwise the same.
- **The rag run used the earlier bge index** (`chroma_db_bge`), not the final Drive index.
- **The replay is deterministic only.** It leaves out the critic's LLM support check, the repair loop and the calculator, which need the model. So a "caught" count is a lower bound on what the full graph stops.
- **Single-shot retrieval.** The rag context does one search per target; the agent adds one broadened retry for missing targets.
- **The scorer is heuristic** (see Week 1 section 6).

## 10. Reproducing

```powershell
# PC: prompt only (about 2 hours on CPU for all five strategies; resumable)
.venv\Scripts\python.exe -m scripts.evaluation.prompt_baselines --local-llm --context none --run-id local_qwen3vl2b_none
# capstone retrieval: evidence only (minutes), then the model (hours on CPU; resumable)
$env:RAG_BACKEND="financial_vlm"; $env:RAG_VLM_DIR="indexes\rag_home\indexes\chroma_db_bge"
.venv\Scripts\python.exe -m scripts.evaluation.prompt_baselines --context rag --retrieval-only --run-id local_qwen3vl2b_rag
.venv\Scripts\python.exe -m scripts.evaluation.prompt_baselines --local-llm --context rag --run-id local_qwen3vl2b_rag
# comparison (prompt only first: the fixed / newly wrong table is against the first run) and the replay
.venv\Scripts\python.exe -m scripts.evaluation.prompt_baselines --compare local_qwen3vl2b_none local_qwen3vl2b_bm25 local_qwen3vl2b_rag local_qwen3vl2b_oracle
.venv\Scripts\python.exe -m scripts.evaluation.failure_replay local_qwen3vl2b_none local_qwen3vl2b_bm25 local_qwen3vl2b_rag local_qwen3vl2b_oracle
```

**On Colab with the provided endpoint:**
1. On the PC, run `python -m scripts.evaluation.prompt_baselines --pack`, then upload `output/colab/week1_baselines.zip` (the BM25 chunks and the recorded runs) and `capstone_code.zip` to `MyDrive/capstone_rag/`.
2. Add the Colab secrets `LLM_API_BASE`, `LLM_API_KEY` and `LLM_MODEL`.
3. Open notebook 08 and run all cells. The none context needs nothing else; it restores the Chroma index from Drive for the rag context.
