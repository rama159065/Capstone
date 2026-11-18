# Chart reading: what works, what it costs, what to decide

Date: 12 November 2026. **Status: decision pending**, to be revisited after the mentor discussion (section 8). Nothing in sections 5 and 6 is built yet unless marked as built.

## 1. Summary

- **Most chart numbers in the annual reports are already printed as text in the PDF**, so they can be read exactly without a vision model. What is missing is tying each label to its bar, series and year.
- **Qwen3-VL is good on labelled charts that fill a page** (Chevron 2022: 4 of 4 values) but **missed a small chart on a dense page** (Nvidia FY2026: 0 of 4), and it lists axis ticks next to data values without telling them apart.
- **A value no chart prints (unlabelled bars) cannot be read exactly by any method.** The system should describe the trend and say the value is not printed.
- **A paid vision model is needed only for about 2% of pages** (`paidVL`), and only when Qwen3-VL disagrees with OCR: roughly **$1–$40 for min500**, depending on the model. Track B rules may not allow it at all.

## 2. What the corpus contains (measured, 25 EDGAR annual reports)

From [financial_rag_final_design.md](financial_rag_final_design.md) §4.1:

| Kind of chart | Pages / regions | Can exact numbers be read? |
|---|---|---|
| Vector chart whose **data labels are real PDF text** | 117 pages | **Yes, exactly**, from the text layer (words and positions) |
| Image **with a text layer** over it | 57 pages | **Yes, exactly**, from the text layer |
| Vector chart in a **garbled font** | 25 pages | Yes, through OCR of the rendered page |
| **Pixel-only** chart with numbers | 21 regions (~0.8 per filing) | Mostly, through OCR or a vision model; needs checks (see the "12000" case below) |
| Bars or lines with **no printed values** | 90 pages | **No.** Only the trend or ranking can be described |

An earlier OCR test on the 21 pixel-only charts read Nvidia's four labels exactly, but on Chevron both OCR passes also returned **"12000", which is not on the chart** (an axis tick read as a value). Two readers agreeing is not proof that a number is a data value.

## 3. What the pipelines do today (built)

| Chart page | Pipeline 1 (Chroma, main index) | Pipeline 2 (Qdrant) |
|---|---|---|
| Labels in the text layer | Text layer + native tables as Markdown; labels arrive as loose text, not tied to bars | Same text; ColPali vectors find the page |
| Garbled or pixel-only labels | OCR, or Qwen3-VL ("one line per labelled value, never estimate") checked against the text layer or OCR; unconfirmed figures go to `unverified_numbers` | OCR text; ColPali vectors find the page |
| Unlabelled bars | Qwen3-VL **search description** (kind of chart, title, measures, series, years; no values) so the page can be found | ColPali vectors find the page |

So chart pages can be **found**, and printed labels reach the answering model as text, but there is no structured chart value ("Net sales 2024 = 391,035") with its series and year.

## 4. Qwen3-VL test (measured, 12 November 2026)

Setup: the pipeline's transcription prompt (`VLM_PROMPT`), **Qwen3-VL-2B, 4-bit (Q4_K_M), llama.cpp on CPU**, whole page rendered at about 1 megapixel. Gold values come from the frozen evaluation set (FIN-067/068 for Nvidia, FIN-069–072 for Chevron).

| Chart | Text layer has the values? | Qwen3-VL-2B | Time (CPU) |
|---|---|---|---|
| **Chevron 2022 p35**: four production and reserves charts, gold 1,440 · 7,677 · 11.2 · 279 | No (garbled) | ✅ **4 of 4 correct** | 97 s |
| **Nvidia FY2026 p60**: a peer revenue bar chart on a dense compensation page, gold $229.70 · $113.27 · $54.03 · $40.93 | No | ❌ **0 of 4**: it transcribed other figures on the page (145, 395, 894) and missed the chart's labels | 71 s |

**Weaknesses seen:**
1. **Small charts on busy pages are missed.** The page is shrunk to about 1 megapixel, so a small chart's labels become too small to read.
2. **Axis ticks are mixed with data labels.** On Chevron it listed `12,000 / 9,000 / 6,000 / 3,000 / 0` next to the real value `7,677` without marking which is which. The figures are exact text and would pass the OCR cross-check (OCR sees the ticks too), but **a confirmed number is not necessarily a data value**.

**This was the weakest setup** (smallest model, 4-bit, CPU, whole page). The 4B and 8B models on a Colab GPU should do better; not measured yet. Qwen reports strong chart and document benchmark results for the Qwen3-VL family, but not on annual reports, so our own test is the one to trust.

A search **description** of a page (no values) worked well with the same 2B model, about 54 s per page on CPU. Walmart 2025 p2: *"an infographic titled 'A message from our CEO'… revenues, operating income, … eCommerce sales, advertising, membership income"*.

## 5. Options to read chart numbers reliably (not built)

| # | Option | What it fixes | Needs a model? | Effort |
|---|---|---|---|---|
| 1 | **Chart crop**: render only the chart area at high resolution and send that to Qwen3-VL | Small charts on busy pages (the Nvidia case) | Qwen3-VL | Small: page analysis already finds chart areas |
| 2 | **Structured prompt**: ask for `series, category, value, unit` rows; drop values that match the axis ticks | Separates data labels from axis ticks | Qwen3-VL | Small |
| 3 | **Exact label pairing for vector charts**: take each label's position in the chart area and match it to its year or category by x position; output a table chunk such as "Net sales chart: 2023 = 383,285 · 2024 = 391,035" | Exact values for the largest group (117 of the measured chart pages) | **No** (CPU) | About a day |
| 4 | **Geometry checks**: axis ticks evenly spaced (rejects "12000"), each label on a bar top, segments add up to the total | Catches misreads from any reader | No | 1–2 days |
| 5 | **Use 4B or 8B on a GPU** in Colab instead of 2B on CPU | Better reading overall, ~5–15 s per page (estimate) | Qwen3-VL | A setting (`QWEN_MODEL`) |
| 6 | **Check against XBRL / the 10-K**: a chart value that matches a tagged figure counts as confirmed | Confirms values; promotes them to exact | No | Needs the facts DB (not built) |

The design doc's **extraction ladder** (T1 XBRL → T2 filing text → T3 PDF label → T4 validated OCR → T5 unvalidated, never used in answers → T6 reviewer-approved) describes how options 3, 4 and 6 fit together.

**Never stated in any option:** a value no document prints. For unlabelled bars the system describes the trend or ranking, says the exact value is not printed, and links the page.

## 6. Paid vision model for `paidVL` pages

**Pages** (page audit of 20 filings: 72 of 3,316 pages = **2.2%** `paidVL`):

| Corpus | Pages (~158 per filing) | `paidVL` pages |
|---|---|---|
| min500 (509 filings) | ~80,000 | **~1,750** |
| full (1,004 filings) | ~159,000 | **~3,450** |

These are **upper limits**: in Pipeline 1 a `paidVL` page goes to the paid model only when more than 10% of Qwen3-VL's figures fail the OCR check. The exact count for min500 comes from running the page audit (`_summary.json`).

**Cost**, worst case (every `paidVL` page escalated). Prices per 1,000 pages are from [deployment.md](deployment.md) §5.3 (researched 9–10 November 2026, ~1,500 input and 800 output tokens per page), **doubled** because the pipeline calls the standard API page by page, not the 50%-off batch API:

| Paid model | $ per 1,000 pages (standard) | **min500 (~1,750 pages)** | full (~3,450) |
|---|---|---|---|
| Claude Haiku 5.5 | ~$0.56 | **~$1** | ~$2 |
| Gemini 3.1 Flash-Lite | ~$1.58 | **~$3** | ~$5 |
| GPT-5.4 mini | ~$4.72 | **~$8** | ~$16 |
| Claude Sonnet 5.5 | ~$11 | **~$19** | ~$38 |
| Claude Opus 5.5 | ~$22 | **~$39** | ~$76 |

Dense tables can use ~1.8× more tokens, so roughly double the top of each range. Prices change on 1 December and 1 January for some models; check before spending. Without a paid model, `paidVL` pages keep the Qwen3-VL reading with unconfirmed figures flagged, so the pipeline still works.

## 7. Track B constraints

- Track B allows only the provided model for answering, and the template says teams "cannot swap in a frontier model". Using a paid model **only to transcribe pages at ingest time** is arguable but not obviously allowed.
- The provided chat model (Qwen3-8B) is text-only, so whatever a vision model reads must be turned into text at ingest time (Pipeline 1's design).
- Whether Qwen3-VL itself (an open model, run on Colab at ingest time only) is allowed should also be confirmed.

## 8. Decisions for the mentor discussion

1. **Qwen3-VL at ingest time:** is an open vision model allowed for reading pages and writing descriptions, if answers still come only from the provided Qwen3-8B?
2. **Paid vision model:** allowed for the ~2% `paidVL` pages? If yes, which price band (section 6)?
3. **Chart values in answers:** may the agent cite a chart value read by a model (labelled "read from chart"), or only values confirmed against the text layer, OCR or XBRL?
4. **Scope before 25 November:** which of options 1–6 to build. Suggested order if approved: 1 + 2 (small, fixes the measured failure), then 3 (exact, no model), then 4. Option 6 depends on the facts DB.
5. **Evaluation:** score chart questions on the 6 frozen chart items (FIN-067–072) plus a hand-labelled set of ~50 charts, reporting precision and coverage per method.

## 9. Proposed default (until decided)

- Keep Qwen3-VL for **search descriptions** and **labelled charts** (it read Chevron correctly), but do not treat its chart numbers as facts on their own: they stay checked against the text layer or OCR and flagged otherwise.
- Do not call a paid model until the mentor approves.
- First build step after approval: **chart crop + structured prompt** (options 1 and 2), then rerun the same two-chart test for a before/after comparison.

## 10. How the measurements were made

- **Corpus counts** (section 2): [financial_rag_final_design.md](financial_rag_final_design.md) §4.1, 25 EDGAR ARS PDFs.
- **Qwen3-VL test** (section 4): `scripts.rag.run_local_multimodal.vision_server()` started the local llama.cpp server with `indexes/models/Qwen3-VL-2B` (Q4_K_M + Q8 projector); `financial_ingest_chroma_db.OpenAIVLM.transcribe` and `.describe` were called on pages rendered by `financial_ingest_chroma_db.render`; gold values were matched as exact strings in the output.
- **`paidVL` share** (section 6): `python -m scripts.rag.page_audit` on 20 filings of `data/raw/financial_report_qa/edgar` (3,316 pages).
- **Prices** (section 6): [deployment.md](deployment.md) §5.3.
