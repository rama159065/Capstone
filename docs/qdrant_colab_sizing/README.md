# Qdrant on Google Colab — sizing for all six Operations Copilot domains

Track B “Multi-Agent Operations Copilot” (hosted ≤8B LLM). Workload per domain: **2,000 PDFs × 50–100 pages**, **two indexes**: OCR text of every page → `docs_text` (text embeddings, dense + BM25), and ColPali multivectors for the ~30% of pages with charts, images or tables → `visual_pages`, Qdrant as the vector DB, all agents and report generation running on top.

> **Free account verdict: not enough** for any domain's full 2,000-document corpus, and far from enough for all six. At 1,000 documents a single domain becomes borderline with optimized settings (see the 1,000-document table). With the local-download + batch-upload approach, storage fits free Drive for any single domain even at 2,000 documents; only GPU time (multiple sessions) remains. Free Colab can run the Qdrant server, agents and a pilot (a few hundred documents per domain); full scale needs paid compute (Colab Pro/Pro+ or a VM) and paid persistent storage.

## Typical-scenario summary

| Domain | Pages | Visual share | Qdrant as coded | Qdrant optimized | Disk working set (opt.) | Persist (opt.) | Qdrant serving RAM (opt.) | GPU ingest time | Free-tier pilot docs |
|---|---|---|---|---|---|---|---|---|---|
| [financial_report_qa](financial_report_qa.md) | 180,000 | 30% | 38.7 GB | 10.6 GB | 47.3 GB | 26.8 GB | 1.1 GB | 44.1 h | 453 |
| [procurement_vendor](procurement_vendor.md) | 120,000 | 30% | 25.6 GB | 6.9 GB | 31.1 GB | 17.7 GB | 1.0 GB | 29.4 h | 679 |
| [travel_expense](travel_expense.md) | 110,000 | 30% | 23.3 GB | 6.2 GB | 28.3 GB | 16.1 GB | 1.0 GB | 27.0 h | 741 |
| [competitive_research](competitive_research.md) | 140,000 | 30% | 27.6 GB | 6.1 GB | 33.2 GB | 18.7 GB | 0.9 GB | 34.3 h | 583 |
| [incident_postmortem](incident_postmortem.md) | 110,000 | 30% | 23.4 GB | 6.3 GB | 28.5 GB | 16.2 GB | 1.0 GB | 27.0 h | 741 |
| [hr_policy](hr_policy.md) | 130,000 | 30% | 27.7 GB | 7.4 GB | 33.7 GB | 19.1 GB | 1.0 GB | 31.9 h | 627 |
| **All six (12,000 docs)** | **790,000** | — | **166.3 GB** | **43.6 GB** | **201.9 GB** | **114.7 GB** | **5.9 GB** | **193.6 h** | — |

Low/high scenarios, RAM per process, embedding-dimension sensitivity and setup steps are in each domain file. Add ~10 GB per VM for models (ColPali, Qwen3-VL, MiniLM, reranker).

## Implementing only one domain

Only one domain will be implemented and written to the RAG index. Then Qdrant holds just that domain's two collections, and the per-domain figures are the full footprint; the all-six totals are for reference only. Candidates, smallest footprint first (typical pages, 30% visual, optimized):

| Domain | Qdrant 1,000 | Persist 1,000 | GPU 1,000 | Qdrant 2,000 | Persist 2,000 | GPU 2,000 | Fit for implementation |
|---|---|---|---|---|---|---|---|
| [travel_expense](travel_expense.md) | 3.1 GB | 8.1 GB | 13.5 h | 6.2 GB | 16.1 GB | 27.0 h | Smallest corpus; simple policy Q&A; receipts may push visual share above 30% |
| [incident_postmortem](incident_postmortem.md) | 3.2 GB | 8.1 GB | 13.5 h | 6.3 GB | 16.2 GB | 27.0 h | Small footprint; drafting fits agentic HITL story; screenshots need OCR + ColPali |
| [procurement_vendor](procurement_vendor.md) | 3.4 GB | 8.8 GB | 14.7 h | 6.9 GB | 17.7 GB | 29.4 h | Shared graph; HITL terms suit vendor approvals; scanned contracts drive OCR time |
| [competitive_research](competitive_research.md) | 3.1 GB | 9.4 GB | 17.1 h | 6.1 GB | 18.7 GB | 34.3 h | Chart-heavy decks showcase ColPali; fewer words/page keeps text index small |
| [hr_policy](hr_policy.md) | 3.7 GB | 9.6 GB | 15.9 h | 7.4 GB | 19.1 GB | 31.9 h | Prose-heavy; PII/refusal angle for responsible-AI section; may have <30% visual pages |
| [financial_report_qa](financial_report_qa.md) | 5.3 GB | 13.4 GB | 22.1 h | 10.6 GB | 26.8 GB | 44.1 h | Most complete agent + report generation; largest index; tables/charts benefit most from ColPali |

- **Free-tier fit:** at 1,000 documents every domain is borderline-feasible with optimized settings; at 2,000 none fits free Drive.
- **Richest existing code:** `financial_report_qa` (8-node graph with critic→answer repair loop, prompting-technique comparison, deterministic report generation) — but it has the largest footprint (90 pages/doc).
- **Cheapest to run:** `travel_expense` and `incident_postmortem` (~55 pages/doc) use the shared 9-node graph.
- Recommendation: implement `financial_report_qa` with 1,000 documents if report generation is part of the demo; otherwise `incident_postmortem` with 1,000–2,000 documents for the most headroom on Colab.
- Each domain file has an **“If this is the one domain implemented”** section with the exact provisioning and ingestion steps.

## Alternative approach: local download + batch upload to Colab

Keep all PDFs on the local PC, upload one batch at a time to Colab, process it (OCR text → `docs_text`, ColPali → `visual_pages`), write to Qdrant, snapshot, delete the batch and push the next. Only Qdrant and its SQLite sidecar accumulate. Values are Approach A → Approach B (typical, optimized, models included in disk):

| Domain | Batch at 2,000 docs | Colab disk 1,000 | Colab disk 2,000 | Drive 1,000 | Drive 2,000 | Local PDFs 2,000 | Upload 2,000 @ 50 Mbps |
|---|---|---|---|---|---|---|---|
| [financial_report_qa](financial_report_qa.md) | 90 docs / 23 batches | 33.6 GB → 20.8 GB | 57.3 GB → 28.7 GB | 13.4 GB → 5.3 GB | 26.8 GB → 10.6 GB | 16.2 GB | 0.7 h |
| [procurement_vendor](procurement_vendor.md) | 130 docs / 16 batches | 25.5 GB → 17.9 GB | 41.1 GB → 23.0 GB | 8.8 GB → 3.4 GB | 17.7 GB → 6.9 GB | 10.8 GB | 0.5 h |
| [travel_expense](travel_expense.md) | 140 docs / 15 batches | 24.1 GB → 17.3 GB | 38.3 GB → 22.0 GB | 8.1 GB → 3.1 GB | 16.1 GB → 6.2 GB | 9.9 GB | 0.4 h |
| [competitive_research](competitive_research.md) | 110 docs / 19 batches | 26.6 GB → 17.2 GB | 43.2 GB → 21.8 GB | 9.4 GB → 3.1 GB | 18.7 GB → 6.1 GB | 12.6 GB | 0.6 h |
| [incident_postmortem](incident_postmortem.md) | 140 docs / 15 batches | 24.2 GB → 17.4 GB | 38.5 GB → 22.1 GB | 8.1 GB → 3.2 GB | 16.2 GB → 6.3 GB | 9.9 GB | 0.4 h |
| [hr_policy](hr_policy.md) | 120 docs / 17 batches | 26.8 GB → 18.3 GB | 43.7 GB → 23.9 GB | 9.6 GB → 3.7 GB | 19.1 GB → 7.4 GB | 11.7 GB | 0.5 h |

- **Storage stops being the blocker:** every domain's snapshot fits free Drive even at 2,000 documents; Colab disk drops to Qdrant + two batches + models.
- **RAM and GPU hours are unchanged** — the free-tier limit becomes compute only (multiple sessions, T4 not guaranteed).
- OCR and ColPali must run in the same ingest pass (backfill commands need the deleted files); persist `metadata.sqlite3` with every snapshot; payload file paths dangle after deletion (store public `source_url`, keep PNGs if VLM answers are needed).
- On Colab call `LocalRAG.ingest()` directly: `run_local_multimodal ingest` starts the Windows `llama-server.exe` Qwen3-VL captioner, which does not run on Colab and adds ~8 s/page.
- Details, batch loop commands and caveats are in each domain file under **“Alternative approach: local download + batch upload to Colab”**.

## Recommended: Qdrant + page images (30% visual pages)

Batch flow, but keep the rendered PNGs of the ~30% visual pages next to Qdrant (PDFs and OCR JSON are deleted). Needed for VLM answers over charts/tables and page-preview citations. Drive persistent = Qdrant snapshot + images:

| Domain | Images at 2,000 docs | PNG size 1k / 2k | Drive 1k (PNG) | Drive 1k (WebP) | Drive 2k (PNG) | Drive 2k (WebP) | Colab disk peak 2k (PNG) |
|---|---|---|---|---|---|---|---|
| [financial_report_qa](financial_report_qa.md) | 54,000 | 6.8 GB / 13.5 GB | ✅ 12.1 GB | ✅ 8.0 GB | ❌ 24.1 GB | ❌ 16.0 GB | 42.2 GB |
| [procurement_vendor](procurement_vendor.md) | 36,000 | 4.5 GB / 9.0 GB | ✅ 7.9 GB | ✅ 5.2 GB | ❌ 15.9 GB | ✅ 10.5 GB | 32.0 GB |
| [travel_expense](travel_expense.md) | 33,000 | 4.1 GB / 8.2 GB | ✅ 7.2 GB | ✅ 4.8 GB | ✅ 14.5 GB | ✅ 9.5 GB | 30.2 GB |
| [competitive_research](competitive_research.md) | 42,000 | 5.2 GB / 10.5 GB | ✅ 8.3 GB | ✅ 5.2 GB | ❌ 16.6 GB | ✅ 10.3 GB | 32.3 GB |
| [incident_postmortem](incident_postmortem.md) | 33,000 | 4.1 GB / 8.2 GB | ✅ 7.3 GB | ✅ 4.8 GB | ✅ 14.6 GB | ✅ 9.6 GB | 30.4 GB |
| [hr_policy](hr_policy.md) | 39,000 | 4.9 GB / 9.8 GB | ✅ 8.6 GB | ✅ 5.7 GB | ❌ 17.2 GB | ✅ 11.3 GB | 33.6 GB |

- At **1,000 documents**: all domains fit free Drive with PNG. At **2,000 documents**: `travel_expense`, `incident_postmortem` fit free Drive with PNG; `procurement_vendor`, `competitive_research`, `hr_policy` fit only after WebP/JPEG conversion; `financial_report_qa` needs paid storage even with WebP. Free Drive's 15 GB is also shared with Gmail/Photos; Google One 100 GB removes the limit.
- RAM does not change: images are read from disk, at most 3 per answer.
- Keep `RAG_LOCAL_DIR` identical across sessions (payload `image_path` is absolute) and archive images as one tar per batch on Drive.

## 1,000 documents per domain (typical pages, 30% visual)

| Domain | Pages | Qdrant as coded | Qdrant optimized | Disk working set (opt.) | Persist (opt.) | Qdrant serving RAM (opt.) | GPU ingest time | Free account |
|---|---|---|---|---|---|---|---|---|
| [financial_report_qa](financial_report_qa.md) | 90,000 | 19.3 GB | 5.3 GB | 23.6 GB | 13.4 GB | 0.9 GB | 22.1 h | ⚠️ borderline (optimized, ~3 sessions) |
| [procurement_vendor](procurement_vendor.md) | 60,000 | 12.8 GB | 3.4 GB | 15.5 GB | 8.8 GB | 0.9 GB | 14.7 h | ⚠️ borderline (optimized, ~2 sessions) |
| [travel_expense](travel_expense.md) | 55,000 | 11.7 GB | 3.1 GB | 14.1 GB | 8.1 GB | 0.9 GB | 13.5 h | ⚠️ borderline (optimized, ~2 sessions) |
| [competitive_research](competitive_research.md) | 70,000 | 13.8 GB | 3.1 GB | 16.6 GB | 9.4 GB | 0.9 GB | 17.1 h | ⚠️ borderline (optimized, ~2 sessions) |
| [incident_postmortem](incident_postmortem.md) | 55,000 | 11.7 GB | 3.2 GB | 14.2 GB | 8.1 GB | 0.9 GB | 13.5 h | ⚠️ borderline (optimized, ~2 sessions) |
| [hr_policy](hr_policy.md) | 65,000 | 13.8 GB | 3.7 GB | 16.8 GB | 9.6 GB | 0.9 GB | 15.9 h | ⚠️ borderline (optimized, ~2 sessions) |
| **All six (6,000 docs)** | **395,000** | **83.2 GB** | **21.8 GB** | **101.0 GB** | **57.3 GB** | **5.3 GB** | **96.8 h** | ❌ |

Halving the corpus halves storage and ingestion time; Qdrant serving RAM barely changes because fixed process and cache overhead dominates. With optimized settings a single domain at 1,000 documents comes close to free Drive's 15 GB; all six together (57.3 GB to persist, 96.8 h of GPU ingestion) still need paid storage and compute.

## Key findings from the code

- **ColPali dominates storage**: ~0.53 MB per visual page as float32 multivectors (≈1,030 × 128). Float16 + token pooling ×3 cuts it ~6×.
- **Embedded Qdrant local mode cannot hold this data** (pure Python, in-process RAM, warns above 20,000 points). Set `QDRANT_URL` and run the Qdrant server binary in Colab.
- **Forced OCR on all pages would make every page a ColPali page** under the current `visual` rule — decouple the two.
- `grounding_text` repeats the whole page in every chunk's payload; payload is on disk, so it costs disk, not RAM.
- Agents use the hosted LLM, so their Colab RAM is small (~1–1.5 GB); RAM pressure comes from ingestion models (ColPali, OCR, optional Qwen3-VL).

## Free tier vs. what is needed (all six domains, typical, optimized)

| Resource | Free limit | Needed | Verdict |
|---|---|---|---|
| Persistent storage | Drive 15 GB | 114.7 GB | ❌ |
| VM disk (ephemeral) | ~78 GB (GPU runtime) | 201.9 GB if all domains on one VM | ❌ (one domain at a time at most) |
| Qdrant serving RAM | ~12.7 GB shared with models | 5.9 GB | ✅ for serving only, if ingestion models are not loaded |
| Ingestion GPU time | ≤12 h/session, T4 not guaranteed | 193.6 h | ❌ |
| Qdrant Cloud free | 1 GB RAM / 4 GB disk | 43.6 GB index | ❌ |

Limits are approximate (Colab does not publish fixed quotas); check the Colab resource panel and Qdrant pricing page before planning.
