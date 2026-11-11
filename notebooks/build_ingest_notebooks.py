"""Build the two Colab ingestion notebooks from shared cells.

    notebooks/colab_ingest_chroma_db.ipynb   text index: PDF parser + OCR + Qwen3-VL (+ paid VL) -> Chroma + BM25
    notebooks/colab_ingest_qdrant_db.ipynb   visual index: text + OCR -> Qdrant text collection, ColPali -> Qdrant visual

Both drive scripts/rag/colab_pipeline.py (pipelines ``vlm_chroma`` and ``colpali_qdrant``), so the notebooks only
set things up. Rebuild after editing:  python notebooks/build_ingest_notebooks.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

SETUP = r'''# @title 2. Drive, code, dependencies
import json, os, subprocess, sys
from pathlib import Path
from google.colab import drive
drive.mount("/content/drive")


def fetch_code(root, repo_url, branch):
    # Unpack the code zip (Drive, else one you upload) into root; without a zip, clone the repo.
    import zipfile
    uploaded = Path("/content") / Path(CODE_ZIP).name
    source = next((p for p in (Path(CODE_ZIP), uploaded) if p.exists()), None)
    if source is None and not root.exists():
        from google.colab import files
        print(f"{CODE_ZIP} not found: upload capstone_code.zip (cancel to clone {repo_url} instead)")
        names = list(files.upload())
        source = Path("/content") / names[0] if names else None
    if source is not None:
        with zipfile.ZipFile(source) as archive:
            info = archive.read("BUNDLE_INFO.json")
            stamp = root / "BUNDLE_INFO.json"
            if not stamp.exists() or stamp.read_bytes() != info:      # a new build: unpack over the old one
                archive.extractall(root)
        print("code:", source, json.loads(info))
    elif not root.exists():
        try:
            from google.colab import userdata
            token = userdata.get("GITHUB_TOKEN")      # needed while the repo is private
        except Exception:
            token = None
        url = repo_url.replace("https://", f"https://{token}@") if token else repo_url
        subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", branch, url, str(root)], check=True)


fetch_code(Path("/content/Capstone"), REPO, BRANCH)
sys.path.insert(0, "/content/Capstone")
%cd /content/Capstone
!pip -q install {PIP}
import importlib.util
assert importlib.util.find_spec("scripts.rag.{MODULE}"), \
    "old code: rebuild capstone_code.zip (python -m scripts.common.colab_bundle) and upload it to Drive"
!nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU: {NO_GPU}"'''

SECRETS = r'''for key in ("LLM_API_BASE", "LLM_API_KEY", "LLM_MODEL", "LLM_CA_CERT"):
    try:
        os.environ[key] = userdata.get(key)
    except Exception:
        print(f"{key} is not a Colab secret" + (": answers use llm_provider=mock" if key == "LLM_MODEL" else ""))'''

SERVE_MD = r'''## Serve this index: FastAPI + Streamlit on this runtime

```
browser -> Colab port proxy (only you) -> Streamlit :8501 -> FastAPI :8000 -> {INDEX}
```

`cp.serve(s)` starts both with `cp.serving_env(s)`: {ENV}. Requests can filter by `ticker`, `year` and `sector`.
For the Track B model add the Colab secrets `LLM_API_BASE`, `LLM_API_KEY`, `LLM_MODEL` (e.g. `Qwen/Qwen3-8B`) and,
for a self-signed endpoint, `LLM_CA_CERT` (path to the CA certificate on Drive). Without them answers use the mock.'''

SERVE_DEPS = r'''# @title Serving dependencies (from pyproject.toml; keeps Colab's CUDA torch) and LLM secrets
import re, tomllib
deps = tomllib.load(open("/content/Capstone/pyproject.toml", "rb"))["project"]["dependencies"]
deps = [d for d in deps if re.split(r"[<>=!~ \[]", d)[0] not in {"torch", "torchvision", "jupyterlab"}]
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *deps])
from google.colab import userdata
''' + SECRETS + "{EXTRA}"

SERVE_START = r'''# @title Start the API and the chat UI
api, ui = cp.serve(s)
import time; time.sleep(8)
from google.colab.output import eval_js
print("Open the chat UI:", eval_js("google.colab.kernel.proxyPort(8501)"))'''

SERVE_CHECK = r'''# @title Check the API without the UI: sector-filtered hybrid retrieval and one answer
import urllib.request
def post(path, body):
    request = urllib.request.Request(f"http://127.0.0.1:8000{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(request, timeout=300))
hits = post("/retrieve/financial_report_qa", {"query": "net interest income", "sector": "banking", "top_k": 3})
for e in hits["evidence"]:
    print(e["chunk_id"], e.get("retrieval_ranks"), (e.get("preview") or "")[:80])
provider = "hosted" if os.getenv("LLM_MODEL") else "mock"
answer = post("/chat/financial_report_qa", {"query": "What was JPMorgan's net interest income?", "ticker": "JPM",
                                            "year": 2024, "llm_provider": provider})
print(provider, "|", answer["final_status"], "|", (answer.get("answer") or "")[:300])'''

SERVE_STOP = r'''# @title Stop serving (the index stays)
for process in (ui, api):
    process.terminate()'''

COMMON_SETTINGS = r'''CORPUS = "min500"  # @param ["min500", "full"]
# min500 = 509 filings (Track B minimum: 41 per sector + every eval-set company); full = all 1,004
SECTORS = ""  # @param {type:"string"}
# "" = all 13 sectors; or e.g. "banking,retail"
MAX_PER_SECTOR = 0  # @param {type:"integer"}
# download only: 0 = every PDF of the CORPUS preset; 20 = a first pass (largest companies first)
SMOKE_TEST = True  # @param {type:"boolean"}
# True: 2 PDFs per sector downloaded, 1 filing per sector ingested, first 30 pages, into a separate *_smoke index
SEC_USER_AGENT = "capstone-research your.name@example.com"  # @param {type:"string"}
DRIVE_ROOT = "/content/drive/MyDrive/capstone_rag"
REPO = "https://github.com/rama159065/Capstone.git"   # used only when there is no code zip
BRANCH = "financial-report-only"
CODE_ZIP = "/content/drive/MyDrive/capstone_rag/capstone_code.zip"   # python -m scripts.common.colab_bundle, then upload'''

SECTOR_LIST = ("banking, insurance, financial_services, oil_and_gas, energy_and_utilities, healthcare, technology_and_it, "
               "retail, telecom_and_media, industrials, consumer_goods, materials_and_chemicals, real_estate")

# ============================================================================= Chroma (text index)

CHROMA_INTRO = r'''# Ingest the annual reports into **Chroma** (text index)

Every page becomes **text**, so the text-only Track B model (Qwen3-8B) can read all of it: tables as Markdown, charts as
labelled values, scanned and garbled pages through OCR. Code: [scripts/rag/financial_ingest_chroma_db.py](../scripts/rag/financial_ingest_chroma_db.py),
driven by [scripts/rag/colab_pipeline.py](../scripts/rag/colab_pipeline.py) (pipeline `vlm_chroma`).

```
Drive PDFs (by sector) -> copy one sector to local disk -> analyse every page -> a reader per page
   -> chunks (1,200/200; tables split with their header) -> MiniLM vectors -> Chroma `annual_reports_vlm`
                                                         -> number-aware terms -> BM25 file chroma/bm25.sqlite3
```

**Readers per page** with `READER = "routed"` (the page audit rules, `scripts/rag/page_audit.route`):

| Page | Reader | Notes |
|---|---|---|
| plain text | PDF text layer | exact, free |
| table / chart / picture with figures in the text layer | text layer + native tables as Markdown | no model |
| garbled text layer, glyphs drawn as outlines, scanned page | RapidOCR (CPU) | |
| chart without printed values, infographic | **Qwen3-VL** (GPU) | figures checked against the text layer |
| table / chart / picture with an unusable text layer (`paidVL`) | OCR -> **Qwen3-VL**, figures checked against the OCR text -> **paid model** only when more than `PAID_THRESHOLD` of them are unconfirmed | without a paid model: Qwen3-VL kept, unconfirmed figures flagged |
| blank page, photo or logo | skipped | |

Every chunk records `reader`, `text_source`, `escalation` and `unverified_numbers`. Readings are cached on Drive per page
and reader (`vlm_pages/`), so a rerun never pays twice, and the Chroma folder is copied to Drive after every sector.

**Runtime:** GPU (T4 runs Qwen3-VL-4B in 4-bit; L4/A100 for 8B). Only about 3% of pages need Qwen3-VL and about 10% need OCR
(page audit of 20 filings). Rough cost for min500 (~80k pages), **estimates**:
the text layer and analysis ~3 CPU-hours; OCR of ~8k pages ~10 CPU-hours; Qwen3-VL on ~2.5k pages ~10-28 GPU-hours on a T4
(2B is faster, or serve Qwen3-VL with vLLM and set `QWEN_READER = "openai"`). The smoke test prints the measured rates.

**Before you start:** on your PC run `python -m scripts.common.colab_bundle` and upload `output/colab/capstone_code.zip`
to `MyDrive/capstone_rag/`. Optional Colab secrets: `PAID_VISION_API_BASE`, `PAID_VISION_MODEL`, `PAID_VISION_API_KEY`
(the paid fallback model, e.g. `https://api.openai.com/v1`, `gpt-4o`), and `LLM_*` for serving.'''

CHROMA_SETTINGS = "# @title 1. Settings\n" + COMMON_SETTINGS + r'''
READER = "routed"  # @param ["routed", "ocr", "transformers", "openai"]
# routed = a reader per page (table above); ocr = OCR for every visual page (CPU only); transformers = Qwen3-VL for
# every visual page; openai = an OpenAI-compatible endpoint for every visual page (RAG_VISION_API_BASE / _MODEL / _KEY)
QWEN_MODEL = "Qwen/Qwen3-VL-4B-Instruct"  # @param ["Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"]
QWEN_READER = "transformers"  # @param ["transformers", "openai"]
# transformers = Qwen3-VL on this GPU; openai = a served Qwen3-VL (vLLM) at the secret RAG_VISION_API_BASE
PAID_THRESHOLD = 0.10  # @param {type:"number"}
# paidVL pages go to the paid model when more than this share of Qwen3-VL's figures is not in the page's OCR text
PAID_MAX_PAGES_PER_PDF = 0  # @param {type:"integer"}
# cap on paid calls per filing (0 = no cap)
VLM_PAGES = "visual"  # @param ["visual", "tables_charts_images", "tables_charts", "all"]
# pages for the non-routed readers (routed chooses per page)'''

CHROMA_STEP3 = r'''# @title 3. Settings object, paid model secrets, restore the index from Drive
from scripts.rag import colab_pipeline as cp
os.environ["SEC_USER_AGENT"] = SEC_USER_AGENT
s = cp.Settings(
    pipeline="vlm_chroma", corpus_preset=CORPUS, drive_root=DRIVE_ROOT, sectors=SECTORS, user_agent=SEC_USER_AGENT,
    max_per_sector=2 if SMOKE_TEST else (MAX_PER_SECTOR or None),
    vlm=READER, vlm_model=QWEN_MODEL, qwen_reader=QWEN_READER, vlm_pages=VLM_PAGES,
    paid_threshold=float(PAID_THRESHOLD), paid_max_pages_per_pdf=PAID_MAX_PAGES_PER_PDF,
    run_name="_smoke" if SMOKE_TEST else "",
    limit_filings=1 if SMOKE_TEST else None,
    max_pages_per_pdf=30 if SMOKE_TEST else None,
    vlm_max_pages_per_pdf=3 if SMOKE_TEST else 0)
from google.colab import userdata
for key in ("PAID_VISION_API_BASE", "PAID_VISION_MODEL", "PAID_VISION_API_KEY",
            "VISION_API_BASE", "VISION_MODEL", "VISION_API_KEY"):
    try:
        if userdata.get(key):
            os.environ["RAG_" + key] = userdata.get(key)
    except Exception:
        pass
print("paid fallback model:", os.getenv("RAG_PAID_VISION_MODEL") or "none (paidVL pages keep the Qwen3-VL reading, flagged)")
cp.prepare(s)
cp.restore(s)          # first session: nothing to restore; later sessions: the Chroma copy on Drive'''

CHROMA_NEXT = r'''## Next

- **Real run:** set `SMOKE_TEST = False` and run cells 3-6. After a disconnect run cells 2-3 and then 5: finished filings are
  skipped and every page reading is cached on Drive.
- **Check the routing first (optional):** `notebooks/colab_page_audit.ipynb` writes a CSV per PDF with the reader each page
  will get, plus the share per sector; it needs only a CPU.
- **Faster Qwen3-VL:** serve it with vLLM on a bigger GPU and set `QWEN_READER = "openai"` with the secrets
  `VISION_API_BASE` / `VISION_MODEL`.
- **Serving on your PC** after copying the index down: set `RAG_BACKEND=financial_vlm` and `RAG_VLM_DIR` to the folder that holds
  `chroma/` and `vlm_ingest.sqlite3`.
- **Sectors:** ''' + SECTOR_LIST + "."

# ============================================================================= Qdrant (visual index)

QDRANT_INTRO = r'''# Ingest the annual reports into **Qdrant** (text + ColPali visual index)

Code: [scripts/rag/financial_ingest_qdrant_db.py](../scripts/rag/financial_ingest_qdrant_db.py), driven by
[scripts/rag/colab_pipeline.py](../scripts/rag/colab_pipeline.py) (pipeline `colpali_qdrant`). Two collections on a Qdrant
server running on this VM:

| Collection | What | Vectors | RAM / disk |
|---|---|---|---|
| `annual_reports_text` | every page's text: the text layer, **OCR** for garbled / outlined / scanned pages, tables as Markdown | `dense` MiniLM 384-d + `bm25` sparse (number-aware terms, IDF) | dense in RAM (HNSW); payloads on disk |
| `annual_reports_visual` | **ColPali** page images of table, chart and image pages, plus the page's text | `colpali` full 1,030 x 128 fp16 | **disk**, no HNSW (MaxSim rerank only) |
| | | `mean_rows` / `mean_cols`: 32 row / column means + 6 special tokens = 38 x 128 | **RAM**, HNSW (first stage) |
| | | `summary`: MiniLM of the page text | RAM |

```
Drive PDFs -> copy one sector to local disk -> analyse pages (parallel) -> OCR unreadable pages (CPU)
   -> text stage:   chunks -> dense + bm25 -> annual_reports_text
   -> visual stage: page images (1.5x) -> ColPali (GPU) -> pooled + full multivectors -> annual_reports_visual
   -> snapshot both collections + ingest state to Drive after every sector
```

There is **no VLM** in this pipeline: ColPali only finds pages; the text the LLM reads comes from the text layer or OCR.
Search is hybrid (dense + BM25, RRF) and, with `RAG_FIN_VISUAL=1`, adds ColPali pages (pooled prefetch, exact MaxSim rerank).
If Track B later provides Qwen3-VL as the chat model, the pages ColPali finds can be sent to it as images.

**Runtime:** GPU for ColPali (T4 is fine, L4 faster). Rough cost for min500, **estimates**: text stage + page analysis
~3 CPU-hours, OCR of ~8k unreadable pages ~10 CPU-hours, ColPali on ~10k table/chart/image pages a few GPU-hours.
Storage: ~0.26 MB per visual page on disk (full vectors), ~20 KB in RAM (pooled). Snapshots copy whole collections to
Drive, so use `SNAPSHOT_EVERY = 0` (once per sector) at full scale.

**Before you start:** on your PC run `python -m scripts.common.colab_bundle` and upload `output/colab/capstone_code.zip`
to `MyDrive/capstone_rag/`.'''

QDRANT_SETTINGS = "# @title 1. Settings\n" + COMMON_SETTINGS + r'''
STAGES = "text,visual"  # @param ["text,visual", "text", "visual"]
# text = annual_reports_text (CPU); visual = ColPali into annual_reports_visual (GPU)
COLPALI_PAGES = "tables_charts_images"  # @param ["tables_charts_images", "tables_charts", "visual", "all"]
# pages that get ColPali vectors; every page's text goes to annual_reports_text either way
OCR = "garbled"  # @param ["garbled", "none"]
# garbled = OCR every unreadable page (garbled text layer, outlined glyphs, scans) so its text is searchable
OCR_MAX_PAGES_PER_FILING = 0  # @param {type:"integer"}
# 0 = every unreadable page; e.g. 20 caps the OCR time per filing
SNAPSHOT_EVERY = 0  # @param {type:"integer"}
# batches of 10 filings between Drive snapshots; 0 = once per sector (each snapshot copies the whole collection)
WORKERS = 2  # @param {type:"integer"}
# processes for page analysis (Colab standard runtimes have 2 CPUs)'''

QDRANT_STEP3 = r'''# @title 3. Settings object, Qdrant server, restore the snapshots from Drive
from scripts.rag import colab_pipeline as cp
os.environ["SEC_USER_AGENT"] = SEC_USER_AGENT
s = cp.Settings(
    pipeline="colpali_qdrant", corpus_preset=CORPUS, drive_root=DRIVE_ROOT, sectors=SECTORS, user_agent=SEC_USER_AGENT,
    max_per_sector=2 if SMOKE_TEST else (MAX_PER_SECTOR or None),
    stages=STAGES, visual_pages=COLPALI_PAGES, ocr=OCR, ocr_max_pages_per_filing=OCR_MAX_PAGES_PER_FILING,
    snapshot_every=SNAPSHOT_EVERY, workers=WORKERS,
    run_name="_smoke" if SMOKE_TEST else "",
    qdrant_url=None if SMOKE_TEST else "http://127.0.0.1:6333",      # smoke runs use embedded Qdrant, off the server
    limit_filings=1 if SMOKE_TEST else None,
    max_pages_per_pdf=30 if SMOKE_TEST else None)
cp.prepare(s)          # starts the Qdrant server on this VM (not for smoke runs)
cp.restore(s)          # first session: nothing to restore; later sessions: snapshots from Drive'''

QDRANT_NEXT = r'''## Next

- **Real run:** set `SMOKE_TEST = False` and run cells 3-6. After a disconnect run cells 2-3 (they restart Qdrant and
  restore the snapshots) and then 5: finished filings and stages are skipped.
- **Text first, ColPali later:** run with `STAGES = "text"` on a CPU runtime, then `STAGES = "visual"` on a GPU runtime.
- **Storage:** the Drive snapshots of the full plan need more than the free 15 GB (about 17-29 GB of ColPali vectors for
  the full corpus; min500 is about half).
- **Serving on your PC or the Docker server** after copying the snapshots down:
  `python -m scripts.rag.financial_ingest_qdrant_db restore --root indexes/financial_qdrant --qdrant-url http://localhost:6333 --snapshot-dir <snapshots>`,
  then `RAG_BACKEND=financial_qdrant` (see `docs/deployment.md`).
- **Sectors:** ''' + SECTOR_LIST + "."


def cell(kind, text):
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    base = {"cell_type": kind, "metadata": {}, "source": source}
    return base if kind == "markdown" else {**base, "execution_count": None, "outputs": []}


def notebook(cells):
    return {"cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def chroma():
    setup = SETUP.replace("{PIP}", '"pymupdf>=1.24,<2" "qdrant-client>=1.12,<2" "chromadb>=1.1" "transformers>=4.57,<5" '
                                   '"sentence-transformers>=3" "rapidocr>=3,<4" onnxruntime "bitsandbytes>=0.48" accelerate') \
                 .replace("{MODULE}", "financial_ingest_chroma_db") \
                 .replace("{NO_GPU}", "OCR and the text layer work; Qwen3-VL will be very slow (use READER = 'ocr')")
    return notebook([
        cell("markdown", CHROMA_INTRO),
        cell("code", CHROMA_SETTINGS),
        cell("code", setup),
        cell("code", CHROMA_STEP3),
        cell("code", "# @title 4. Download the chosen sectors to Drive (resumable; skips PDFs already there)\n"
                     "cp.download(s)\n!du -sh {DRIVE_ROOT}/annual_reports/*/ 2>/dev/null | sort -k2"),
        cell("code", "# @title 5. Ingest into Chroma, one sector at a time (rerun after a disconnect: cells 2-3, then this one)\n"
                     "# Each sector's PDFs are copied to local disk first; the Chroma folder is copied to Drive after each sector.\n"
                     "results = cp.ingest(s)\nresults"),
        cell("code", "# @title 6. Status: filings done / failed, chunks, readings per reader, unverified figures\n"
                     "report = cp.status(s)"),
        cell("code", "# @title 6b. Coverage of the plan: planned -> downloaded -> ingested, per sector\ncov = cp.coverage(s)"),
        cell("code", "# @title 7. Retrieval checks (hybrid: Chroma dense + BM25, fused by RRF)\n"
                     'cp.query(s, "net interest income", sector="banking")\n'
                     'cp.query(s, "net oil-equivalent production", sector="oil_and_gas")\n'
                     'cp.query(s, "revenue by segment", sector="retail")'),
        cell("markdown", SERVE_MD.replace("{INDEX}", "the Chroma index").replace(
            "{ENV}", "`RAG_BACKEND=financial_vlm`, `RAG_VLM_DIR=/content/rag/vlm_chroma`")),
        cell("code", "# @title 8. " + SERVE_DEPS.split("# @title ", 1)[1].replace("{EXTRA}", "")),
        cell("code", "# @title 9. " + SERVE_START.split("# @title ", 1)[1]),
        cell("code", "# @title 10. " + SERVE_CHECK.split("# @title ", 1)[1]),
        cell("code", "# @title 11. " + SERVE_STOP.split("# @title ", 1)[1]),
        cell("markdown", CHROMA_NEXT),
    ])


def qdrant():
    setup = SETUP.replace("{PIP}", '"qdrant-client>=1.12,<2" "pymupdf>=1.24,<2" "transformers>=4.57,<5" '
                                   '"sentence-transformers>=3" "rapidocr>=3,<4" onnxruntime') \
                 .replace("{MODULE}", "financial_ingest_qdrant_db") \
                 .replace("{NO_GPU}", "the text stage works; ColPali will be very slow (use STAGES = 'text')")
    extra = '\n# os.environ["RAG_FIN_VISUAL"] = "1"     # add ColPali visual hits to answers (loads ColPali on this GPU)'
    return notebook([
        cell("markdown", QDRANT_INTRO),
        cell("code", QDRANT_SETTINGS),
        cell("code", setup),
        cell("code", QDRANT_STEP3),
        cell("code", "# @title 4. Download the chosen sectors to Drive (resumable; skips PDFs already there)\n"
                     "cp.download(s)\n!du -sh {DRIVE_ROOT}/annual_reports/*/ 2>/dev/null | sort -k2"),
        cell("code", "# @title 5. Ingest into Qdrant, one sector at a time (rerun after a disconnect: cells 2-3, then this one)\n"
                     "# Each sector's PDFs are copied to local disk first; snapshots go to Drive after each sector.\n"
                     "results = cp.ingest(s)\nresults"),
        cell("code", "# @title 6. Status: points per collection and sector, failures, what is in RAM and on disk\n"
                     "report = cp.status(s)"),
        cell("code", "# @title 6b. Coverage of the plan: planned -> downloaded -> ingested, per sector\ncov = cp.coverage(s)"),
        cell("code", "# @title 7. Retrieval checks: hybrid text search, then ColPali (pooled prefetch -> full MaxSim rerank)\n"
                     'cp.query(s, "net interest income", sector="banking")\n'
                     'cp.query(s, "net oil-equivalent production", sector="oil_and_gas")\n'
                     'if "visual" in STAGES:\n'
                     '    cp.query(s, "revenue by segment bar chart", sector="banking", visual=True)'),
        cell("markdown", SERVE_MD.replace("{INDEX}", "Qdrant (this VM)").replace(
            "{ENV}", "`RAG_BACKEND=financial_qdrant`, `RAG_INGEST_DIR=/content/rag/colpali_qdrant`, `QDRANT_URL` "
                     "(serving needs the server, so `SMOKE_TEST = False`)")),
        cell("code", "# @title 8. " + SERVE_DEPS.split("# @title ", 1)[1].replace("{EXTRA}", extra)),
        cell("code", "# @title 9. " + SERVE_START.split("# @title ", 1)[1]),
        cell("code", "# @title 10. " + SERVE_CHECK.split("# @title ", 1)[1]),
        cell("code", "# @title 11. " + SERVE_STOP.split("# @title ", 1)[1]),
        cell("markdown", QDRANT_NEXT),
    ])


def main():
    for name, build in (("colab_ingest_chroma_db.ipynb", chroma), ("colab_ingest_qdrant_db.ipynb", qdrant)):
        path = HERE / name
        path.write_text(json.dumps(build(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print("wrote", path)


if __name__ == "__main__":
    main()
