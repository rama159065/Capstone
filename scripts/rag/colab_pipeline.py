"""Build either financial RAG index on Colab (or locally) from one Settings object.

Two pipelines over the sector corpus (scripts/downloads/download_annual_report_corpus.py):

``colpali_qdrant`` (scripts/rag/financial_ingest.py), Qdrant server, two collections:
    annual_reports_text    all the text: every page's text layer, OCR for unreadable pages, and table chunks;
                   dense vector (RAM, HNSW) + number-aware BM25 sparse vector
    annual_reports_visual  ColPali on table, chart and image pages only: mean-pooled rows + cols (38 x 128 each,
                   RAM, HNSW) for the first stage; full 1030 x 128 multivector on disk (float16, mmap,
                   no HNSW) for the MaxSim rerank; payloads on disk
``vlm_chroma`` (scripts/rag/vlm_ingest.py), Chroma, text embeddings only:
    a VLM reads each table, chart, image and unreadable page image and writes it as Markdown text;
    clean narrative pages keep their text layer; all chunks are embedded as text into Chroma
    (persisted on disk, HNSW loaded into RAM when queried)

Where things live on Colab:
    <drive_root>/annual_reports/       PDFs by sector + manifests (durable)
    <work>/annual_reports/             the sector being ingested, copied from Drive to fast local disk
    <work>/<pipeline>/                 the live index on the VM's fast local disk (lost at disconnect)
    <drive_root>/snapshots/colpali_qdrant/   Qdrant snapshots + ingest state, written after every sector
                                       (a snapshot copies the whole collection; snapshot_every=N for every N batches)
    <drive_root>/indexes/vlm_chroma/         copy of the Chroma index + state, written after every sector
    <drive_root>/vlm_pages/            VLM transcriptions, one JSONL per filing (append-only, so safe on Drive)

``restore`` brings the Drive copy back to local disk at the start of a session, and
``ingest`` processes one sector at a time, skipping finished filings. So a new
session continues where the last one stopped. Models (embedder, ColPali, VLM) are
loaded once and reused for every sector.

Notebook: notebooks/colab_rag_pipelines.ipynb. From a terminal:

    python -m scripts.rag.colab_pipeline run --pipeline colpali_qdrant --sectors banking --max-per-sector 20
    python -m scripts.rag.colab_pipeline run --pipeline vlm_chroma --sectors banking --vlm-pages tables_charts
    python -m scripts.rag.colab_pipeline query --pipeline vlm_chroma "net interest income" --sector banking

Locally (no Colab) the defaults are repo folders and embedded Qdrant (``--qdrant-url ""``).
"""
from __future__ import annotations

import argparse
import atexit
from collections import Counter
import contextlib
from dataclasses import dataclass, fields
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

from scripts.common.paths import ROOT
from scripts.common.sector_corpus import MANIFEST_NAME, parse_sectors, read_manifest
from scripts.rag import financial_ingest as fi
from scripts.rag import vlm_ingest as vi

PIPELINES = ("colpali_qdrant", "vlm_chroma")
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
QDRANT_RELEASE = os.getenv("QDRANT_VERSION", "latest")     # e.g. v1.15.4 to pin the server
QDRANT_PORT = 6333


@dataclass
class Settings:
    pipeline: str = "colpali_qdrant"
    drive_root: Path = Path("/content/drive/MyDrive/capstone_rag") if IN_COLAB else ROOT / "indexes/pipelines_drive"
    work: Path = Path("/content/rag") if IN_COLAB else ROOT / "indexes/pipelines"
    sectors: str = ""                       # names or slugs, comma-separated; "" = every downloaded sector
    max_per_sector: int | None = 20         # download: largest companies first; None = the whole plan
    years: str | None = None                # download: e.g. "2024,2025"
    user_agent: str | None = os.getenv("SEC_USER_AGENT")
    qdrant_url: str | None = f"http://127.0.0.1:{QDRANT_PORT}" if IN_COLAB else None   # None = embedded (local runs)
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # colpali_qdrant
    stages: str = "text,visual"
    visual_pages: str = "tables_charts_images"   # ColPali pages; or "tables_charts" / "rule" (+ unreadable pages) / "all"
    colpali_dtype: str = "auto"
    ocr: str = "garbled"                    # RapidOCR on unreadable text pages, so their text reaches the text collection; "none" to skip
    ocr_max_pages_per_filing: int = 0       # 0 = every unreadable page (all the text); e.g. 20 caps the OCR time
    workers: int = 2
    batch_size: int = 10                    # filings analysed together
    snapshot_every: int = 0                 # batches between Drive snapshots; 0 = once, at the end of each sector
    stage_pdfs: bool = IN_COLAB             # copy each sector's PDFs from Drive to local disk before ingesting
    # vlm_chroma
    vlm: str = "transformers"               # or "openai" / "mock"
    vlm_model: str | None = None            # default Qwen/Qwen2.5-VL-3B-Instruct
    vlm_pages: str = "visual"               # tables, charts, images + unreadable pages (no OCR here: skipped ones are lost)
                                            # or "tables_charts_images" / "tables_charts" / "all"
    vlm_max_pages_per_pdf: int = 0
    # smoke runs
    run_name: str = ""                      # suffix for the index folders, e.g. "_smoke": PDFs and VLM cache stay shared
    limit_filings: int | None = None        # per sector
    max_pages_per_pdf: int | None = None

    def __post_init__(self):
        if self.pipeline not in PIPELINES:
            raise ValueError(f"pipeline must be one of {PIPELINES}")
        self.drive_root, self.work = Path(self.drive_root), Path(self.work)

    @property
    def corpus(self):
        return self.drive_root / "annual_reports"

    @property
    def staged(self):
        return self.work / "annual_reports"

    @property
    def root(self):
        return self.work / (self.pipeline + self.run_name)

    @property
    def backup(self):
        return self.drive_root / ("snapshots" if self.pipeline == "colpali_qdrant" else "indexes") / (self.pipeline + self.run_name)

    @property
    def vlm_cache(self):
        return self.drive_root / "vlm_pages"


_loaded = {}          # models and clients shared by every sector and every call in this process


@atexit.register
def _close_clients():
    for key, value in list(_loaded.items()):
        if isinstance(key, tuple) and key[0] == "qdrant":
            value.close()


# ----------------------------------------------------------------------------- setup

def prepare(s: Settings):
    """Folders, and for colpali_qdrant a local Qdrant server (started if it isn't running)."""
    for path in (s.corpus, s.root, s.backup):
        path.mkdir(parents=True, exist_ok=True)
    if s.pipeline == "colpali_qdrant" and s.qdrant_url:
        start_qdrant(s)
    print(json.dumps({"pipeline": s.pipeline, "corpus": str(s.corpus), "live_index": str(s.root),
                      "drive_copy": str(s.backup), "qdrant": s.qdrant_url or "embedded" if s.pipeline == "colpali_qdrant" else None},
                     indent=2))


def qdrant_ready(url):
    try:
        with urllib.request.urlopen(f"{fi.rest_url(url)}/readyz", timeout=2):
            return True
    except OSError:
        return False


def start_qdrant(s: Settings):
    """Download the Linux Qdrant binary into <work>/bin and run it on localhost with storage on local disk."""
    if qdrant_ready(s.qdrant_url):
        print("Qdrant already running")
        return None
    if not sys.platform.startswith("linux"):
        raise RuntimeError("start_qdrant runs the Linux binary (Colab). Locally use qdrant_url=None (embedded) "
                           "or start a server yourself (docker run -p 6333:6333 qdrant/qdrant).")
    binary = s.work / "bin" / "qdrant"
    if not binary.exists():
        binary.parent.mkdir(parents=True, exist_ok=True)
        base = "https://github.com/qdrant/qdrant/releases"
        url = (f"{base}/latest/download" if QDRANT_RELEASE == "latest" else f"{base}/download/{QDRANT_RELEASE}") \
            + "/qdrant-x86_64-unknown-linux-gnu.tar.gz"
        archive = binary.parent / "qdrant.tar.gz"
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(binary.parent, filter="data")
        archive.unlink()
    env = {**os.environ, "QDRANT__SERVICE__HOST": "127.0.0.1", "QDRANT__SERVICE__HTTP_PORT": str(QDRANT_PORT),
           "QDRANT__STORAGE__STORAGE_PATH": str(s.root / "qdrant_storage"),
           "QDRANT__STORAGE__SNAPSHOTS_PATH": str(s.root / "snapshots"),     # where financial_ingest.restore copies to
           "QDRANT__TELEMETRY_DISABLED": "true"}
    process = subprocess.Popen([str(binary)], cwd=binary.parent, env=env, stdout=open(s.work / "qdrant.log", "ab"),
                               stderr=subprocess.STDOUT)
    for _ in range(60):
        if qdrant_ready(s.qdrant_url):
            print(f"Qdrant started (pid {process.pid}), storage {s.root / 'qdrant_storage'}")
            return process
        time.sleep(1)
    raise RuntimeError(f"Qdrant did not start; see {s.work / 'qdrant.log'}")


def download(s: Settings):
    """Fetch the planned PDFs of the chosen sectors to Drive (resumable; verified SEC User-Agent required)."""
    from scripts.downloads import download_annual_report_corpus as dl
    argv = ["--output-dir", str(s.corpus)]
    argv += ["--user-agent", s.user_agent] if s.user_agent else []
    argv += ["--sectors", s.sectors] if s.sectors else []
    argv += ["--max-per-sector", str(s.max_per_sector)] if s.max_per_sector else []
    argv += ["--years", s.years] if s.years else []
    return dl.main(argv)


def restore(s: Settings):
    """Bring the Drive copy back to local disk when this session has no index yet."""
    if s.pipeline == "colpali_qdrant":
        if (s.root / "ingest.sqlite3").exists() or not list(s.backup.glob("*.snapshot")):
            print("nothing to restore (first session, or this runtime already has the index)")
            return False
        if not s.qdrant_url:
            print("snapshots exist but restore needs the Qdrant server (qdrant_url); embedded runs start fresh")
            return False
        fi.restore(fi.build_parser().parse_args(["restore", "--root", str(s.root), "--qdrant-url", s.qdrant_url,
                                                 "--snapshot-dir", str(s.backup)]))
        return True
    if (s.root / "vlm_ingest.sqlite3").exists() or not (s.backup / "vlm_ingest.sqlite3").exists():
        print("nothing to restore (first session, or this runtime already has the index)")
        return False
    shutil.copytree(s.backup, s.root, dirs_exist_ok=True)
    print(f"Chroma index and state restored from {s.backup}")
    return True


def backup_vlm(s: Settings):
    """Copy the Chroma index, state and page analysis to Drive (Chroma's SQLite stays on local disk while in use)."""
    for name in ("chroma", "pages"):
        if (s.root / name).exists():
            shutil.copytree(s.root / name, s.backup / name, dirs_exist_ok=True)
    state = s.root / "vlm_ingest.sqlite3"
    if state.exists():
        fi.State(state).backup(s.backup / "vlm_ingest.sqlite3")


# ----------------------------------------------------------------------------- ingest

def present_sectors(s: Settings):
    """Sector slugs with downloaded PDFs."""
    try:
        return sorted({r["sector_slug"] for r in read_manifest(s.corpus)})
    except FileNotFoundError:                                   # nothing downloaded yet
        return []


def sectors_to_run(s: Settings):
    """Downloaded sectors (slugs) in folder order, limited to s.sectors."""
    wanted = parse_sectors(s.sectors)
    present = present_sectors(s)
    missing = sorted((wanted or set()) - set(present))
    if missing:
        print(f"not downloaded yet, skipped: {missing}")
    return [slug for slug in present if not wanted or slug in wanted]


def stage(s: Settings, sector):
    """Copy one sector's PDFs and manifest from Drive to local disk (reading from Drive while indexing is slow).

    Files already copied with the same size are kept, so a rerun copies only what is new. Returns the corpus
    folder to ingest from.
    """
    if not s.stage_pdfs:
        return s.corpus
    source, dest = s.corpus / sector, s.staged / sector
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source.glob("*.pdf")):
        target = dest / path.name
        if not target.exists() or target.stat().st_size != path.stat().st_size:
            shutil.copyfile(path, target.with_name(target.name + ".part"))
            target.with_name(target.name + ".part").replace(target)
            copied += 1
    shutil.copyfile(source / MANIFEST_NAME, dest / MANIFEST_NAME)          # always the latest list
    print(f"staged {sector}: {copied} PDFs copied to {dest}", flush=True)
    return s.staged


def common_argv(s: Settings, sector, corpus=None):
    argv = ["--root", str(s.root), "--manifest", str(corpus or s.corpus), "--sectors", sector,
            "--embedding-provider", s.embedding_provider, "--embedding-model", s.embedding_model]
    argv += ["--limit-filings", str(s.limit_filings)] if s.limit_filings else []
    argv += ["--max-pages-per-pdf", str(s.max_pages_per_pdf)] if s.max_pages_per_pdf else []
    return argv


def qdrant(s: Settings):
    key = ("qdrant", s.qdrant_url or str(s.root))
    if key not in _loaded:
        _loaded[key] = fi.connect(s.qdrant_url, s.root)
    return _loaded[key]


def ingest_colpali(s: Settings, sector, corpus=None):
    pages = {"visual": "rule"}.get(s.visual_pages, s.visual_pages)        # the two ingesters name this rule differently
    argv = ["ingest", *common_argv(s, sector, corpus), "--stages", s.stages, "--visual-pages", pages,
            "--colpali-dtype", s.colpali_dtype, "--ocr", s.ocr, "--ocr-max-pages-per-filing", str(s.ocr_max_pages_per_filing),
            "--workers", str(s.workers), "--batch-size", str(s.batch_size), "--snapshot-every", str(s.snapshot_every)]
    argv += ["--qdrant-url", s.qdrant_url, "--snapshot-dir", str(s.backup)] if s.qdrant_url else []
    run = fi.Ingest(fi.build_parser().parse_args(argv), client=qdrant(s),
                    embedder=_loaded.get("embedder"), colpali=_loaded.get("colpali"))
    _loaded["embedder"] = run.embedder
    try:
        return run.run()
    finally:
        if run._colpali is not None:
            _loaded["colpali"] = run._colpali


def ingest_vlm(s: Settings, sector, corpus=None):
    pages = {"rule": "visual"}.get(s.vlm_pages, s.vlm_pages)
    argv = ["ingest", *common_argv(s, sector, corpus), "--store", "chroma", "--cache-dir", str(s.vlm_cache),
            "--vlm", s.vlm, "--vlm-pages", pages, "--vlm-max-pages-per-pdf", str(s.vlm_max_pages_per_pdf)]
    argv += ["--vlm-model", s.vlm_model] if s.vlm_model else []
    run = vi.VlmIngest(vi.build_parser().parse_args(argv), vlm=_loaded.get("vlm"), embedder=_loaded.get("embedder"))
    _loaded["embedder"] = run.embedder
    try:
        return run.run()
    finally:
        if run._vlm is not None:
            _loaded["vlm"] = run._vlm
        backup_vlm(s)


def ingest(s: Settings):
    """One sector at a time; finished filings are skipped, so rerunning after a disconnect continues."""
    results = {}
    for sector in sectors_to_run(s):
        started = time.time()
        print(f"=== {s.pipeline}: {sector}", flush=True)
        corpus = stage(s, sector)
        summary = ingest_colpali(s, sector, corpus) if s.pipeline == "colpali_qdrant" else ingest_vlm(s, sector, corpus)
        results[sector] = {**summary, "minutes": round((time.time() - started) / 60, 1)}
        print(json.dumps({sector: results[sector]}), flush=True)
    return results


# ----------------------------------------------------------------------------- status and query

def folder_gb(path):
    path = Path(path)
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30, 3) if path.exists() else 0.0


def status(s: Settings):
    """Points/chunks per sector, failures, and what sits in RAM vs on disk."""
    report = {"pipeline": s.pipeline, "live_index_gb": folder_gb(s.root), "drive_copy_gb": folder_gb(s.backup)}
    if s.pipeline == "colpali_qdrant":
        client = qdrant(s)
        layout = {}
        for name in (fi.TEXT, fi.VISUAL):
            if not client.collection_exists(name):
                continue
            info = client.get_collection(name)
            layout[name] = {"points": client.count(name, exact=True).count,
                            "vectors": {v: ("disk" if c.on_disk else "RAM") for v, c in info.config.params.vectors.items()},
                            "payload": "disk" if info.config.params.on_disk_payload else "RAM"}
            layout[name]["by_sector"] = {slug: client.count(name, count_filter=fi.query_filter(sector=slug), exact=True).count
                                         for slug in present_sectors(s)}
        report["collections"] = layout
        state = s.root / "ingest.sqlite3"
    else:
        store = vi.ChromaStore(s.root) if (s.root / "chroma").exists() else None
        report["chroma"] = {"chunks": store.count() if store else 0, "where": "disk (persistent), HNSW in RAM when queried"}
        if store and store.count():
            metas = store.collection.get(include=["metadatas"])["metadatas"]
            report["chroma"]["by_sector"] = dict(sorted(Counter(m.get("sector_slug") for m in metas).items()))
        state = s.root / "vlm_ingest.sqlite3"
        with contextlib.redirect_stdout(io.StringIO()):          # vi.status prints; keep one report
            cache = vi.status(argparse.Namespace(root=s.root, cache_dir=s.vlm_cache))
        report["vlm_cache"] = {k: cache[k] for k in ("transcribed_pages", "pages_with_unverified_numbers",
                                                     "pages_without_text_layer_to_check", "vlm_seconds_per_page")}
    if state.exists():
        import sqlite3
        with sqlite3.connect(state) as db:
            report["filings"] = [dict(zip(("stage", "status", "count"), r)) for r in db.execute(
                "SELECT stage, status, COUNT(*) FROM filings GROUP BY stage, status")]
            report["failed"] = [dict(zip(("doc_id", "stage", "error"), r)) for r in db.execute(
                "SELECT doc_id, stage, error FROM filings WHERE status='failed'")]
    print(json.dumps(report, indent=2, default=str))
    return report


def state_path(s: Settings):
    return s.root / ("ingest.sqlite3" if s.pipeline == "colpali_qdrant" else "vlm_ingest.sqlite3")


def coverage(s: Settings, show=True):
    """Every planned filing (corpus_plan.jsonl, 1,004) by sector: downloaded? ingested? failed?

    A filing counts as ingested when every stage of this pipeline is complete (colpali_qdrant: s.stages;
    vlm_chroma: its one stage). The totals answer "is the whole plan in the index?".
    """
    import sqlite3
    from scripts.common.sector_corpus import PLAN_NAME, read_jsonl, sector_slug
    from scripts.downloads.download_annual_report_corpus import DEFAULT_DIR, document_id
    plan_file = s.corpus / PLAN_NAME if (s.corpus / PLAN_NAME).exists() else DEFAULT_DIR / PLAN_NAME
    planned = {document_id(r): sector_slug(r["sector"]) for r in read_jsonl(plan_file)}
    try:
        downloaded = {r["document_id"] for r in read_manifest(s.corpus)}
    except FileNotFoundError:
        downloaded = set()
    stages = set(s.stages.split(",")) if s.pipeline == "colpali_qdrant" else None
    status = {}
    if state_path(s).exists():
        with sqlite3.connect(state_path(s)) as db:
            for doc_id, stage, state in db.execute("SELECT doc_id, stage, status FROM filings"):
                if stages is None or stage in stages:
                    status.setdefault(doc_id, {})[stage] = state
    wanted = len(stages) if stages else 1
    rows, failed = {}, []
    for doc_id, slug in sorted(planned.items(), key=lambda kv: (kv[1], kv[0])):
        row = rows.setdefault(slug, Counter())
        row["planned"] += 1
        states = status.get(doc_id, {})
        if doc_id in downloaded:
            row["downloaded"] += 1
        if len(states) == wanted and set(states.values()) == {"complete"}:
            row["ingested"] += 1
        elif "failed" in states.values():
            row["failed"] += 1
            failed.append(doc_id)
        elif doc_id in downloaded:
            row["pending"] += 1
    total = sum(rows.values(), Counter())
    report = {"pipeline": s.pipeline, "plan": str(plan_file),
              "by_sector": {slug: {k: c[k] for k in ("planned", "downloaded", "ingested", "pending", "failed")}
                            for slug, c in rows.items()},
              "total": {k: total[k] for k in ("planned", "downloaded", "ingested", "pending", "failed")},
              "failed": failed,
              "not_in_plan": sorted(set(status) - set(planned))}
    if show:
        print(f"{'sector':26s} {'planned':>7s} {'downloaded':>10s} {'ingested':>8s} {'pending':>7s} {'failed':>6s}")
        for slug, c in report["by_sector"].items():
            print(f"{slug:26s} {c['planned']:7d} {c['downloaded']:10d} {c['ingested']:8d} {c['pending']:7d} {c['failed']:6d}")
        t = report["total"]
        print(f"{'total':26s} {t['planned']:7d} {t['downloaded']:10d} {t['ingested']:8d} {t['pending']:7d} {t['failed']:6d}")
        if failed:
            print(f"failed ({len(failed)}; rerun ingest to retry, see status for errors): {failed[:20]}")
    return report


def serving_env(s: Settings):
    """Environment for FastAPI/agents so they answer from this pipeline's index (see runtime_profile)."""
    env = {"RAG_PROFILE": "colab" if IN_COLAB else "local", "RAG_INGEST_DIR": str(s.work / "colpali_qdrant"),
           "RAG_VLM_DIR": str(s.work / "vlm_chroma")}
    if s.pipeline == "colpali_qdrant":
        env.update(RAG_BACKEND="financial_qdrant", RAG_INGEST_DIR=str(s.root))
        if s.qdrant_url:
            env["QDRANT_URL"] = s.qdrant_url
    else:
        env.update(RAG_BACKEND="financial_vlm", RAG_VLM_DIR=str(s.root), RAG_VLM_STORE="chroma")
    return env


def serve(s: Settings, repo=ROOT, api_port=8000, ui_port=8501):
    """Start FastAPI (one worker) and Streamlit on localhost against this pipeline's index; returns both processes.

    colpali_qdrant needs the Qdrant server (prepare/restore first): embedded Qdrant allows one process, and the
    notebook's own client would hold it.
    """
    if s.pipeline == "colpali_qdrant" and not s.qdrant_url:
        raise RuntimeError("serving colpali_qdrant needs the Qdrant server (qdrant_url); embedded Qdrant is single-process")
    if not state_path(s).exists():
        raise RuntimeError(f"no index under {s.root}: run ingest, or restore from Drive, first")
    env = {**os.environ, **serving_env(s)}
    logs = s.work / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def start(args, log):
        return subprocess.Popen(args, cwd=repo, env=env, stdout=open(logs / log, "ab"), stderr=subprocess.STDOUT)

    api = start([sys.executable, "-m", "uvicorn", "scripts.api.api_app:app", "--host", "127.0.0.1",
                 "--port", str(api_port), "--workers", "1"], "api.log")
    for _ in range(180):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{api_port}/health", timeout=2) as response:
                health = json.load(response)
            break
        except OSError:
            if api.poll() is not None:
                raise RuntimeError(f"API exited; see {logs / 'api.log'}")
            time.sleep(1)
    else:
        api.terminate()
        raise RuntimeError(f"API did not start; see {logs / 'api.log'}")
    print("API runtime:", json.dumps(health.get("runtime", {})))
    ui = start([sys.executable, "-m", "streamlit", "run", "scripts/ui/api_streamlit_app.py", "--server.port", str(ui_port),
                "--server.address", "127.0.0.1", "--server.headless", "true", "--server.enableCORS", "false",
                "--server.enableXsrfProtection", "false"], "streamlit.log")
    return api, ui


def query(s: Settings, text, sector=None, ticker=None, year=None, top_k=5, visual=False):
    """Top hits from the pipeline's index; visual=True also runs the ColPali path (loads the model)."""
    if s.pipeline == "vlm_chroma":
        hits = vi.retrieve(text, top_k, ticker, year, sector, root=s.root, store="chroma", embedder=_loaded.get("embedder"))
        rows = [{"score": round(h["score"], 4), "chunk_id": h["chunk_id"], "sector": h["sector"], "page": h["page"],
                 "source": h["text_source"], "type": h["content_type"], "preview": h["preview"][:200]} for h in hits]
    else:
        client = qdrant(s)
        embedder = _loaded.get("embedder") or fi.recorded_embedder(argparse.Namespace(
            root=s.root, embedding_provider=s.embedding_provider, embedding_model=s.embedding_model))
        _loaded["embedder"] = embedder
        flt = fi.query_filter(ticker, year, sector)
        found = [("text", h) for h in fi.search_text(client, embedder, text, top_k, flt)]
        if visual:
            if "colpali" not in _loaded:                       # load once; setdefault would load it every call
                _loaded["colpali"] = fi.ColPali(dtype=s.colpali_dtype)
            found += [("visual", h) for h in fi.search_visual(client, _loaded["colpali"], embedder, text, top_k, flt)]
        rows = [{"path": path, "score": round(h.score, 4), "chunk_id": h.payload["chunk_id"], "sector": h.payload.get("sector"),
                 "page": h.payload["page"], "type": h.payload.get("content_type"),
                 "preview": " ".join(h.payload.get("text", "").split())[:200]} for path, h in found]
    for row in rows:
        print(json.dumps(row))
    return rows


def run(s: Settings, download_first=True):
    prepare(s)
    restore(s)
    if download_first:
        download(s)
    results = ingest(s)
    status(s)
    coverage(s)
    return results


# ----------------------------------------------------------------------------- CLI

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["run", "prepare", "download", "restore", "ingest", "status", "coverage", "query"])
    parser.add_argument("text", nargs="?", help="query text")
    parser.add_argument("--sector", help="query filter")
    parser.add_argument("--ticker", help="query filter")
    parser.add_argument("--year", type=int, help="query filter")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--visual", action="store_true", help="query: also the ColPali path")
    parser.add_argument("--no-download", action="store_true", help="run: use the PDFs already downloaded")
    defaults = Settings()
    for field in fields(Settings):
        value = getattr(defaults, field.name)
        kind = int if field.name in {"max_per_sector", "workers", "batch_size", "snapshot_every", "ocr_max_pages_per_filing",
                                     "vlm_max_pages_per_pdf",
                                     "limit_filings", "max_pages_per_pdf"} \
            else (lambda v: v.strip().lower() in {"1", "true", "yes"}) if isinstance(value, bool) \
            else Path if isinstance(value, Path) else str
        parser.add_argument("--" + field.name.replace("_", "-"), type=kind, default=value)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    values = {f.name: getattr(args, f.name) for f in fields(Settings)}
    values["qdrant_url"] = values["qdrant_url"] or None
    s = Settings(**values)
    if args.command == "query":
        if not args.text:
            raise SystemExit("query needs the question text")
        return query(s, args.text, args.sector, args.ticker, args.year, args.top_k, args.visual)
    if args.command == "run":
        return run(s, download_first=not args.no_download)
    return {"prepare": prepare, "download": download, "restore": restore, "ingest": ingest, "status": status,
            "coverage": coverage}[args.command](s)


if __name__ == "__main__":
    main()
