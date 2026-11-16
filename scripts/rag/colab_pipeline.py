"""Build either financial RAG index on Colab (or locally) from one Settings object.

Two pipelines over the sector corpus (scripts/downloads/download_annual_report_corpus.py):

``colpali_qdrant`` (scripts/rag/financial_ingest_qdrant_db.py), Qdrant server, two collections:
    annual_reports_text    all the text: every page's text layer, OCR for unreadable pages, and table chunks;
                   dense vector (RAM, HNSW) + number-aware BM25 sparse vector
    annual_reports_visual  ColPali on table, chart and image pages only: mean-pooled rows + cols (38 x 128 each,
                   RAM, HNSW) for the first stage; full 1030 x 128 multivector on disk (float16, mmap,
                   no HNSW) for the MaxSim rerank; payloads on disk
``vlm_chroma`` (scripts/rag/financial_ingest_chroma_db.py), Chroma only, text embeddings + a BM25 sidecar:
    OCR or a VLM (local Qwen-VL, or an OpenAI-compatible API, self-hosted or paid) reads each table,
    chart, image and unreadable page image and writes it as Markdown text;
    clean narrative pages keep their text layer; all chunks are embedded as text into Chroma
    (persisted on disk, HNSW loaded into RAM when queried)

Where things live (the storage layout of scripts/common/storage.py; <work> = RAG_HOME):
    <drive_root>/annual_reports/       PDFs by sector + manifests (durable)
    <work>/annual_reports/             the sector being ingested, copied from Drive to fast local disk
    <work>/indexes/qdrant_db/          live index of colpali_qdrant on the VM's fast local disk (lost at disconnect)
    <work>/indexes/chroma_db/          live index of vlm_chroma
    <drive_root>/snapshots/qdrant_db/  Qdrant snapshots + ingest state, written after every sector
                                       (a snapshot copies the whole collection; snapshot_every=N for every N batches)
    <drive_root>/indexes/chroma_db/    copy of the Chroma index + state, written after every sector
    <drive_root>/cache/vlm_pages/      VLM transcriptions, one JSONL per filing (append-only, so safe on Drive)
Folders written before this layout (<work>/colpali_qdrant, <work>/vlm_chroma, <drive_root>/snapshots/colpali_qdrant,
<drive_root>/indexes/vlm_chroma, <drive_root>/vlm_pages) are found and reused, so nothing needs renaming.
``export`` copies the Drive copies to a Google Cloud Storage bucket for the GCP deployment (deploy/sync_from_gcs.sh).

``restore`` brings the Drive copy back to local disk at the start of a session, and
``ingest`` processes one sector at a time, skipping finished filings. So a new
session continues where the last one stopped. Models (embedder, ColPali, VLM) are
loaded once and reused for every sector.

Notebook: notebooks/colab_rag_pipelines.ipynb. From a terminal:

    python -m scripts.rag.colab_pipeline run --pipeline colpali_qdrant --sectors banking --max-per-sector 20
    python -m scripts.rag.colab_pipeline run --pipeline vlm_chroma --sectors banking --vlm-pages tables_charts
    python -m scripts.rag.colab_pipeline query --pipeline vlm_chroma "net interest income" --sector banking

Locally (no Colab) <work> is the local RAG_HOME (<repo>/indexes/rag_home), the folder the local API serves, and
Qdrant is embedded (``--qdrant-url ""``).
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

from scripts.common import storage
from scripts.common.paths import ROOT
from scripts.common.sector_corpus import MANIFEST_NAME, parse_sectors, read_manifest
from scripts.rag import financial_ingest_qdrant_db as fi
from scripts.rag import financial_ingest_chroma_db as vi

PIPELINES = ("colpali_qdrant", "vlm_chroma")
IN_COLAB = "google.colab" in sys.modules or "COLAB_RELEASE_TAG" in os.environ
QDRANT_RELEASE = os.getenv("QDRANT_VERSION", "latest")     # e.g. v1.15.4 to pin the server
QDRANT_PORT = 6333


@dataclass
class Settings:
    pipeline: str = "colpali_qdrant"
    drive_root: Path = Path("/content/drive/MyDrive/capstone_rag") if IN_COLAB else ROOT / "indexes/pipelines_drive"
    work: Path = storage.layout("colab" if IN_COLAB else "local").home    # RAG_HOME: the live copy
    corpus_preset: str = "full"             # "full" = 1,004 filings; "min500" = the 509-filing variant (Track B minimum)
    sectors: str = ""                       # names or slugs, comma-separated; "" = every downloaded sector
    max_per_sector: int | None = 20         # download: largest companies first; None = the whole plan
    years: str | None = None                # download: e.g. "2024,2025"
    user_agent: str | None = os.getenv("SEC_USER_AGENT")
    qdrant_url: str | None = f"http://127.0.0.1:{QDRANT_PORT}" if IN_COLAB else None   # None = embedded (local runs)
    embedding_provider: str = "local"       # "hosted" = the shared embedding API (docs/embedding_key.md)
    embedding_model: str | None = None      # None: BAAI/bge-small-en-v1.5 for vlm_chroma, all-MiniLM-L6-v2 for colpali_qdrant
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
    vlm: str = "routed"                     # page reader: "routed" (per page, paidVL fallback) / "ocr" / "transformers"
                                            # (Qwen3-VL) / "openai" (paid or self-hosted) / "mock"
    qwen_reader: str = "transformers"       # routed: Qwen3-VL in-process, or "openai" (endpoint at RAG_VISION_API_BASE)
    paid_threshold: float = 0.10            # routed: paidVL page -> paid model when > 10% of Qwen3-VL's figures fail OCR
    paid_max_pages_per_pdf: int = 0         # routed: cap paid calls per filing (0 = no cap)
    paid_first: bool = False                # routed: paidVL pages -> paid model first, Qwen3-VL only as a fallback
    describe_visuals: bool = True           # routed: Qwen3-VL search description of chart and image pages
    vlm_model: str | None = None            # default Qwen/Qwen3-VL-4B-Instruct (or -2B / -8B)
    vlm_pages: str = "visual"               # tables, charts, images + unreadable pages
                                            # or "tables_charts_images" / "tables_charts" / "all"
    vlm_max_pages_per_pdf: int = 0
    # smoke runs
    run_name: str = ""                      # suffix for the index folders, e.g. "_smoke": PDFs and VLM cache stay shared
    limit_filings: int | None = None        # per sector
    max_pages_per_pdf: int | None = None

    def __post_init__(self):
        if self.pipeline not in PIPELINES:
            raise ValueError(f"pipeline must be one of {PIPELINES}")
        if self.corpus_preset not in ("full", "min500"):
            raise ValueError("corpus_preset must be 'full' or 'min500'")
        self.drive_root, self.work = Path(self.drive_root), Path(self.work)
        if self.embedding_model is None:      # pipeline 1 uses bge-small (also served by the shared API)
            self.embedding_model = fi.LOCAL_BGE_SMALL if self.pipeline == "vlm_chroma" else "sentence-transformers/all-MiniLM-L6-v2"

    @property
    def corpus(self):
        return self.drive_root / "annual_reports"

    @property
    def staged(self):
        return self.work / "annual_reports"

    @property
    def kind(self):
        return storage.PIPELINE_KIND[self.pipeline]

    @property
    def root(self):
        return storage.work_index(self.work, self.kind, self.run_name)

    @property
    def backup(self):
        return storage.durable_index(self.drive_root, self.kind, self.run_name)

    @property
    def vlm_cache(self):
        return storage.first_existing([self.drive_root / "cache" / "vlm_pages", self.drive_root / "vlm_pages"])


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
           "QDRANT__STORAGE__SNAPSHOTS_PATH": str(s.root / "snapshots"),     # where financial_ingest_qdrant_db.restore copies to
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
    argv = ["--output-dir", str(s.corpus), "--preset", s.corpus_preset]
    argv += ["--user-agent", s.user_agent] if s.user_agent else []
    argv += ["--sectors", s.sectors] if s.sectors else []
    argv += ["--max-per-sector", str(s.max_per_sector)] if s.max_per_sector else []
    argv += ["--years", s.years] if s.years else []
    return dl.main(argv)


FUND_DIR = "mutual_fund"               # <home>/indexes/mutual_fund, the same name on Drive, in the bucket and on the VM
FUND_ZIP = "mutual_fund_data.zip"       # what pack_mutual_fund writes; upload it to MyDrive/capstone_rag/
FUND_DB = "nport_mutual_fund.sqlite3"


def pack_mutual_fund(source=None, out=None) -> Path:
    """Zip the local mutual fund data (N-PORT database, fund-card Chroma folder, raw filings) for Colab and GCP:
    upload the zip to MyDrive/capstone_rag/ and ``restore`` unpacks it. The database is copied with SQLite's backup
    API, so a running API that reads it doesn't matter."""
    import sqlite3
    import tempfile
    import zipfile
    source = Path(source or storage.layout().mutual_fund)
    if not (source / FUND_DB).exists():
        raise FileNotFoundError(f"no {FUND_DB} in {source}: load the funds first (scripts.rag.nport_mutual_fund load)")
    out = Path(out or ROOT / "output" / "colab" / FUND_ZIP)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(".zip.partial")
    with tempfile.TemporaryDirectory() as tmp:
        db_copy = Path(tmp) / FUND_DB
        src, dst = sqlite3.connect(f"file:{(source / FUND_DB).as_posix()}?mode=ro", uri=True), sqlite3.connect(db_copy)
        try:
            src.backup(dst)
        finally:
            src.close()
            dst.close()
        with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(db_copy, FUND_DB)
            for path in sorted(source.rglob("*")):
                if path.is_file() and path.name != FUND_DB and not path.name.startswith(FUND_DB + "-"):
                    archive.write(path, path.relative_to(source).as_posix())
    partial.replace(out)
    print(f"mutual fund data packed: {out} ({out.stat().st_size / 1e6:.1f} MB); upload it to MyDrive/capstone_rag/")
    return out


def durable_mutual_fund(s: Settings) -> Path | None:
    """<drive_root>/indexes/mutual_fund, unpacked from <drive_root>/mutual_fund_data.zip the first time."""
    import zipfile
    folder = s.drive_root / "indexes" / FUND_DIR
    if (folder / FUND_DB).exists():
        return folder
    archive = s.drive_root / FUND_ZIP
    if not archive.exists():
        return None
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():                               # no absolute paths or ".." out of the folder
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"unsafe path in {archive}: {name}")
        zf.extractall(folder)
    print(f"mutual fund data unpacked from {archive} into {folder}")
    return folder


def restore_mutual_fund(s: Settings) -> bool:
    """Copy the mutual fund data from Drive to <work>/indexes/mutual_fund, where the API finds it (RAG_HOME layout).
    Without it the router's fund route simply stays off."""
    live = s.work / "indexes" / FUND_DIR
    if (live / FUND_DB).exists():
        print(f"mutual fund data already in place: {live}")
        return False
    durable = durable_mutual_fund(s)
    if durable is None:
        print(f"no mutual fund data on Drive ({s.drive_root / FUND_ZIP}); the fund route stays off")
        return False
    shutil.copytree(durable, live, dirs_exist_ok=True)
    print(f"mutual fund data restored from {durable} into {live}")
    return True


def restore(s: Settings):
    """Bring the Drive copy back to local disk when this session has no index yet (and the mutual fund data)."""
    restore_mutual_fund(s)
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
            "--embedding-provider", s.embedding_provider, "--embedding-model", s.embedding_model,
            "--corpus-preset", s.corpus_preset]
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
            "--vlm", s.vlm, "--vlm-pages", pages, "--vlm-max-pages-per-pdf", str(s.vlm_max_pages_per_pdf),
            "--qwen-reader", s.qwen_reader, "--paid-threshold", str(s.paid_threshold),
            "--paid-max-pages-per-pdf", str(s.paid_max_pages_per_pdf)]
    argv += [] if s.describe_visuals else ["--no-describe-visuals"]
    argv += ["--paid-first"] if s.paid_first else []
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
    """Every planned filing by sector: downloaded? ingested? failed? The plan is ``corpus_plan.jsonl`` narrowed
    to ``s.corpus_preset`` (full: 1,004 filings; min500: 509).

    A filing counts as ingested when every stage of this pipeline is complete (colpali_qdrant: s.stages;
    vlm_chroma: its one stage). The totals answer "is the whole plan in the index?".
    """
    import sqlite3
    from scripts.common.sector_corpus import PLAN_NAME, read_jsonl, sector_slug
    from scripts.downloads.download_annual_report_corpus import DEFAULT_DIR, document_id, preset_rows
    plan_file = s.corpus / PLAN_NAME if (s.corpus / PLAN_NAME).exists() else DEFAULT_DIR / PLAN_NAME
    planned = {document_id(r): sector_slug(r["sector"]) for r in preset_rows(read_jsonl(plan_file), s.corpus_preset)}
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
    report = {"pipeline": s.pipeline, "plan": str(plan_file), "corpus_preset": s.corpus_preset,
              "by_sector": {slug: {k: c[k] for k in ("planned", "downloaded", "ingested", "pending", "failed")}
                            for slug, c in rows.items()},
              "total": {k: total[k] for k in ("planned", "downloaded", "ingested", "pending", "failed")},
              "failed": failed,
              "not_in_plan": sorted(set(status) - set(planned)),
              "by_doc": planned}
    if show:
        print(f"{'sector':26s} {'planned':>7s} {'downloaded':>10s} {'ingested':>8s} {'pending':>7s} {'failed':>6s}")
        for slug, c in report["by_sector"].items():
            print(f"{slug:26s} {c['planned']:7d} {c['downloaded']:10d} {c['ingested']:8d} {c['pending']:7d} {c['failed']:6d}")
        t = report["total"]
        print(f"{'total':26s} {t['planned']:7d} {t['downloaded']:10d} {t['ingested']:8d} {t['pending']:7d} {t['failed']:6d}")
        if failed:
            print(f"failed ({len(failed)}; rerun ingest to retry, see status for errors): {failed[:20]}")
    return report


def chroma_metadatas(store, batch=20000, fields=None):
    """Every chunk's metadata, read in batches so 300k+ chunks don't need one huge response."""
    total = store.count()
    for offset in range(0, total, batch):
        for meta in store.collection.get(include=["metadatas"], limit=batch, offset=offset)["metadatas"]:
            yield {k: meta.get(k) for k in fields} if fields else meta


def chroma_check(root):
    """Chunks in Chroma, BM25 and the state file, and the filings (doc_ids) Chroma holds."""
    import sqlite3
    out = {"chroma_chunks": 0, "bm25_chunks": 0, "state_chunks": 0, "state_complete": 0, "chroma_filings": set()}
    state = Path(root) / "vlm_ingest.sqlite3"
    if state.exists():
        with contextlib.closing(sqlite3.connect(f"file:{state}?mode=ro", uri=True)) as db:
            out["state_complete"], out["state_chunks"] = db.execute(
                "SELECT COUNT(*), COALESCE(SUM(points), 0) FROM filings WHERE status='complete'").fetchone()
            out["embedding"] = dict(db.execute("SELECT key, value FROM settings")).get("embedding")
    if (Path(root) / "chroma").exists():
        store = vi.ChromaStore(root)
        out["chroma_chunks"], out["bm25_chunks"] = store.count(), store.bm25.count()
        out["chroma_filings"] = {m["doc_id"] for m in chroma_metadatas(store, fields=("doc_id",))}
    return out


def embedding_matches(s: Settings, recorded):
    """The index's recorded embedding (e.g. local:BAAI/bge-small-en-v1.5:384) is the one this run would use:
    the same model name, or the hashing embedder of tests and smoke runs. Local and hosted bge-small are one model."""
    recorded = recorded or ""
    if s.embedding_provider == "hashing":
        return recorded.startswith("hashing")
    return Path(s.embedding_model).name in recorded


def verify_upload(s: Settings, expected_filings=None):
    """Is everything the PC staged on Drive, and did restore bring the index to local disk intact?

    Checks the code and fund zips, every planned PDF (present, non-empty), the Drive index copy (state file,
    page analyses, Chroma and BM25 files), the OCR cache, and - once ``restore`` ran - the local copy: Chroma chunks =
    BM25 chunks = the state file's chunks, and every planned filing in Chroma. Prints one line per check.
    """
    from scripts.common.sector_corpus import resolve_local_path
    planned = coverage(s, show=False)
    expected = expected_filings or planned["total"]["planned"]
    rows = read_manifest(s.corpus) if s.corpus.exists() else []
    pdfs = [resolve_local_path(r) for r in rows]
    missing = [str(p) for p in pdfs if not p.exists() or p.stat().st_size == 0]
    durable = s.backup
    pages = len(list((durable / "pages").glob("*.json"))) if (durable / "pages").exists() else 0
    cached = len(list(s.vlm_cache.glob("*.jsonl"))) if s.vlm_cache.exists() else 0
    checks = [
        ("code zip on Drive", (s.drive_root / "capstone_code.zip").exists(), str(s.drive_root / "capstone_code.zip")),
        ("mutual fund zip on Drive", (s.drive_root / FUND_ZIP).exists(), str(s.drive_root / FUND_ZIP)),
        ("PDFs in the manifests", len(rows) >= expected, f"{len(rows)} rows (expect {expected})"),
        ("PDF files present and non-empty", bool(pdfs) and not missing,
         f"{len(pdfs) - len(missing)} ok, {len(missing)} missing, {folder_gb(s.corpus)} GB"),
        ("planned filings downloaded", planned["total"]["downloaded"] == expected,
         f"{planned['total']['downloaded']} of {expected}"),
        ("Drive index: state file", (durable / "vlm_ingest.sqlite3").exists(), str(durable / "vlm_ingest.sqlite3")),
        ("Drive index: page analyses", pages >= expected, f"{pages} json (expect {expected})"),
        ("Drive index: Chroma + BM25 files", (durable / "chroma" / "chroma.sqlite3").exists()
         and (durable / "chroma" / "bm25.sqlite3").exists(), f"{folder_gb(durable / 'chroma')} GB"),
        ("OCR / page reading cache", cached > 0, f"{cached} filings cached in {s.vlm_cache}"),
    ]
    local = chroma_check(s.root) if (s.root / "vlm_ingest.sqlite3").exists() else None
    if local:
        absent = sorted(set(planned["by_doc"]) - local["chroma_filings"])
        checks += [
            ("local copy: filings complete in state", local["state_complete"] >= expected,
             f"{local['state_complete']} (expect {expected})"),
            ("local copy: embedding model = this run's", embedding_matches(s, local.get("embedding")),
             f"{local.get('embedding')} (run: {s.embedding_provider} {s.embedding_model})"),
            ("local copy: Chroma = BM25 = state chunks",
             local["chroma_chunks"] == local["bm25_chunks"] == local["state_chunks"] > 0,
             f"{local['chroma_chunks']:,} / {local['bm25_chunks']:,} / {local['state_chunks']:,}"),
            ("local copy: every planned filing in Chroma", not absent,
             f"{len(local['chroma_filings'])} filings" + (f", missing {absent[:5]}" if absent else "")),
        ]
    else:
        checks.append(("local copy restored", False, f"nothing at {s.root}: run cp.restore(s) (cell 3) first"))
    for name, ok, detail in checks:
        print(f"{'OK  ' if ok else 'FAIL'} {name:42s} {detail}")
    failed = [name for name, ok, _ in checks if not ok]
    print("\nALL CHECKS PASSED" if not failed else f"\n{len(failed)} check(s) failed: {failed}")
    return {"ok": not failed, "failed": failed, "missing_pdfs": missing[:50], "checks": checks}


PAGE_READERS = ("pdfParser", "ocr", "qwen3VL", "paidVL")
READER_NAMES = {"pdfParser": "PDF text layer / native tables", "ocr": "RapidOCR", "qwen3VL": "Qwen3-VL",
                "paidVL": "paid model (Gemini Flash)"}


def reader_summary(s: Settings, expected_filings=None):
    """Who read each page of the final index, and whether all planned filings are in it.

    Counts distinct (filing, page) pairs per ``reader`` (pdfParser text layer, ocr, qwen3VL, paidVL = the paid model,
    e.g. Gemini Flash) from the Chroma chunks, the paidVL escalation outcomes, the Qwen3-VL page descriptions, and the
    filings still carrying a ``skip=`` pipeline (read on the PC without Qwen3-VL, not yet redone).
    """
    import sqlite3
    planned = coverage(s, show=False)
    expected = expected_filings or planned["total"]["planned"]
    store = vi.ChromaStore(s.root)
    pages, escalations, descriptions, by_route = {}, Counter(), set(), Counter()
    filings, unverified = set(), set()
    for m in chroma_metadatas(store, fields=("doc_id", "page", "reader", "content_type", "escalation", "page_route",
                                               "unverified_numbers")):
        key = (m["doc_id"], m["page"])
        filings.add(m["doc_id"])
        if m.get("content_type") == "page_description":
            descriptions.add(key)
            continue
        if key not in pages:
            pages[key] = m.get("reader") or "pdfParser"
            by_route[m.get("page_route") or "?"] += 1
            if m.get("page_route") == "paidVL":
                escalations[m.get("escalation") or "none"] += 1
        if m.get("unverified_numbers") not in (None, "[]"):
            unverified.add(key)
    readers = Counter(pages.values())
    with contextlib.closing(sqlite3.connect(state_path(s))) as db:
        rows = db.execute("SELECT doc_id, status, pipeline FROM filings").fetchall()
    complete = [d for d, st, _ in rows if st == "complete"]
    still_skipped = [d for d, st, p in rows if st == "complete" and "skip=" in (p or "")]
    absent = sorted(set(planned["by_doc"]) - filings)
    report = {
        "filings": {"planned": expected, "complete_in_state": len(complete), "in_chroma": len(filings),
                    "missing_from_chroma": absent, "failed": planned["failed"],
                    "still_without_qwen3vl": len(still_skipped)},
        "chunks": {"chroma": store.count(), "bm25": store.bm25.count()},
        "pages_indexed": len(pages),
        "pages_by_reader": {r: readers.get(r, 0) for r in (*PAGE_READERS, *sorted(set(readers) - set(PAGE_READERS)))},
        "pages_by_route": dict(by_route.most_common()),
        "paidVL_pages_by_outcome": dict(escalations.most_common()),
        "qwen3vl_page_descriptions": len(descriptions),
        "pages_with_unverified_figures": len(unverified),
    }
    report["all_filings_present"] = not absent and len(filings) >= expected and not planned["failed"]
    total = len(pages) or 1
    print(f"Filings: {len(filings)} in Chroma, {len(complete)} complete in state, planned {expected}  -> "
          + ("ALL PRESENT" if report["all_filings_present"] else f"MISSING {absent[:10]} FAILED {planned['failed'][:10]}"))
    chunks = report["chunks"]
    print(f"Chunks: Chroma {chunks['chroma']:,} | BM25 {chunks['bm25']:,}  ("
          + ("match" if chunks["chroma"] == chunks["bm25"] else "MISMATCH") + ")")
    print(f"Filings still without Qwen3-VL (PC run, skip=): {len(still_skipped)}")
    audit = s.drive_root / "page_audit" / "_summary.json"
    if audit.exists():
        usage = json.loads(audit.read_text(encoding="utf-8")).get("parser_usage", {})
        report["page_audit_routes"] = usage
        print(f"Page audit: {usage.get('qwen3VL', 0):,} pages routed to qwen3VL, {usage.get('paidVL', 0):,} to paidVL"
              f" (Qwen3-VL, then Gemini when it disagrees with OCR); {usage.get('pdfParser+colpali', 0):,} chart/image"
              " pages get a Qwen3-VL description")
    print(f"\nPages indexed: {len(pages):,}")
    for reader, count in report["pages_by_reader"].items():
        print(f"  {READER_NAMES.get(reader, reader):34s} {count:8,d}  {100 * count / total:5.1f}%")
    print(f"  {'Qwen3-VL chart/image descriptions':34s} {len(descriptions):8,d}  (extra search chunks)")
    print("\npaidVL pages by outcome: " + (", ".join(f"{k}={v:,}" for k, v in escalations.most_common()) or "none")
          + "\n  not_needed = Qwen3-VL matched OCR | paid = Gemini's reading kept | paid_not_better = Qwen kept"
            "\n  paid_first = Gemini read it first and matched OCR | paid_first_qwen_better = Qwen's reading kept"
            " | paid_first_qwen_not_better = Gemini kept after Qwen also tried"
            "\n  no_paid_model / paid_limit = Gemini not called | models_failed_ocr_fallback = OCR kept")
    print(f"Pages with figures not confirmed by OCR or the text layer: {len(unverified):,}")
    return report


def serving_env(s: Settings):
    """Environment for FastAPI/agents so they answer from this pipeline's index (see runtime_profile)."""
    env = {"RAG_PROFILE": "colab" if IN_COLAB else "local", "RAG_HOME": str(s.work),
           "RAG_INGEST_DIR": str(storage.work_index(s.work, "qdrant_db", s.run_name)),
           "RAG_VLM_DIR": str(storage.work_index(s.work, "chroma_db", s.run_name))}
    if s.pipeline == "colpali_qdrant":
        env.update(RAG_BACKEND="financial_qdrant", RAG_INGEST_DIR=str(s.root))
        if s.qdrant_url:
            env["QDRANT_URL"] = s.qdrant_url
    else:
        env.update(RAG_BACKEND="financial_vlm", RAG_VLM_DIR=str(s.root))
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


def page_audit(s: Settings, workers=None, force=False, limit=None, out=None):
    """Which parser each downloaded page needs: <drive_root>/page_audit/<sector>/<pdf name>.csv (pageNumber,parserUsage)
    plus _summary.csv / _summary.json. Reads the PDFs on Drive (CPU only: no GPU needed); finished filings are
    skipped on a rerun, so after a disconnect just run it again. ``limit``: first N filings (a timing run)."""
    from scripts.rag import page_audit as pa
    return pa.run(s.corpus, out or s.drive_root / "page_audit", preset=s.corpus_preset, sectors=s.sectors or None,
                  workers=workers or os.cpu_count(), force=force, limit=limit)


def export_to_gcs(s: Settings, bucket=None, dry_run=False):
    """Copy the durable index copies (Drive) to a Google Cloud Storage bucket, in the storage layout the GCP VM reads
    (deploy/sync_from_gcs.sh): <bucket>/indexes/chroma_db, <bucket>/snapshots/qdrant_db and <bucket>/indexes/mutual_fund.
    Needs ``gcloud`` (preinstalled on Colab) and ``google.colab.auth.authenticate_user()`` first. Returns the commands
    it ran."""
    bucket = (bucket or os.getenv("RAG_GCS_BUCKET") or "").strip()
    if not bucket:
        raise ValueError("give the bucket (gs://name) or set RAG_GCS_BUCKET")
    bucket = bucket if bucket.startswith("gs://") else f"gs://{bucket}"
    commands = []
    for kind, target in (("chroma_db", "indexes/chroma_db"), ("qdrant_db", "snapshots/qdrant_db")):
        source = storage.durable_index(s.drive_root, kind, s.run_name)
        if source.exists():
            commands.append(["gcloud", "storage", "rsync", "-r", str(source), f"{bucket.rstrip('/')}/{target}"])
        else:
            print(f"skipped {kind}: no copy at {source}")
    funds = durable_mutual_fund(s)
    if funds is not None:
        commands.append(["gcloud", "storage", "rsync", "-r", str(funds), f"{bucket.rstrip('/')}/indexes/{FUND_DIR}"])
    else:
        print(f"skipped mutual_fund: no {FUND_ZIP} or indexes/{FUND_DIR} under {s.drive_root}")
    for command in commands:
        print(" ".join(command))
        if not dry_run:
            subprocess.run(command, check=True)
    return commands


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
    parser.add_argument("command", choices=["run", "prepare", "download", "restore", "ingest", "status", "coverage", "query",
                                            "audit", "export", "pack-funds"])
    parser.add_argument("--bucket", help="export: gs://bucket (default RAG_GCS_BUCKET)")
    parser.add_argument("--out", type=Path, help="pack-funds: zip path (default output/colab/mutual_fund_data.zip)")
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
        kind = float if isinstance(value, float) \
            else int if field.name in {"max_per_sector", "workers", "batch_size", "snapshot_every", "ocr_max_pages_per_filing",
                                     "vlm_max_pages_per_pdf", "paid_max_pages_per_pdf",
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
    if args.command == "export":
        return export_to_gcs(s, args.bucket)
    if args.command == "pack-funds":
        return pack_mutual_fund(out=args.out)
    return {"prepare": prepare, "download": download, "restore": restore, "ingest": ingest, "status": status,
            "coverage": coverage, "audit": page_audit}[args.command](s)


if __name__ == "__main__":
    main()
