"""One storage layout for every environment: this PC (``local``), Google Colab (``colab``), any Docker host
(``server``) and a Google Cloud VM (``gcp``).

Every environment uses the same folder structure under one root, ``RAG_HOME``. Only the root changes:

    <RAG_HOME>/                              live copy: fast local disk (Chroma, Qdrant and SQLite never run on Drive)
      annual_reports/<sector>/*.pdf          ingestion only (Colab stages each sector here from Drive)
      indexes/qdrant_db/                     pipeline 2: ingest.sqlite3, pages/, qdrant_embedded/ = RAG_INGEST_DIR
      indexes/chroma_db/                     pipeline 1: chroma/ (+ bm25.sqlite3), vlm_ingest.sqlite3  = RAG_VLM_DIR
      snapshots/qdrant_db/                   Qdrant snapshots + ingest.sqlite3, for restore on another machine
      cache/vlm_pages/                       OCR / Qwen3-VL / paid page readings (never paid twice)
      cache/hf/                              HF_HOME on Colab and servers
      models/                                local model copies (cross-encoder reranker, ...)
      page_audit/<sector>/*.csv
      state/conversations.sqlite3            = CONVERSATION_DB
      secrets/llm_ca.crt                     = LLM_CA_CERT when present (never in git, Drive or a bucket)
      secrets/embedding_ca.crt               = EMBEDDING_CA_CERT when present (same rule)
      logs/

| Profile | ``RAG_HOME`` (live) | Durable copy (same layout) | Qdrant |
|---|---|---|---|
| local  | ``<repo>/indexes/rag_home`` | the same folder | embedded, in ``indexes/qdrant_db`` |
| colab  | ``/content/rag`` (VM disk) | ``/content/drive/MyDrive/capstone_rag`` | server on the VM, ``127.0.0.1:6333`` |
| server | ``/app/rag_home`` in the containers (``RAG_HOME_DIR`` on the host) | the host folder | compose ``qdrant`` service |
| gcp    | as server; host folder on the VM's persistent disk | ``gs://$RAG_GCS_BUCKET`` | compose ``qdrant`` service |

**Older folder names keep working.** An index is looked up under its new name first (``qdrant_db``,
``chroma_db``) and then under the names used before (``colpali_qdrant`` / ``financial_qdrant``,
``vlm_chroma`` / ``financial_vlm``, in ``RAG_HOME``, ``RAG_HOME/indexes`` and ``<repo>/indexes``); the first one
holding an ingest state file wins. Indexes built before this layout therefore need no re-ingest or renaming.

Any variable you set yourself (``RAG_INGEST_DIR``, ``RAG_VLM_DIR``, ``CONVERSATION_DB``, ...) wins over the layout.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from scripts.common.paths import ROOT

PROFILES = ("local", "colab", "server", "gcp")
STATE_FILE = {"qdrant_db": "ingest.sqlite3", "chroma_db": "vlm_ingest.sqlite3"}
LEGACY_NAMES = {"qdrant_db": ("colpali_qdrant", "financial_qdrant"), "chroma_db": ("vlm_chroma", "financial_vlm")}
PIPELINE_KIND = {"colpali_qdrant": "qdrant_db", "vlm_chroma": "chroma_db"}     # colab_pipeline ids -> folder kind
RERANKER = "ms-marco-MiniLM-L-6-v2"
CA_CERT = "llm_ca.crt"
EMBEDDING_CA_CERT = "embedding_ca.crt"
DRIVE_ROOT = Path("/content/drive/MyDrive/capstone_rag")


def detect() -> str:
    """``RAG_PROFILE`` if set (``server`` and ``gcp`` are never guessed), else ``colab`` on Colab, else ``local``."""
    if os.getenv("RAG_PROFILE"):
        profile = os.environ["RAG_PROFILE"].strip().lower()
        if profile not in PROFILES:
            raise ValueError(f"RAG_PROFILE must be one of {', '.join(PROFILES)}")
        return profile
    on_colab = bool(os.getenv("COLAB_RELEASE_TAG")) or importlib.util.find_spec("google.colab") is not None
    return "colab" if on_colab else "local"


def default_home(profile: str) -> Path:
    if profile == "colab":
        return Path("/content/rag")
    if profile in ("server", "gcp"):
        container = Path("/app/rag_home")
        return container if container.exists() else ROOT / "indexes" / "rag_home"
    return ROOT / "indexes" / "rag_home"


def first_existing(candidates, marker: str | None = None) -> Path:
    """The first candidate that exists (holding ``marker`` when given), else the first one (the canonical name)."""
    candidates = [Path(c) for c in candidates]
    for path in candidates:
        if (path / marker).exists() if marker else path.exists():
            return path
    return candidates[0]


@dataclass(frozen=True)
class Layout:
    profile: str
    home: Path                       # RAG_HOME: the live copy
    durable: str | None = None       # Drive folder or gs:// bucket holding the same layout; None = home itself

    # ------------------------------------------------------------------ folders
    @property
    def indexes(self) -> Path:
        return self.home / "indexes"

    def index(self, kind: str, suffix: str = "") -> Path:
        """Live folder of an index (``qdrant_db`` or ``chroma_db``), with fallback to the older folder names."""
        if kind not in STATE_FILE:
            raise ValueError(f"unknown index kind {kind!r}")
        bases = [self.indexes, self.home]
        if self.profile in ("local", "server", "gcp"):
            bases.append(ROOT / "indexes")
        candidates = [self.indexes / f"{kind}{suffix}"]
        candidates += [base / f"{name}{suffix}" for base in bases for name in LEGACY_NAMES[kind]]
        return first_existing(dict.fromkeys(candidates), STATE_FILE[kind])

    @property
    def qdrant_db(self) -> Path:
        return self.index("qdrant_db")

    @property
    def chroma_db(self) -> Path:
        return self.index("chroma_db")

    @property
    def mutual_fund(self) -> Path:
        """Mutual fund data (``nport_mutual_fund.sqlite3``, ``chroma_mutual_fund/``, ``raw_mutual_fund/``);
        ``RAG_MUTUAL_FUND_DIR`` wins. Kept apart from ``chroma_db`` so the annual-report index is never touched."""
        return Path(os.getenv("RAG_MUTUAL_FUND_DIR") or self.indexes / "mutual_fund")

    @property
    def corpus(self) -> Path:
        return self.home / "annual_reports"

    @property
    def snapshots(self) -> Path:
        return self.home / "snapshots" / "qdrant_db"

    @property
    def vlm_pages(self) -> Path:
        return self.home / "cache" / "vlm_pages"

    @property
    def hf_home(self) -> Path:
        return self.home / "cache" / "hf"

    @property
    def models(self) -> Path:
        return Path(os.getenv("RAG_MODELS_DIR") or self.home / "models")

    @property
    def page_audit(self) -> Path:
        return self.home / "page_audit"

    @property
    def state(self) -> Path:
        return self.home / "state"

    @property
    def conversations(self) -> Path:
        return self.state / "conversations.sqlite3"

    @property
    def secrets(self) -> Path:
        return self.home / "secrets"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    # ------------------------------------------------------------------ optional files
    def reranker(self) -> Path | None:
        """A local copy of the cross-encoder, if there is one (``models/``; locally also ``<repo>/indexes/models``)."""
        candidates = [self.models / RERANKER]
        if self.profile == "local" and not os.getenv("RAG_MODELS_DIR"):
            candidates.append(ROOT / "indexes" / "models" / RERANKER)
        return next((p for p in candidates if (p / "config.json").exists()), None)

    def ca_cert(self) -> Path | None:
        path = self.secrets / CA_CERT
        return path if path.is_file() else None

    def embedding_ca_cert(self) -> Path | None:
        path = self.secrets / EMBEDDING_CA_CERT
        return path if path.is_file() else None

    def describe(self) -> dict:
        """Paths only (for /health and logs): no file contents, keys or secrets."""
        return {"profile": self.profile, "home": str(self.home), "durable": self.durable or str(self.home),
                "qdrant_db": str(self.qdrant_db), "chroma_db": str(self.chroma_db), "corpus": str(self.corpus),
                "snapshots": str(self.snapshots), "vlm_pages": str(self.vlm_pages), "state": str(self.state)}


def durable_for(profile: str, home: Path) -> str | None:
    if os.getenv("RAG_DURABLE"):
        return os.environ["RAG_DURABLE"]
    if profile == "colab":
        return str(DRIVE_ROOT)
    if profile == "gcp" and os.getenv("RAG_GCS_BUCKET"):
        bucket = os.environ["RAG_GCS_BUCKET"].strip()
        return bucket if bucket.startswith("gs://") else f"gs://{bucket}"
    return None


def layout(profile: str | None = None) -> Layout:
    """The layout for ``profile`` (default: ``detect()``), rooted at ``RAG_HOME`` when set."""
    profile = profile or detect()
    if profile not in PROFILES:
        raise ValueError(f"RAG_PROFILE must be one of {', '.join(PROFILES)}")
    home = Path(os.getenv("RAG_HOME") or default_home(profile))
    return Layout(profile=profile, home=home, durable=durable_for(profile, home))


def durable_index(durable_root: Path, kind: str, suffix: str = "") -> Path:
    """The Drive (or other mounted) copy of an index: snapshots/qdrant_db or indexes/chroma_db, or the older names."""
    group = "snapshots" if kind == "qdrant_db" else "indexes"
    names = [kind, *LEGACY_NAMES[kind][:1]]          # colpali_qdrant / vlm_chroma were the Drive names
    return first_existing([Path(durable_root) / group / f"{name}{suffix}" for name in names])


def work_index(work: Path, kind: str, suffix: str = "") -> Path:
    """An index folder under one working root only (``<work>/indexes/<kind>``, or ``<work>/<older name>`` when only
    that holds an index). Used by colab_pipeline, which must never pick up an index outside its own root."""
    work = Path(work)
    names = [work / "indexes" / f"{kind}{suffix}"]
    names += [base / f"{name}{suffix}" for base in (work / "indexes", work) for name in LEGACY_NAMES[kind]]
    return first_existing(dict.fromkeys(names), STATE_FILE[kind])
