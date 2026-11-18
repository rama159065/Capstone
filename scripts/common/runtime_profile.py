"""Load the right settings for where the stack runs: ``local`` (this PC), ``colab``, ``server`` (any Docker host)
or ``gcp`` (a Google Cloud VM running the compose stack).

FastAPI calls :func:`apply_profile` at import, and the notebooks do the same before starting the API and Streamlit.
Every value is a default only: an environment variable you set yourself always wins (an empty value counts as
unset). Choose a profile with ``RAG_PROFILE``; without it, Colab is detected automatically and anything else is
``local``. Paths come from one layout (:mod:`scripts.common.storage`), rooted at ``RAG_HOME``:

| Setting | local | colab | server / gcp |
|---|---|---|---|
| ``RAG_HOME`` | ``<repo>/indexes/rag_home`` | ``/content/rag`` | ``/app/rag_home`` (host: ``RAG_HOME_DIR``) |
| ``RAG_INGEST_DIR`` (pipeline 2, Qdrant) | ``<RAG_HOME>/indexes/qdrant_db`` (or an older folder holding an index) | same | same |
| ``RAG_VLM_DIR`` (pipeline 1, Chroma) | ``<RAG_HOME>/indexes/chroma_db`` (or an older folder) | same | same |
| ``QDRANT_URL`` | unset: embedded Qdrant (one process at a time) | ``http://127.0.0.1:6333`` | ``http://qdrant:6333`` |
| ``CONVERSATION_DB`` | ``<RAG_HOME>/state/conversations.sqlite3`` | same | same |
| ``HF_HOME`` | your usual cache (unchanged) | ``<RAG_HOME>/cache/hf`` | the image's ``/app/indexes/hf_cache`` |
| ``RAG_DEVICE`` | ``cpu`` | ``cuda`` when a GPU is present | ``cpu`` |
| ``API_BASE_URL`` (Streamlit -> FastAPI) | ``http://127.0.0.1:8000`` | same | ``http://api:8000`` |
| ``RAG_RERANKER_MODEL`` | the local cross-encoder copy, when there is one | same | same |
| ``LLM_CA_CERT`` | ``<RAG_HOME>/secrets/llm_ca.crt``, when it exists | same | same (compose mounts ``secrets/``) |
| ``EMBEDDING_CA_CERT`` | ``<RAG_HOME>/secrets/embedding_ca.crt``, when it exists | same | same |
| ``RAG_EMBEDDING_PROVIDER`` (pipeline 1, bge-small) | unset: the provider that built the index (``local``) | same | ``hosted`` (the shared API; local model if unreachable) |
| durable copy | ``RAG_HOME`` itself | ``MyDrive/capstone_rag`` | host folder (server), ``gs://$RAG_GCS_BUCKET`` (gcp) |

``RAG_BACKEND`` for financial QA: ``financial_qdrant`` once its ingest exists, else ``financial_vlm`` once that one
exists, else the older Chroma index. Set it yourself to choose (for example one API per pipeline).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from scripts.common import storage
from scripts.common.storage import detect  # noqa: F401 - public: runtime_profile.detect()

LOGGER = logging.getLogger(__name__)
PROFILES = {profile: {} for profile in storage.PROFILES}       # kept for callers that list the profile names
SERVICES = {
    "local": {"RAG_DEVICE": "cpu", "API_BASE_URL": "http://127.0.0.1:8000"},
    "colab": {"QDRANT_URL": "http://127.0.0.1:6333", "API_BASE_URL": "http://127.0.0.1:8000"},
    # Deployed servers embed questions with the shared embedding API; an index of bge-small built locally (or on
    # Colab) is served with it, and the local model takes over if the API can't be reached.
    "server": {"QDRANT_URL": "http://qdrant:6333", "RAG_DEVICE": "cpu", "API_BASE_URL": "http://api:8000",
               "RAG_EMBEDDING_PROVIDER": "hosted"},
    "gcp": {"QDRANT_URL": "http://qdrant:6333", "RAG_DEVICE": "cpu", "API_BASE_URL": "http://api:8000",
            "RAG_EMBEDDING_PROVIDER": "hosted"},
}
_applied: dict | None = None


def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - torch missing or broken: treat as no GPU
        return False


def defaults_for(profile: str) -> dict[str, str]:
    """Every default the profile would set (paths from the storage layout, services from SERVICES)."""
    lay = storage.layout(profile)
    values = {"RAG_HOME": str(lay.home)}
    # Index folders: an explicit RAG_HOME or layout decides; an index under an older folder name is found too.
    values.update(RAG_INGEST_DIR=str(lay.qdrant_db), RAG_VLM_DIR=str(lay.chroma_db),
                  CONVERSATION_DB=str(lay.conversations))
    values.update(SERVICES[profile])
    if profile == "colab":
        values["HF_HOME"] = str(lay.hf_home)
        values["RAG_DEVICE"] = "cuda" if gpu_available() else "cpu"
    reranker = lay.reranker()
    if reranker:
        values["RAG_RERANKER_MODEL"] = str(reranker)
    cert = lay.ca_cert()
    if cert:
        values["LLM_CA_CERT"] = str(cert)
    embedding_cert = lay.embedding_ca_cert()
    if embedding_cert:
        values["EMBEDDING_CA_CERT"] = str(embedding_cert)
    return values


def apply_profile(profile=None, *, force=False):
    """Fill unset environment variables for the profile; returns what was decided (idempotent)."""
    global _applied
    if _applied is not None and not force and profile in (None, _applied["profile"]):
        return _applied
    profile = profile or detect()
    if profile not in SERVICES:
        raise ValueError(f"RAG_PROFILE must be one of {', '.join(SERVICES)}")
    chosen = {}
    for key, value in defaults_for(profile).items():
        if not os.getenv(key):
            os.environ[key] = value
        chosen[key] = os.environ[key]
    index_ready = (Path(os.environ["RAG_INGEST_DIR"]) / "ingest.sqlite3").exists()
    vlm_ready = (Path(os.environ["RAG_VLM_DIR"]) / "vlm_ingest.sqlite3").exists()
    if not os.getenv("RAG_BACKEND") and (index_ready or vlm_ready):
        os.environ["RAG_BACKEND"] = "financial_qdrant" if index_ready else "financial_vlm"
    backend = os.getenv("RAG_BACKEND") or "chroma"
    lay = storage.layout(profile)
    _applied = {"profile": profile, "financial_backend": backend, "index_ready": index_ready, "vlm_index_ready": vlm_ready,
                "qdrant": os.getenv("QDRANT_URL") or "embedded", "durable": lay.durable or os.environ["RAG_HOME"],
                "reranker": bool(os.getenv("RAG_RERANKER_MODEL")), "ca_cert": bool(os.getenv("LLM_CA_CERT")), **chosen}
    if backend == "chroma":
        LOGGER.info("runtime_profile=%s financial_backend=chroma (no ingest under %s or %s yet)",
                    profile, os.environ["RAG_INGEST_DIR"], os.environ["RAG_VLM_DIR"])
    else:
        LOGGER.info("runtime_profile=%s financial_backend=%s qdrant=%s home=%s", profile, backend, _applied["qdrant"],
                    os.environ["RAG_HOME"])
    return _applied


def describe():
    """Safe summary for /health: paths and switches only, no keys, certificates or secrets."""
    info = apply_profile()
    return {key: info.get(key) for key in ("profile", "financial_backend", "index_ready", "vlm_index_ready", "qdrant",
                                           "RAG_HOME", "durable", "RAG_INGEST_DIR", "RAG_VLM_DIR", "reranker", "ca_cert")}
