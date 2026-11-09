"""Load the right settings for where the stack runs: this PC (``local``) or Google Colab (``colab``).

FastAPI calls :func:`apply_profile` at import, and the notebook does the same before
starting the API and Streamlit. Every value is a default only: an environment variable
you set yourself always wins. Choose a profile with ``RAG_PROFILE=local|colab``; without
it, Colab is detected automatically.

| Setting | local | colab |
|---|---|---|
| ``RAG_INGEST_DIR`` (ColPali/Qdrant index: state, page cache) | ``indexes/financial_qdrant`` | ``/content/rag/colpali_qdrant`` |
| ``RAG_VLM_DIR`` (VLM/Chroma index) | ``indexes/financial_vlm`` | ``/content/rag/vlm_chroma`` |
| ``QDRANT_URL`` | unset: embedded Qdrant in ``RAG_INGEST_DIR`` (one process at a time) | ``http://127.0.0.1:6333`` (server started by the notebook) |
| ``RAG_BACKEND`` for financial QA | ``financial_qdrant`` once its ingest exists, else ``financial_vlm`` once that one exists, else the older Chroma index | same |

The Colab folders are the ones ``scripts/rag/colab_pipeline.py`` writes (``<work>/<pipeline>``), so the API
serves the corpus that notebook ingested. The older ``colab_financial_ingest.ipynb`` sets ``RAG_INGEST_DIR``
itself, which still wins.
| ``RAG_DEVICE`` (text models) | ``cpu`` | ``cuda`` if a GPU is present |
| ``API_BASE_URL`` (Streamlit → FastAPI) | ``http://127.0.0.1:8000`` | ``http://127.0.0.1:8000`` (same VM) |
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

from scripts.common.paths import ROOT  # noqa: E402 - repo root
LOGGER = logging.getLogger(__name__)
PROFILES = {
    "local": {"RAG_INGEST_DIR": str(ROOT / "indexes/financial_qdrant"), "RAG_VLM_DIR": str(ROOT / "indexes/financial_vlm"),
              "RAG_DEVICE": "cpu", "API_BASE_URL": "http://127.0.0.1:8000"},
    "colab": {"RAG_INGEST_DIR": "/content/rag/colpali_qdrant", "RAG_VLM_DIR": "/content/rag/vlm_chroma",
              "QDRANT_URL": "http://127.0.0.1:6333",
              "API_BASE_URL": "http://127.0.0.1:8000", "HF_HOME": "/content/hf_cache"},
}
_applied: dict | None = None


def detect():
    if os.getenv("RAG_PROFILE"):
        profile = os.environ["RAG_PROFILE"].strip().lower()
        if profile not in PROFILES:
            raise ValueError(f"RAG_PROFILE must be one of {', '.join(PROFILES)}")
        return profile
    on_colab = bool(os.getenv("COLAB_RELEASE_TAG")) or importlib.util.find_spec("google.colab") is not None
    return "colab" if on_colab else "local"


def gpu_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - torch missing or broken: treat as no GPU
        return False


def apply_profile(profile=None, *, force=False):
    """Fill unset environment variables for the profile; returns what was decided (idempotent)."""
    global _applied
    if _applied is not None and not force and profile in (None, _applied["profile"]):
        return _applied
    profile = profile or detect()
    defaults = dict(PROFILES[profile])
    if profile == "colab" and gpu_available():
        defaults["RAG_DEVICE"] = "cuda"
    chosen = {}
    for key, value in defaults.items():
        if not os.getenv(key):
            os.environ[key] = value
        chosen[key] = os.environ[key]
    index_ready = (Path(os.environ["RAG_INGEST_DIR"]) / "ingest.sqlite3").exists()
    vlm_ready = (Path(os.environ["RAG_VLM_DIR"]) / "vlm_ingest.sqlite3").exists()
    if not os.getenv("RAG_BACKEND") and (index_ready or vlm_ready):
        os.environ["RAG_BACKEND"] = "financial_qdrant" if index_ready else "financial_vlm"
    backend = os.getenv("RAG_BACKEND") or "chroma"
    _applied = {"profile": profile, "financial_backend": backend, "index_ready": index_ready, "vlm_index_ready": vlm_ready,
                "qdrant": os.getenv("QDRANT_URL") or "embedded", **chosen}
    if backend == "chroma":
        LOGGER.info("runtime_profile=%s financial_backend=chroma (no ingest under %s or %s yet)",
                    profile, os.environ["RAG_INGEST_DIR"], os.environ["RAG_VLM_DIR"])
    else:
        LOGGER.info("runtime_profile=%s financial_backend=%s qdrant=%s", profile, backend, _applied["qdrant"])
    return _applied


def describe():
    """Safe summary for /health: no keys or secrets."""
    info = apply_profile()
    return {key: info[key] for key in ("profile", "financial_backend", "index_ready", "vlm_index_ready", "qdrant",
                                       "RAG_INGEST_DIR", "RAG_VLM_DIR")}
