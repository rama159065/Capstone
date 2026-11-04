import os
from unittest import mock

import pytest

from scripts.common import runtime_profile as rp

KEYS = ("RAG_PROFILE", "RAG_INGEST_DIR", "QDRANT_URL", "RAG_BACKEND", "RAG_DEVICE", "API_BASE_URL",
        "HF_HOME", "COLAB_RELEASE_TAG")


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(rp, "_applied", None)        # restored after the test, so later imports are unaffected
    with mock.patch.dict(os.environ, {}, clear=False):
        for key in KEYS:
            os.environ.pop(key, None)
        yield os.environ


def test_local_profile_keeps_chroma_until_an_ingest_exists(env, tmp_path):
    env["RAG_PROFILE"], env["RAG_INGEST_DIR"] = "local", str(tmp_path)
    info = rp.apply_profile(force=True)
    assert info["profile"] == "local" and info["financial_backend"] == "chroma" and info["qdrant"] == "embedded"
    (tmp_path / "ingest.sqlite3").write_bytes(b"")
    assert rp.apply_profile(force=True)["financial_backend"] == "financial_qdrant"


def test_colab_profile_defaults_and_explicit_values_win(env, monkeypatch):
    monkeypatch.setattr(rp, "gpu_available", lambda: True)
    env["COLAB_RELEASE_TAG"] = "release-colab-test"
    env["QDRANT_URL"] = "http://10.0.0.5:6333"
    info = rp.apply_profile(force=True)
    assert info["profile"] == "colab" and info["RAG_INGEST_DIR"] == "/content/rag"
    assert info["qdrant"] == "http://10.0.0.5:6333" and env["RAG_DEVICE"] == "cuda"


def test_explicit_backend_and_bad_profile(env, tmp_path):
    env.update(RAG_PROFILE="local", RAG_INGEST_DIR=str(tmp_path), RAG_BACKEND="qdrant")
    (tmp_path / "ingest.sqlite3").write_bytes(b"")
    assert rp.apply_profile(force=True)["financial_backend"] == "qdrant"     # never overrides a choice
    env["RAG_PROFILE"] = "cloud"
    with pytest.raises(ValueError):
        rp.apply_profile(force=True)
