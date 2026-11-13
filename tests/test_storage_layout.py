"""One storage layout for local, Colab, Docker and GCP (scripts/common/storage.py), the profiles built on it and
colab_pipeline's use of it. Nothing outside tmp_path is read or written."""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import pytest
import yaml

from scripts.common import runtime_profile as rp
from scripts.common import storage
from scripts.common.paths import ROOT

KEYS = ("RAG_PROFILE", "RAG_HOME", "RAG_INGEST_DIR", "RAG_VLM_DIR", "QDRANT_URL", "RAG_BACKEND", "RAG_DEVICE",
        "API_BASE_URL", "HF_HOME", "CONVERSATION_DB", "RAG_RERANKER_MODEL", "LLM_CA_CERT", "RAG_GCS_BUCKET",
        "RAG_DURABLE", "RAG_MODELS_DIR", "COLAB_RELEASE_TAG")


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(rp, "_applied", None)
    monkeypatch.setattr(rp, "gpu_available", lambda: False)
    with mock.patch.dict(os.environ, {}, clear=False):
        for key in KEYS:
            os.environ.pop(key, None)
        os.environ["RAG_HOME"] = str(tmp_path / "home")
        os.environ["RAG_MODELS_DIR"] = str(tmp_path / "home" / "models")
        yield os.environ


def make_index(folder: Path, kind: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / storage.STATE_FILE[kind]).write_bytes(b"")
    return folder


@pytest.mark.parametrize("profile", storage.PROFILES)
def test_every_profile_has_the_same_structure_under_its_home(env, profile, tmp_path):
    lay = storage.layout(profile)
    home = tmp_path / "home"
    assert lay.home == home
    assert lay.qdrant_db == home / "indexes" / "qdrant_db" and lay.chroma_db == home / "indexes" / "chroma_db"
    assert lay.conversations == home / "state" / "conversations.sqlite3"
    assert lay.snapshots == home / "snapshots" / "qdrant_db" and lay.vlm_pages == home / "cache" / "vlm_pages"


def test_default_homes_per_environment(env):
    env.pop("RAG_HOME")
    assert storage.layout("local").home == ROOT / "indexes" / "rag_home"
    assert storage.layout("colab").home == Path("/content/rag")
    assert storage.layout("colab").durable.replace("\\", "/").endswith("MyDrive/capstone_rag")
    env["RAG_GCS_BUCKET"] = "capstone-rag-demo"
    assert storage.layout("gcp").durable == "gs://capstone-rag-demo"
    assert storage.layout("server").durable is None                 # the host folder is the durable copy


def test_older_index_folders_are_found_without_renaming(env, tmp_path):
    home = tmp_path / "home"
    old_qdrant = make_index(home / "colpali_qdrant", "qdrant_db")          # the former colab_pipeline name
    old_chroma = make_index(home / "indexes" / "financial_vlm", "chroma_db")
    lay = storage.layout("local")
    assert lay.qdrant_db == old_qdrant and lay.chroma_db == old_chroma
    new_qdrant = make_index(home / "indexes" / "qdrant_db", "qdrant_db")
    assert storage.layout("local").qdrant_db == new_qdrant                 # the new name wins once it holds an index


def test_local_profile_serves_what_the_local_pipeline_builds(env, tmp_path):
    from scripts.rag import colab_pipeline as cp
    s = cp.Settings(pipeline="vlm_chroma", work=tmp_path / "home", drive_root=tmp_path / "drive")
    make_index(s.root, "chroma_db")
    info = rp.apply_profile("local", force=True)
    assert Path(info["RAG_VLM_DIR"]) == s.root == tmp_path / "home" / "indexes" / "chroma_db"
    assert info["financial_backend"] == "financial_vlm"
    assert Path(cp.serving_env(s)["RAG_VLM_DIR"]) == s.root


def test_profiles_fill_services_state_reranker_and_certificate(env, tmp_path):
    home = tmp_path / "home"
    (home / "models" / storage.RERANKER).mkdir(parents=True)
    (home / "models" / storage.RERANKER / "config.json").write_text("{}")
    (home / "secrets").mkdir()
    (home / "secrets" / storage.CA_CERT).write_text("-----BEGIN CERTIFICATE-----")
    env["RAG_GCS_BUCKET"] = "gs://capstone-rag-demo"
    info = rp.apply_profile("gcp", force=True)
    assert info["QDRANT_URL"] == "http://qdrant:6333" and info["API_BASE_URL"] == "http://api:8000"
    assert info["durable"] == "gs://capstone-rag-demo" and info["reranker"] and info["ca_cert"]
    assert Path(env["CONVERSATION_DB"]) == home / "state" / "conversations.sqlite3"
    assert Path(env["RAG_RERANKER_MODEL"]) == home / "models" / storage.RERANKER
    described = rp.describe()
    assert "BEGIN CERTIFICATE" not in json.dumps(described) and described["durable"] == "gs://capstone-rag-demo"


def test_explicit_settings_win_and_an_empty_value_counts_as_unset(env, tmp_path):
    env.update(RAG_VLM_DIR=str(tmp_path / "mine"), LLM_CA_CERT="", QDRANT_URL="http://10.0.0.9:6333")
    info = rp.apply_profile("server", force=True)
    assert info["RAG_VLM_DIR"] == str(tmp_path / "mine") and info["QDRANT_URL"] == "http://10.0.0.9:6333"
    assert not info["ca_cert"]                                            # no certificate file: stays unset


def test_colab_pipeline_reuses_older_drive_copies(env, tmp_path):
    from scripts.rag import colab_pipeline as cp
    drive = tmp_path / "drive"
    (drive / "snapshots" / "colpali_qdrant").mkdir(parents=True)
    (drive / "vlm_pages").mkdir()
    qdrant = cp.Settings(pipeline="colpali_qdrant", work=tmp_path / "work", drive_root=drive, qdrant_url=None)
    chroma = cp.Settings(pipeline="vlm_chroma", work=tmp_path / "work", drive_root=drive)
    assert qdrant.backup == drive / "snapshots" / "colpali_qdrant"         # existing Drive copy kept
    assert chroma.backup == drive / "indexes" / "chroma_db"                # nothing there yet: the new name
    assert chroma.vlm_cache == drive / "vlm_pages"
    assert qdrant.root == tmp_path / "work" / "indexes" / "qdrant_db"


def test_export_to_gcs_copies_the_durable_indexes(env, tmp_path):
    from scripts.rag import colab_pipeline as cp
    drive = tmp_path / "drive"
    (drive / "indexes" / "vlm_chroma").mkdir(parents=True)
    (drive / "snapshots" / "qdrant_db").mkdir(parents=True)
    s = cp.Settings(pipeline="vlm_chroma", work=tmp_path / "work", drive_root=drive)
    commands = cp.export_to_gcs(s, "capstone-rag-demo", dry_run=True)
    assert [c[-1] for c in commands] == ["gs://capstone-rag-demo/indexes/chroma_db",
                                        "gs://capstone-rag-demo/snapshots/qdrant_db"]
    assert commands[0][-2] == str(drive / "indexes" / "vlm_chroma")
    with pytest.raises(ValueError):
        cp.export_to_gcs(s, "", dry_run=True)


def test_compose_mounts_the_layout_and_secrets():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    api = compose["services"]["api"]
    assert api["environment"]["RAG_HOME"] == "/app/rag_home"
    assert api["environment"]["RAG_PROFILE"] == "${RAG_PROFILE:-server}"
    volumes = " ".join(api["volumes"])
    assert ":/app/rag_home" in volumes and "/app/rag_home/secrets:ro" in volumes
    for script in ("deploy/restore_index.sh", "deploy/sync_from_gcs.sh"):
        assert (ROOT / script).exists(), script
