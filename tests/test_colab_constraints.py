"""Colab installs the versions tested on the PC: deploy/colab_constraints.txt must match uv.lock and pyproject."""
from __future__ import annotations

import json
import tomllib

from scripts.common import colab_constraints as cc
from scripts.common.paths import ROOT


def test_constraints_file_matches_uv_lock() -> None:
    assert cc.main(["--check"]) == 0, "run: python -m scripts.common.colab_constraints"


def test_chromadb_pin_matches_the_index_writer_and_pyproject() -> None:
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["dependencies"]
    assert "chromadb==1.5.9" in deps
    assert cc.pinned()["chromadb"] == "1.5.9"


def test_colab_runtime_packages_are_left_alone() -> None:
    names = set(cc.pinned())
    for name in names:
        assert not name.startswith(("torch", "nvidia", "cuda", "triton", "jupyter", "ipykernel", "tornado")), name
    assert not names & {"numpy", "pandas", "scikit-learn", "matplotlib", "protobuf", "pillow"}
    assert {"langsmith", "langgraph", "langchain-core", "fastapi", "streamlit", "transformers"} <= names


def test_every_colab_install_uses_the_constraints() -> None:
    for name in ("colab_ingest_chroma_db.ipynb", "colab_ingest_qdrant_db.ipynb", "colab_rag_pipelines.ipynb"):
        nb = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            source = "".join(cell["source"])
            for line in source.splitlines():
                if line.lstrip().startswith("!pip") and " install" in line:
                    assert "{PIN}" in line, (name, line)
                if '"pip", "install"' in line:
                    assert "pin" in line or "colab_constraints" in line, (name, line)
    chroma = (ROOT / "notebooks" / "colab_ingest_chroma_db.ipynb").read_text(encoding="utf-8")
    assert "version_mismatches" in chroma and "chromadb must be 1.5.9" in chroma
