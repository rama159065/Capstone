"""Actual Qdrant integration with deterministic test vectors, not model accuracy tests."""
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from qdrant_client import models

from scripts.rag import multimodal_rag as mm


def vector(axis):
    result = [0.0] * 128
    result[axis] = 1.0
    return result


class TestEncoder:
    __test__ = False
    signature = "deterministic-test-vectors"
    dimensions = 128

    def __init__(self):
        self.calls = 0

    def embed_image(self, image):
        self.calls += 1
        page = int(Path(image).stem.split("_")[-1])
        return [vector(page), vector(20), vector(21)]

    def embed_query(self, query):
        return [vector(2 if "table" in query else 3), vector(20)]


def sample():
    return next((mm.ROOT / "data/samples/multimodal/hr_policy").glob("*.pdf"))


def test_visual_multivectors_written_and_maxsim_retrieves_chart(tmp_path):
    encoder = TestEncoder()
    rag = mm.LocalRAG(tmp_path / "index", colpali_encoder=encoder)
    try:
        rag.ingest("hr_policy", sample())
        assert rag.client.count("visual_pages", exact=True).count == 3
        config = rag.client.get_collection("visual_pages").config.params.vectors["colpali"]
        assert config.size == 128
        assert config.multivector_config.comparator == models.MultiVectorComparator.MAX_SIM
        assert config.hnsw_config.m == 0
        points, _ = rag.client.scroll("visual_pages", with_vectors=True)
        assert all(len(p.vector["colpali"]) == 3 for p in points)
        assert all(len(row) == 128 for p in points for row in p.vector["colpali"])
        chart = rag.retrieve_visual("hr_policy", "training chart", 1)[0]
        assert chart["page"] == 3
        assert chart["colpali_score"] == pytest.approx(2.0)
        assert rag.retrieve_visual("hr_policy", "training table", 1)[0]["page"] == 2
        assert rag.retrieve_visual("financial_report_qa", "chart") == []
        assert rag.retrieve_visual("hr_policy", "chart", ticker="UNKNOWN") == []
        merged = rag.retrieve("hr_policy", "bar chart trend graph", 1)[0]
        assert "visual" in merged["retrieval_paths"]
        assert merged["colpali_score"] is not None
    finally:
        rag.close()
    reopened = mm.LocalRAG(tmp_path / "index", colpali_encoder=encoder)
    try:
        assert reopened.client.count("visual_pages", exact=True).count == 3
        assert reopened.retrieve_visual("hr_policy", "table", 1)[0]["page"] == 2
    finally:
        reopened.close()


def test_backfill_existing_pages_is_idempotent(tmp_path):
    root = tmp_path / "index"
    rag = mm.LocalRAG(root)
    rag.ingest("hr_policy", sample())
    rag.close()
    encoder = TestEncoder()
    rag = mm.LocalRAG(root, colpali_encoder=encoder)
    try:
        assert rag.backfill_visual()["visual_pages_written"] == 3
        assert rag.backfill_visual()["visual_pages_skipped"] == 3
        assert encoder.calls == 3
        assert rag.backfill_visual(force=True)["visual_pages_written"] == 3
        assert rag.client.count("docs_text", exact=True).count == 4
        assert rag.client.count("visual_pages", exact=True).count == 3
    finally:
        rag.close()


def test_changed_document_and_failure_clear_both_collections(tmp_path):
    import shutil
    path = tmp_path / "changing.pdf"
    shutil.copyfile(sample(), path)
    encoder = TestEncoder()
    rag = mm.LocalRAG(tmp_path / "index", colpali_encoder=encoder)
    try:
        rag.ingest("hr_policy", path)
        with pymupdf.open() as pdf:
            page = pdf.new_page()
            page.insert_text((50, 50), "New text-only policy.")
            pdf.save(path)
        rag.ingest("hr_policy", path)
        assert rag.client.count("docs_text", exact=True).count == 1
        assert rag.client.count("visual_pages", exact=True).count == 0
        def fail(image):
            raise RuntimeError("encoder unavailable")
        encoder.embed_image = fail
        with pytest.raises(RuntimeError, match="encoder unavailable"):
            rag.ingest("hr_policy", sample())
        assert rag.client.count("visual_pages", exact=True).count == 0
        assert rag.client.count("docs_text", exact=True).count == 1
    finally:
        rag.close()


def test_colpali_api_and_model_signature(tmp_path, monkeypatch):
    from scripts.api import api_app
    root = tmp_path / "index"
    rag = mm.LocalRAG(root, colpali_encoder=TestEncoder())
    try:
        rag.ingest("financial_report_qa", sample())
        monkeypatch.setattr(mm, "_instance", rag)
        client = TestClient(api_app.app)
        response = client.post("/multimodal/financial_report_qa/visual/retrieve",
                               json={"query": "training table", "top_k": 1})
        assert response.status_code == 200
        assert response.json()["retriever"] == "colpali_maxsim"
        assert response.json()["evidence"][0]["page"] == 2
    finally:
        rag.close()
    changed = TestEncoder()
    changed.signature = "different-model"
    with pytest.raises(ValueError, match="ColPali configuration differs"):
        mm.LocalRAG(root, colpali_encoder=changed)
