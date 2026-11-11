"""Qdrant annual_reports_text and Chroma annual_reports_vlm store the same chunk schema and return the same evidence keys."""
import hashlib
import json

import pytest

from scripts.rag import chunk_schema
from scripts.rag import financial_ingest_chroma_db as chroma
from scripts.rag import financial_ingest_qdrant_db as qdrant
from tests.test_financial_ingest_qdrant_db import args_for as qdrant_args, make_pdf
from tests.test_financial_ingest_chroma_db import args_for as chroma_args


@pytest.fixture
def corpus(tmp_path):
    folder = tmp_path / "corpus" / "banking"
    folder.mkdir(parents=True)
    pdf = folder / "EDGAR_ARS_TEST_2025.pdf"
    make_pdf(pdf)
    row = {"document_id": "edgar_ars_TEST_2025", "company": "Test Bank", "ticker": "TEST", "year": 2025,
           "title": "Test Bank annual report 2025", "source_url": "https://example.invalid/x.pdf",
           "sector": "Banking", "sector_slug": "banking", "local_path": pdf.name,
           "content_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    (folder / "pdf_manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return tmp_path / "corpus"


def test_both_indexes_store_the_same_fields_and_return_the_same_evidence(tmp_path, corpus, monkeypatch):
    client = qdrant.connect(None, tmp_path / "root")
    run = qdrant.Ingest(qdrant_args(tmp_path, corpus, stages="text"), client=client)
    assert run.run() == {"complete": 1}
    q_payloads = [p.payload for p in client.scroll(qdrant.TEXT, limit=200, with_payload=True)[0]]

    c_run = chroma.VlmIngest(chroma_args(tmp_path, corpus, "chroma", "--vlm", "routed", "--qwen-reader", "mock"))
    assert c_run.run() == {"complete": 1}
    c_meta = c_run.store.collection.get(include=["metadatas", "documents"])
    c_payloads = [chunk_schema.normalize({**m, "text": d}) for m, d in zip(c_meta["metadatas"], c_meta["documents"])]

    assert q_payloads and c_payloads
    for payload in q_payloads + c_payloads:
        assert set(chunk_schema.FIELDS) <= set(payload), set(chunk_schema.FIELDS) - set(payload)
    assert {p["pipeline"] for p in q_payloads} == {"qdrant_db"} and {p["pipeline"] for p in c_payloads} == {"chroma_db"}
    for name in chunk_schema.LIST_FIELDS:                      # lists in both, decoded from Chroma's JSON strings
        assert all(isinstance(p[name], list) for p in q_payloads + c_payloads)
    same_page = lambda payloads: next(p for p in payloads if p["page"] == 1)
    for field in chunk_schema.FILING_FIELDS + ("page", "section", "page_type", "scale", "garbled", "text_source", "reader"):
        assert same_page(q_payloads)[field] == same_page(c_payloads)[field], field

    monkeypatch.setattr(qdrant, "_serving", {"client": client, "embedder": run.embedder})
    q_hit = qdrant.retrieve_for_agent("total net sales 391,035", top_k=1, ticker="TEST")[0]
    c_hit = chroma.retrieve("total net sales 391,035", 1, root=tmp_path / "chroma", embedder=c_run.embedder)[0]
    assert set(q_hit) == set(c_hit) == set(chunk_schema.EVIDENCE_KEYS)
    assert set(q_hit["metadata"]) >= set(chunk_schema.FIELDS) and set(c_hit["metadata"]) >= set(chunk_schema.FIELDS)


def test_normalize_decodes_chroma_metadata():
    stored = {"chunk_id": "c1", "text": "x", "unverified_numbers": '["120,580"]', "table_markdown": "[]"}
    out = chunk_schema.normalize(stored)
    assert out["unverified_numbers"] == ["120,580"] and out["table_markdown"] == [] and out["unverified_in_chunk"] == []
    assert out["reader"] is None and set(chunk_schema.FIELDS) <= set(out)
