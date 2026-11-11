from __future__ import annotations

import json

import scripts.downloads.download_rag_corpora as corpora


def test_text_and_filename_helpers() -> None:
    assert corpora.safe_filename("A/B C?!") == "A_B_C"
    assert corpora.normalize_text("<p>Hello&nbsp;world</p>\r\n\n\nNext") == "Hello world\n\nNext"
    assert corpora.sha256_text("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_corpus_writer_writes_document_and_skips_duplicate(tmp_path) -> None:
    writer = corpora.CorpusWriter(tmp_path, "financial_report_qa")
    doc = corpora.RawDocument(
        domain="financial_report_qa",
        document_id="doc-1",
        title="Test Doc",
        text="Useful text",
        source_name="unit",
        source_url="https://example.test",
        metadata={"source_type": "unit"},
    )

    assert writer.write(doc) is True
    assert writer.write(doc) is False
    assert writer.count == 1

    written = tmp_path / "financial_report_qa" / "documents" / "doc-1.json"
    manifest = tmp_path / "financial_report_qa" / "manifest.jsonl"
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert written.exists()
    assert manifest.exists()
    assert payload["title"] == "Test Doc"
    assert payload["metadata"]["source_type"] == "unit"


def test_helpers_and_financial_only_dispatch(tmp_path) -> None:
    import pytest
    assert corpora.safe_list_get(["a"], 3, "missing") == "missing"
    assert list(corpora.DOMAINS) == ["financial_report_qa"]
    with pytest.raises(ValueError, match="Unsupported domain"):
        corpora.fetch_domain("hr_policy", corpora.CorpusWriter(tmp_path, "hr_policy"), 1, "ua")
