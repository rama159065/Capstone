from types import SimpleNamespace

import pymupdf
import pytest

from scripts.rag.multimodal_rag import LocalRAG, ROOT
from scripts.rag.selective_ocr import SelectiveOCR

SAMPLE = ROOT / "data/samples/ocr_validation/financial_report_qa/selective_ocr_sample.pdf"


class Recognizer:
    def __init__(self):
        self.calls = []

    def __call__(self, image):
        self.calls.append(image.size)
        return SimpleNamespace(txts=["Hidden reference OCRSECRET6789", "unreliable"],
            scores=[.99, .2], boxes=[[[20, 20], [400, 20], [400, 60], [20, 60]],
                                    [[20, 80], [400, 80], [400, 120], [20, 120]]])


def test_routing_skips_native_and_crops_mixed_image_region():
    recognizer = Recognizer()
    ocr = SelectiveOCR(recognizer)
    with pymupdf.open(SAMPLE) as pdf:
        assert ocr.extract(pdf[0]).mode == "none"
        assert recognizer.calls == []
        scan = ocr.extract(pdf[1])
        assert scan.mode == "full_page"
        assert "OCRSECRET6789" in scan.text
        assert "unreliable" not in scan.text
        assert scan.rejected_lines == 1
        mixed = ocr.extract(pdf[2])
        assert mixed.mode == "image_regions"
        assert len(mixed.regions) == 1
        assert mixed.blocks[0]["polygon"][0][0] > 50
        assert mixed.blocks[0]["polygon"][0][1] > 300
        assert recognizer.calls[-1][1] < recognizer.calls[0][1]


def test_native_duplicate_lines_are_not_added_again():
    recognizer = lambda image: SimpleNamespace(txts=["Maintenance field guide"], scores=[.99],
        boxes=[[[0, 0], [100, 0], [100, 30], [0, 30]]])
    with pymupdf.open(SAMPLE) as pdf:
        result = SelectiveOCR(recognizer).extract(pdf[2])
    assert result.text == ""
    assert result.blocks[0]["included_in_text"] is False


def test_disabled_ocr_and_blank_page_never_call_engine(monkeypatch):
    def fail(_):
        raise AssertionError("OCR should not run")
    monkeypatch.setenv("RAG_OCR_ENABLED", "0")
    with pymupdf.open(SAMPLE) as pdf:
        assert SelectiveOCR(fail).extract(pdf[1]).mode == "none"
    monkeypatch.setenv("RAG_OCR_ENABLED", "1")
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        assert SelectiveOCR(fail).extract(page).mode == "none"


def test_existing_searchable_ocr_layer_skips_recognition():
    def fail(_):
        raise AssertionError("Existing text layer should be sufficient")
    with pymupdf.open(SAMPLE) as source, pymupdf.open() as pdf:
        pdf.insert_pdf(source, from_page=1, to_page=1)
        page = pdf[0]
        page.insert_textbox(pymupdf.Rect(45, 60, 560, 600),
            ("This scanned document already contains a complete searchable OCR text layer with readable invoice and payment information. " * 5),
            fontsize=17, render_mode=3)
        assert SelectiveOCR(fail).extract(page).mode == "none"


def test_rotated_image_page_uses_full_page_and_pdf_coordinates():
    with pymupdf.open(SAMPLE) as pdf:
        page = pdf[2]
        page.set_rotation(90)
        result = SelectiveOCR(Recognizer()).extract(page)
        assert result.mode == "full_page"
        assert all(0 <= x <= 612 and 0 <= y <= 792 for block in result.blocks for x, y in block["polygon"])


def test_ocr_ingestion_persists_and_retrieves_text(tmp_path):
    rag = LocalRAG(tmp_path / "index", ocr=SelectiveOCR(Recognizer()))
    try:
        result = rag.ingest("financial_report_qa", SAMPLE)
        assert result["ocr_pages"] == 2
        hits = rag.retrieve("financial_report_qa", "OCRSECRET6789", 3)
        assert any("OCRSECRET6789" in h["text"] for h in hits)
        scan = next(h for h in hits if h["page"] == 2)
        assert scan["ocr_applied"] and scan["ocr_mode"] == "full_page"
        assert scan["ocr_blocks"][0]["confidence"] == .99
        assert "ocr" in scan["extraction_methods"]
        assert rag.ingest("financial_report_qa", SAMPLE)["status"] == "skipped"
    finally:
        rag.close()
    reopened = LocalRAG(tmp_path / "index", ocr=SelectiveOCR(Recognizer()))
    try:
        assert any("OCRSECRET6789" in h["text"] for h in reopened.retrieve("financial_report_qa", "OCRSECRET6789"))
    finally:
        reopened.close()


def test_backfill_preserves_colpali_and_is_idempotent(tmp_path, monkeypatch):
    class Encoder:
        signature = "test-visual"
        def embed_image(self, path):
            return [[1.0] + [0.0]*127, [0.0, 1.0] + [0.0]*126]
    monkeypatch.setenv("RAG_OCR_ENABLED", "0")
    rag = LocalRAG(tmp_path / "index", colpali_encoder=Encoder())
    try:
        rag.ingest("financial_report_qa", SAMPLE)
        before, _ = rag.client.scroll("visual_pages", with_vectors=True)
        matrices = {p.id: p.vector["colpali"] for p in before}
        summaries = {p.id: p.vector["summary"] for p in before}
        monkeypatch.setenv("RAG_OCR_ENABLED", "1")
        rag.ocr = SelectiveOCR(Recognizer())
        stats = rag.backfill_ocr()
        assert stats["pages_text_enriched"] == 2
        assert stats["visual_summaries_updated"] == 2
        after, _ = rag.client.scroll("visual_pages", with_vectors=True)
        assert {p.id: p.vector["colpali"] for p in after} == matrices
        assert any(p.vector["summary"] != summaries[p.id] for p in after)
        assert rag.backfill_ocr()["pages_skipped"] == 3
        rag.backfill_ocr(force=True)
        points, _ = rag.client.scroll("docs_text", with_vectors=False)
        assert len(points) == 3
        assert all(p.payload["text"].count("OCR text:") <= 1 for p in points)
    finally:
        rag.close()


def test_recognition_failure_records_failed_ingestion(tmp_path):
    def fail(image):
        raise RuntimeError("OCR failed")
    rag = LocalRAG(tmp_path / "index", ocr=SelectiveOCR(fail))
    try:
        with pytest.raises(RuntimeError, match="OCR failed"):
            rag.ingest("financial_report_qa", SAMPLE)
        assert rag.client.count("docs_text", exact=True).count == 0
        assert rag.db.execute("SELECT status FROM documents").fetchone()[0] == "failed"
    finally:
        rag.close()


@pytest.mark.parametrize("score,accepted", [(.6, True), (1., True), (.599, False),
    (-.1, False), (1.01, False), (float("nan"), False), (float("inf"), False)])
def test_confidence_boundary_and_invalid_scores(monkeypatch, score, accepted):
    monkeypatch.setenv("RAG_OCR_MIN_CONFIDENCE", "0.6")
    recognizer = lambda image: SimpleNamespace(txts=["CONFIDENCE7429"], scores=[score],
        boxes=[[[0, 0], [100, 0], [100, 30], [0, 30]]])
    with pymupdf.open(SAMPLE) as pdf:
        result = SelectiveOCR(recognizer).extract(pdf[1])
    assert bool(result.text) is accepted
    assert result.rejected_lines == int(not accepted)


@pytest.mark.parametrize("variable,value", [
    ("RAG_OCR_MIN_CONFIDENCE", "-0.1"), ("RAG_OCR_MIN_CONFIDENCE", "1.1"),
    ("RAG_OCR_MIN_CONFIDENCE", "nan"), ("RAG_OCR_SCALE", "0.49"),
    ("RAG_OCR_SCALE", "4.01"), ("RAG_OCR_SCALE", "inf")])
def test_invalid_configuration_fails_before_processing(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValueError, match="OCR confidence"):
        SelectiveOCR()


def test_no_detections_keeps_region_provenance():
    recognizer = lambda image: SimpleNamespace(txts=None, scores=None, boxes=None)
    with pymupdf.open(SAMPLE) as pdf:
        result = SelectiveOCR(recognizer).extract(pdf[1])
    assert result.text == ""
    assert result.blocks == []
    assert result.regions
    assert result.payload()["ocr_mode"] == "full_page"


def test_backfill_removes_obsolete_chunks_and_stale_ocr_text(tmp_path):
    recognizer = Recognizer()
    long_text = " ".join(f"obsoleteword{i}" for i in range(1000))
    recognizer_output = lambda image: SimpleNamespace(txts=[long_text], scores=[.99],
        boxes=[[[0, 0], [100, 0], [100, 30], [0, 30]]])
    rag = LocalRAG(tmp_path / "index", ocr=SelectiveOCR(recognizer_output))
    try:
        rag.ingest("financial_report_qa", SAMPLE)
        assert rag.client.count("docs_text", exact=True).count > 3
        rag.ocr = SelectiveOCR(recognizer)
        rag.backfill_ocr(force=True)
        points, _ = rag.client.scroll("docs_text", limit=100, with_vectors=True)
        assert len(points) == 3
        assert all("obsoleteword" not in p.payload["text"] for p in points)
        assert all(p.payload["ocr_backfill_complete"] for p in points)
        assert all(set(p.vector) == {"dense", "bm25"} for p in points)
        native = next(p for p in points if p.payload["page"] == 1)
        assert "IRIS" in native.payload["text"]
        assert rag.backfill_ocr()["pages_skipped"] == 3
    finally:
        rag.close()


def test_real_ocr_codes_are_retrievable_from_text_vectors(tmp_path):
    # Bundled ONNX models run locally; no mocked recognizer or external service.
    rag = LocalRAG(tmp_path / "index", ocr=SelectiveOCR())
    try:
        assert rag.ingest("financial_report_qa", SAMPLE)["ocr_pages"] == 2
        for code, page, mode in [("SCAN7429", 2, "full_page"),
                                 ("IMAGE8635", 3, "image_regions")]:
            hits = rag.retrieve("financial_report_qa", code, 3)
            hit = next(h for h in hits if code in h["text"])
            assert (hit["page"], hit["ocr_mode"]) == (page, mode)
            assert "ocr" in hit["extraction_methods"]
        assert not rag.retrieve("hr_policy", "SCAN7429", 3)
    finally:
        rag.close()


def test_interrupted_backfill_retries_incomplete_page(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_OCR_ENABLED", "0")
    rag = LocalRAG(tmp_path / "index")
    try:
        rag.ingest("financial_report_qa", SAMPLE)
        monkeypatch.setenv("RAG_OCR_ENABLED", "1")
        rag.ocr = SelectiveOCR(Recognizer())
        original = rag.client.set_payload

        def interrupt(collection_name, payload, **kwargs):
            if collection_name == "docs_text" and payload.get("ocr_backfill_complete"):
                raise RuntimeError("Interrupted before completion marker")
            return original(collection_name, payload, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(rag.client, "set_payload", interrupt)
            with pytest.raises(RuntimeError, match="Interrupted"):
                rag.backfill_ocr(force=True)
        points, _ = rag.client.scroll("docs_text", with_vectors=False)
        assert any(p.payload.get("ocr_backfill_complete") is False for p in points)
        rag.backfill_ocr()
        points, _ = rag.client.scroll("docs_text", with_vectors=False)
        assert len(points) == 3
        assert all(p.payload["ocr_backfill_complete"] for p in points)
        assert all(p.payload["text"].count("OCR text:") <= 1 for p in points)
        assert rag.backfill_ocr()["pages_skipped"] == 3
    finally:
        rag.close()
