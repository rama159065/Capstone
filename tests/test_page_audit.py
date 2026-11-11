import csv
import hashlib
import json

import pytest

from scripts.rag import page_audit as pa
from tests.test_financial_ingest_qdrant_db import make_pdf


def page(kind, text="", garbled=False, image_share=0.0):
    return {"page": 1, "page_type": kind, "garbled": garbled, "text": text, "image_share": image_share}


PROSE = "the company reported results for the year and the board approved the plan for growth " * 3


@pytest.mark.parametrize("analysis,label", [
    (page("text", PROSE), "pdfParser"),
    (page("blank"), "none"),
    (page("table", "Revenue 391,035 383,285"), "pdfParser+colpali"),
    (page("chart", "Net sales 2023 2024 2025 391,035 383,285 365,817 $6.08 2.1% 10%"), "pdfParser+colpali"),
    (page("chart", "Net sales by year"), "qwen3VL"),
    ({**page("chart"), "shapes": 3371}, "ocr"),                              # text drawn as outlines (Costco 2025)
    (page("blank", "Ñ«® ¿ª»®¿¹» ©¿®»¸±«­» ­°¿½» ·­ ¿°°®±¨·³¿¬»´§ ïìéôððð", garbled=True), "ocr"),   # Costco 2024
    (page("image", PROSE), "pdfParser+colpali"),
    (page("image", "", image_share=.9), "ocr"),
    (page("image", "Our stores", image_share=.3), "none"),                 # a photo: no text, no figures
    (page("image", "Sales 2025 41% 3,420", image_share=.3), "qwen3VL"),    # an infographic
    (page("text", "èçé´±½¿¬", garbled=True), "ocr"),
    (page("outlined_text"), "ocr"),
    (page("table", "èçé", garbled=True), "paidVL"),
    (page("chart", "", garbled=True), "paidVL"),
])
def test_route_rules(analysis, label):
    assert pa.route(analysis) == label and label in pa.LABELS


def test_audit_writes_one_csv_per_pdf_in_sector_folders_and_resumes(tmp_path):
    folder = tmp_path / "annual_reports" / "banking"
    folder.mkdir(parents=True)
    pdf = folder / "EDGAR_ARS_TEST_2025.pdf"
    make_pdf(pdf)
    row = {"document_id": "edgar_ars_TEST_2025", "ticker": "TEST", "year": 2025, "sector": "Banking",
           "sector_slug": "banking", "local_path": pdf.name, "content_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    missing = {**row, "document_id": "edgar_ars_GONE_2025", "local_path": "EDGAR_ARS_GONE_2025.pdf", "content_sha256": None}
    (folder / "pdf_manifest.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(missing) + "\n", encoding="utf-8")

    out = tmp_path / "page_audit"
    summary = pa.run(tmp_path / "annual_reports", out, workers=1, quiet=True)
    target = out / "banking" / "EDGAR_ARS_TEST_2025.csv"
    rows = list(csv.DictReader(target.open(encoding="utf-8")))
    assert list(rows[0]) == ["pageNumber", "parserUsage"]
    assert [r["pageNumber"] for r in rows] == ["1", "2", "3", "4"]
    assert [r["parserUsage"] for r in rows] == ["pdfParser", "qwen3VL", "ocr", "none"]   # text, chart, garbled, blank
    assert summary["status"] == {"complete": 1, "failed": 1} and summary["pages"] == 4
    assert summary["failed"][0]["document_id"] == "edgar_ars_GONE_2025"
    assert (out / "_summary.csv").exists() and json.loads((out / "_summary.json").read_text())["pages"] == 4

    again = pa.run(tmp_path / "annual_reports", out, workers=1, quiet=True)
    assert again["status"] == {"skipped": 1, "failed": 1} and again["parser_usage"] == summary["parser_usage"]
