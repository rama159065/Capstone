"""Actual OCR inference, learned-vector Qdrant writes, and text-only retrieval checks."""
import hashlib
import json
import time

import pymupdf

from scripts.rag.ocr_index import configure_ocr
from scripts.rag.multimodal_rag import LocalRAG, ROOT


def visual_hashes(rag):
    if not rag.client.collection_exists("visual_pages"):
        return {}
    points, _ = rag.client.scroll("visual_pages", limit=100, with_vectors=True)
    return {str(p.id): hashlib.sha256(json.dumps(p.vector["colpali"]).encode()).hexdigest() for p in points}


def main():
    configure_ocr()
    path = ROOT / "data/samples/ocr_validation/financial_report_qa/selective_ocr_sample.pdf"
    report = {"ocr_engine": "rapidocr", "mocked": False, "collection": "docs_text", "checks": []}
    with pymupdf.open(path) as pdf:
        assert not pdf[1].get_text().strip()
        assert "IMAGE8635" not in pdf[2].get_text()
    rag = LocalRAG()
    try:
        before = visual_hashes(rag)
        start = time.monotonic()
        report["ingestion"] = rag.ingest("financial_report_qa", path)
        report["backfill"] = rag.backfill_ocr("financial_report_qa")
        report["write_seconds"] = round(time.monotonic()-start, 2)
        report["colpali_unchanged"] = visual_hashes(rag) == before
        for query, page, code in [("IRIS native review", 1, "IRIS"),
                                  ("SCAN7429 invoice reference", 2, "SCAN7429"),
                                  ("IMAGE8635 calibration access", 3, "IMAGE8635")]:
            hit = rag.retrieve("financial_report_qa", query, 1)[0]
            expected_mode = {1: "none", 2: "full_page", 3: "image_regions"}[page]
            passed = hit["title"] == path.stem and hit["page"] == page and code in hit["text"] and hit["ocr_mode"] == expected_mode
            row = {"query": query, "expected_page": page, "retrieved_page": hit["page"],
                   "ocr_mode": hit["ocr_mode"], "ocr_text": hit["ocr_text"],
                   "ocr_regions": hit["ocr_regions"], "accepted_lines": len(hit["ocr_blocks"]), "passed": passed}
            report["checks"].append(row)
            print(json.dumps(row), flush=True)
        report["idempotent"] = rag.backfill_ocr("financial_report_qa")["pages_text_enriched"] == 0
        report["text_points"] = rag.client.count("docs_text", exact=True).count
        report["visual_points"] = len(before)
    finally:
        rag.close()
    reopened = LocalRAG()
    try:
        points, _ = reopened.client.scroll("docs_text", scroll_filter=reopened.filter(
            "financial_report_qa", doc_id=report["ingestion"]["doc_id"]), with_vectors=True, limit=100)
        report["persisted_ocr_vectors"] = any(p.payload.get("ocr_text") and "dense" in p.vector and "bm25" in p.vector for p in points)
    finally:
        reopened.close()
    folder = ROOT / "output/ocr_validation"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Selective OCR: actual inference and Qdrant validation", "", "Actual RapidOCR CPU inference; learned MiniLM dense and BM25-style sparse vectors stored in the existing docs_text collection. No mocked OCR outputs.", "",
             "| Query | Page | OCR route | Result |", "|---|---|---|---|"]
    lines += [f"| {r['query']} | {r['retrieved_page']} | {r['ocr_mode']} | {'PASS' if r['passed'] else 'FAIL'} |" for r in report["checks"]]
    lines += ["", f"Existing ColPali matrices unchanged: {report['colpali_unchanged']}. OCR vectors persisted after database reopen: {report['persisted_ocr_vectors']}. Idempotent backfill: {report['idempotent']}.", "",
              f"Database counts: {report['text_points']} text points, {report['visual_points']} existing ColPali page points.", "",
              "The scanned-page code and image-region code are absent from their native PDF text layers. The native control page takes no OCR route. New OCR test-fixture pages were indexed into docs_text; existing ColPali records were preserved, not recomputed."]
    (ROOT / "docs/ocr_validation_results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    assert all(r["passed"] for r in report["checks"])
    assert report["colpali_unchanged"] and report["persisted_ocr_vectors"] and report["idempotent"]


if __name__ == "__main__":
    main()
