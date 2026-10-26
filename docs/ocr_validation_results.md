# Selective OCR: actual inference and Qdrant validation

Actual RapidOCR CPU inference; learned MiniLM dense and BM25-style sparse vectors stored in the existing docs_text collection. No mocked OCR outputs.

| Query | Page | OCR route | Result |
|---|---|---|---|
| IRIS native review | 1 | none | PASS |
| SCAN7429 invoice reference | 2 | full_page | PASS |
| IMAGE8635 calibration access | 3 | image_regions | PASS |

Existing ColPali matrices unchanged: True. OCR vectors persisted after database reopen: True. Idempotent backfill: True.

Database counts: 6 text points, 2 existing ColPali page points.

The scanned-page code and image-region code are absent from their native PDF text layers. The native control page takes no OCR route. New OCR test-fixture pages were indexed into docs_text; existing ColPali records were preserved, not recomputed.
