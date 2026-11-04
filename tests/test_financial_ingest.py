import hashlib
import json

import numpy as np
import pymupdf
import pytest

from scripts import financial_ingest as fi


class FakeColPali:
    """1024 image patches + 6 special tokens per page, seeded by the rendered pixels."""
    signature = fi.colpali_signature("fake-colpali")
    dtype_name = "float32"

    def embed_images(self, images):
        out = []
        for image in images:
            rng = np.random.default_rng(int(hashlib.sha256(image.tobytes()).hexdigest()[:8], 16))
            out.append((fi.l2(rng.normal(size=(1030, 128))), np.arange(1030) < 1024))
        return out

    def embed_query(self, text):
        return fi.l2(np.random.default_rng(len(text)).normal(size=(12, 128)))


def test_terms_keep_numbers_whole():
    found = set(fi.terms("Net sales were $391,035 million, up 2.1% in FY2024 ($6.08 per share)."))
    assert {"391,035", "391035", "$391,035", "2.1%", "2.1", "fy2024", "$6.08", "6.08", "net"} <= found
    assert "035" not in found


def test_windows_cover_text_with_overlap():
    text = " ".join(f"w{i}" for i in range(600))
    parts = list(fi.windows(text, size=300, overlap=60))
    assert all(len(p) <= 300 for p in parts)
    assert parts[0].split()[0] == "w0" and parts[-1].split()[-1] == "w599"
    assert set(parts[0].split()) & set(parts[1].split())          # overlap, no word cut in half
    assert all(w.startswith("w") for p in parts for w in p.split())


def test_split_table_repeats_header():
    lines = ["Item | 2025", "--- | ---"] + [f"row{i} | {i},000" for i in range(50)]
    parts = list(fi.split_table(lines, budget=200))
    assert len(parts) > 1 and all(p[:2] == lines[:2] for p in parts)
    assert sum(len(p) - 2 for p in parts) == 50


def test_pool_keeps_specials_and_shapes():
    vectors = fi.l2(np.random.default_rng(0).normal(size=(1030, 128)))
    rows, cols = fi.pool_rows_cols(vectors, np.arange(1030) < 1024)
    assert rows.shape == cols.shape == (38, 128)
    assert np.allclose(rows[-6:], vectors[-6:], atol=1e-6)
    grid = vectors[:1024].reshape(32, 32, 128)
    assert np.allclose(rows[0], fi.l2(grid[0].mean(axis=0)), atol=1e-6)     # row 0 = mean across its columns
    assert np.allclose(cols[0], fi.l2(grid[:, 0].mean(axis=0)), atol=1e-6)
    with pytest.raises(ValueError):
        fi.pool_rows_cols(vectors[:500], np.ones(500, bool))


def make_pdf(path):
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((50, 60), "Item 7. Management's Discussion and Analysis", fontsize=11)
    y = 90
    for line in ["(in millions, except per share amounts)",
                 "Total net sales for the year were 391,035 and the margin of the company is",
                 "reported in the table of the annual report that is shown on this page for the year."] * 3:
        page.insert_text((50, y), line, fontsize=9)
        y += 14
    chart = pdf.new_page()
    chart.insert_text((50, 60), "Revenue by segment chart", fontsize=11)
    for i in range(70):
        chart.draw_rect(pymupdf.Rect(60 + i * 6, 400 - i * 3, 64 + i * 6, 400), color=(0, 0, 1), fill=(0, 0, 1))
    garbled = pdf.new_page()
    y = 60
    for _ in range(6):
        garbled.insert_text((50, y), "xq zvk prt mbl qwx zzk vrn plk trs bnm gfh jkl", fontsize=9)
        y += 14
    pdf.new_page()                                                          # blank page
    pdf.save(path)


def args_for(tmp_path, manifest, **overrides):
    parser = fi.build_parser()
    argv = ["ingest", "--root", str(tmp_path / "root"), "--manifest", str(manifest),
            "--pdf-dir", str(tmp_path / "pdfs"), "--embedding-provider", "hashing",
            "--ocr", "none", "--workers", "1", "--colpali-batch", "2"]
    args = parser.parse_args(argv)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "pdfs").mkdir()
    pdf = tmp_path / "pdfs" / "EDGAR_ARS_TEST_2025.pdf"
    make_pdf(pdf)
    row = {"document_id": "edgar_ars_TEST_2025", "company": "Test Corp", "ticker": "TEST", "cik": "1", "year": 2025,
           "title": "Test Corp annual report 2025", "source_url": "https://example.invalid/x.pdf",
           "accession_number": "0000000001-25-000001", "local_path": "data\\raw\\x\\EDGAR_ARS_TEST_2025.pdf",
           "content_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def test_analysis_classifies_pages(tmp_path, corpus):
    pages = fi.analyse_pdf(tmp_path / "pdfs" / "EDGAR_ARS_TEST_2025.pdf")
    assert [p["page_type"] for p in pages] == ["text", "chart", "text", "blank"]
    assert [p["visual"] for p in pages] == [False, True, True, False]
    assert pages[2]["garbled"] and not pages[0]["garbled"]
    assert pages[0]["section"] == "item_7" and pages[0]["scale"].startswith("in millions")
    _, source = fi.readable_text(pages[2])
    assert source == "ocr" and fi.readable_text(pages[2])[0] == ""


def test_ingest_end_to_end_and_resume(tmp_path, corpus):
    client = fi.connect(None, tmp_path / "root")
    run = fi.Ingest(args_for(tmp_path, corpus), colpali=FakeColPali(), client=client)
    assert run.run() == {"complete": 2}

    texts = client.scroll(fi.TEXT, limit=100, with_payload=True)[0]
    assert texts and {p.payload["page"] for p in texts} == {1, 2}           # garbled page 3 never indexed as text
    assert all(p.payload["filing_id"] == "TEST_FY2025" and p.payload["doc_kind"] == "ARS" for p in texts)
    visuals = client.scroll(fi.VISUAL, limit=100, with_payload=True, with_vectors=True)[0]
    assert sorted(p.payload["page"] for p in visuals) == [2, 3]
    vector = visuals[0].vector
    assert len(vector["colpali"]) == 1030 and len(vector["mean_rows"]) == len(vector["mean_cols"]) == 38

    hit = fi.search_text(client, run.embedder, "391035", top_k=1)[0]       # number without commas still matches
    assert hit.payload["page"] == 1 and "391,035" in hit.payload["text"]
    assert hit.payload["scale"].startswith("in millions")
    visual = fi.search_visual(client, FakeColPali(), run.embedder, "segment chart", top_k=2,
                              flt=fi.query_filter("test", 2025))
    assert {h.payload["page"] for h in visual} == {2, 3}

    again = fi.Ingest(args_for(tmp_path, corpus), colpali=FakeColPali(), client=client)
    assert again.run() == {} and client.count(fi.TEXT, exact=True).count == len(texts)

    with pytest.raises(ValueError, match="colpali differs"):
        class Other(FakeColPali):
            signature = fi.colpali_signature("other-model")
        fi.Ingest(args_for(tmp_path, corpus), colpali=Other(), client=client)


def test_failed_pdf_is_recorded_and_skipped(tmp_path, corpus):
    row = json.loads(corpus.read_text(encoding="utf-8"))
    corpus.write_text(json.dumps({**row, "content_sha256": "0" * 64}) + "\n", encoding="utf-8")
    run = fi.Ingest(args_for(tmp_path, corpus, stages="text"), client=fi.connect(None, tmp_path / "root"))
    assert run.run() == {}
    assert not run.state.done("edgar_ars_TEST_2025", "text", run.pipeline["text"])
    status = run.state.db.execute("SELECT status, error FROM filings").fetchone()
    assert status[0] == "failed" and "SHA-256" in status[1]
