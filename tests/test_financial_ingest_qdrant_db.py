import hashlib
import json

import numpy as np
import pymupdf
import pytest

from scripts.rag import financial_ingest_qdrant_db as fi


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


def test_agent_adapter_returns_citable_evidence(tmp_path, corpus, monkeypatch):
    from scripts.agents.financial_prompting_agent import extract_citations
    client = fi.connect(None, tmp_path / "root")
    run = fi.Ingest(args_for(tmp_path, corpus, stages="text"), client=client)
    run.run()
    monkeypatch.setattr(fi, "_serving", {"client": client, "embedder": run.embedder})
    hits = fi.retrieve_for_agent("total net sales 391,035", top_k=2, ticker="TEST", year=2025)
    assert hits[0]["page"] == 1 and hits[0]["ticker"] == "TEST" and hits[0]["year"] == 2025
    assert hits[0]["chunk_id"].startswith("annual_report_edgar_ars_TEST_2025_p0001_")
    assert extract_citations(f"Sales were $391,035 million [{hits[0]['chunk_id']}].") == [hits[0]["chunk_id"]]
    assert {"text", "preview", "source_url", "rerank_score", "metadata"} <= set(hits[0])
    assert fi.retrieve_for_agent("total net sales", ticker="OTHER") == []


def rerank_hits():
    return [{"chunk_id": name, "text": name, "rerank_score": 0.03 - i / 100, "retrieval_paths": paths}
            for i, (name, paths) in enumerate([("narrative", ["text"]), ("page", ["visual"]), ("table", ["text"]), ("other", ["text"])])]


def test_cross_encoder_orders_text_and_keeps_the_best_visual_page(monkeypatch):
    class Model:
        def predict(self, pairs):
            return [{"narrative": -2.0, "table": 5.0, "other": 1.0}[text] for _, text in pairs]
    monkeypatch.setenv("RAG_RERANKER_MODEL", "stub")
    monkeypatch.setattr("scripts.common.local_models.reranker", lambda name: Model())
    hits = fi.cross_encoder_rerank("q", rerank_hits(), 3)
    assert [h["chunk_id"] for h in hits] == ["table", "other", "page"]
    assert hits[0]["rerank_score"] == 5.0 and hits[0]["reranked"] and "reranked" not in hits[2]


@pytest.mark.parametrize("failure", ["raises", "nan", "short"])
def test_cross_encoder_failure_keeps_the_existing_order_and_warns(monkeypatch, failure):
    class Model:
        def predict(self, pairs):
            if failure == "raises":
                raise RuntimeError("model missing")
            return [float("nan")] * len(pairs) if failure == "nan" else [1.0]
    monkeypatch.setenv("RAG_RERANKER_MODEL", "stub")
    monkeypatch.setattr("scripts.common.local_models.reranker", lambda name: Model())
    hits = fi.cross_encoder_rerank("q", rerank_hits(), 2)
    assert [h["chunk_id"] for h in hits] == ["narrative", "page"]
    assert hits[0]["retrieval_warnings"] == ["reranker_unavailable"] and "reranked" not in hits[0]


def test_retrieve_for_agent_uses_the_reranker_only_when_configured(tmp_path, corpus, monkeypatch):
    client = fi.connect(None, tmp_path / "root")
    run = fi.Ingest(args_for(tmp_path, corpus, stages="text"), client=client)
    run.run()
    monkeypatch.setattr(fi, "_serving", {"client": client, "embedder": run.embedder})
    monkeypatch.delenv("RAG_RERANKER_MODEL", raising=False)
    assert "reranked" not in fi.retrieve_for_agent("total net sales", ticker="TEST")[0]
    class Model:
        def predict(self, pairs):
            return [1.0] * len(pairs)
    monkeypatch.setenv("RAG_RERANKER_MODEL", "stub")
    monkeypatch.setattr("scripts.common.local_models.reranker", lambda name: Model())
    assert fi.retrieve_for_agent("total net sales", ticker="TEST")[0]["reranked"]


def test_failed_pdf_is_recorded_and_skipped(tmp_path, corpus):
    row = json.loads(corpus.read_text(encoding="utf-8"))
    corpus.write_text(json.dumps({**row, "content_sha256": "0" * 64}) + "\n", encoding="utf-8")
    run = fi.Ingest(args_for(tmp_path, corpus, stages="text"), client=fi.connect(None, tmp_path / "root"))
    assert run.run() == {}
    assert not run.state.done("edgar_ars_TEST_2025", "text", run.pipeline["text"])
    status = run.state.db.execute("SELECT status, error FROM filings").fetchone()
    assert status[0] == "failed" and "SHA-256" in status[1]


@pytest.fixture
def sector_corpus(tmp_path):
    """<corpus>/banking/{pdf_manifest.jsonl, EDGAR_ARS_TEST_2025.pdf}, as the sector downloader writes it."""
    folder = tmp_path / "corpus" / "banking"
    folder.mkdir(parents=True)
    pdf = folder / "EDGAR_ARS_TEST_2025.pdf"
    make_pdf(pdf)
    row = {"document_id": "edgar_ars_TEST_2025", "company": "Test Bank", "ticker": "TEST", "cik": "0000000001",
           "year": 2025, "title": "Test Bank annual report 2025", "source_url": "https://example.invalid/x.pdf",
           "sector": "Banking", "sector_slug": "banking", "sic": "6021", "local_path": pdf.name,
           "content_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
    (folder / "pdf_manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return tmp_path / "corpus"


def test_sector_corpus_folder_is_ingested_with_a_sector_field(tmp_path, sector_corpus, monkeypatch):
    client = fi.connect(None, tmp_path / "root")
    run = fi.Ingest(args_for(tmp_path, sector_corpus, stages="text", pdf_dir=None), client=client)
    assert run.run() == {"complete": 1}
    points = client.scroll(fi.TEXT, limit=100, with_payload=True)[0]
    assert points and all(p.payload["sector"] == "Banking" and p.payload["sector_slug"] == "banking" for p in points)

    assert fi.search_text(client, run.embedder, "391035", 3, fi.query_filter(sector="Banking"))
    assert fi.search_text(client, run.embedder, "391035", 3, fi.query_filter(sector="banking,insurance"))
    assert not fi.search_text(client, run.embedder, "391035", 3, fi.query_filter(sector="insurance"))
    monkeypatch.setattr(fi, "_serving", {"client": client, "embedder": run.embedder})
    hits = fi.retrieve_for_agent("total net sales 391,035", top_k=1, sector="banking")
    assert hits[0]["sector"] == "Banking"
    assert fi.retrieve_for_agent("total net sales", sector="oil_and_gas") == []

    other = fi.Ingest(args_for(tmp_path, sector_corpus, stages="text", pdf_dir=None, sectors="insurance",
                               root=str(tmp_path / "root2")), client=fi.connect(None, tmp_path / "root2"))
    assert other.run() == {}                                               # sector filter: nothing selected


def test_tag_sectors_adds_the_field_to_older_points(tmp_path, corpus, sector_corpus):
    client = fi.connect(None, tmp_path / "root")
    fi.Ingest(args_for(tmp_path, corpus, stages="text"), client=client).run()       # legacy manifest: no sector
    assert {p.payload["sector"] for p in client.scroll(fi.TEXT, limit=100, with_payload=True)[0]} == {"unknown"}
    client.close()
    args = fi.build_parser().parse_args(["tag-sectors", "--root", str(tmp_path / "root"), "--manifest", str(sector_corpus)])
    assert fi.tag_sectors(args) == {"banking": 1}
    client = fi.connect(None, tmp_path / "root")
    assert {p.payload["sector_slug"] for p in client.scroll(fi.TEXT, limit=100, with_payload=True)[0]} == {"banking"}


def test_colpali_can_be_limited_to_table_and_chart_pages(tmp_path, corpus):
    client = fi.connect(None, tmp_path / "root")
    fi.Ingest(args_for(tmp_path, corpus, stages="visual", visual_pages="tables_charts"),
              colpali=FakeColPali(), client=client).run()
    assert [p.payload["page"] for p in client.scroll(fi.VISUAL, limit=100, with_payload=True)[0]] == [2]


@pytest.mark.parametrize("every, expected", [(1, 3), (2, 2), (0, 1)])
def test_snapshot_every_n_batches_and_always_after_the_last(tmp_path, corpus, monkeypatch, every, expected):
    row = json.loads(corpus.read_text(encoding="utf-8"))
    corpus.write_text("".join(json.dumps({**row, "document_id": f"edgar_ars_TEST_{n}"}) + "\n" for n in range(3)),
                      encoding="utf-8")
    taken = []
    monkeypatch.setattr(fi, "is_server", lambda client: True)          # snapshots need a server; record them instead
    monkeypatch.setattr(fi, "set_bulk", lambda *a: None)
    monkeypatch.setattr(fi, "wait_green", lambda *a: None)
    monkeypatch.setattr(fi, "snapshot", lambda client, url, name, dest, keep: taken.append(name) or dest / name)
    args = args_for(tmp_path, corpus, stages="text", batch_size=1, snapshot_every=every,
                    snapshot_dir=tmp_path / "drive")
    assert fi.Ingest(args, client=fi.connect(None, tmp_path / "root")).run() == {"complete": 3}
    assert taken == [fi.TEXT] * expected
    assert (tmp_path / "drive" / "ingest.sqlite3").exists()


def test_visual_page_selection_is_shared_by_colpali_and_the_vlm():
    from scripts.rag import financial_ingest_chroma_db as vi
    kinds = ["text", "table", "chart", "image", "outlined_text", "blank", "text"]
    pages = [{"page": n, "page_type": k, "garbled": n == 7,
              "visual": k in {"table", "chart", "image", "outlined_text"} or n == 7} for n, k in enumerate(kinds, 1)]
    picked = lambda mode: [p["page"] for p in fi.visual_targets(pages, mode)]
    assert picked("tables_charts_images") == [2, 3, 4]          # no unreadable or plain text pages
    assert picked("tables_charts") == [2, 3]
    assert picked("rule") == picked("visual") == [2, 3, 4, 5, 7]
    assert picked("all") == [1, 2, 3, 4, 5, 7]
    assert [p["page"] for p in vi.vlm_targets(pages, "tables_charts_images", limit=2)] == [2, 3]
    with pytest.raises(ValueError, match="unknown page selection"):
        fi.visual_targets(pages, "charts")


def test_collections_have_their_own_names_and_an_old_index_is_reported(tmp_path, corpus):
    from qdrant_client import models
    assert (fi.TEXT, fi.VISUAL) == ("annual_reports_text", "annual_reports_visual")
    assert not {fi.TEXT, fi.VISUAL} & {"docs_text", "visual_pages"}          # LocalRAG's names

    client = fi.connect(None, tmp_path / "localrag")                            # LocalRAG's docs_text: not ours
    client.create_collection("docs_text", vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)})
    client.upsert("docs_text", points=[models.PointStruct(id=1, vector={"dense": [1, 0, 0, 0]}, payload={"domain": "hr_policy"})])
    assert fi.legacy_index(client) == []
    fi.ensure_collections(client, 384, visual=False)                            # shares the store without touching it
    assert client.count("docs_text").count == 1 and client.collection_exists(fi.TEXT)

    old = fi.connect(None, tmp_path / "old")                                     # an index from before the rename
    old.create_collection("docs_text", vectors_config={"dense": models.VectorParams(size=4, distance=models.Distance.COSINE)})
    old.upsert("docs_text", points=[models.PointStruct(id=1, vector={"dense": [1, 0, 0, 0]}, payload={"doc_kind": "ARS"})])
    assert fi.legacy_index(old) == ["docs_text"]
    with pytest.raises(ValueError, match="before the rename"):
        fi.Ingest(args_for(tmp_path, corpus, root=tmp_path / "old", stages="text"), client=old)


def test_font_encoding_garbage_counts_as_garbled_but_normal_text_does_not():
    costco = "×¬»³ è‰Ú·²¿²½·¿´ Í¬¿¬»³»²¬­ ¿²¼ Í«°°´»³»²¬¿®§ Ü¿¬¿ ÝÑÍÌÝÑ ÉØÑÔÛÍßÔÛ ÝÑÎÐÑÎßÌ×ÑÒ ×ÒÜÛÈ ÌÑ Ý " * 2
    clean = ("Net sales increased 5% to $249.6 billion — driven by membership fees, Café and Société Générale "
             "deposits; the Company’s results are described in Item 7. ") * 4
    assert fi.non_ascii_garbage(costco)
    assert not fi.non_ascii_garbage(clean)                   # a few dashes, quotes and accents are normal
    assert not fi.non_ascii_garbage("±²³ ¤ µ")              # too short to judge
    assert fi.ANALYSIS_VERSION == 2                          # cached page analyses are redone with the new rule


def test_needs_ocr_covers_pages_without_a_text_layer():
    page = lambda kind, text="", garbled=False: {"page_type": kind, "text": text, "garbled": garbled}
    many = "net sales increased in fiscal 2025 driven by membership fee income and comparable sales growth " * 2
    assert fi.needs_ocr(page("text", many, garbled=True))               # garbled text layer
    assert fi.needs_ocr(page("outlined_text"))                          # glyphs drawn as shapes
    assert fi.needs_ocr(page("chart", "Revenue 2025"))                  # vector page with almost no text (Costco FY2025)
    assert fi.needs_ocr(page("image", ""))                              # text baked into an image
    assert not fi.needs_ocr(page("chart", many))                        # a chart whose labels are real text
    assert not fi.needs_ocr(page("text", many)) and not fi.needs_ocr(page("table", "a b"))
    assert not fi.needs_ocr(page("blank"))
    assert fi.readable_text({**page("chart", "x"), "ocr_text": "COSTCO 914 warehouses"}) == ("COSTCO 914 warehouses", "ocr")
    assert fi.readable_text(page("chart", "Revenue 2025")) == ("Revenue 2025", "native_pdf")   # OCR off: keep the few words
    assert fi.readable_text(page("text", many, garbled=True)) == ("", "ocr")                   # garbage is never indexed
