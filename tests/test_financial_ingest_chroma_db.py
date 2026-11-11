import hashlib
import json

import pytest

from scripts.rag import financial_ingest_qdrant_db as fi
from scripts.rag import financial_ingest_chroma_db as vi
from tests.test_financial_ingest_qdrant_db import make_pdf


class FakeVLM:
    """A table with one misread figure (391,053 for 391,035) on the chart page."""
    signature = "fake-vlm-v1"

    def __init__(self):
        self.calls = 0

    def transcribe(self, image, page):
        self.calls += 1
        return ("```markdown\nRevenue by segment (in millions)\n| Segment | 2025 |\n|---|---|\n"
                "| Upstream | 391,035 |\n| Downstream | 391,053 |\n```")


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


def args_for(tmp_path, corpus, store, *extra):
    return vi.build_parser().parse_args(["ingest", "--root", str(tmp_path / store), "--store", store,
                                         "--manifest", str(corpus), "--embedding-provider", "hashing", *extra])


def test_markdown_tables_keep_their_header_and_title():
    rows = "\n".join(f"| row{i} | {i},000 |" for i in range(400))
    parts = vi.markdown_parts(f"Intro paragraph about the year and the segments.\nNet sales (in millions)\n| Item | 2025 |\n|---|---|\n{rows}")
    tables = [chunk for kind, chunk in parts if kind == "table"]
    assert len(tables) > 1
    assert all(t.startswith("Net sales (in millions)\n| Item | 2025 |\n|---|---|") for t in tables)
    assert sum(t.count("| row") for t in tables) == 400


def test_unverified_numbers_need_a_text_layer():
    layer = "Total net sales for the year were 391,035 and the margin of the company is reported in the table " * 2
    assert vi.unverified_numbers("| Upstream | 391,035 |\n| Downstream | 391,053 |", layer) == ["391,053"]
    assert vi.unverified_numbers("$324B total assets", "&#()*+),-)-.)/)01") is None


@pytest.mark.parametrize("store", ["chroma"])
def test_vlm_pipeline_writes_sector_tagged_chunks_and_resumes(tmp_path, corpus, store):
    vlm = FakeVLM()
    run = vi.VlmIngest(args_for(tmp_path, corpus, store), vlm=vlm)
    assert run.run() == {"complete": 1}
    assert vlm.calls == 2                                       # pages 2 (chart) and 3 (garbled text) are "visual"

    hits = vi.retrieve("Downstream segment revenue 391,053", 3, sector="banking", root=tmp_path / store, store=store,
                       embedder=run.embedder)
    table = next(h for h in hits if h["content_type"] == "table")
    assert table["text_source"] == "vlm" and table["sector"] == "Banking" and table["ticker"] == "TEST"
    assert table["chunk_id"].startswith("annual_report_edgar_ars_TEST_2025_p000")
    assert "```" not in table["text"] and table["text"].startswith("Revenue by segment (in millions)")
    assert vi.retrieve("revenue", 3, sector="insurance", root=tmp_path / store, store=store, embedder=run.embedder) == []
    native = vi.retrieve("Total net sales 391,035 margin", 5, root=tmp_path / store, store=store, embedder=run.embedder)
    assert any(h["text_source"] == "native_pdf" and h["page"] == 1 for h in native)   # plain text page: no VLM

    chart = json.loads((tmp_path / store / "vlm_pages" / "edgar_ars_TEST_2025.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert chart["page"] == 2 and chart["number_check"] in {"text_layer", "no_text_layer"}

    count = run.store.count()
    again = vi.VlmIngest(args_for(tmp_path, corpus, store), vlm=vlm)
    assert again.run() == {} and vlm.calls == 2                # finished filing skipped
    forced = vi.VlmIngest(args_for(tmp_path, corpus, store, "--force"), vlm=vlm)
    assert forced.run() == {"complete": 1} and vlm.calls == 2   # rewritten from the transcription cache
    assert forced.store.count() == count                        # replaced, not duplicated


def test_page_choice_and_signature_without_loading_a_model(tmp_path, corpus):
    pages = fi.analyse_pdf(corpus / "banking" / "EDGAR_ARS_TEST_2025.pdf")
    assert [p["page"] for p in vi.vlm_targets(pages, "tables_charts")] == [2]
    assert [p["page"] for p in vi.vlm_targets(pages, "visual")] == [2, 3]
    assert [p["page"] for p in vi.vlm_targets(pages, "all")] == [1, 2, 3]
    args = args_for(tmp_path, corpus, "chroma", "--vlm", "transformers", "--vlm-model", "Qwen/Qwen3-VL-8B-Instruct")
    assert vi.vlm_signature(args) == "transformers:Qwen/Qwen3-VL-8B-Instruct:1200"


class FakeOCR:
    signature = "fake-ocr"

    def read(self, page):
        return "Revenue by quarter chart, fiscal 2025. Data label for the fourth quarter: 412,700"


def test_ocr_reader_indexes_chart_labels_and_native_tables_as_ocr_text(tmp_path, corpus):
    reader = vi.OcrReader(ocr=FakeOCR())
    run = vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "ocr"), vlm=reader)
    assert run.vlm_signature == vi.ocr_reader_signature() == vi.vlm_signature(args_for(tmp_path, corpus, "chroma", "--vlm", "ocr"))
    assert run.run() == {"complete": 1}
    hits = vi.retrieve("412,700", 3, root=tmp_path / "chroma", embedder=run.embedder)
    assert hits[0]["text_source"] == "ocr" and "412,700" in hits[0]["text"]
    assert hits[0]["page"] in {2, 3}                             # the chart and the unreadable page


def test_bm25_sidecar_matches_exact_figures_and_is_replaced_not_duplicated(tmp_path, corpus):
    run = vi.VlmIngest(args_for(tmp_path, corpus, "chroma"), vlm=FakeVLM())
    run.run()
    store = run.store
    chunks = store.bm25.count()
    assert chunks == store.count() > 0                           # one BM25 row per Chroma chunk
    assert store.bm25.search("391,053", 5) and store.bm25.search("391053", 5) == store.bm25.search("$391,053", 5)
    assert store.bm25.search("391,053", 5, sectors=["insurance"]) == []
    top = vi.retrieve("391,053", 1, root=tmp_path / "chroma", embedder=run.embedder)[0]
    assert "391,053" in top["text"]                              # the exact figure wins the fusion
    vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--force"), vlm=FakeVLM()).run()
    assert vi.ChromaStore(tmp_path / "chroma").bm25.count() == chunks


def test_paid_api_reader_backs_off_on_rate_limits(monkeypatch):
    import email.message
    import io
    import urllib.error
    from PIL import Image
    monkeypatch.setenv("RAG_VISION_API_BASE", "https://api.example.invalid/v1")
    monkeypatch.setenv("RAG_VISION_API_KEY", "sk-test")
    monkeypatch.setenv("RAG_VISION_RETRY_DELAYS", "2,4")
    reader = vi.OpenAIVLM(model="gpt-4o-mini")
    waits, calls = [], []
    monkeypatch.setattr(vi.time, "sleep", waits.append)

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(request, timeout):
        calls.append(request.get_header("Authorization"))
        if len(calls) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "rate", email.message.Message(), None)
        return Response(json.dumps({"choices": [{"message": {"content": "| a | 1 |"}}]}).encode())

    monkeypatch.setattr(vi.urllib.request, "urlopen", fake_urlopen)
    assert reader.transcribe(Image.new("RGB", (8, 8)), None) == "| a | 1 |"
    assert waits == [2.0] and calls == ["Bearer sk-test"] * 2
    assert reader.signature == "openai:gpt-4o-mini:1200"


def test_chroma_is_the_only_store():
    with pytest.raises(SystemExit):
        vi.build_parser().parse_args(["ingest", "--store", "qdrant"])
    with pytest.raises(ValueError, match="Chroma only"):
        vi.retrieve("revenue", store="qdrant", embedder=object())
