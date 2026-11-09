import hashlib
import json

import pytest

from scripts.rag import financial_ingest as fi
from scripts.rag import vlm_ingest as vi
from tests.test_financial_ingest import make_pdf


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


@pytest.mark.parametrize("store", ["chroma", "qdrant"])
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
    args = args_for(tmp_path, corpus, "chroma", "--vlm", "transformers", "--vlm-model", "Qwen/Qwen2.5-VL-7B-Instruct")
    assert vi.vlm_signature(args) == "transformers:Qwen/Qwen2.5-VL-7B-Instruct:1200"
