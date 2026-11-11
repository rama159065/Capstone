"""--vlm routed: one reader per page, and the paidVL fallback (OCR -> Qwen3-VL -> paid model)."""
import json

import pytest

from scripts.rag import page_audit
from scripts.rag import vlm_ingest as vi
from tests.test_vlm_ingest import args_for, corpus  # noqa: F401  (Chroma corpus fixture)

OCR_TEXT = ("Revenue by segment in millions for fiscal 2025 as reported by the company. Upstream segment "
            "revenue 391,035 and downstream segment revenue 120,500 with other items 8,250 for the year.")


class Reader:
    def __init__(self, name, text):
        self.signature, self.text, self.calls = f"fake-{name}-v1", text, 0

    def transcribe(self, image, page):
        self.calls += 1
        return self.text


def table(downstream):
    return f"Revenue by segment (in millions)\n| Segment | 2025 |\n|---|---|\n| Upstream | 391,035 |\n| Downstream | {downstream} |"


def routed(qwen_text, paid_text=None, threshold=.10):
    ocr, qwen = Reader("ocr", OCR_TEXT), Reader("qwen", qwen_text)
    paid = Reader("paid", paid_text) if paid_text else None
    return vi.RoutedReader(lambda: qwen, qwen.signature, ocr=ocr, paid=paid, threshold=threshold), ocr, qwen, paid


@pytest.fixture
def paid_chart(monkeypatch):
    """Page 2 of the test PDF (a chart) routed as paidVL: the text layer can't be used to check figures."""
    real = page_audit.route
    monkeypatch.setattr(page_audit, "route", lambda page: "paidVL" if page["page"] == 2 else real(page))


def run(tmp_path, corpus, reader):
    ingest = vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "routed"), vlm=reader)
    assert ingest.run() == {"complete": 1}
    chunks = ingest.store.collection.get(include=["metadatas"])["metadatas"]
    return ingest, {m["page"]: m for m in chunks}


def test_qwen_reading_confirmed_by_ocr_is_kept_without_paying(tmp_path, corpus, paid_chart):
    reader, ocr, qwen, paid = routed(table("120,500"), paid_text=table("120,500"))
    _, by_page = run(tmp_path, corpus, reader)
    chart = by_page[2]
    assert chart["reader"] == "qwen3VL" and chart["escalation"] == "not_needed" and chart["number_check"] == "ocr"
    assert json.loads(chart["unverified_numbers"]) == [] and paid.calls == 0


def test_misread_figure_escalates_to_the_paid_model(tmp_path, corpus, paid_chart):
    reader, ocr, qwen, paid = routed(table("120,580"), paid_text=table("120,500"))   # Qwen misreads one figure
    ingest, by_page = run(tmp_path, corpus, reader)
    chart = by_page[2]
    assert chart["reader"] == "paidVL" and chart["escalation"] == "paid" and paid.calls == 1
    assert json.loads(chart["unverified_numbers"]) == []
    from scripts.rag.vlm_ingest import retrieve
    hit = retrieve("Downstream segment revenue", 5, root=tmp_path / "chroma", embedder=ingest.embedder)
    assert any("120,500" in h["text"] and h["metadata"]["reader"] == "paidVL" for h in hit)


def test_without_a_paid_model_the_qwen_reading_is_kept_and_flagged(tmp_path, corpus, paid_chart):
    reader, ocr, qwen, paid = routed(table("120,580"))
    _, by_page = run(tmp_path, corpus, reader)
    chart = by_page[2]
    assert chart["reader"] == "qwen3VL" and chart["escalation"] == "no_paid_model"
    assert json.loads(chart["unverified_numbers"]) == ["120,580"]


def test_routing_per_page_and_cached_readings(tmp_path, corpus, paid_chart):
    reader, ocr, qwen, paid = routed(table("120,580"), paid_text=table("120,500"))
    _, by_page = run(tmp_path, corpus, reader)
    assert by_page[1]["reader"] == "pdfParser" and by_page[1]["text_source"] == "native_pdf"     # plain text page
    assert by_page[3]["reader"] == "ocr" and by_page[3]["text_source"] == "ocr"                 # garbled text page
    assert 4 not in by_page                                                                      # blank page skipped
    calls = (ocr.calls, qwen.calls, paid.calls)
    again = vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "routed", "--force"), vlm=reader)
    assert again.run() == {"complete": 1}
    assert (ocr.calls, qwen.calls, paid.calls) == calls                                          # every reading cached

    stricter, *_ = routed(table("120,580"), paid_text=table("120,500"), threshold=.9)            # new threshold:
    stricter.ocr, stricter._qwen = ocr, qwen                                                     # same readers,
    vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "routed", "--force"), vlm=stricter).run()
    assert (ocr.calls, qwen.calls) == calls[:2]                                                  # reused from cache


def test_cross_check_and_signature():
    assert vi.cross_check(table("120,500"), OCR_TEXT) == {"unverified": [], "figures": 3, "share": 0.0}
    assert vi.cross_check(table("120,580"), OCR_TEXT)["unverified"] == ["120,580"]
    assert vi.cross_check(table("120,500"), "too short")["share"] is None                         # nothing to check
    args = vi.build_parser().parse_args(["ingest", "--vlm", "routed", "--qwen-reader", "mock", "--paid-threshold", "0.2"])
    assert vi.vlm_signature(args).startswith("routed-v1:qwen=mock-v1:ocr=ocr:") and "threshold=0.2" in vi.vlm_signature(args)
