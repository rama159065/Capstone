"""--vlm routed: one reader per page, and the paidVL fallback (OCR -> Qwen3-VL -> paid model)."""
import json

import pytest

from scripts.rag import page_audit
from scripts.rag import financial_ingest_chroma_db as vi
from tests.test_financial_ingest_chroma_db import args_for, corpus  # noqa: F401  (Chroma corpus fixture)

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
    from scripts.rag.financial_ingest_chroma_db import retrieve
    hit = retrieve("Downstream segment revenue", 5, root=tmp_path / "chroma", embedder=ingest.embedder)
    assert any("120,500" in h["text"] and h["metadata"]["reader"] == "paidVL" for h in hit)


@pytest.mark.parametrize("qwen_fails", ["empty", "error", "no_model"])
def test_paid_model_reads_the_page_when_qwen_cannot(tmp_path, corpus, paid_chart, qwen_fails):
    """Qwen3-VL returns nothing, raises (e.g. out of GPU memory) or doesn't load: Gemini's reading is used."""
    ocr, paid = Reader("ocr", OCR_TEXT), Reader("paid", table("120,500"))
    qwen = Reader("qwen", "" if qwen_fails == "empty" else table("120,500"))
    if qwen_fails == "error":
        qwen.transcribe = lambda image, page: (_ for _ in ()).throw(RuntimeError("CUDA out of memory"))

    def factory():
        if qwen_fails == "no_model":
            raise OSError("Qwen3-VL weights not found")
        return qwen

    reader = vi.RoutedReader(factory, qwen.signature, ocr=ocr, paid=paid, threshold=.10)
    _, by_page = run(tmp_path, corpus, reader)
    chart = by_page[2]
    assert chart["reader"] == "paidVL" and chart["escalation"] == "paid" and paid.calls == 1
    assert json.loads(chart["unverified_numbers"]) == [] and chart["reader_model"] == paid.signature


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
    assert vi.vlm_signature(args).startswith("routed-v5:qwen=mock-v1:ocr=ocr:") and "threshold=0.2" in vi.vlm_signature(args)
    assert ":skip=" not in vi.vlm_signature(args)
    skipping = vi.build_parser().parse_args(["ingest", "--vlm", "routed", "--qwen-reader", "mock",
                                             "--skip-routes", "qwen3VL,paidVL"])
    assert vi.vlm_signature(skipping).endswith(":skip=paidVL,qwen3VL")          # a later full run redoes the filings
    assert vi.make_vlm(skipping).signature == vi.vlm_signature(skipping)
    with pytest.raises(ValueError):
        vi.RoutedReader(lambda: None, "q", ocr=Reader("ocr", ""), skip_routes=("ocr",))


def test_skipped_routes_load_no_model_and_keep_a_usable_text_layer(tmp_path, corpus, monkeypatch):
    real = page_audit.route
    monkeypatch.setattr(page_audit, "route", lambda page: {2: "qwen3VL", 3: "paidVL"}.get(page["page"], real(page)))
    ocr, broken = Reader("ocr", OCR_TEXT), Broken()
    reader = vi.RoutedReader(lambda: broken, broken.signature, ocr=ocr, describe=False, skip_routes=("qwen3VL", "paidVL"))
    _, by_page = run(tmp_path, corpus, reader)
    assert broken.calls == 0 and reader._qwen is None and ocr.calls == 0     # no model, no OCR for skipped pages
    assert {m["reader"] for m in by_page.values()} == {"pdfParser"}          # text layers only (the test chart's is
    assert 3 not in by_page                                                   # too short); garbled: left for later


RICH_OCR = ("Segment results in millions | Upstream 391,035 | Downstream 120,500 | Chemicals 8,250 | "
            "Other 1,775 | Eliminations (2,310) | Total 519,250 | Prior year 498,120 | Change 4.2%")


def test_table_without_figures_in_its_text_layer_is_read_by_ocr(tmp_path, corpus, monkeypatch):
    real = page_audit.route
    monkeypatch.setattr(page_audit, "route", lambda page: "pdfParser+colpali" if page["page"] == 2 else real(page))
    monkeypatch.setattr(vi, "lacks_figures", lambda page: page["page"] == 2)
    ocr = Reader("ocr", RICH_OCR)
    reader = vi.RoutedReader(lambda: Broken(), "q", ocr=ocr, describe=False)
    _, by_page = run(tmp_path, corpus, reader)
    assert by_page[2]["reader"] == "ocr" and by_page[2]["escalation"] == "text_layer_lacks_figures"

    few = Reader("ocr-few", OCR_TEXT)                                         # OCR finds no more figures:
    reader = vi.RoutedReader(lambda: Broken(), "q", ocr=few, describe=False)
    again = vi.VlmIngest(args_for(tmp_path / "few", corpus, "chroma", "--vlm", "routed"), vlm=reader)
    assert again.run() == {"complete": 1}
    pages = {m["page"]: m for m in again.store.collection.get(include=["metadatas"])["metadatas"]}
    assert few.calls >= 1 and pages.get(2, {}).get("reader", "pdfParser") == "pdfParser"   # checked; layer stays


def test_lacks_figures_rule():
    assert vi.lacks_figures({"page_type": "table", "text": "Board of directors | Name | Since 2019"})
    assert not vi.lacks_figures({"page_type": "table", "text": "Revenue 1,200 1,340 2.5 3.1 4,400 5,600"})
    assert not vi.lacks_figures({"page_type": "text", "text": "no figures here"})       # plain text: never


class Broken:
    signature = "broken-qwen-v1"

    def __init__(self):
        self.calls = 0

    def transcribe(self, image, page):
        self.calls += 1
        raise RuntimeError("CUDA out of memory")


def test_qwen_failure_or_empty_reading_falls_back_to_ocr(tmp_path, corpus, monkeypatch):
    real = page_audit.route
    monkeypatch.setattr(page_audit, "route", lambda page: {2: "qwen3VL", 3: "paidVL"}.get(page["page"], real(page)))
    ocr, broken = Reader("ocr", OCR_TEXT), Broken()
    reader = vi.RoutedReader(lambda: broken, broken.signature, ocr=ocr)
    _, by_page = run(tmp_path, corpus, reader)
    assert by_page[2]["reader"] == "ocr" and by_page[2]["escalation"] == "qwen_failed_ocr_fallback"
    assert by_page[3]["reader"] == "ocr" and by_page[3]["escalation"] == "models_failed_ocr_fallback"
    assert broken.calls == 2                                     # the filing completed; failures are not cached

    empty = Reader("qwen-empty", "   ")
    reader = vi.RoutedReader(lambda: empty, empty.signature, ocr=ocr)
    again = vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "routed", "--force"), vlm=reader)
    assert again.run() == {"complete": 1}
    pages = {m["page"]: m for m in again.store.collection.get(include=["metadatas"])["metadatas"]}
    assert pages[2]["reader"] == "ocr" and pages[2]["escalation"] == "qwen_failed_ocr_fallback"


def paid_first(qwen_text, paid_text):
    ocr, qwen, paid = Reader("ocr", OCR_TEXT), Reader("qwen", qwen_text), Reader("paid", paid_text)
    reader = vi.RoutedReader(lambda: qwen, qwen.signature, ocr=ocr, paid=paid, threshold=.10, paid_first=True)
    return reader, qwen, paid


def test_paid_first_reads_paidvl_pages_with_the_paid_model_and_skips_qwen(tmp_path, corpus, paid_chart):
    reader, qwen, paid = paid_first(table("120,500"), table("120,500"))
    _, by_page = run(tmp_path, corpus, reader)
    chart = by_page[2]
    assert chart["reader"] == "paidVL" and chart["escalation"] == "paid_first"
    assert paid.calls == 1 and qwen.calls == 0                        # no GPU time on this page
    assert reader.signature.endswith(":paid_first")


def test_paid_first_falls_back_to_qwen_when_the_paid_reading_disagrees_or_fails(tmp_path, corpus, paid_chart):
    reader, qwen, paid = paid_first(table("120,500"), table("120,580"))      # the paid model misreads a figure
    _, by_page = run(tmp_path, corpus, reader)
    assert by_page[2]["reader"] == "qwen3VL" and by_page[2]["escalation"] == "paid_first_qwen_better"
    assert paid.calls == 1 and qwen.calls == 1

    reader, qwen, paid = paid_first(table("120,500"), "")                    # the paid model returns nothing
    _, by_page = run(tmp_path / "failed", corpus, reader)
    assert by_page[2]["reader"] == "qwen3VL" and json.loads(by_page[2]["unverified_numbers"]) == []


def test_paid_first_without_a_paid_model_is_the_normal_chain():
    qwen = Reader("qwen", "x")
    reader = vi.RoutedReader(lambda: qwen, qwen.signature, ocr=Reader("ocr", OCR_TEXT), paid=None, paid_first=True)
    assert reader.paid_first is False and "paid_first" not in reader.signature


class SlowPaid(Reader):
    """A paid model that takes a moment per page and records how many pages it reads at once."""

    def __init__(self, text):
        super().__init__("slow-paid", text)
        import threading
        self.active = self.peak = 0
        self.guard = threading.Lock()

    def transcribe(self, image, page):
        import time
        with self.guard:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.05)
        with self.guard:
            self.active -= 1
        return self.text


@pytest.mark.parametrize("prefetch", ["1", "0"])
def test_prefetch_reads_paid_pages_in_parallel_and_indexes_the_same(tmp_path, corpus, monkeypatch, prefetch):
    monkeypatch.setenv("RAG_PAGE_PREFETCH", prefetch)
    monkeypatch.setattr(page_audit, "route", lambda page: "paidVL")             # every page needs the paid model
    ocr, qwen, paid = Reader("ocr", OCR_TEXT), Reader("qwen", table("120,500")), SlowPaid(table("120,500"))
    reader = vi.RoutedReader(lambda: qwen, qwen.signature, ocr=ocr, paid=paid, paid_first=True)
    _, by_page = run(tmp_path, corpus, reader)
    assert {m["reader"] for m in by_page.values()} == {"paidVL"} and paid.calls == len(by_page)
    assert qwen.calls == 0                                                      # Gemini agreed with OCR everywhere
    assert paid.peak == (min(4, len(by_page)) if prefetch == "1" else 1)


def test_prefetch_plan_follows_the_page_rules_and_limits():
    pages = [{"page": n, "page_type": "chart", "text": ""} for n in range(1, 7)]
    routes = {1: "qwen3VL", 2: "paidVL", 3: "paidVL", 4: "ocr", 5: "pdfParser", 6: "paidVL"}
    import scripts.rag.page_audit as audit
    real, audit.route = audit.route, lambda page: routes[page["page"]]
    try:
        qwen = Reader("qwen", "x")
        first = vi.RoutedReader(lambda: qwen, qwen.signature, ocr=Reader("ocr", OCR_TEXT), paid=Reader("paid", "y"),
                                paid_first=True, describe=False)
        plan = {lane: [(n, p["page"]) for n, p in items] for lane, items in vi.prefetch_plan(first, pages, limit=3).items()}
        assert plan["gpu"] == [("qwen3VL", 1)]
        assert plan["paid"] == [("paidVL", 2), ("paidVL", 3)]                   # page 6 is past the limit of 3 model pages
        assert plan["ocr"] == [("ocr", 2), ("ocr", 3), ("ocr", 4), ("ocr", 6)]
        chain = vi.RoutedReader(lambda: qwen, qwen.signature, ocr=Reader("ocr", OCR_TEXT), paid=Reader("paid", "y"))
        plan = {lane: [(n, p["page"]) for n, p in items] for lane, items in vi.prefetch_plan(chain, pages).items()}
        assert plan["paid"] == [] and plan["gpu"][:3] == [("describe", 1), ("qwen3VL", 1), ("describe", 2)]
    finally:
        audit.route = real


LONG_OCR = ("Certain of our non-bank subsidiaries are subject to regulation and supervision by various federal and state "
            "authorities, and broker-dealers must maintain minimum net capital. " * 12) + "Total assets 391,035."


def test_a_short_paid_reading_of_a_text_page_keeps_the_ocr_text(tmp_path, corpus, paid_chart):
    """A page of paragraphs the paid model refused to copy (RECITATION) and answered with a few figures only."""
    ocr, qwen, paid = Reader("ocr", LONG_OCR), Reader("qwen", table("120,500")), Reader("paid", "Total assets | 391,035")
    reader = vi.RoutedReader(lambda: qwen, qwen.signature, ocr=ocr, paid=paid, paid_first=True)
    ingest, by_page = run(tmp_path, corpus, reader)
    chart = by_page[2]
    assert chart["reader"] == "ocr" and chart["escalation"] == "paidVL_reading_short_ocr_kept"
    assert qwen.calls == 0                                        # OCR already has the page: no GPU time
    texts = ingest.store.collection.get(where={"page": 2}, include=["documents"])["documents"]
    body = "\n".join(texts)
    assert "broker-dealers must maintain minimum net capital" in body and "Total assets | 391,035" in body


def test_too_short_needs_a_real_page_of_ocr_text():
    assert vi.too_short("Total assets 391,035", LONG_OCR)
    assert not vi.too_short("Total assets 391,035", "Total assets 391,035 and 120,500")   # OCR found little
    assert not vi.too_short(LONG_OCR[:900], LONG_OCR)                                       # a full reading
