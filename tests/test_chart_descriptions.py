"""Pipeline 1 (Chroma): Qwen3-VL search descriptions make chart and image pages findable."""
import json

import pytest

from scripts.rag import financial_ingest_chroma_db as vi
from scripts.rag import page_audit
from tests.test_financial_ingest_chroma_db import args_for, corpus  # noqa: F401  (Chroma corpus fixture)

DESCRIPTION = ("Bar chart titled Revenue by segment, comparing Upstream and Downstream segment revenue "
               "for fiscal 2024 and fiscal 2025.")


class Qwen:
    """Stand-in Qwen3-VL: a transcription and a page description."""
    signature = "fake-qwen3vl-v1"

    def __init__(self, description=DESCRIPTION, fail=False):
        self.description, self.fail, self.transcribed, self.described = description, fail, 0, 0

    def transcribe(self, image, page):
        self.transcribed += 1
        return "Revenue by segment (in millions)\n| Segment | 2025 |\n|---|---|\n| Upstream | 391,035 |"

    def describe(self, image, page):
        self.described += 1
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        return self.description


class OCR:
    signature = "fake-ocr-v1"

    def transcribe(self, image, page):
        return "Revenue by segment in millions for fiscal 2025 and fiscal 2024 Upstream 391,035 Downstream 120,500"


def ingest(tmp_path, corpus, qwen, *extra):
    reader = vi.RoutedReader(lambda: qwen, qwen.signature, ocr=OCR(), describe="--no-describe-visuals" not in extra)
    run = vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "routed", *extra), vlm=reader)
    assert run.run() == {"complete": 1}
    metas = run.store.collection.get(include=["metadatas", "documents"])
    return run, [(m, d) for m, d in zip(metas["metadatas"], metas["documents"])]


def test_chart_page_gets_a_searchable_description_chunk(tmp_path, corpus):
    qwen = Qwen()
    run, chunks = ingest(tmp_path, corpus, qwen)
    described = [(m, d) for m, d in chunks if m["content_type"] == "page_description"]
    assert len(described) == 1 and qwen.described == 1                  # page 2 is the only chart page
    meta, text = described[0]
    assert meta["page"] == 2 and meta["reader"] == "qwen3VL" and meta["text_source"] == "vlm"
    assert meta["chunk_id"].endswith("_p0002_desc0") and text.startswith("Chart on page 2 (page description): Bar chart")
    assert json.loads(meta.get("unverified_numbers", "[]")) == []        # only years, which the page prints
    hit = vi.retrieve("bar chart of revenue by segment", 1, root=tmp_path / "chroma", embedder=run.embedder)[0]
    assert hit["content_type"] == "page_description" and hit["page"] == 2

    again = vi.VlmIngest(args_for(tmp_path, corpus, "chroma", "--vlm", "routed", "--force"),
                         vlm=vi.RoutedReader(lambda: qwen, qwen.signature, ocr=OCR()))
    assert again.run() == {"complete": 1} and qwen.described == 1        # cached: no second call


def test_readable_chart_routed_to_the_text_layer_is_described_too(tmp_path, corpus, monkeypatch):
    real = page_audit.route
    monkeypatch.setattr(page_audit, "route", lambda p: "pdfParser+colpali" if p["page"] == 2 else real(p))
    qwen = Qwen()
    _, chunks = ingest(tmp_path, corpus, qwen)
    kinds = {(m["page"], m["content_type"]) for m, _ in chunks}
    assert (2, "page_description") in kinds and qwen.transcribed == 0    # described, but not transcribed


def test_switch_off_failures_and_the_mock_add_nothing(tmp_path, corpus):
    _, chunks = ingest(tmp_path / "off", corpus, Qwen(), "--no-describe-visuals")
    assert not [m for m, _ in chunks if m["content_type"] == "page_description"]
    _, chunks = ingest(tmp_path / "fail", corpus, Qwen(fail=True))      # the filing still completes
    assert not [m for m, _ in chunks if m["content_type"] == "page_description"]
    mock = vi.RoutedReader(lambda: vi.MockVLM(), vi.MockVLM.signature, ocr=OCR())
    run = vi.VlmIngest(args_for(tmp_path / "mock", corpus, "chroma", "--vlm", "routed"), vlm=mock)
    run.run()
    assert not [m for m in run.store.collection.get(include=["metadatas"])["metadatas"]
                if m["content_type"] == "page_description"]               # the mock invents no descriptions


def test_description_figures_not_on_the_page_are_flagged(tmp_path, corpus):
    _, chunks = ingest(tmp_path, corpus, Qwen(DESCRIPTION + " Revenue peaked at 987,654."))
    meta = next(m for m, _ in chunks if m["content_type"] == "page_description")
    assert json.loads(meta["unverified_numbers"]) == ["987,654"]


def test_signature_includes_the_description_switch():
    on = vi.vlm_signature(vi.build_parser().parse_args(["ingest", "--vlm", "routed", "--qwen-reader", "mock"]))
    off = vi.vlm_signature(vi.build_parser().parse_args(["ingest", "--vlm", "routed", "--qwen-reader", "mock",
                                                         "--no-describe-visuals"]))
    assert on.endswith(":describe=v1") and "describe" not in off
