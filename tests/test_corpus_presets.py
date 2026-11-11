"""Corpus presets: full (1,004 filings) and min500 (509) drive download, ingest and coverage alike."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest

from scripts.common.sector_corpus import read_jsonl, sector_slug
from scripts.downloads import download_annual_report_corpus as dl
from scripts.rag import colab_pipeline as cp
from scripts.rag import financial_ingest as fi

ROOT = Path(__file__).resolve().parents[1]
PLAN = read_jsonl(dl.DEFAULT_DIR / dl.PLAN_NAME)
EVAL_FILINGS = set(re.findall(r"edgar_ars_[A-Za-z.-]+_\d{4}", dl.EVAL_SET.read_text(encoding="utf-8")))


def ids(rows):
    return {dl.document_id(r) for r in rows}


def test_full_preset_is_the_whole_plan():
    assert len(dl.preset_rows(PLAN, "full")) == len(PLAN) == 1004


def test_min500_meets_track_b_and_keeps_every_eval_filing():
    rows = dl.preset_rows(PLAN, "min500")
    assert 500 <= len(rows) <= 520                                  # Track B: 500–2,000 documents
    assert len(rows) == 509 and len({r["ticker"] for r in rows}) == 143
    assert EVAL_FILINGS and EVAL_FILINGS <= ids(rows)               # the frozen eval set loses no source
    per_sector = Counter(sector_slug(r["sector"]) for r in rows)
    assert len(per_sector) == 13                                    # every sector represented
    assert all(n <= dl.MIN500_PER_SECTOR for slug, n in per_sector.items() if slug != "technology_and_it")
    assert ids(rows) <= ids(PLAN)


def test_unknown_preset_is_rejected():
    with pytest.raises(SystemExit, match="unknown corpus preset"):
        dl.preset_rows(PLAN, "tiny")


def test_download_selection_applies_the_preset_before_other_filters(tmp_path):
    args = dl.parse_args(["--plan", str(dl.DEFAULT_DIR / dl.PLAN_NAME), "--output-dir", str(tmp_path),
                          "--preset", "min500", "--max-per-sector", "2"])
    rows = dl.select(PLAN, args)
    assert len(rows) == 26 and ids(rows) <= dl.preset_document_ids("min500")
    assert len(dl.select(PLAN, dl.parse_args(["--output-dir", str(tmp_path)]))) == 1004    # default stays full


def test_ingest_keeps_only_the_presets_filings():
    inside, outside = sorted(dl.preset_document_ids("min500"))[0], sorted(ids(PLAN) - dl.preset_document_ids("min500"))[0]
    rows = [{"document_id": inside}, {"document_id": outside}]
    assert fi.in_preset(rows, "full") == rows
    assert fi.in_preset(rows, "min500") == [{"document_id": inside}]
    for parser in (fi.build_parser(),):
        assert parser.parse_args(["ingest", "--corpus-preset", "min500"]).corpus_preset == "min500"
    from scripts.rag import vlm_ingest as vi
    assert vi.build_parser().parse_args(["ingest", "--corpus-preset", "min500"]).corpus_preset == "min500"


def test_colab_settings_and_coverage_use_the_preset(tmp_path):
    with pytest.raises(ValueError, match="corpus_preset"):
        cp.Settings(corpus_preset="tiny", drive_root=tmp_path / "d", work=tmp_path / "w")
    for preset, planned in (("full", 1004), ("min500", 509)):
        s = cp.Settings(corpus_preset=preset, drive_root=tmp_path / "d", work=tmp_path / "w")
        report = cp.coverage(s, show=False)
        assert report["total"]["planned"] == planned and report["corpus_preset"] == preset
        assert "--corpus-preset" in cp.common_argv(s, "banking") and preset in cp.common_argv(s, "banking")
    assert cp.build_parser().parse_args(["coverage", "--corpus-preset", "min500"]).corpus_preset == "min500"


def test_download_passes_the_preset(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(dl, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    cp.download(cp.Settings(corpus_preset="min500", drive_root=tmp_path / "d", work=tmp_path / "w", max_per_sector=None))
    assert seen["argv"][:4] == ["--output-dir", str(tmp_path / "d" / "annual_reports"), "--preset", "min500"]


def test_notebook_exposes_the_corpus_setting():
    nb = json.loads((ROOT / "notebooks/colab_rag_pipelines.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert 'CORPUS = "full"  # @param ["full", "min500"]' in source
    assert "corpus_preset=CORPUS" in source


def test_preset_report_lists_every_filing_by_sector(tmp_path):
    from scripts.downloads import corpus_preset_report as report
    out = tmp_path / "min500.html"
    assert report.main(["--preset", "min500", "--out", str(out)]) == 0
    page = out.read_text(encoding="utf-8")
    assert page.count("<tbody class='filings'>") == 13
    assert page.count("<tr><td class='num'>") == 509
    assert page.count("<td>✓</td>") == len(EVAL_FILINGS)
    assert page.count(">eval-set company<") == 8                    # Adobe and Salesforce, 4 years each
