import hashlib
import json

import pytest

from scripts.rag import colab_pipeline as cp
from tests.test_financial_ingest_qdrant_db import FakeColPali, make_pdf


@pytest.fixture
def drive(tmp_path, monkeypatch):
    """<drive>/annual_reports/{banking,oil_and_gas}/ with one filing each, as the downloader writes it."""
    monkeypatch.setattr(cp, "_loaded", {})
    for sector, slug, ticker in (("Banking", "banking", "JPM"), ("Oil and gas", "oil_and_gas", "CVX")):
        folder = tmp_path / "drive" / "annual_reports" / slug
        folder.mkdir(parents=True)
        pdf = folder / f"EDGAR_ARS_{ticker}_2025.pdf"
        make_pdf(pdf)
        row = {"document_id": f"edgar_ars_{ticker}_2025", "company": ticker, "ticker": ticker, "year": 2025,
               "title": f"{ticker} annual report 2025", "source_url": "https://example.invalid/x.pdf", "sector": sector,
               "sector_slug": slug, "local_path": pdf.name, "content_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest()}
        (folder / "pdf_manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return tmp_path


def settings(tmp_path, pipeline, **overrides):
    values = dict(pipeline=pipeline, drive_root=tmp_path / "drive", work=tmp_path / "work", qdrant_url=None,
                  embedding_provider="hashing", vlm="mock", workers=1, ocr="none")
    return cp.Settings(**{**values, **overrides})


def test_colpali_pipeline_puts_pooled_vectors_in_ram_and_full_vectors_on_disk(drive):
    s = settings(drive, "colpali_qdrant")
    cp._loaded["colpali"] = FakeColPali()
    cp.prepare(s)
    assert cp.ingest(s).keys() == {"banking", "oil_and_gas"}

    report = cp.status(s)
    visual, text = report["collections"][cp.fi.VISUAL], report["collections"][cp.fi.TEXT]
    assert visual["vectors"] == {"colpali": "disk", "mean_rows": "RAM", "mean_cols": "RAM", "summary": "RAM"}
    assert text["vectors"] == {"dense": "RAM"}            # payload placement is only reported by a Qdrant server
    assert visual["by_sector"] == {"banking": 1, "oil_and_gas": 1}            # tables_charts_images: the chart page only
    assert text["by_sector"]["banking"] == text["by_sector"]["oil_and_gas"] > 0

    rows = cp.query(s, "391035", sector="oil_and_gas")
    assert rows and {r["sector"] for r in rows} == {"Oil and gas"}
    assert {r["path"] for r in cp.query(s, "segment chart", sector="banking", visual=True)} == {"text", "visual"}
    assert cp.ingest(s) == {"banking": {"minutes": 0.0}, "oil_and_gas": {"minutes": 0.0}}   # nothing left to do


def test_vlm_pipeline_writes_chroma_backs_up_to_drive_and_restores(drive):
    s = settings(drive, "vlm_chroma", sectors="banking")
    cp.prepare(s)
    assert list(cp.ingest(s)) == ["banking"]                                    # only the chosen sector
    report = cp.status(s)
    assert report["chroma"]["by_sector"] == {"banking": report["chroma"]["chunks"]}
    assert report["vlm_cache"]["transcribed_pages"] == 2                        # visual: the chart and the unreadable page
    assert (s.backup / "vlm_ingest.sqlite3").exists() and (s.backup / "chroma").exists()
    assert (s.vlm_cache / "edgar_ars_JPM_2025.jsonl").exists()                  # transcriptions straight on Drive

    later = settings(drive, "vlm_chroma", sectors="banking", work=drive / "new_session")   # a fresh Colab VM
    assert cp.restore(later) is True
    assert cp.status(later)["chroma"]["chunks"] == report["chroma"]["chunks"]
    assert cp.query(later, "Total net sales 391,035", sector="banking")[0]["sector"] == "Banking"
    assert cp.ingest(later) == {"banking": {"minutes": 0.0}}                    # state came back too
    assert cp.restore(later) is False                                           # already has the index


def test_cli_builds_settings_and_rejects_unknown_pipelines(drive, capsys):
    cp.main(["status", "--pipeline", "vlm_chroma", "--drive-root", str(drive / "drive"), "--work", str(drive / "work")])
    assert json.loads(capsys.readouterr().out)["chroma"]["chunks"] == 0
    with pytest.raises(ValueError, match="pipeline must be one of"):
        cp.Settings(pipeline="faiss")
    assert cp.sectors_to_run(settings(drive, "vlm_chroma", sectors="Oil and gas,insurance")) == ["oil_and_gas"]


def test_sector_pdfs_are_staged_on_local_disk_before_ingest(drive):
    s = settings(drive, "colpali_qdrant", sectors="banking", stage_pdfs=True, stages="text")
    cp.prepare(s)
    assert cp.ingest(s)["banking"]["complete"] == 1
    staged = s.work / "annual_reports" / "banking"
    assert sorted(p.name for p in staged.iterdir()) == ["EDGAR_ARS_JPM_2025.pdf", "pdf_manifest.jsonl"]
    assert not (s.work / "annual_reports" / "oil_and_gas").exists()           # only the sector being ingested
    pdf = staged / "EDGAR_ARS_JPM_2025.pdf"
    before = pdf.stat().st_mtime_ns
    assert cp.stage(s, "banking") == s.work / "annual_reports" and pdf.stat().st_mtime_ns == before   # not copied again


def test_both_names_of_the_visual_page_rule_work_for_both_pipelines(drive, monkeypatch):
    seen = {}
    monkeypatch.setattr(cp.fi, "Ingest", lambda args, **kw: seen.setdefault("colpali", args.visual_pages) and None)
    monkeypatch.setattr(cp.vi, "VlmIngest", lambda args, **kw: seen.setdefault("vlm", args.vlm_pages) and None)
    for pipeline, call in (("colpali_qdrant", cp.ingest_colpali), ("vlm_chroma", cp.ingest_vlm)):
        with pytest.raises(AttributeError):                    # stops after parsing the arguments
            call(settings(drive, pipeline, visual_pages="visual", vlm_pages="rule"), "banking")
    assert seen == {"colpali": "rule", "vlm": "visual"}


def test_cli_parses_booleans_and_zero_means_the_whole_plan(drive):
    args = cp.build_parser().parse_args(["download", "--stage-pdfs", "false", "--max-per-sector", "0",
                                         "--snapshot-every", "3"])
    assert args.stage_pdfs is False and args.max_per_sector == 0 and args.snapshot_every == 3
    assert cp.build_parser().parse_args(["download", "--stage-pdfs", "True"]).stage_pdfs is True


def write_plan(drive, extra=()):
    rows = [{"sector": "Banking", "ticker": "JPM", "year": 2025}, {"sector": "Oil and gas", "ticker": "CVX", "year": 2025},
            *extra]
    (drive / "drive" / "annual_reports" / "corpus_plan.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_coverage_counts_every_planned_filing_by_sector(drive):
    write_plan(drive, [{"sector": "Banking", "ticker": "BAC", "year": 2025}])      # planned, never downloaded
    s = settings(drive, "vlm_chroma")
    before = cp.coverage(s)
    assert before["total"] == {"planned": 3, "downloaded": 2, "ingested": 0, "pending": 2, "failed": 0}
    cp.prepare(s)
    cp.ingest(s)
    after = cp.coverage(s)
    assert after["by_sector"]["banking"] == {"planned": 2, "downloaded": 1, "ingested": 1, "pending": 0, "failed": 0}
    assert after["total"]["ingested"] == 2 and after["failed"] == [] and after["not_in_plan"] == []


def test_colpali_coverage_needs_every_stage(drive):
    write_plan(drive)
    s = settings(drive, "colpali_qdrant", stages="text")
    cp.prepare(s)
    cp.ingest(s)
    assert cp.coverage(s)["total"]["ingested"] == 2
    both = settings(drive, "colpali_qdrant", stages="text,visual")       # visual not run yet: not ingested
    assert cp.coverage(both, show=False)["total"]["ingested"] == 0


def test_serving_env_points_the_api_at_this_pipelines_index(drive):
    env = cp.serving_env(settings(drive, "colpali_qdrant", qdrant_url="http://127.0.0.1:6333"))
    assert env["RAG_BACKEND"] == "financial_qdrant" and env["QDRANT_URL"] == "http://127.0.0.1:6333"
    assert env["RAG_INGEST_DIR"] == str(drive / "work" / "indexes" / "qdrant_db")     # the storage layout
    env = cp.serving_env(settings(drive, "vlm_chroma", run_name="_smoke"))
    assert env["RAG_BACKEND"] == "financial_vlm"
    assert env["RAG_VLM_DIR"] == str(drive / "work" / "indexes" / "chroma_db_smoke")
    assert env["RAG_HOME"] == str(drive / "work")
    with pytest.raises(RuntimeError, match="Qdrant server"):
        cp.serve(settings(drive, "colpali_qdrant"))


def test_agent_answers_from_the_vlm_index_with_a_sector_filter(drive, monkeypatch):
    from scripts.agents import financial_prompting_agent as agent
    from scripts.rag import financial_ingest_chroma_db
    s = settings(drive, "vlm_chroma")
    cp.prepare(s)
    cp.ingest(s)
    for key, value in cp.serving_env(s).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(financial_ingest_chroma_db, "_serving", {})
    config = agent.AgentConfig(query="391035 revenue", ticker=None, year=None, source_type=None, top_k=5, dense_top_k=5,
                               bm25_top_k=5, retrieval_mode="hybrid", embedding_provider="hashing", hash_dimensions=384,
                               max_context_chars=4000, self_consistency_samples=1, sector="oil_and_gas")
    hits = agent.retrieve_evidence(config)
    assert hits and {h["sector"] for h in hits} == {"Oil and gas"}
    assert all(h["chunk_id"].startswith("annual_report_edgar_ars_CVX_2025") and "rerank_score" in h for h in hits)


def test_verify_upload_and_reader_summary_after_a_restore(drive):
    write_plan(drive, [])
    s = settings(drive, "vlm_chroma")
    cp.prepare(s)
    cp.ingest(s)
    (s.drive_root / "capstone_code.zip").write_bytes(b"zip")              # no mutual fund zip: reported, not fatal
    later = settings(drive, "vlm_chroma", work=drive / "new_session")         # the Colab VM after the Drive upload
    first = cp.verify_upload(later)
    assert first["failed"] == ["mutual fund zip on Drive", "local copy restored"]   # nothing restored yet
    cp.restore(later)
    assert cp.verify_upload(later)["failed"] == ["mutual fund zip on Drive"]
    report = cp.reader_summary(later)
    assert report["all_filings_present"] and report["filings"]["in_chroma"] == 2
    assert report["chunks"]["chroma"] == report["chunks"]["bm25"] > 0
    assert sum(report["pages_by_reader"].values()) == report["pages_indexed"] > 0
    assert set(cp.PAGE_READERS) <= set(report["pages_by_reader"])
    (s.drive_root / "annual_reports" / "banking" / "EDGAR_ARS_JPM_2025.pdf").unlink()   # an upload that lost a PDF
    assert "PDF files present and non-empty" in cp.verify_upload(later)["failed"]
