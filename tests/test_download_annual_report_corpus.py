import io
import json
import urllib.error

import pytest

from scripts.common import sector_corpus as sc
from scripts.downloads import download_annual_report_corpus as dl

PDF = b"%PDF-1.7\n" + b"x" * 200
REUSED = b"%PDF-1.4\n" + b"r" * 120


def plan_row(sector, ticker, year, size, rank=1):
    return {"sector": sector, "rank": rank, "ticker": ticker, "company": f"{ticker} Corp", "cik": 123,
            "sic": "6021", "sic_desc": "National Commercial Banks", "year": year, "filed": f"{year + 1}-03-01",
            "accession": f"0000000123-{year % 100}-000001", "url": f"https://www.sec.gov/x/{ticker}_{year}.pdf",
            "bytes": size}


@pytest.fixture
def plan(tmp_path):
    rows = [plan_row("Banking", "JPM", 2025, len(PDF)), plan_row("Banking", "BAC", 2025, len(REUSED), rank=2),
            plan_row("Oil and gas", "XOM", 2025, len(PDF)), plan_row("Oil and gas", "BAD", 2025, len(PDF), rank=2)]
    path = tmp_path / "plan" / sc.PLAN_NAME
    path.parent.mkdir()
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    reuse = tmp_path / "reuse"
    reuse.mkdir()
    (reuse / "EDGAR_ARS_BAC_2025.pdf").write_bytes(REUSED)
    return path, reuse


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def fake_sec(calls):
    def urlopen(request, timeout=None):
        calls.append(request.full_url)
        assert "@" in request.get_header("User-agent")
        if "BAD" in request.full_url:
            return FakeResponse(b"<html>not found</html>".ljust(len(PDF)))
        return FakeResponse(PDF)
    return urlopen


def argv(plan, tmp_path, *extra):
    path, reuse = plan
    return ["--plan", str(path), "--output-dir", str(tmp_path / "corpus"), "--reuse-dir", str(reuse),
            "--user-agent", "test you@example.com", "--max-requests-per-second", "10", *extra]


def args(plan, tmp_path, *extra):
    return dl.parse_args(argv(plan, tmp_path, *extra))


def test_sector_slug_and_parse():
    assert sc.sector_slug("Oil and gas") == "oil_and_gas"
    assert sc.sector_slug("Technology and IT") == "technology_and_it"
    assert sc.sector_slug(None) == "unknown"
    assert sc.parse_sectors("Banking, oil_and_gas") == {"banking", "oil_and_gas"}
    assert sc.parse_sectors("") is None


def test_select_filters_by_sector_name_or_slug(plan, tmp_path):
    rows = dl.read_jsonl(plan[0])
    assert [r["ticker"] for r in dl.select(rows, args(plan, tmp_path, "--sectors", "Oil and gas"))] == ["XOM", "BAD"]
    assert [r["ticker"] for r in dl.select(rows, args(plan, tmp_path, "--max-per-sector", "1"))] == ["JPM", "XOM"]
    with pytest.raises(SystemExit, match="unknown sector"):
        dl.select(rows, args(plan, tmp_path, "--sectors", "mining"))


def test_download_writes_sector_folders_and_manifests(plan, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_sec(calls))
    assert dl.main(argv(plan, tmp_path)) == 1             # BAD is not a PDF

    corpus = tmp_path / "corpus"
    assert (corpus / sc.PLAN_NAME).exists()
    assert (corpus / "banking" / "EDGAR_ARS_JPM_2025.pdf").read_bytes() == PDF
    assert (corpus / "banking" / "EDGAR_ARS_BAC_2025.pdf").read_bytes() == REUSED      # copied, not fetched
    assert not list(corpus.rglob("*.part")) and not (corpus / "oil_and_gas" / "EDGAR_ARS_BAD_2025.pdf").exists()
    assert sorted(calls) == sorted(["https://www.sec.gov/x/JPM_2025.pdf", "https://www.sec.gov/x/XOM_2025.pdf",
                                    "https://www.sec.gov/x/BAD_2025.pdf"])

    banking = sc.read_jsonl(corpus / "banking" / sc.MANIFEST_NAME)
    assert {r["ticker"] for r in banking} == {"JPM", "BAC"}
    jpm = next(r for r in banking if r["ticker"] == "JPM")
    assert jpm["local_path"] == "EDGAR_ARS_JPM_2025.pdf" and jpm["sector"] == "Banking"
    assert jpm["sector_slug"] == "banking" and jpm["document_id"] == "edgar_ars_JPM_2025"
    assert jpm["cik"] == "0000000123" and jpm["bytes"] == len(PDF) and len(jpm["content_sha256"]) == 64
    combined = sc.read_jsonl(corpus / sc.MANIFEST_NAME)
    assert {r["local_path"] for r in combined} == {"banking/EDGAR_ARS_BAC_2025.pdf", "banking/EDGAR_ARS_JPM_2025.pdf",
                                                   "oil_and_gas/EDGAR_ARS_XOM_2025.pdf"}
    summary = json.loads((corpus / "download_summary.json").read_text(encoding="utf-8"))
    assert summary["by_sector"]["banking"] == {"downloaded": 1, "reused": 1}
    assert [f["document_id"] for f in summary["failures"]] == ["edgar_ars_BAD_2025"]

    calls.clear()                                                       # rerun: only the failure is retried
    dl.main(argv(plan, tmp_path))
    assert calls == ["https://www.sec.gov/x/BAD_2025.pdf"]
    assert len(sc.read_jsonl(corpus / "banking" / sc.MANIFEST_NAME)) == 2


def test_reader_reads_the_corpus_folder_by_sector(plan, tmp_path, monkeypatch):
    monkeypatch.setattr(dl.urllib.request, "urlopen", fake_sec([]))
    dl.main(argv(plan, tmp_path))
    corpus = tmp_path / "corpus"
    rows = sc.read_manifest(corpus)
    assert {r["sector_slug"] for r in rows} == {"banking", "oil_and_gas"}
    assert all(sc.resolve_local_path(r).exists() for r in rows)
    assert [r["ticker"] for r in sc.read_manifest(corpus, "Oil and gas")] == ["XOM"]
    combined = sc.read_manifest(corpus / sc.MANIFEST_NAME, "banking")       # the all-sector file works too
    assert sorted(r["ticker"] for r in combined) == ["BAC", "JPM"]
    assert all(sc.resolve_local_path(r).exists() for r in combined)


def test_throttling_is_retried_and_user_agent_required(plan, tmp_path, monkeypatch):
    attempts = []

    def urlopen(request, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise urllib.error.HTTPError(request.full_url, 429, "Too Many Requests", {}, None)
        return FakeResponse(PDF)
    monkeypatch.setattr(dl.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(dl.time, "sleep", lambda s: None)
    assert dl.main(argv(plan, tmp_path, "--tickers", "JPM")) == 0
    assert len(attempts) == 2
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(SystemExit, match="contact e-mail"):
        dl.main(["--plan", str(plan[0]), "--output-dir", str(tmp_path / "c2")])

