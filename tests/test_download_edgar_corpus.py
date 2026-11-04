import hashlib
import json

from scripts.downloads import download_edgar_ars_pdfs as ars
from scripts.downloads import download_edgar_corpus as dc

CIK = "0000000042"
PDF = b"%PDF-1.7 fake annual report"


def filing(form, filed, report, accession, primary):
    return {"form": form, "filingDate": filed, "reportDate": report, "accessionNumber": accession,
            "primaryDocument": primary}


FILINGS = [
    filing("10-K", "2025-02-20", "2024-12-31", "0000000042-25-000010", "x-20241231.htm"),
    filing("10-K/A", "2025-04-01", "2024-12-31", "0000000042-25-000020", "x-20241231a.htm"),
    filing("DEF 14A", "2025-03-10", "", "0000000042-25-000015", "proxy2025.htm"),
    filing("DEF 14A", "2026-03-12", "", "0000000042-26-000015", "proxy2026.htm"),
    filing("10-K", "2026-02-19", "2025-12-31", "0000000042-26-000010", "x-20251231_d2.htm"),
]


def test_match_tenk_and_proxy():
    row = {"year": 2024, "filing_date": "2025-03-10"}
    tenk = dc.match_tenk(row, FILINGS)
    assert tenk["accessionNumber"] == "0000000042-25-000010"          # amendment ignored
    assert dc.match_proxy(tenk, FILINGS)["primaryDocument"] == "proxy2025.htm"
    assert dc.match_tenk({"year": 2021, "filing_date": "2025-03-10"}, FILINGS) is None


def fake_sec(monkeypatch):
    columns = {key: [f[key] for f in FILINGS] for key in FILINGS[0]}
    base = "https://www.sec.gov/Archives/edgar/data/42/000000004226000010/"
    responses = {
        f"https://data.sec.gov/submissions/CIK{CIK}.json": json.dumps({"filings": {"recent": columns, "files": []}}).encode(),
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json": b'{"facts": {}}',
        "https://example.invalid/ars.pdf": PDF,
        base + "x-20251231_d2.htm": b"<html><ix:nonFraction>1</ix:nonFraction></html>",
        base + "index.json": json.dumps({"directory": {"item": [
            {"name": "x-20251231_d2.htm"}, {"name": "x-20251231.htm"}, {"name": "R1.htm"},
            {"name": "x-1231x2025xex21.htm"}]}}).encode(),
        base + "x-20251231.htm": b"<html>" + b"<ix:nonFraction>9</ix:nonFraction>" * 600 + b"</html>",
        "https://www.sec.gov/Archives/edgar/data/42/000000004226000015/proxy2026.htm": b"<html>proxy</html>",
    }
    calls = []

    def get(url, user_agent, timeout):
        calls.append(url)
        return responses[url]
    monkeypatch.setattr(ars, "http_get", get)
    return calls


def test_download_links_filing_and_resumes(tmp_path, monkeypatch):
    calls = fake_sec(monkeypatch)
    manifest = tmp_path / "pdf_manifest.jsonl"
    manifest.write_text(json.dumps({
        "document_id": "edgar_ars_X_2025", "ticker": "X", "cik": CIK, "year": 2025, "company": "X Corp",
        "accession_number": "0000000042-26-000015", "filing_date": "2026-03-12",
        "source_url": "https://example.invalid/ars.pdf", "local_path": "data\\raw\\EDGAR_ARS_X_2025.pdf",
        "content_sha256": hashlib.sha256(PDF).hexdigest()}) + "\n", encoding="utf-8")
    argv = ["--manifest", str(manifest), "--output-dir", str(tmp_path / "edgar"), "--user-agent", "t a@b.c"]
    assert dc.main(argv) == 0
    record = json.loads((tmp_path / "edgar" / "corpus_manifest.jsonl").read_text(encoding="utf-8"))
    assert record["filing_id"] == "X_FY2025" and record["errors"] == {}
    assert record["tenk_accession"] == "0000000042-26-000010" and record["proxy_url"].endswith("proxy2026.htm")
    assert [d["name"] for d in record["tenk_companion_documents"]] == ["x-20251231.htm"]   # no R pages, no exhibits
    assert record["tenk_ixbrl_numeric_facts_total"] == 601
    assert (tmp_path / "edgar" / "pdfs" / "EDGAR_ARS_X_2025.pdf").read_bytes() == PDF

    first = len(calls)
    assert dc.main(argv) == 0
    assert len(calls) - first == 1          # only the companion index is re-read; every file is reused


def test_pdf_with_wrong_checksum_is_deleted(tmp_path, monkeypatch):
    fake_sec(monkeypatch)
    manifest = tmp_path / "pdf_manifest.jsonl"
    manifest.write_text(json.dumps({
        "document_id": "edgar_ars_X_2025", "ticker": "X", "cik": CIK, "year": 2025, "filing_date": "2026-03-12",
        "source_url": "https://example.invalid/ars.pdf", "local_path": "EDGAR_ARS_X_2025.pdf",
        "content_sha256": "0" * 64}) + "\n", encoding="utf-8")
    assert dc.main(["--manifest", str(manifest), "--output-dir", str(tmp_path / "edgar"),
                    "--user-agent", "t a@b.c", "--only", "pdf"]) == 1
    assert not (tmp_path / "edgar" / "pdfs" / "EDGAR_ARS_X_2025.pdf").exists()
