from __future__ import annotations

import io

import scripts.download_financial_report_pdfs as pdfs


class FakeUrlopen:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> io.BytesIO:
        return io.BytesIO(self.payload)

    def __exit__(self, *args: object) -> None:
        return None


def test_annual_report_urls_and_safe_filename() -> None:
    company = pdfs.Company("NASDAQ", "AAPL", "Apple Inc.")

    urls = list(pdfs.annual_report_urls(company, 2024))

    assert len(urls) == 3
    assert "NASDAQ_AAPL_2024.pdf" in urls[0]
    assert pdfs.safe_filename("NASDAQ/AAPL 2024?.pdf") == "NASDAQ_AAPL_2024_.pdf"


def test_download_pdf_accepts_only_pdf_payload(monkeypatch) -> None:
    monkeypatch.setattr(
        pdfs.urllib.request,
        "urlopen",
        lambda request, timeout: FakeUrlopen(b"%PDF-1.7 fake"),
    )

    assert pdfs.download_pdf("https://example.test/report.pdf", "ua", 1) == b"%PDF-1.7 fake"

    monkeypatch.setattr(
        pdfs.urllib.request,
        "urlopen",
        lambda request, timeout: FakeUrlopen(b"<html>not a pdf</html>"),
    )

    assert pdfs.download_pdf("https://example.test/report.pdf", "ua", 1) is None


def test_manifest_helpers_round_trip(tmp_path) -> None:
    manifest = tmp_path / "pdf_manifest.jsonl"
    row = {"ticker": "AAPL", "year": 2024, "content_sha256": pdfs.sha256_bytes(b"abc")}

    pdfs.write_manifest_row(manifest, row)

    assert pdfs.existing_keys(manifest) == {("AAPL", 2024)}
    assert row["content_sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
