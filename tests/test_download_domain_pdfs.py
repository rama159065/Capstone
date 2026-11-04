from __future__ import annotations

import io

import scripts.downloads.download_domain_pdfs as domain_pdfs


class FakeUrlopen:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> io.BytesIO:
        return io.BytesIO(self.payload)

    def __exit__(self, *args: object) -> None:
        return None


def test_source_domains_cover_remaining_track_b_domains() -> None:
    domains = {source.domain for source in domain_pdfs.PDF_SOURCES}

    assert {
        "procurement_vendor",
        "travel_expense",
        "competitive_research",
        "incident_postmortem",
        "hr_policy",
    } <= domains


def test_download_pdf_accepts_only_pdf(monkeypatch) -> None:
    monkeypatch.setattr(
        domain_pdfs.urllib.request,
        "urlopen",
        lambda request, timeout: FakeUrlopen(b"%PDF-1.7 fake"),
    )

    assert domain_pdfs.download_pdf("https://example.test/doc.pdf", "ua", 1) == b"%PDF-1.7 fake"

    monkeypatch.setattr(
        domain_pdfs.urllib.request,
        "urlopen",
        lambda request, timeout: FakeUrlopen(b"<html>not a pdf</html>"),
    )

    assert domain_pdfs.download_pdf("https://example.test/doc.pdf", "ua", 1) is None


def test_manifest_id_helpers(tmp_path) -> None:
    manifest = tmp_path / "pdf_manifest.jsonl"

    domain_pdfs.write_manifest_row(manifest, {"document_id": "doc-1", "title": "One"})

    assert domain_pdfs.safe_filename("A/B C?.pdf") == "A_B_C_.pdf"
    assert domain_pdfs.existing_ids(manifest) == {"doc-1"}
    assert domain_pdfs.sha256_bytes(b"abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
