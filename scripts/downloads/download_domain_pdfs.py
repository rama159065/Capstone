"""Download public PDF corpora for the non-financial Track B domains.

PDFs are written under:

    data/raw/<domain>/pdfs/*.pdf
    data/raw/<domain>/pdf_manifest.jsonl

The sources are intentionally public government or standards documents where
possible, so the PDF corpus can be cited and reprocessed for RAG.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path("data/raw")
SLEEP_SECONDS = 0.3


@dataclass(frozen=True)
class PdfSource:
    domain: str
    document_id: str
    title: str
    source_name: str
    source_url: str
    source_type: str


PDF_SOURCES = [
    # Procurement and vendor review.
    PdfSource(
        "procurement_vendor",
        "worldbank_procurement_regulations_2025",
        "World Bank Procurement Regulations for IPF Borrowers, September 2025",
        "World Bank",
        "https://thedocs.worldbank.org/en/doc/c84273d1b230aeb2b0b8134de5dc8cd7-0290012025/original/Procurement-Regulations-7th-Edition-Sep-2025.pdf",
        "procurement_policy_pdf",
    ),
    PdfSource(
        "procurement_vendor",
        "worldbank_ppsd_guidance_2024",
        "Project Procurement Strategy for Development Guidance, August 2024",
        "World Bank",
        "https://thedocs.worldbank.org/en/doc/ae0f8a5bf130bce9c3bf23f088a6c3a9-0290012024/original/PPSD-Procurement-Guidance-FINAL-Aug-24-WEB.pdf",
        "procurement_guidance_pdf",
    ),
    PdfSource(
        "procurement_vendor",
        "worldbank_beginner_guide_borrowers_2023",
        "Beginner's Guide for Borrowers - Procurement under World Bank IPF",
        "World Bank",
        "https://thedocs.worldbank.org/en/doc/caf6a3f677305292d799dbe55fd83bfe-0290012023/original/PROCUREMENT-Beginners-Guide-to-IPF-WEB-VERSION-Nov-2023.pdf",
        "procurement_guidance_pdf",
    ),
    PdfSource(
        "procurement_vendor",
        "worldbank_procurement_directive_2024",
        "Procurement Directive in IPF and Other Operational Procurement Matters, April 2024",
        "World Bank",
        "https://thedocs.worldbank.org/en/doc/2d4449540430ed046e159dc5d1ff45ed-0290012024/original/Procurement-Directive-April-2024.pdf",
        "procurement_directive_pdf",
    ),
    PdfSource(
        "procurement_vendor",
        "worldbank_evaluating_bids_rated_criteria_2025",
        "Evaluating Bids and Proposals with Rated Criteria, February 2025",
        "World Bank",
        "https://thedocs.worldbank.org/en/doc/9dcb7971706bf29b2732779c39922b77-0290012025/original/Evaluating-Bids-and-Proposals-with-Rated-Criteria-Feb-4-2025.pdf",
        "vendor_evaluation_guidance_pdf",
    ),
    PdfSource(
        "procurement_vendor",
        "federal_acquisition_regulation_current",
        "Federal Acquisition Regulation PDF",
        "Acquisition.gov",
        "https://www.acquisition.gov/sites/default/files/current/far/pdf/FAR.pdf",
        "federal_procurement_regulation_pdf",
    ),
    # Travel and expense.
    PdfSource(
        "travel_expense",
        "irs_publication_463_2025",
        "IRS Publication 463: Travel, Gift, and Car Expenses",
        "Internal Revenue Service",
        "https://www.irs.gov/pub/irs-pdf/p463.pdf",
        "tax_travel_expense_pdf",
    ),
    PdfSource(
        "travel_expense",
        "dod_joint_travel_regulations",
        "Joint Travel Regulations",
        "Defense Travel Management Office",
        "https://media.defense.gov/2022/Jan/04/2002917147/-1/-1/0/JTR.PDF",
        "travel_policy_pdf",
    ),
    PdfSource(
        "travel_expense",
        "fema_federal_travel_regulation",
        "Federal Travel Regulation PDF",
        "FEMA / GSA",
        "https://www.fema.gov/pdf/about/regions/regionx/federal_travel_regulation.pdf",
        "travel_policy_pdf",
    ),
    PdfSource(
        "travel_expense",
        "gsa_foreign_travel_policy_2025",
        "GSA Foreign Travel Policy",
        "General Services Administration",
        "https://www.gsa.gov/directives/files?file=2025-09%2FOAS+5775.1A+Foreign+Travel+Policy+%283%29_0.pdf",
        "travel_policy_pdf",
    ),
    # Competitive research.
    PdfSource(
        "competitive_research",
        "annual_report_aapl_2024",
        "Apple 2024 Annual Report",
        "AnnualReports.com",
        "https://www.annualreports.com/HostedData/AnnualReportArchive/a/NASDAQ_AAPL_2024.pdf",
        "competitor_annual_report_pdf",
    ),
    PdfSource(
        "competitive_research",
        "annual_report_msft_2024",
        "Microsoft 2024 Annual Report",
        "AnnualReports.com",
        "https://www.annualreports.com/HostedData/AnnualReportArchive/m/NASDAQ_MSFT_2024.pdf",
        "competitor_annual_report_pdf",
    ),
    PdfSource(
        "competitive_research",
        "annual_report_amzn_2024",
        "Amazon 2024 Annual Report",
        "AnnualReports.com",
        "https://www.annualreports.com/HostedData/AnnualReportArchive/a/NASDAQ_AMZN_2024.pdf",
        "competitor_annual_report_pdf",
    ),
    PdfSource(
        "competitive_research",
        "annual_report_nvda_2024",
        "NVIDIA 2024 Annual Report",
        "AnnualReports.com",
        "https://www.annualreports.com/HostedData/AnnualReportArchive/n/NASDAQ_NVDA_2024.pdf",
        "competitor_annual_report_pdf",
    ),
    PdfSource(
        "competitive_research",
        "annual_report_googl_2024",
        "Alphabet 2024 Annual Report",
        "AnnualReports.com",
        "https://www.annualreports.com/HostedData/AnnualReportArchive/g/NASDAQ_GOOGL_2024.pdf",
        "competitor_annual_report_pdf",
    ),
    # Incident postmortem / incident response drafting.
    PdfSource(
        "incident_postmortem",
        "nist_sp_800_61r3",
        "NIST SP 800-61 Rev. 3 Incident Response Recommendations",
        "NIST",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r3.pdf",
        "incident_response_guidance_pdf",
    ),
    PdfSource(
        "incident_postmortem",
        "nist_sp_800_61r2",
        "NIST SP 800-61 Rev. 2 Computer Security Incident Handling Guide",
        "NIST",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r2.pdf",
        "incident_response_guidance_pdf",
    ),
    PdfSource(
        "incident_postmortem",
        "nist_sp_800_184",
        "NIST SP 800-184 Guide for Cybersecurity Event Recovery",
        "NIST",
        "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-184.pdf",
        "incident_recovery_guidance_pdf",
    ),
    PdfSource(
        "incident_postmortem",
        "nist_sp_800_86",
        "NIST SP 800-86 Guide to Integrating Forensic Techniques into Incident Response",
        "NIST",
        "https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-86.pdf",
        "incident_forensics_guidance_pdf",
    ),
    # HR policy navigator.
    PdfSource(
        "hr_policy",
        "dol_fmla_employee_guide",
        "Employee Guide to the Family and Medical Leave Act",
        "U.S. Department of Labor",
        "https://www.dol.gov/sites/dolgov/files/WHD/legacy/files/employeeguide.pdf",
        "hr_leave_policy_pdf",
    ),
    PdfSource(
        "hr_policy",
        "dol_fmla_employer_guide",
        "Employer's Guide to the Family and Medical Leave Act",
        "U.S. Department of Labor",
        "https://www.dol.gov/sites/dolgov/files/WHD/legacy/files/employerguide.pdf",
        "hr_leave_policy_pdf",
    ),
    PdfSource(
        "hr_policy",
        "opm_telework_remote_work_guide_2021",
        "Guide to Telework and Remote Work in the Federal Government",
        "Office of Personnel Management",
        "https://www.opm.gov/telework/documents-for-telework/2021-guide-to-telework-and-remote-work.pdf",
        "hr_telework_policy_pdf",
    ),
    PdfSource(
        "hr_policy",
        "dol_flsa_worker_rights_poster",
        "FLSA Notice of Workers' Rights",
        "U.S. Department of Labor",
        "https://www.dol.gov/sites/dolgov/files/WHD/legacy/files/FLSA-MSPA-H2A-091925-Public.pdf",
        "hr_wage_hour_policy_pdf",
    ),
    PdfSource(
        "hr_policy",
        "osha_job_safety_health_poster",
        "Job Safety and Health: It's the Law Poster",
        "Occupational Safety and Health Administration",
        "https://www.osha.gov/sites/default/files/publications/osha3165.pdf",
        "hr_safety_policy_pdf",
    ),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in value).strip("._")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def download_pdf(url: str, user_agent: str, timeout: int) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept": "application/pdf,*/*"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 404, 410}:
            return None
        raise
    except (TimeoutError, socket.timeout, urllib.error.URLError):
        return None
    if not payload.startswith(b"%PDF"):
        return None
    return payload


def write_manifest_row(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def existing_ids(manifest_path: Path) -> set[str]:
    ids: set[str] = set()
    if not manifest_path.exists():
        return ids
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                ids.add(str(json.loads(line).get("document_id")))
    return ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--domains", nargs="+", default=["all"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--user-agent", default="capstone-domain-pdf/0.1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    domains = sorted({source.domain for source in PDF_SOURCES}) if "all" in args.domains else args.domains
    invalid = sorted(set(domains) - {source.domain for source in PDF_SOURCES})
    if invalid:
        raise SystemExit(f"Unknown domain(s): {', '.join(invalid)}")

    summary: dict[str, dict[str, object]] = {}
    for domain in domains:
        domain_root = args.output_dir / domain
        pdf_dir = domain_root / "pdfs"
        manifest_path = domain_root / "pdf_manifest.jsonl"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        if args.overwrite:
            for old_pdf in pdf_dir.glob("*.pdf"):
                old_pdf.unlink()
            if manifest_path.exists():
                manifest_path.unlink()

        done = existing_ids(manifest_path)
        saved = 0
        failed: list[str] = []
        for source in [item for item in PDF_SOURCES if item.domain == domain]:
            if source.document_id in done:
                continue
            print(f"FETCH {domain}: {source.document_id}", flush=True)
            payload = download_pdf(source.source_url, args.user_agent, args.timeout)
            if payload is None:
                failed.append(source.document_id)
                print(f"SKIP {domain}: {source.document_id}", flush=True)
                continue
            pdf_path = pdf_dir / safe_filename(f"{source.document_id}.pdf")
            pdf_path.write_bytes(payload)
            row = {
                "domain": domain,
                "document_id": source.document_id,
                "title": source.title,
                "source_name": source.source_name,
                "source_url": source.source_url,
                "source_type": source.source_type,
                "local_path": str(pdf_path),
                "bytes": len(payload),
                "content_sha256": sha256_bytes(payload),
                "downloaded_at": utc_now(),
            }
            write_manifest_row(manifest_path, row)
            saved += 1
            print(f"SAVED {domain}: {source.document_id}", flush=True)
            time.sleep(SLEEP_SECONDS)

        summary[domain] = {
            "saved_this_run": saved,
            "manifest": str(manifest_path),
            "pdf_dir": str(pdf_dir),
            "failed": failed,
        }

    summary_path = args.output_dir / "domain_pdf_download_summary.json"
    summary_path.write_text(json.dumps({"created_at": utc_now(), "domains": summary}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
