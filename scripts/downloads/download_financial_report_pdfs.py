"""Download public annual report PDFs for the financial-report QA corpus.

PDFs are stored separately from the JSON companyfacts corpus:

    data/raw/financial_report_qa/pdfs/*.pdf
    data/raw/financial_report_qa/pdf_manifest.jsonl

The URLs are public AnnualReports.com PDF locations. The script tries the
archive path first, then a current-reports path for compatibility.
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
from typing import Iterable


DEFAULT_OUTPUT_DIR = Path("data/raw/financial_report_qa")
DEFAULT_TARGET_PDFS = 100
DEFAULT_YEARS = list(range(2024, 2014, -1))
SLEEP_SECONDS = 0.4


@dataclass(frozen=True)
class Company:
    exchange: str
    ticker: str
    name: str


COMPANIES = [
    Company("NASDAQ", "AAPL", "Apple Inc."),
    Company("NASDAQ", "MSFT", "Microsoft Corporation"),
    Company("NASDAQ", "NVDA", "NVIDIA Corporation"),
    Company("NASDAQ", "AMZN", "Amazon.com, Inc."),
    Company("NASDAQ", "GOOGL", "Alphabet Inc."),
    Company("NASDAQ", "META", "Meta Platforms, Inc."),
    Company("NASDAQ", "TSLA", "Tesla, Inc."),
    Company("NASDAQ", "AVGO", "Broadcom Inc."),
    Company("NYSE", "JPM", "JPMorgan Chase & Co."),
    Company("NYSE", "WMT", "Walmart Inc."),
    Company("NYSE", "V", "Visa Inc."),
    Company("NYSE", "XOM", "Exxon Mobil Corporation"),
    Company("NYSE", "UNH", "UnitedHealth Group Incorporated"),
    Company("NYSE", "MA", "Mastercard Incorporated"),
    Company("NYSE", "ORCL", "Oracle Corporation"),
    Company("NASDAQ", "COST", "Costco Wholesale Corporation"),
    Company("NYSE", "HD", "The Home Depot, Inc."),
    Company("NYSE", "PG", "The Procter & Gamble Company"),
    Company("NASDAQ", "NFLX", "Netflix, Inc."),
    Company("NYSE", "JNJ", "Johnson & Johnson"),
    Company("NYSE", "ABBV", "AbbVie Inc."),
    Company("NYSE", "BAC", "Bank of America Corporation"),
    Company("NYSE", "CRM", "Salesforce, Inc."),
    Company("NYSE", "KO", "The Coca-Cola Company"),
    Company("NASDAQ", "AMD", "Advanced Micro Devices, Inc."),
    Company("NYSE", "CVX", "Chevron Corporation"),
    Company("NASDAQ", "PEP", "PepsiCo, Inc."),
    Company("NYSE", "TMO", "Thermo Fisher Scientific Inc."),
    Company("NASDAQ", "ADBE", "Adobe Inc."),
    Company("NYSE", "WFC", "Wells Fargo & Company"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._=-" else "_" for ch in value)


def annual_report_urls(company: Company, year: int) -> Iterable[str]:
    exchange = company.exchange.upper()
    ticker = company.ticker.upper()
    first_letter = ticker[0].lower()
    yield (
        "https://www.annualreports.com/HostedData/"
        f"AnnualReportArchive/{first_letter}/{exchange}_{ticker}_{year}.pdf"
    )
    yield (
        "https://www.annualreports.com/HostedData/"
        f"AnnualReports/PDF/{exchange}_{ticker}_{year}.pdf"
    )
    yield (
        "https://www.annualreports.com/hosteddata/"
        f"annualreports/pdf/{exchange.lower()}_{ticker.lower()}_{year}.pdf"
    )


def download_pdf(url: str, user_agent: str, timeout: int) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code in {403, 404, 410}:
            return None
        if exc.code == 429:
            time.sleep(max(5, timeout))
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


def existing_keys(manifest_path: Path) -> set[tuple[str, int]]:
    keys = set()
    if not manifest_path.exists():
        return keys
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keys.add((str(row.get("ticker")), int(row.get("year"))))
    return keys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-pdfs", type=int, default=DEFAULT_TARGET_PDFS)
    parser.add_argument("--years", nargs="*", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--tickers", nargs="*", help="Optional ticker filter, e.g. AAPL MSFT NVDA")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--user-agent", default="capstone-rag-pdf/0.1 ramakris99@gmail.com")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pdf_dir = args.output_dir / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "pdf_manifest.jsonl"

    if args.overwrite:
        for old_pdf in pdf_dir.glob("*.pdf"):
            old_pdf.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    wanted_tickers = {ticker.upper() for ticker in args.tickers or []}
    companies = [
        company for company in COMPANIES
        if not wanted_tickers or company.ticker.upper() in wanted_tickers
    ]
    done_keys = existing_keys(manifest_path)
    saved = 0

    for company in companies:
        for year in args.years:
            if saved >= args.target_pdfs:
                print(f"Saved {saved} PDFs under {pdf_dir}")
                return 0
            key = (company.ticker, year)
            if key in done_keys:
                continue
            for url in annual_report_urls(company, year):
                payload = download_pdf(url, args.user_agent, args.timeout)
                if payload is None:
                    continue
                file_name = safe_filename(f"{company.exchange}_{company.ticker}_{year}.pdf")
                pdf_path = pdf_dir / file_name
                pdf_path.write_bytes(payload)
                row = {
                    "domain": "financial_report_qa",
                    "document_id": f"annual_report_{company.exchange}_{company.ticker}_{year}",
                    "company": company.name,
                    "ticker": company.ticker,
                    "exchange": company.exchange,
                    "year": year,
                    "title": f"{company.name} annual report {year}",
                    "source_name": "AnnualReports.com",
                    "source_url": url,
                    "local_path": str(pdf_path),
                    "bytes": len(payload),
                    "content_sha256": sha256_bytes(payload),
                    "downloaded_at": utc_now(),
                    "source_type": "annual_report_pdf",
                }
                write_manifest_row(manifest_path, row)
                saved += 1
                done_keys.add(key)
                print(f"Saved {saved:03d}: {company.ticker} {year}")
                break
            time.sleep(SLEEP_SECONDS)

    print(f"Saved {saved} PDFs under {pdf_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
