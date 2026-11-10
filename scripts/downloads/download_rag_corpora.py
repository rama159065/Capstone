"""Download the raw SEC XBRL company-facts corpus for the financial report copilot.

The script writes one JSON file per source document/record so later steps can
run cleaning, chunking experiments, hybrid retrieval, reranking, and citation
evaluation without re-contacting the original sources.

Examples:
    python scripts/downloads/download_rag_corpora.py --domains all --target-docs 500
    python scripts/downloads/download_rag_corpora.py --sec-user-agent "capstone/0.1 you@example.com"
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DOMAINS = {
    "financial_report_qa": "Financial-report question-answering agent",
}

DEFAULT_OUTPUT_DIR = Path("data/raw")
DEFAULT_TARGET_DOCS = 500
SLEEP_SECONDS = 0.25

SEC_DEFAULT_USER_AGENT = (
    "capstone-rag-corpus/0.1 "
    "(set --sec-user-agent or SEC_USER_AGENT with your email)"
)

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "BRK-B",
    "LLY",
    "TSLA",
    "AVGO",
    "JPM",
    "WMT",
    "V",
    "XOM",
    "UNH",
    "MA",
    "ORCL",
    "COST",
    "HD",
    "PG",
    "NFLX",
    "JNJ",
    "ABBV",
    "BAC",
    "CRM",
    "KO",
    "AMD",
    "CVX",
    "PEP",
    "TMO",
    "ADBE",
    "WFC",
    "CSCO",
    "MCD",
    "IBM",
    "GE",
    "ABT",
    "QCOM",
    "PM",
    "TXN",
    "INTU",
    "AMAT",
    "CAT",
    "NOW",
    "DIS",
    "ISRG",
    "VZ",
    "UBER",
    "PFE",
    "SPGI",
]

@dataclass(frozen=True)
class RawDocument:
    domain: str
    document_id: str
    title: str
    text: str
    source_name: str
    source_url: str
    metadata: dict[str, Any]


class CorpusWriter:
    def __init__(self, output_dir: Path, domain: str, overwrite: bool = False) -> None:
        self.domain = domain
        self.root = output_dir / domain
        self.docs_dir = self.root / "documents"
        self.sources_dir = self.root / "source_files"
        self.overwrite = overwrite
        self.count = 0
        self.seen_ids: set[str] = set()
        self.root.mkdir(parents=True, exist_ok=True)
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for old in self.docs_dir.glob("*.json"):
                old.unlink()
            manifest = self.root / "manifest.jsonl"
            if manifest.exists():
                manifest.unlink()

    def write_source_bytes(self, name: str, payload: bytes) -> Path:
        path = self.sources_dir / safe_filename(name)
        path.write_bytes(payload)
        return path

    def write(self, doc: RawDocument) -> bool:
        if not doc.text.strip():
            return False
        if doc.document_id in self.seen_ids:
            return False
        self.seen_ids.add(doc.document_id)
        payload = {
            "domain": doc.domain,
            "document_id": doc.document_id,
            "title": doc.title,
            "text": normalize_text(doc.text),
            "source_name": doc.source_name,
            "source_url": doc.source_url,
            "metadata": doc.metadata,
            "downloaded_at": utc_now(),
            "content_sha256": sha256_text(doc.text),
        }
        path = self.docs_dir / f"{safe_filename(doc.document_id)}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        with (self.root / "manifest.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload | {"path": str(path)}, ensure_ascii=False) + "\n")
        self.count += 1
        return True


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_filename(value: str, max_len: int = 140) -> str:
    value = re.sub(r"[^A-Za-z0-9._=-]+", "_", value).strip("._")
    return (value or "doc")[:max_len]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()


def normalize_text(value: str) -> str:
    value = html.unescape(value)
    value = value.replace("\xa0", " ")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = "\n".join(line.strip() for line in value.splitlines())
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def http_get(url: str, headers: dict[str, str] | None = None, timeout: int = 60) -> bytes:
    merged_headers = {
        "Accept": "application/json,text/xml,application/xml,text/html,*/*",
        "User-Agent": "capstone-rag-corpus/0.1",
    }
    if headers:
        merged_headers.update(headers)
    request = urllib.request.Request(url, headers=merged_headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def get_json(url: str, headers: dict[str, str] | None = None) -> Any:
    payload = http_get(url, headers=headers)
    return json.loads(payload.decode("utf-8-sig"))


def write_run_summary(output_dir: Path, summaries: dict[str, int], errors: list[str]) -> None:
    discovered = discover_domain_counts(output_dir)
    merged_summaries = discovered | summaries
    payload = {
        "created_at": utc_now(),
        "domains": merged_summaries,
        "errors": errors,
        "layout": {
            "documents": "data/raw/<domain>/documents/*.json",
            "manifest": "data/raw/<domain>/manifest.jsonl",
            "source_files": "data/raw/<domain>/source_files/",
        },
    }
    (output_dir / "download_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def discover_domain_counts(output_dir: Path) -> dict[str, int]:
    counts = {}
    for domain in DOMAINS:
        docs_dir = output_dir / domain / "documents"
        if docs_dir.exists():
            counts[domain] = len(list(docs_dir.glob("*.json")))
    return counts


def safe_list_get(values: list[Any], index: int, default: Any) -> Any:
    return values[index] if index < len(values) else default


def fetch_financial_sec_facts(writer: CorpusWriter, target_docs: int, sec_user_agent: str) -> None:
    headers = {"User-Agent": sec_user_agent}
    tickers_data = get_json("https://www.sec.gov/files/company_tickers.json", headers=headers)
    ticker_map = {
        str(item["ticker"]).upper(): item
        for item in tickers_data.values()
        if isinstance(item, dict) and "ticker" in item
    }
    for ticker in TICKERS:
        item = ticker_map.get(ticker)
        if not item:
            continue
        cik = f"{int(item['cik_str']):010d}"
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        try:
            facts = get_json(url, headers=headers)
        except Exception:
            continue
        entity_name = facts.get("entityName") or item.get("title") or ticker
        for taxonomy, concepts in facts.get("facts", {}).items():
            if not isinstance(concepts, dict):
                continue
            for concept_name, concept in concepts.items():
                units = concept.get("units", {})
                rows = []
                for unit_name, filings in units.items():
                    for filing in filings[-12:]:
                        rows.append(
                            f"{unit_name} {filing.get('fy')} {filing.get('fp')} "
                            f"{filing.get('form')} filed {filing.get('filed')}: {filing.get('val')}"
                        )
                if not rows:
                    continue
                label = concept.get("label") or concept_name
                text = "\n".join(
                    [
                        f"Company: {entity_name}",
                        f"Ticker: {ticker}",
                        f"CIK: {cik}",
                        f"Taxonomy: {taxonomy}",
                        f"Concept: {concept_name}",
                        f"Label: {label}",
                        f"Description: {concept.get('description', '')}",
                        "Recent reported values:",
                        *rows,
                    ]
                )
                writer.write(
                    RawDocument(
                        domain=writer.domain,
                        document_id=f"sec_facts_{ticker}_{taxonomy}_{safe_filename(concept_name)}",
                        title=f"{entity_name} - {label}",
                        text=text,
                        source_name="SEC EDGAR companyfacts",
                        source_url=url,
                        metadata={
                            "ticker": ticker,
                            "cik": cik,
                            "taxonomy": taxonomy,
                            "concept": concept_name,
                            "source_type": "sec_xbrl_company_fact",
                        },
                    )
                )
                if writer.count >= target_docs:
                    return
        time.sleep(SLEEP_SECONDS)


def fetch_domain(domain: str, writer: CorpusWriter, target_docs: int, sec_user_agent: str) -> None:
    if domain != "financial_report_qa":
        raise ValueError(f"Unsupported domain: {domain}")
    fetch_financial_sec_facts(writer, target_docs, sec_user_agent)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["all"],
        help=f"Domains to download, or all. Choices: {', '.join(DOMAINS)}",
    )
    parser.add_argument("--target-docs", type=int, default=DEFAULT_TARGET_DOCS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sec-user-agent",
        default=os.environ.get("SEC_USER_AGENT", SEC_DEFAULT_USER_AGENT),
        help="SEC asks automated clients to send a descriptive User-Agent with contact email.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    domains = list(DOMAINS) if "all" in args.domains else args.domains
    invalid = sorted(set(domains) - set(DOMAINS))
    if invalid:
        raise SystemExit(f"Unknown domain(s): {', '.join(invalid)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, int] = {}
    errors: list[str] = []
    for domain in domains:
        print(f"\n==> Downloading {domain}: {DOMAINS[domain]}")
        writer = CorpusWriter(args.output_dir, domain, overwrite=args.overwrite)
        try:
            fetch_domain(domain, writer, args.target_docs, args.sec_user_agent)
        except Exception as exc:  # noqa: BLE001 - record and continue other domains.
            errors.append(f"{domain}: {exc}")
            print(f"ERROR {domain}: {exc}")
        summaries[domain] = writer.count
        print(f"Saved {writer.count} documents under {writer.root}")
    write_run_summary(args.output_dir, summaries, errors)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
