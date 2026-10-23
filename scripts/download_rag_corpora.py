"""Download raw RAG corpora for Track B capstone domains.

The script writes one JSON file per source document/record so later steps can
run cleaning, chunking experiments, hybrid retrieval, reranking, and citation
evaluation without re-contacting the original sources.

Examples:
    python scripts/download_rag_corpora.py --domains all --target-docs 500
    python scripts/download_rag_corpora.py --domains travel_expense hr_policy
    python scripts/download_rag_corpora.py --sec-user-agent "capstone/0.1 you@example.com"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DOMAINS = {
    "procurement_vendor": "Procurement and vendor-review assistant",
    "travel_expense": "Travel-and-expense policy agent",
    "competitive_research": "Competitive-research analyst",
    "incident_postmortem": "Incident postmortem drafter",
    "financial_report_qa": "Financial-report question-answering agent",
    "hr_policy": "HR policy navigator",
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

STATUSPAGE_BASES = [
    "https://www.githubstatus.com",
    "https://www.cloudflarestatus.com",
    "https://status.openai.com",
    "https://status.atlassian.com",
    "https://status.vercel.com",
    "https://status.datadoghq.com",
    "https://status.zoom.us",
    "https://status.anthropic.com",
    "https://status.mongodb.com",
    "https://status.linear.app",
    "https://status.box.com",
    "https://status.dropbox.com",
    "https://www.githubstatus.com",
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


def fetch_world_bank_procurement(writer: CorpusWriter, target_docs: int) -> None:
    endpoints = [
        (
            "World Bank procurement notices",
            "https://search.worldbank.org/api/v2/procnotices",
            {"format": "json", "rows": "100", "os": "0", "srt": "noticedate desc,id asc"},
        ),
        (
            "World Bank contract awards",
            "https://search.worldbank.org/api/contractdata",
            {"format": "json", "rows": "100", "os": "0", "srt": "contr_sgn_date desc,id asc"},
        ),
    ]
    for source_name, base_url, params in endpoints:
        offset = 0
        while writer.count < target_docs:
            params["os"] = str(offset)
            url = f"{base_url}?{urllib.parse.urlencode(params)}"
            data = get_json(url)
            writer.write_source_bytes(f"{source_name}_{offset}.json", json.dumps(data).encode("utf-8"))
            records = extract_world_bank_records(data)
            if not records:
                break
            for record in records:
                doc_id = str(record.get("id") or record.get("contract_id") or record.get("wbcontract_number") or "")
                if not doc_id:
                    doc_id = sha256_text(json.dumps(record, sort_keys=True))[:16]
                title = first_present(
                    record,
                    ["notice_title", "contract_desc", "procurement_notice_title", "project_name", "supplier"],
                    f"{source_name} {doc_id}",
                )
                text = record_to_text(record)
                writer.write(
                    RawDocument(
                        domain=writer.domain,
                        document_id=f"worldbank_{safe_filename(doc_id)}",
                        title=title,
                        text=text,
                        source_name=source_name,
                        source_url=url,
                        metadata={"record": record, "source_type": "api_record"},
                    )
                )
                if writer.count >= target_docs:
                    return
            offset += len(records)
            time.sleep(SLEEP_SECONDS)


def extract_world_bank_records(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for key in ("procnotices", "contracts", "data", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
            if isinstance(value, dict):
                return [x for x in value.values() if isinstance(x, dict)]
        return [x for x in data.values() if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def first_present(record: dict[str, Any], keys: list[str], fallback: str) -> str:
    for key in keys:
        value = record.get(key)
        if value:
            return str(value)
    return fallback


def record_to_text(record: dict[str, Any]) -> str:
    lines = []
    for key in sorted(record):
        value = record[key]
        if value is None or value == "":
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def fetch_ecfr_title_sections(
    writer: CorpusWriter,
    target_docs: int,
    title_number: int,
    source_label: str,
    include_path_patterns: list[str] | None = None,
) -> None:
    candidates = [
        f"https://www.ecfr.gov/api/versioner/v1/full/current/title-{title_number}.xml",
        f"https://www.govinfo.gov/bulkdata/ECFR/title-{title_number}/ECFR-title{title_number}.xml",
    ]
    last_error = None
    xml_payload = None
    source_url = candidates[0]
    for url in candidates:
        try:
            xml_payload = http_get(url, timeout=120)
            source_url = url
            break
        except Exception as exc:  # noqa: BLE001 - keep fallback robust for source APIs.
            last_error = exc
    if xml_payload is None:
        raise RuntimeError(f"Could not download eCFR title {title_number}: {last_error}")

    writer.write_source_bytes(f"ecfr_title_{title_number}.xml", xml_payload)
    root = ET.fromstring(xml_payload)
    for section in root.iter():
        if not is_ecfr_section(section):
            continue
        heading_path = section_heading_path(section)
        if include_path_patterns and not any(
            re.search(pattern, heading_path, flags=re.IGNORECASE)
            for pattern in include_path_patterns
        ):
            continue
        section_id = section.attrib.get("N") or section.attrib.get("NODE") or sha256_text(heading_path)[:16]
        title = section.findtext("./HEAD") or section.findtext("./{*}HEAD") or heading_path
        text = ecfr_section_text(section, heading_path)
        writer.write(
            RawDocument(
                domain=writer.domain,
                document_id=f"ecfr_title{title_number}_{safe_filename(section_id)}",
                title=normalize_text(title),
                text=text,
                source_name=source_label,
                source_url=source_url,
                metadata={
                    "title_number": title_number,
                    "section": section_id,
                    "heading_path": heading_path,
                    "source_type": "regulatory_section",
                },
            )
        )
        if writer.count >= target_docs:
            return


def strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def is_ecfr_section(element: ET.Element) -> bool:
    tag = strip_ns(element.tag)
    return tag == "SECTION" or element.attrib.get("TYPE", "").upper() == "SECTION"


def section_heading_path(section: ET.Element) -> str:
    headings = []
    node = section
    while node is not None:
        head = node.findtext("./HEAD") or node.findtext("./{*}HEAD")
        if head:
            headings.append(normalize_text(head))
        node = None  # ElementTree does not expose parents; section heading still gives a citation anchor.
    return " / ".join(reversed(headings))


def ecfr_section_text(section: ET.Element, heading_path: str) -> str:
    chunks = [heading_path]
    for element in section.iter():
        tag = strip_ns(element.tag)
        if tag in {"HEAD", "P", "FP", "I", "E"} and element.text:
            chunks.append(element.text)
    return normalize_text("\n".join(chunks))


def fetch_competitive_sec(writer: CorpusWriter, target_docs: int, sec_user_agent: str) -> None:
    headers = {"User-Agent": sec_user_agent}
    tickers_data = get_json("https://www.sec.gov/files/company_tickers.json", headers=headers)
    writer.write_source_bytes("sec_company_tickers.json", json.dumps(tickers_data).encode("utf-8"))
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
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        try:
            submissions = get_json(url, headers=headers)
        except Exception:
            continue
        company_name = submissions.get("name") or item.get("title") or ticker
        recent = submissions.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])
        for idx, form in enumerate(forms):
            if form not in {"10-K", "10-Q", "8-K", "DEF 14A"}:
                continue
            accession = safe_list_get(accessions, idx, "")
            accession_no_dash = accession.replace("-", "")
            primary_doc = safe_list_get(primary_docs, idx, "")
            filing_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(item['cik_str'])}/"
                f"{accession_no_dash}/{primary_doc}"
            )
            title = f"{company_name} {form} filed {safe_list_get(filing_dates, idx, '')}"
            text = "\n".join(
                [
                    f"Company: {company_name}",
                    f"Ticker: {ticker}",
                    f"CIK: {cik}",
                    f"SIC: {submissions.get('sic')} {submissions.get('sicDescription')}",
                    f"Business address: {submissions.get('addresses', {}).get('business')}",
                    f"Form: {form}",
                    f"Filing date: {safe_list_get(filing_dates, idx, '')}",
                    f"Accession number: {accession}",
                    f"Primary document: {primary_doc}",
                    f"Primary document description: {safe_list_get(descriptions, idx, '')}",
                    f"Filing URL: {filing_url}",
                ]
            )
            writer.write(
                RawDocument(
                    domain=writer.domain,
                    document_id=f"sec_competitive_{ticker}_{safe_filename(accession)}",
                    title=title,
                    text=text,
                    source_name="SEC EDGAR submissions",
                    source_url=url,
                    metadata={
                        "ticker": ticker,
                        "cik": cik,
                        "form": form,
                        "filing_url": filing_url,
                        "source_type": "sec_filing_metadata",
                    },
                )
            )
            if writer.count >= target_docs:
                return
        time.sleep(SLEEP_SECONDS)


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


def fetch_incidents(writer: CorpusWriter, target_docs: int) -> None:
    for base_url in STATUSPAGE_BASES:
        url = f"{base_url.rstrip('/')}/api/v2/incidents.json"
        try:
            data = get_json(url)
        except Exception:
            continue
        incidents = data.get("incidents", []) if isinstance(data, dict) else []
        page_name = data.get("page", {}).get("name", base_url) if isinstance(data, dict) else base_url
        writer.write_source_bytes(f"statuspage_{safe_filename(page_name)}.json", json.dumps(data).encode("utf-8"))
        for incident in incidents:
            updates = incident.get("incident_updates", [])
            update_text = "\n".join(
                f"- {u.get('created_at')}: {u.get('status')} - {u.get('body')}"
                for u in updates
            )
            text = "\n".join(
                [
                    f"Service: {page_name}",
                    f"Incident: {incident.get('name')}",
                    f"Status: {incident.get('status')}",
                    f"Impact: {incident.get('impact')}",
                    f"Started: {incident.get('started_at')}",
                    f"Resolved: {incident.get('resolved_at')}",
                    "Updates:",
                    update_text,
                ]
            )
            writer.write(
                RawDocument(
                    domain=writer.domain,
                    document_id=f"statuspage_{safe_filename(page_name)}_{incident.get('id')}",
                    title=f"{page_name} - {incident.get('name')}",
                    text=text,
                    source_name="Atlassian Statuspage public incident API",
                    source_url=url,
                    metadata={"incident": incident, "source_type": "status_incident"},
                )
            )
            if writer.count >= target_docs:
                return
        time.sleep(SLEEP_SECONDS)

        maintenance_url = f"{base_url.rstrip('/')}/api/v2/scheduled-maintenances.json"
        try:
            maintenance_data = get_json(maintenance_url)
        except Exception:
            continue
        maintenances = (
            maintenance_data.get("scheduled_maintenances", [])
            if isinstance(maintenance_data, dict)
            else []
        )
        writer.write_source_bytes(
            f"statuspage_maintenance_{safe_filename(page_name)}.json",
            json.dumps(maintenance_data).encode("utf-8"),
        )
        for maintenance in maintenances:
            updates = maintenance.get("incident_updates", [])
            update_text = "\n".join(
                f"- {u.get('created_at')}: {u.get('status')} - {u.get('body')}"
                for u in updates
            )
            text = "\n".join(
                [
                    f"Service: {page_name}",
                    f"Maintenance: {maintenance.get('name')}",
                    f"Status: {maintenance.get('status')}",
                    f"Impact: {maintenance.get('impact')}",
                    f"Scheduled for: {maintenance.get('scheduled_for')}",
                    f"Scheduled until: {maintenance.get('scheduled_until')}",
                    "Updates:",
                    update_text,
                ]
            )
            writer.write(
                RawDocument(
                    domain=writer.domain,
                    document_id=f"statuspage_maintenance_{safe_filename(page_name)}_{maintenance.get('id')}",
                    title=f"{page_name} - {maintenance.get('name')}",
                    text=text,
                    source_name="Atlassian Statuspage public maintenance API",
                    source_url=maintenance_url,
                    metadata={"incident": maintenance, "source_type": "status_maintenance"},
                )
            )
            if writer.count >= target_docs:
                return
        time.sleep(SLEEP_SECONDS)

    fetch_ntsb_latest_update(writer, target_docs)


def fetch_ntsb_latest_update(writer: CorpusWriter, target_docs: int) -> None:
    if writer.count >= target_docs:
        return
    directory_url = "https://data.ntsb.gov/avdata/filedirectory"
    html_payload = http_get(directory_url)
    writer.write_source_bytes("ntsb_file_directory.html", html_payload)
    names = re.findall(rb"up\d{2}[A-Z]{3}\.zip", html_payload)
    for raw_name in sorted({name.decode("ascii") for name in names}, reverse=True):
        if writer.count >= target_docs:
            return
        zip_url = f"https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C%3A%5Cavdata%5C{raw_name}"
        try:
            payload = http_get(zip_url, timeout=180)
        except Exception:
            continue
        writer.write_source_bytes(f"ntsb_{raw_name}", payload)
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                parse_ntsb_zip(writer, archive, zip_url, target_docs)
        except zipfile.BadZipFile:
            continue


def parse_ntsb_zip(
    writer: CorpusWriter,
    archive: zipfile.ZipFile,
    source_url: str,
    target_docs: int,
) -> None:
    for name in archive.namelist():
        if writer.count >= target_docs:
            return
        if not name.lower().endswith((".csv", ".txt")):
            continue
        with archive.open(name) as handle:
            raw = handle.read()
        text_stream = io.StringIO(raw.decode("utf-8-sig", errors="replace"))
        sample = text_stream.read(4096)
        text_stream.seek(0)
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        reader = csv.DictReader(text_stream, delimiter=delimiter)
        for row in reader:
            if writer.count >= target_docs:
                return
            event_id = (
                row.get("ev_id")
                or row.get("EventId")
                or row.get("NtsbNo")
                or row.get("ntsb_no")
                or sha256_text(json.dumps(row, sort_keys=True))[:16]
            )
            title = first_present(row, ["ntsb_no", "NtsbNo", "ev_id", "EventId"], f"NTSB event {event_id}")
            writer.write(
                RawDocument(
                    domain=writer.domain,
                    document_id=f"ntsb_{safe_filename(event_id)}_{sha256_text(str(row))[:8]}",
                    title=title,
                    text=record_to_text(row),
                    source_name="NTSB downloadable aviation dataset",
                    source_url=source_url,
                    metadata={"source_file": name, "source_type": "incident_record"},
                )
            )


def fetch_domain(domain: str, writer: CorpusWriter, target_docs: int, sec_user_agent: str) -> None:
    if domain == "procurement_vendor":
        fetch_world_bank_procurement(writer, target_docs)
        if writer.count < target_docs:
            fetch_ecfr_title_sections(
                writer,
                target_docs,
                title_number=48,
                source_label="eCFR Title 48 Federal Acquisition Regulation",
            )
    elif domain == "travel_expense":
        fetch_ecfr_title_sections(
            writer,
            target_docs,
            title_number=41,
            source_label="eCFR Title 41 Public Contracts and Property Management",
            include_path_patterns=[r"300", r"301", r"302", r"303", r"304", r"travel"],
        )
    elif domain == "competitive_research":
        fetch_competitive_sec(writer, target_docs, sec_user_agent)
    elif domain == "incident_postmortem":
        fetch_incidents(writer, target_docs)
    elif domain == "financial_report_qa":
        fetch_financial_sec_facts(writer, target_docs, sec_user_agent)
    elif domain == "hr_policy":
        fetch_ecfr_title_sections(
            writer,
            target_docs,
            title_number=29,
            source_label="eCFR Title 29 Labor",
        )
    else:
        raise ValueError(f"Unsupported domain: {domain}")


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
