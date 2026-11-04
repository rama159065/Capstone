"""Download everything the financial copilot needs for the filings in the ARS manifest.

For every row of data/raw/financial_report_qa/edgar/pdf_manifest.jsonl (100 ARS
PDFs) this fetches, from SEC EDGAR:

* the ARS PDF itself (skipped if present; SHA-256 checked against the manifest);
* the matching **10-K** primary document (HTML with inline XBRL): the latest
  10-K for the same fiscal year filed on or before the ARS. If the primary
  document carries few tagged numbers, its sibling iXBRL documents in the same
  filing (where the statements are) are downloaded too;
* the **proxy statement** (DEF 14A) filed after that 10-K (governance, executive
  pay, pay-versus-performance XBRL);
* per company, once: **companyfacts** (all XBRL facts, JSON) and **submissions**
  (filing index, JSON).

It writes ``corpus_manifest.jsonl`` with one row per filing that links the ARS,
10-K and proxy through one ``filing_id`` (design section 3.3). It is resumable:
files already on disk are kept, so a rerun after a Colab disconnect only fetches
what is missing. SEC allows 10 requests/second; requests are spaced 0.15 s
apart and retried only on HTTP 429/503.

    python -m scripts.downloads.download_edgar_corpus --user-agent "capstone/0.1 you@example.com"
    # Colab: keep the corpus on Drive
    python -m scripts.downloads.download_edgar_corpus --output-dir /content/drive/MyDrive/capstone_rag/edgar
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error

from scripts.downloads import download_edgar_ars_pdfs as ars

from scripts.common.paths import ROOT  # noqa: E402 - repo root
MANIFEST = ROOT / "data/raw/financial_report_qa/edgar/pdf_manifest.jsonl"
ARTIFACTS = ("pdf", "10k", "proxy", "companyfacts", "submissions")
TENK_FORMS = {"10-K", "10-K405", "10-KT"}
PROXY_FORMS = {"DEF 14A"}
MIN_PRIMARY_FACTS = 500    # below this the 10-K is split over several iXBRL documents (e.g. Wells Fargo)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").split("\n") if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def archive_url(cik, accession, name):
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{name}"


def fetch(url, path, user_agent, timeout):
    """Download url to path atomically unless it already exists; return (path, downloaded?)."""
    if path.exists() and path.stat().st_size:
        return path, False
    payload = ars.http_get(url, user_agent, timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(payload)
    tmp.replace(path)
    return path, True


def submissions(cik, out_dir, user_agent, timeout):
    """The company's submissions JSON (saved) plus every filing row since 2023 from all pages."""
    path, _ = fetch(f"https://data.sec.gov/submissions/CIK{cik}.json", out_dir / "submissions" / f"CIK{cik}.json",
                    user_agent, timeout)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = ars.filing_rows(data["filings"]["recent"])
    for extra in data["filings"].get("files", []):
        if extra.get("filingTo", "") < ars.EARLIEST_FILING_DATE:
            continue
        page, _ = fetch(f"https://data.sec.gov/submissions/{extra['name']}", out_dir / "submissions" / extra["name"],
                        user_agent, timeout)
        rows += ars.filing_rows(json.loads(page.read_text(encoding="utf-8")))
    return rows


def match_tenk(row, filings):
    """Latest original 10-K whose period ends in the row's fiscal year, filed on or before the ARS."""
    year, filed = str(row["year"]), row["filing_date"]
    same_year = [f for f in filings if f["form"] in TENK_FORMS and (f["reportDate"] or "")[:4] == year]
    before = [f for f in same_year if f["filingDate"] <= filed]
    pool = before or same_year
    return max(pool, key=lambda f: f["filingDate"]) if pool else None


def match_proxy(tenk, filings, max_days=400):
    """First DEF 14A filed from a week before the 10-K up to max_days after it."""
    start = date.fromisoformat(tenk["filingDate"])
    pool = [f for f in filings if f["form"] in PROXY_FORMS
            and -7 <= (date.fromisoformat(f["filingDate"]) - start).days <= max_days]
    return min(pool, key=lambda f: f["filingDate"]) if pool else None


def html_stats(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
            "ixbrl_numeric_facts": raw.count(b"<ix:nonFraction"), "ixbrl_text_facts": raw.count(b"<ix:nonNumeric")}


def companion_documents(cik, accession, primary, user_agent, timeout):
    """Other iXBRL documents of the same 10-K (same name stem, not R*.htm viewer pages or exhibits)."""
    index = json.loads(ars.http_get(archive_url(cik, accession, "index.json"), user_agent, timeout))
    stem = Path(primary).stem.split("_")[0]
    return [item["name"] for item in index["directory"]["item"]
            if item["name"].lower().endswith(".htm") and item["name"] != primary
            and item["name"].startswith(stem) and "ex" not in item["name"][len(stem):].lower()]


def process(row, args, cache):
    cik, ticker, year = row["cik"], row["ticker"], int(row["year"])
    out, ua, timeout = args.output_dir, args.user_agent, args.timeout
    record = {"filing_id": f"{ticker}_FY{year}", "ticker": ticker, "cik": cik, "fiscal_year": year,
              "company": row.get("company"), "ars_document_id": row["document_id"],
              "ars_accession": row.get("accession_number"), "ars_url": row.get("source_url"),
              "ars_filing_date": row.get("filing_date"), "errors": {}}
    downloads = 0

    if "pdf" in args.only:
        try:
            pdf, new = fetch(row["source_url"], out / "pdfs" / Path(row["local_path"].replace("\\", "/")).name, ua, timeout)
            downloads += new
            digest = sha256(pdf)
            if row.get("content_sha256") and digest != row["content_sha256"]:
                pdf.unlink()        # never keep a file that differs from the manifest
                raise ValueError("SHA-256 differs from pdf_manifest.jsonl; deleted, rerun to fetch again")
            record.update(ars_local_path=str(pdf.relative_to(out)), ars_sha256=digest)
        except Exception as exc:
            record["errors"]["pdf"] = repr(exc)

    if cik not in cache and {"10k", "proxy", "submissions"} & set(args.only):
        try:
            cache[cik] = submissions(cik, out, ua, timeout)
        except Exception as exc:
            cache[cik] = exc
    filings = cache.get(cik)
    if isinstance(filings, Exception):
        record["errors"]["submissions"] = repr(filings)
        filings = None

    tenk = match_tenk(row, filings) if filings is not None else None
    if filings is not None and not tenk:
        record["errors"]["10k"] = f"no 10-K for fiscal {year} in EDGAR submissions"
    if tenk:
        url = archive_url(cik, tenk["accessionNumber"], tenk["primaryDocument"])
        record.update(tenk_accession=tenk["accessionNumber"], tenk_url=url, tenk_filing_date=tenk["filingDate"],
                      tenk_report_date=tenk["reportDate"])
        if "10k" in args.only:
            try:
                suffix = Path(tenk["primaryDocument"]).suffix or ".htm"
                path, new = fetch(url, out / "10k" / f"EDGAR_10K_{ticker}_{year}{suffix}", ua, timeout)
                downloads += new
                stats = html_stats(path)
                record.update(tenk_local_path=str(path.relative_to(out)), tenk_sha256=stats["sha256"],
                              tenk_bytes=stats["bytes"], tenk_ixbrl_numeric_facts=stats["ixbrl_numeric_facts"],
                              tenk_ixbrl_text_facts=stats["ixbrl_text_facts"])
                total = stats["ixbrl_numeric_facts"]
                if total < MIN_PRIMARY_FACTS:
                    parts = []
                    for name in companion_documents(cik, tenk["accessionNumber"], tenk["primaryDocument"], ua, timeout):
                        part, new = fetch(archive_url(cik, tenk["accessionNumber"], name),
                                          out / "10k" / f"EDGAR_10K_{ticker}_{year}__{name}", ua, timeout)
                        downloads += new
                        part_stats = html_stats(part)
                        total += part_stats["ixbrl_numeric_facts"]
                        parts.append({"name": name, "local_path": str(part.relative_to(out)), **part_stats})
                    record["tenk_companion_documents"] = parts
                record["tenk_ixbrl_numeric_facts_total"] = total
            except Exception as exc:
                record["errors"]["10k"] = repr(exc)

    proxy = match_proxy(tenk, filings) if tenk else None
    if tenk and not proxy:
        record["errors"]["proxy"] = "no DEF 14A filed within 400 days of the 10-K (possibly not filed yet)"
    if proxy:
        url = archive_url(cik, proxy["accessionNumber"], proxy["primaryDocument"])
        record.update(proxy_accession=proxy["accessionNumber"], proxy_url=url, proxy_filing_date=proxy["filingDate"])
        if "proxy" in args.only:
            try:
                suffix = Path(proxy["primaryDocument"]).suffix or ".htm"
                path, new = fetch(url, out / "proxy" / f"EDGAR_DEF14A_{ticker}_{year}{suffix}", ua, timeout)
                downloads += new
                stats = html_stats(path)
                record.update(proxy_local_path=str(path.relative_to(out)), proxy_sha256=stats["sha256"],
                              proxy_ixbrl_numeric_facts=stats["ixbrl_numeric_facts"])
            except Exception as exc:
                record["errors"]["proxy"] = repr(exc)

    if "companyfacts" in args.only:
        try:
            path, new = fetch(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                              out / "companyfacts" / f"CIK{cik}.json", ua, timeout)
            downloads += new
            record["companyfacts_local_path"] = str(path.relative_to(out))
        except Exception as exc:
            record["errors"]["companyfacts"] = repr(exc)
    if filings is not None:
        record["submissions_local_path"] = f"submissions/CIK{cik}.json"
    return record, downloads


def write_manifest(path, records):
    tmp = path.with_name(path.name + ".part")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    tmp.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help="ARS manifest (from download_edgar_ars_pdfs.py)")
    parser.add_argument("--output-dir", type=Path, default=MANIFEST.parent,
                        help="writes pdfs/, 10k/, proxy/, companyfacts/, submissions/ and corpus_manifest.jsonl")
    parser.add_argument("--only", default=",".join(ARTIFACTS), help=f"comma-separated subset of {','.join(ARTIFACTS)}")
    parser.add_argument("--tickers", help="comma-separated subset")
    parser.add_argument("--limit", type=int, help="first N manifest rows (trial runs)")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--user-agent", default=os.getenv("SEC_USER_AGENT", ""))
    args = parser.parse_args(argv)
    args.only = [a.strip() for a in args.only.split(",") if a.strip()]
    unknown = set(args.only) - set(ARTIFACTS)
    if unknown:
        parser.error(f"unknown --only value(s): {', '.join(sorted(unknown))}")
    return args


def main(argv=None):
    args = parse_args(argv)
    if "@" not in args.user_agent:
        print("SEC requires contact details: pass --user-agent 'name/0.1 you@example.com' or set SEC_USER_AGENT")
        return 2
    rows = read_jsonl(args.manifest)
    if args.tickers:
        wanted = {t.strip().upper() for t in args.tickers.split(",")}
        rows = [r for r in rows if r["ticker"].upper() in wanted]
    rows = rows[:args.limit] if args.limit else rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_manifest = args.output_dir / "corpus_manifest.jsonl"
    previous = {r["filing_id"]: r for r in read_jsonl(out_manifest)} if out_manifest.exists() else {}

    records, cache, started, fetched = [], {}, time.time(), 0
    for number, row in enumerate(rows, 1):
        record, downloads = process(row, args, cache)
        records.append(record)
        fetched += downloads
        previous[record["filing_id"]] = record
        write_manifest(out_manifest, sorted(previous.values(), key=lambda r: (r["ticker"], -r["fiscal_year"])))
        status = "ok" if not record["errors"] else "errors: " + ", ".join(record["errors"])
        print(f"{number:03d}/{len(rows)} {record['filing_id']:<14} 10-K {record.get('tenk_accession', '-'):<22} "
              f"proxy {record.get('proxy_accession', '-'):<22} {status}", flush=True)

    failed = [r for r in records if r["errors"]]
    sizes = {name: sum(p.stat().st_size for p in (args.output_dir / name).glob("*") if p.is_file()) / 1e6
             for name in ("pdfs", "10k", "proxy", "companyfacts", "submissions") if (args.output_dir / name).exists()}
    print(json.dumps({"filings": len(records), "new_files": fetched, "with_errors": len(failed),
                      "with_10k": sum("tenk_local_path" in r for r in records),
                      "with_proxy": sum("proxy_local_path" in r for r in records),
                      "megabytes": {k: round(v, 1) for k, v in sizes.items()},
                      "minutes": round((time.time() - started) / 60, 1), "manifest": str(out_manifest)}, indent=2))
    for r in failed:
        print(f"  {r['filing_id']}: {r['errors']}")
    return 0 if not any(k in r["errors"] for r in failed for k in ("pdf", "10k", "companyfacts")) else 1


if __name__ == "__main__":
    sys.exit(main())
