"""Download the sector annual-report corpus (docs/annual_report_corpus_plan.html) from SEC EDGAR.

Reads ``corpus_plan.jsonl`` (1,004 Form ARS PDFs, 281 companies, 13 sectors; every
URL was checked with a HEAD request when the plan was made) and writes one folder
per sector, each with its own manifest::

    <output-dir>/
        corpus_plan.jsonl                  copied here when --output-dir is elsewhere (e.g. Drive)
        pdf_manifest.jsonl                 all sectors; local_path = "<sector>/<file>.pdf"
        download_summary.json
        banking/pdf_manifest.jsonl         local_path = "<file>.pdf"
        banking/EDGAR_ARS_JPM_2025.pdf
        ...

Every manifest row carries ``sector`` / ``sector_slug`` / ``sic``, and the same
fields as the EDGAR ARS manifest (``document_id = edgar_ars_<TICKER>_<YEAR>``,
SHA-256, pages), so both indexers can read it:

    python -m scripts.rag.financial_ingest ingest --manifest data/raw/financial_report_qa/annual_reports --sectors banking
    python scripts/rag/index_financial_report_pdfs.py --manifest data/raw/financial_report_qa/annual_reports --sectors banking

Reruns are resumable: PDFs already listed with the right size are skipped, and
PDFs present in --reuse-dir (the earlier 100-PDF EDGAR download) are copied
instead of fetched. SEC asks for a User-Agent with a contact e-mail and at most
10 requests/second; the limit here is shared by all worker threads.

    python -m scripts.downloads.download_annual_report_corpus --dry-run
    python -m scripts.downloads.download_annual_report_corpus --user-agent "capstone-research you@example.com" \
        --sectors banking,insurance --max-per-sector 20
    # Colab, from a clone of the repo, PDFs on Drive:
    python -m scripts.downloads.download_annual_report_corpus --output-dir /content/drive/MyDrive/capstone_rag/annual_reports
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.common.sector_corpus import MANIFEST_NAME, PLAN_NAME, parse_sectors, read_jsonl, sector_slug  # noqa: E402

DEFAULT_DIR = ROOT / "data/raw/financial_report_qa/annual_reports"
DEFAULT_REUSE_DIR = ROOT / "data/raw/financial_report_qa/edgar/pdfs"
DOMAIN = "financial_report_qa"
MAX_REQUESTS_PER_SECOND = 8          # SEC allows 10
CHUNK = 1 << 20


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def pdf_name(row: dict) -> str:
    ticker = "".join(ch if ch.isalnum() else "-" for ch in row["ticker"].upper())
    return f"EDGAR_ARS_{ticker}_{int(row['year'])}.pdf"


def document_id(row: dict) -> str:
    return f"edgar_ars_{row['ticker'].upper()}_{int(row['year'])}"


def page_count(path: Path) -> int | None:
    try:
        import pymupdf
    except ImportError:
        return None
    try:
        with pymupdf.open(path) as document:
            return document.page_count
    except Exception:
        return None


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def jsonl(rows) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


class RateLimiter:
    """At most `per_second` request starts per second across threads."""

    def __init__(self, per_second: float):
        self.interval, self.next, self.lock = 1.0 / per_second, 0.0, threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next)
            self.next = start + self.interval
        time.sleep(max(0.0, start - now))


def select(plan: list[dict], args) -> list[dict]:
    sectors = parse_sectors(args.sectors)
    tickers = {t.strip().upper() for t in args.tickers.split(",")} if args.tickers else None
    years = {int(y) for y in args.years.split(",")} if args.years else None
    unknown = sectors - {sector_slug(r["sector"]) for r in plan} if sectors else set()
    if unknown:
        raise SystemExit(f"unknown sector(s) {sorted(unknown)}; choose from {sorted({sector_slug(r['sector']) for r in plan})}")
    rows, per_sector = [], Counter()
    for row in sorted(plan, key=lambda r: (sector_slug(r["sector"]), r.get("rank", 0), r["ticker"], -int(r["year"]))):
        slug = sector_slug(row["sector"])
        if (sectors and slug not in sectors) or (tickers and row["ticker"].upper() not in tickers) \
                or (years and int(row["year"]) not in years):
            continue
        if args.max_per_sector and per_sector[slug] >= args.max_per_sector:
            continue
        per_sector[slug] += 1
        rows.append(row)
    return rows[:args.limit] if args.limit else rows


class Downloader:
    def __init__(self, args):
        self.args, self.out = args, Path(args.output_dir)
        self.limiter = RateLimiter(args.max_requests_per_second)
        self.lock = threading.Lock()
        self.existing = self.load_existing()

    # --------------------------------------------------------------------- manifests

    def sector_manifest(self, slug: str) -> Path:
        return self.out / slug / MANIFEST_NAME

    def load_existing(self) -> dict[str, dict]:
        rows = {}
        for manifest in sorted(self.out.glob(f"*/{MANIFEST_NAME}")):
            for row in read_jsonl(manifest):
                rows[row["document_id"]] = row
        return rows

    def done(self, plan_row: dict) -> dict | None:
        row = self.existing.get(document_id(plan_row))
        if not row:
            return None
        path = self.out / row["sector_slug"] / row["local_path"]
        if not path.exists() or path.stat().st_size != row["bytes"]:
            return None
        if self.args.verify and sha256_file(path) != row["content_sha256"]:
            return None
        return row

    def record(self, row: dict) -> None:
        with self.lock:
            self.existing[row["document_id"]] = row
            with self.sector_manifest(row["sector_slug"]).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def rewrite_manifests(self) -> int:
        """Sector manifests deduplicated and sorted; the combined manifest rebuilt from them."""
        by_sector = defaultdict(list)
        for row in self.existing.values():
            if (self.out / row["sector_slug"] / row["local_path"]).exists():
                by_sector[row["sector_slug"]].append(row)
        combined = []
        for slug in sorted(by_sector):
            rows = sorted(by_sector[slug], key=lambda r: (r["ticker"], -r["year"]))
            write_atomic(self.sector_manifest(slug), jsonl(rows))
            combined += [{**r, "local_path": f"{slug}/{r['local_path']}"} for r in rows]
        write_atomic(self.out / MANIFEST_NAME, jsonl(combined))
        return len(combined)

    # --------------------------------------------------------------------- fetching

    def fetch(self, url: str, part: Path, expected: int | None) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": self.args.user_agent, "Accept-Encoding": "identity"})
        for attempt in range(4):
            self.limiter.wait()
            try:
                with urllib.request.urlopen(request, timeout=self.args.timeout) as response, part.open("wb") as handle:
                    while block := response.read(CHUNK):
                        handle.write(block)
                break
            except urllib.error.HTTPError as exc:
                if exc.code in {429, 503} and attempt < 3:
                    time.sleep(10 * (attempt + 1))          # SEC throttling: back off
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead):
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        size = part.stat().st_size
        if expected and size != expected:
            raise ValueError(f"got {size} bytes, the plan recorded {expected}")

    def download(self, plan_row: dict) -> tuple[str, dict]:
        slug, name = sector_slug(plan_row["sector"]), pdf_name(plan_row)
        dest = self.out / slug / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(name + ".part")
        expected = plan_row.get("bytes")
        reuse = Path(self.args.reuse_dir) / name if self.args.reuse_dir else None
        try:
            if dest.exists() and (not expected or dest.stat().st_size == expected):
                how = "present"                               # file there, manifest row missing
            else:
                if reuse and reuse.exists() and (not expected or reuse.stat().st_size == expected):
                    try:
                        os.link(reuse, part)                  # same disk: no second copy
                    except OSError:
                        shutil.copyfile(reuse, part)
                    how = "reused"
                else:
                    self.fetch(plan_row["url"], part, expected)
                    how = "downloaded"
                with part.open("rb") as handle:
                    if handle.read(5) != b"%PDF-":
                        raise ValueError("not a PDF (no %PDF- header)")
                part.replace(dest)
        except Exception as exc:
            part.unlink(missing_ok=True)
            return "failed", {"document_id": document_id(plan_row), "sector": plan_row["sector"],
                              "url": plan_row["url"], "error": f"{type(exc).__name__}: {exc}"}
        row = manifest_row(plan_row, dest)
        self.record(row)
        return how, row

    # --------------------------------------------------------------------- run

    def run(self, rows: list[dict]) -> dict:
        counts, failures = defaultdict(Counter), []
        todo = []
        for row in rows:
            if self.done(row):
                counts[sector_slug(row["sector"])]["skipped"] += 1
            else:
                todo.append(row)
        print(f"{len(rows)} selected, {len(rows) - len(todo)} already downloaded, {len(todo)} to fetch "
              f"({sum(r.get('bytes') or 0 for r in todo) / 1e9:.2f} GB)", flush=True)
        started = time.time()
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            futures = {pool.submit(self.download, row): row for row in todo}
            for n, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                how, info = future.result()
                counts[sector_slug(row["sector"])][how] += 1
                if how == "failed":
                    failures.append(info)
                print(f"[{n}/{len(todo)}] {how:10s} {sector_slug(row['sector'])}/{pdf_name(row)}"
                      + (f"  {info['error']}" if how == "failed" else ""), flush=True)
        total = self.rewrite_manifests()
        summary = {"finished_at": utc_now(), "seconds": round(time.time() - started, 1),
                   "selected": len(rows), "in_manifest": total,
                   "by_sector": {slug: dict(c) for slug, c in sorted(counts.items())},
                   "failures": failures}
        write_atomic(self.out / "download_summary.json", json.dumps(summary, indent=2))
        return summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def manifest_row(plan_row: dict, path: Path) -> dict:
    year, cik = int(plan_row["year"]), plan_row.get("cik")
    return {
        "domain": DOMAIN, "document_id": document_id(plan_row),
        "company": plan_row["company"], "ticker": plan_row["ticker"].upper(),
        "cik": f"{int(cik):010d}" if cik is not None else None, "year": year,
        "title": f"{plan_row['company']} annual report to shareholders {year}",
        "sector": plan_row["sector"], "sector_slug": sector_slug(plan_row["sector"]),
        "sic": plan_row.get("sic"), "sic_desc": plan_row.get("sic_desc"),
        "source_name": "SEC EDGAR", "source_url": plan_row["url"], "form": "ARS",
        "accession_number": plan_row.get("accession"), "filing_date": plan_row.get("filed"),
        "local_path": path.name, "pages": page_count(path), "bytes": path.stat().st_size,
        "content_sha256": sha256_file(path), "downloaded_at": utc_now(), "source_type": "annual_report_pdf",
    }


def dry_run(rows: list[dict], args) -> dict:
    reuse = Path(args.reuse_dir) if args.reuse_dir else None
    table = defaultdict(Counter)
    for row in rows:
        slug = sector_slug(row["sector"])
        table[slug]["pdfs"] += 1
        table[slug]["bytes"] += row.get("bytes") or 0
        table[slug]["companies_" + row["ticker"]] = 1
        if reuse and (reuse / pdf_name(row)).exists():
            table[slug]["reusable"] += 1
    report = {slug: {"pdfs": c["pdfs"], "companies": sum(1 for k in c if k.startswith("companies_")),
                     "gb": round(c["bytes"] / 1e9, 2), "reusable_local": c["reusable"]}
              for slug, c in sorted(table.items())}
    print(f"{'sector':26s} {'pdfs':>5s} {'cos':>4s} {'GB':>6s} {'local':>6s}")
    for slug, r in report.items():
        print(f"{slug:26s} {r['pdfs']:5d} {r['companies']:4d} {r['gb']:6.2f} {r['reusable_local']:6d}")
    print(f"{'total':26s} {sum(r['pdfs'] for r in report.values()):5d} "
          f"{sum(r['companies'] for r in report.values()):4d} {sum(r['gb'] for r in report.values()):6.2f} "
          f"{sum(r['reusable_local'] for r in report.values()):6d}")
    print(f"would write to {Path(args.output_dir)}/<sector>/")
    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=DEFAULT_DIR / PLAN_NAME)
    parser.add_argument("--output-dir", type=Path, help="corpus root (default: the plan's folder)")
    parser.add_argument("--user-agent", default=os.getenv("SEC_USER_AGENT"), help="must contain a contact e-mail")
    parser.add_argument("--sectors", help="comma-separated names or slugs, e.g. banking,oil_and_gas (default: all 13)")
    parser.add_argument("--tickers", help="comma-separated subset")
    parser.add_argument("--years", help="comma-separated fiscal years, e.g. 2024,2025")
    parser.add_argument("--max-per-sector", type=int, help="first N per sector, largest companies first")
    parser.add_argument("--limit", type=int, help="first N overall (smoke runs)")
    parser.add_argument("--reuse-dir", type=Path, default=DEFAULT_REUSE_DIR if DEFAULT_REUSE_DIR.exists() else None,
                        help="copy PDFs of the same name and size from here instead of downloading")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-requests-per-second", type=float, default=MAX_REQUESTS_PER_SECOND)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--verify", action="store_true", help="re-hash finished PDFs instead of trusting their size")
    parser.add_argument("--dry-run", action="store_true", help="show what would be fetched per sector; no network")
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir or args.plan.parent
    if args.max_requests_per_second > 10:
        parser.error("SEC allows at most 10 requests/second")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.plan.exists():
        raise SystemExit(f"{args.plan} not found (it ships with the repo under data/raw/financial_report_qa/annual_reports)")
    rows = select(read_jsonl(args.plan), args)
    if args.dry_run:
        dry_run(rows, args)
        return 0
    if not args.user_agent or "@" not in args.user_agent:
        raise SystemExit('SEC requires a contact e-mail: --user-agent "capstone-research you@example.com" or SEC_USER_AGENT')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.plan.resolve() != (out / PLAN_NAME).resolve():
        shutil.copyfile(args.plan, out / PLAN_NAME)            # the corpus folder describes itself
    summary = Downloader(args).run(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=2))
    if summary["failures"]:
        print(f"{len(summary['failures'])} failed; rerun to retry them (see download_summary.json)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
