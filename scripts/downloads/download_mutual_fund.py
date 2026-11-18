"""Download SEC Form N-PORT data for the mutual fund layer (design ``docs/mutual_fund_rag_design.md`` section 2.1).

Two sources:

* ``latest``: for each fund in ``configs/funds_mutual_fund.json`` (registrant CIK + series name fragment), the most
  recent ``NPORT-P`` filing's ``primary_doc.xml`` from EDGAR, saved as ``nport_<accession>_mutual_fund.xml``, with its
  filing date in ``manifest_mutual_fund.jsonl``. A trust files one NPORT-P per series, so the newest filings are read
  until the series name matches (``--scan`` filings at most).
* ``bulk``: one quarterly data set ZIP (about 440 MB, ``2026q3`` -> ``2026q3_nport_mutual_fund.zip``) for many funds;
  load it with ``nport_mutual_fund load --tsv-zip``.

SEC asks for a User-Agent with a contact e-mail and at most 10 requests per second: ``--user-agent`` or
``SEC_USER_AGENT``. Files go to ``<RAG_HOME>/indexes/mutual_fund/raw_mutual_fund`` (gitignored, like all data).

    python -m scripts.downloads.download_mutual_fund latest
    python -m scripts.downloads.download_mutual_fund latest --cik 36405 --series "500 INDEX"
    python -m scripts.downloads.download_mutual_fund bulk --quarter 2026q3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

from scripts.common.paths import ROOT
from scripts.rag import nport_mutual_fund as nport

CONFIG = ROOT / "configs" / "funds_mutual_fund.json"
SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{folder}/primary_doc.xml"
BULK = "https://www.sec.gov/files/dera/data/form-n-port-data-sets/{quarter}_nport.zip"
PAUSE = 0.15                                   # well under SEC's 10 requests per second


class Sec:
    def __init__(self, user_agent: str):
        if not user_agent or "@" not in user_agent:
            raise SystemExit('SEC requires a contact e-mail: --user-agent "capstone-research you@example.com" or SEC_USER_AGENT')
        self.headers = {"User-Agent": user_agent, "Accept-Encoding": "identity"}
        self.cache: dict[str, bytes] = {}          # several funds of one trust read the same newest filings

    def get(self, url: str) -> bytes:
        if url in self.cache:
            return self.cache[url]
        time.sleep(PAUSE)
        with urllib.request.urlopen(urllib.request.Request(url, headers=self.headers), timeout=120) as response:
            self.cache[url] = response.read()
        return self.cache[url]

    def save(self, url: str, path: Path) -> int:
        """Stream a large file to ``path`` (written to ``.part`` first, then renamed)."""
        time.sleep(PAUSE)
        part = path.with_suffix(path.suffix + ".part")
        size = 0
        with urllib.request.urlopen(urllib.request.Request(url, headers=self.headers), timeout=600) as response, \
                part.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
                size += len(chunk)
        part.replace(path)
        return size


def output_dir(args) -> Path:
    folder = Path(args.out) if args.out else nport.default_dir() / nport.RAW_DIR
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def latest(sec: Sec, cik: int, series: str, scan: int = 40) -> dict[str, Any]:
    """The newest NPORT-P of a series: reads the trust's newest filings until the series name matches."""
    recent = json.loads(sec.get(SUBMISSIONS.format(cik=cik)))["filings"]["recent"]
    pattern = re.compile(re.escape(series), re.I)
    seen = 0
    for form, accession, filed, report in zip(recent["form"], recent["accessionNumber"], recent["filingDate"],
                                              recent["reportDate"]):
        if form != "NPORT-P":
            continue
        seen += 1
        if seen > scan:
            break
        xml = sec.get(ARCHIVE.format(cik=cik, folder=accession.replace("-", "")))
        name = re.search(rb"<seriesName>(.*?)</seriesName>", xml)
        if name and pattern.search(name.group(1).decode("utf-8", "replace")):
            return {"accession": accession, "cik": cik, "series_name": name.group(1).decode("utf-8", "replace"),
                    "filing_date": filed, "report_date": report, "xml": xml}
    raise LookupError(f"no NPORT-P for {series!r} in the newest {scan} filings of CIK {cik}")


def write_manifest(folder: Path, row: dict[str, Any]) -> None:
    path = folder / nport.MANIFEST_NAME
    rows = nport.read_manifest(folder)
    rows[row["accession"]] = row
    path.write_text("".join(json.dumps(r) + "\n" for r in rows.values()), encoding="utf-8")


def run_latest(args) -> list[dict[str, Any]]:
    sec = Sec(args.user_agent)
    folder = output_dir(args)
    funds = ([{"cik": args.cik, "series": args.series}] if args.cik and args.series
             else json.loads(CONFIG.read_text(encoding="utf-8"))["funds"])
    done = []
    for fund in funds:
        try:
            found = latest(sec, int(fund["cik"]), fund["series"], args.scan)
        except (LookupError, OSError, KeyError, ValueError) as exc:
            done.append({"cik": fund["cik"], "series": fund["series"], "error": str(exc)[:300]})
            continue
        path = folder / f"nport_{found['accession']}_mutual_fund.xml"
        path.write_bytes(found.pop("xml"))
        row = {**found, "file": path.name, "source": "edgar_nport_p"}
        write_manifest(folder, row)
        done.append(row)
    return done


def run_bulk(args) -> dict[str, Any]:
    if not re.fullmatch(r"\d{4}q[1-4]", args.quarter):
        raise SystemExit("--quarter looks like 2026q3")
    folder = output_dir(args)
    path = folder / f"{args.quarter}_nport_mutual_fund.zip"
    if path.exists() and not args.force:
        return {"file": str(path), "bytes": path.stat().st_size, "skipped": "exists (use --force)"}
    size = Sec(args.user_agent).save(BULK.format(quarter=args.quarter), path)
    return {"file": str(path), "bytes": size}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("latest", "bulk"):
        one = sub.add_parser(name)
        one.add_argument("--user-agent", default=os.getenv("SEC_USER_AGENT"), help="must contain a contact e-mail")
        one.add_argument("--out", help="default: <RAG_HOME>/indexes/mutual_fund/raw_mutual_fund")
        if name == "latest":
            one.add_argument("--cik", type=int, help="registrant (trust) CIK; default: every fund in the config")
            one.add_argument("--series", help="series name fragment, e.g. '500 INDEX'")
            one.add_argument("--scan", type=int, default=40, help="newest NPORT-P filings to read per trust")
        else:
            one.add_argument("--quarter", required=True, help="e.g. 2026q3")
            one.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = run_latest(args) if args.command == "latest" else run_bulk(args)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
