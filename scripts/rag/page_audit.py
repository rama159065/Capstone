"""Page audit: which parser each page of each downloaded filing needs, one CSV per PDF.

Run it after downloading the corpus (``full`` or ``min500``). It reads the sector manifests, analyses every
page with ``financial_ingest_qdrant_db.analyse_pdf`` (the same analysis both ingestion pipelines use) and writes

    <out>/<sector_slug>/<pdf file name without .pdf>.csv     pageNumber,parserUsage   (one row per page)
    <out>/_summary.csv                                       one row per filing: pages per parser + seconds
    <out>/_summary.json                                      totals per parser and per sector, run time

so the output tree mirrors ``annual_reports/<sector_slug>/``. Finished filings are skipped on a rerun
(``--force`` redoes them), so a Colab disconnect costs at most the filings in progress.

parserUsage values (decided from the page analysis, before any OCR or VLM runs):

    none               nothing any parser can turn into answerable content: a blank page, or a picture page
                       (photo, logo) with no text layer and no figures; skip it, don't pay a reader for it
    pdfParser          plain text: the PDF text layer is exact and enough
    pdfParser+colpali  a table, chart or picture whose figures are in the text layer: parse the text (tables via
                       PyMuPDF find_tables), and index the page image with ColPali so the visual is findable
    ocr                printed text with an unusable text layer: font-encoding garbage, glyphs drawn as vector
                       outlines (thousands of shapes, no text layer), or a scanned page (a full-page image with
                       almost no text): RapidOCR reads it
    qwen3VL            a chart without printed values in the text layer, or a figure/infographic with little text:
                       needs a vision-language model (Qwen3-VL) to read the layout
    paidVL             a table, chart or picture whose text layer is unusable: nothing to check a model's figures
                       against, so it goes to the strongest (paid) model

These are predictions from cheap signals. Whether OCR or Qwen actually succeeds is only known by running them
(pipeline 1 flags figures a model writes that the text layer doesn't print as ``unverified_numbers``).
A chart with no printed values is ``qwen3VL``: a model can only estimate its bars, so treat those figures as estimates.
The label is lower-case ``none`` on purpose: pandas reads the text ``None`` in a CSV as a missing value.
A PDF that can't be opened (corrupt, encrypted, missing, checksum mismatch) gets no CSV; it is listed under
``failed`` in ``_summary.json``.

    python -m scripts.rag.page_audit --manifest data/raw/financial_report_qa/annual_reports --corpus-preset min500 \
        --out output/page_audit --workers 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scripts.common.paths import ROOT
from scripts.rag import financial_ingest_qdrant_db as fi

LABELS = ("pdfParser", "pdfParser+colpali", "ocr", "qwen3VL", "paidVL", "none")
MIN_TEXT_WORDS = 15          # fewer real words than this: the page has (almost) no text layer
MIN_CHART_NUMBERS = 6        # a chart with at least this many figures in its text layer has printed data labels
SCANNED_IMAGE_SHARE = .6     # an image covering this much of a page with no text layer looks like a scan
TEXT_PAGE_WORDS = 600       # a "chart" with this many real words is a text page whose frame/rules are vector art;
                             # Qwen3-VL would truncate it at 1,200 tokens, so the complete text layer is kept instead
OUTLINED_SHAPES = 1000      # a "chart" with this many vector shapes and no text layer is text drawn as outlines
                             # (Costco 2025: 2,800-3,400 shapes per page); real charts have tens to hundreds
WORD = re.compile(r"[a-z]{2,}")
NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+%?|\d+%|\$\s?\d+")


def route(page: dict) -> str:
    """One page analysis (financial_ingest_qdrant_db.analyse_page) -> parserUsage."""
    kind, garbled = page["page_type"], page["garbled"]
    words = len(WORD.findall(page.get("text", "").lower()))
    numbers = len(NUMBER.findall(page.get("text", "")))
    if kind == "blank":
        # analyse_page calls a page blank when it has no English words, which includes a fully garbled text layer
        return "ocr" if garbled or len(page.get("text", "").strip()) >= 40 else "none"
    if kind == "outlined_text":
        return "ocr"
    if garbled:
        return "paidVL" if kind in {"table", "chart", "image"} else "ocr"
    if kind == "table":
        return "pdfParser+colpali"
    if kind == "chart":
        if words >= TEXT_PAGE_WORDS:
            return "pdfParser"                 # a text page with decorative vector art (BAC 2025, TRV 2023, CME 2022)
        if words < MIN_TEXT_WORDS and page.get("shapes", 0) >= OUTLINED_SHAPES:
            return "ocr"                       # glyphs drawn as outlines, not a chart
        return "pdfParser+colpali" if numbers >= MIN_CHART_NUMBERS else "qwen3VL"
    if kind == "image":
        if words >= MIN_TEXT_WORDS:
            return "pdfParser+colpali"
        if page.get("image_share", 0) >= SCANNED_IMAGE_SHARE:
            return "ocr"                       # a full-page image: maybe a scan (OCR finds nothing on a photo, cheaply)
        return "qwen3VL" if numbers else "none"   # figures nearby: an infographic; else a photo or logo
    return "pdfParser"


def csv_path(out: Path, row: dict) -> Path:
    name = Path(str(row["local_path"]).replace("\\", "/")).stem
    return Path(out) / (row.get("sector_slug") or "unknown") / f"{name}.csv"


def audit_filing(row: dict, out: str, pdf_dir: str | None, pages_root: str | None, force: bool) -> dict:
    """Analyse one filing and write its CSV (atomically). Runs in a worker process."""
    target = csv_path(Path(out), row)
    result = {"document_id": row["document_id"], "sector_slug": row.get("sector_slug"), "ticker": row.get("ticker"),
              "year": row.get("year"), "csv": str(target)}
    if target.exists() and not force:
        with target.open(encoding="utf-8") as handle:
            counts = Counter(r["parserUsage"] for r in csv.DictReader(handle))
        return {**result, "status": "skipped", "pages": sum(counts.values()), "counts": dict(counts), "seconds": 0.0}
    started = time.time()
    try:
        pdf = fi.resolve_pdf(row, pdf_dir)
        pages = fi.load_pages(pages_root, row, pdf, None) if pages_root else fi.analyse_pdf(pdf)
    except Exception as exc:                                  # a bad or missing PDF must not stop the run
        return {**result, "status": "failed", "error": repr(exc), "pages": 0, "counts": {},
                "seconds": round(time.time() - started, 1)}
    labels = [(page["page"], route(page)) for page in pages]
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pageNumber", "parserUsage"])
        writer.writerows(labels)
    tmp.replace(target)
    counts = Counter(label for _, label in labels)
    return {**result, "status": "complete", "pages": len(labels), "counts": dict(counts),
            "seconds": round(time.time() - started, 1)}


def run(manifest, out, *, preset="full", sectors=None, pdf_dir=None, pages_root=None, workers=None, force=False,
        limit=None, quiet=False):
    rows = fi.in_preset(fi.read_manifest(Path(manifest), sectors), preset)
    rows = rows[:limit] if limit else rows
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    workers = max(1, workers or os.cpu_count() or 1)
    started, results = time.time(), []
    args = (str(out), str(pdf_dir) if pdf_dir else None, str(pages_root) if pages_root else None, force)
    if not quiet:
        print(f"{len(rows)} filings ({preset}) -> {out} with {workers} worker(s)", flush=True)
    if workers == 1:
        done = (audit_filing(row, *args) for row in rows)
    else:
        pool = ProcessPoolExecutor(workers)
        done = (f.result() for f in as_completed([pool.submit(audit_filing, row, *args) for row in rows]))
    try:
        for n, result in enumerate(done, 1):
            results.append(result)
            if not quiet:
                elapsed = time.time() - started
                eta = elapsed / n * (len(rows) - n)
                print(json.dumps({"n": f"{n}/{len(rows)}", "doc": result["document_id"], "status": result["status"],
                                  "pages": result["pages"], "seconds": result["seconds"], **result["counts"],
                                  "eta_min": round(eta / 60, 1)}), flush=True)
    finally:
        if workers > 1:
            pool.shutdown()
    return write_summary(out, results, time.time() - started, preset, workers)


def write_summary(out: Path, results: list[dict], seconds: float, preset: str, workers: int) -> dict:
    results = sorted(results, key=lambda r: (r.get("sector_slug") or "", r["document_id"]))
    with (out / "_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sector_slug", "document_id", "ticker", "year", "status", "pages", *LABELS, "seconds"])
        for r in results:
            writer.writerow([r.get("sector_slug"), r["document_id"], r.get("ticker"), r.get("year"), r["status"],
                             r["pages"], *(r["counts"].get(label, 0) for label in LABELS), r["seconds"]])
    totals, sectors = Counter(), {}
    for r in results:
        totals.update(r["counts"])
        sectors.setdefault(r.get("sector_slug"), Counter()).update(r["counts"])
    analysed = [r for r in results if r["status"] == "complete"]
    pages = sum(totals.values())
    summary = {
        "preset": preset, "workers": workers, "filings": len(results),
        "status": dict(Counter(r["status"] for r in results)), "pages": pages,
        "parser_usage": {label: totals.get(label, 0) for label in LABELS},
        "parser_share": {label: round(totals.get(label, 0) / pages, 3) if pages else 0 for label in LABELS},
        "by_sector": {s: {label: c.get(label, 0) for label in LABELS} for s, c in sorted(sectors.items())},
        "failed": [{"document_id": r["document_id"], "error": r.get("error")} for r in results if r["status"] == "failed"],
        "wall_seconds": round(seconds, 1),
        "seconds_per_page": round(sum(r["seconds"] for r in analysed) / max(1, sum(r["pages"] for r in analysed)), 3),
    }
    (out / "_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/raw/financial_report_qa/annual_reports",
                        help="the sector corpus folder (<dir>/<sector>/pdf_manifest.jsonl) or one manifest file")
    parser.add_argument("--out", type=Path, default=ROOT / "output/page_audit")
    parser.add_argument("--corpus-preset", choices=["full", "min500"], default="full")
    parser.add_argument("--sectors", help="comma-separated sector names or slugs")
    parser.add_argument("--pdf-dir", type=Path, help="where the PDFs are (default: next to each manifest)")
    parser.add_argument("--pages-root", type=Path,
                        help="also cache the page analysis in <pages-root>/pages, where ingestion reuses it "
                             "(pass the ingest --root)")
    parser.add_argument("--workers", type=int, help="parallel processes (default: every CPU)")
    parser.add_argument("--limit", type=int, help="first N filings only (timing runs)")
    parser.add_argument("--force", action="store_true", help="redo filings that already have a CSV")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    summary = run(args.manifest, args.out, preset=args.corpus_preset, sectors=args.sectors, pdf_dir=args.pdf_dir,
                  pages_root=args.pages_root, workers=args.workers, force=args.force, limit=args.limit)
    print(json.dumps({k: summary[k] for k in ("filings", "status", "pages", "parser_usage", "parser_share",
                                              "wall_seconds", "seconds_per_page")}, indent=2))


if __name__ == "__main__":
    main()
