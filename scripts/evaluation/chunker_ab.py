"""Chunker A/B test on separate test indexes: the current chunker (v1) vs chunker v2, same filings, same pages, same
embedder, same search. The served indexes are not opened.

    python -m scripts.evaluation.chunker_ab build      # analyse the eval-set filings, write <out>/v1 and <out>/v2
    python -m scripts.evaluation.chunker_ab evaluate   # the eval set's single-figure questions on both indexes

Both indexes hold the text-layer pages of the filings the frozen evaluation set asks about (pipeline 1 schema,
bge-small, Chroma + BM25). v1 chunks with ``financial_ingest_qdrant_db.windows``; v2 with ``rag.chunking_v2`` (column
order on two-column prose pages, line-end cuts, table header carried). Pages that need OCR are left out of both.

For each answerable single-figure question, on each index, through ``financial_ingest_chroma_db.retrieve`` (top 5,
filtered by company and year):
  (also "<index>_agent": the question's hits plus the statement search the Q&A agent adds, top 3)
  in_top5        the gold figure (as the filing prints it: millions, thousands, billions or per share) is in a hit
  year_in_chunk  ...with the fiscal year in the same line or in a column header above it
  code_correct   for statement measures, label_match reads exactly the gold figure for that year
  code_wrong     ...reads a different figure (a wrong number, worse than none)
v1 is scored as stored (``v1_raw``) and with the query-time page header (``v1``, what the served index does).
Results: <out>/results.json and a table on stdout.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from scripts.common.companies import NAMES
from scripts.common.paths import ROOT

EVAL_SET = ROOT / "data/evaluation/financial_report_qa/eval_set_v1.jsonl"
REPORTS = ROOT / "data/raw/financial_report_qa/annual_reports"
OUT = ROOT / "indexes/rag_home/indexes/chunk_test"
MODEL = "BAAI/bge-small-en-v1.5"


def questions() -> list[dict]:
    from scripts.agents.financial_clarifier import default_clarifier
    clarifier = default_clarifier()
    out = []
    for line in EVAL_SET.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        if item.get("category") != "single_figure" or item.get("answerability") != "answerable":
            continue
        gold = item.get("gold_value")
        gold = ast.literal_eval(gold) if isinstance(gold, str) else gold
        companies, years = clarifier.companies_in(item["question"]), clarifier.years_in(item["question"])
        if companies and years and gold:
            out.append({"id": item["id"], "question": item["question"], "ticker": companies[0], "year": years[0],
                        "metric": clarifier.metric_in(item["question"]), "value": gold["value"],
                        "unit": gold.get("unit")})
    return out


def pdf_for(ticker: str, year: int) -> Path | None:
    return next(iter(REPORTS.glob(f"*/EDGAR_ARS_{ticker}_{year}.pdf")), None)


def gold_strings(value: float, unit: str | None) -> set[str]:
    """How a filing may print the gold figure."""
    if unit and "/" in unit:                                    # per share
        return {f"{value:.2f}"}
    out = set()
    for scale in (1e6, 1e3, 1e9, 1):
        scaled = value / scale
        if abs(scaled - round(scaled)) < 1e-9 and abs(scaled) >= 1:
            out.add(f"{abs(round(scaled)):,}")
        elif scale == 1e9:
            out.update({f"{abs(scaled):,.1f}", f"{abs(scaled):,.2f}"})
    return out


# ----------------------------------------------------------------------------- build

def filing_records(job: dict) -> dict:
    """Both variants' (identifier, payload, text-to-embed) records for one filing (run in a worker process)."""
    import pymupdf
    from scripts.rag import chunk_schema, chunking_v2
    from scripts.rag import financial_ingest_qdrant_db as fi
    pages = fi.analyse_pdf(job["pdf"])
    base = fi.filing_payload({"document_id": job["doc_id"], "ticker": job["ticker"], "year": job["year"],
                              "local_path": job["pdf"], "sector": job.get("sector")})
    out = {"v1": [], "v2": []}
    with pymupdf.open(job["pdf"]) as pdf:
        for page in pages:
            body, source = fi.readable_text(page)
            if source != "native_pdf" or not body.strip():
                continue                                        # OCR pages: left out of both variants
            title = next((line.strip() for line in body.splitlines() if line.strip()), "")[:120]
            variants = {"v1": [c for c in fi.windows(body) if len(c) >= 40],
                        "v2": chunking_v2.chunks(chunking_v2.column_text(pdf[page["page"] - 1], body), title=title)}
            for name, parts in variants.items():
                for index, chunk in enumerate(parts):
                    identifier = fi.point_id(job["doc_id"], page["page"], "vlm", index)
                    payload = chunk_schema.chunk_payload(
                        base, page, chunk_id=fi.chunk_name(job["doc_id"], page["page"], f"vlm{index}"),
                        point_id=identifier, chunk_index=index, content_type="text", text=chunk,
                        pipeline=f"chunk_test_{name}", text_source="native_pdf", reader="pdfParser",
                        number_check="text_layer")
                    out[name].append((identifier, payload, fi.context_line(payload) + "\n" + chunk))
    return {"doc_id": job["doc_id"], **out}


def write_state(root: Path, signature: str, rows: list[tuple]) -> None:
    with sqlite3.connect(root / "vlm_ingest.sqlite3") as db:
        db.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS filings (doc_id TEXT, stage TEXT, status TEXT, pipeline TEXT, "
                   "points INTEGER, seconds REAL, error TEXT, updated_at TEXT, PRIMARY KEY (doc_id, stage))")
        db.executemany("INSERT OR REPLACE INTO settings VALUES (?, ?)", [("embedding", signature), ("store", "chroma")])
        db.executemany("INSERT OR REPLACE INTO filings VALUES (?, 'vlm', 'complete', ?, ?, 0, NULL, datetime('now'))",
                       rows)


def build(args) -> None:
    from scripts.rag import financial_ingest_chroma_db as fc
    from scripts.rag import financial_ingest_qdrant_db as fi
    out = Path(args.out)
    for name in ("v1", "v2"):
        if (out / name / "chroma").exists() and not args.force:
            raise SystemExit(f"{out / name} exists; use --force to rebuild it")
    from scripts.common.companies import SECTOR_OF
    jobs = {}
    for q in questions():
        pdf = pdf_for(q["ticker"], q["year"])
        if pdf:
            doc_id = f"edgar_ars_{q['ticker']}_{q['year']}"
            jobs[doc_id] = {"doc_id": doc_id, "pdf": str(pdf), "ticker": q["ticker"], "year": q["year"],
                            "sector": SECTOR_OF.get(q["ticker"])}
    print(f"{len(jobs)} filings to index", flush=True)
    embedder = fi.Embedder("local", MODEL)
    stores = {name: fc.ChromaStore(out / name) for name in ("v1", "v2")}
    counts = {name: [] for name in stores}
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, result in enumerate(pool.map(filing_records, jobs.values()), 1):
            for name, store in stores.items():
                records = result[name]
                vectors = embedder([text for _, _, text in records])
                store.replace(result["doc_id"], records, vectors)
                counts[name].append((result["doc_id"], f"chunk_test_{name}", len(records)))
            print(f"  {n}/{len(jobs)} {result['doc_id']}: v1 {len(result['v1'])} chunks, v2 {len(result['v2'])}, "
                  f"{time.time() - started:.0f} s", flush=True)
    for name in stores:
        write_state(out / name, embedder.portable, counts[name])
        print(f"{name}: {sum(c for *_, c in counts[name])} chunks in {out / name}", flush=True)


# ----------------------------------------------------------------------------- evaluate

def score(hits: list[dict], q: dict, gold: set[str]) -> dict:
    from scripts.agents import financial_sector_trend as st
    from scripts.rag.chunking_v2 import year_header
    in_top5 = year_ok = False
    for hit in hits:
        lines = str(hit.get("text") or "").splitlines()
        for i, line in enumerate(lines):
            if any(g in line for g in gold):
                in_top5 = True
                if (str(q["year"]) in line                         # "...to $62.6 billion in 2025"
                        or any(year_header(prev) and str(q["year"]) in prev for prev in lines[:i])):
                    year_ok = True
    result = {"in_top5": in_top5, "year_in_chunk": year_ok}
    if q["metric"] in st.STATEMENT_LABELS:
        read = st.label_match(hits, q["metric"], q["year"])
        value = (read or {}).get("value", "").replace("$", "").strip("() ")
        result.update(statement=True, code_correct=bool(read) and value in gold,
                      code_wrong=bool(read) and value not in gold, code_value=value or None)
    return result


def evaluate(args) -> None:
    from scripts.rag import financial_ingest_chroma_db as fc
    from scripts.rag import financial_ingest_qdrant_db as fi
    out = Path(args.out)
    embedder = fi.Embedder("local", MODEL)
    rows, totals = [], {}
    for q in questions():
        gold = gold_strings(q["value"], q["unit"])
        row = {**q, "gold": sorted(gold)}
        for name in ("v1", "v2"):
            hits = fc.retrieve(q["question"], 5, q["ticker"], q["year"], root=out / name, store="chroma",
                               embedder=embedder)
            variants = {name: hits}
            if q["metric"]:                                       # as the Q&A agent searches: + the statement search
                statement = fc.retrieve(f"{NAMES.get(q['ticker'], q['ticker'])} {q['metric']} fiscal year {q['year']}",
                                        3, q["ticker"], q["year"], root=out / name, store="chroma", embedder=embedder)
                seen = {h["chunk_id"] for h in hits}
                variants[f"{name}_agent"] = hits + [h for h in statement if h["chunk_id"] not in seen]
            if name == "v1":                                      # as stored: without the query-time page header
                raw = []
                for hit in hits:
                    text = hit["text"]
                    if hit.get("page_context"):
                        text = text[len(f"{fc.CONTEXT_MARK}\n{hit['page_context']}\n"):]
                    raw.append({**hit, "text": text})
                variants["v1_raw"] = raw
            for label, chosen in variants.items():
                row[label] = score(chosen, q, gold)
                row[label]["pages"] = [h.get("page") for h in chosen]
                bucket = totals.setdefault(label, {})
                for key in ("in_top5", "year_in_chunk", "code_correct", "code_wrong"):
                    if key in row[label]:
                        bucket[key] = bucket.get(key, 0) + bool(row[label][key])
                bucket["questions"] = bucket.get("questions", 0) + 1
                bucket["statement_questions"] = bucket.get("statement_questions", 0) + bool(row[label].get("statement"))
        rows.append(row)
    (out / "results.json").write_text(json.dumps({"totals": totals, "rows": rows}, indent=1), encoding="utf-8")
    print(f"\n{'index':8}{'questions':>10}{'in top 5':>10}{'year in chunk':>15}{'code correct':>14}{'code wrong':>12}"
          f"  (code: of {totals['v1']['statement_questions']} statement-measure questions)")
    for label in ("v1_raw", "v1", "v2", "v1_agent", "v2_agent"):
        t = totals[label]
        print(f"{label:8}{t['questions']:>10}{t['in_top5']:>10}{t['year_in_chunk']:>15}{t.get('code_correct', 0):>14}"
              f"{t.get('code_wrong', 0):>12}")
    print("\nquestion-level differences (v1 -> v2):")
    for row in rows:
        a, b = row["v1"], row["v2"]
        if (a["in_top5"], a["year_in_chunk"], a.get("code_correct")) != (b["in_top5"], b["year_in_chunk"], b.get("code_correct")):
            print(f"  {row['id']} {row['ticker']} {row['year']} {row['metric']}: "
                  f"top5 {a['in_top5']}->{b['in_top5']}, year {a['year_in_chunk']}->{b['year_in_chunk']}, "
                  f"code {a.get('code_value')}->{b.get('code_value')} (gold {row['gold']})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("build", "evaluate"))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    (build if args.command == "build" else evaluate)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
