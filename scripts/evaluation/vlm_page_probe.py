"""Before/after probe for the pages Qwen3-VL and the paid model (Gemini) read.

Each case (data/evaluation/financial_report_qa/vlm_page_probe.jsonl) names a filing page, a question and the figures
the page prints. The probe searches the Chroma index (pipeline 1, hybrid dense + BM25, filtered by ticker and year)
and reports, per case:

    page_found   a hit from the target page is in the top k
    value_found  that hit's full text contains every expected figure
    reader       who read the target page's hit (pdfParser, ocr, qwen3VL, paidVL)

Optionally (--api) it also asks the running API's /chat/financial_report_qa and checks the answer for the figures.

    # before the Colab run (the PC index)
    python -m scripts.evaluation.vlm_page_probe --root indexes/rag_home/indexes/chroma_db_bge --label before
    # after copying the Colab index back
    python -m scripts.evaluation.vlm_page_probe --root indexes/rag_home/indexes/chroma_db --label after
    python -m scripts.evaluation.vlm_page_probe --compare output/eval/vlm_page_probe_before.json output/eval/vlm_page_probe_after.json

Kinds: gemini (garbled text layer: nothing indexed before), qwen (figures only in an image), control (in the text
layer: found before and after), regression (long text page: must stay complete after the routing fix).
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path

from scripts.common.paths import ROOT

CASES = ROOT / "data" / "evaluation" / "financial_report_qa" / "vlm_page_probe.jsonl"
OUT = ROOT / "output" / "eval"


def squash(text):
    """Compare figures without spaces, '$' and thousands separators: '$ 3,097' == '3097', '46 %' == '46%'."""
    return re.sub(r"[\s$,]", "", str(text or "")).lower()


def contains_all(text, expected):
    body = squash(text)
    return all(squash(value) in body for value in expected)


def on_page(hit, case):
    chunk = str(hit.get("chunk_id") or "")
    return f"_{case['doc_id']}_p{int(case['page']):04d}_" in chunk


def ask_api(api, case, llm_provider, timeout=300):
    body = {"query": case["query"], "ticker": case["ticker"], "year": case["year"], "llm_provider": llm_provider,
            "technique": "agentic_graph"}
    request = urllib.request.Request(api.rstrip("/") + "/chat/financial_report_qa", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except Exception as exc:                                    # report, don't stop the probe
        return {"status": "error", "answer": repr(exc)[:200], "answer_ok": False}
    answer = result.get("answer") or ""
    return {"status": result.get("final_status"), "answer": answer[:400], "answer_ok": contains_all(answer, case["expected"])}


def probe(root, cases, top_k=8, api=None, llm_provider="mock"):
    from scripts.rag import financial_ingest_chroma_db as vi
    from scripts.rag import financial_ingest_qdrant_db as fi
    embedder = fi.recorded_embedder(argparse.Namespace(root=Path(root), embedding_provider="hashing", embedding_model=None),
                                    state_name="vlm_ingest.sqlite3")         # the model the index was built with, once
    rows = []
    for case in cases:
        started = time.time()
        hits = vi.retrieve(case["query"], top_k, case["ticker"], case["year"], root=root, embedder=embedder)
        page_hits = [h for h in hits if on_page(h, case)]
        valued = [h for h in page_hits if contains_all(h.get("text"), case["expected"])]
        best = (valued or page_hits or [None])[0]
        row = {"id": case["id"], "kind": case["kind"], "doc_id": case["doc_id"], "page": case["page"],
               "expected": case["expected"], "page_found": bool(page_hits), "value_found": bool(valued),
               "value_anywhere": any(contains_all(h.get("text"), case["expected"]) for h in hits),
               "rank": hits.index(best) + 1 if best else None,
               "reader": (best or {}).get("reader") or ((best or {}).get("metadata") or {}).get("reader"),
               "seconds": round(time.time() - started, 2)}
        if api:
            row.update(ask_api(api, case, llm_provider))
        rows.append(row)
        print(f"{row['id']:30s} {row['kind']:10s} page={'Y' if row['page_found'] else '-'} "
              f"value={'Y' if row['value_found'] else '-'} rank={row['rank'] or '-'} reader={row['reader'] or '-'}"
              + (f" answer={'Y' if row.get('answer_ok') else '-'} ({row.get('status')})" if api else ""), flush=True)
    return rows


def summary(rows):
    kinds = sorted({r["kind"] for r in rows})
    return {kind: {"cases": sum(r["kind"] == kind for r in rows),
                   "value_found": sum(r["kind"] == kind and r["value_found"] for r in rows)} for kind in kinds}


def compare(before_path, after_path):
    before = {r["id"]: r for r in json.loads(Path(before_path).read_text(encoding="utf-8"))["rows"]}
    after = {r["id"]: r for r in json.loads(Path(after_path).read_text(encoding="utf-8"))["rows"]}
    print(f"{'case':30s} {'kind':10s} {'before':16s} {'after':16s} change")
    for case_id, row in after.items():
        old = before.get(case_id, {})
        show = lambda r: (f"{'Y' if r.get('value_found') else '-'} {r.get('reader') or '-'}" if r else "n/a")
        change = ("FIXED" if row["value_found"] and not old.get("value_found")
                  else "LOST" if old.get("value_found") and not row["value_found"]
                  else "same")
        print(f"{case_id:30s} {row['kind']:10s} {show(old):16s} {show(row):16s} {change}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", help="Chroma index folder (holds chroma/ and vlm_ingest.sqlite3)")
    parser.add_argument("--cases", default=str(CASES))
    parser.add_argument("--label", default="run", help="before / after: names the output file")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--api", help="also ask the API's /chat, e.g. http://127.0.0.1:8000")
    parser.add_argument("--llm-provider", default="mock", choices=["mock", "hosted"])
    parser.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    args = parser.parse_args(argv)
    if args.compare:
        compare(*args.compare)
        return
    if not args.root:
        parser.error("--root is required (or use --compare)")
    cases = [json.loads(line) for line in Path(args.cases).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = probe(Path(args.root), cases, args.top_k, args.api, args.llm_provider)
    result = {"label": args.label, "root": str(args.root), "top_k": args.top_k, "summary": summary(rows), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"vlm_page_probe_{args.label}.json"
    target.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"]), "->", target)


if __name__ == "__main__":
    main()
