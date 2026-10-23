"""Evaluate financial-report RAG retrieval recall.

This evaluates retrieval only. It does not call an LLM and does not grade
answer quality. Use a human-authored eval file for final capstone metrics.

Smoke-test path with current local hashing embeddings:
    python scripts/evaluate_financial_rag.py --embedding-provider hashing --generate-silver
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import chromadb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.domain_common.langsmith_utils import (
    add_langsmith_args,
    maybe_enable_from_args,
    run_traced,
)


DEFAULT_EVAL_FILE = Path("eval/financial_report_qa/retrieval_eval_silver.json")
DEFAULT_RESULTS_FILE = Path("eval/financial_report_qa/retrieval_eval_results.json")
DEFAULT_CHUNKS = Path("data/processed/financial_report_qa/pdf_chunks.jsonl")
DEFAULT_INDEX_DIR = Path("indexes/chroma/financial_report_qa_pdfs")
DEFAULT_COLLECTION = "financial_report_qa_pdfs"


def load_indexer_module():
    module_path = Path("scripts/index_financial_report_pdfs.py")
    spec = importlib.util.spec_from_file_location("financial_pdf_indexer", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["financial_pdf_indexer"] = module
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def pick_phrase(text: str) -> str:
    text = normalize_space(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    priority_terms = [
        "net sales",
        "revenue",
        "operating",
        "cash",
        "risk",
        "services",
        "product",
        "business",
        "table",
        "annual report",
    ]
    candidates = [
        sentence for sentence in sentences
        if 45 <= len(sentence) <= 220
    ]
    for term in priority_terms:
        for sentence in candidates:
            if term in sentence.lower():
                return sentence[:180]
    if candidates:
        return candidates[0][:180]
    return text[:180]


def make_silver_query(row: dict, query_id: str) -> dict:
    metadata = row["metadata"]
    company = metadata.get("company")
    ticker = metadata.get("ticker")
    year = metadata.get("year")
    page = metadata.get("page")
    source_type = metadata.get("source_type")
    text = row.get("text", "")

    if source_type == "table_summary":
        phrase = pick_phrase(text)
        query = f"Find the table in {company} {year} annual report about: {phrase}"
    elif source_type == "chart_summary":
        query = f"Find the chart or figure in {company} {year} annual report on page {page}."
    else:
        phrase = pick_phrase(text)
        query = f"In {company} {year} annual report, where is this discussed: {phrase}"

    return {
        "query_id": query_id,
        "query": query,
        "ticker": ticker,
        "year": year,
        "source_type": source_type,
        "relevant_chunk_ids": [row["chunk_id"]],
        "answerable": True,
        "eval_type": "silver_generated_from_indexed_chunk",
    }


def generate_silver_eval(chunks_path: Path, output_path: Path, num_queries: int) -> list[dict]:
    rows = read_jsonl(chunks_path)
    by_type = {
        "annual_report_pdf": [
            row for row in rows
            if row["metadata"].get("source_type") == "annual_report_pdf"
            and len(row.get("text", "")) >= 350
        ],
        "table_summary": [
            row for row in rows
            if row["metadata"].get("source_type") == "table_summary"
        ],
        "chart_summary": [
            row for row in rows
            if row["metadata"].get("source_type") == "chart_summary"
        ],
    }
    targets = {
        "annual_report_pdf": max(0, num_queries - 20),
        "table_summary": min(10, len(by_type["table_summary"])),
        "chart_summary": min(10, len(by_type["chart_summary"])),
    }
    targets["annual_report_pdf"] = num_queries - targets["table_summary"] - targets["chart_summary"]

    selected = []
    for source_type, target_count in targets.items():
        candidates = sorted(
            by_type[source_type],
            key=lambda row: (
                str(row["metadata"].get("ticker")),
                int(row["metadata"].get("year") or 0),
                int(row["metadata"].get("page") or 0),
                row["chunk_id"],
            ),
        )
        if not candidates:
            continue
        step = max(1, len(candidates) // max(1, target_count))
        selected.extend(candidates[::step][:target_count])

    selected = selected[:num_queries]
    eval_rows = [
        make_silver_query(row, f"silver_{index:03d}")
        for index, row in enumerate(selected, start=1)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(eval_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return eval_rows


def load_eval(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for index, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / index
    return 0.0


def evaluate(args: argparse.Namespace) -> dict:
    indexer = load_indexer_module()
    eval_rows = load_eval(args.eval_file)
    chunk_rows = indexer.load_chunk_rows(args.chunks)
    client = chromadb.PersistentClient(path=str(args.index_dir))
    collection = client.get_collection(args.collection)
    embedder = indexer.get_embedder(args.embedding_provider, args.hash_dimensions)

    max_k = max(args.k_values)
    per_query = []
    for item in eval_rows:
        source_type = item.get("filter_source_type") or item.get("source_type")
        ticker = item.get("filter_ticker") or item.get("ticker")
        year = item.get("filter_year") or item.get("year")
        if args.ignore_year_filter:
            year = None
        if args.ignore_source_type_filter:
            source_type = None

        if args.retrieval_mode == "dense":
            hits = indexer.retrieve_top_k(
                collection=collection,
                embedder=embedder,
                query=item["query"],
                top_k=max_k,
                source_type=source_type,
                ticker=ticker,
                year=year,
            )
        else:
            hits = indexer.retrieve_hybrid_top_k(
                collection=collection,
                embedder=embedder,
                chunk_rows=chunk_rows,
                query=item["query"],
                top_k=max_k,
                dense_top_k=args.dense_top_k,
                bm25_top_k=args.bm25_top_k,
                rrf_k=args.rrf_k,
                reranker=args.reranker,
                source_type=source_type,
                ticker=ticker,
                year=year,
            )

        retrieved_ids = [str(hit["chunk_id"]) for hit in hits]
        relevant_ids = set(item["relevant_chunk_ids"])
        recalls = {
            f"recall@{k}": any(chunk_id in relevant_ids for chunk_id in retrieved_ids[:k])
            for k in args.k_values
        }
        per_query.append(
            {
                "query_id": item["query_id"],
                "query": item["query"],
                "relevant_chunk_ids": sorted(relevant_ids),
                "retrieved_chunk_ids": retrieved_ids,
                "first_relevant_rank": next(
                    (
                        index for index, chunk_id in enumerate(retrieved_ids, start=1)
                        if chunk_id in relevant_ids
                    ),
                    None,
                ),
                "mrr": reciprocal_rank(retrieved_ids, relevant_ids),
                "source_type": source_type,
                "ticker": ticker,
                "year": year,
                **recalls,
            }
        )

    miss_k = 10 if 10 in args.k_values else max(args.k_values)
    miss_key = f"recall@{miss_k}"
    summary = {
        "eval_file": str(args.eval_file),
        "results_file": str(args.results_file),
        "retrieval_mode": args.retrieval_mode,
        "embedding_provider": args.embedding_provider,
        "queries": len(per_query),
        "k_values": args.k_values,
        "metrics": {
            f"recall@{k}": mean(1.0 if row[f"recall@{k}"] else 0.0 for row in per_query)
            for k in args.k_values
        },
        "mrr": mean(row["mrr"] for row in per_query),
        "source_type_counts": dict(Counter(str(row.get("source_type")) for row in per_query)),
        f"misses@{miss_k}": [
            row for row in per_query
            if not row.get(miss_key, False)
        ],
    }
    results = {"summary": summary, "per_query": per_query}
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    args.results_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-file", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-provider", choices=["hashing", "hosted"], default="hashing")
    parser.add_argument("--hash-dimensions", type=int, default=384)
    parser.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--dense-top-k", type=int, default=30)
    parser.add_argument("--bm25-top-k", type=int, default=30)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker", choices=["simple", "none"], default="simple")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--generate-silver", action="store_true")
    parser.add_argument("--num-silver-queries", type=int, default=100)
    parser.add_argument("--ignore-year-filter", action="store_true")
    parser.add_argument("--ignore-source-type-filter", action="store_true")
    add_langsmith_args(parser, default_project="capstone-financial-rag-eval")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    langsmith_status = maybe_enable_from_args(args)
    if args.generate_silver or not args.eval_file.exists():
        generate_silver_eval(args.chunks, args.eval_file, args.num_silver_queries)
    summary = run_traced(
        "financial_report_qa.retrieval_eval",
        evaluate,
        args,
        run_type="chain",
        tags=[
            "capstone",
            "financial_report_qa",
            "retrieval_eval",
            args.retrieval_mode,
            args.embedding_provider,
            *args.langsmith_tag,
        ],
        metadata={
            "eval_file": str(args.eval_file),
            "chunks": str(args.chunks),
            "collection": args.collection,
            "k_values": args.k_values,
            "dense_top_k": args.dense_top_k,
            "bm25_top_k": args.bm25_top_k,
            "reranker": args.reranker,
            "ignore_year_filter": args.ignore_year_filter,
            "ignore_source_type_filter": args.ignore_source_type_filter,
        },
    )
    summary["langsmith"] = langsmith_status
    sys.stdout.buffer.write(
        (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8", errors="replace")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
