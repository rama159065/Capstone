"""Run real model inference through FastAPI and save independently checked results.

Run with RAG_EMBEDDING_MODEL and RAG_TRANSFORMERS_MODEL pointing at local models.
No model output is mocked; chart labels are absent from the PDF text layer.
"""
import json
from pathlib import Path
import re
import time

import pymupdf
from fastapi.testclient import TestClient

from scripts.domain_common import multimodal_rag as mm

CASES = [
    {"id": "text_code", "type": "text", "query": "What is the annual review code?", "file": "text_operations", "expected": [r"\borchid\b"]},
    {"id": "text_deadline", "type": "text", "query": "Within how many days must expense reports be submitted?", "file": "text_operations", "expected": [r"\b14\b"]},
    {"id": "table_north", "type": "table", "query": "What is the North regional budget in million USD?", "file": "table_budget", "expected": [r"\b73\b"]},
    {"id": "table_south", "type": "table", "query": "What is the South regional budget in million USD?", "file": "table_budget", "expected": [r"\b46\b"]},
    {"id": "table_difference", "type": "table", "query": "How much larger is the North budget than the South budget in million USD?", "file": "table_budget", "expected": [r"\b27\b"]},
    {"id": "chart_trend", "type": "chart", "query": "Does quarterly revenue increase or decrease from Q1 to Q4?", "file": "chart_revenue", "expected": [r"increas|ris|upward|grow"]},
    {"id": "chart_maximum", "type": "chart", "query": "Which quarter has the highest revenue in the chart?", "file": "chart_revenue", "expected": [r"q\s*4|fourth|quarter 4"]},
    {"id": "chart_value", "type": "chart", "query": "What is Q4 revenue in million USD in the chart?", "file": "chart_revenue", "expected": [r"\b40\b"]},
    {"id": "unsupported", "type": "unsupported", "query": "What is the chief executive's private phone number?", "file": None, "expected": [r"no evidence|insufficient evidence|no (verified )?answer found"]},
]


def main():
    from scripts import api_app
    output = mm.ROOT / "output/multimodal_validation"
    output.mkdir(parents=True, exist_ok=True)
    folder = mm.ROOT / "data/samples/multimodal_validation/financial_report_qa"
    with pymupdf.open(folder / "chart_revenue.pdf") as pdf:
        assert not pdf[0].get_text().strip(), "Chart fixture must have no text layer"
    rag = mm.LocalRAG()
    mm._instance = rag
    rows = []
    try:
        ingestion = []
        for path in sorted(folder.glob("*.pdf")):
            print("Ingesting " + path.name, flush=True)
            ingestion.append(rag.ingest("financial_report_qa", path))
            print(json.dumps(ingestion[-1]), flush=True)
        with TestClient(api_app.app) as client:
            for case in CASES:
                start = time.monotonic()
                response = client.post("/multimodal/financial_report_qa/chat", json={
                    "query": case["query"], "provider": "llamacpp", "top_k": 1})
                result = response.json()
                answer = result.get("answer", "")
                evidence = result.get("evidence", [])
                correct_source = bool(evidence) and evidence[0]["title"] == case["file"] if case["file"] else not evidence
                generated = answer.split("\nSources:")[0]
                correct_answer = all(re.search(p, generated, re.I) for p in case["expected"])
                # A fallback excerpt containing the expected number is not a correct answer.
                correct_answer = correct_answer and result.get("final_status") == (
                    "insufficient_evidence" if case["type"] == "unsupported" else "evidence_found")
                if case["id"] == "chart_trend" and re.search(r"decreas|declin", generated, re.I):
                    correct_answer = False
                visual_used = result.get("images_used", 0) > 0 if case["type"] == "chart" else True
                citations_ok = result.get("citation_check", {}).get("valid", False) if case["file"] else result.get("final_status") == "insufficient_evidence"
                passed = response.status_code == 200 and correct_source and correct_answer and visual_used and citations_ok
                row = {**case, "answer": answer, "retrieved_titles": [h["title"] for h in evidence],
                       "images_used": result.get("images_used", 0), "status_code": response.status_code,
                       "final_status": result.get("final_status"), "warnings": result.get("warnings", []),
                       "trace_id": result.get("trace_id"), "trace": result.get("trace", []),
                       "rerank_scores": [h["rerank_score"] for h in evidence],
                       "correct_source": correct_source, "correct_answer": bool(correct_answer),
                       "passed": passed, "elapsed_seconds": round(time.monotonic()-start, 2)}
                rows.append(row)
                print(json.dumps(row), flush=True)
                (output / "results.json").write_text(json.dumps({"ingestion": ingestion, "cases": rows}, indent=2), encoding="utf-8")
        report = ["# Local multimodal RAG: actual inference results", "", "Model: Qwen3-VL-2B-Instruct Q4_K_M with Q8 image projector, llama.cpp CPU.", "",
                  "Retrieval: MiniLM dense embeddings + sparse BM25-style vectors, Qdrant RRF, MS MARCO cross-encoder reranking.", "",
                  "Chart PDF text layer: empty. No mocked model outputs.", "",
                  "| Case | Modality | Source | Correct answer | Images | Result |", "|---|---|---|---|---|---|"]
        report += [f"| {r['id']} | {r['type']} | {r['file'] + '.pdf, page 1' if r['file'] else 'No evidence'} | {r['correct_answer']} | {r['images_used']} | {'PASS' if r['passed'] else 'FAIL'} |" for r in rows]
        for r in rows:
            report += ["", f"## {r['id']}", "", r["query"], "", r["answer"]]
        (output / "report.md").write_text("\n".join(report)+"\n", encoding="utf-8")
        (mm.ROOT / "docs/multimodal_validation_results.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    finally:
        mm._instance = None
        rag.close()
    if not all(row["passed"] for row in rows):
        raise SystemExit("Some actual inference cases failed; inspect output/multimodal_validation/report.md")


if __name__ == "__main__":
    main()
