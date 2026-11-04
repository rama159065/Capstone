"""Generate real ColPali vectors, persist them in Qdrant, and verify MaxSim queries."""
import json
import time

from scripts.rag.colpali_index import configure_colpali
from scripts.rag.multimodal_rag import LocalRAG, ROOT


def main():
    configure_colpali()
    report = {"model": "vidore/colpali-v1.3-hf", "quantization": "4bit-nf4",
              "synthetic_vectors": False, "checks": []}
    rag = LocalRAG()
    try:
        print("Backfilling real ColPali page embeddings into " + str(rag.root), flush=True)
        start = time.monotonic()
        report["backfill"] = rag.backfill_visual("financial_report_qa")
        report["backfill_seconds"] = round(time.monotonic()-start, 2)
        report["storage"] = str(rag.root)
        report["counts"] = {name: rag.client.count(name, exact=True).count for name in ["docs_text", "visual_pages"]}
        points, _ = rag.client.scroll("visual_pages", scroll_filter=rag.filter("financial_report_qa"),
                                      limit=100, with_vectors=True)
        report["stored_pages"] = []
        for point in points:
            matrix = point.vector["colpali"]
            assert len(matrix) > 1 and all(len(row) == 128 for row in matrix)
            assert point.payload["image_path"]
            report["stored_pages"].append({"point_id": str(point.id), "document": point.payload["title"],
                "page": point.payload["page"], "patches": len(matrix), "dimensions": 128})
        assert points, "No actual ColPali page vectors were stored"
        for query, expected in [
            ("What is the North regional budget in million USD?", "table_budget"),
            ("Which quarter has the highest revenue in the chart?", "chart_revenue")]:
            start = time.monotonic()
            hits = rag.retrieve_visual("financial_report_qa", query, 2)
            passed = bool(hits) and hits[0]["title"] == expected
            row = {"query": query, "expected": expected, "retrieved": [h["title"] for h in hits],
                   "maxsim_scores": [h["colpali_score"] for h in hits], "passed": passed,
                   "elapsed_seconds": round(time.monotonic()-start, 2)}
            merged = rag.retrieve("financial_report_qa", query, 1)
            row["merged_retrieval_passed"] = bool(merged) and merged[0]["title"] == expected and "visual" in merged[0]["retrieval_paths"]
            report["checks"].append(row)
            print(json.dumps(row), flush=True)
        report["domain_isolation"] = rag.retrieve_visual("hr_policy", "revenue chart") == []
        report["idempotent_backfill"] = rag.backfill_visual("financial_report_qa")["visual_pages_written"] == 0
        from fastapi.testclient import TestClient
        from scripts.api import api_app
        from scripts.rag import multimodal_rag as mm
        mm._instance = rag
        try:
            with TestClient(api_app.app) as client:
                response = client.post("/multimodal/financial_report_qa/visual/retrieve", json={
                    "query": "Which quarter has the highest revenue in the chart?", "top_k": 1})
                report["api_passed"] = response.status_code == 200 and response.json()["evidence"][0]["title"] == "chart_revenue"
        finally:
            mm._instance = None
    finally:
        rag.close()
    reopened = LocalRAG()
    try:
        report["persisted_after_reopen"] = reopened.client.count("visual_pages", exact=True).count == report["counts"]["visual_pages"]
    finally:
        reopened.close()
    output = ROOT / "output/colpali_validation"
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# ColPali Qdrant: actual model validation", "", "Model: official vidore/colpali-v1.3-hf, CPU NF4 quantization. No synthetic vectors.", "",
             "Storage: `indexes/multimodal_full/qdrant`; summary vectors plus 128-dimensional ColPali multivectors with MaxSim.", "",
             f"Collection counts: {report['counts']}", "", "| Document | Page | Patch vectors | Dimension |", "|---|---|---|---|"]
    lines += [f"| {p['document']} | {p['page']} | {p['patches']} | {p['dimensions']} |" for p in report["stored_pages"]]
    lines += ["", "| Query | Top result | MaxSim | Result |", "|---|---|---|---|"]
    lines += [f"| {r['query']} | {r['retrieved'][0] if r['retrieved'] else 'None'} | {r['maxsim_scores'][0] if r['maxsim_scores'] else 'None'} | {'PASS' if r['passed'] else 'FAIL'} |" for r in report["checks"]]
    lines += ["", f"Domain isolation: {report['domain_isolation']}. Idempotent backfill: {report['idempotent_backfill']}. Reopened database persistence: {report['persisted_after_reopen']}."]
    lines += ["", f"Merged text/visual retrieval: {all(r['merged_retrieval_passed'] for r in report['checks'])}. Visual FastAPI endpoint: {report['api_passed']}."]
    (ROOT / "docs/colpali_validation_results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    if not all(r["passed"] and r["merged_retrieval_passed"] for r in report["checks"]) or not all(report[k] for k in ["domain_isolation", "idempotent_backfill", "persisted_after_reopen", "api_passed"]):
        raise SystemExit("ColPali validation failed; inspect the persisted report")


if __name__ == "__main__":
    main()
