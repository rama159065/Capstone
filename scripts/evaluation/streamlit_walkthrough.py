"""Send the end-to-end walkthrough questions through Streamlit's own request code to a running API.

    python -m scripts.evaluation.streamlit_walkthrough                    # every case, nothing traced
    python -m scripts.evaluation.streamlit_walkthrough --trace            # + one LangSmith trace per agent
    python -m scripts.evaluation.streamlit_walkthrough trend fund         # only these cases

Each question goes through ``api_streamlit_app.build_payload`` and ``post_json`` to
``/conversation/financial_report_qa`` with clarifying questions on, exactly as the Streamlit chat sends it (hosted LLM,
hybrid retrieval, top k 5, guardrails and human review on). With ``--trace`` and ``LANGSMITH_API_KEY`` set here and
in the API, the cases marked ``evidence`` ask for LangSmith, so each trace starts at ``streamlit.conversation``; the
others are never traced. Results go to ``output/walkthrough/<case>.json`` and ``summary.json``.
Guide and results: docs/end_to_end_walkthrough.md.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from scripts.common.paths import ROOT

# (case, technique, evidence trace?, question)
CASES = [
    ("router_qa", "router", True, "What was Citi's U.S. Personal Banking net income in 2025 compared with 2024?"),
    ("qa_compare", "agentic_graph", True, "Which company had higher net income attributable to the company in its "
                                          "fiscal year ending in 2025: JPMorgan Chase or Bank of America?"),
    ("qa_text", "agentic_graph", False, "Who was Bank of America's independent auditor for fiscal year 2025?"),
    ("qa_chart", "agentic_graph", False, "According to the production charts in Chevron's 2022 annual report, what "
                                         "was Chevron's net crude oil production in 2022?"),
    ("trend", "trend_graph", True, "How did JPMorgan Chase's net income trend from 2022 to 2025?"),
    ("sector_trend", "sector_trend_graph", True, "How did banking sector net income trend from 2022 to 2025?"),
    ("fund", "fund_graph", True, "What are the top 10 holdings of the Vanguard 500 Index Fund?"),
    ("router_refuse", "router", False, "Should I buy NVIDIA stock now?"),
    ("router_clarify", "router", False, "What was the revenue?"),
    ("router_report", "router", False, "Generate a full financial report for Costco for fiscal 2025"),
    ("router_not_available", "router", False, "What was Apple's net income in fiscal 2024?"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", help="case names (default: all)")
    parser.add_argument("--api-base", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--trace", action="store_true", help="one LangSmith trace per agent (the 'evidence' cases)")
    parser.add_argument("--project", default=os.getenv("LANGSMITH_PROJECT", "annual-report-copilot"))
    parser.add_argument("--out", default=str(ROOT / "output" / "walkthrough"))
    args = parser.parse_args(argv)
    os.environ.setdefault("API_TIMEOUT_SECONDS", "900")          # the sector trend takes about two minutes
    from scripts.ui import api_streamlit_app as ui

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    summary_path = os.path.join(out, "summary.json")
    summary = json.load(open(summary_path, encoding="utf-8")) if os.path.exists(summary_path) else {}
    settings = {"api_base": args.api_base, "ticker": "", "year": None, "sectors": [], **ui.DEMO_DEFAULTS,
                "llm_provider": "hosted", "retrieval_mode": "hybrid", "top_k": 5, "enable_guardrails": True,
                "enable_human_review": True}
    for name, technique, evidence, query in CASES:
        if args.cases and name not in args.cases:
            continue
        trace = bool(args.trace and evidence)
        payload = {**ui.build_payload(query, {**settings, "technique": technique, "enable_langsmith": trace,
                                              "langsmith_project": args.project if trace else ""}),
                   "clarify": True, "conversation_id": None}
        start = time.time()
        try:
            turn = ui.post_json(args.api_base, "/conversation/financial_report_qa", payload)
        except Exception as exc:  # noqa: BLE001 - recorded, the next case still runs
            turn = {"status": "error", "error": str(exc)}
        seconds = round(time.time() - start, 1)
        with open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8") as handle:
            json.dump({"technique": technique, "query": query, "traced": trace, "seconds": seconds, "turn": turn},
                      handle, indent=1, ensure_ascii=False)
        result = turn.get("result") or {}
        summary[name] = {"technique": technique, "traced": trace, "seconds": seconds, "status": turn.get("status"),
                         "final_status": result.get("final_status")}
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=1)
        print(f"{name:22} {technique:19} {seconds:6.1f} s  {turn.get('status')}  {result.get('final_status')}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
