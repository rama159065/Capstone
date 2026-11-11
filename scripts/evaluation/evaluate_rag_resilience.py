"""Offline fault-injection evaluation; independent of live-model accuracy scoring."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output/resilience_evaluation"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = args.output / "junit.xml"
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/test_resilience.py",
                             "tests/test_complex_query_audit.py", "-q", f"--junitxml={report}"],
                            cwd=root, timeout=180)
    suites = ET.parse(report).getroot()
    cases = [{"name": case.attrib["name"],
              "passed": not any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))}
             for case in suites.iter("testcase")]
    summary = {"suite": "offline_fault_injection", "live_model_accuracy_tested": False,
               "passed": sum(c["passed"] for c in cases), "total": len(cases), "cases": cases,
               "elapsed_seconds": round(time.monotonic() - started, 2)}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cases"}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
