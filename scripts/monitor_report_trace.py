"""Summarize one financial-report trace or all runs in a logical thread."""
import argparse
from collections import Counter
import json
from pathlib import Path
from uuid import UUID


def summarize(root, *, trace_id=None, thread_id=None):
    if not trace_id and not thread_id:
        raise ValueError("A trace ID or thread ID is required")
    trace_id = str(UUID(trace_id)) if trace_id else None
    thread_id = str(UUID(thread_id)) if thread_id else None
    paths = Path(root).glob(f"**/logs/{trace_id or '*'}.jsonl")
    runs = []
    for path in paths:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows or (thread_id and rows[0]["thread_id"] != thread_id):
            continue
        manifest = path.parent.parent / "manifest.json"
        state = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        runs.append({"trace_id": rows[0]["trace_id"], "thread_id": rows[0]["thread_id"],
                     "status": state.get("status", "unknown"), "log_path": str(path.resolve()),
                     "events": len(rows), "event_counts": dict(Counter(r["event"] for r in rows)),
                     "failures": [r for r in rows if r["event"].endswith(("failed", "withheld")) or r.get("status") in {"error", "timeout"}],
                     "agents": [r for r in rows if r["event"] in {"span.completed", "span.failed"}],
                     "calls": [r for r in rows if r["event"] == "call.completed"],
                     "artifacts": state.get("artifacts", [])})
    return {"runs": runs, "count": len(runs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "output/pdf")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--trace-id")
    selection.add_argument("--thread-id")
    args = parser.parse_args()
    result = summarize(args.root, trace_id=args.trace_id, thread_id=args.thread_id)
    print(json.dumps(result, indent=2))
    return 0 if result["count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
