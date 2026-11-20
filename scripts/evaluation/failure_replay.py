"""Replay the Week 1 single-prompt failures through the capstone's deterministic layers (no LLM, no network).

    python -m scripts.evaluation.failure_replay local_qwen3vl2b_oracle local_qwen3vl2b_bm25 [local_qwen3vl2b_rag]

For every incorrect answer recorded by ``prompt_baselines`` (``output/prompt_baselines/<run>/scores.jsonl``) it asks
which check of the agentic graph would have stopped it before release, using the same functions the graph runs:

    policy            financial_prompting_agent.policy_refusal: advice, forecasts, personal or non-public data and
                      creative requests are refused by rule in the graph's first step (security_step), no LLM call
    input_guard       financial_prompting_agent.deterministic_input_guard: injection / code / encoded payloads
    coverage          financial_clarifier.question_scope: a company or year the filings don't hold -> not_available
    company_scope     guardrails.checks.company_scope_check: a cited chunk from a company the question didn't name
    numeric_grounding financial_tools.numeric_grounding: a figure that appears in no cited chunk
    output_filter     financial_prompting_agent.output_content_filter: advice wording -> human review

A caught failure becomes a safe outcome (refused, not available, withheld for evidence, or sent to review), not a
correct answer. Failures no layer catches are the ones only better retrieval, the calculator or the critic's repair
loop can turn into correct answers; ``prompt_baselines --context rag`` measures the retrieval part.
Writes ``output/prompt_baselines/failure_replay.json`` and ``.md``.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict

from scripts.evaluation import prompt_baselines as pb

LAYERS = ("policy", "input_guard", "coverage", "company_scope", "numeric_grounding", "output_filter")
CITED = re.compile(r"\bc\d+\b")


def _tools():
    from scripts.agents import financial_prompting_agent as agent
    from scripts.agents import financial_router, financial_tools
    from scripts.agents.financial_clarifier import default_clarifier
    from scripts.guardrails import checks
    return agent, financial_router, financial_tools, default_clarifier(), checks


def replay_answer(item: dict, row: dict, hits: list[dict], tools=None) -> dict:
    """Which layers stop one recorded answer. ``row`` is a scores.jsonl row, ``hits`` its evidence."""
    agent, router, ftools, clarifier, checks = tools or _tools()
    question = item["question"]
    answer = row.get("answer") or ""
    parsed = pb.parse_json(answer) if row["strategy"] in pb.JSON_STRATEGIES else None
    text = pb.answer_text(answer, parsed)
    caught = {}
    policy = agent.policy_refusal(question)
    if policy["refuse"]:
        caught["policy"] = ", ".join(policy["labels"])
    route = router.rule_route(question, clarifier)[0]            # recorded for context; not a catching layer
    guard = agent.deterministic_input_guard(question)
    if guard.get("risk_labels"):
        caught["input_guard"] = ", ".join(guard["risk_labels"])
    scope = clarifier.scope(question)
    if scope.get("coverage"):
        caught["coverage"] = scope["coverage"].get("reason") or scope["coverage"].get("status")
    declined = pb.declined(text, parsed)
    cited_ids = sorted(set(CITED.findall(text)) | set((parsed or {}).get("citations") or []))
    cited = [h for h in hits if h["chunk_id"] in cited_ids] or ([] if declined else hits)
    if not declined:
        if scope.get("expected") and cited_ids:
            check = checks.company_scope_check(hits, cited_ids, scope["expected"])
            if not check["valid"]:
                caught["company_scope"] = "cites " + ", ".join(
                    sorted({str(h.get("ticker")) for h in hits if h["chunk_id"] in check["foreign_chunk_ids"]}))
        grounding = ftools.numeric_grounding(text, cited, [])
        if not grounding["grounded"]:
            caught["numeric_grounding"] = "not in cited evidence: " + ", ".join(grounding["ungrounded"][:4])
        decision = agent.output_content_filter(text).get("decision")
        if decision in {"review", "block"}:
            caught["output_filter"] = decision
    return {"route": route, "caught_by": caught, "declined": declined}


def replay_run(run_id: str, tools=None) -> list[dict]:
    out = pb.OUT_ROOT / run_id
    items = {i["id"]: i for i in pb.read_jsonl(pb.EVAL_SET)}
    hits = {r["id"]: r["hits"] for r in pb.read_jsonl(out / "retrieval.jsonl")}
    tools = tools or _tools()
    rows = []
    for row in pb.read_jsonl(out / "scores.jsonl"):
        if row["correct"]:
            continue
        item = items[row["id"]]
        result = replay_answer(item, row, hits[row["id"]], tools)
        rows.append({"run": run_id, "id": row["id"], "category": row["category"], "strategy": row["strategy"],
                     "failure": row["failure"], **result, "answer": (row.get("answer") or "")[-300:]})
    return rows


def summarize(rows: list[dict]) -> dict:
    by_failure = defaultdict(lambda: {"incorrect": 0, "caught": 0, "layers": Counter()})
    for r in rows:
        entry = by_failure[(r["run"], r["failure"])]
        entry["incorrect"] += 1
        entry["caught"] += bool(r["caught_by"])
        for layer in r["caught_by"]:
            entry["layers"][layer] += 1
    return {f"{run}|{failure}": {**v, "layers": dict(v["layers"])} for (run, failure), v in sorted(by_failure.items())}


def render(rows: list[dict]) -> str:
    runs = list(dict.fromkeys(r["run"] for r in rows))
    lines = ["# Week 1 failures replayed through the capstone's checks", "",
             "Incorrect single-prompt answers (every strategy scored in each run), and how many a deterministic layer of the "
             "agentic graph would stop before release. Caught = refused, not available, withheld or sent to review; "
             "it is a safe outcome, not a correct answer.", ""]
    for run in runs:
        rr = [r for r in rows if r["run"] == run]
        lines += [f"## {run}", "", "| Failure label | Incorrect | Caught | " + " | ".join(LAYERS) + " |",
                  "|---" * (3 + len(LAYERS)) + "|"]
        labels = Counter(r["failure"] for r in rr)
        for label, n in labels.most_common():
            sub = [r for r in rr if r["failure"] == label]
            layer_counts = Counter(layer for r in sub for layer in r["caught_by"])
            lines.append(f"| {label} | {n} | {sum(bool(r['caught_by']) for r in sub)} | "
                         + " | ".join(str(layer_counts.get(layer, 0)) for layer in LAYERS) + " |")
        total = len(rr)
        lines += ["", f"**Total:** {sum(bool(r['caught_by']) for r in rr)} of {total} incorrect answers caught.", ""]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    run_ids = argv if argv is not None else sys.argv[1:]
    if not run_ids:
        run_ids = ["local_qwen3vl2b_oracle", "local_qwen3vl2b_bm25"]
    tools = _tools()
    rows = [row for run_id in run_ids if (pb.OUT_ROOT / run_id / "scores.jsonl").exists()
            for row in replay_run(run_id, tools)]
    (pb.OUT_ROOT / "failure_replay.json").write_text(
        json.dumps({"summary": summarize(rows), "rows": rows}, indent=1, ensure_ascii=False), encoding="utf-8")
    text = render(rows)
    (pb.OUT_ROOT / "failure_replay.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
