"""Check that a running API sends complete traces to LangSmith (local PC, Colab or a Google Cloud VM).

    python -m scripts.observability.check_langsmith                       # API at API_BASE_URL or 127.0.0.1:8000
    python -m scripts.observability.check_langsmith --api-base http://127.0.0.1:8010 --llm-provider hosted
    docker compose exec api python -m scripts.observability.check_langsmith      # on the VM, inside the API container

Steps (each prints PASS, FAIL or SKIP):

1. LangSmith is reachable from this machine (``GET <endpoint>/info``).
2. ``GET /health``: the API has a key (``langsmith.api_key_set``); nothing is sent without one.
3. One ``/conversation`` turn with ``enable_langsmith=true`` and a fresh ``X-Request-ID``: the response says
   ``langsmith.enabled`` and names the project.
4. Looking the runs up in LangSmith by that request ID (needs ``LANGSMITH_API_KEY`` here too; the runs arrive within
   a few seconds): the conversation turn, the agent graph nodes and at least one LLM run, all with the request ID.

The question goes to whatever backend the API serves (``--ticker``/``--year`` pick a company the index holds), with the
mock LLM unless ``--llm-provider hosted``. The key is read from the environment and never printed.
Exit code 0 when every step that ran passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

EXPECTED_NODES = ("security_step", "plan_step", "retrieve_step", "answer_step", "critic_step", "governance_step")


def http_json(url: str, body: dict[str, Any] | None = None, headers: dict[str, str] | None = None,
              timeout: float = 30) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if body is not None else "GET",
                                     headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        return exc.code, {"detail": exc.read()[:300].decode("utf-8", "replace")}
    except (urllib.error.URLError, OSError) as exc:
        return 599, {"detail": f"{type(exc).__name__}: {exc}"}


def endpoint() -> str:
    return (os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
            or "https://api.smith.langchain.com").rstrip("/")


def find_runs(project: str, request_id: str, wait: float) -> list[Any]:
    """The runs of one request in LangSmith, polled until the graph's last node arrives or ``wait`` runs out."""
    from langsmith import Client
    client = Client()
    query = f'and(eq(metadata_key, "request_id"), eq(metadata_value, "{request_id}"))'
    deadline, runs = time.monotonic() + wait, []
    while time.monotonic() < deadline:
        runs = list(client.list_runs(project_name=project, filter=query, limit=100))
        if any(run.name == "governance_step" for run in runs):
            break
        time.sleep(3)
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api-base", default=os.getenv("API_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--llm-provider", default="mock", choices=["mock", "hosted"])
    parser.add_argument("--question", default="What was net income attributable to the company in its annual report?")
    parser.add_argument("--ticker", default="CVX")
    parser.add_argument("--year", type=int, default=2022)
    parser.add_argument("--project", help="LangSmith project (default: the one the API reports)")
    parser.add_argument("--wait", type=float, default=60, help="Seconds to wait for the runs to appear.")
    args = parser.parse_args(argv)
    api = args.api_base.rstrip("/")
    headers = {"X-API-Key": os.environ["API_AUTH_TOKEN"]} if os.getenv("API_AUTH_TOKEN") else {}
    results: list[tuple[str, str, str]] = []

    def report(step: str, status: str, detail: str) -> None:
        results.append((step, status, detail))
        print(f"[{status}] {step}: {detail}", flush=True)

    code, info = http_json(endpoint() + "/info", timeout=15)
    report("1 LangSmith reachable", "PASS" if code == 200 else "FAIL",
           f"{endpoint()} answered HTTP {code}" + ("" if code == 200 else f" ({info.get('detail', '')[:120]})"))

    code, health = http_json(api + "/health", headers=headers)
    if code != 200:
        report("2 API health", "FAIL", f"{api}/health answered HTTP {code}: start the API first")
        return 1
    ls = health.get("langsmith") or {}
    report("2 API has a key", "PASS" if ls.get("api_key_set") else "FAIL",
           f"api_key_set={ls.get('api_key_set')}, tracing_all_requests={ls.get('tracing_all_requests')}, "
           f"project={ls.get('project')}, backend={(health.get('runtime') or {}).get('financial_backend')}"
           + (f", warning: {ls['warning']}" if ls.get("warning") else ""))

    request_id = str(uuid.uuid4())
    body = {"query": args.question, "ticker": args.ticker, "year": args.year, "clarify": False,
            "technique": "agentic_graph", "llm_provider": args.llm_provider, "enable_langsmith": True}
    if args.project:
        body["langsmith_project"] = args.project
    started = time.perf_counter()
    code, turn = http_json(api + "/conversation/financial_report_qa", body, {**headers, "X-Request-ID": request_id},
                           timeout=600)
    traced = ((turn.get("result") or {}).get("langsmith") or {})
    project = args.project or traced.get("project")
    report("3 Traced request", "PASS" if code == 200 and traced.get("enabled") else "FAIL",
           f"HTTP {code} in {time.perf_counter() - started:.1f}s, request_id={request_id}, "
           f"final_status={(turn.get('result') or {}).get('final_status')}, langsmith={traced}")

    if not (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        report("4 Runs in LangSmith", "SKIP", "set LANGSMITH_API_KEY in this shell to look the runs up; or open the "
               f"project '{project}' and filter by metadata request_id = {request_id}")
    elif not traced.get("enabled"):
        report("4 Runs in LangSmith", "SKIP", "the request was not traced (see step 3)")
    else:
        try:
            runs = find_runs(project, request_id, args.wait)
        except Exception as exc:                       # wrong key, project not created yet, network
            report("4 Runs in LangSmith", "FAIL", f"{type(exc).__name__}: {str(exc)[:200]}")
        else:
            names = {run.name for run in runs}
            missing = [node for node in ("api.conversation_turn", *EXPECTED_NODES) if node not in names]
            llm_runs = [run for run in runs if run.run_type == "llm"]
            report("4 Runs in LangSmith", "PASS" if runs and not missing and llm_runs else "FAIL",
                   f"{len(runs)} runs, {len(llm_runs)} LLM runs, project '{project}'"
                   + (f"; missing {missing}" if missing else "") + f"; filter metadata request_id = {request_id}")
    failed = [step for step, status, _ in results if status == "FAIL"]
    print("RESULT:", "all checks passed" if not failed else f"failed: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
