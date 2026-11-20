"""Tools of the financial agent graph, and the rules that keep the planner's choices safe.

The planner LLM proposes a plan (``tools``, ``needs_calculation``, ...). ``normalize_plan`` turns it into
the plan the executor runs:

* only known tools survive (legacy names are mapped), ``text_search`` is always kept because every answer
  must cite text, and tools the deployment can't run (``visual_search`` without ColPali) are dropped;
* deterministic hints add what the question clearly needs (``calculator`` for growth/ratios,
  ``coverage_lookup`` when it names a company or year), so a weak or mock LLM can't skip them;
* ``targets`` split a comparison or trend into one filtered search per company and year.

Every change to the LLM's plan is listed in ``plan["adjustments"]``, so the trace shows who decided what.
The calculator evaluates ``CALC[...]`` expressions the answer agent writes, with a whitelist-only parser,
and checks that each operand appears in the retrieved evidence.
"""
from __future__ import annotations

import ast
import operator
import os
import re
from typing import Any

TOOLS = {
    "coverage_lookup": "Check which companies and fiscal years the served index holds; skip searches it can't answer.",
    "text_search": "Hybrid dense + BM25 search over filing text and tables, filtered by company, year and sector.",
    "visual_search": "ColPali search over chart, table and image pages (needs RAG_FIN_VISUAL=1 or auto).",
    "calculator": "Evaluate CALC[...] arithmetic from the answer, using only numbers found in the evidence.",
}
TOOL_ORDER = ("coverage_lookup", "text_search", "visual_search", "calculator")
LEGACY_TOOLS = {"hybrid_financial_search": "text_search", "hybrid_search": "text_search", "table_lookup": "text_search",
                "chart_lookup": "visual_search", "search": "text_search", "calc": "calculator"}
CALC_TERMS = re.compile(r"\b(grow(?:s|n|th|ing)?|grew|change[sd]?|increase[sd]?|decrease[sd]?|decline[sd]?|ratio|margin|difference|"
                        r"compare[sd]?|comparison|versus|vs\.?|cagr|percent(age)?|trend|how much (more|less))\b|%", re.I)
VISUAL_TERMS = re.compile(r"\b(chart|graph|figure|image|diagram|map|infographic|segment mix|breakdown)s?\b", re.I)
MAX_TARGETS = 6
MAX_RETRIEVAL_ATTEMPTS = 2           # first search + one broadened retry


def visual_mode() -> str:
    """'always' (RAG_FIN_VISUAL=1, every query), 'auto' (the planner decides) or 'off'."""
    if os.getenv("RAG_BACKEND") == "financial_hybrid":       # ColPali is on by default there ('auto')
        from scripts.rag.financial_hybrid import visual_setting
        return visual_setting()
    if os.getenv("RAG_BACKEND") != "financial_qdrant":
        return "off"
    value = os.getenv("RAG_FIN_VISUAL", "").strip().lower()
    return {"1": "always", "true": "always", "auto": "auto"}.get(value, "off")


def plan_targets(query: str, ticker: str | None, year: int | None, clarifier) -> list[dict[str, Any]]:
    """One search per company x year the question names (a request's ticker/year win over the text)."""
    companies = [ticker.upper()] if ticker else clarifier.companies_in(query)
    years = [year] if year else clarifier.years_in(query)
    if len(years) > 3:                  # long spans: one search per company, the query carries the years
        years = []
    targets = [{"ticker": c, "year": y} for c in (companies or [None]) for y in (years or [None])]
    return targets[:MAX_TARGETS]


def normalize_plan(raw: dict[str, Any] | None, query: str, *, ticker=None, year=None, source_type=None,
                   clarifier=None) -> dict[str, Any]:
    raw = raw if isinstance(raw, dict) else {}
    adjustments: list[str] = []
    proposed = [str(t).strip() for t in (raw.get("tools") or []) if isinstance(t, (str, bytes))]
    tools: list[str] = []
    for name in proposed:
        mapped = LEGACY_TOOLS.get(name, name)
        if mapped not in TOOLS:
            adjustments.append(f"dropped unknown tool {name!r}")
        elif mapped not in tools:
            tools.append(mapped)
    if "text_search" not in tools:
        tools.append("text_search")
        adjustments.append("added text_search: every answer must cite filing text")
    needs_calc = bool(raw.get("needs_calculation")) or bool(CALC_TERMS.search(query))
    if needs_calc and "calculator" not in tools:
        tools.append("calculator")
        adjustments.append("added calculator: the question asks for a change, ratio or comparison")
    targets = plan_targets(query, ticker, year, clarifier) if clarifier is not None else [{"ticker": ticker, "year": year}]
    if any(t["ticker"] or t["year"] for t in targets) and "coverage_lookup" not in tools:
        tools.append("coverage_lookup")
        adjustments.append("added coverage_lookup: the question names a company or year")
    mode = visual_mode()
    if mode == "always" and "visual_search" not in tools:
        tools.append("visual_search")
        adjustments.append("added visual_search: RAG_FIN_VISUAL=1 searches page images for every question")
    elif mode == "auto" and "visual_search" not in tools and VISUAL_TERMS.search(query):
        tools.append("visual_search")
        adjustments.append("added visual_search: the question asks about a chart or figure")
    elif mode == "off" and "visual_search" in tools:
        tools.remove("visual_search")
        adjustments.append("dropped visual_search: no ColPali index is served (RAG_FIN_VISUAL)")
    return {
        "intent": str(raw.get("intent") or "annual_report_qa"),
        "tools": [t for t in TOOL_ORDER if t in tools],
        "targets": targets,
        "source_type": raw.get("source_type") or source_type,
        "needs_calculation": needs_calc,
        "planner": "llm" if raw.get("tools") else "fallback",
        "llm_tools": proposed,
        "adjustments": adjustments,
    }


def target_label(target: dict[str, Any]) -> str:
    return " ".join(str(v) for v in (target.get("ticker"), target.get("year")) if v) or "all filings"


def coverage(targets: list[dict[str, Any]], directory: dict[str, list[int]]) -> dict[str, Any]:
    """Split targets into searchable ones and ones the index can't answer (an empty directory = unknown)."""
    if not directory:
        return {"known": False, "searchable": targets, "skipped": []}
    searchable, skipped = [], []
    for target in targets:
        ticker, year = target.get("ticker"), target.get("year")
        if ticker and ticker not in directory:
            skipped.append({**target, "reason": "company_not_in_index"})
        elif ticker and year and year not in directory[ticker]:
            skipped.append({**target, "reason": "year_not_in_index", "available_years": directory[ticker]})
        else:
            searchable.append(target)
    return {"known": True, "searchable": searchable, "skipped": skipped}


STOP_WORDS = {"what", "was", "were", "is", "are", "the", "a", "an", "of", "for", "in", "on", "to", "and", "did", "does",
              "do", "how", "much", "many", "which", "who", "tell", "me", "about", "please", "their", "its", "by", "from",
              "with", "between", "compare", "versus", "vs"}


def broaden_query(query: str, metric: str | None, metric_phrases: list[str]) -> str:
    """Retry query: the question's content words plus the metric's synonyms and statement names."""
    words = [w for w in re.findall(r"[A-Za-z0-9$%.,'-]+", query) if w.lower().strip("?.,") not in STOP_WORDS]
    extra = list(dict.fromkeys(metric_phrases))[:4] if metric else []
    return " ".join(words + extra + ["consolidated statements", "annual report"])


# ----------------------------------------------------------------------------- calculator

CALC_TOKEN = re.compile(r"CALC\[([^\]]{1,200})\]")
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
ALLOWED_CONSTANTS = {1.0, 2.0, 100.0, 1000.0, 1_000_000.0}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _eval(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    raise ValueError("only numbers, + - * / and parentheses are allowed")


def safe_calculate(expression: str) -> float:
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression.replace("$", "")).strip()
    return _eval(ast.parse(cleaned, mode="eval"))


def evidence_numbers(evidence: list[dict[str, Any]]) -> set[float]:
    found = set()
    for hit in evidence:
        for match in NUMBER.findall(str(hit.get("text") or hit.get("preview") or "")):
            try:
                found.add(float(match.replace(",", "")))
            except ValueError:
                pass
    return found


def run_calculations(answer: str, evidence: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Replace each CALC[...] with its value; record whether every operand came from the evidence."""
    known = evidence_numbers(evidence)
    records: list[dict[str, Any]] = []

    def replace(match):
        expression = match.group(1)
        operands = [float(n.replace(",", "")) for n in NUMBER.findall(expression)]
        ungrounded = [n for n in operands if n not in known and n not in ALLOWED_CONSTANTS]
        try:
            value = safe_calculate(expression)
        except (ValueError, SyntaxError, ZeroDivisionError) as exc:
            records.append({"expression": expression, "value": None, "error": str(exc), "grounded": False})
            return "[calculation failed]"
        records.append({"expression": expression, "value": round(value, 4), "grounded": not ungrounded,
                        "ungrounded_operands": ungrounded})
        return f"{value:,.2f}"

    return CALC_TOKEN.sub(replace, answer), records


def strip_calculations(answer: str) -> str:
    return CALC_TOKEN.sub("[not calculated]", answer)


# ----------------------------------------------------------------------------- support check (critic)

CHUNK_ID_TOKEN = re.compile(r"\S*_p\d{3,}\S*")              # citation IDs are not figures
# Dates and page references are not figures either: "December 31, 2025", "Dec. 31", "2025-12-31", "page 49", "pp. 49-50".
NOT_FIGURES = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?\b"
    r"|\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b"
    r"|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b(?:pages?|pp?\.)\s*\d+(?:\s*(?:-|–|,|and|to)\s*\d+)*", re.I)
SCALES = (1.0, 1e3, 1e-3, 1e6, 1e-6, 1e9, 1e-9)             # "$254.5 billion" vs "254,453" (millions)
VERDICTS = ("supported", "partially_supported", "unsupported", "unverified")


def _matches(value: float, decimals: int, known: set[float]) -> bool:
    """True when some known figure equals ``value`` at the precision the answer printed, in any unit scale."""
    tolerance = 0.5 * 10 ** -decimals
    return any(abs(value * scale - k) <= tolerance * scale + 1e-9 for k in known for scale in SCALES)


def numeric_grounding(answer: str, cited: list[dict[str, Any]], calculations: list[dict[str, Any]]) -> dict[str, Any]:
    """Every figure in the answer must appear in a cited chunk or come from the calculator.

    Years (1990-2049), small counts (<= 10), dates and page references are not checked. Rounding is allowed only to
    the precision the answer shows, in any unit scale (thousands, millions, billions), so "$254.5 billion" matches
    254,453 million.
    """
    calculated = {abs(float(c["value"])) for c in calculations if c.get("value") is not None}
    known = evidence_numbers(cited) | calculated
    confirmed = confirmed_numbers(cited) | calculated
    text = NOT_FIGURES.sub(" ", CHUNK_ID_TOKEN.sub(" ", CALC_TOKEN.sub(" ", answer)))
    checked, ungrounded, unverified = 0, [], []
    for raw in NUMBER.findall(text):
        raw = raw.rstrip(",")                                 # "revenue was 61,160, up 9%" -> "61,160"
        value = float(raw.replace(",", ""))
        if ("," not in raw and "." not in raw and 1990 <= value <= 2049) or (value <= 10 and value == int(value)):
            continue
        checked += 1
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if not _matches(value, decimals, known):
            ungrounded.append(raw)
        elif not _matches(value, decimals, confirmed):
            unverified.append(raw)
    return {"grounded": not ungrounded, "checked": checked, "ungrounded": ungrounded, "unverified": unverified}


def confirmed_numbers(evidence: list[dict[str, Any]]) -> set[float]:
    """Figures the cited chunks state, minus those only a VLM read: a chunk's ``unverified_numbers`` are figures its
    page reader wrote that neither the page's text layer nor OCR prints. Such a figure still counts as confirmed when
    another cited chunk states it without that flag (another page, or pipeline 2's text of the same page)."""
    found = set()
    for hit in evidence:
        flagged = set()
        for item in hit.get("unverified_numbers") or []:
            for match in NUMBER.findall(str(item)):
                try:
                    flagged.add(float(match.replace(",", "")))
                except ValueError:
                    pass
        found |= evidence_numbers([hit]) - flagged
    return found


def parse_support(text: str) -> dict[str, Any]:
    """The judge's JSON -> claims + verdict. Anything unparseable or claim-less is 'unverified' (fail closed)."""
    import json
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    claims = [c for c in data.get("claims") or [] if isinstance(c, dict)]
    if not claims:
        # A reply that starts JSON but doesn't parse was usually cut off by the output-token cap (hosted Qwen3-8B,
        # max 1,024 tokens, on an 11-claim fund answer).
        cut = bool(text and "{" in text and not data)
        return {"verdict": "unverified", "claims": [], "judge_cut_off": cut,
                "reason": "the support judge's reply was cut off or not valid JSON" if cut
                else "the support judge returned no claims"}
    unsupported = [c for c in claims if c.get("supported") is not True]
    stated = str(data.get("verdict") or "").strip().lower()
    if unsupported:
        verdict = "unsupported" if len(unsupported) == len(claims) else "partially_supported"
    else:
        verdict = "supported" if stated in ("", "supported") else "partially_supported"   # claims and verdict disagree
    return {"verdict": verdict, "claims": claims, "unsupported_claims": [str(c.get("claim", ""))[:300] for c in unsupported]}


# ---------------------------------------------------------------------------- parallel search fan-out
MAX_SEARCH_WORKERS = 8


def search_workers() -> int:
    """Searches a node may run at once: RAG_SEARCH_WORKERS (default 4, 1 = sequential). Embedded Qdrant (one in-process
    client, no QDRANT_URL) stays sequential; Chroma opens a handle per search and a Qdrant server takes concurrent
    queries. LLM calls are never fanned out: the hosted client sends one request at a time."""
    try:
        workers = int(os.getenv("RAG_SEARCH_WORKERS", "4"))
    except ValueError:
        workers = 4
    backend = os.getenv("RAG_BACKEND", "")
    if backend == "qdrant" or (backend in ("financial_qdrant", "financial_hybrid") and not os.getenv("QDRANT_URL")):
        return 1
    return max(1, min(workers, MAX_SEARCH_WORKERS))


def fan_out(fn, items: list, workers: int | None = None) -> tuple[list, dict]:
    """``[fn(item) for item in items]`` with up to ``workers`` running at once; results keep the items' order.

    The first item runs alone, so lazily loaded models and clients (query embedder, reranker) are created once before
    the rest run in parallel. Each task runs in a copy of the caller's context (trace IDs). The first exception is
    raised, as in a loop. Returns the results and a log entry for ``agent_run``."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context
    items = list(items)
    workers = search_workers() if workers is None else max(1, workers)
    started = time.perf_counter()
    if workers <= 1 or len(items) <= 2:
        results = [fn(item) for item in items]
        used = 1
    else:
        results = [fn(items[0])]
        used = min(workers, len(items) - 1)
        with ThreadPoolExecutor(max_workers=used, thread_name_prefix="rag-search") as pool:
            futures = [pool.submit(copy_context().run, fn, item) for item in items[1:]]
            results += [future.result() for future in futures]
    return results, {"tool": "search_fan_out", "searches": len(items), "workers": used,
                     "ms": round((time.perf_counter() - started) * 1000)}
