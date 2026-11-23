"""Near-miss checks: is a cited figure the one the question asks about?

The figure check (``financial_tools.numeric_grounding``) proves a figure is printed in a cited chunk and the LLM judge
reads the claims, but a figure from the wrong row passes both: on the 23 Nov scenario run Qwen3-8B answered Citi's
"U.S. Personal Banking" net income with another segment's, Chevron's worldwide crude oil with "U.S. ... crude oil and
natural gas liquids", and NVIDIA's peer-chart revenue with the axis label "$250". These checks are in code, run in the
critic after the judge, and can only withhold an answer (one repair first), never approve one:

* ``subject``: "U.S. Personal Banking net income", "data center revenue": the words that narrow the measure must be
  named on the page of every figure's chunk.
* ``region`` / ``product``: a question that names no region is company-wide; a question about crude oil is not answered
  by a row about crude oil **and** natural gas liquids.
* ``axis``: a chart's axis label is not a value of the chart.
* ``term``: "how does X define conduct risk" must cite a chunk that contains "conduct risk".
* ``parent``: a parent-company-only statement is not the company's.

``clean_chart_tables`` drops what a vision model wrote for bars the chart doesn't label (the same value for every year,
or all zeros). It edits the retrieved copy only, never the index.
"""
from __future__ import annotations

import re
from typing import Any

from scripts.agents import financial_tools as tools

# Words that may stand between the company and the measure without narrowing it.
_PLAIN = frozenset("""total net consolidated annual reported overall full fiscal year year's yearly company company's
companies its their his the a an of for in on at to and or by from with as is was were are be been did do does has have
had how what which who when much many more less most least higher lower highest lowest larger bigger greater than between
compare compared compares versus vs show give tell me list report reports reporting state states say says growth change
changed grow grew trend trended over across during last latest current recent past years gaap per amount figure figures
value combined that this these those it they about please quarter full-year year-over-year yoy any all each every both
made make earn earned earning generate generated record recorded if i we us our income statement sheet balance cash flow
financial same group sector sector's industry industry's holdings holding top fund fund's etf portfolio""".split())
_SEGMENT_PHRASE = re.compile(r"\b(?:from|in|for|of|by)\s+(?:the\s+|its\s+)?([A-Za-z0-9&.,'\- ]{2,50}?)\s+"
                             r"(?:segment|division|business unit|business|unit|region)\b", re.I)
_YEARISH = re.compile(r"^(?:fy)?\d{2,4}(?:s|['’]s)?$|^q[1-4]$", re.I)


# Measures a qualifier doesn't narrow ("independent auditor").
_NOT_NARROWED = frozenset(("auditor", "employees", "segment revenue"))


def _blank_names(query: str, aliases: list[str], sectors: list[str]) -> str:
    """The query with company and sector names replaced by " . ", so they end the walk back from the measure. A sector
    word after a capitalised one is part of a name ("U.S. Personal Banking"), not the sector."""
    text = query
    for alias in sorted(aliases, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z]){re.escape(alias)}(?:['’]s|['’])?(?![A-Za-z])", " . ", text, flags=re.I)

    def blank(match):
        before = text[:match.start()].split()
        named = match.group(0)[:1].isupper() and len(before) > 1 and before[-1][:1].isupper()
        return match.group(0) if named else " . "
    for phrase in sorted(sectors, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z]){re.escape(phrase)}(?![A-Za-z])", blank, text, flags=re.I)
    return text


def subject(query: str, metric: str | None, metric_phrases: list[str], aliases: list[str], sectors: list[str] = (),
            tickers: list[str] = ()) -> str | None:
    """The words that narrow the measure ("U.S. Personal Banking" net income, "data center" revenue, revenue from the
    "Domestic Card" segment), or None when the question asks for the company's own figure."""
    if not metric or metric in _NOT_NARROWED:
        return None
    text = _blank_names(query, aliases, list(sectors))
    for ticker in tickers:
        text = re.sub(rf"\b{re.escape(ticker)}\b", " . ", text)
    found = _SEGMENT_PHRASE.search(text)
    if found and not any(w.lower() in ("entire", "whole") for w in found.group(1).split()):
        words = [w for w in found.group(1).split() if w.strip(".,'’") and w.lower().strip(".,'’") not in _PLAIN]
        if words:
            return " ".join(words)
    best = None
    for phrase in metric_phrases:
        match = re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", text.lower())
        if match and (best is None or len(phrase) > best[1] - best[0]):
            best = (match.start(), match.end())
    if best is None:
        return None
    words: list[str] = []
    for word in reversed(text[:best[0]].split()):
        bare = word.strip(",;:()\"").lower()
        if word.endswith((",", ";", ":")) and not words:
            break
        if not bare or bare in (".", "&") or _YEARISH.match(bare):
            break
        if bare.rstrip(".") in _PLAIN:
            if words or bare not in ("total", "net", "consolidated", "annual", "reported", "overall", "fiscal", "full-year"):
                break
            continue
        if not re.match(r"^[a-z0-9][a-z0-9.&'’\-/]*$", bare) or bare.endswith(("'s", "’s")):
            break
        words.insert(0, word.strip(",;:()\""))
        if len(words) >= 5:
            break
    return " ".join(words) or None


_WORD = re.compile(r"[a-z0-9]+")


def names(text: str, phrase: str) -> bool:
    """The text has the phrase's words together, in any case and punctuation ("U.S. PERSONAL BANKING")."""
    squash = lambda s: " ".join(_WORD.findall(s.lower().replace("u.s.", "us").replace("united states", "us")))  # noqa: E731
    return f" {squash(phrase)} " in f" {squash(text)} "


def _page(hit: dict[str, Any]) -> tuple:
    return (hit.get("ticker"), hit.get("year"), hit.get("page"))


def pages_naming(evidence: list[dict[str, Any]], phrase: str) -> set[tuple]:
    return {_page(h) for h in evidence if names(str(h.get("text") or ""), phrase)}


_FIGURE = tools.NUMBER


def answer_figures(answer: str) -> list[tuple[str, float, int]]:
    """(as printed, value, decimals) of the answer's figures, with years, small counts, dates and IDs left out."""
    text = tools.NOT_FIGURES.sub(" ", tools.CHUNK_ID_TOKEN.sub(" ", answer))
    out = []
    for raw in _FIGURE.findall(text):
        raw = raw.rstrip(",")
        value = float(raw.replace(",", ""))
        if ("," not in raw and "." not in raw and 1990 <= value <= 2049) or (value <= 10 and value == int(value)):
            continue
        out.append((raw, value, len(raw.split(".")[1]) if "." in raw else 0))
    return out


def is_heading(line: str) -> bool:
    """A short title-case line without figures ("U.S. Upstream", "| **International Upstream** | | |")."""
    stripped = line.strip(" |*#:\t")
    words = stripped.split()
    if not words or len(words) > 8 or stripped.endswith((".", ",", ";")):
        return False
    if any(not 1990 <= float(n.replace(",", "")) <= 2049 for n in _FIGURE.findall(stripped)):
        return False
    return all(w[:1].isupper() or not w[:1].isalpha() or len(w) <= 3 for w in words)


def _lines_with(hit: dict[str, Any], value: float, decimals: int, before: str = "") -> list[tuple[str, str]]:
    """(line, nearest heading above it) for every line of the chunk that prints the figure. ``before`` is the page's
    text above the chunk: a paragraph that continues from the previous chunk is under that chunk's heading."""
    out, heading = [], next((line.strip(" |*#:\t") for line in reversed(before.splitlines()) if is_heading(line)), "")
    for line in str(hit.get("text") or "").splitlines():
        numbers = {float(n.replace(",", "")) for n in _FIGURE.findall(line)}
        if tools._matches(value, decimals, numbers):
            out.append((f"{hit['chart_title']}: {line}" if hit.get("chart_title") else line, heading))
        elif is_heading(line):
            heading = line.strip(" |*#:\t")
    return out


_REGIONS = {"U.S.": re.compile(r"\bU\.?S\.?(?=\W|$)(?!\s*(?:dollars?|\$|GAAP))|\bUnited\s+States\b|\bdomestic\b", re.I),
            "international": re.compile(r"\binternational\b|\bforeign\b", re.I)}
# One product's row is not another's, and a row that adds two products is neither.
_PRODUCTS = (("natural gas liquids", re.compile(r"\bnatural\s+gas\s+liquids\b|\bNGLs?\b", re.I)),
             ("natural gas", re.compile(r"\bnatural\s+gas\b(?!\s+liquids)", re.I)),
             ("crude oil", re.compile(r"\bcrude(?:\s+oil)?\b", re.I)),
             ("oil-equivalent", re.compile(r"\boil[- ]equivalent\b|\bBOE\b", re.I)))


def products_in(text: str) -> set[str]:
    return {name for name, pattern in _PRODUCTS if pattern.search(text)}


def scope_problem(query: str, line: str, heading: str) -> str | None:
    """Why this row is not what the question asks (a region or product it doesn't ask for), or None."""
    asked = {name for name, pattern in _REGIONS.items() if pattern.search(query)}
    row = {name for name, pattern in _REGIONS.items() if pattern.search(line)}
    under = {name for name, pattern in _REGIONS.items() if pattern.search(heading)}
    if not asked and len(row or under) == 1:
        return f"it is the {next(iter(row or under))} figure and the question asks for the whole company"
    if len(asked) == 1 and len(row or under) == 1 and asked != (row or under):
        return f"it is the {next(iter(row or under))} figure, not the {next(iter(asked))} one"
    wanted, printed = products_in(query), products_in(line)
    if wanted and printed and wanted != printed and len(wanted) == 1:
        return f"its row is {' and '.join(sorted(printed))}, not {next(iter(wanted))}"
    return None


CHART_WORDS = re.compile(r"\b(chart|graph|figure|diagram|infographic|plot)s?\b", re.I)


def axis_ticks(text: str) -> set[float]:
    """Axis labels printed in a chart page's text: bare-number lines that count up from 0 in equal steps (0, 50, ...)."""
    bare = set()
    for line in text.splitlines():
        match = re.fullmatch(r"\s*\$?\s*(\d[\d,]*(?:\.\d+)?)\s*(?:%|billion|million)?\s*", line)
        if match:
            bare.add(float(match.group(1).replace(",", "")))
    ticks: set[float] = set()
    if 0.0 in bare:
        for step in sorted(v for v in bare if v > 0):
            run = []
            while step * (len(run) + 1) in bare:
                run.append(step * (len(run) + 1))
            if len(run) >= 4:
                ticks |= set(run)
    return ticks


RANKS = re.compile(r"\b(higher|lower|larger|smaller|greater|bigger|more than|less than|exceed(?:s|ed)?|highest|lowest|outperform\w*)\b", re.I)


def ranked_without_figure(answer: str, companies: dict[str, str], cited: list[dict[str, Any]]) -> list[str]:
    """A ranking of the compared companies (ticker -> name) needs a cited chunk of each: "Walmart's 284,668 is clearly
    higher than Home Depot's", with no Home Depot figure, is a guess."""
    if len(companies) < 2 or not RANKS.search(answer):
        return []
    have = {str(h.get("ticker") or "").upper() for h in cited}
    return [f"no cited chunk of {name} ({ticker}); give its figure or say the comparison can't be made"
            for ticker, name in companies.items() if ticker not in have]


DEFINE = re.compile(r"\b(?:define[sd]?|definition\s+of|meaning\s+of)\s+(?:the\s+term\s+)?[\"“']?([A-Za-z][A-Za-z\- ]{2,60}?)"
                    r"[\"”']?(?=\s+(?:and|in|as|for|or|under|within)\b|[?.,;]|$)", re.I)


def defined_term(query: str) -> str | None:
    match = DEFINE.search(query)
    return match.group(1).strip() if match else None


def defines(text: str, term: str) -> bool:
    """The text states what the term is ("Conduct risk is the risk of ...")."""
    return bool(re.search(rf"\b{re.escape(term)}\s+(?:is|are|means|refers\s+to|is\s+defined\s+as)\b", text, re.I))


def problems(query: str, answer: str, cited: list[dict[str, Any]], evidence: list[dict[str, Any]],
             narrowed_by: str | None, parent_only, text_before=lambda hit: "") -> list[str]:
    """What is wrong with the answer's figures, as notes the answer agent can act on; empty when nothing is."""
    notes: list[str] = []
    figures = answer_figures(answer)
    term = defined_term(query)
    if term:
        holding = [str(h.get("chunk_id")) for h in evidence if names(str(h.get("text") or ""), term)]
        defining = [h for h in evidence if defines(str(h.get("text") or ""), term)]
        if defining and not any(defines(str(h.get("text") or ""), term) for h in cited):
            notes.append(f"'{term}' is defined in {', '.join(str(h.get('chunk_id')) for h in defining[:2])}; give that definition")
        elif not any(names(str(h.get("text") or ""), term) for h in cited):
            notes.append(f"no cited chunk contains the term '{term}'"
                         + (f"; these chunks do: {', '.join(holding[:3])}" if holding else
                            "; say the evidence does not define it"))
    if narrowed_by:
        named = pages_naming(evidence, narrowed_by)
        inside = [h for h in cited if _page(h) in named]
        chunks = [str(h.get("chunk_id")) for h in evidence if _page(h) in named][:4]
        for raw, value, decimals in figures:
            anywhere = any(_lines_with(h, value, decimals) for h in cited)
            if anywhere and not any(_lines_with(h, value, decimals) for h in inside):
                notes.append(f"{raw} is not from a page about '{narrowed_by}'"
                             + (f"; use these chunks: {', '.join(chunks)}" if chunks else ""))
        if not named:
            notes.append(f"no evidence page names '{narrowed_by}'")
    # A scope problem counts when the answer has no figure without one (a breakdown after the right figure is fine).
    scoped, free = [], 0
    for raw, value, decimals in figures:
        rows = [(h, line, heading) for h in cited for line, heading in _lines_with(h, value, decimals, text_before(h))]
        if not rows:
            continue
        reasons = [scope_problem(query, line, heading) for _, line, heading in rows]
        if all(reasons):
            scoped.append(f"{raw}: {reasons[0]}")
        else:
            free += 1
    if scoped and not free and not narrowed_by:
        notes += scoped[:3]
    if CHART_WORDS.search(query):
        for raw, value, decimals in figures:
            hits = [h for h in cited if _lines_with(h, value, decimals)]
            if hits and all(value in axis_ticks(str(h.get("text") or "")) for h in hits):
                notes.append(f"{raw} is an axis label of the chart, not a value it shows")
    if not re.search(r"\bparent\b", query, re.I):
        for raw, value, decimals in figures:
            hits = [h for h in cited if _lines_with(h, value, decimals)]
            if hits and all(parent_only(str(h.get("text") or "")) for h in hits):
                notes.append(f"{raw} is from a parent-company-only statement; use the consolidated statement")
    return list(dict.fromkeys(notes))[:5]


# ----------------------------------------------------------------------------- chart tables read by a vision model
_ROW_YEAR = re.compile(r"^\|\s*(?:'|20)?(\d{2}|20\d{2})\s*\|")


def clean_chart_text(text: str) -> str:
    """A chart table whose value repeats for three or more years keeps its last year only (the model copied the one
    label the chart prints); a table of zeros keeps none (the model invented bars without labels)."""
    lines = text.splitlines()
    rows = [(i, [c.strip() for c in line.strip().strip("|").split("|")]) for i, line in enumerate(lines)
            if _ROW_YEAR.match(line.strip())]
    drop: set[int] = set()
    if len(rows) >= 3:
        values = [cells[1:] for _, cells in rows]
        if all(v == values[0] for v in values[1:]) and any(re.search(r"\d", c) for c in values[0]):
            drop = {i for i, _ in rows[:-1]}
    body = [(i, [c.strip() for c in line.strip().strip("|").split("|")]) for i, line in enumerate(lines)
            if line.strip().startswith("|") and not set(line.strip()) <= set("|-: ")]
    cells = [c for _, r in body[1:] for c in r[1:]]
    numeric = [c for c in cells if re.fullmatch(r"\(?-?[\d.,]+\)?", c)]
    if len(numeric) >= 6 and all(float(re.sub(r"[^\d.]", "", c) or 0) == 0 for c in numeric if re.sub(r"[^\d.]", "", c)):
        drop |= {i for i, r in body[1:] if any(c in numeric for c in r[1:])}
    if not drop:
        return text
    kept = [line for i, line in enumerate(lines) if i not in drop]
    return "\n".join([*kept, "(Values the chart does not label were left out.)"])


def off_measure(answer: str, cited: list[dict[str, Any]], phrases: list[str]) -> list[str]:
    """Figures of an answer about a statement measure that no line near them names ("Balance at January 31, 2024 ...
    59,646" for total stockholders' equity), when none of the answer's figures has such a line."""
    words = [p.lower().replace("’", "'") for p in phrases]
    bad, named = [], 0
    for raw, value, decimals in answer_figures(answer):
        found = False
        rows = 0
        for hit in cited:
            lines = [line.lower().replace("’", "'") for line in str(hit.get("text") or "").splitlines()]
            for i, line in enumerate(lines):
                if not tools._matches(value, decimals, {float(n.replace(",", "")) for n in _FIGURE.findall(line)}):
                    continue
                rows += 1
                near = " ".join(lines[max(0, i - 2):i + 2])
                found = found or any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", near) for w in words)
        if rows:
            named += found
            if not found:
                bad.append(raw)
    return [] if named else bad[:3]


_FENCE = re.compile(r"^\s*```\w*\s*$", re.M)


def clean_chart_tables(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for hit in hits:
        if "```" in str(hit.get("text") or "") and str(hit.get("reader") or "").lower() in ("qwen3vl", "paidvl", "vlm"):
            # The page reader wrapped a chart's axis in a Markdown code fence, which the injection guard reads as a code
            # payload (Chevron 2025 p20). The fence is the reader's, not the filing's; the text inside is still checked.
            hit = {**hit, "text": _FENCE.sub("", str(hit["text"]))}
        if hit.get("content_type") == "table" and str(hit.get("reader") or "").lower() in ("qwen3vl", "paidvl", "vlm"):
            text = clean_chart_text(str(hit.get("text") or ""))
            if text != hit.get("text"):
                hit = {**hit, "text": text, "preview": re.sub(r"\s+", " ", text)[:500]}
        out.append(hit)
    return out
