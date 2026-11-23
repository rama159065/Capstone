"""Chunker v2 for text-layer pages: line-end cuts, the table header carried into continuation chunks, and column
order for two-column prose pages. Not used by the ingest yet; ``scripts.evaluation.chunker_ab`` compares it with the
current chunker (``financial_ingest_qdrant_db.windows``) on separate test indexes.

Why: the current chunker cuts 1,200-character windows at a space with a 200-character overlap. A statement read from
the text layer is then split mid-table, and the continuation chunk has no year header and no scale: on the 509
filings, 18.6% of the statement figures (revenue, net income, total assets, ...) could not be read from the chunk
that stores them, 32.7% in banking. ``get_text(sort=True)`` also interleaves side-by-side prose columns (BAC 2025
p.129: "...has been audited by to the maintenance of records that..., PricewaterhouseCoopers, LLP").

Checked on every page of the 509 filings before any index was built (figures read in code with label_match):
readable from one chunk 81.4% -> 94.9%, every sector better, 0 of 3,936 figures lost or changed; column order is
applied only to pages with fewer than ``PROSE_MAX_FIGURES`` figures, as applying it to every page lost 19% of the
bank figures (full-width tables were split as if they were text columns).
"""
from __future__ import annotations

import re

CHUNKER_VERSION = "v2-lines-header-1"
CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 300              # a smaller chunk is joined to the next one
PROSE_MAX_FIGURES = 5              # column order only for pages with fewer figures (prose, not tables)
CONTINUED = "[continued]"

YEAR = re.compile(r"(?<!\d)(20[0-4]\d)(?!\d)")
FIGURE = re.compile(r"\(?\$?\s?\d{1,3}(?:,\d{3})+(?:\.\d+)?\)?|\b\d+\.\d+\b")
SCALE = re.compile(r"(?i)\bin\s+(thousands|millions|billions)\b")


def year_header(line: str) -> bool:
    """Two or more fiscal years and no figures: a table's column header."""
    return len(set(YEAR.findall(line))) >= 2 and not FIGURE.search(YEAR.sub(" ", line))


def column_text(page, text: str) -> str:
    """``text`` (the page's text layer) in column order when the page is two-column prose; otherwise unchanged.

    ``page`` is the PyMuPDF page. Text blocks wholly in the left half are read top to bottom, then the right half;
    blocks spanning both halves (headings, full-width tables) come first. Pages with ``PROSE_MAX_FIGURES`` or more
    figures keep their order, so tables are never reordered."""
    if len(FIGURE.findall(text)) >= PROSE_MAX_FIGURES:
        return text
    blocks = [b for b in page.get_text("blocks") if str(b[4]).strip()]
    mid = page.rect.width / 2
    left = [b for b in blocks if b[2] <= mid + 5]
    right = [b for b in blocks if b[0] >= mid - 5]
    if len(left) < 3 or len(right) < 3:
        return text
    full = [b for b in blocks if b not in left and b not in right]
    ordered = sorted(full, key=lambda b: b[1]) + sorted(left, key=lambda b: b[1]) + sorted(right, key=lambda b: b[1])
    out = "\n".join(re.sub(r"\s*\n\s*", " ", str(b[4])).strip() for b in ordered)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", out)).strip()


def _split_long(line: str, size: int) -> list[str]:
    """A single line longer than a chunk, cut at spaces."""
    parts, start = [], 0
    while start < len(line):
        end = min(len(line), start + size)
        if end < len(line):
            cut = line.rfind(" ", start, end)
            end = cut if cut > start + size // 2 else end
        parts.append(line[start:end].strip())
        start = end
    return [p for p in parts if p]


def chunks(text: str, title: str = "", size: int = CHUNK_CHARS) -> list[str]:
    """Chunks cut at line ends only. A chunk that starts inside a figure table repeats ``title`` and the table's
    year header (with its scale), marked ``[continued]``. Chunks under ``MIN_CHUNK_CHARS`` join the next one."""
    lines = [piece for line in text.splitlines() if line.strip() for piece in _split_long(line.strip(), size)]
    out: list[list[str]] = []
    current: list[str] = []
    header = scale = None
    for line in lines:
        found = SCALE.search(line)
        if found:
            scale = found.group(0)
        if year_header(line):
            header = line
        if current and sum(len(x) + 1 for x in current) + len(line) > size:
            out.append(current)
            current = []
            if header and FIGURE.search(line) and not year_header(line):
                carried = header if (not scale or scale.lower() in header.lower()) else f"{header} ({scale})"
                current = [f"{CONTINUED} {title}".strip(), carried]
        current.append(line)
    if current:
        out.append(current)
    joined = ["\n".join(c) for c in out]
    merged: list[str] = []
    for i, chunk in enumerate(joined):
        if merged and len(merged[-1]) < MIN_CHUNK_CHARS and len(merged[-1]) + len(chunk) <= size + MIN_CHUNK_CHARS:
            merged[-1] = merged[-1] + "\n" + chunk
        else:
            merged.append(chunk)
    return merged
