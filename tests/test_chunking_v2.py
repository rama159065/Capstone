"""Chunker v2: line-end cuts, carried table header, prose-only column order."""
from scripts.agents import financial_sector_trend as st
from scripts.rag import chunking_v2 as cv

STATEMENT = ("Bank of America Corporation and Subsidiaries\nConsolidated Statement of Income\n"
             "(In millions, except per share information) 2024 2023 2022\n"
             + "\n".join(f"Line item {i} {i},{i:03d} {i},{i + 1:03d} {i},{i + 2:03d}" for i in range(1, 60))
             + "\nNet income $ 27,132 $ 26,515 $ 27,528\nNet income applicable to common shareholders $ 25,503 $ 24,866 $ 26,015")


def test_chunks_cut_at_line_ends_and_carry_the_table_header():
    parts = cv.chunks(STATEMENT, title="BAC 2024 Consolidated Statement of Income")
    assert len(parts) >= 2
    for part in parts:
        assert all(not line.endswith("applicable") for line in part.splitlines())       # no row cut in half
    last = parts[-1]
    assert last.startswith(f"{cv.CONTINUED} BAC 2024 Consolidated Statement of Income")
    assert "(In millions, except per share information) 2024 2023 2022" in last
    assert st.label_match([{"chunk_id": "x", "text": last}], "net income", 2024)["value"] == "27,132"


def test_prose_is_not_given_a_header_and_tiny_chunks_are_joined():
    prose = "Short heading\n" + "\n".join("A sentence about risk management without any figures." for _ in range(60))
    parts = cv.chunks(prose)
    assert all(cv.CONTINUED not in p for p in parts)
    assert all(len(p) >= cv.MIN_CHUNK_CHARS for p in parts[:-1])


class Page:
    class Rect:
        width = 600

    rect = Rect()

    def __init__(self, blocks):
        self.blocks = blocks

    def get_text(self, kind):
        return self.blocks


def test_column_order_only_for_prose_pages():
    left = [(50, 100 + 20 * i, 280, 115 + 20 * i, f"left {i}") for i in range(4)]
    right = [(320, 100 + 20 * i, 560, 115 + 20 * i, f"right {i}") for i in range(4)]
    page = Page(left + right)
    interleaved = "\n".join(f"left {i} right {i}" for i in range(4))
    assert cv.column_text(page, interleaved).splitlines() == [f"left {i}" for i in range(4)] + [f"right {i}" for i in range(4)]
    table = interleaved + "\n" + "\n".join(f"Row {i} 1,{i:03d} 2,{i:03d}" for i in range(5))
    assert cv.column_text(page, table) == table                                        # figures: never reordered
