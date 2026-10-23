from __future__ import annotations

import json

import scripts.evaluate_financial_rag as eval_rag


def write_jsonl(path, rows) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_phrase_and_reciprocal_rank_helpers() -> None:
    text = "Tiny. Net sales increased because product demand was strong across services and devices."

    assert eval_rag.normalize_space("A\n  B") == "A B"
    assert "Net sales" in eval_rag.pick_phrase(text)
    assert eval_rag.reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert eval_rag.reciprocal_rank(["a"], {"z"}) == 0.0


def test_make_silver_query_for_source_types() -> None:
    base_row = {
        "chunk_id": "chunk-1",
        "text": "Revenue table information with enough words for phrase picking.",
        "metadata": {
            "company": "Apple Inc.",
            "ticker": "AAPL",
            "year": 2024,
            "page": 12,
            "source_type": "chart_summary",
        },
    }

    query = eval_rag.make_silver_query(base_row, "q1")

    assert query["query_id"] == "q1"
    assert query["ticker"] == "AAPL"
    assert query["relevant_chunk_ids"] == ["chunk-1"]
    assert "chart or figure" in query["query"]


def test_generate_silver_eval_uses_chunk_metadata(tmp_path) -> None:
    chunks = tmp_path / "chunks.jsonl"
    output = tmp_path / "eval.json"
    rows = [
        {
            "chunk_id": "annual-1",
            "text": "Net sales and product demand are discussed in this annual report chunk. " * 8,
            "metadata": {
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "year": 2024,
                "page": 3,
                "source_type": "annual_report_pdf",
            },
        },
        {
            "chunk_id": "table-1",
            "text": "Table extracted from annual report with revenue values.",
            "metadata": {
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "year": 2024,
                "page": 4,
                "source_type": "table_summary",
            },
        },
    ]
    write_jsonl(chunks, rows)

    generated = eval_rag.generate_silver_eval(chunks, output, num_queries=2)

    assert output.exists()
    assert len(generated) == 2
    assert {item["relevant_chunk_ids"][0] for item in generated} == {"annual-1", "table-1"}
