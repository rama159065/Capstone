"""term_id values are stored in every BM25 sparse vector: they must never change."""
import pytest

from scripts.rag import financial_ingest_qdrant_db as fi
from scripts.rag import hashing, multimodal_rag


@pytest.mark.parametrize("term,expected", [("revenue", 924810818), ("391,035", 1303547064), ("$6.08", 2527192811)])
def test_term_ids_are_pinned(term, expected):
    assert hashing.term_id(term) == expected


def test_dense_is_deterministic_and_normalised():
    vector = hashing.dense("Net income was $93,736 million in fiscal 2025.")
    assert len(vector) == 384 and abs(sum(x * x for x in vector) - 1) < 1e-9
    assert vector == hashing.dense("net INCOME was $93,736 million in fiscal 2025")
    assert hashing.dense("") == [0.0] * 384


def test_financial_pipeline_does_not_need_the_multimodal_module_for_hashing():
    assert fi.term_id is hashing.term_id and fi.dense is hashing.dense
    assert multimodal_rag.term_id is hashing.term_id and multimodal_rag.dense is hashing.dense   # re-exported
    assert "multimodal_rag" not in open(fi.__file__, encoding="utf-8").read().split('"""', 2)[2].split("\n\n\n")[0]
