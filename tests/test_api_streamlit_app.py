import sys
import types

from scripts.common.companies import SECTOR_NAMES

sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))   # build_payload doesn't use Streamlit itself
from scripts.ui import api_streamlit_app as ui  # noqa: E402


def settings(**overrides):
    base = {"llm_provider": "mock", "embedding_provider": "hashing", "hash_dimensions": 384, "top_k": 5, "dense_top_k": 30,
            "bm25_top_k": 30, "source_type": "", "max_context_chars": 12000, "enable_guardrails": True,
            "enable_human_review": True, "enable_langsmith": False, "langsmith_project": "", "technique": "agentic_graph",
            "retrieval_mode": "hybrid", "ticker": "", "year": None, "sectors": [], "self_consistency_samples": 3}
    return {**base, **overrides}


def test_financial_payload_sends_the_chosen_sectors():
    payload = ui.build_payload("net interest income", settings(sectors=["Banking", "Oil and gas"]))
    assert payload["sector"] == "Banking,Oil and gas"
    assert ui.build_payload("q", settings())["sector"] is None
    assert ui.DOMAIN == "financial_report_qa" and not hasattr(ui, "DOMAINS")   # financial-only UI
    assert len(SECTOR_NAMES) == 13


def test_typewriter_sends_long_answers_in_few_pieces_and_keeps_the_text():
    long_text = "excerpt " * 4600 + "end"          # a withheld answer of many cited excerpts
    pieces = list(ui.typewriter(long_text, seconds=0))
    assert "".join(pieces) == long_text
    assert len(pieces) <= ui.TYPEWRITER_PIECES
    short = "Revenue was $1 million."
    assert "".join(ui.typewriter(short, seconds=0)) == short
