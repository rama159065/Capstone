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
    payload = ui.build_payload("financial_report_qa", "net interest income",
                               settings(sectors=["Banking", "Oil and gas"]))
    assert payload["sector"] == "Banking,Oil and gas"
    assert ui.build_payload("financial_report_qa", "q", settings())["sector"] is None
    assert "sector" not in ui.build_payload("hr_policy", "q", settings(sectors=["Banking"]))   # financial only
    assert len(SECTOR_NAMES) == 13
