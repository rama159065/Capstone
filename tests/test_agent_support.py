"""The critic's support check: every figure must be in a cited chunk, every claim judged supported by the
evidence. Only a verified answer is shown as generated prose with a hosted LLM; otherwise excerpts."""
from __future__ import annotations

import json

import pytest

import scripts.agents.financial_prompting_agent as agent
from scripts.agents import financial_clarifier as fc
from scripts.agents import financial_tools as tools

DIRECTORY = {"COST": [2024, 2025]}
CHUNK = "annual_report_edgar_ars_COST_2025_p0041_text0"
EVIDENCE_TEXT = "Costco net sales were 254,453 million in fiscal 2025 and 237,710 million in fiscal 2024."


def config(query="What was Costco revenue in 2025?"):
    return agent.AgentConfig(query=query, ticker=None, year=None, source_type=None, top_k=4, dense_top_k=4,
                             bm25_top_k=4, retrieval_mode="hybrid", embedding_provider="hashing", hash_dimensions=16,
                             max_context_chars=6000, self_consistency_samples=1)


def hit(chunk_id=CHUNK, text=EVIDENCE_TEXT):
    return {"chunk_id": chunk_id, "ticker": "COST", "year": 2025, "page": 41, "source_type": "annual_report_pdf",
            "text": text, "preview": text[:80], "metadata": {"chunk_id": chunk_id}}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setattr(fc, "default_clarifier", lambda: fc.Clarifier(directory=DIRECTORY))
    monkeypatch.delenv("RAG_FIN_VISUAL", raising=False)
    monkeypatch.delenv("RAG_BACKEND", raising=False)
    monkeypatch.delenv("RAG_SUPPORT_CHECK", raising=False)
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")
    monkeypatch.setattr(agent, "retrieve_evidence", lambda run_config: [hit()])


class ScriptedLLM(agent.MockLLM):
    """Answers and judges from scripts (one entry per call); falls back to the mock for other prompts."""

    def __init__(self, answers=(), verdicts=()):
        self.answers, self.verdicts, self.prompts = list(answers), list(verdicts), []

    def invoke(self, messages):
        joined = "\n".join(getattr(m, "content", str(m)) for m in messages)
        self.prompts.append(joined)
        if "Support check JSON" in joined and self.verdicts:
            supported = self.verdicts.pop(0)
            return agent.MockResponse(json.dumps({
                "claims": [{"claim": "Costco net sales were 254,453 million in 2025", "chunk_ids": [CHUNK],
                            "supported": supported, "reason": "stated on page 41" if supported else "not in the evidence"}],
                "verdict": "supported" if supported else "unsupported"}))
        if "Answer with citations." in joined and self.answers:
            return agent.MockResponse(self.answers.pop(0))
        return super().invoke(messages)


# ----------------------------------------------------------------------------- numeric grounding

def test_figures_must_appear_in_cited_chunks_at_the_printed_precision():
    cited = [hit()]
    ok = tools.numeric_grounding(f"Net sales were $254.5 billion, up from 237,710 million [{CHUNK}].", cited, [])
    assert ok["grounded"] and ok["checked"] == 2                     # 254.5 bn = 254,453 m at 1 decimal
    wrong = tools.numeric_grounding(f"Net sales were $255 billion [{CHUNK}].", cited, [])
    assert not wrong["grounded"] and wrong["ungrounded"] == ["255"]
    years_and_ids = tools.numeric_grounding(f"In fiscal 2025 the 3 segments grew [{CHUNK}].", cited, [])
    assert years_and_ids == {"grounded": True, "checked": 0, "ungrounded": []}
    calc = tools.numeric_grounding("Growth was 7.04%.", cited, [{"value": 7.0434, "grounded": True}])
    assert calc["grounded"]


def test_judge_output_is_parsed_fail_closed():
    assert tools.parse_support("not json")["verdict"] == "unverified"
    assert tools.parse_support('{"claims": [], "verdict": "supported"}')["verdict"] == "unverified"
    mixed = tools.parse_support('{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}],'
                                ' "verdict": "supported"}')
    assert mixed["verdict"] == "partially_supported" and mixed["unsupported_claims"] == ["b"]


# ----------------------------------------------------------------------------- critic in the graph

def test_supported_answer_passes_the_critic_first_time():
    llm = ScriptedLLM(answers=[f"Costco net sales were 254,453 million in fiscal 2025 [{CHUNK}]."], verdicts=[True])
    result = agent.agentic_graph_prompt(llm, config())
    assert result["support_check"]["verdict"] == "supported" and result["support_check"]["numeric"]["grounded"]
    assert result["agent_run"]["repairs"] == 0 and result["final_status"] == "answered"


def test_unsupported_claim_is_repaired_with_the_judges_feedback():
    llm = ScriptedLLM(answers=[f"Costco net sales doubled in 2025 [{CHUNK}].",
                               f"Costco net sales were 254,453 million in fiscal 2025 [{CHUNK}]."],
                      verdicts=[False, True])
    result = agent.agentic_graph_prompt(llm, config())
    assert result["agent_run"]["repairs"] == 1 and result["support_check"]["verdict"] == "supported"
    repair_prompt = next(p for p in llm.prompts if "Critic feedback" in p)
    assert "not stated by the cited evidence" in repair_prompt and CHUNK in repair_prompt


def test_invented_figure_fails_even_when_the_judge_agrees():
    llm = ScriptedLLM(answers=[f"Costco net sales were 300,000 million in fiscal 2025 [{CHUNK}]."] * 2,
                      verdicts=[True, True])
    result = agent.agentic_graph_prompt(llm, config())
    support = result["support_check"]
    assert support["verdict"] == "partially_supported" and support["numeric"]["ungrounded"] == ["300,000"]
    assert result["agent_run"]["repairs"] == 1                          # tried once, still wrong
    assert result["final_status"] == "pending_human_review"
    assert "unsupported_claims" in result["human_review"]["triggers"]
    assert "300,000" in next(p for p in llm.prompts if "Critic feedback" in p)


def test_support_check_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("RAG_SUPPORT_CHECK", "0")
    llm = ScriptedLLM(answers=[f"Costco net sales were 300,000 million [{CHUNK}]."])
    result = agent.agentic_graph_prompt(llm, config())
    assert result["support_check"] is None and not any("Support check JSON" in p for p in llm.prompts)


# ----------------------------------------------------------------------------- what a hosted LLM shows

def hosted(monkeypatch, llm):
    monkeypatch.setattr(agent, "get_llm", lambda *args, **kwargs: llm)
    return agent.run_technique("agentic_graph", "hosted", config())


def test_hosted_answer_is_shown_only_when_verified(monkeypatch):
    answer = f"Costco net sales were 254,453 million in fiscal 2025 [{CHUNK}]."
    result = hosted(monkeypatch, ScriptedLLM(answers=[answer], verdicts=[True]))
    assert result["answer"] == answer and result["final_status"] == "answered"
    assert result["grounding"] == {"method": "llm_support_check", "semantic_answer_verified": True}


def test_hosted_answer_falls_back_to_excerpts_when_unverified(monkeypatch):
    result = hosted(monkeypatch, ScriptedLLM(answers=[f"Costco sales doubled [{CHUNK}]."] * 2, verdicts=[False, False]))
    assert result["final_status"] == "evidence_only" and result["answer"].startswith("Unverified generation withheld")
    assert result["grounding"] == {"method": "source_excerpts", "semantic_answer_verified": False}
    assert result["support_check"]["verdict"] == "unsupported"


def test_single_prompt_techniques_still_return_excerpts_with_a_hosted_llm(monkeypatch):
    monkeypatch.setattr(agent, "get_llm", lambda *args, **kwargs: ScriptedLLM())
    result = agent.run_technique("baseline_rag", "hosted", config())
    assert result["final_status"] == "evidence_only" and not result["grounding"]["semantic_answer_verified"]


def test_api_grounding_policy_keeps_the_support_verdict(monkeypatch):
    from scripts.api import api_app
    answer = f"Costco net sales were 254,453 million in fiscal 2025 [{CHUNK}]."
    result = api_app.apply_grounding_policy(hosted(monkeypatch, ScriptedLLM(answers=[answer], verdicts=[True])))
    grounding = result["grounding"]
    assert grounding["status"] == "grounded" and grounding["semantic_answer_verified"] is True
    assert grounding["method"] == "llm_support_check" and "support check" in grounding["reason"]
    withheld = api_app.apply_grounding_policy({"answer": "x", "final_status": "answered", "evidence": [hit()],
                                               "citation_check": {"valid": False, "cited_chunk_ids": []},
                                               "grounding": {"method": "llm_support_check",
                                                             "semantic_answer_verified": True}})
    assert withheld["grounding"]["status"] == "not_grounded" and withheld["grounding"]["semantic_answer_verified"] is False
