"""Diagnostic tests documenting current complex-query support and limitations.

Model calls are deliberately mocked: these verify evidence transport and guards,
not vision recognition or generated-answer accuracy.
"""
from uuid import uuid4

import pymupdf
import pytest

from scripts.rag import multimodal_rag as mm


@pytest.fixture
def rag(tmp_path):
    instance = mm.LocalRAG(tmp_path / "store")
    yield instance
    instance.close()


def evidence(count, *, long_first=False):
    return [dict(chunk_id=str(uuid4()), title="audit", page=i + 1,
                 text=f"PAGE_{i+1}_FACT", page_context=("x " * 4500 if long_first and i == 0
                                                          else f"PAGE_{i+1}_FACT"),
                 image_path=f"page_{i+1}.png") for i in range(count)]


def test_real_ingestion_retrieves_facts_across_pages(rag, tmp_path):
    path = tmp_path / "cross_page.pdf"
    with pymupdf.open() as pdf:
        for fact in ["Orchid project revenue is 120 million dollars.",
                     "Orchid project costs are 75 million dollars.",
                     "Orchid project target margin is 30 percent."]:
            pdf.new_page().insert_text((50, 50), fact)
        pdf.save(path)
    rag.ingest("financial_report_qa", path)
    result = rag.answer("financial_report_qa",
                        "What are Orchid revenue, costs and target margin?", top_k=3)
    assert {h["page"] for h in result["evidence"]} == {1, 2, 3}
    assert all(value in result["answer"] for value in ["120", "75", "30"])
    assert result["generation_mode"] == "extractive"


def test_four_visual_pages_report_omitted_images(rag, monkeypatch):
    hits = evidence(4)
    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: hits)
    calls = []
    def model(prompt, images):
        calls.append((prompt, images))
        return "Insufficient evidence"
    monkeypatch.setattr(mm, "ollama", model)
    result = rag.answer("hr_policy", "Compare all four charts", provider="ollama")
    assert len(result["evidence"]) == 4
    assert result["images_used"] == 0
    assert "visual_evidence_omitted" in result["warnings"]
    assert "page_4.png" not in calls[0][1]


def test_local_prompt_preserves_later_page_and_rejects_unstructured_answer(rag, monkeypatch):
    from scripts.common import local_models
    hits = evidence(2, long_first=True)
    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: hits)
    prompts = []
    def model(prompt, images):
        prompts.append(prompt)
        return "A partial answer."
    monkeypatch.setattr(local_models, "llamacpp", model)
    result = rag.answer("hr_policy", "Compare both pages", provider="llamacpp")
    assert "PAGE_2_FACT" in prompts[0]
    assert hits[1]["chunk_id"] in result["citation_check"]["cited_chunk_ids"]
    assert result["final_status"] == "partial_evidence"


def test_invented_claim_with_valid_source_id_is_not_returned(rag, monkeypatch):
    hits = evidence(1)
    monkeypatch.setattr(rag, "retrieve", lambda *a, **kw: hits)
    monkeypatch.setattr(mm, "ollama", lambda *a: "Invented value: 999999. [" + hits[0]["chunk_id"] + "]")
    result = rag.answer("hr_policy", "What is the value?", provider="ollama")
    assert "999999" not in result["answer"]
    assert result["final_status"] == "partial_evidence"


def test_compound_question_retrieves_each_question(rag, monkeypatch):
    calls = []
    def retrieve(*args, **kwargs):
        calls.append((args, kwargs))
        return []
    monkeypatch.setattr(rag, "retrieve", retrieve)
    query = "What is revenue? What are costs? Compute margin and compare the chart."
    rag.answer("financial_report_qa", query)
    assert len(calls) == 3
    assert calls[0][0][1] == "What is revenue"
