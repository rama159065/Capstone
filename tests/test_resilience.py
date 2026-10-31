import io
import json
import threading
import time
import urllib.error
from uuid import uuid4

import pytest

from scripts.domain_common.resilience import Calls, CallFailure
from scripts.domain_common.safe_answer import answer


def test_completed_call_is_not_repeated():
    invoked = []
    key = str(uuid4())
    calls = Calls(1)
    assert calls.run("tool", lambda: invoked.append(1) or "ok", key=key) == "ok"
    assert calls.run("tool", lambda: pytest.fail("duplicate"), key=key) == "ok"
    assert invoked == [1]
    assert calls.events[-1]["status"] == "cached"


def test_concurrent_call_joins_same_work():
    key, invoked = str(uuid4()), []
    ready, release = threading.Event(), threading.Event()
    def work():
        invoked.append(1)
        ready.set()
        release.wait(1)
        return "ok"
    result = []
    thread = threading.Thread(target=lambda: result.append(Calls(2).run("tool", work, key=key)))
    thread.start()
    assert ready.wait(1)
    release.set()
    assert Calls(2).run("tool", work, key=key) == "ok"
    thread.join(2)
    assert result == ["ok"] and invoked == [1]


def test_timeout_returns_promptly_and_does_not_duplicate_inflight_work():
    key, release, invoked = str(uuid4()), threading.Event(), []
    def work():
        invoked.append(1)
        release.wait(1)
        return "done"
    calls = Calls(.03)
    started = time.monotonic()
    try:
        with pytest.raises(CallFailure, match="deadline"):
            calls.run("tool", work, key=key)
        assert time.monotonic() - started < .5
        with pytest.raises(CallFailure):
            Calls(.03).run("tool", work, key=key)
        assert invoked == [1]
        assert calls.events[0]["status"] == "timeout"
    finally:
        release.set()


@pytest.mark.parametrize("code,retries", [(429, 2), (503, 2), (400, 1), (401, 1)])
def test_only_explicit_transient_rejections_are_retried(code, retries):
    attempts = []
    def work():
        attempts.append(1)
        raise urllib.error.HTTPError("local", code, "rejected", {"Retry-After": "0"}, io.BytesIO())
    with pytest.raises(CallFailure):
        Calls(1).run("model", work, key=str(uuid4()), retry=True)
    assert len(attempts) == retries


class FakeRAG:
    reranker_model = None
    pipeline = "test"
    def __init__(self, root, hits):
        self.root, self.hits = root, hits
    def retrieve(self, *a, **kw):
        return self.hits


def hit():
    return {"chunk_id": str(uuid4()), "page": 1, "title": "test", "text": "Revenue was 120 million.",
            "page_context": "Revenue was 120 million."}


def test_no_evidence_never_calls_model(tmp_path, monkeypatch):
    from scripts.domain_common import multimodal_rag as mm
    monkeypatch.setattr(mm, "ollama", lambda *a: pytest.fail("must not generate"))
    result = answer(FakeRAG(tmp_path, []), "hr_policy", "What is revenue?", provider="ollama")
    assert result["final_status"] == "insufficient_evidence"
    assert result["images_used"] == 0
    assert not result["citation_check"]["valid"]


def test_verified_passage_and_replayed_request(tmp_path, monkeypatch):
    from scripts.domain_common import multimodal_rag as mm
    row, invoked = hit(), []
    def model(*args):
        invoked.append(1)
        return json.dumps({"selections": [{"source_id": row["chunk_id"], "quote": row["text"]}], "unanswered": []})
    monkeypatch.setattr(mm, "ollama", model)
    rag = FakeRAG(tmp_path, [row])
    for _ in range(2):
        result = answer(rag, "hr_policy", "What is revenue?", provider="ollama", idempotency_key="request-1")
        assert result["final_status"] == "evidence_found"
        assert "120" in result["answer"]
    assert invoked == [1]
    # Same caller key with changed data must run again.
    row["page_context"] = "Revenue was 121 million."
    answer(rag, "hr_policy", "What is revenue?", provider="ollama", idempotency_key="request-1")
    assert len(invoked) == 2


@pytest.mark.parametrize("response", ["", "not json", '{"selections":null}',
    '{"selections":[{"source_id":"unknown","quote":"999"}],"unanswered":[]}'])
def test_bad_generation_degrades_without_returning_invention(tmp_path, monkeypatch, response):
    from scripts.domain_common import multimodal_rag as mm
    monkeypatch.setattr(mm, "ollama", lambda *a: response)
    result = answer(FakeRAG(tmp_path, [hit()]), "hr_policy", "revenue", provider="ollama")
    assert result["final_status"] == "partial_evidence"
    assert "999" not in result["answer"]
    assert "generation_unavailable_or_ungrounded" in result["warnings"]


def test_retrieval_failure_is_visible_and_safe(tmp_path):
    rag = FakeRAG(tmp_path, [])
    def fail(*a, **kw):
        raise OSError("private service address")
    rag.retrieve = fail
    result = answer(rag, "hr_policy", "revenue", provider="ollama")
    assert result["final_status"] == "insufficient_evidence"
    assert result["warnings"] == ["retrieval_unavailable"]
    assert "private service" not in json.dumps(result)


@pytest.mark.parametrize("hits", [None, [None, {"chunk_id": "x", "text": None}],
                                   [{"chunk_id": "x", "text": "Image-only page; no readable text extracted."}]])
def test_empty_and_malformed_tool_results_do_not_generate(tmp_path, monkeypatch, hits):
    from scripts.domain_common import multimodal_rag as mm
    monkeypatch.setattr(mm, "ollama", lambda *a: pytest.fail("must not generate"))
    result = answer(FakeRAG(tmp_path, hits), "hr_policy", "revenue", provider="ollama")
    assert result["final_status"] == "insufficient_evidence"


def test_model_caption_cannot_verify_a_fact(tmp_path, monkeypatch):
    from scripts.domain_common import multimodal_rag as mm
    row = hit() | {"visual_captioned": True}
    monkeypatch.setattr(mm, "ollama", lambda *a: pytest.fail("caption is not grounding evidence"))
    result = answer(FakeRAG(tmp_path, [row]), "hr_policy", "revenue", provider="ollama")
    assert result["final_status"] == "insufficient_evidence"
    assert "no_verifiable_text_reingest_or_review_visual" in result["warnings"]


def test_omitted_context_cannot_receive_a_citation(tmp_path, monkeypatch):
    from scripts.domain_common import multimodal_rag as mm
    row = hit() | {"page_context": "x" * 25000}
    monkeypatch.setattr(mm, "ollama", lambda *a: pytest.fail("nothing fits the budget"))
    result = answer(FakeRAG(tmp_path, [row]), "hr_policy", "revenue", provider="ollama")
    assert not result["citation_check"]["valid"]
    assert "context_budget_exceeded" in result["warnings"]


def test_retry_requires_a_key():
    with pytest.raises(ValueError, match="idempotency"):
        Calls(1).run("unsafe_retry", lambda: None, retry=True)
