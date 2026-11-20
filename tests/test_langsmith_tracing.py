"""LangSmith tracing without the network: RunTree.post/patch are replaced, so the test sees every run that would be
sent (name, type, parent, project, metadata) and nothing leaves the machine."""
from __future__ import annotations

from typing import Any, TypedDict

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langsmith import run_trees

import scripts.api.api_app as api_app
from scripts.agents import financial_prompting_agent as agent
from scripts.observability import langsmith_utils as ls

FAKE_KEY = "test-key-never-sent"


@pytest.fixture
def runs(monkeypatch) -> list[dict[str, Any]]:
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT",
                 "LANGCHAIN_API_KEY", "ANSWER_CACHE_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LANGSMITH_API_KEY", FAKE_KEY)
    sent: list[dict[str, Any]] = []
    monkeypatch.setattr(run_trees.RunTree, "post", lambda self, *a, **k: sent.append({
        "name": self.name, "run_type": self.run_type, "id": self.id, "parent": self.parent_run_id,
        "project": self.session_name, "metadata": dict((self.extra or {}).get("metadata") or {}),
        "inputs": self.inputs}))
    monkeypatch.setattr(run_trees.RunTree, "patch", lambda self, *a, **k: None)
    return sent


class State(TypedDict):
    q: str
    a: str


def tiny_graph():
    def answer(state):
        return {"a": agent.MockLLM().invoke([SystemMessage(content="sys"), HumanMessage(content=state["q"])]).content}
    graph = StateGraph(State)
    graph.add_node("answer", answer)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)
    return graph.compile()


def test_request_tracing_traces_only_inside_the_block(runs) -> None:
    graph = tiny_graph()
    graph.invoke({"q": "hi", "a": ""})
    assert runs == [] and not ls.langsmith_is_enabled()

    with ls.request_tracing(enabled=True, project_name="probe", metadata={"request_id": "r1"}) as status:
        assert status == {"enabled": True, "project": "probe", "endpoint": None}
        ls.run_traced("api.chat", graph.invoke, {"q": "hi", "a": ""})

    assert not ls.langsmith_is_enabled()                  # no environment change survives the request
    assert [r["name"] for r in runs] == ["api.chat", "LangGraph", "answer", "mock_llm"]
    root = runs[0]["id"]
    assert runs[0]["parent"] is None and all(r["parent"] is not None for r in runs[1:])
    assert all(r["project"] == "probe" and r["metadata"]["request_id"] == "r1" for r in runs)
    llm = runs[-1]
    assert llm["run_type"] == "llm" and llm["metadata"]["ls_provider"] == "mock"
    assert llm["inputs"]["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert root is not None


def test_nested_blocks_keep_the_outer_project_and_metadata(runs) -> None:
    with ls.request_tracing(enabled=True, project_name="outer", metadata={"thread_id": "t1"}):
        with ls.request_tracing(enabled=True, project_name="inner", metadata={"request_id": "r2"}) as status:
            ls.run_traced("step", lambda: 1)
    assert status["project"] == "outer"
    assert runs[0]["project"] == "outer" and runs[0]["metadata"]["thread_id"] == "t1"
    assert runs[0]["metadata"]["request_id"] == "r2"


def test_no_key_or_not_requested_sends_nothing(runs, monkeypatch) -> None:
    with ls.request_tracing(enabled=False) as status:
        ls.run_traced("step", lambda: 1)
    assert status["reason"] == "not_requested"
    monkeypatch.delenv("LANGSMITH_API_KEY")
    with ls.request_tracing(enabled=True) as status:
        assert agent.MockLLM().invoke([HumanMessage(content="x")]).content
    assert status["reason"] == "missing_LANGSMITH_API_KEY" and runs == []


def test_hosted_llm_run_records_tokens_and_never_the_key(runs, monkeypatch) -> None:
    llm = agent.HostedChatLLM(api_base="http://llm.invalid/v1", api_key="secret-llm-key", model="qwen3-8b",
                              temperature=0.0)
    monkeypatch.setattr(llm, "_invoke", lambda messages: agent.MockResponse(
        "42", usage={"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}))
    with ls.request_tracing(enabled=True, project_name="p"):
        assert llm.invoke([HumanMessage(content="q")]).content == "42"
    llm_run = next(r for r in runs if r["name"] == "hosted_llm")
    assert llm_run["run_type"] == "llm" and llm_run["metadata"]["ls_model_name"] == "qwen3-8b"
    assert "secret-llm-key" not in repr(runs)


def test_api_request_flag_does_not_turn_tracing_on_for_later_requests(runs, monkeypatch) -> None:
    monkeypatch.setattr(api_app, "run_technique", lambda technique, provider, config: {
        "technique": technique, "final_status": "answered", "answer": "Revenue was $1 [c1].",
        "citation_check": {"valid": True, "cited_chunk_ids": ["c1"], "allowed_chunk_ids": ["c1"]},
        "evidence": [{"chunk_id": "c1", "source_type": "annual_report_pdf", "ticker": "COST", "year": 2025,
                      "page": 7, "preview": "evidence"}]})
    client = TestClient(api_app.app)
    request_id = "11111111-2222-3333-4444-555555555555"
    traced = client.post("/chat/financial_report_qa", headers={"X-Request-ID": request_id},
                         json={"query": "What was Costco revenue in 2025?", "enable_langsmith": True})
    assert traced.status_code == 200 and traced.json()["langsmith"]["enabled"] is True
    assert {r["name"] for r in runs} >= {"api.financial_report_qa.agentic_graph", "api.grounding_policy"}
    assert all(r["metadata"]["request_id"] == request_id for r in runs)
    assert runs[0]["project"] == "capstone-financial-report-qa-api"

    count = len(runs)
    plain = client.post("/chat/financial_report_qa", json={"query": "What was Costco revenue in 2024?"})
    assert plain.json()["langsmith"] == {"enabled": False, "reason": "not_requested"}
    assert len(runs) == count                             # the earlier request left tracing off


def test_conversation_turns_share_a_thread_id(runs, monkeypatch) -> None:
    monkeypatch.setattr(api_app, "run_technique", lambda technique, provider, config: {
        "technique": technique, "final_status": "answered", "answer": "A [c1].",
        "citation_check": {"valid": True, "cited_chunk_ids": ["c1"], "allowed_chunk_ids": ["c1"]},
        "evidence": [{"chunk_id": "c1", "source_type": "annual_report_pdf", "page": 1, "preview": "e"}]})
    client = TestClient(api_app.app)
    body = {"query": "What was Costco revenue in 2025?", "clarify": False, "enable_langsmith": True}
    first = client.post("/conversation/financial_report_qa", json=body).json()
    conversation_id = first["conversation_id"]
    client.post("/conversation/financial_report_qa", json={**body, "conversation_id": conversation_id})
    turns = [r for r in runs if r["name"] == "api.conversation_turn"]
    assert len(turns) == 2 and all(r["metadata"]["thread_id"] == conversation_id for r in turns)
    chat_runs = [r for r in runs if r["name"].startswith("api.financial_report_qa.")]
    assert chat_runs and all(r["parent"] is not None for r in chat_runs)    # nested under the turn


def test_health_reports_langsmith_without_the_key(runs, monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    health = TestClient(api_app.app).get("/health").json()
    assert health["langsmith"]["api_key_set"] is True and health["langsmith"]["tracing_all_requests"] is True
    assert FAKE_KEY not in repr(health)
    monkeypatch.delenv("LANGSMITH_API_KEY")
    assert "warning" in ls.tracing_status()


def test_agentic_graph_trace_shows_every_node_and_llm_call(runs, monkeypatch) -> None:
    for name in ("RAG_FIN_VISUAL", "RAG_BACKEND", "RAG_SUPPORT_CHECK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")
    chunk = "annual_report_edgar_ars_COST_2025_p0040_text0"
    monkeypatch.setattr(agent, "retrieve_evidence", lambda config: [{
        "chunk_id": chunk, "ticker": "COST", "year": 2025, "page": 40, "source_type": "annual_report_pdf",
        "text": "Total revenue was 275,235 million in fiscal 2025.", "preview": "revenue",
        "metadata": {"chunk_id": chunk}}])
    config = agent.AgentConfig(query="What was Costco total revenue in its 2025 annual report?", ticker="COST",
                               year=2025, source_type=None, top_k=4, dense_top_k=4, bm25_top_k=4,
                               retrieval_mode="hybrid", embedding_provider="hashing", hash_dimensions=16,
                               max_context_chars=6000, self_consistency_samples=1)
    with ls.request_tracing(enabled=True, project_name="p", metadata={"request_id": "r9"}):
        result = ls.run_traced("api.chat", agent.run_technique, "agentic_graph", "mock", config)
    names = [r["name"] for r in runs]
    for node in ("financial_workflow", "security_step", "plain_text_step", "plan_step", "retrieve_step",
                 "evidence_guard_step", "answer_step", "critic_step", "governance_step"):
        assert node in names, (node, names)
    assert "mock_llm" in names and result["final_status"]
    assert all(r["metadata"]["request_id"] == "r9" for r in runs)


def test_check_script_reports_each_step(monkeypatch, capsys) -> None:
    from scripts.observability import check_langsmith as check
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    sent = {}

    def fake_http(url, body=None, headers=None, timeout=30):
        if url.endswith("/info"):
            return 200, {}
        if url.endswith("/health"):
            return 200, {"langsmith": {"api_key_set": True, "project": "p"}, "runtime": {"financial_backend": "financial_vlm"}}
        sent.update(body=body, headers=headers)
        return 200, {"result": {"final_status": "answered", "langsmith": {"enabled": True, "project": "p"}}}

    monkeypatch.setattr(check, "http_json", fake_http)
    assert check.main(["--api-base", "http://api"]) == 0
    out = capsys.readouterr().out
    assert "[PASS] 3 Traced request" in out and "[SKIP] 4 Runs in LangSmith" in out
    assert sent["body"]["enable_langsmith"] is True and "X-Request-ID" in sent["headers"]

    monkeypatch.setattr(check, "http_json", lambda url, *a, **k: (200, {"langsmith": {"api_key_set": False}})
                        if url.endswith("/health") else (200, {}))
    assert check.main(["--api-base", "http://api"]) == 1


@pytest.mark.parametrize("backend", ["financial_vlm", "financial_qdrant", "financial_hybrid"])
def test_each_backend_search_is_a_retriever_run(runs, monkeypatch, backend) -> None:
    monkeypatch.setenv("RAG_BACKEND", backend)
    hits = [{"chunk_id": "c1", "page": 35, "ticker": "CVX", "year": 2022, "text": "Net oil production 1,440",
             "retrieval_paths": ["vlm_text"], "rrf_score": 0.03}]
    monkeypatch.setattr(agent, "_retrieve_evidence", lambda config: hits)
    config = agent.AgentConfig(query="net oil production chart", ticker="CVX", year=2022, source_type=None, top_k=4,
                               dense_top_k=4, bm25_top_k=4, retrieval_mode="hybrid", embedding_provider="hashing",
                               hash_dimensions=16, max_context_chars=6000, self_consistency_samples=1)
    with ls.request_tracing(enabled=True, project_name="p", metadata={"request_id": "r5"}):
        assert agent.retrieve_evidence(config) == hits
    run = runs[0]
    assert run["name"] == f"retrieve.{backend}" and run["run_type"] == "retriever"
    assert run["metadata"]["rag_backend"] == backend and run["metadata"]["request_id"] == "r5"
    assert run["inputs"]["query"] == "net oil production chart" and run["inputs"]["filters"]["ticker"] == "CVX"


def test_hybrid_stages_are_child_runs(runs, monkeypatch) -> None:
    from scripts.rag import financial_hybrid as hybrid
    monkeypatch.setenv("RAG_FIN_VISUAL", "auto")
    text = [{"chunk_id": "t1", "page": 35, "text": "1,440"}]
    monkeypatch.setattr(hybrid.chroma_db, "retrieve_for_agent", lambda *a, **k: text)
    monkeypatch.setattr(hybrid, "visual_pages", lambda *a, **k: [{"doc_id": "d", "page": 35, "rank": 1, "payload": {}}])
    monkeypatch.setattr(hybrid.chroma_db, "ChromaStore", lambda root: None)
    monkeypatch.setattr(hybrid, "page_evidence", lambda store, found: [{"chunk_id": "t1", "page": 35, "text": "1,440"}])
    with ls.request_tracing(enabled=True, project_name="p"):
        hits = hybrid.retrieve_for_agent("crude oil chart", 5, ticker="CVX", year=2022, visual=True)
    assert [h["chunk_id"] for h in hits] == ["t1"]
    assert [r["name"] for r in runs] == ["hybrid.chroma_text", "hybrid.colpali_pages", "hybrid.fusion"]
    assert all(r["run_type"] == "retriever" for r in runs)


def test_colpali_pages_show_as_documents() -> None:
    docs = ls.as_documents([{"doc_id": "d", "page": 35, "rank": 1, "payload": {"text": "Net production", "ticker": "CVX"}}])
    assert docs["documents"][0]["page_content"] == "Net production"
    assert docs["documents"][0]["metadata"] == {"doc_id": "d", "ticker": "CVX", "page": 35, "rank": 1}


def test_api_continues_a_callers_trace_and_nests_the_chat_under_the_turn(runs, monkeypatch) -> None:
    monkeypatch.setattr(api_app, "run_technique", lambda technique, provider, config: {
        "technique": technique, "final_status": "answered", "answer": "A [c1].",
        "citation_check": {"valid": True, "cited_chunk_ids": ["c1"], "allowed_chunk_ids": ["c1"]},
        "evidence": [{"chunk_id": "c1", "source_type": "annual_report_pdf", "page": 1, "preview": "e"}]})
    client = TestClient(api_app.app)
    with ls.request_tracing(enabled=True, project_name="ui-project", metadata={"client": "streamlit"}):
        def ui_call():
            return client.post("/conversation/financial_report_qa", headers=ls.outgoing_trace_headers(), json={
                "query": "What was Costco revenue in 2025?", "clarify": False, "enable_langsmith": True}).json()
        turn = ls.run_traced("streamlit.conversation", ui_call)
    assert turn["status"] == "answered"
    by_name = {r["name"]: r for r in runs}
    root = by_name["streamlit.conversation"]
    assert root["parent"] is None
    assert by_name["api.conversation_turn"]["parent"] == root["id"]
    chat = next(r for r in runs if r["name"].startswith("api.financial_report_qa."))
    assert chat["parent"] == by_name["api.conversation_turn"]["id"]           # not re-parented to the UI run
    assert {r["project"] for r in runs} == {"ui-project"}
    assert by_name["api.conversation_turn"]["metadata"]["client"] == "streamlit"


def test_streamlit_post_json_sends_trace_headers_only_when_traced(runs, monkeypatch) -> None:
    from scripts.ui import api_streamlit_app as ui
    sent = []
    monkeypatch.setattr(ui, "_post_json", lambda base, path, payload, extra=None: sent.append(extra) or {"ok": True})
    ui.post_json("http://api", "/conversation/financial_report_qa", {"query": "q", "enable_langsmith": True})
    assert "langsmith-trace" in sent[0] and "X-Request-ID" in sent[0]
    assert [r["name"] for r in runs] == ["streamlit.conversation"]
    assert runs[0]["metadata"]["client"] == "streamlit"
    ui.post_json("http://api", "/conversation/financial_report_qa", {"query": "q", "enable_langsmith": False})
    assert sent[1] is None and len(runs) == 1
