"""Deployment layer: persistent conversations, index coverage, step cap, API auth/cache/companies, UI pickers."""
from __future__ import annotations

import os

import pytest
import sqlite3
import sys
import time
import types
from pathlib import Path

from fastapi.testclient import TestClient

import scripts.agents.rag_agent as rag
from scripts.agents import financial_clarifier as fc
from scripts.api import api_app
from scripts.common import agent_limits, runtime_profile
from sample_domain import SAMPLE_DOMAIN as PROCUREMENT_CONFIG  # noqa: E402

sys.modules.setdefault("streamlit", types.ModuleType("streamlit"))
from scripts.ui import api_streamlit_app as ui  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ conversations

def test_sqlite_conversations_survive_a_new_store(tmp_path):
    path = tmp_path / "state" / "conversations.sqlite3"
    first = fc.ConversationStore(path=path)
    conversation = first.get()
    conversation.original_query = "revenue"
    conversation.slots.companies = ["AAPL"]
    conversation.slots.years = [2024]
    conversation.pending = ["metric"]
    first.save(conversation)

    restored = fc.ConversationStore(path=path).get(conversation.conversation_id)   # e.g. after an API restart
    assert restored.original_query == "revenue"
    assert restored.slots.companies == ["AAPL"] and restored.slots.years == [2024]
    assert isinstance(restored.slots, fc.Slots) and restored.pending == ["metric"]

    fc.ConversationStore(path=path).drop(conversation.conversation_id)
    assert fc.ConversationStore(path=path).get(conversation.conversation_id).original_query == ""


def test_sqlite_conversations_expire(tmp_path):
    store = fc.ConversationStore(path=tmp_path / "c.sqlite3", ttl_seconds=60)
    conversation = store.get("abc")
    conversation.original_query = "old"
    store.save(conversation)
    with sqlite3.connect(tmp_path / "c.sqlite3") as db:
        db.execute("UPDATE conversations SET updated=?", (time.time() - 120,))
    assert store.get("abc").original_query == ""


def test_clarifier_turns_persist_through_the_store(tmp_path):
    store = fc.ConversationStore(path=tmp_path / "c.sqlite3")
    clarifier = fc.Clarifier(directory={"AAPL": [2023, 2024]})
    conversation = store.get()
    first = clarifier.step(conversation, "What was Apple's revenue?")
    store.save(conversation)
    assert first["status"] == "needs_clarification"
    again = fc.ConversationStore(path=tmp_path / "c.sqlite3").get(conversation.conversation_id)
    second = clarifier.step(again, "2024")
    assert second["status"] == "ready" and second["ticker"] == "AAPL" and second["year"] == 2024


# ------------------------------------------------------------------ coverage from the served index

def make_state(path: Path, rows, with_stage=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as db:
        if with_stage:
            db.execute("CREATE TABLE filings (doc_id TEXT, stage TEXT, status TEXT)")
            db.executemany("INSERT INTO filings VALUES (?,?,?)", rows)
        else:
            db.execute("CREATE TABLE filings (doc_id TEXT, status TEXT)")
            db.executemany("INSERT INTO filings VALUES (?,?)", rows)


def test_directory_includes_filings_the_index_holds(tmp_path, monkeypatch):
    make_state(tmp_path / "fq" / "ingest.sqlite3", [
        ("edgar_ars_JPM_2025", "text", "complete"),
        ("edgar_ars_BRK-B_2024", "text", "complete"),
        ("edgar_ars_XOM_2023", "text", "failed"),          # not searchable
        ("edgar_ars_XOM_2024", "visual", "complete"),      # images only: text stage decides
    ])
    make_state(tmp_path / "vlm" / "vlm_ingest.sqlite3", [("edgar_ars_COST_2024", "complete")], with_stage=False)
    monkeypatch.setattr(fc, "MANIFESTS", ())
    monkeypatch.setenv("RAG_INGEST_DIR", str(tmp_path / "fq"))
    monkeypatch.setenv("RAG_VLM_DIR", str(tmp_path / "vlm"))
    monkeypatch.delenv("RAG_CORPUS_DIR", raising=False)
    assert fc.load_directory() == {"BRK-B": [2024], "COST": [2024], "JPM": [2025]}
    assert fc.load_directory(paths=[]) == {}                 # explicit manifests only
    assert fc.indexed_filings(tmp_path / "missing.sqlite3") == []


# ------------------------------------------------------------------ step cap

def test_step_cap_config(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")
    assert agent_limits.graph_config() == {"recursion_limit": 11}
    monkeypatch.setenv("AGENT_MAX_STEPS", "nonsense")
    assert agent_limits.max_steps() == agent_limits.DEFAULT_MAX_STEPS


def run_procurement(**overrides):
    params = {"query": "What procurement criteria apply to vendor bids?", "llm_provider": "mock",
              "embedding_provider": "hashing", "hash_dimensions": 128, "top_k": 3, "dense_top_k": 3, "bm25_top_k": 3,
              "source_type": None, "max_context_chars": 1000, "enable_guardrails": True, "enable_human_review": True}
    return rag.run_domain_agent(PROCUREMENT_CONFIG, **{**params, **overrides})


def sample_hit():
    return {"rank": 1, "chunk_id": "chunk_1", "text": "Vendor evaluation criteria evidence.",
            "metadata": {"title": "Procurement Guide", "source_type": "procurement_policy_pdf", "page": 1,
                         "source_url": "https://example.test/procurement.pdf"}, "rerank_score": 0.9}


def test_domain_graph_fits_the_default_cap_and_fails_closed_below_it(monkeypatch):
    monkeypatch.setattr(rag, "retrieve_hybrid", lambda *args, **kwargs: [sample_hit()])
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")
    assert run_procurement()["final_status"] != "step_limit"
    monkeypatch.setenv("AGENT_MAX_STEPS", "3")
    result = run_procurement()
    assert result["final_status"] == "step_limit"
    assert result["answer"] == agent_limits.STEP_LIMIT_ANSWER


# ------------------------------------------------------------------ API

def test_api_key_is_required_when_configured(monkeypatch):
    monkeypatch.setattr(api_app, "API_AUTH_TOKEN", "s3cret")
    client = TestClient(api_app.app)
    assert client.get("/health").status_code == 200                      # public, for health checks
    assert client.get("/domains").status_code == 401
    assert client.get("/domains", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/domains", headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/domains", headers={"Authorization": "Bearer s3cret"}).status_code == 200
    body = client.get("/companies").json()
    assert body["error"] == "unauthorized" and "request_id" in body


def test_no_api_key_configured_means_open(monkeypatch):
    monkeypatch.setattr(api_app, "API_AUTH_TOKEN", "")
    health = TestClient(api_app.app).get("/health").json()
    assert health["auth_required"] is False and health["agent_max_steps"] >= 1
    assert health["conversation_store"] in {"memory", "sqlite"}


def test_companies_lists_names_sectors_and_years(monkeypatch):
    monkeypatch.setattr(api_app, "_clarifier", fc.Clarifier(directory={"JPM": [2024, 2025], "XOM": [2024]}))
    client = TestClient(api_app.app)
    body = client.get("/companies").json()
    assert body["count"] == 2 and len(body["sectors"]) == 13
    jpm = next(row for row in body["companies"] if row["ticker"] == "JPM")
    assert jpm["sector"] == "Banking" and jpm["years"] == [2024, 2025] and jpm["company"]
    banking = client.get("/companies", params={"sector": "banking"}).json()
    assert [row["ticker"] for row in banking["companies"]] == ["JPM"]


def test_answer_cache_reuses_identical_requests_only():
    cache = api_app.AnswerCache(ttl_seconds=60, max_items=2)
    calls = []

    def run(status="answered"):
        calls.append(1)
        return {"final_status": status, "answer": "x"}

    first = api_app.ChatRequest(query="q1")
    assert cache.get_or_run("r", first, run)["answer"] == "x"
    assert cache.get_or_run("r", api_app.ChatRequest(query="q1", enable_langsmith=True), run)["cache"] == "hit"
    assert len(calls) == 1
    cache.get_or_run("r", api_app.ChatRequest(query="q2"), run)
    assert len(calls) == 2
    cache.get_or_run("r", api_app.ChatRequest(query="q3"), lambda: run("tool_error"))
    cache.get_or_run("r", api_app.ChatRequest(query="q3"), lambda: run("tool_error"))
    assert len(calls) == 4                                               # failures are never cached
    assert api_app.AnswerCache(0, 10).get_or_run("r", first, run) == {"final_status": "answered", "answer": "x"}


def test_request_defaults_come_from_the_environment():
    request = api_app.ChatRequest(query="q")
    assert request.llm_provider == api_app.DEFAULT_LLM_PROVIDER
    assert request.embedding_provider == api_app.DEFAULT_EMBEDDING_PROVIDER


# ------------------------------------------------------------------ UI, profile, image

def test_ui_pickers_follow_sector_and_ticker():
    companies = [{"ticker": "JPM", "sector": "Banking", "years": [2024, 2025]},
                 {"ticker": "XOM", "sector": "Oil and gas", "years": [2023]}]
    assert ui.ticker_choices(companies, []) == ["", "JPM", "XOM"]
    assert ui.ticker_choices(companies, ["Banking"]) == ["", "JPM"]
    assert ui.year_choices(companies, "JPM") == [None, 2025, 2024]
    assert ui.year_choices(companies, "") == [None, 2025, 2024, 2023]


def test_ui_sends_the_api_key(monkeypatch):
    monkeypatch.setattr(ui, "API_AUTH_TOKEN", "s3cret")
    assert ui.api_headers() == {"X-API-Key": "s3cret"}
    monkeypatch.setattr(ui, "API_AUTH_TOKEN", "")
    assert ui.api_headers() == {}


def test_server_profile_points_at_the_compose_services():
    # apply_profile writes os.environ directly, so restore all of it (monkeypatch can't undo keys it never saw).
    saved, applied = dict(os.environ), runtime_profile._applied
    try:
        for key in ("QDRANT_URL", "API_BASE_URL", "CONVERSATION_DB", "RAG_INGEST_DIR", "RAG_VLM_DIR", "RAG_BACKEND",
                    "RAG_DEVICE"):
            os.environ.pop(key, None)
        info = runtime_profile.apply_profile("server", force=True)
        assert info["QDRANT_URL"] == "http://qdrant:6333" and info["API_BASE_URL"] == "http://api:8000"
        assert info["CONVERSATION_DB"].endswith("conversations.sqlite3")
    finally:
        os.environ.clear()
        os.environ.update(saved)
        runtime_profile._applied = applied


def test_docker_requirements_skip_gpu_torch_and_jupyter():
    sys.path.insert(0, str(ROOT / "deploy"))
    try:
        from requirements_from_pyproject import requirements
    finally:
        sys.path.remove(str(ROOT / "deploy"))
    reqs = requirements(ROOT / "pyproject.toml")
    names = {r.split(">")[0].split("=")[0].split("<")[0].strip().lower() for r in reqs}
    assert {"torch", "torchvision", "jupyterlab"}.isdisjoint(names)
    assert {"fastapi", "langgraph", "streamlit", "qdrant-client", "uvicorn"} <= names


def test_compose_wires_the_services():
    import yaml
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert {"qdrant", "api", "ui", "caddy"} <= set(services)
    assert "--workers" in services["api"]["command"] and "1" in services["api"]["command"]
    assert services["api"]["environment"]["RAG_PROFILE"] == "server"
    assert "qdrant_snapshots:/qdrant/snapshots" in services["qdrant"]["volumes"]
    assert "qdrant_snapshots:/qdrant/snapshots" in services["api"]["volumes"]
    keys = {line.split("=")[0] for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")}
    assert {"LLM_API_BASE", "API_AUTH_TOKEN", "AGENT_MAX_STEPS", "ANSWER_CACHE_SECONDS", "DOMAIN"} <= keys


def test_directory_prefers_the_served_index_over_manifests(tmp_path, monkeypatch):
    manifest = tmp_path / "pdf_manifest.jsonl"
    manifest.write_text('{"ticker": "AAPL", "year": 2024}\n{"ticker": "COST", "year": 2024}\n', encoding="utf-8")
    make_state(tmp_path / "fq" / "ingest.sqlite3", [("edgar_ars_COST_2025", "text", "complete")])
    monkeypatch.setattr(fc, "MANIFESTS", (manifest,))
    monkeypatch.setenv("RAG_INGEST_DIR", str(tmp_path / "fq"))
    monkeypatch.delenv("RAG_VLM_DIR", raising=False)
    monkeypatch.delenv("RAG_CORPUS_DIR", raising=False)
    monkeypatch.delenv("COVERAGE_SOURCE", raising=False)
    assert fc.load_directory() == {"COST": [2025]}         # downloaded-only AAPL is not answerable
    monkeypatch.setenv("COVERAGE_SOURCE", "corpus")
    assert fc.load_directory() == {"AAPL": [2024], "COST": [2024, 2025]}
    monkeypatch.delenv("COVERAGE_SOURCE")
    monkeypatch.setenv("RAG_INGEST_DIR", str(tmp_path / "empty"))       # no index: manifests decide
    assert fc.load_directory() == {"AAPL": [2024], "COST": [2024]}


def test_company_outside_the_index_is_not_available_without_retrieval():
    scope = fc.Clarifier(directory={"COST": [2024, 2025]}).scope("What was Apple's net income for FY2024?")
    assert scope["coverage"]["status"] == "not_available"
    assert "Apple" in scope["coverage"]["message"]


def test_warm_up_is_skipped_off_the_ingest_backends(monkeypatch):
    from scripts.api import api_app
    monkeypatch.setenv("RAG_BACKEND", "chroma")
    assert api_app.warm_up_retrieval() == "skipped"
    monkeypatch.setenv("RAG_BACKEND", "financial_qdrant")
    monkeypatch.setenv("API_WARMUP", "false")
    assert api_app.warm_up_retrieval() == "skipped"


def test_domain_output_filter_block_goes_straight_to_governance(monkeypatch):
    monkeypatch.setattr(rag, "retrieve_hybrid", lambda *args, **kwargs: [sample_hit()])
    monkeypatch.setattr(rag, "output_content_filter", lambda answer, config: {
        "decision": "block", "risk_labels": ["test"], "redacted_preview": ""})
    monkeypatch.setattr(rag, "validate_citations", lambda *a, **k: pytest.fail("a blocked answer is not critiqued"))
    result = run_procurement()
    assert result["final_status"] == "refused"
    assert result["citation_check"]["reason"] == "answer blocked by the output filter"
    assert result["human_review"]["status"] == "not_required"
