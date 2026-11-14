"""Mutual fund API: read-only /mutual_fund routes (404 until data is loaded) and fund answers through the chat route."""
from __future__ import annotations

from fastapi.testclient import TestClient

import scripts.api.api_app as api_app
from scripts.agents import financial_clarifier as fc
from tests.data_mutual_fund import SERIES_500, load_funds


def test_routes_are_404_without_fund_data(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_MUTUAL_FUND_DIR", str(tmp_path / "none"))
    client = TestClient(api_app.app)
    response = client.get("/mutual_fund/funds")
    assert response.status_code == 404 and "No mutual fund data" in response.json()["message"]
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/mutual_fund/funds", "/mutual_fund/holdings", "/mutual_fund/holders/{ticker}", "/mutual_fund/search"} <= set(paths)
    assert "/chat/financial_report_qa" in paths and "/health" in paths


def test_routes_return_loaded_rows(tmp_path, monkeypatch):
    load_funds(tmp_path)
    monkeypatch.setenv("RAG_MUTUAL_FUND_DIR", str(tmp_path))
    client = TestClient(api_app.app)
    funds = client.get("/mutual_fund/funds").json()
    assert funds["count"] == 2
    holdings = client.get("/mutual_fund/holdings", params={"series_id": SERIES_500, "limit": 3}).json()
    assert [h["ticker"] for h in holdings["holdings"]] == ["NVDA", "GOOGL", "MSFT"]
    assert holdings["coverage"]["corpus_companies"] == 5
    holders = client.get("/mutual_fund/holders/nvda").json()
    assert holders["ticker"] == "NVDA" and len(holders["holders"]) == 2
    assert client.get("/mutual_fund/holders/ZZZZ").status_code == 404


def test_chat_route_returns_the_fund_table(tmp_path, monkeypatch):
    load_funds(tmp_path)
    monkeypatch.setenv("RAG_MUTUAL_FUND_DIR", str(tmp_path))
    monkeypatch.setenv("AGENT_MAX_STEPS", "10")
    monkeypatch.setattr(fc, "default_clarifier", lambda: fc.Clarifier(directory={"NVDA": [2025]}))
    client = TestClient(api_app.app)
    body = client.post("/chat/financial_report_qa", json={
        "query": "What are the top holdings of the Vanguard 500 Index Fund?", "technique": "router",
        "llm_provider": "mock", "embedding_provider": "hashing"}).json()
    assert body["route"]["route"] == "fund" and body["mutual_fund"]["kind"] == "top_holdings"
    assert body["mutual_fund"]["rows"][0]["ticker"] == "NVDA"
    missing = client.post("/chat/financial_report_qa", json={
        "query": "What does the Vanguard fund hold?", "technique": "fund_graph", "llm_provider": "mock",
        "embedding_provider": "hashing"}).json()
    assert missing["final_status"] == "not_available" and "which Vanguard fund" in missing["answer"]
