from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

import scripts.api_app as api_app


def test_health_and_domains() -> None:
    client = TestClient(api_app.app)

    health = client.get("/health").json()
    domains = client.get("/domains").json()["domains"]

    assert health["status"] == "ok"
    assert "financial_report_qa" in health["domains"]
    assert {row["domain"] for row in domains} >= {"travel_expense", "financial_report_qa"}


def test_request_logs_and_response_include_correlation_id(caplog) -> None:
    request_id = "12345678-1234-5678-1234-567812345678"
    client = TestClient(api_app.app)

    with caplog.at_level("INFO", logger="scripts.api_app"):
        response = client.get("/health", headers={"X-Request-ID": request_id})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id
    assert f"api_request_completed request_id={request_id}" in caplog.text
    assert "method=GET path=/health status_code=200" in caplog.text


def test_langsmith_trace_metadata_contains_request_id(monkeypatch) -> None:
    request_id = "12345678-1234-5678-1234-567812345678"
    captured_metadata: dict[str, Any] = {}

    def fake_run_domain_agent(domain_config: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "domain": domain_config.domain,
            "display_name": domain_config.display_name,
            "final_status": "answered",
            "query": kwargs["query"],
        }

    def capture_traced_call(name: str, func: Any, *args: Any, **kwargs: Any) -> Any:
        captured_metadata.update(kwargs["metadata"])
        return func(*args, **kwargs)

    monkeypatch.setattr(api_app, "run_domain_agent", fake_run_domain_agent)
    monkeypatch.setattr(api_app, "run_traced", capture_traced_call)
    client = TestClient(api_app.app)

    response = client.post(
        "/chat/travel_expense",
        headers={"X-Request-ID": request_id},
        json={"query": "test query"},
    )

    assert response.status_code == 200
    assert captured_metadata["request_id"] == request_id


def test_chat_domain_uses_shared_agent(monkeypatch) -> None:
    def fake_run_domain_agent(domain_config: Any, **kwargs: Any) -> dict[str, Any]:
        return {
            "domain": domain_config.domain,
            "display_name": domain_config.display_name,
            "final_status": "answered",
            "query": kwargs["query"],
            "answer": "Travel evidence answer with chunk_1.",
            "citation_check": {"valid": True, "cited_chunk_ids": ["chunk_1"]},
            "human_review": {"required": False, "status": "auto_approved", "triggers": []},
            "evidence": [{"chunk_id": "chunk_1", "preview": "receipt evidence"}],
        }

    monkeypatch.setattr(api_app, "run_domain_agent", fake_run_domain_agent)
    client = TestClient(api_app.app)

    response = client.post(
        "/chat/travel_expense",
        json={"query": "What documentation is required for travel expenses?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["domain"] == "travel_expense"
    assert payload["final_status"] == "answered"
    assert payload["langsmith"]["enabled"] is False


def test_chat_domain_validates_agent_response(monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "run_domain_agent",
        lambda domain_config, **kwargs: {
            "domain": None,
            "final_status": "answered",
            "query": kwargs["query"],
        },
    )
    client = TestClient(api_app.app, raise_server_exceptions=False)

    response = client.post(
        "/chat/travel_expense",
        json={"query": "What documentation is required for travel expenses?"},
    )

    assert response.status_code == 500
    assert response.json() == {
        "error": "internal_server_error",
        "message": "An unexpected error occurred. Please try again later.",
    }


def test_invalid_chat_request_returns_validation_error() -> None:
    client = TestClient(api_app.app)

    response = client.post("/chat/travel_expense", json={"query": ""})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "validation_error"
    assert payload["message"] == "Request validation failed."
    assert payload["details"][0]["field"] == "body.query"


def test_unknown_domain_returns_http_error_message() -> None:
    client = TestClient(api_app.app)

    response = client.post("/chat/not_a_domain", json={"query": "hello"})

    assert response.status_code == 404
    assert response.json() == {
        "error": "http_error",
        "message": "Unknown domain: not_a_domain",
    }


def test_unhandled_service_exception_returns_safe_error(monkeypatch) -> None:
    def raise_service_error(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("internal service detail")

    monkeypatch.setattr(api_app, "run_domain_agent", raise_service_error)
    client = TestClient(api_app.app, raise_server_exceptions=False)

    response = client.post("/chat/travel_expense", json={"query": "hello"})

    assert response.status_code == 500
    assert response.json()["error"] == "internal_server_error"
    assert "internal service detail" not in response.text


def test_fastapi_documents_response_models() -> None:
    openapi = api_app.app.openapi()
    schemas = openapi["components"]["schemas"]

    assert "ChatResponse" in schemas
    assert "FinancialChatResponse" in schemas
    assert "RetrieveResponse" in schemas
    assert api_app.app.docs_url == "/docs"
    assert api_app.app.redoc_url == "/redoc"
    chat_operation = openapi["paths"]["/chat/{domain}"]["post"]
    assert chat_operation["summary"] == "Answer a domain question"
    assert chat_operation["description"]
    assert "429" in chat_operation["responses"]
    query_field = schemas["ChatRequest"]["properties"]["query"]
    assert query_field["minLength"] == 1
    assert query_field["maxLength"] == 4000
    assert query_field["description"]


def test_rate_limit_returns_429_with_retry_after(monkeypatch) -> None:
    monkeypatch.setattr(
        api_app.app.state,
        "rate_limiter",
        api_app.InMemoryRateLimiter(limit=2, window_seconds=60),
    )
    client = TestClient(api_app.app)

    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200
    response = client.get("/health")

    assert response.status_code == 429
    assert response.json() == {
        "error": "rate_limit_exceeded",
        "message": "Rate limit exceeded. Please try again later.",
    }
    assert 1 <= int(response.headers["Retry-After"]) <= 60


def test_chat_financial_uses_financial_agent(monkeypatch) -> None:
    def fake_run_technique(technique: str, llm_provider: str, config: Any) -> dict[str, Any]:
        return {
            "technique": technique,
            "final_status": "answered",
            "answer": f"{llm_provider} financial answer with fin_chunk.",
            "citation_check": {"valid": True, "cited_chunk_ids": ["fin_chunk"]},
            "evidence": [
                {
                    "chunk_id": "fin_chunk",
                    "source_type": "annual_report_pdf",
                    "ticker": config.ticker,
                    "year": config.year,
                    "page": 7,
                    "preview": "financial evidence",
                }
            ],
        }

    monkeypatch.setattr(api_app, "run_technique", fake_run_technique)
    client = TestClient(api_app.app)

    response = client.post(
        "/chat/financial_report_qa",
        json={
            "query": "What does Apple say about revenue?",
            "ticker": "AAPL",
            "year": 2024,
            "llm_provider": "mock",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["domain"] == "financial_report_qa"
    assert payload["technique"] == "agentic_graph"
    assert payload["evidence_preview"][0]["ticker"] == "AAPL"


def test_retrieve_domain_summarizes_evidence(monkeypatch) -> None:
    monkeypatch.setattr(
        api_app,
        "retrieve_hybrid",
        lambda *args, **kwargs: [
            {
                "chunk_id": "hr_1",
                "rank": 1,
                "rerank_score": 0.7,
                "text": "FMLA leave request evidence.",
                "metadata": {
                    "title": "FMLA Guide",
                    "source_type": "hr_leave_policy_pdf",
                    "page": 2,
                    "source_url": "https://example.test/fmla.pdf",
                },
            }
        ],
    )
    client = TestClient(api_app.app)

    response = client.post(
        "/retrieve/hr_policy",
        json={"query": "How do employees request FMLA leave?", "top_k": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["evidence"][0]["chunk_id"] == "hr_1"


def test_unknown_domain_returns_404() -> None:
    client = TestClient(api_app.app)

    response = client.post("/chat/not_a_domain", json={"query": "hello"})

    assert response.status_code == 404
