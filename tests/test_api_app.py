from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

import json
import scripts.api.api_app as api_app


def fake_financial(answer, cited, evidence_ids=("fin_chunk",)):
    """A run_technique stand-in: the given answer and citations over the given evidence."""
    def run(technique, llm_provider, config):
        return {"technique": technique, "final_status": "answered", "answer": answer,
                "citation_check": {"valid": True, "cited_chunk_ids": list(cited), "allowed_chunk_ids": list(evidence_ids)},
                "evidence": [{"chunk_id": cid, "source_type": "annual_report_pdf", "ticker": "COST", "year": 2025,
                              "page": 7, "preview": "financial evidence"} for cid in evidence_ids]}
    return run


def test_health_and_domains() -> None:
    client = TestClient(api_app.app)

    health = client.get("/health").json()
    domains = client.get("/domains").json()["domains"]

    assert health["status"] == "ok"
    assert health["domains"] == ["financial_report_qa"]
    assert [row["domain"] for row in domains] == ["financial_report_qa"]


def test_request_logs_and_response_include_correlation_id(caplog) -> None:
    request_id = "12345678-1234-5678-1234-567812345678"
    client = TestClient(api_app.app)

    with caplog.at_level("INFO", logger="scripts.api.api_app"):
        response = client.get("/health", headers={"X-Request-ID": request_id})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id
    assert f"api_request_completed request_id={request_id}" in caplog.text
    assert "method=GET path=/health status_code=200" in caplog.text


def test_langsmith_trace_metadata_contains_request_id(monkeypatch) -> None:
    request_id = "87654321-4321-8765-4321-876543218765"
    captured_metadata: dict[str, Any] = {}

    def capture_traced_call(name: str, func: Any, *args: Any, run_type: str = "chain", tags: Any = None,
                            metadata: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        captured_metadata.update(metadata or {})
        return func(*args, **kwargs)

    monkeypatch.setattr(api_app, "run_technique", fake_financial("Answer [fin_chunk].", ["fin_chunk"]))
    monkeypatch.setattr(api_app, "run_traced", capture_traced_call)
    client = TestClient(api_app.app)

    response = client.post(
        "/chat/financial_report_qa",
        headers={"X-Request-ID": request_id},
        json={"query": "What was Costco revenue in 2025?"},
    )

    assert response.status_code == 200
    assert captured_metadata["request_id"] == request_id


def test_chat_validates_agent_response(monkeypatch) -> None:
    monkeypatch.setattr(api_app, "run_technique", lambda *args: {"final_status": "answered", "evidence": "not a list"})
    client = TestClient(api_app.app, raise_server_exceptions=False)

    response = client.post("/chat/financial_report_qa", json={"query": "What was Costco revenue in 2025?"})

    assert response.status_code == 500
    payload = response.json()
    assert payload["error"] == "internal_server_error"
    assert payload["message"] == "An unexpected error occurred. Please try again later."
    assert payload["request_id"] == response.headers["X-Request-ID"]


def test_chat_explanation_describes_refusal_and_review_reasons() -> None:
    refused = api_app.explain_chat_result(
        {
            "final_status": "refused",
            "security": {"decision": "block", "risk_labels": ["prompt_injection"]},
        }
    )
    review = api_app.explain_chat_result(
        {
            "final_status": "pending_human_review",
            "human_review": {"required": True, "triggers": ["citation_failure"]},
        }
    )

    assert "input security checks" in refused
    assert "prompt_injection" in refused
    assert "human review" in review
    assert "citation_failure" in review


def test_chat_endpoint_withholds_answer_without_valid_evidence_citations(monkeypatch) -> None:
    monkeypatch.setattr(api_app, "run_technique", fake_financial("Unsupported answer [unknown_chunk].", ["unknown_chunk"]))
    client = TestClient(api_app.app)

    response = client.post("/chat/financial_report_qa", json={"query": "What was Costco revenue in 2025?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] is None
    assert payload["final_status"] == "not_grounded"
    assert payload["grounding"]["status"] == "not_grounded"
    assert "withheld" in payload["explanation"]


def test_grounding_policy_marks_citation_checked_answer() -> None:
    result = {
        "final_status": "answered",
        "answer": "The policy says to keep receipts (chunk_1).",
        "citation_check": {
            "valid": True,
            "cited_chunk_ids": ["chunk_1"],
            "allowed_chunk_ids": ["chunk_1"],
        },
        "evidence": [{"chunk_id": "chunk_1", "preview": "receipt policy"}],
    }

    response = api_app.apply_grounding_policy(result)

    assert response["answer"] == "The policy says to keep receipts (chunk_1)."
    assert response["grounding"]["status"] == "grounded"
    assert response["grounding"]["cited_chunk_ids"] == ["chunk_1"]


def test_invalid_chat_request_returns_validation_error() -> None:
    client = TestClient(api_app.app)

    response = client.post("/chat/financial_report_qa", json={"query": ""})

    assert response.status_code == 422
    payload = response.json()
    assert payload["error"] == "validation_error"
    assert payload["message"] == "Request validation failed."
    assert payload["details"][0]["field"] == "body.query"
    assert payload["request_id"] == response.headers["X-Request-ID"]


def test_removed_domains_are_not_served() -> None:
    client = TestClient(api_app.app)

    for path in ("/chat/hr_policy", "/retrieve/travel_expense"):
        response = client.post(path, json={"query": "hello"})
        assert response.status_code == 404
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
    response = client.post("/multimodal/hr_policy/retrieve", json={"query": "hello"})
    assert response.status_code == 404
    payload = response.json()
    assert payload["error"] == "http_error"
    assert payload["message"].startswith("Unknown domain")


def test_unknown_route_explains_how_to_find_endpoints() -> None:
    client = TestClient(api_app.app)

    response = client.get("/not-a-route")

    assert response.status_code == 404
    payload = response.json()
    assert "Check the route path in the API docs." in payload["message"]
    assert payload["details"][0]["field"] == "path"


def test_unsupported_method_returns_actionable_error() -> None:
    client = TestClient(api_app.app)

    response = client.put("/health")

    assert response.status_code == 405
    assert response.json()["message"] == "This HTTP method is not supported for the requested endpoint."
    assert response.headers["allow"] == "GET"


def test_unhandled_service_exception_returns_safe_error(monkeypatch) -> None:
    def raise_service_error(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("internal service detail")

    monkeypatch.setattr(api_app, "run_technique", raise_service_error)
    client = TestClient(api_app.app, raise_server_exceptions=False)

    response = client.post("/chat/financial_report_qa", json={"query": "hello"})

    assert response.status_code == 500
    assert response.json()["error"] == "internal_server_error"
    assert "internal service detail" not in response.text


def test_fastapi_documents_response_models() -> None:
    openapi = api_app.app.openapi()
    schemas = openapi["components"]["schemas"]

    assert "FinancialChatResponse" in schemas
    assert "RetrieveResponse" in schemas
    assert api_app.app.docs_url == "/docs"
    assert api_app.app.redoc_url == "/redoc"
    chat_operation = openapi["paths"]["/chat/financial_report_qa"]["post"]
    assert chat_operation["summary"] == "Answer a financial-report question"
    assert "/chat/{domain}" not in openapi["paths"]
    assert chat_operation["description"]
    assert "429" in chat_operation["responses"]
    assert "explanation" in schemas["FinancialChatResponse"]["properties"]
    assert "grounding" in schemas["FinancialChatResponse"]["properties"]
    query_field = schemas["FinancialChatRequest"]["properties"]["query"]
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
    payload = response.json()
    assert payload["error"] == "rate_limit_exceeded"
    assert payload["message"] == "Rate limit exceeded. Please try again later."
    assert payload["details"][0]["type"] == "rate_limit"
    assert payload["request_id"] == response.headers["X-Request-ID"]
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
    assert "citation validation passed" in payload["explanation"]


def test_unknown_domain_returns_404() -> None:
    client = TestClient(api_app.app)

    response = client.post("/chat/not_a_domain", json={"query": "hello"})

    assert response.status_code == 404


def test_grounding_policy_keeps_not_available_reply() -> None:
    result = {"final_status": "not_available", "answer": "I don't have filings for Exxon Mobil.",
              "coverage": {"reason": "company_not_in_filings"}, "citation_check": {"valid": False, "cited_chunk_ids": []}, "evidence": []}

    response = api_app.apply_grounding_policy(result)

    assert response["answer"] == "I don't have filings for Exxon Mobil." and response["final_status"] == "not_available"
    assert response["grounding"]["status"] == "not_applicable" and "company_not_in_filings" in response["grounding"]["reason"]
    assert "company_not_in_filings" in api_app.explain_chat_result(response)


def test_grounding_policy_explains_citation_from_another_company() -> None:
    result = {"final_status": "not_grounded", "answer": "Withheld.", "company_check": {"valid": False, "foreign_chunk_ids": ["c1"]},
              "citation_check": {"valid": False, "cited_chunk_ids": ["c1"]}, "evidence": [{"chunk_id": "c1"}]}

    response = api_app.apply_grounding_policy(result)

    assert response["answer"] is None and "didn't ask about" in response["grounding"]["reason"]


def test_financial_requests_pass_the_sector_filter_to_the_agent(monkeypatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run_technique(technique: str, llm_provider: str, config: Any) -> dict[str, Any]:
        seen["chat"] = config.sector
        return {"technique": technique, "final_status": "insufficient_evidence", "answer": "", "evidence": []}

    def fake_retrieve(config: Any) -> list[dict[str, Any]]:
        seen["retrieve"] = config.sector
        return []

    monkeypatch.setattr(api_app, "run_technique", fake_run_technique)
    monkeypatch.setattr(api_app, "retrieve_evidence", fake_retrieve)
    client = TestClient(api_app.app)
    body = {"query": "net interest income", "sector": "banking", "llm_provider": "mock"}
    assert client.post("/chat/financial_report_qa", json=body).status_code == 200
    assert client.post("/retrieve/financial_report_qa", json={"query": "net interest income", "sector": "banking"}).status_code == 200
    assert seen == {"chat": "banking", "retrieve": "banking"}


def test_openapi_declares_the_api_key_and_public_health():
    api_app.app.openapi_schema = None
    spec = api_app.app.openapi()
    schemes = spec["components"]["securitySchemes"]
    assert schemes["ApiKeyHeader"] == {**schemes["ApiKeyHeader"], "type": "apiKey", "in": "header", "name": "X-API-Key"}
    assert schemes["BearerAuth"]["scheme"] == "bearer"
    assert spec["paths"]["/health"]["get"]["security"] == []
    assert "401" in spec["paths"]["/chat/financial_report_qa"]["post"]["responses"]
    assert "401" not in spec["paths"]["/health"]["get"]["responses"]
    assert spec["paths"]["/multimodal/{domain}/chat"]["post"]["summary"] == "Answer with the local multimodal pipeline"


def test_export_openapi_writes_spec_yaml_and_swagger(tmp_path):
    import json
    import yaml
    from scripts.api import export_openapi
    files = export_openapi.export(tmp_path, ["http://example.test:8000/"])
    spec = json.loads(files["json"].read_text(encoding="utf-8"))
    assert spec["servers"] == [{"url": "http://example.test:8000"}]
    assert "servers" not in api_app.app.openapi()                       # the live schema is not changed
    assert yaml.safe_load(files["yaml"].read_text(encoding="utf-8"))["paths"].keys() == spec["paths"].keys()
    page = files["html"].read_text(encoding="utf-8")
    assert "SwaggerUIBundle" in page and export_openapi.CDN in page and "</script>" not in page.split("const spec")[1].split("window.ui")[0]


def _fake_report(tmp_path, monkeypatch):
    import uuid as _uuid
    report_id = str(_uuid.uuid4())
    folder = tmp_path / "AAPL_2024" / report_id
    (folder / "preview").mkdir(parents=True)
    (folder / "aapl_fy2024_financial_report.pdf").write_bytes(b"%PDF-1.4 test")
    (folder / "preview" / "page_01.png").write_bytes(b"\x89PNG test")
    (folder / "manifest.json").write_text(json.dumps({
        "status": "generated_pending_visual_review", "visual_review": "pending", "trace_id": report_id,
        "company": "Apple Inc.", "year": 2024, "pdf_pages": 1}), encoding="utf-8")
    monkeypatch.setattr(api_app, "REPORT_ROOT", tmp_path)
    return report_id, folder


def test_report_files_and_approval(tmp_path, monkeypatch):
    from scripts.reports import generate_financial_report as reports
    report_id, folder = _fake_report(tmp_path, monkeypatch)
    approved = []

    def record(path):
        approved.append(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**manifest, "status": "completed", "visual_review": "passed"}), encoding="utf-8")
    monkeypatch.setattr(reports, "record_visual_review", record)
    client = TestClient(api_app.app)
    base = f"/reports/financial_report_qa/{report_id}"
    assert client.get(base).json()["status"] == "generated_pending_visual_review"
    pdf = client.get(f"{base}/pdf")
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    assert client.get(f"{base}/pages/1").content == b"\x89PNG test"
    assert client.get(f"{base}/pages/2").status_code == 404
    assert client.get("/reports/financial_report_qa/../../etc/pdf").status_code == 404
    assert client.get("/reports/financial_report_qa/not-a-uuid").status_code == 404
    assert client.post(f"{base}/review", json={"decision": "publish"}).status_code == 422
    response = client.post(f"{base}/review", json={"decision": "approve"})
    assert response.status_code == 200 and response.json()["status"] == "completed"
    assert approved == [folder / "manifest.json"]
    assert client.post(f"{base}/review", json={"decision": "approve"}).status_code == 409


def test_report_rejection_keeps_it_unpublished(tmp_path, monkeypatch):
    report_id, folder = _fake_report(tmp_path, monkeypatch)
    client = TestClient(api_app.app)
    response = client.post(f"/reports/financial_report_qa/{report_id}/review", json={"decision": "reject"})
    assert response.json()["status"] == "rejected"
    assert json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["visual_review"] == "rejected"
