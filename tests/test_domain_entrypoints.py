from __future__ import annotations

import importlib

import pytest

from scripts.domain_common.rag_agent import DomainConfig


DOMAINS = (
    "procurement_vendor",
    "travel_expense",
    "competitive_research",
    "incident_postmortem",
    "hr_policy",
)


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_config_integrity(domain: str) -> None:
    module = importlib.import_module(f"scripts.{domain}.domain_config")
    config = module.DOMAIN_CONFIG

    assert isinstance(config, DomainConfig)
    assert config.domain == domain
    assert config.display_name
    assert config.description
    assert config.intent_terms
    assert config.review_terms
    assert config.source_types
    assert config.default_query


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_index_entrypoint_delegates_config(domain: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module(f"scripts.{domain}.index")
    called: dict[str, str] = {}

    def fake_index_cli(config: DomainConfig) -> int:
        called["domain"] = config.domain
        return 17

    monkeypatch.setattr(module, "index_cli", fake_index_cli)

    assert module.index_cli(module.DOMAIN_CONFIG) == 17
    assert called["domain"] == domain


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_agent_entrypoint_delegates_config(domain: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module(f"scripts.{domain}.agent")
    called: dict[str, str] = {}

    def fake_agent_cli(config: DomainConfig) -> int:
        called["domain"] = config.domain
        return 23

    monkeypatch.setattr(module, "agent_cli", fake_agent_cli)

    assert module.agent_cli(module.DOMAIN_CONFIG) == 23
    assert called["domain"] == domain


@pytest.mark.parametrize("domain", DOMAINS)
def test_domain_html_app_entrypoint_delegates_config(domain: str, monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module(f"scripts.{domain}.html_app")
    called: dict[str, str] = {}

    def fake_streamlit_app(config: DomainConfig) -> None:
        called["domain"] = config.domain

    monkeypatch.setattr(module, "streamlit_app", fake_streamlit_app)

    module.streamlit_app(module.DOMAIN_CONFIG)

    assert called["domain"] == domain
