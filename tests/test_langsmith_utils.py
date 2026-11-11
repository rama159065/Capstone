from __future__ import annotations

import argparse

from scripts.observability import langsmith_utils


def test_configure_langsmith_is_optional_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    status = langsmith_utils.configure_langsmith(enabled=True, project_name="test-project")

    assert status["enabled"] is False
    assert status["reason"] == "missing_LANGSMITH_API_KEY"
    assert langsmith_utils.run_traced("test", lambda value: value + 1, 2) == 3


def test_configure_langsmith_sets_expected_env(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    status = langsmith_utils.configure_langsmith(enabled=True, project_name="capstone-test")

    assert status["enabled"] is True
    assert status["project"] == "capstone-test"
    assert langsmith_utils.langsmith_is_enabled() is True


def test_add_langsmith_args_defaults() -> None:
    parser = argparse.ArgumentParser()
    langsmith_utils.add_langsmith_args(parser, default_project="default-project")

    args = parser.parse_args([])

    assert args.enable_langsmith is False
    assert args.langsmith_project == "default-project"
    assert args.langsmith_tag == []
