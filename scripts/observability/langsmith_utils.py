"""Optional LangSmith tracing helpers for capstone agents and evals."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, TypeVar


T = TypeVar("T")


TRUE_VALUES = {"1", "true", "yes", "on"}


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def configure_langsmith(
    *,
    enabled: bool,
    project_name: str | None = None,
    endpoint: str | None = None,
) -> dict[str, Any]:
    """Enable LangSmith tracing for this process when credentials are present."""
    api_key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not enabled:
        return {"enabled": False, "reason": "not_requested"}
    if not api_key:
        return {
            "enabled": False,
            "reason": "missing_LANGSMITH_API_KEY",
            "hint": "Set LANGSMITH_API_KEY or LANGCHAIN_API_KEY to send traces.",
        }

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if project_name:
        os.environ["LANGSMITH_PROJECT"] = project_name
        os.environ["LANGCHAIN_PROJECT"] = project_name
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    return {
        "enabled": True,
        "project": os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT"),
        "endpoint": os.environ.get("LANGSMITH_ENDPOINT") or os.environ.get("LANGCHAIN_ENDPOINT"),
    }


def langsmith_is_enabled() -> bool:
    has_key = bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))
    tracing_requested = env_flag("LANGSMITH_TRACING") or env_flag("LANGCHAIN_TRACING_V2")
    return has_key and tracing_requested


def add_langsmith_args(parser: Any, *, default_project: str) -> None:
    parser.add_argument(
        "--enable-langsmith",
        action="store_true",
        help="Enable LangSmith tracing for this run. Requires LANGSMITH_API_KEY.",
    )
    parser.add_argument(
        "--langsmith-project",
        default=default_project,
        help="LangSmith project name used when --enable-langsmith is set.",
    )
    parser.add_argument(
        "--langsmith-endpoint",
        help="Optional LangSmith endpoint override.",
    )
    parser.add_argument(
        "--langsmith-tag",
        action="append",
        default=[],
        help="Extra LangSmith tag. Can be provided multiple times.",
    )


def maybe_enable_from_args(args: Any) -> dict[str, Any]:
    enabled = bool(getattr(args, "enable_langsmith", False)) or env_flag("LANGSMITH_TRACING")
    return configure_langsmith(
        enabled=enabled,
        project_name=getattr(args, "langsmith_project", None),
        endpoint=getattr(args, "langsmith_endpoint", None),
    )


def run_traced(
    name: str,
    func: Callable[..., T],
    *args: Any,
    run_type: str = "chain",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> T:
    """Run a callable inside a LangSmith trace when tracing is configured."""
    if not langsmith_is_enabled():
        return func(*args, **kwargs)
    try:
        from langsmith import traceable
    except Exception:
        return func(*args, **kwargs)

    wrapped = traceable(
        name=name,
        run_type=run_type,
        tags=tags or [],
        metadata=metadata or {},
    )(func)
    return wrapped(*args, **kwargs)
