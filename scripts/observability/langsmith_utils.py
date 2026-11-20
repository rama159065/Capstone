"""Optional LangSmith tracing helpers for capstone agents, the API and evals.

Two ways to turn tracing on, both needing ``LANGSMITH_API_KEY`` (or ``LANGCHAIN_API_KEY``):

* process-wide, for CLIs and evals: :func:`configure_langsmith` / ``--enable-langsmith`` set ``LANGSMITH_TRACING``;
  ``LANGSMITH_TRACING=true`` in the API's environment traces every request;
* per request, for the API: :func:`request_tracing` opens a ``langsmith.tracing_context`` for one request only, so a
  request with ``enable_langsmith=true`` never turns tracing on for other requests, and concurrent requests keep their
  own project and metadata.

Without a key nothing is sent and every helper just runs the function. The key is never logged or returned.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
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


def has_api_key() -> bool:
    return bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))


def tracing_requested_by_env() -> bool:
    return env_flag("LANGSMITH_TRACING") or env_flag("LANGCHAIN_TRACING_V2")


def _context_enabled() -> bool | None:
    """The enabled flag of the active ``tracing_context`` (None outside one, or without the langsmith package)."""
    try:
        from langsmith.run_helpers import get_tracing_context
    except Exception:
        return None
    enabled = get_tracing_context().get("enabled")
    return None if enabled is None else bool(enabled)


def langsmith_is_enabled() -> bool:
    """True when runs started now would be sent: a key, and tracing on for this request or this process."""
    if not has_api_key():
        return False
    context = _context_enabled()
    return context if context is not None else tracing_requested_by_env()


def default_project(fallback: str) -> str:
    return os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT") or fallback


def tracing_status() -> dict[str, Any]:
    """What /health reports about LangSmith (never the key)."""
    status: dict[str, Any] = {
        "api_key_set": has_api_key(),
        "tracing_all_requests": tracing_requested_by_env() and has_api_key(),
        "project": default_project("") or None,
        "endpoint": os.environ.get("LANGSMITH_ENDPOINT") or os.environ.get("LANGCHAIN_ENDPOINT") or None,
    }
    if tracing_requested_by_env() and not has_api_key():
        status["warning"] = "LANGSMITH_TRACING is set but LANGSMITH_API_KEY is not; nothing is traced"
    return status


@contextlib.contextmanager
def request_tracing(*, enabled: bool, project_name: str | None = None, tags: list[str] | None = None,
                    metadata: dict[str, Any] | None = None,
                    parent_headers: dict[str, str] | None = None) -> Iterator[dict[str, Any]]:
    """Trace what runs inside this block to LangSmith when ``enabled`` (or ``LANGSMITH_TRACING``) and a key is set.

    Yields the status dict the API returns as ``langsmith``. Unlike :func:`configure_langsmith` it changes no
    environment variable: tracing, project, tags and metadata live in a ``tracing_context`` for this request (and the
    worker threads it starts with ``copy_context``). Nested blocks keep the outer project, metadata and tags.

    ``parent_headers`` (the ``langsmith-trace`` and ``baggage`` headers of an incoming HTTP request, see
    :func:`outgoing_trace_headers`) continue the caller's trace: the runs here become children of the caller's run,
    in the caller's project, so Streamlit -> FastAPI -> agents is one trace.
    """
    if not (enabled or tracing_requested_by_env()):
        yield {"enabled": False, "reason": "not_requested"}
        return
    if not has_api_key():
        yield {"enabled": False, "reason": "missing_LANGSMITH_API_KEY",
               "hint": "Set LANGSMITH_API_KEY or LANGCHAIN_API_KEY to send traces."}
        return
    try:
        from langsmith import tracing_context
        from langsmith.run_helpers import get_tracing_context
    except Exception:
        yield {"enabled": False, "reason": "langsmith_not_installed"}
        return
    outer = get_tracing_context()
    merged_metadata = {**(outer.get("metadata") or {}), **(metadata or {})}
    merged_tags = list(dict.fromkeys([*(outer.get("tags") or []), *(tags or [])]))
    project = (outer.get("project_name") if outer.get("enabled") else None) or project_name
    parent = dict(parent_headers) if parent_headers and parent_headers.get("langsmith-trace") else None
    if parent:                     # the caller's trace applies once, at the top: inside a run, keep that run as parent
        from langsmith.run_helpers import get_current_run_tree
        if get_current_run_tree() is not None:
            parent = None
    if parent:
        project = baggage_project(parent.get("baggage")) or project
    with tracing_context(enabled=True, project_name=project, tags=merged_tags or None,
                         metadata=merged_metadata or None, **({"parent": parent} if parent else {})):
        yield {"enabled": True, "project": project or default_project("default"),
               "endpoint": os.environ.get("LANGSMITH_ENDPOINT") or os.environ.get("LANGCHAIN_ENDPOINT")}


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


def traced_llm(name: str, call: Callable[[], Any], messages: list[dict[str, str]], *, provider: str, model: str,
               temperature: float | None = None) -> Any:
    """Run ``call`` (one chat completion returning an object with ``.content``) as a LangSmith ``llm`` run.

    The run shows the role/content messages, the reply and, when the response carries ``usage`` (OpenAI format),
    token counts, so LangSmith can chart tokens and latency per model. Only the messages are recorded: never the
    endpoint's key or headers.
    """
    if not langsmith_is_enabled():
        return call()
    try:
        from langsmith import traceable
    except Exception:
        return call()
    holder: dict[str, Any] = {}

    def chat_completion(messages: list[dict[str, str]]) -> dict[str, Any]:
        response = holder["response"] = call()
        output: dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": str(response.content)}}]}
        usage = getattr(response, "usage", None) or {}
        if usage:
            output["usage_metadata"] = {"input_tokens": int(usage.get("prompt_tokens") or 0),
                                        "output_tokens": int(usage.get("completion_tokens") or 0),
                                        "total_tokens": int(usage.get("total_tokens") or 0)}
        return output

    metadata: dict[str, Any] = {"ls_provider": provider, "ls_model_name": model}
    if temperature is not None:
        metadata["ls_temperature"] = temperature
    traceable(name=name, run_type="llm", metadata=metadata)(chat_completion)(messages)
    return holder["response"]


DOCUMENT_FIELDS = ("chunk_id", "doc_id", "ticker", "year", "sector", "page", "rank", "content_type", "reader",
                   "text_source", "retrieval_paths", "retrieval_ranks", "rrf_score", "rerank_score", "score")


def as_documents(hits: list[dict[str, Any]], text_chars: int = 1500) -> dict[str, Any]:
    """Evidence hits in the form LangSmith shows for a retriever run: page_content plus a few metadata fields."""
    documents = []
    for hit in hits or []:
        payload = hit.get("payload") if isinstance(hit.get("payload"), dict) else {}   # ColPali pages: {doc_id, page, rank, payload}
        hit = {**{k: v for k, v in payload.items() if k in DOCUMENT_FIELDS}, **hit}
        text = str(hit.get("text") or hit.get("preview") or payload.get("text") or payload.get("preview") or "")
        documents.append({"type": "Document", "page_content": text[:text_chars],
                          "metadata": {key: hit[key] for key in DOCUMENT_FIELDS if hit.get(key) is not None}})
    return {"documents": documents}


def traced_retrieval(name: str, call: Callable[[], list[dict[str, Any]]], *, query: str,
                     filters: dict[str, Any] | None = None, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Run one search as a LangSmith ``retriever`` run named ``name`` (for example ``retrieve.financial_vlm``).

    The run's input is the query and its filters; its output is the hits as documents (text cut to 1,500 characters),
    so a trace shows which backend answered and exactly what evidence the agent received.
    """
    if not langsmith_is_enabled():
        return call()
    try:
        from langsmith import traceable
    except Exception:
        return call()
    holder: dict[str, Any] = {}

    def search(query: str, filters: dict[str, Any]) -> dict[str, Any]:
        hits = holder["hits"] = call()
        return as_documents(hits)

    traceable(name=name, run_type="retriever", metadata=metadata or {})(search)(query, filters or {})
    return holder["hits"]


TRACE_HEADERS = ("langsmith-trace", "baggage")


def baggage_project(baggage: str | None) -> str | None:
    """The ``langsmith-project`` entry of a W3C baggage header (the caller's project), if any."""
    from urllib.parse import unquote
    for item in (baggage or "").split(","):
        key, _, value = item.strip().partition("=")
        if key == "langsmith-project" and value:
            return unquote(value)
    return None


def outgoing_trace_headers() -> dict[str, str]:
    """Headers that let a downstream service continue the current run's trace ({} outside a traced run)."""
    if not langsmith_is_enabled():
        return {}
    try:
        from langsmith.run_helpers import get_current_run_tree
    except Exception:
        return {}
    run = get_current_run_tree()
    return dict(run.to_headers()) if run is not None else {}
