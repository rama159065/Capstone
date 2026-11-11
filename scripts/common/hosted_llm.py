"""Settings shared by the hosted OpenAI-compatible chat clients (e.g. the course's Qwen/Qwen3-8B service).

Environment variables (all optional except the usual LLM_API_BASE / LLM_API_KEY / LLM_MODEL):
    LLM_CA_CERT          CA certificate (PEM/CRT) that signed the endpoint's TLS certificate, e.g. vm04-ca.crt.
                         Used only for LLM calls (other HTTPS traffic keeps the system trust store).
    LLM_MAX_TOKENS       Upper limit on generated tokens per call (sent as max_tokens). The course endpoint
                         allows 1-1024 (default 512) and 8192 tokens of input + output combined.
    LLM_MAX_CONCURRENCY  Requests in flight from this process (default 1: one active request per group key).
    LLM_RETRY_DELAYS     Back-off in seconds for HTTP 429/503 (default "2,4,8"; "" disables retries).
    LLM_ENABLE_THINKING  "1" keeps Qwen3's reasoning mode. Default "0": we send
                         chat_template_kwargs={"enable_thinking": false}, so answers come back faster and JSON
                         replies (planner, support check) parse cleanly.

Smoke test (same request as the course's curl example):
    python -m scripts.common.hosted_llm "Explain RAG in three steps."
"""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


@lru_cache(maxsize=4)
def _context_for(cafile: str) -> ssl.SSLContext:
    path = Path(os.path.expandvars(cafile)).expanduser()
    if not path.is_file():
        raise RuntimeError(f"LLM_CA_CERT not found: {path}")
    return ssl.create_default_context(cafile=str(path))


def ssl_context() -> ssl.SSLContext | None:
    cafile = os.getenv("LLM_CA_CERT", "").strip()
    return _context_for(cafile) if cafile else None


@lru_cache(maxsize=1)
def _gate() -> threading.BoundedSemaphore:
    return threading.BoundedSemaphore(max(1, int(os.getenv("LLM_MAX_CONCURRENCY", "1") or 1)))


def retry_delays() -> list[float]:
    raw = os.getenv("LLM_RETRY_DELAYS", "2,4,8")
    return [float(part) for part in raw.split(",") if part.strip()]


def _urlopen(request: urllib.request.Request, timeout: float):
    context = ssl_context()
    if context is None:
        return urllib.request.urlopen(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout, context=context)


def open_url(request: urllib.request.Request, timeout: float):
    """POST to the hosted LLM: trusts LLM_CA_CERT, sends one request at a time, backs off on 429/503.

    The course endpoint allows one active request per group key (60/min), so calls from this process
    queue on a semaphore (LLM_MAX_CONCURRENCY, default 1) instead of colliding into 429s. Explicit
    429/503 rejections are retried after LLM_RETRY_DELAYS seconds (default 2,4,8; Retry-After wins when
    larger); timeouts and other errors are never retried here.
    """
    if not _gate().acquire(timeout=timeout):
        raise RuntimeError(f"Hosted LLM busy: waited {timeout:.0f}s for the active request to finish")
    try:
        delays = retry_delays()
        for attempt in range(len(delays) + 1):
            try:
                return _urlopen(request, timeout)
            except urllib.error.HTTPError as exc:
                if exc.code in {429, 503} and delays and attempt >= len(delays):
                    # Retries are spent: report it without letting an outer layer retry all over again.
                    raise RuntimeError(f"Hosted LLM HTTP {exc.code} after {len(delays)} retries: "
                                       f"{HINTS[exc.code]}") from exc
                if exc.code not in {429, 503} or attempt >= len(delays):
                    if exc.code in HINTS:
                        logger.warning("Hosted LLM HTTP %s: %s", exc.code, HINTS[exc.code])
                    raise
                try:
                    wait = max(delays[attempt], float(exc.headers.get("Retry-After") or 0))
                except (TypeError, ValueError):
                    wait = delays[attempt]
                logger.info("Hosted LLM HTTP %s; retry %d/%d in %.0fs", exc.code, attempt + 1, len(delays), wait)
                time.sleep(wait)
    finally:
        _gate().release()


HINTS = {
    400: "check the JSON, model ID and context length (input + output <= 8192 tokens; max_tokens 1-1024)",
    401: "check LLM_API_KEY (your group key) and LLM_API_BASE (your assigned URL)",
    404: "LLM_API_BASE must end in /v1 (the client adds /chat/completions; do not repeat /v1)",
    429: "too many requests: one active request per key, 60 per minute",
    503: "the VM is busy or the model is not ready yet",
}


def payload_extras(*, max_tokens: int | None = None) -> dict:
    """Extra request fields: max_tokens and Qwen3's thinking switch."""
    extras: dict = {}
    limit = max_tokens or int(os.getenv("LLM_MAX_TOKENS", "0") or 0)
    if limit:
        extras["max_tokens"] = limit
    if os.getenv("LLM_ENABLE_THINKING", "0").strip().lower() not in {"1", "true", "yes"}:
        extras["chat_template_kwargs"] = {"enable_thinking": False}
    return extras


def clean_content(text: str | None) -> str:
    """Remove Qwen3 <think> blocks; an unclosed block (cut off by max_tokens) leaves no answer."""
    text = THINK_BLOCK.sub("", text or "")
    if text.lstrip().lower().startswith("<think>"):
        return ""
    return text.strip()


def chat(messages: list[dict], *, api_base: str | None = None, api_key: str | None = None,
         model: str | None = None, temperature: float = 0.0, max_tokens: int | None = None,
         timeout: float = 90) -> str:
    """One /chat/completions call; returns choices[0].message.content without <think> blocks."""
    api_base = (api_base or os.environ["LLM_API_BASE"]).rstrip("/")
    body = {"model": model or os.environ["LLM_MODEL"], "messages": messages, "temperature": temperature,
            **payload_extras(max_tokens=max_tokens)}
    request = urllib.request.Request(
        f"{api_base}/chat/completions", data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Authorization": f"Bearer {api_key or os.environ['LLM_API_KEY']}",
                 "Content-Type": "application/json"})
    with open_url(request, timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return clean_content(data["choices"][0]["message"]["content"])


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Explain RAG in three steps."
    print(chat([{"role": "user", "content": question}], max_tokens=256))
