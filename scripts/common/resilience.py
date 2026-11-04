"""Bounded, observable calls. Never retry ambiguous model timeouts automatically."""
from collections import OrderedDict
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from contextvars import copy_context
from threading import BoundedSemaphore, RLock
import hashlib
import json
import time
import urllib.error
from uuid import uuid4
from scripts.observability.trace_log import current_trace_id, emit, SPAN

POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rag-tool")
SLOTS = BoundedSemaphore(4)
LOCK = RLock()
CACHE = OrderedDict()
INFLIGHT = {}


class CallFailure(RuntimeError):
    pass


@contextmanager
def timed_lock(lock, timeout):
    if not lock.acquire(timeout=timeout):
        raise TimeoutError("store_busy")
    try:
        yield
    finally:
        lock.release()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class Calls:
    def __init__(self, timeout=60):
        if not 0 < timeout <= 600:
            raise ValueError("timeout must be in (0, 600]")
        self.deadline = time.monotonic() + timeout
        self.trace_id = current_trace_id() or str(uuid4())
        self.events = []

    def run(self, name, fn, *, key=None, retry=False):
        if retry and not key:
            raise ValueError("Retries require an idempotency key")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise CallFailure("deadline_exceeded")
        started = time.monotonic()
        identifier = str(uuid4())
        event = {"node": name, "attempts": 0, "status": "error", "span_id": identifier}
        self.events.append(event)
        emit("call.started", name=name, span_id=identifier, parent_span_id=SPAN.get(),
             idempotency_key_hash=fingerprint(key) if key else None, remaining_seconds=round(remaining, 3))
        try:
            with LOCK:
                now = time.monotonic()
                for old in list(CACHE):
                    if CACHE[old][0] < now:
                        del CACHE[old]
                if key and key in CACHE:
                    event["status"] = "cached"
                    return CACHE[key][1]
                future = INFLIGHT.get(key) if key else None
                if future is None:
                    if not SLOTS.acquire(blocking=False):
                        raise CallFailure("capacity_exceeded")
                    def execute():
                        token = SPAN.set(identifier)
                        try:
                            from scripts.observability.langsmith_utils import run_traced
                            for attempt in range(2 if retry else 1):
                                event["attempts"] = attempt + 1
                                emit("call.attempt", name=name, attempt=attempt + 1)
                                try:
                                    return run_traced(name, fn, run_type="tool",
                                                      metadata={"trace_id": self.trace_id})
                                except urllib.error.HTTPError as exc:
                                    # Only explicit transient rejection is safe to retry.
                                    if exc.code not in {429, 503} or attempt or time.monotonic() >= self.deadline:
                                        raise
                                    delay = min(float(exc.headers.get("Retry-After", "0.1")), 2)
                                    if delay >= self.deadline - time.monotonic():
                                        raise
                                    emit("call.retry", name=name, http_status=exc.code, delay_seconds=max(0, delay))
                                    time.sleep(max(0, delay))
                        finally:
                            SPAN.reset(token)
                            SLOTS.release()
                    context = copy_context()
                    future = POOL.submit(context.run, execute)
                    if key:
                        INFLIGHT[key] = future
                        def complete(done):
                            with LOCK:
                                INFLIGHT.pop(key, None)
                                if not done.cancelled() and done.exception() is None:
                                    CACHE[key] = (time.monotonic() + 300, done.result())
                                    while len(CACHE) > 128:
                                        CACHE.popitem(last=False)
                        future.add_done_callback(complete)
                else:
                    emit("call.joined", name=name, span_id=identifier)
            result = future.result(timeout=max(0.001, self.deadline - time.monotonic()))
            event["status"] = "ok"
            return result
        except FutureTimeout as exc:
            event["status"] = "timeout"
            raise CallFailure("deadline_exceeded") from exc
        except Exception as exc:
            event["error_type"] = type(exc).__name__
            raise CallFailure(type(exc).__name__) from exc
        finally:
            event["elapsed_ms"] = round((time.monotonic() - started) * 1000)
            emit("call.completed", name=name, **event)
