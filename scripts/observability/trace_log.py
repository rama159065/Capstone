"""Persistent JSONL traces with logical thread/run IDs, propagated into workers."""
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from threading import RLock, get_ident
import time
from uuid import UUID, uuid4

ACTIVE = ContextVar("rag_trace", default=None)
SPAN = ContextVar("rag_span", default=None)


def current_trace_id():
    trace = ACTIVE.get()
    return trace.trace_id if trace else None


def emit(event, **fields):
    trace = ACTIVE.get()
    if trace:
        trace.emit(event, **fields)


class TraceLog:
    def __init__(self, directory, *, thread_id=None, trace_id=None):
        self.thread_id = str(UUID(thread_id)) if thread_id else str(uuid4())
        self.trace_id = str(UUID(trace_id)) if trace_id else str(uuid4())
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{self.trace_id}.jsonl"
        # Unique run IDs cannot silently append to a previous run.
        self.path.open("x", encoding="utf-8").close()
        self.lock, self.sequence = RLock(), 0

    def emit(self, event, **fields):
        with self.lock:
            self.sequence += 1
            row = {**fields, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                   "sequence": self.sequence, "thread_id": self.thread_id, "trace_id": self.trace_id,
                   "span_id": fields.get("span_id", SPAN.get()), "event": event,
                   "worker_thread_id": get_ident()}
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    @contextmanager
    def activate(self):
        token = ACTIVE.set(self)
        try:
            yield self
        finally:
            ACTIVE.reset(token)

    @contextmanager
    def span(self, name, **fields):
        identifier, parent = str(uuid4()), SPAN.get()
        token = SPAN.set(identifier)
        started = time.monotonic()
        self.emit("span.started", name=name, parent_span_id=parent, **fields)
        try:
            yield identifier
        except Exception as exc:
            self.emit("span.failed", name=name, error_type=type(exc).__name__,
                      elapsed_ms=round((time.monotonic() - started) * 1000))
            raise
        else:
            self.emit("span.completed", name=name, elapsed_ms=round((time.monotonic() - started) * 1000))
        finally:
            SPAN.reset(token)


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
