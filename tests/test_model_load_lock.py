"""Local models load one at a time per process: parallel loads (the API's warm-up and a first request) made
transformers leave a global meta-device patch behind, and every later load failed with "Cannot copy out of meta tensor"."""
from __future__ import annotations

import sys
import threading
import time
import types

from scripts.common import local_models
from scripts.rag import financial_ingest_qdrant_db as q


class FakeModel:
    active = peak = 0
    guard = threading.Lock()

    def __init__(self, name, device=None):
        with FakeModel.guard:
            FakeModel.active += 1
            FakeModel.peak = max(FakeModel.peak, FakeModel.active)
        time.sleep(0.05)
        with FakeModel.guard:
            FakeModel.active -= 1

    def get_sentence_embedding_dimension(self):
        return 384


def run_parallel(target, count=6):
    barrier = threading.Barrier(count)

    def go(i):
        barrier.wait()
        target(i)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_local_model_loads_never_overlap(monkeypatch):
    fake = types.SimpleNamespace(SentenceTransformer=FakeModel, CrossEncoder=FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    FakeModel.peak = 0
    local_models.text_model.cache_clear()
    local_models.reranker.cache_clear()
    loaders = [lambda i: q.Embedder("local", f"model-{i}"),
               lambda i: local_models.text_model(f"text-{i}"),
               lambda i: local_models.reranker(f"rerank-{i}")]
    run_parallel(lambda i: loaders[i % 3](i))
    local_models.text_model.cache_clear()
    local_models.reranker.cache_clear()
    assert FakeModel.peak == 1
