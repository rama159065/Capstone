from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Tests never read or write the real storage layout (scripts/common/storage.py): RAG_HOME points at an empty
# temporary folder, and no local model copy is found, so the profile turns on no reranker and no CA certificate.
import os
import tempfile

_TEST_HOME = tempfile.mkdtemp(prefix="rag_home_test_")
os.environ.setdefault("RAG_HOME", _TEST_HOME)
os.environ.setdefault("RAG_MODELS_DIR", os.path.join(_TEST_HOME, "models"))

import pytest


@pytest.fixture(autouse=True)
def _fresh_api_rate_limiter():
    """Each test starts with an empty per-IP rate limiter. Every TestClient shares one client IP and the limiter
    lives for the whole process, so without this the suite's total request count (not any one test) hits the
    60-per-minute limit and later API tests get 429."""
    api = sys.modules.get("scripts.api.api_app")
    if api is not None:
        api.app.state.rate_limiter = api.InMemoryRateLimiter()
    yield
