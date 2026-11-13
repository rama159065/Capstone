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
