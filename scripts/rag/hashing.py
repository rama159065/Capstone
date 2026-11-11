"""Hashed term ids and the offline hashing embedding, shared by every index.

``term_id`` turns a word or number into the integer id of a sparse BM25 dimension. Qdrant's ``bm25`` vectors
(``financial_ingest_qdrant_db.sparse_vector``, the local multimodal ``docs_text``) store these ids, so they must never
change: a different hash would leave every existing index unable to match new queries.

``dense`` is the deterministic 384-d hashing "embedding" used by ``--embedding-provider hashing`` (offline tests and
smoke runs; it carries no meaning beyond shared words).

Moved here from ``multimodal_rag.py`` so the financial pipelines don't depend on the local multimodal stack;
``multimodal_rag`` re-exports them unchanged.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter


def tokens(text):
    """Lower-case word tokens (letters, digits, underscore)."""
    return re.findall(r"\b\w+\b", text.lower())


def term_id(term):
    """First 4 bytes of the term's SHA-256, as an unsigned int: the sparse vector dimension for this term."""
    return int.from_bytes(hashlib.sha256(term.encode()).digest()[:4], "big")


def dense(text, size=384):
    """Hashing embedding: log-scaled term counts folded into ``size`` dimensions, L2-normalised."""
    vector = [0.0] * size
    for term, count in Counter(tokens(text)).items():
        vector[term_id(term) % size] += 1 + math.log(count)
    norm = math.sqrt(sum(x*x for x in vector)) or 1
    return [x / norm for x in vector]
