"""Input guard, evidence guard and output filter, parameterised by pattern set.

Decisions: any blocking label -> ``block``; any other label -> ``review``; none -> ``allow``.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from scripts.guardrails.patterns import (
    BASE64_LIKE_PATTERN,
    INPUT_BLOCKING_LABELS,
    OUTPUT_BLOCKING_LABELS,
    SECRET_PATTERNS,
    SENSITIVE_DATA_PATTERNS,
)


def canonicalize(message: str) -> str:
    """Drop NUL and zero-width characters and collapse whitespace."""
    message = re.sub(r"[​-‏﻿]", "", message.replace("\x00", " "))
    return re.sub(r"\s+", " ", message).strip()


def regex_labels(message: str, patterns: list[str], label: str) -> list[str]:
    return [label for pattern in patterns if re.search(pattern, message, flags=re.IGNORECASE | re.DOTALL)]


def redact(message: str) -> str:
    redacted = message
    for pattern in SECRET_PATTERNS:
        redacted = re.sub(pattern, "[REDACTED_SECRET]", redacted, flags=re.IGNORECASE)
    for pattern in SENSITIVE_DATA_PATTERNS:
        redacted = re.sub(pattern, "[REDACTED_SENSITIVE_DATA]", redacted, flags=re.IGNORECASE)
    return redacted


def decide(labels: Iterable[str], blocking: set[str]) -> str:
    labels = set(labels)
    return "block" if labels & blocking else "review" if labels else "allow"


def input_guard(message: str, injection: list[str], code: list[str], *, detect_encoded: bool = False,
                max_chars: int | None = None) -> dict[str, Any]:
    sanitized = canonicalize(message)
    labels = (regex_labels(sanitized, injection, "prompt_injection")
              + regex_labels(sanitized, code, "code_or_tool_payload")
              + regex_labels(sanitized, SECRET_PATTERNS, "secret_or_credential")
              + regex_labels(sanitized, SENSITIVE_DATA_PATTERNS, "sensitive_data"))
    if detect_encoded and re.search(BASE64_LIKE_PATTERN, sanitized):
        labels.append("encoded_or_obfuscated_payload")
    if max_chars is not None and len(sanitized) > max_chars:
        labels.append("oversized_message")
    labels = sorted(set(labels))
    return {"decision": decide(labels, INPUT_BLOCKING_LABELS), "risk_labels": labels,
            "sanitized_query": redact(sanitized)}


def injection_labels(text: str, injection: list[str], code: list[str]) -> list[str]:
    return sorted(set(regex_labels(text, injection, "prompt_injection") + regex_labels(text, code, "code_or_tool_payload")))


def evidence_guard(hits: list[dict[str, Any]], injection: list[str], code: list[str]) -> dict[str, Any]:
    """Flag retrieved chunks that carry instructions or code (indirect prompt injection)."""
    flagged = []
    for hit in hits:
        labels = injection_labels(str(hit.get("text", "")), injection, code)
        if labels:
            flagged.append({"chunk_id": hit.get("chunk_id"), "risk_labels": labels})
    return {"safe": not flagged, "flagged_chunks": flagged,
            "risk_labels": sorted({label for row in flagged for label in row["risk_labels"]})}


def output_filter(answer: str, injection: list[str], code: list[str],
                  review_terms: Iterable[tuple[Iterable[str], str]] = ()) -> dict[str, Any]:
    """Block echoed injections, code and secrets; send policy-sensitive wording to review."""
    text = canonicalize(answer)
    labels = (regex_labels(text, injection, "prompt_injection_echo")
              + regex_labels(text, code, "code_or_tool_payload")
              + regex_labels(text, SECRET_PATTERNS, "secret_or_credential")
              + regex_labels(text, SENSITIVE_DATA_PATTERNS, "sensitive_data"))
    lower = text.lower()
    for terms, label in review_terms:
        if any(term in lower for term in terms):
            labels.append(label)
    labels = sorted(set(labels))
    return {"decision": decide(labels, OUTPUT_BLOCKING_LABELS), "risk_labels": labels,
            "redacted_preview": redact(text)[:800]}
