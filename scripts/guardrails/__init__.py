"""Deterministic safety guardrails shared by the financial agent and the domain agents.

- ``patterns``: the regex sets (prompt injection, code/tool payloads, secrets, sensitive data).
- ``checks``: input guard, evidence (prompt-injection) guard, output content filter, redaction.

Each agent binds these checks to its own pattern set; see ``scripts/agents``.
"""
