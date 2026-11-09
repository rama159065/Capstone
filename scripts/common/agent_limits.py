"""Step cap for the LangGraph agents (Track B: at most 10 steps per query).

``AGENT_MAX_STEPS`` counts node executions. LangGraph's ``recursion_limit`` stops a run
*at* the limit, so the graphs are invoked with ``max_steps() + 1``. The longest path is
the financial graph with its one repair: security, plain text, plan, retrieve, evidence
guard, answer, critic, answer, critic, governance = 10 steps. The domain graph needs 9.

A run that hits the cap fails closed: no answer, ``final_status = "step_limit"``.
"""
from __future__ import annotations

import os

DEFAULT_MAX_STEPS = 10
STEP_LIMIT_ANSWER = "No answer: the agent reached its step limit before a verified answer was ready."


def max_steps() -> int:
    try:
        return max(1, int(os.getenv("AGENT_MAX_STEPS", DEFAULT_MAX_STEPS)))
    except ValueError:
        return DEFAULT_MAX_STEPS


def graph_config() -> dict:
    """``config`` for ``compiled_graph.invoke``."""
    return {"recursion_limit": max_steps() + 1}
