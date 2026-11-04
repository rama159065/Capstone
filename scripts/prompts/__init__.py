"""Prompt templates used by the agents ("chat templates"), kept out of the agent code.

- ``financial``: financial agent techniques (baseline, few-shot, decomposition, self-consistency, agentic graph).
- ``domain``: the shared domain agent; ``{display_name}`` is the domain's display name.

Templates use ``str.format`` fields; literal braces are doubled. The Week 1 prompt-baseline
strategies are part of that frozen experiment and stay in ``scripts/evaluation/prompt_baselines.py``.
"""
