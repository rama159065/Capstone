"""Pydantic schemas for structured financial answers and tool calls, plus grounding checks.

``FinancialAnswer`` is the JSON shape the chat templates ask for (scripts/evaluation/prompt_baselines.SCHEMA).
``validate_answer`` parses a model reply into it and then checks it against the evidence: every
citation must name an evidence chunk, and the stated value must appear in the evidence text.
Schema errors and grounding errors are reported separately, because they are different failures
(malformed output vs. unsupported claim).
"""
from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

Answerability = Literal["answerable", "not_answerable_from_corpus", "refuse"]


class FinancialAnswer(BaseModel):
    answer: str = Field(min_length=1, description="One or two sentences with [cN] citations.")
    answerability: Answerability
    value: str | None = Field(None, description="Main number exactly as written, e.g. '61,160'.")
    unit: str | None = Field(None, description="e.g. 'USD millions', 'USD per share', 'percent'.")
    period: str | None = Field(None, description="Fiscal period, e.g. 'fiscal 2025'.")
    citations: list[str] = Field(default_factory=list)

    @field_validator("value", mode="before")
    @classmethod
    def number_as_text(cls, value: Any) -> str | None:
        return None if value in (None, "", "null") else str(value)

    @field_validator("answerability", mode="before")
    @classmethod
    def normalise_answerability(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def consistent(self) -> "FinancialAnswer":
        if self.answerability == "answerable" and not self.citations:
            raise ValueError("an answerable reply must cite at least one evidence chunk")
        if self.answerability != "answerable" and self.value is not None:
            raise ValueError(f"answerability={self.answerability} but a value was given")
        return self


class ToolCall(BaseModel):
    """What the model may ask for before answering: one exact XBRL figure from SEC."""
    tool: Literal["get_xbrl_fact", "none"]
    ticker: str | None = Field(None, pattern=r"^[A-Z.]{1,6}$")
    metric: str | None = None
    fiscal_year: int | None = Field(None, ge=2000, le=2100)

    @model_validator(mode="after")
    def complete(self) -> "ToolCall":
        if self.tool == "get_xbrl_fact" and not (self.ticker and self.metric and self.fiscal_year):
            raise ValueError("get_xbrl_fact needs ticker, metric and fiscal_year")
        return self


def digits(text: str) -> str:
    return re.sub(r"[^\d.]", "", text or "").strip(".")


def validate_answer(parsed: dict | None, evidence: list[dict]) -> tuple[FinancialAnswer | None, list[str]]:
    """Return (model, errors). Errors: 'no_json', 'schema: ...', 'citation_not_in_evidence', 'value_not_in_evidence'."""
    if parsed is None:
        return None, ["no_json"]
    try:
        model = FinancialAnswer.model_validate(parsed)
    except ValidationError as exc:
        return None, [f"schema: {e['loc'][0] if e['loc'] else 'model'}: {e['msg']}" for e in exc.errors()]
    errors = []
    ids = {str(hit["chunk_id"]) for hit in evidence}
    if any(c.strip("[]") not in ids for c in model.citations):
        errors.append("citation_not_in_evidence")
    if model.value is not None:
        wanted = digits(model.value)
        haystack = " ".join(digits(t) for t in re.findall(r"\d[\d,.]*", " ".join(h.get("text", "") for h in evidence)))
        if wanted and wanted not in haystack.split():
            errors.append("value_not_in_evidence")
    return model, errors
