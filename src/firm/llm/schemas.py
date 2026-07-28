"""Pydantic schemas for validating structured (``json_mode``) LLM responses.

Every LLM-enhanced agent asks the model for a specific JSON shape in its
prompt (documented on each model below), but previously the parsed dict
was consumed ad hoc at each call site via ``.get(...)`` plus scattered
``float()`` casts with no single source of truth for what a valid
response actually looks like — e.g. ``trader_llm.py`` did a bare
``{k: float(v) for k, v in adjusted.items()}`` with no per-field context
on failure, and ``rationale``/``notes``/``reasoning`` fields were never
type-checked at all before flowing into contracts that declare them
``str``.

These schemas make each prompt's JSON contract explicit and centralise
the coercion:

- Missing fields fall back to sensible defaults instead of raising.
- Unexpected extra keys are ignored rather than erroring.
- Free-text fields (``rationale``/``notes``/``reasoning``) are coerced to
  ``str`` so a hallucinated non-string value can never violate the
  ``str``-typed contracts (:class:`firm.contracts.models.Thesis`,
  :class:`~firm.contracts.models.TradeProposal`, ...) downstream.
- Numeric fields that fail to coerce become ``NaN`` rather than raising,
  preserving the existing *per-field* fallback contract used by
  :meth:`firm.agents.llm.base_llm_agent.LLMAgentMixin._bounded_override`
  and friends (which already treat NaN exactly like a missing value —
  see their ``score != score`` checks) — one malformed numeric field
  doesn't have to discard an otherwise-valid response.
- List/bool fields (``additional_violations``, ``additional_actions``,
  ``override_approval``) filter out wrong-typed entries instead of
  rejecting the whole response, matching the ``isinstance`` filtering
  ``risk_llm.py`` did by hand before.
- ``adjusted_targets`` (portfolio review) is deliberately *not* given
  this leniency: a value that can't coerce to ``float`` fails the whole
  model, exactly as the original bare dict-comprehension crashed and
  fell back to the quant proposal in full — partial application of a
  weight adjustment is not a safe degradation for portfolio targets.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

log = logging.getLogger(__name__)


def _coerce_float_or_nan(v: Any) -> Any:
    """Best-effort numeric coercion; unparseable values become NaN.

    NaN (rather than raising) lets the whole response still validate so
    a bad numeric field doesn't discard sibling fields that *were* valid
    (e.g. a good ``rationale`` alongside a hallucinated ``score``) —
    the NaN is then caught by the existing bounded-override NaN check,
    which falls back to the quant value for that field alone.
    """
    if isinstance(v, bool):  # bool is a numeric subtype in Python; reject explicitly
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _coerce_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    return str(v)


def _coerce_str_list(v: Any) -> list[str]:
    if not isinstance(v, list):
        return []
    return [item for item in v if isinstance(item, str)]


class AnalystEnhancementResponse(BaseModel):
    """Fundamental/sentiment/technical analyst enhancement prompt contract.

    ``{"score": float (-1 to 1), "confidence": float (0 to 1), "rationale": "..."}``
    """

    model_config = ConfigDict(extra="ignore")

    score: float = 0.0
    confidence: float = 0.0
    rationale: str = ""

    _coerce_score = field_validator("score", mode="before")(_coerce_float_or_nan)
    _coerce_confidence = field_validator("confidence", mode="before")(_coerce_float_or_nan)
    _coerce_rationale = field_validator("rationale", mode="before")(_coerce_str)


class ThesisEnhancementResponse(BaseModel):
    """Bull/bear researcher enhancement prompt contract.

    ``{"conviction": float (0 to 1), "rationale": "..."}``
    """

    model_config = ConfigDict(extra="ignore")

    conviction: float = 0.0
    rationale: str = ""

    _coerce_conviction = field_validator("conviction", mode="before")(_coerce_float_or_nan)
    _coerce_rationale = field_validator("rationale", mode="before")(_coerce_str)


class DebateEnhancementResponse(BaseModel):
    """Debate/synthesis enhancement prompt contract.

    ``{"net_conviction": float (-1 to 1), "reasoning": "..."}``
    """

    model_config = ConfigDict(extra="ignore")

    net_conviction: float = 0.0
    reasoning: str = ""

    _coerce_net_conviction = field_validator("net_conviction", mode="before")(_coerce_float_or_nan)
    _coerce_reasoning = field_validator("reasoning", mode="before")(_coerce_str)


class RiskReviewResponse(BaseModel):
    """LLM risk-review prompt contract.

    ``{"additional_violations": [...], "additional_actions": [...], "override_approval": null|bool}``
    """

    model_config = ConfigDict(extra="ignore")

    additional_violations: list[str] = []
    additional_actions: list[str] = []
    override_approval: bool | None = None

    _coerce_violations = field_validator("additional_violations", mode="before")(_coerce_str_list)
    _coerce_actions = field_validator("additional_actions", mode="before")(_coerce_str_list)

    @field_validator("override_approval", mode="before")
    @classmethod
    def _coerce_override(cls, v: Any) -> Any:
        return v if isinstance(v, bool) or v is None else None


class PortfolioReviewResponse(BaseModel):
    """LLM portfolio-review prompt contract.

    ``{"adjusted_targets": {"SYMBOL": weight, ...}, "notes": "..."}``

    ``adjusted_targets`` intentionally has no lenient before-validator: an
    unparseable weight fails the whole model (see module docstring).
    """

    model_config = ConfigDict(extra="ignore")

    adjusted_targets: dict[str, float] = {}
    notes: str = ""

    _coerce_notes = field_validator("notes", mode="before")(_coerce_str)


def parse_llm_response(
    model_cls: type[BaseModel], raw: Any, *, context: str,
) -> BaseModel | None:
    """Validate a parsed LLM JSON response against *model_cls*.

    Returns ``None`` on validation failure (logged at ``warning`` with
    *context*, e.g. ``"AAPL/momentum"``) so callers can fall back to
    their quant-only value — this never raises, matching the "an LLM
    hiccup must never take down the pipeline" contract every call site
    already relies on.
    """
    try:
        return model_cls.model_validate(raw)
    except ValidationError as exc:
        log.warning(
            "LLM response failed %s schema validation for %s: %s",
            model_cls.__name__, context, exc,
        )
        return None
