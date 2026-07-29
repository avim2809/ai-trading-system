"""Unit tests for firm.llm.schemas — Pydantic validation of LLM json_mode
responses, replacing the ad hoc `.get()` + `float()` parsing that used to
live at each LLM-enhanced agent's call site.
"""

from __future__ import annotations

import math

import pytest

from firm.llm.schemas import (
    AnalystEnhancementResponse,
    DebateEnhancementResponse,
    DecisionReflection,
    PortfolioReviewResponse,
    RiskReviewResponse,
    ThesisEnhancementResponse,
    parse_llm_response,
)


class TestAnalystEnhancementResponse:
    def test_valid_response_parses_as_is(self):
        r = AnalystEnhancementResponse.model_validate(
            {"score": 0.4, "confidence": 0.8, "rationale": "solid fundamentals"}
        )
        assert r.score == pytest.approx(0.4)
        assert r.confidence == pytest.approx(0.8)
        assert r.rationale == "solid fundamentals"

    def test_missing_fields_default_sensibly(self):
        r = AnalystEnhancementResponse.model_validate({})
        assert r.score == 0.0
        assert r.confidence == 0.0
        assert r.rationale == ""

    def test_extra_keys_are_ignored(self):
        r = AnalystEnhancementResponse.model_validate(
            {"score": 0.1, "confidence": 0.2, "rationale": "x", "unexpected_key": 123}
        )
        assert r.score == pytest.approx(0.1)

    def test_out_of_range_values_pass_through_unclamped(self):
        """Clamping to the documented [-1,1]/[0,1] ranges is deliberately
        left to LLMAgentMixin._bounded_override (which needs symbol/strategy
        context for its log messages) — the schema's job is just structural
        validation and type coercion."""
        r = AnalystEnhancementResponse.model_validate({"score": 5.0, "confidence": -0.3})
        assert r.score == 5.0
        assert r.confidence == -0.3

    def test_numeric_strings_coerce_to_float(self):
        r = AnalystEnhancementResponse.model_validate({"score": "0.25", "confidence": "0.9"})
        assert r.score == pytest.approx(0.25)
        assert r.confidence == pytest.approx(0.9)

    def test_non_numeric_score_becomes_nan_not_a_raised_error(self):
        """NaN (not an exception) so a bad `score` doesn't discard a
        sibling `confidence`/`rationale` that parsed fine — the NaN is
        then caught by _bounded_override's existing NaN check."""
        r = AnalystEnhancementResponse.model_validate(
            {"score": "not-a-number", "confidence": 0.7, "rationale": "kept"}
        )
        assert math.isnan(r.score)
        assert r.confidence == pytest.approx(0.7)
        assert r.rationale == "kept"

    def test_nan_literal_passes_through_as_nan(self):
        r = AnalystEnhancementResponse.model_validate({"score": float("nan")})
        assert math.isnan(r.score)

    def test_none_rationale_becomes_empty_string(self):
        r = AnalystEnhancementResponse.model_validate({"rationale": None})
        assert r.rationale == ""

    def test_non_string_rationale_is_stringified_not_dropped(self):
        """The original `.get()`-based call sites never type-checked
        `rationale` before handing it to a `str`-typed dataclass field —
        a hallucinated non-string value would violate that contract."""
        r = AnalystEnhancementResponse.model_validate({"rationale": {"a": 1}})
        assert r.rationale == "{'a': 1}"

    def test_bool_score_treated_as_invalid_not_as_0_or_1(self):
        """bool is a numeric subtype in Python (float(True) == 1.0) but an
        LLM returning `true` for a numeric field is a schema violation, not
        a legitimate 1.0/0.0 score."""
        r = AnalystEnhancementResponse.model_validate({"score": True})
        assert math.isnan(r.score)


class TestThesisEnhancementResponse:
    def test_valid_response(self):
        r = ThesisEnhancementResponse.model_validate({"conviction": 0.6, "rationale": "strong moat"})
        assert r.conviction == pytest.approx(0.6)
        assert r.rationale == "strong moat"

    def test_defaults_when_empty(self):
        r = ThesisEnhancementResponse.model_validate({})
        assert r.conviction == 0.0
        assert r.rationale == ""

    def test_non_numeric_conviction_becomes_nan(self):
        r = ThesisEnhancementResponse.model_validate({"conviction": "high"})
        assert math.isnan(r.conviction)


class TestDebateEnhancementResponse:
    def test_valid_response(self):
        r = DebateEnhancementResponse.model_validate({"net_conviction": -0.3, "reasoning": "bear wins"})
        assert r.net_conviction == pytest.approx(-0.3)
        assert r.reasoning == "bear wins"

    def test_non_numeric_net_conviction_becomes_nan(self):
        r = DebateEnhancementResponse.model_validate({"net_conviction": None})
        assert math.isnan(r.net_conviction)


class TestDecisionReflection:
    def test_valid_response(self):
        r = DecisionReflection.model_validate({
            "verdict": "correct",
            "what_worked": "momentum thesis held",
            "what_failed": "",
            "lesson": "trust the signal in trending regimes",
        })
        assert r.verdict == "correct"
        assert r.what_worked == "momentum thesis held"
        assert r.what_failed == ""
        assert r.lesson == "trust the signal in trending regimes"

    def test_defaults_when_empty(self):
        r = DecisionReflection.model_validate({})
        assert r.verdict == "partial"
        assert r.what_worked == ""
        assert r.what_failed == ""
        assert r.lesson == ""

    def test_invalid_verdict_falls_back_to_partial(self):
        r = DecisionReflection.model_validate({"verdict": "definitely_maybe"})
        assert r.verdict == "partial"

    def test_non_string_text_fields_are_coerced(self):
        r = DecisionReflection.model_validate({"what_worked": None, "lesson": 42})
        assert r.what_worked == ""
        assert r.lesson == "42"


class TestRiskReviewResponse:
    def test_valid_response(self):
        r = RiskReviewResponse.model_validate({
            "additional_violations": ["litigation risk"],
            "additional_actions": ["reduce position"],
            "override_approval": False,
        })
        assert r.additional_violations == ["litigation risk"]
        assert r.additional_actions == ["reduce position"]
        assert r.override_approval is False

    def test_defaults_when_empty(self):
        r = RiskReviewResponse.model_validate({})
        assert r.additional_violations == []
        assert r.additional_actions == []
        assert r.override_approval is None

    def test_non_list_violations_default_to_empty(self):
        r = RiskReviewResponse.model_validate({"additional_violations": "not a list"})
        assert r.additional_violations == []

    def test_non_string_items_filtered_out_of_lists(self):
        r = RiskReviewResponse.model_validate({
            "additional_violations": ["real risk", 42, None, {"x": 1}],
        })
        assert r.additional_violations == ["real risk"]

    def test_non_bool_override_approval_becomes_none(self):
        r = RiskReviewResponse.model_validate({"override_approval": "true"})
        assert r.override_approval is None

    def test_override_approval_true_and_false_preserved(self):
        assert RiskReviewResponse.model_validate({"override_approval": True}).override_approval is True
        assert RiskReviewResponse.model_validate({"override_approval": False}).override_approval is False


class TestPortfolioReviewResponse:
    def test_valid_response(self):
        r = PortfolioReviewResponse.model_validate({
            "adjusted_targets": {"AAPL": 0.1, "MSFT": "0.2"},
            "notes": "trim tech concentration",
        })
        assert r.adjusted_targets == {"AAPL": 0.1, "MSFT": 0.2}
        assert r.notes == "trim tech concentration"

    def test_defaults_when_empty(self):
        r = PortfolioReviewResponse.model_validate({})
        assert r.adjusted_targets == {}
        assert r.notes == ""

    def test_none_notes_becomes_empty_string(self):
        r = PortfolioReviewResponse.model_validate({"notes": None})
        assert r.notes == ""

    def test_unparseable_weight_raises_validation_error(self):
        """Deliberately not NaN-tolerant like the numeric override fields:
        a bad weight must fail the whole model so the call site falls back
        to the quant proposal in full, matching the original bare
        dict-comprehension's all-or-nothing crash+fallback behaviour."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            PortfolioReviewResponse.model_validate({"adjusted_targets": {"AAPL": "not-a-weight"}})


class TestParseLlmResponse:
    def test_returns_validated_model_on_success(self):
        result = parse_llm_response(
            AnalystEnhancementResponse, {"score": 0.3, "confidence": 0.5}, context="AAPL/momentum",
        )
        assert isinstance(result, AnalystEnhancementResponse)
        assert result.score == pytest.approx(0.3)

    def test_returns_none_and_logs_on_validation_failure(self, caplog):
        with caplog.at_level("WARNING"):
            result = parse_llm_response(
                PortfolioReviewResponse,
                {"adjusted_targets": {"AAPL": "bad"}},
                context="portfolio review",
            )
        assert result is None
        assert any("portfolio review" in r.message for r in caplog.records)

    def test_returns_none_for_non_dict_input(self):
        assert parse_llm_response(AnalystEnhancementResponse, ["not", "a", "dict"], context="x") is None
        assert parse_llm_response(AnalystEnhancementResponse, "just a string", context="x") is None
        assert parse_llm_response(AnalystEnhancementResponse, None, context="x") is None

    def test_never_raises_regardless_of_input(self):
        for bad_input in [object(), 42, [], {"score": {"nested": "dict"}}]:
            parse_llm_response(AnalystEnhancementResponse, bad_input, context="x")
