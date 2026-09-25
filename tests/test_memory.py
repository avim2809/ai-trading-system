"""Tests for firm.agents.memory.TradingMemoryLog — decision storage,
structured LLM reflection, and the lessons-learned aggregation.

Structured reflection replaces the prior free-text "2-4 sentences of prose"
format: what worked / what failed / one lesson are now separate fields
(firm.llm.schemas.DecisionReflection), so a recurring mistake is visible via
summarize_lessons() instead of buried inside per-decision prose blobs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from firm.agents.memory import TradingMemoryLog
from firm.rag.models import RetrievedDoc


def _log(tmp_path) -> TradingMemoryLog:
    return TradingMemoryLog(config={"memory_log_path": str(tmp_path / "decisions.jsonl")})


class TestStoreDecision:
    def test_stores_pending_entry(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, notes="n")
        entries = log.list_decisions()
        assert len(entries) == 1
        assert entries[0]["status"] == "pending"
        assert entries[0]["verdict"] is None

    def test_idempotent_for_same_date(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        log.store_decision(date="2026-01-01", proposal_weights={"MSFT": 0.2})
        assert len(log.list_decisions()) == 1

    def test_stores_per_strategy_attribution(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(
            date="2026-01-01",
            proposal_weights={"AAPL": 0.1, "MSFT": -0.05},
            per_strategy={"momentum": {"AAPL": 0.1}, "trend": {"MSFT": -0.05}},
        )
        entry = log.list_decisions()[0]
        assert entry["per_strategy"] == {"momentum": {"AAPL": 0.1}, "trend": {"MSFT": -0.05}}

    def test_per_strategy_defaults_to_empty_dict_when_not_given(self, tmp_path):
        """Backward compatible: callers that don't pass per_strategy (or an
        older entry re-read from disk) must not crash reflect()'s lookup."""
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        entry = log.list_decisions()[0]
        assert entry["per_strategy"] == {}


class TestReflect:
    def test_structured_response_populates_fields_and_renders_prose(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})

        llm = MagicMock()
        llm.chat_json.return_value = {
            "verdict": "correct",
            "what_worked": "momentum thesis held",
            "what_failed": "",
            "lesson": "trust the signal",
        }
        reflection = log.reflect(
            date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm,
        )

        assert reflection == "CORRECT. What worked: momentum thesis held Lesson: trust the signal"
        entry = log.list_decisions()[0]
        assert entry["status"] == "reflected"
        assert entry["verdict"] == "correct"
        assert entry["what_worked"] == "momentum thesis held"
        assert entry["what_failed"] == ""
        assert entry["lesson"] == "trust the signal"
        assert entry["reflection"] == reflection

    def test_incorrect_verdict_includes_what_failed(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        llm = MagicMock()
        llm.chat_json.return_value = {
            "verdict": "incorrect",
            "what_worked": "",
            "what_failed": "regime shifted against the thesis",
            "lesson": "add a regime filter",
        }
        reflection = log.reflect(
            date="2026-01-01", raw_return=-0.03, benchmark_return=0.01, llm_service=llm,
        )
        assert reflection == (
            "INCORRECT. What failed: regime shifted against the thesis "
            "Lesson: add a regime filter"
        )

    def test_llm_failure_falls_back_to_unknown_verdict(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        llm = MagicMock()
        llm.chat_json.side_effect = RuntimeError("no provider")

        reflection = log.reflect(
            date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm,
        )

        assert "reflection unavailable" in reflection
        entry = log.list_decisions()[0]
        assert entry["verdict"] == "unknown"
        assert entry["lesson"] == ""

    def test_malformed_llm_json_falls_back_to_unknown_verdict(self, tmp_path):
        """chat_json returning something that fails schema validation (e.g.
        not a dict) must degrade the same as a hard failure, not raise."""
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        llm = MagicMock()
        llm.chat_json.return_value = "not a dict"

        reflection = log.reflect(
            date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm,
        )

        assert "reflection unavailable" in reflection
        assert log.list_decisions()[0]["verdict"] == "unknown"

    def test_retries_before_falling_back_to_unknown(self, tmp_path):
        """Garbled/degenerate LLM output fails schema validation often
        enough (historically ~1/3 of reflected decisions) that a
        single-shot call was silently losing real self-assessment data.
        All attempts must be exhausted before giving up."""
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        llm = MagicMock()
        llm.chat_json.return_value = "not a dict"

        reflection = log.reflect(
            date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm,
        )

        assert llm.chat_json.call_count == 3
        assert "reflection unavailable" in reflection
        assert log.list_decisions()[0]["verdict"] == "unknown"

    def test_retry_forces_default_model_instead_of_reusing_the_same_pick(self, tmp_path):
        """Regression: LLMService's load-balance routing is a deterministic
        hash of the message content, so a same-message retry with no model
        override reliably re-picks the exact same (possibly unreliable
        free-tier) model that just failed -- defeating the retry's whole
        purpose. Reflection is low-volume enough that it doesn't need
        load-balancing; retries must force the service's own default_model
        instead of re-rolling the same dice."""
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        llm = MagicMock()
        llm.default_model = "groq/openai/gpt-oss-120b"
        llm.chat_json.return_value = "not a dict"

        log.reflect(
            date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm,
        )

        calls = llm.chat_json.call_args_list
        assert calls[0].kwargs.get("model") is None  # first attempt: normal load-balanced pick
        assert calls[1].kwargs.get("model") == "groq/openai/gpt-oss-120b"
        assert calls[2].kwargs.get("model") == "groq/openai/gpt-oss-120b"

    def test_recovers_on_second_attempt_after_a_bad_first_sample(self, tmp_path):
        """The whole point of the retry: a bad first sample must not
        permanently discard a decision's self-assessment when a second
        attempt would have produced a valid one."""
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        llm = MagicMock()
        llm.chat_json.side_effect = [
            "not a dict",  # first sample: garbled, fails schema validation
            {
                "verdict": "correct", "what_worked": "momentum's long AAPL call",
                "what_failed": "", "lesson": "size up on high-confidence momentum calls",
            },
        ]

        reflection = log.reflect(
            date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm,
        )

        assert llm.chat_json.call_count == 2
        assert reflection.startswith("CORRECT")
        entry = log.list_decisions()[0]
        assert entry["verdict"] == "correct"
        assert entry["lesson"] == "size up on high-confidence momentum calls"

    def test_per_strategy_attribution_included_in_reflection_prompt(self, tmp_path):
        """The reflection-generating LLM call must actually SEE the
        per-strategy breakdown, not just have it stored for display —
        otherwise the LLM can only judge "the portfolio" as a whole and
        can never name which strategy's call was right or wrong."""
        log = _log(tmp_path)
        log.store_decision(
            date="2026-01-01",
            proposal_weights={"AAPL": 0.1, "MSFT": -0.05},
            per_strategy={"momentum": {"AAPL": 0.1}, "danelfin_ai_score": {"MSFT": -0.05}},
        )
        llm = MagicMock()
        llm.chat_json.return_value = {
            "verdict": "correct", "what_worked": "momentum's AAPL call",
            "what_failed": "", "lesson": "trust momentum in trends",
        }
        log.reflect(date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm)

        user_prompt = llm.chat_json.call_args[0][0][1]["content"]
        assert "momentum" in user_prompt
        assert "danelfin_ai_score" in user_prompt
        assert '"AAPL": 0.1' in user_prompt

    def test_no_per_strategy_omits_the_block_cleanly(self, tmp_path):
        """A decision stored before this field existed (or with none given)
        must not render a dangling/empty "Per-strategy attribution:" label
        with nothing after it."""
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        llm = MagicMock()
        llm.chat_json.return_value = {
            "verdict": "correct", "what_worked": "x", "what_failed": "", "lesson": "y",
        }
        log.reflect(date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm)

        user_prompt = llm.chat_json.call_args[0][0][1]["content"]
        assert "Per-strategy attribution" not in user_prompt

    def test_no_pending_entry_returns_none(self, tmp_path):
        log = _log(tmp_path)
        llm = MagicMock()
        result = log.reflect(
            date="2026-01-01", raw_return=0.0, benchmark_return=0.0, llm_service=llm,
        )
        assert result is None
        llm.chat_json.assert_not_called()


class TestGetContext:
    def test_renders_structured_reflection_in_markdown(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        llm = MagicMock()
        llm.chat_json.return_value = {
            "verdict": "partial",
            "what_worked": "sizing was right",
            "what_failed": "entry timing was early",
            "lesson": "wait for confirmation",
        }
        log.reflect(date="2026-01-01", raw_return=0.0, benchmark_return=0.0, llm_service=llm)

        context = log.get_context()
        assert "PARTIAL" in context
        assert "wait for confirmation" in context


class TestSummarizeLessons:
    def _reflect(self, log, date, llm_result, raw_return=0.0, benchmark_return=0.0):
        log.store_decision(date=date, proposal_weights={"AAPL": 0.1})
        llm = MagicMock()
        llm.chat_json.return_value = llm_result
        log.reflect(date=date, raw_return=raw_return, benchmark_return=benchmark_return, llm_service=llm)

    def test_no_reflections_yet(self, tmp_path):
        log = _log(tmp_path)
        summary = log.summarize_lessons()
        assert summary == {
            "total": 0,
            "counts": {"correct": 0, "incorrect": 0, "partial": 0, "unknown": 0},
            "recent_lessons": [],
        }

    def test_aggregates_verdict_counts_and_lessons(self, tmp_path):
        log = _log(tmp_path)
        self._reflect(log, "2026-01-01", {
            "verdict": "correct", "what_worked": "x", "what_failed": "", "lesson": "lesson one",
        })
        self._reflect(log, "2026-01-02", {
            "verdict": "incorrect", "what_worked": "", "what_failed": "y", "lesson": "lesson two",
        })
        self._reflect(log, "2026-01-03", {
            "verdict": "correct", "what_worked": "z", "what_failed": "", "lesson": "",
        })

        summary = log.summarize_lessons()
        assert summary["total"] == 3
        assert summary["counts"] == {"correct": 2, "incorrect": 1, "partial": 0, "unknown": 0}
        # Empty lessons are skipped; non-empty ones are most-recent-first.
        assert summary["recent_lessons"] == ["lesson two", "lesson one"]

    def test_respects_n_limit(self, tmp_path):
        log = _log(tmp_path)
        for i in range(5):
            self._reflect(log, f"2026-01-0{i+1}", {
                "verdict": "correct", "what_worked": "", "what_failed": "", "lesson": f"lesson {i}",
            })
        summary = log.summarize_lessons(n=2)
        assert summary["recent_lessons"] == ["lesson 4", "lesson 3"]

    def test_unknown_verdict_from_failed_reflection_is_counted(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        llm = MagicMock()
        llm.chat_json.side_effect = RuntimeError("down")
        log.reflect(date="2026-01-01", raw_return=0.0, benchmark_return=0.0, llm_service=llm)

        summary = log.summarize_lessons()
        assert summary["counts"]["unknown"] == 1
        assert summary["recent_lessons"] == []


class TestCycleIdStorage:
    """store_decision's cycle_id key -- added 2026-09-20 to fix a real
    incident: under the ~7-cycles/trading-day "hourly_market_hours"
    schedule, the old bare-date idempotency key meant only the day's very
    first cycle's decision was ever stored; the other ~6 silently
    vanished."""

    def test_multiple_cycles_same_day_all_stored(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.2}, cycle_id=2)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.3}, cycle_id=3)

        entries = log.list_decisions()
        assert len(entries) == 3
        assert {e["cycle_id"] for e in entries} == {1, 2, 3}

    def test_same_cycle_id_is_idempotent(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.99}, cycle_id=1)

        entries = log.list_decisions()
        assert len(entries) == 1
        assert entries[0]["proposal_weights"] == {"AAPL": 0.1}

    def test_legacy_no_cycle_id_still_one_entry_per_date(self, tmp_path):
        """cycle_id=None (the old call shape) must reproduce exactly the
        original bare-date idempotency -- no behavior change for a caller
        that doesn't opt in."""
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        log.store_decision(date="2026-01-01", proposal_weights={"MSFT": 0.2})
        assert len(log.list_decisions()) == 1


class TestReflectDay:
    def _llm(self, **overrides):
        llm = MagicMock()
        payload = {
            "verdict": "correct", "what_worked": "w", "what_failed": "", "lesson": "l",
            "recommendation": {"action": "no_action"},
        }
        payload.update(overrides)
        llm.chat_json.return_value = payload
        return llm

    def test_aggregates_all_of_a_day_s_cycles_into_one_reflection(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(
            date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1,
            per_strategy={"momentum": {"AAPL": 0.1}}, nav_at_decision=100_000,
        )
        log.store_decision(
            date="2026-01-01", proposal_weights={"AAPL": 0.15, "MSFT": 0.05}, cycle_id=2,
            per_strategy={"momentum": {"AAPL": 0.15}, "trend": {"MSFT": 0.05}},
        )

        llm = self._llm()
        reflection = log.reflect_day(
            date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm,
        )

        assert reflection is not None
        assert llm.chat_json.call_count == 1  # one LLM call for the whole day, not per cycle
        prompt = llm.chat_json.call_args[0][0][1]["content"]
        assert "2 decision cycle(s)" in prompt

        entries = log.list_decisions()
        rollup = next(e for e in entries if e.get("cycle_id") == "rollup")
        assert rollup["status"] == "reflected"
        assert rollup["n_cycles"] == 2
        # Later cycle's AAPL target supersedes the earlier one.
        assert rollup["per_strategy"]["momentum"]["AAPL"] == 0.15
        assert rollup["per_strategy"]["trend"]["MSFT"] == 0.05

    def test_per_cycle_entries_excluded_from_pending_after_rollup(self, tmp_path):
        """The real incident's fix: after a daily rollup, find_all_pending()
        must stop re-surfacing that day's per-cycle entries forever."""
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.2}, cycle_id=2)
        assert len(log.find_all_pending()) == 2

        log.reflect_day(date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=self._llm())

        assert log.find_all_pending() == []

    def test_calling_twice_for_same_date_is_a_no_op_second_time(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)
        llm = self._llm()
        log.reflect_day(date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm)

        result = log.reflect_day(date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm)

        assert result is None
        assert llm.chat_json.call_count == 1  # not called again

    def test_no_pending_entries_returns_none(self, tmp_path):
        log = _log(tmp_path)
        result = log.reflect_day(
            date="2026-01-01", raw_return=0.0, benchmark_return=0.0, llm_service=self._llm(),
        )
        assert result is None

    def test_recommendation_is_stored_and_listed(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)
        llm = self._llm(recommendation={
            "action": "reduce_position_limit", "strategy": "stat_arb",
            "reduce_by_pct": 0.3, "rationale": "repeated veto pattern",
        })

        log.reflect_day(date="2026-01-01", raw_return=-0.05, benchmark_return=0.0, llm_service=llm)

        recs = log.list_recommendations()
        assert len(recs) == 1
        assert recs[0]["action"] == "reduce_position_limit"
        assert recs[0]["strategy"] == "stat_arb"
        assert recs[0]["reduce_by_pct"] == 0.3
        assert recs[0]["date"] == "2026-01-01"

    def test_no_action_recommendation_not_listed_by_default(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)
        log.reflect_day(date="2026-01-01", raw_return=0.01, benchmark_return=0.01, llm_service=self._llm())

        assert log.list_recommendations() == []
        assert log.list_recommendations(pending_only=False) != []

    def test_mark_recommendation_applied_excludes_from_pending(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)
        log.reflect_day(date="2026-01-01", raw_return=-0.05, benchmark_return=0.0, llm_service=self._llm(
            recommendation={
                "action": "reduce_position_limit", "strategy": "stat_arb",
                "reduce_by_pct": 0.3, "rationale": "x",
            },
        ))
        assert len(log.list_recommendations()) == 1

        applied = log.mark_recommendation_applied("2026-01-01")

        assert applied is True
        assert log.list_recommendations() == []
        assert log.list_recommendations(pending_only=False)[0]["applied"] is True

    def test_mark_recommendation_applied_returns_false_when_nothing_to_apply(self, tmp_path):
        log = _log(tmp_path)
        assert log.mark_recommendation_applied("2026-01-01") is False


class TestStrategyPerformanceBlock:
    """reflect_day()'s strategy_performance param (2026-09-25) -- real
    realized returns fed into the prompt alongside (not instead of) the
    target-weight breakdown, plus the block's own rendering."""

    def _llm(self, **overrides):
        llm = MagicMock()
        payload = {
            "verdict": "correct", "what_worked": "w", "what_failed": "", "lesson": "l",
            "recommendation": {"action": "no_action"},
        }
        payload.update(overrides)
        llm.chat_json.return_value = payload
        return llm

    def test_realized_returns_appear_in_the_prompt_sorted_worst_first(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)
        llm = self._llm()

        log.reflect_day(
            date="2026-01-01", raw_return=-0.05, benchmark_return=0.0, llm_service=llm,
            strategy_performance={
                "momentum": {"return": 0.01},
                "stat_arb": {
                    "return": -0.03, "trailing_mean": -0.001,
                    "trailing_std": 0.01, "trailing_n": 10.0,
                },
            },
        )

        prompt = llm.chat_json.call_args[0][0][1]["content"]
        assert "ACTUAL REALIZED" in prompt
        assert "stat_arb: -3.000%" in prompt
        assert "z=" in prompt
        # Worst return (stat_arb) listed before momentum.
        assert prompt.index("stat_arb") < prompt.index("momentum")

    def test_omitted_entirely_when_none_supplied(self, tmp_path):
        """Backward compat: a caller that doesn't pass strategy_performance
        gets the exact old prompt, no new section at all."""
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)
        llm = self._llm()

        log.reflect_day(date="2026-01-01", raw_return=0.01, benchmark_return=0.0, llm_service=llm)

        prompt = llm.chat_json.call_args[0][0][1]["content"]
        assert "ground truth for P&L" not in prompt


class TestRecommendationSanityGate:
    """_sanity_check_recommendation (2026-09-25) -- a mechanical, code-
    enforced second layer behind the prompt's own "one bad day is not
    sufficient evidence" instruction, added after an independent audit
    found 3 of 4 ever-issued recommendations were factually wrong and not
    caught by that instruction alone."""

    def _llm(self, **overrides):
        llm = MagicMock()
        payload = {
            "verdict": "correct", "what_worked": "w", "what_failed": "", "lesson": "l",
            "recommendation": {"action": "no_action"},
        }
        payload.update(overrides)
        llm.chat_json.return_value = payload
        return llm

    def _reflect(self, tmp_path, *, recommendation, strategy_performance):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)
        llm = self._llm(recommendation=recommendation)
        log.reflect_day(
            date="2026-01-01", raw_return=-0.05, benchmark_return=0.0, llm_service=llm,
            strategy_performance=strategy_performance,
        )
        return log

    def test_downgrades_when_named_strategy_has_no_performance_data(self, tmp_path):
        """Real incident: a strategy flagged with no way to verify it at
        all must not reach list_recommendations() as actionable."""
        log = self._reflect(
            tmp_path,
            recommendation={
                "action": "flag_strategy_for_review", "strategy": "regime_hmm",
                "rationale": "large loss",
            },
            strategy_performance={"momentum": {"return": -0.02}},  # regime_hmm absent
        )
        assert log.list_recommendations() == []
        all_recs = log.list_recommendations(pending_only=False)
        assert all_recs[0]["action"] == "no_action"
        assert "auto-downgraded" in all_recs[0]["rationale"]

    def test_downgrades_when_strategy_actually_had_a_positive_day(self, tmp_path):
        """Real incident this closes exactly: regime_hmm flagged for
        "dominating portfolio drag" on a day it actually returned +0.38%."""
        log = self._reflect(
            tmp_path,
            recommendation={
                "action": "flag_strategy_for_review", "strategy": "regime_hmm",
                "rationale": "NVDA short generated large loss",
            },
            strategy_performance={"regime_hmm": {"return": 0.0038}},
        )
        assert log.list_recommendations() == []
        rec = log.list_recommendations(pending_only=False)[0]
        assert rec["action"] == "no_action"
        assert "not a loss" in rec["rationale"]

    def test_downgrades_negative_but_ordinary_day_for_that_strategy(self, tmp_path):
        """Real incident this closes: volatility_breakout flagged for a
        genuine but tiny (-0.016%) loss well within its own normal range
        (lifetime Sharpe +0.36) -- sign alone isn't enough, z-score must
        also indicate a genuine anomaly."""
        log = self._reflect(
            tmp_path,
            recommendation={
                "action": "flag_strategy_for_review", "strategy": "volatility_breakout",
                "rationale": "negative net contribution dragged portfolio down",
            },
            strategy_performance={
                "volatility_breakout": {
                    "return": -0.00016, "trailing_mean": 0.0001,
                    "trailing_std": 0.002, "trailing_n": 10.0,
                },
            },
        )
        assert log.list_recommendations() == []
        rec = log.list_recommendations(pending_only=False)[0]
        assert rec["action"] == "no_action"
        assert "within its own normal range" in rec["rationale"]

    def test_allows_through_a_genuine_negative_outlier(self, tmp_path):
        """The one case that SHOULD survive: a real, negative, genuinely
        unusual (z well below -1) return for the named strategy."""
        log = self._reflect(
            tmp_path,
            recommendation={
                "action": "reduce_position_limit", "strategy": "multi_factor",
                "reduce_by_pct": 0.2, "rationale": "recurring negative alpha in chop",
            },
            strategy_performance={
                "multi_factor": {
                    "return": -0.028, "trailing_mean": -0.001,
                    "trailing_std": 0.01, "trailing_n": 15.0,
                },
            },
        )
        recs = log.list_recommendations()
        assert len(recs) == 1
        assert recs[0]["action"] == "reduce_position_limit"
        assert recs[0]["strategy"] == "multi_factor"

    def test_never_gates_when_strategy_performance_not_supplied_at_all(self, tmp_path):
        """None (not supplied) means the caller hasn't opted into this
        check -- must reproduce the exact old ungated behavior."""
        log = self._reflect(
            tmp_path,
            recommendation={
                "action": "reduce_position_limit", "strategy": "stat_arb",
                "reduce_by_pct": 0.3, "rationale": "x",
            },
            strategy_performance=None,
        )
        recs = log.list_recommendations()
        assert len(recs) == 1
        assert recs[0]["action"] == "reduce_position_limit"

    def test_no_action_recommendations_are_never_touched(self, tmp_path):
        log = self._reflect(
            tmp_path,
            recommendation={"action": "no_action", "rationale": "fine day"},
            strategy_performance={},
        )
        rec = log.list_recommendations(pending_only=False)[0]
        assert rec["rationale"] == "fine day"  # unmodified, no auto-downgrade note


class TestRecommendationRAGLoop:
    """The self-improvement loop: reflect_day() writes its recommendation
    into the RAG "recommendations" collection (even a no_action one) and
    retrieves relevant past recommendations before building its prompt;
    mark_recommendation_applied() mirrors the applied flag into that same
    stored doc. RAG is mocked directly onto the instance (_rag_store /
    _rag_retriever) rather than exercising real Chroma/embeddings — matches
    how reflect()/reflect_day() already mock llm_service in this file."""

    def _llm(self, **overrides):
        llm = MagicMock()
        payload = {
            "verdict": "correct", "what_worked": "w", "what_failed": "", "lesson": "l",
            "recommendation": {"action": "no_action"},
        }
        payload.update(overrides)
        llm.chat_json.return_value = payload
        return llm

    def _wire_rag(self, log, retrieved=None):
        mock_store = MagicMock()
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = retrieved or []
        log._rag_store = mock_store
        log._rag_retriever = mock_retriever
        return mock_store, mock_retriever

    def test_recommendation_written_to_rag_with_correct_metadata(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(
            date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1,
            per_strategy={"momentum": {"AAPL": 0.1}}, nav_at_decision=100_000,
        )
        mock_store, _ = self._wire_rag(log)
        llm = self._llm(recommendation={
            "action": "reduce_position_limit", "strategy": "momentum",
            "reduce_by_pct": 0.3, "rationale": "repeated veto pattern",
        })

        log.reflect_day(date="2026-01-01", raw_return=-0.05, benchmark_return=0.0, llm_service=llm)

        mock_store.add_documents.assert_called_once()
        collection_name, docs = mock_store.add_documents.call_args[0]
        assert collection_name == "recommendations"
        assert len(docs) == 1
        doc = docs[0]
        assert doc.doc_id == "recommendation:2026-01-01"
        assert doc.metadata == {
            "date": "2026-01-01",
            "doc_type": "recommendation",
            "action": "reduce_position_limit",
            "strategy": "momentum",
            "reduce_by_pct": 0.3,
            "rationale": "repeated veto pattern",
            "applied": False,
        }
        assert "momentum" in doc.text
        assert "repeated veto pattern" in doc.text

    def test_no_action_recommendation_is_still_written(self, tmp_path):
        """The *absence* of an actionable recommendation for a situation is
        itself useful history -- it must not be skipped just because
        action == "no_action"."""
        log = _log(tmp_path)
        log.store_decision(
            date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000,
        )
        mock_store, _ = self._wire_rag(log)

        log.reflect_day(date="2026-01-01", raw_return=0.01, benchmark_return=0.01, llm_service=self._llm())

        mock_store.add_documents.assert_called_once()
        _, docs = mock_store.add_documents.call_args[0]
        assert docs[0].metadata["action"] == "no_action"
        assert docs[0].metadata["applied"] is False

    def test_no_llm_recommendation_at_all_is_not_written(self, tmp_path):
        """A total LLM failure (parsed is None) yields recommendation=None,
        distinct from a real "no_action" verdict -- there's no rationale to
        store, so this must not write a hollow RAG doc."""
        log = _log(tmp_path)
        log.store_decision(
            date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000,
        )
        mock_store, _ = self._wire_rag(log)
        llm = MagicMock()
        llm.chat_json.side_effect = RuntimeError("down")

        log.reflect_day(date="2026-01-01", raw_return=0.01, benchmark_return=0.01, llm_service=llm)

        mock_store.add_documents.assert_not_called()

    def test_past_recommendations_are_retrieved_and_injected_into_prompt(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(
            date="2026-01-02", proposal_weights={"AAPL": 0.1}, cycle_id=1,
            per_strategy={"momentum": {"AAPL": 0.1}}, nav_at_decision=100_000,
        )
        _, mock_retriever = self._wire_rag(log, retrieved=[
            RetrievedDoc(
                doc_id="recommendation:2026-01-01",
                text="...",
                metadata={
                    "date": "2026-01-01",
                    "action": "flag_strategy_for_review",
                    "strategy": "momentum",
                    "applied": True,
                    "rationale": "repeated losing days",
                },
                score=0.9,
            ),
        ])
        llm = self._llm()

        log.reflect_day(date="2026-01-02", raw_return=0.02, benchmark_return=0.0, llm_service=llm)

        _, kwargs = mock_retriever.retrieve.call_args
        assert kwargs["collection"] == "recommendations"
        assert kwargs["n_results"] == 3  # bounded, matches memory_rag_n_results default

        prompt = llm.chat_json.call_args[0][0][1]["content"]
        assert "Past recommendations for similar situations" in prompt
        assert "flag_strategy_for_review" in prompt
        assert "momentum" in prompt
        assert "applied: yes" in prompt
        assert "repeated losing days" in prompt

    def test_no_past_recommendations_omits_the_block_cleanly(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)
        llm = self._llm()
        self._wire_rag(log, retrieved=[])

        log.reflect_day(date="2026-01-01", raw_return=0.01, benchmark_return=0.01, llm_service=llm)

        prompt = llm.chat_json.call_args[0][0][1]["content"]
        assert "Past recommendations" not in prompt

    def test_mark_recommendation_applied_updates_rag_doc_in_place(self, tmp_path):
        """The update-vs-dedup subtlety: applying a recommendation must
        patch the existing RAG doc's metadata via update_metadata (in
        place), not re-add it through add_documents -- add_documents skips
        any id already present in the collection before ever calling
        upsert(), so a same-id/changed-metadata re-add would silently do
        nothing and leave the stale applied=False behind."""
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)
        mock_store, _ = self._wire_rag(log)
        mock_store.update_metadata.return_value = True
        llm = self._llm(recommendation={
            "action": "reduce_position_limit", "strategy": "momentum",
            "reduce_by_pct": 0.3, "rationale": "x",
        })
        log.reflect_day(date="2026-01-01", raw_return=-0.05, benchmark_return=0.0, llm_service=llm)
        assert mock_store.add_documents.call_count == 1  # the original write, from reflect_day

        applied = log.mark_recommendation_applied("2026-01-01")

        assert applied is True
        mock_store.update_metadata.assert_called_once_with(
            "recommendations", "recommendation:2026-01-01", {"applied": True},
        )
        # Applying never re-invokes add_documents -- that path is a dedup
        # no-op for an unchanged doc_id and would leave applied=False stale.
        assert mock_store.add_documents.call_count == 1

    def test_mark_recommendation_applied_still_true_when_rag_doc_missing(self, tmp_path):
        """The JSONL log (the source of truth) must still record "applied"
        even if the RAG mirror has nothing to update for this date."""
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)
        mock_store, _ = self._wire_rag(log)
        mock_store.update_metadata.return_value = False
        log.reflect_day(date="2026-01-01", raw_return=-0.05, benchmark_return=0.0, llm_service=self._llm(
            recommendation={"action": "flag_strategy_for_review", "strategy": "momentum", "rationale": "x"},
        ))

        applied = log.mark_recommendation_applied("2026-01-01")

        assert applied is True
        assert log.list_recommendations(pending_only=False)[0]["applied"] is True


class TestRecommendationRAGFailsSoft:
    """reflect_day()/mark_recommendation_applied() must work exactly as
    they do today when RAG is unavailable (extras not installed, vector
    store down, etc.) -- the recommendation loop only enriches the prompt,
    it must never be a new way for reflection to break."""

    def _llm(self, **overrides):
        llm = MagicMock()
        payload = {
            "verdict": "correct", "what_worked": "w", "what_failed": "", "lesson": "l",
            "recommendation": {"action": "no_action"},
        }
        payload.update(overrides)
        llm.chat_json.return_value = payload
        return llm

    def test_reflect_day_completes_when_rag_store_construction_fails(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)

        with patch.object(TradingMemoryLog, "_get_rag_store", side_effect=ImportError("rag extra not installed")):
            reflection = log.reflect_day(
                date="2026-01-01", raw_return=0.01, benchmark_return=0.0, llm_service=self._llm(),
            )

        assert reflection is not None
        assert "CORRECT" in reflection
        rollup = next(e for e in log.list_decisions() if e.get("cycle_id") == "rollup")
        assert rollup["status"] == "reflected"

    def test_reflect_day_completes_when_retriever_query_raises(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)
        mock_retriever = MagicMock()
        mock_retriever.retrieve.side_effect = RuntimeError("chroma query failed")
        log._rag_retriever = mock_retriever
        log._rag_store = MagicMock()
        log._rag_store.add_documents.side_effect = RuntimeError("chroma write failed")

        reflection = log.reflect_day(
            date="2026-01-01", raw_return=0.01, benchmark_return=0.0, llm_service=self._llm(),
        )

        assert reflection is not None
        assert "CORRECT" in reflection

    def test_mark_recommendation_applied_still_succeeds_when_rag_unavailable(self, tmp_path):
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1}, cycle_id=1, nav_at_decision=100_000)

        with patch.object(TradingMemoryLog, "_get_rag_store", side_effect=RuntimeError("chroma down")):
            log.reflect_day(date="2026-01-01", raw_return=-0.05, benchmark_return=0.0, llm_service=self._llm(
                recommendation={
                    "action": "reduce_position_limit", "strategy": "momentum",
                    "reduce_by_pct": 0.2, "rationale": "x",
                },
            ))
            applied = log.mark_recommendation_applied("2026-01-01")

        assert applied is True  # JSONL log updated correctly regardless of RAG mirror failure
        assert log.list_recommendations(pending_only=False)[0]["applied"] is True
