"""Tests for firm.agents.memory.TradingMemoryLog — decision storage,
structured LLM reflection, and the lessons-learned aggregation.

Structured reflection replaces the prior free-text "2-4 sentences of prose"
format: what worked / what failed / one lesson are now separate fields
(firm.llm.schemas.DecisionReflection), so a recurring mistake is visible via
summarize_lessons() instead of buried inside per-decision prose blobs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from firm.agents.memory import TradingMemoryLog


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

    def test_retries_once_before_falling_back_to_unknown(self, tmp_path):
        """Confirmed live: garbled/degenerate LLM output fails schema
        validation often enough (~1/3 of reflected decisions) that a
        single-shot call was silently losing real self-assessment data.
        Both attempts must be exhausted before giving up."""
        log = _log(tmp_path)
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.1})
        llm = MagicMock()
        llm.chat_json.return_value = "not a dict"

        reflection = log.reflect(
            date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm,
        )

        assert llm.chat_json.call_count == 2
        assert "reflection unavailable" in reflection
        assert log.list_decisions()[0]["verdict"] == "unknown"

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
