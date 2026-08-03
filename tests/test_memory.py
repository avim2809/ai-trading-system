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
