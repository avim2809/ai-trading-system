"""Tests for scripts/pattern_recognition_rollout_gate.py's pure decision
function, ``evaluate_rollout_gate``.

Pure-function tests only -- no live engine or network required. Imports the
script module directly by adding ``scripts/`` to ``sys.path`` (mirroring how
the scripts themselves add ``src/`` to import ``firm``), since ``scripts/``
is not a package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from pattern_recognition_rollout_gate import (  # noqa: E402
    DEFAULT_DRAWDOWN_MULTIPLE_THRESHOLD,
    DEFAULT_MIN_LIVE_DAYS,
    DEFAULT_MIN_LIVE_SIGNALS,
    DEFAULT_SHARPE_RATIO_THRESHOLD_FRACTION,
    evaluate_rollout_gate,
)


def _call(**overrides):
    """evaluate_rollout_gate with sensible passing-scenario defaults, so each
    test only needs to override what it's actually exercising."""
    kwargs = {
        "backtest_verdict": "pass",
        "backtest_oos_sharpe_mean": 0.6,
        "backtest_oos_drawdown_p95": 0.05,
        "live_sample_days": DEFAULT_MIN_LIVE_DAYS,
        "live_sample_signals": None,
        "live_sharpe_estimate": 0.6,
        "live_max_drawdown": 0.02,
    }
    kwargs.update(overrides)
    return evaluate_rollout_gate(**kwargs)


class TestInsufficientSample:
    def test_too_few_days_holds_regardless_of_great_live_performance(self):
        result = _call(
            live_sample_days=5,
            live_sample_signals=None,
            live_sharpe_estimate=5.0,  # would clearly pass a rollback check
            live_max_drawdown=0.0,
        )
        assert result["sample_ok"] is False
        assert result["recommendation"].startswith("HOLD")
        assert "insufficient live sample" in result["recommendation"]
        assert "pass" in result["recommendation"]  # backtest verdict echoed
        assert result["rollback_triggered"] is False

    def test_too_few_days_holds_regardless_of_terrible_live_performance(self):
        """The whole point of the staged gate: bad early numbers must not
        trigger a rollback before the sample-size gate is met."""
        result = _call(
            live_sample_days=3,
            live_sample_signals=None,
            live_sharpe_estimate=-5.0,
            live_max_drawdown=0.9,
        )
        assert result["sample_ok"] is False
        assert result["recommendation"].startswith("HOLD")
        assert result["rollback_triggered"] is False

    def test_signal_count_can_satisfy_the_gate_even_with_few_days(self):
        result = _call(
            live_sample_days=5,
            live_sample_signals=DEFAULT_MIN_LIVE_SIGNALS,
        )
        assert result["sample_ok"] is True
        assert not result["recommendation"].startswith("HOLD (insufficient")

    def test_days_alone_can_satisfy_the_gate_with_no_signal_count(self):
        result = _call(
            live_sample_days=DEFAULT_MIN_LIVE_DAYS,
            live_sample_signals=None,
        )
        assert result["sample_ok"] is True


class TestSufficientSampleGoodLivePassingBacktest:
    def test_keep(self):
        result = _call(
            backtest_verdict="pass",
            backtest_oos_sharpe_mean=0.6,
            backtest_oos_drawdown_p95=0.05,
            live_sample_days=25,
            live_sharpe_estimate=0.55,  # close to backtest, well above 50% floor
            live_max_drawdown=0.03,  # below 2x backtest p95
        )
        assert result["sample_ok"] is True
        assert result["rollback_triggered"] is False
        assert result["recommendation"] == "KEEP"

    def test_keep_when_live_actually_beats_backtest(self):
        result = _call(
            backtest_verdict="pass",
            backtest_oos_sharpe_mean=0.6,
            backtest_oos_drawdown_p95=0.05,
            live_sample_days=40,
            live_sharpe_estimate=0.9,
            live_max_drawdown=0.01,
        )
        assert result["recommendation"] == "KEEP"


class TestSufficientSampleBadLivePerformance:
    def test_sharpe_trip_triggers_rollback(self):
        result = _call(
            backtest_verdict="pass",
            backtest_oos_sharpe_mean=0.6,
            backtest_oos_drawdown_p95=0.05,
            live_sample_days=25,
            live_sharpe_estimate=0.6 * DEFAULT_SHARPE_RATIO_THRESHOLD_FRACTION - 0.01,
            live_max_drawdown=0.02,
        )
        assert result["sample_ok"] is True
        assert result["sharpe_trip"] is True
        assert result["drawdown_trip"] is False
        assert result["rollback_triggered"] is True
        assert result["recommendation"] == "ROLLBACK"

    def test_drawdown_trip_triggers_rollback(self):
        result = _call(
            backtest_verdict="pass",
            backtest_oos_sharpe_mean=0.6,
            backtest_oos_drawdown_p95=0.05,
            live_sample_days=25,
            live_sharpe_estimate=0.6,
            live_max_drawdown=0.05 * DEFAULT_DRAWDOWN_MULTIPLE_THRESHOLD + 0.01,
        )
        assert result["drawdown_trip"] is True
        assert result["sharpe_trip"] is False
        assert result["rollback_triggered"] is True
        assert result["recommendation"] == "ROLLBACK"

    def test_bad_live_overrides_a_passing_backtest(self):
        """Once the sample is sufficient, bad live performance is real
        signal and rolls back even though the backtest audit passed."""
        result = _call(
            backtest_verdict="pass",
            live_sample_days=30,
            live_sharpe_estimate=-1.0,
            live_max_drawdown=0.9,
        )
        assert result["recommendation"] == "ROLLBACK"

    def test_both_trips_still_reports_each_flag(self):
        result = _call(
            backtest_verdict="pass",
            backtest_oos_sharpe_mean=0.6,
            backtest_oos_drawdown_p95=0.05,
            live_sample_days=25,
            live_sharpe_estimate=-0.5,
            live_max_drawdown=0.5,
        )
        assert result["sharpe_trip"] is True
        assert result["drawdown_trip"] is True
        assert result["recommendation"] == "ROLLBACK"


class TestSufficientSampleFailingBacktestNeutralLive:
    """Documented judgment call: sufficient live sample, live performance
    itself doesn't trip a rollback threshold, but the backtest side never
    cleared PBO/DSR. This implementation holds rather than rolling back --
    fine live performance alone should not be punished, but it also can't
    KEEP a change whose backtest evidence failed."""

    def test_holds_not_keeps_and_not_rollback(self):
        result = _call(
            backtest_verdict="fail",
            backtest_oos_sharpe_mean=0.6,
            backtest_oos_drawdown_p95=0.05,
            live_sample_days=25,
            live_sharpe_estimate=0.55,
            live_max_drawdown=0.03,
        )
        assert result["sample_ok"] is True
        assert result["rollback_triggered"] is False
        assert result["recommendation"] == "HOLD"

    def test_holds_when_backtest_verdict_missing_entirely(self):
        result = _call(
            backtest_verdict=None,
            backtest_oos_sharpe_mean=None,
            backtest_oos_drawdown_p95=None,
            live_sample_days=25,
            live_sharpe_estimate=0.55,
            live_max_drawdown=0.03,
        )
        assert result["recommendation"] == "HOLD"


class TestMissingBenchmarkData:
    def test_sharpe_check_skipped_when_backtest_sharpe_missing(self):
        result = _call(
            backtest_oos_sharpe_mean=None,
            live_sample_days=25,
            live_sharpe_estimate=-10.0,  # would trip if the check ran
        )
        assert result["sharpe_trip"] is False

    def test_drawdown_check_skipped_when_backtest_drawdown_missing(self):
        result = _call(
            backtest_oos_drawdown_p95=None,
            live_sample_days=25,
            live_max_drawdown=0.99,  # would trip if the check ran
        )
        assert result["drawdown_trip"] is False

    def test_sharpe_check_skipped_when_live_estimate_missing(self):
        result = _call(live_sample_days=25, live_sharpe_estimate=None)
        assert result["sharpe_trip"] is False

    def test_non_positive_backtest_sharpe_benchmark_is_skipped(self):
        """A fraction of a non-positive Sharpe is not a meaningful floor."""
        result = _call(
            backtest_oos_sharpe_mean=-0.2,
            live_sample_days=25,
            live_sharpe_estimate=-5.0,
        )
        assert result["sharpe_trip"] is False


class TestReasoningTrail:
    def test_reasoning_is_a_nonempty_list_of_strings(self):
        result = _call(live_sample_days=25)
        assert isinstance(result["reasoning"], list)
        assert len(result["reasoning"]) > 0
        assert all(isinstance(line, str) for line in result["reasoning"])

    def test_insufficient_sample_reasoning_mentions_thresholds(self):
        result = _call(live_sample_days=1, live_sample_signals=None)
        joined = " ".join(result["reasoning"])
        assert str(DEFAULT_MIN_LIVE_DAYS) in joined
