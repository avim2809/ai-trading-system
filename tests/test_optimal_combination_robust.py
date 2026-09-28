"""Tests for the robust `optimal` signal-combination estimator, the standalone
signal-book tracker, and the random-weights research placebo.

See docs/optimal_combination_fix_2026_09.md for the defects these cover.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from firm.agents.analysts import (
    _daily_compound,
    combine_signals_optimal,
    combine_signals_optimal_robust,
    combine_signals_random_weights,
    optimal_signal_weights,
    robust_optimal_signal_weights,
)
from firm.agents.research._combine import net_scores_for_blackboard
from firm.contracts.models import Signal
from firm.portfolio.attribution import PerformanceAttribution


def _sig(strategy: str, symbol: str = "AAA", score: float = 1.0, confidence: float = 1.0) -> Signal:
    return Signal(
        symbol=symbol, strategy=strategy, score=score, confidence=confidence,
        horizon="1d", asof=datetime(2024, 6, 3),
    )


def _ragged_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=300, freq="B")
    return pd.DataFrame({
        "a": pd.Series(rng.normal(0, 0.01, 300), idx),
        "b": pd.Series(rng.normal(0, 0.01, 300), idx),
        "late": pd.Series(rng.normal(0, 0.002, 100), idx[200:]),
    })


class TestLegacyDefectReproduction:
    """Pins the legacy defect so the fix's motivation stays checkable."""

    def test_legacy_zeroes_a_late_starting_strategy(self):
        frame = _ragged_frame().dropna(how="all")  # exactly what combine_signals_optimal builds
        w, _ = optimal_signal_weights(frame)
        assert w["late"] == 0.0

    def test_legacy_combiner_ignores_late_starter_entirely(self):
        frame = _ragged_frame()
        returns = {c: frame[c].dropna() for c in frame.columns}
        sigs = [_sig("a", score=0.0), _sig("b", score=0.0), _sig("late", score=1.0)]
        assert combine_signals_optimal(sigs, returns)["AAA"] == 0.0


class TestRobustWeights:
    def test_late_starter_participates(self):
        w, _ = robust_optimal_signal_weights(_ragged_frame())
        assert w["late"] > 0
        assert w.sum() == pytest.approx(1.0)
        assert (w >= 0).all()

    def test_immature_strategy_gets_equal_share(self):
        frame = _ragged_frame()
        frame.loc[frame.index[:-5], "late"] = np.nan  # only 5 obs < min_obs
        w, _ = robust_optimal_signal_weights(frame, min_obs=20)
        assert w["late"] == pytest.approx(1 / 3)
        assert w.sum() == pytest.approx(1.0)

    def test_no_history_is_equal_weight(self):
        frame = pd.DataFrame(columns=["a", "b", "c"], dtype=float)
        w, n_eff = robust_optimal_signal_weights(frame)
        assert list(w) == pytest.approx([1 / 3] * 3)
        assert n_eff == pytest.approx(3.0)

    def test_lookback_limits_window(self):
        # No late starter here: with one, the complete-case window would
        # already be capped at its 100 rows regardless of lookback.
        frame = _ragged_frame()[["a", "b"]].copy()
        # Make 'a' hugely volatile only in the distant past: with a short
        # lookback that must not affect its weight.
        frame.iloc[:150, frame.columns.get_loc("a")] *= 50
        w_short, _ = robust_optimal_signal_weights(frame, lookback_days=100)
        w_long, _ = robust_optimal_signal_weights(frame, lookback_days=300)
        assert w_short["a"] > w_long["a"]

    def test_lower_vol_gets_more_weight_when_uncorrelated(self):
        rng = np.random.default_rng(1)
        idx = pd.date_range("2024-01-01", periods=200, freq="B")
        frame = pd.DataFrame({"lo": rng.normal(0, 0.005, 200), "hi": rng.normal(0, 0.02, 200)}, idx)
        w, _ = robust_optimal_signal_weights(frame)
        assert w["lo"] > w["hi"] > 0


class TestDailyCompound:
    def test_intraday_points_compound_to_one_daily_row(self):
        idx = pd.DatetimeIndex([
            "2024-01-02 10:00", "2024-01-02 12:00", "2024-01-02 15:00", "2024-01-03 10:00",
        ])
        frame = pd.DataFrame({"a": [0.01, 0.02, -0.01, 0.005], "b": [np.nan, np.nan, np.nan, 0.01]}, idx)
        daily = _daily_compound(frame)
        assert len(daily) == 2
        assert daily["a"].iloc[0] == pytest.approx(1.01 * 1.02 * 0.99 - 1)
        assert np.isnan(daily["b"].iloc[0])  # not started: stays NaN, not 0
        assert daily["b"].iloc[1] == pytest.approx(0.01)

    def test_per_cycle_and_daily_inputs_give_same_weights(self):
        frame = _ragged_frame()
        # Split each daily return into 3 equal compounding intraday legs.
        rows = []
        for ts, row in frame.iterrows():
            leg = (1 + row) ** (1 / 3) - 1
            for h in (10, 12, 15):
                rows.append((ts + pd.Timedelta(hours=h), leg))
        intraday = pd.DataFrame([r for _, r in rows], index=[t for t, _ in rows])
        w_daily, _ = robust_optimal_signal_weights(_daily_compound(frame))
        w_intra, _ = robust_optimal_signal_weights(_daily_compound(intraday))
        pd.testing.assert_series_equal(w_daily, w_intra, atol=1e-9)


class TestRobustCombiner:
    def test_strategy_without_any_history_still_counts(self):
        frame = _ragged_frame()
        returns = {"a": frame["a"], "b": frame["b"]}
        sigs = [_sig("a", score=0.0), _sig("b", score=0.0), _sig("new", score=0.9)]
        out = combine_signals_optimal_robust(sigs, returns)
        assert out["AAA"] == pytest.approx(0.9 / 3)

    def test_empty_history_is_equal_weight_mean(self):
        sigs = [_sig("a", score=0.3), _sig("b", score=-0.9)]
        assert combine_signals_optimal_robust(sigs, {})["AAA"] == pytest.approx(-0.3)


class TestRandomWeightsPlacebo:
    def test_deterministic_and_stable_within_month(self):
        sigs = [_sig("a", score=1.0), _sig("b", score=-1.0), _sig("c", score=0.5)]
        d1 = combine_signals_random_weights(sigs, seed=7, asof=datetime(2024, 3, 4))
        d2 = combine_signals_random_weights(sigs, seed=7, asof=datetime(2024, 3, 28))
        d3 = combine_signals_random_weights(sigs, seed=7, asof=datetime(2024, 4, 2))
        assert d1 == d2
        assert d1 != d3

    def test_is_a_convex_combination(self):
        sigs = [_sig("a", score=1.0), _sig("b", score=1.0)]
        out = combine_signals_random_weights(sigs, seed=1, asof=datetime(2024, 1, 2))
        assert out["AAA"] == pytest.approx(1.0)


class TestStandaloneSignalBooks:
    def test_unit_gross_book_marked_to_market(self):
        pa = PerformanceAttribution()
        pa.update_daily(datetime(2024, 1, 2), {"X": 100.0, "Y": 50.0}, 1e6)
        pa.record_signals([
            _sig("s1", "X", score=1.0, confidence=0.5),
            _sig("s1", "Y", score=-1.0, confidence=0.5),
        ])
        pa.update_daily(datetime(2024, 1, 3), {"X": 110.0, "Y": 45.0}, 1e6)
        r = pa.get_all_signal_returns(min_points=1)["s1"]
        # 0.5 long X (+10%) and 0.5 short Y (-10%) -> +10%
        assert r.iloc[-1] == pytest.approx(0.10)

    def test_independent_of_blended_book(self):
        """A strategy never traded by the blended book still gets a stream."""
        pa = PerformanceAttribution()
        pa.update_daily(datetime(2024, 1, 2), {"X": 100.0}, 1e6)
        pa.record_signals([_sig("never_traded", "X", score=1.0)])
        pa.update_daily(datetime(2024, 1, 3), {"X": 101.0}, 1e6)
        assert "never_traded" not in pa.get_all_strategy_returns(min_points=1)
        assert pa.get_all_signal_returns(min_points=1)["never_traded"].iloc[-1] == pytest.approx(0.01)

    def test_strategy_that_stops_signalling_goes_flat(self):
        pa = PerformanceAttribution()
        pa.update_daily(datetime(2024, 1, 2), {"X": 100.0}, 1e6)
        pa.record_signals([_sig("s1", "X", score=1.0)])
        pa.update_daily(datetime(2024, 1, 3), {"X": 101.0}, 1e6)
        pa.record_signals([])
        pa.update_daily(datetime(2024, 1, 4), {"X": 150.0}, 1e6)
        assert pa.get_all_signal_returns(min_points=1)["s1"].iloc[-1] == 0.0

    def test_export_restore_roundtrip(self):
        pa = PerformanceAttribution()
        pa.update_daily(datetime(2024, 1, 2), {"X": 100.0}, 1e6)
        pa.record_signals([_sig("s1", "X", score=1.0)])
        pa.update_daily(datetime(2024, 1, 3), {"X": 102.0}, 1e6)
        pb = PerformanceAttribution()
        pb.restore_state(pa.export_state())
        pd.testing.assert_series_equal(
            pa.get_all_signal_returns(min_points=1)["s1"],
            pb.get_all_signal_returns(min_points=1)["s1"],
        )
        pb.update_daily(datetime(2024, 1, 4), {"X": 104.04}, 1e6)
        assert pb.get_all_signal_returns(min_points=1)["s1"].iloc[-1] == pytest.approx(0.02)

    def test_restore_tolerates_old_blob_without_signal_keys(self):
        pb = PerformanceAttribution()
        pb.restore_state({"strategy_returns": {}, "strategy_dates": {}})
        assert pb.get_all_signal_returns() == {}


class _BB:
    def __init__(self, sigs):
        self._sigs = sigs

    def get_all_symbols(self):
        return sorted({s.symbol for s in self._sigs})

    def get_signals_by_symbol(self, sym):
        return [s for s in self._sigs if s.symbol == sym]


class TestCombineRouting:
    def _ctx(self, returns=None, signal_returns=None):
        return SimpleNamespace(
            strategy_returns=returns, strategy_signal_returns=signal_returns,
            market_regime=None, now=datetime(2024, 6, 3),
        )

    def test_default_estimator_is_legacy_and_unchanged(self):
        frame = _ragged_frame()
        returns = {c: frame[c].dropna() for c in frame.columns}
        sigs = [_sig("a", score=0.2), _sig("b", score=-0.4), _sig("late", score=1.0)]
        got = net_scores_for_blackboard(_BB(sigs), self._ctx(returns), {"signal_combination": {"method": "optimal"}})
        assert got == combine_signals_optimal(sigs, returns)

    def test_robust_standalone_reads_signal_returns(self):
        frame = _ragged_frame()
        sig_returns = {c: frame[c].dropna() for c in frame.columns}
        sigs = [_sig("a", score=0.2), _sig("b", score=-0.4), _sig("late", score=1.0)]
        cfg = {"signal_combination": {"method": "optimal", "estimator": "robust", "returns_source": "standalone"}}
        got = net_scores_for_blackboard(_BB(sigs), self._ctx(returns=None, signal_returns=sig_returns), cfg)
        assert got == combine_signals_optimal_robust(sigs, sig_returns)

    def test_invalid_estimator_raises(self):
        cfg = {"signal_combination": {"method": "optimal", "estimator": "bogus"}}
        with pytest.raises(ValueError):
            net_scores_for_blackboard(_BB([_sig("a")]), self._ctx({}), cfg)

    def test_invalid_returns_source_raises(self):
        cfg = {"signal_combination": {"method": "optimal", "estimator": "robust", "returns_source": "x"}}
        with pytest.raises(ValueError):
            net_scores_for_blackboard(_BB([_sig("a")]), self._ctx({}), cfg)
