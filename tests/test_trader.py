"""Tests for the new signal-calibrated-probability path in
firm.agents.trader.TraderAgent._kelly (see TraderAgent._signal_calibrated_edge).

Pre-existing Kelly allocation tests (return-history edge, conviction sign,
fallback behavior) already live in tests/test_trader_kelly.py -- this file
follows that file's exact TraderAgent/AgentContext/DebateResult/mock-PitView
construction conventions, adding a minimal mock Blackboard for the new
``meta["calibrated_probability"]``/``meta["risk_reward"]`` convention this
change introduces.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from firm.agents.base import AgentContext
from firm.agents.trader import TraderAgent
from firm.contracts.models import DebateResult, Signal

NOW = datetime(2026, 1, 2)


def _make_signal(symbol: str, strategy: str, meta: dict | None = None) -> Signal:
    return Signal(
        symbol=symbol,
        strategy=strategy,
        score=0.5,
        confidence=0.8,
        horizon="5d",
        asof=NOW,
        meta=meta or {},
    )


class _FakeBlackboard:
    """Minimal stand-in exposing only what TraderAgent._kelly /
    _signal_calibrated_edge actually call: get_signals_by_symbol.
    """

    def __init__(self, signals_by_symbol: dict[str, list[Signal]] | None = None):
        self._signals_by_symbol = signals_by_symbol or {}

    def get_signals_by_symbol(self, symbol: str) -> list[Signal]:
        return self._signals_by_symbol.get(symbol, [])


def _prices_from_returns(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    return start * np.cumprod(1.0 + returns)


class _PitView:
    """Same convention as test_trader_kelly.py's _PitView: a positive-edge
    return series for WIN, negative-edge for LOSE, and (here) a symbol with
    no usable history at all (NOHIST is simply absent) to exercise the
    signal-calibrated path when the return-history fallback would return
    None.
    """

    asof = NOW

    def __init__(self):
        rng = np.random.default_rng(0)
        n = 252
        win_ret = rng.choice([0.02, -0.01], size=n, p=[0.6, 0.4])
        lose_ret = rng.choice([0.01, -0.02], size=n, p=[0.4, 0.6])
        dates = pd.date_range("2025-01-01", periods=n, freq="D")
        frames = []
        for sym, ret in (("WIN", win_ret), ("LOSE", lose_ret)):
            px = _prices_from_returns(ret)
            frames.append(pd.DataFrame({
                "date": dates, "symbol": sym, "close": px, "adj_close": px,
            }))
        self._df = pd.concat(frames, ignore_index=True)

    def prices(self, symbols=None, lookback_days=252):
        return self._df[self._df["symbol"].isin(symbols)]


class TestSignalCalibratedEdge:
    def test_no_signals_returns_none(self):
        bb = _FakeBlackboard({})
        edge = TraderAgent._signal_calibrated_edge(bb, "AAPL")
        assert edge is None

    def test_signals_without_calibrated_probability_key_return_none(self):
        bb = _FakeBlackboard({"AAPL": [_make_signal("AAPL", "momentum", meta={"foo": 1})]})
        edge = TraderAgent._signal_calibrated_edge(bb, "AAPL")
        assert edge is None

    def test_single_signal_default_risk_reward_one(self):
        # p=0.7, b=1.0 (default) -> f = (0.7*1 - 0.3)/1 = 0.4
        bb = _FakeBlackboard({
            "AAPL": [_make_signal("AAPL", "pattern_recognition", meta={"calibrated_probability": 0.7})],
        })
        edge = TraderAgent._signal_calibrated_edge(bb, "AAPL")
        assert edge == pytest.approx(0.4)

    def test_single_signal_with_risk_reward(self):
        # p=0.6, b=2.0 -> f = (0.6*2 - 0.4)/2 = 0.4
        bb = _FakeBlackboard({
            "AAPL": [_make_signal(
                "AAPL", "pattern_recognition",
                meta={"calibrated_probability": 0.6, "risk_reward": 2.0},
            )],
        })
        edge = TraderAgent._signal_calibrated_edge(bb, "AAPL")
        assert edge == pytest.approx(0.4)

    def test_multiple_signals_averaged(self):
        # sig1: p=0.7,b=1 -> f=0.4 ; sig2: p=0.6,b=2 -> f=0.4 -> average 0.4
        bb = _FakeBlackboard({
            "AAPL": [
                _make_signal("AAPL", "pattern_recognition", meta={"calibrated_probability": 0.7}),
                _make_signal("AAPL", "stat_arb", meta={"calibrated_probability": 0.6, "risk_reward": 2.0}),
            ],
        })
        edge = TraderAgent._signal_calibrated_edge(bb, "AAPL")
        assert edge == pytest.approx(0.4)

    def test_multiple_signals_averaged_differing_values(self):
        # sig1: p=0.9,b=1 -> f=0.8 ; sig2: p=0.5,b=1 -> f=0.0 -> average 0.4
        bb = _FakeBlackboard({
            "AAPL": [
                _make_signal("AAPL", "pattern_recognition", meta={"calibrated_probability": 0.9}),
                _make_signal("AAPL", "stat_arb", meta={"calibrated_probability": 0.5}),
            ],
        })
        edge = TraderAgent._signal_calibrated_edge(bb, "AAPL")
        assert edge == pytest.approx(0.4)

    def test_mixed_signals_only_tagged_ones_counted(self):
        bb = _FakeBlackboard({
            "AAPL": [
                _make_signal("AAPL", "pattern_recognition", meta={"calibrated_probability": 0.7}),
                _make_signal("AAPL", "momentum", meta={}),  # no key -- ignored
            ],
        })
        edge = TraderAgent._signal_calibrated_edge(bb, "AAPL")
        assert edge == pytest.approx(0.4)  # same as the single-signal case

    def test_out_of_range_probability_skipped(self):
        bb = _FakeBlackboard({
            "AAPL": [_make_signal("AAPL", "pattern_recognition", meta={"calibrated_probability": 1.5})],
        })
        edge = TraderAgent._signal_calibrated_edge(bb, "AAPL")
        assert edge is None

    def test_non_numeric_probability_skipped(self):
        bb = _FakeBlackboard({
            "AAPL": [_make_signal("AAPL", "pattern_recognition", meta={"calibrated_probability": "high"})],
        })
        edge = TraderAgent._signal_calibrated_edge(bb, "AAPL")
        assert edge is None

    def test_non_positive_risk_reward_falls_back_to_one(self):
        # p=0.7, b invalid (0) -> should behave as b=1.0 -> f=0.4
        bb = _FakeBlackboard({
            "AAPL": [_make_signal(
                "AAPL", "pattern_recognition",
                meta={"calibrated_probability": 0.7, "risk_reward": 0.0},
            )],
        })
        edge = TraderAgent._signal_calibrated_edge(bb, "AAPL")
        assert edge == pytest.approx(0.4)

    def test_blackboard_lookup_exception_returns_none(self):
        class _BrokenBlackboard:
            def get_signals_by_symbol(self, symbol):
                raise RuntimeError("boom")

        edge = TraderAgent._signal_calibrated_edge(_BrokenBlackboard(), "AAPL")
        assert edge is None


class TestKellyWithBlackboard:
    def test_signal_calibrated_edge_used_when_present(self):
        # NOHIST has no PitView history at all -- the return-history path
        # would return None -- but a strong calibrated signal should still
        # produce a positive weight via the new path.
        trader = TraderAgent(config={"allocation_method": "kelly", "kelly_fraction": 0.5})
        ctx = AgentContext(now=NOW, pit_view=_PitView())
        bb = _FakeBlackboard({
            "NOHIST": [_make_signal("NOHIST", "pattern_recognition", meta={"calibrated_probability": 0.8})],
        })
        results = [DebateResult(symbol="NOHIST", net_conviction=0.5)]
        targets = trader._kelly(results, ctx, bb)
        assert targets.get("NOHIST", 0.0) > 0.0

    def test_falls_back_to_kelly_edge_when_no_calibrated_signal(self):
        # Regression: identical to test_trader_kelly.py's
        # test_positive_edge_gets_weight_negative_edge_zero, but exercised
        # via the new 3-arg _kelly signature with an empty blackboard.
        trader = TraderAgent(config={"allocation_method": "kelly", "kelly_fraction": 0.5})
        ctx = AgentContext(now=NOW, pit_view=_PitView())
        bb = _FakeBlackboard({})
        results = [
            DebateResult(symbol="WIN", net_conviction=0.5),
            DebateResult(symbol="LOSE", net_conviction=0.5),
        ]
        targets = trader._kelly(results, ctx, bb)
        assert targets.get("WIN", 0.0) > 0.0
        assert targets.get("LOSE", 0.0) == 0.0

    def test_blackboard_none_matches_pre_change_behavior(self):
        # Explicit blackboard=None must be byte-for-byte identical to the
        # pre-change 2-arg call (full backward compatibility).
        trader = TraderAgent(config={"allocation_method": "kelly", "kelly_fraction": 0.5})
        ctx = AgentContext(now=NOW, pit_view=_PitView())
        results = [
            DebateResult(symbol="WIN", net_conviction=0.5),
            DebateResult(symbol="LOSE", net_conviction=0.5),
        ]
        targets_with_none = trader._kelly(results, ctx, None)
        targets_default = trader._kelly(results, ctx)
        assert targets_with_none == targets_default
        assert targets_with_none.get("WIN", 0.0) > 0.0

    def test_calibrated_probability_overrides_return_history_edge(self):
        # WIN has a positive return-history edge on its own -- but when a
        # calibrated signal is also present, it takes precedence per the
        # documented _kelly precedence order. A near-certain calibrated
        # signal (p=0.99) should produce a materially larger weight than the
        # return-history-only edge would for the same symbol.
        trader = TraderAgent(config={"allocation_method": "kelly", "kelly_fraction": 1.0})
        ctx = AgentContext(now=NOW, pit_view=_PitView())
        history_only_edge = TraderAgent._kelly_edge(ctx.pit_view, "WIN")

        bb = _FakeBlackboard({
            "WIN": [_make_signal("WIN", "pattern_recognition", meta={"calibrated_probability": 0.99})],
        })
        results = [DebateResult(symbol="WIN", net_conviction=0.5)]
        targets = trader._kelly(results, ctx, bb)
        # Single selected name -> L1-normalized weight is just its own sign,
        # so instead verify the *edge* actually used differs from the
        # history-only edge by checking the calibrated-edge helper directly
        # produces a distinctly different (much higher) value.
        calibrated_edge = TraderAgent._signal_calibrated_edge(bb, "WIN")
        assert calibrated_edge is not None
        assert calibrated_edge != pytest.approx(history_only_edge, rel=0.05)
        assert targets["WIN"] > 0.0

    def test_end_to_end_mixed_symbols_blended_and_normalized(self):
        # WIN: real positive return-history edge, no calibrated signal.
        # NOHIST: no return history, but a calibrated signal.
        # LOSE: negative return-history edge, no calibrated signal -> excluded.
        trader = TraderAgent(config={"allocation_method": "kelly", "kelly_fraction": 0.5})
        ctx = AgentContext(now=NOW, pit_view=_PitView())
        bb = _FakeBlackboard({
            "NOHIST": [_make_signal("NOHIST", "pattern_recognition", meta={"calibrated_probability": 0.75})],
        })
        results = [
            DebateResult(symbol="WIN", net_conviction=0.5),
            DebateResult(symbol="LOSE", net_conviction=0.5),
            DebateResult(symbol="NOHIST", net_conviction=-0.3),
        ]
        targets = trader.run(ctx, debate_results=results, blackboard=bb).targets

        assert targets.get("WIN", 0.0) > 0.0
        assert targets.get("LOSE", 0.0) == 0.0
        assert targets.get("NOHIST", 0.0) < 0.0  # negative conviction on a positive edge
        # L1-normalized across the surviving (positive-edge) names.
        total_abs = sum(abs(w) for w in targets.values())
        assert total_abs == pytest.approx(1.0)
