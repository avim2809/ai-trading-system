"""Tests for the Kelly allocation method in firm.agents.trader.TraderAgent."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from firm.agents.base import AgentContext
from firm.agents.trader import TraderAgent
from firm.contracts.models import DebateResult

NOW = datetime(2026, 1, 2)


def _prices_from_returns(returns: np.ndarray, start: float = 100.0) -> np.ndarray:
    return start * np.cumprod(1.0 + returns)


class _PitView:
    """Returns a positive-edge series for WIN and a negative-edge one for LOSE."""

    asof = NOW

    def __init__(self):
        rng = np.random.default_rng(0)
        n = 252
        # Positive edge: p_win high and avg win > avg loss.
        win_ret = rng.choice([0.02, -0.01], size=n, p=[0.6, 0.4])
        # Negative edge: mostly losses.
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


class TestKellyAllocation:
    def test_positive_edge_gets_weight_negative_edge_zero(self):
        trader = TraderAgent(config={"allocation_method": "kelly", "kelly_fraction": 0.5})
        ctx = AgentContext(now=NOW, pit_view=_PitView())
        results = [
            DebateResult(symbol="WIN", net_conviction=0.5),
            DebateResult(symbol="LOSE", net_conviction=0.5),
        ]
        proposal = trader.run(ctx, debate_results=results)
        assert proposal.targets.get("WIN", 0.0) > 0.0
        assert proposal.targets.get("LOSE", 0.0) == 0.0

    def test_conviction_sign_respected(self):
        trader = TraderAgent(config={"allocation_method": "kelly"})
        ctx = AgentContext(now=NOW, pit_view=_PitView())
        results = [DebateResult(symbol="WIN", net_conviction=-0.5)]
        proposal = trader.run(ctx, debate_results=results)
        # A short conviction on a positive-edge name → negative weight.
        assert proposal.targets["WIN"] < 0.0

    def test_full_kelly_edge_value(self):
        # p=0.6, b=2 → f = (0.6*2 - 0.4)/2 = 0.4
        pv = _PitView()
        edge = TraderAgent._kelly_edge(pv, "WIN")
        assert edge is not None
        assert edge > 0.0

    def test_short_edge_is_direction_conditional_not_just_negated_long_edge(self):
        """Regression (2026-09-27): shorting a persistently-uptrending
        symbol must be evaluated on its OWN (down-day) win rate/payoff, not
        inherit WIN's strong long-edge magnitude with the sign flipped by
        the caller. WIN is 60% up-days at +2% / 40% down-days at -1% -- a
        short's true win rate is the 40% down-day frequency with an
        unfavorable payoff ratio (down moves are smaller than the up moves
        it's fighting against), so the short edge should be negative, not
        the mirror of the strong positive long edge."""
        pv = _PitView()
        long_edge = TraderAgent._kelly_edge(pv, "WIN", direction="long")
        short_edge = TraderAgent._kelly_edge(pv, "WIN", direction="short")
        assert long_edge > 0.0
        assert short_edge is not None
        assert short_edge < 0.0
        assert short_edge != pytest.approx(-long_edge)

    def test_short_edge_positive_when_symbol_favors_shorts(self):
        # LOSE is 40% up / 60% down (mirror of WIN) -- a short there should
        # have a genuinely positive edge, not just "not as bad as long".
        pv = _PitView()
        short_edge = TraderAgent._kelly_edge(pv, "LOSE", direction="short")
        long_edge = TraderAgent._kelly_edge(pv, "LOSE", direction="long")
        assert short_edge is not None
        assert short_edge > 0.0
        assert long_edge < 0.0

    def test_kelly_allocation_sizes_short_by_its_own_direction_edge(self):
        # End-to-end: a short conviction on WIN (long-favorable symbol)
        # should NOT get sized as if it had WIN's strong long edge -- with
        # a genuinely negative short edge, it must fall through to the
        # conviction-weighted fallback instead of a Kelly-sized weight.
        trader = TraderAgent(config={"allocation_method": "kelly", "kelly_fraction": 0.5})
        ctx = AgentContext(now=NOW, pit_view=_PitView())
        results = [DebateResult(symbol="WIN", net_conviction=-0.5)]
        proposal = trader.run(ctx, debate_results=results)
        assert proposal.targets["WIN"] < 0.0
        # Conviction-weighted fallback for a single name allocates its full
        # (signed) conviction magnitude normalized to 1.0 -- i.e. -1.0 here
        # -- whereas a genuine (bugged) Kelly-sized weight would be scaled
        # by kelly_fraction against WIN's strong long-edge magnitude and
        # look different. Pin the fallback behavior explicitly.
        assert proposal.targets["WIN"] == pytest.approx(-1.0)

    def test_no_history_falls_back_to_conviction(self):
        trader = TraderAgent(config={"allocation_method": "kelly"})
        ctx = AgentContext(now=NOW, pit_view=None)
        results = [
            DebateResult(symbol="AAPL", net_conviction=0.6),
            DebateResult(symbol="GOOG", net_conviction=-0.4),
        ]
        proposal = trader.run(ctx, debate_results=results)
        # Falls back to conviction weighting → both names allocated.
        assert proposal.targets["AAPL"] > 0.0
        assert proposal.targets["GOOG"] < 0.0
