"""Tests for per-strategy regime-conditional score multipliers."""

from __future__ import annotations

from datetime import datetime

import pytest

from firm.agents.base import AgentContext
from firm.agents.blackboard import Blackboard
from firm.agents.research._combine import net_scores_for_blackboard
from firm.agents.research._regime_weights import apply_strategy_regime_weights
from firm.contracts.models import Signal, SignalSet
from firm.regime.model import RegimeState

NOW = datetime(2026, 1, 2)


def _signal(strategy: str, score: float = 1.0) -> Signal:
    return Signal("AAPL", strategy, score, 0.8, "5d", NOW)


def _regime(label: str, confidence: float = 0.8) -> RegimeState:
    return RegimeState(
        label=label,
        confidence=confidence,
        state_idx=0,
        posterior=[confidence],
        separation=1.0,
    )


class TestApplyStrategyRegimeWeights:
    def test_disabled_is_noop(self):
        sigs = [_signal("momentum")]
        out = apply_strategy_regime_weights(sigs, _regime("Bull"), {"enabled": False})
        assert out[0].score == 1.0

    def test_no_regime_is_noop(self):
        cfg = {"enabled": True, "weights": {"Bull": {"momentum": 1.5}}}
        out = apply_strategy_regime_weights([_signal("momentum")], None, cfg)
        assert out[0].score == 1.0

    def test_boosts_configured_strategy_in_regime(self):
        cfg = {
            "enabled": True,
            "weights": {"Bull": {"momentum": 1.5}},
        }
        out = apply_strategy_regime_weights(
            [_signal("momentum")], _regime("Bull", confidence=1.0), cfg,
        )
        assert out[0].score == pytest.approx(1.5)

    def test_unlisted_strategy_unchanged(self):
        cfg = {
            "enabled": True,
            "weights": {"Bull": {"momentum": 1.5}},
        }
        out = apply_strategy_regime_weights(
            [_signal("trend")], _regime("Bull", confidence=1.0), cfg,
        )
        assert out[0].score == 1.0

    def test_confidence_blends_toward_neutral(self):
        cfg = {
            "enabled": True,
            "weights": {"Bear": {"stat_arb": 0.5}},
        }
        out = apply_strategy_regime_weights(
            [_signal("stat_arb")], _regime("Bear", confidence=0.5), cfg,
        )
        # effective = 1 + (0.5 - 1) * 0.5 = 0.75
        assert out[0].score == pytest.approx(0.75)


class TestNetScoresIntegration:
    def test_regime_weights_applied_before_combination(self):
        bb = Blackboard(asof=NOW)
        bb.signal_sets.append(
            SignalSet(domain="technical", asof=NOW, signals=[_signal("momentum", score=1.0)]),
        )
        ctx = AgentContext(
            now=NOW,
            config={
                "strategy_regime_weights": {
                    "enabled": True,
                    "weights": {"Bull": {"momentum": 2.0}},
                },
            },
            market_regime=_regime("Bull", confidence=1.0),
        )
        scores = net_scores_for_blackboard(bb, ctx, ctx.config)
        assert scores["AAPL"] == pytest.approx(2.0)
