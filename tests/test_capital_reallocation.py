"""Tests for adaptive per-sleeve capital reweighting (2026-09-22):
``firm.live.capital_reallocation`` (pure weight-tilt math),
``firm.live.capital_reallocation_job`` (the scheduled recommendation
check), and the ``LiveTradingEngine``/API wiring around them.

Disabled-by-default, human-gated capability. ``Orchestrator.
get_sleeve_metrics()`` (this module's only performance input) and
``Orchestrator.apply_capital_reallocation`` (the actual cash-moving
mechanism) are covered in ``tests/test_sleeves.py`` alongside the rest of
the sleeve-portfolio mechanics they depend on; this file covers the
weight-tilt math itself, the "never auto-applies" scheduled-job
behavior, and the read/apply wrappers on top of it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from firm.live.approval import ApprovalQueue
from firm.live.capital_reallocation import compute_reallocated_weights, weights_only
from firm.live.data_feed import LiveDataFeed
from firm.live.engine import LiveTradingEngine
from tests.test_brokers import MockBroker


def _metrics(score: float, n_days: int = 60, metric: str = "sortino_ratio") -> dict[str, float]:
    return {metric: score, "n_days": float(n_days)}


class TestComputeReallocatedWeights:
    def test_empty_current_weights_returns_empty(self):
        assert compute_reallocated_weights({}, {}) == {}

    def test_zero_tilt_strength_reproduces_current_weights_exactly(self):
        """tilt_strength=0 is the mechanism's own no-op knob, distinct from
        the scheduler-level ``enabled`` flag -- must reproduce today's
        static split exactly regardless of how different the underlying
        scores are."""
        current = {"momentum": 0.5, "mean_reversion": 0.3, "stat_arb": 0.2}
        metrics = {
            "momentum": _metrics(3.0),
            "mean_reversion": _metrics(0.0),
            "stat_arb": _metrics(-3.0),
        }
        # cap_pct loosened: the default (0.35) is deliberately tight for
        # realistic sleeve counts (11 strategies) and would otherwise clip
        # momentum's already-legitimate 0.5 static weight regardless of
        # tilt_strength -- a separate, real property of floor/cap (see
        # test_cap_protects_against_one_strategy_taking_the_whole_book),
        # not what this test isolates.
        result = compute_reallocated_weights(metrics, current, tilt_strength=0.0, cap_pct=0.9)
        for strategy, weight in current.items():
            assert result[strategy]["new_weight"] == pytest.approx(weight)

    def test_tilts_toward_better_relative_performer(self):
        current = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        metrics = {"a": _metrics(2.0), "b": _metrics(0.0), "c": _metrics(-2.0)}
        result = compute_reallocated_weights(metrics, current, floor_pct=0.03, cap_pct=0.9)
        assert result["a"]["new_weight"] > result["b"]["new_weight"] > result["c"]["new_weight"]
        assert sum(r["new_weight"] for r in result.values()) == pytest.approx(1.0)

    def test_insufficient_history_keeps_current_weight_exactly(self):
        current = {"a": 0.5, "b": 0.5}
        metrics = {"a": _metrics(5.0, n_days=60), "b": _metrics(-5.0, n_days=5)}
        result = compute_reallocated_weights(metrics, current, min_track_days=30)
        assert result["b"]["new_weight"] == pytest.approx(0.5)
        assert result["b"]["reason"] == "insufficient_history"
        assert result["a"]["reason"] == "reweighted"

    def test_missing_metric_key_is_insufficient_history(self):
        current = {"a": 0.5, "b": 0.5}
        metrics = {"a": _metrics(5.0), "b": {"n_days": 90.0}}  # no sortino_ratio at all
        result = compute_reallocated_weights(metrics, current)
        assert result["b"]["reason"] == "insufficient_history"
        assert result["b"]["new_weight"] == pytest.approx(0.5)

    def test_no_sleeve_metrics_at_all_is_a_full_no_op(self):
        current = {"a": 0.6, "b": 0.4}
        result = compute_reallocated_weights({}, current)
        assert result["a"]["new_weight"] == pytest.approx(0.6)
        assert result["b"]["new_weight"] == pytest.approx(0.4)

    def test_floor_protects_extreme_underperformer(self):
        current = {"a": 0.34, "b": 0.33, "c": 0.33}
        metrics = {"a": _metrics(0.1), "b": _metrics(0.1), "c": _metrics(-20.0)}
        result = compute_reallocated_weights(
            metrics, current, floor_pct=0.05, cap_pct=0.9, tilt_strength=1.0,
        )
        assert result["c"]["new_weight"] == pytest.approx(0.05, abs=1e-6)
        assert sum(r["new_weight"] for r in result.values()) == pytest.approx(1.0)

    def test_cap_protects_against_one_strategy_taking_the_whole_book(self):
        current = {"a": 0.34, "b": 0.33, "c": 0.33}
        metrics = {"a": _metrics(20.0), "b": _metrics(0.1), "c": _metrics(0.1)}
        result = compute_reallocated_weights(
            metrics, current, floor_pct=0.03, cap_pct=0.4, tilt_strength=1.0,
        )
        assert result["a"]["new_weight"] == pytest.approx(0.4, abs=1e-6)
        assert sum(r["new_weight"] for r in result.values()) == pytest.approx(1.0)

    def test_infeasible_floor_falls_back_to_equal_split(self):
        """floor_pct * n > budget can't be satisfied simultaneously for
        every eligible sleeve -- must not raise or produce a lopsided
        result, just the least-arbitrary fallback (equal split)."""
        current = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        metrics = {s: _metrics(v) for s, v in (("a", 5.0), ("b", 0.0), ("c", -5.0))}
        result = compute_reallocated_weights(metrics, current, floor_pct=0.5, cap_pct=0.9)
        for s in current:
            assert result[s]["new_weight"] == pytest.approx(1 / 3)

    def test_z_score_zero_when_every_eligible_peer_ties(self):
        """Std-dev-zero guard: identical peer scores must not divide by
        zero or produce a spurious tilt."""
        current = {"a": 0.5, "b": 0.5}
        metrics = {"a": _metrics(1.0), "b": _metrics(1.0)}
        result = compute_reallocated_weights(metrics, current, tilt_strength=1.0)
        assert result["a"]["new_weight"] == pytest.approx(0.5)
        assert result["b"]["new_weight"] == pytest.approx(0.5)

    def test_weights_only_extracts_flat_map(self):
        current = {"a": 0.5, "b": 0.5}
        metrics = {"a": _metrics(1.0), "b": _metrics(-1.0)}
        result = compute_reallocated_weights(metrics, current)
        flat = weights_only(result)
        assert set(flat) == {"a", "b"}
        assert sum(flat.values()) == pytest.approx(1.0)


class TestComputeRecommendation:
    @staticmethod
    def _engine(*, mode="sleeved", current_weights=None, sleeve_metrics=None):
        engine = MagicMock()
        engine._orchestrator.capital_allocation_mode = mode
        engine._orchestrator.sleeve_capital_weights.return_value = current_weights or {}
        engine._orchestrator.get_sleeve_metrics.return_value = sleeve_metrics or {}
        return engine

    def test_blended_mode_returns_empty_with_note(self):
        from firm.live.capital_reallocation_job import compute_recommendation

        result = compute_recommendation(self._engine(mode="blended"))
        assert result["sleeves"] == {}
        assert "not 'sleeved'" in result["note"]

    def test_no_sleeve_has_traded_yet_returns_empty_with_note(self):
        from firm.live.capital_reallocation_job import compute_recommendation

        result = compute_recommendation(self._engine(mode="sleeved", current_weights={}))
        assert result["sleeves"] == {}
        assert "no sleeve has traded" in result["note"]

    def test_sleeved_mode_computes_a_real_recommendation(self):
        from firm.live.capital_reallocation_job import compute_recommendation

        # 3 sleeves, not 2: the default cap_pct=0.35 is only feasible for
        # cap_pct * n_eligible >= 1.0 (n >= ~3) -- with only 2 eligible
        # sleeves it's mathematically infeasible and the water-fill's own
        # defensive fallback (equal split) would mask the tilt this test
        # means to demonstrate. Realistic production sleeve counts (11
        # strategies) are well clear of this edge case.
        engine = self._engine(
            mode="sleeved",
            current_weights={"momentum": 1 / 3, "mean_reversion": 1 / 3, "stat_arb": 1 / 3},
            sleeve_metrics={
                "momentum": _metrics(2.0),
                "mean_reversion": _metrics(-2.0),
                "stat_arb": _metrics(0.0),
            },
        )
        result = compute_recommendation(engine)
        sleeves = result["sleeves"]
        assert sleeves["momentum"]["new_weight"] > sleeves["mean_reversion"]["new_weight"]
        assert "params" in result


class TestRunScheduledCapitalReallocationCheck:
    def test_noop_when_disabled(self):
        from firm.live.capital_reallocation_job import run_scheduled_capital_reallocation_check

        engine = MagicMock()
        engine._config = {"capital_reallocation": {"enabled": False}}
        run_scheduled_capital_reallocation_check(engine)
        engine._orchestrator.get_sleeve_metrics.assert_not_called()

    def test_noop_when_config_block_absent(self):
        from firm.live.capital_reallocation_job import run_scheduled_capital_reallocation_check

        engine = MagicMock()
        engine._config = {}
        run_scheduled_capital_reallocation_check(engine)
        engine._orchestrator.get_sleeve_metrics.assert_not_called()

    def test_computes_but_never_applies_when_enabled(self):
        """The whole point of shipping this as a recommendation: enabling
        the scheduled check must never call the mutating apply path."""
        from firm.live.capital_reallocation_job import run_scheduled_capital_reallocation_check

        engine = MagicMock()
        engine._config = {"capital_reallocation": {"enabled": True}}
        engine._orchestrator.capital_allocation_mode = "sleeved"
        engine._orchestrator.sleeve_capital_weights.return_value = {
            "momentum": 0.5, "mean_reversion": 0.5,
        }
        engine._orchestrator.get_sleeve_metrics.return_value = {
            "momentum": _metrics(2.0), "mean_reversion": _metrics(-2.0),
        }
        run_scheduled_capital_reallocation_check(engine)
        engine._orchestrator.get_sleeve_metrics.assert_called_once()
        engine._orchestrator.apply_capital_reallocation.assert_not_called()
        engine.apply_capital_reallocation.assert_not_called()

    def test_never_raises_on_internal_failure(self):
        from firm.live.capital_reallocation_job import run_scheduled_capital_reallocation_check

        engine = MagicMock()
        engine._config = {"capital_reallocation": {"enabled": True}}
        engine._orchestrator.capital_allocation_mode = "sleeved"
        engine._orchestrator.sleeve_capital_weights.side_effect = RuntimeError("boom")
        run_scheduled_capital_reallocation_check(engine)  # must not raise


@pytest.fixture()
def engine_components(tmp_path):
    broker = MockBroker()
    feed = LiveDataFeed(providers={}, universe=["AAPL", "MSFT"])
    queue = ApprovalQueue(broker=broker)
    config = {"initial_capital": 100_000, "memory_log_path": str(tmp_path / "decisions.jsonl")}
    return broker, feed, queue, config


class TestEngineCapitalReallocationWrappers:
    @staticmethod
    def _sleeved_engine(broker, feed, queue, config, **extra_cfg):
        cfg = {
            **config,
            "capital_allocation_mode": "sleeved",
            "strategies": ["momentum", "mean_reversion"],
            **extra_cfg,
        }
        return LiveTradingEngine(config=cfg, broker=broker, data_feed=feed, approval_queue=queue)

    def test_recommendation_blended_engine_returns_empty(self, engine_components):
        broker, feed, queue, config = engine_components
        engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)
        result = engine.get_capital_reallocation_recommendation()
        assert result["sleeves"] == {}

    def test_apply_raises_when_nothing_to_apply(self, engine_components):
        broker, feed, queue, config = engine_components
        engine = self._sleeved_engine(broker, feed, queue, config)
        with pytest.raises(ValueError):
            engine.apply_capital_reallocation()

    def test_recommendation_uses_configured_params(self, engine_components):
        broker, feed, queue, config = engine_components
        engine = self._sleeved_engine(
            broker, feed, queue, config,
            capital_reallocation={"floor_pct": 0.1, "cap_pct": 0.5, "metric": "sharpe_ratio"},
        )
        engine._orchestrator._get_or_create_sleeve_portfolio("momentum", 0.5)
        engine._orchestrator._get_or_create_sleeve_portfolio("mean_reversion", 0.5)

        result = engine.get_capital_reallocation_recommendation()

        assert result["params"]["floor_pct"] == 0.1
        assert result["params"]["cap_pct"] == 0.5
        assert result["params"]["metric"] == "sharpe_ratio"

    def test_apply_moves_cash_and_updates_config(self, engine_components, monkeypatch):
        broker, feed, queue, config = engine_components
        engine = self._sleeved_engine(broker, feed, queue, config)
        momentum = engine._orchestrator._get_or_create_sleeve_portfolio("momentum", 0.5)
        mean_reversion = engine._orchestrator._get_or_create_sleeve_portfolio("mean_reversion", 0.5)
        assert momentum.nav == pytest.approx(50_000.0)
        assert mean_reversion.nav == pytest.approx(50_000.0)

        # apply_capital_reallocation always re-derives from
        # get_capital_reallocation_recommendation rather than trusting a
        # caller-supplied weights blob -- fake that one call so this test
        # doesn't need 30+ days of synthetic NAV history just to make
        # get_sleeve_metrics produce an eligible (non-frozen) sleeve.
        fake_recommendation = {
            "sleeves": {
                "momentum": {"current_weight": 0.5, "new_weight": 0.7, "reason": "reweighted"},
                "mean_reversion": {"current_weight": 0.5, "new_weight": 0.3, "reason": "reweighted"},
            },
            "params": {},
        }
        monkeypatch.setattr(
            engine, "get_capital_reallocation_recommendation", lambda **kw: fake_recommendation,
        )

        result = engine.apply_capital_reallocation()

        assert momentum.nav == pytest.approx(70_000.0)
        assert mean_reversion.nav == pytest.approx(30_000.0)
        assert engine._config["strategy_capital_weights"] == {"momentum": 0.7, "mean_reversion": 0.3}
        assert result["summary"]["momentum"]["weight_applied"] == pytest.approx(0.7)
        assert result["recommendation"] is fake_recommendation


class TestCapitalReallocationEndpoints:
    """firm.api.routers.live's capital_reallocation_recommendation/apply --
    thin wrappers, tested the same direct-call way as
    TestLiveAttributionEndpointSleeveMerge in test_sleeves.py."""

    @staticmethod
    def _fake_request(engine):
        request = MagicMock()
        request.app.state.live_engine = engine
        return request

    def test_recommendation_no_engine_raises_400(self):
        from firm.api.routers.live import capital_reallocation_recommendation

        request = MagicMock()
        request.app.state.live_engine = None
        with pytest.raises(HTTPException) as exc:
            capital_reallocation_recommendation(request)
        assert exc.value.status_code == 400

    def test_recommendation_delegates_to_engine(self):
        from firm.api.routers.live import capital_reallocation_recommendation

        engine = MagicMock()
        engine.get_capital_reallocation_recommendation.return_value = {"sleeves": {}, "note": "x"}
        result = capital_reallocation_recommendation(self._fake_request(engine))
        assert result == {"sleeves": {}, "note": "x"}

    def test_apply_no_engine_raises_400(self):
        from firm.api.routers.live import capital_reallocation_apply

        request = MagicMock()
        request.app.state.live_engine = None
        with pytest.raises(HTTPException) as exc:
            capital_reallocation_apply(request)
        assert exc.value.status_code == 400

    def test_apply_converts_value_error_to_400(self):
        from firm.api.routers.live import capital_reallocation_apply

        engine = MagicMock()
        engine.apply_capital_reallocation.side_effect = ValueError("nothing to apply")
        with pytest.raises(HTTPException) as exc:
            capital_reallocation_apply(self._fake_request(engine))
        assert exc.value.status_code == 400

    def test_apply_delegates_to_engine(self):
        from firm.api.routers.live import capital_reallocation_apply

        engine = MagicMock()
        engine.apply_capital_reallocation.return_value = {"recommendation": {}, "summary": {}}
        result = capital_reallocation_apply(self._fake_request(engine))
        assert result == {"recommendation": {}, "summary": {}}
