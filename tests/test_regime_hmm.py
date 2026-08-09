"""Tests for the HMM market-regime detection feature.

Covers the shared regime package (features, Laplace smoothing, Gaussian HMM
wrapper), the per-symbol ``regime_hmm`` strategy (signals, labelling,
no-look-ahead, determinism, graceful degradation), the RiskAgent market-regime
exposure overlay, and the runtime strategy wiring.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from firm.agents.base import AgentContext
from firm.agents.risk import RiskAgent
from firm.backtest.firm_strategy import PitViewAdapter
from firm.contracts.models import TradeProposal
from firm.data.pit_store import PointInTimeDataStore
from firm.data.synthetic import make_synthetic_prices
from firm.regime.features import (
    REGIME_FEATURES,
    apply_laplace_smoothing,
    compute_regime_features,
)
from firm.regime.model import (
    BEAR,
    BULL,
    CHOP,
    GaussianRegimeModel,
    RegimeState,
    RegimeUnavailable,
    hmm_available,
)
from firm.strategies.registry import get, list_strategies

pytestmark = pytest.mark.skipif(
    not hmm_available(), reason="hmmlearn not installed"
)

SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]


def _store(n_days: int = 500, end_date: str = "2023-12-31") -> PointInTimeDataStore:
    df = make_synthetic_prices(SYMBOLS, n_days=n_days, end_date=end_date)
    store = PointInTimeDataStore()
    store.load(prices=df)
    return store


def _view(store: PointInTimeDataStore, asof: datetime | None = None) -> PitViewAdapter:
    if asof is None:
        asof = store.get_prices(SYMBOLS, datetime(2100, 1, 1), 5)["date"].max()
    return PitViewAdapter(store, asof, SYMBOLS)


# ---------------------------------------------------------------------------
# Features + Laplace smoothing
# ---------------------------------------------------------------------------


class TestFeatures:
    def test_feature_matrix_shape_and_order(self):
        df = make_synthetic_prices(["AAPL"], n_days=300)
        feats = compute_regime_features(df)
        assert list(feats.columns) == REGIME_FEATURES
        assert len(feats) > 0
        assert np.isfinite(feats.values).all()

    def test_close_only_series_supported(self):
        # A close-only proxy (no high/low/volume) must still yield features.
        df = make_synthetic_prices(["AAPL"], n_days=200)[["date", "close"]]
        feats = compute_regime_features(df)
        assert list(feats.columns) == REGIME_FEATURES
        assert len(feats) > 0
        assert np.isfinite(feats.values).all()

    def test_empty_input(self):
        assert compute_regime_features(pd.DataFrame()).empty

    def test_laplace_smoothing_positive_and_normalized(self):
        transmat = np.array([[1.0, 0.0, 0.0], [0.3, 0.7, 0.0], [0.0, 0.0, 1.0]])
        smoothed = apply_laplace_smoothing(transmat, epsilon=1e-6)
        assert (smoothed > 0).all()
        np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# Gaussian HMM wrapper
# ---------------------------------------------------------------------------


class TestGaussianRegimeModel:
    def test_fit_classify_and_labels(self):
        feats = compute_regime_features(make_synthetic_prices(["AAPL"], n_days=400))
        model = GaussianRegimeModel(n_states=3).fit(feats.values)
        labels = set(model.label_map().values())
        assert BULL in labels and BEAR in labels  # extremes always present
        state = model.classify(feats.values)
        assert isinstance(state, RegimeState)
        assert state.label in {BULL, CHOP, BEAR}
        assert 0.0 <= state.confidence <= 1.0
        assert len(state.posterior) == 3

    def test_labelling_orders_by_mean_return(self):
        # Highest mean-return state -> Bull, lowest -> Bear, middle -> Chop.
        model = GaussianRegimeModel(n_states=3)
        means = np.array([[0.01, 0, 0, 0], [-0.02, 0, 0, 0], [0.0, 0, 0, 0]])
        label_map = model._build_label_map(means)
        assert label_map[0] == BULL  # +0.01
        assert label_map[1] == BEAR  # -0.02
        assert label_map[2] == CHOP  # 0.0

    def test_transmat_is_laplace_smoothed_after_fit(self):
        feats = compute_regime_features(make_synthetic_prices(["AAPL"], n_days=400))
        model = GaussianRegimeModel(n_states=3).fit(feats.values)
        assert (model._model.transmat_ > 0).all()

    def test_unfitted_raises(self):
        with pytest.raises(RegimeUnavailable):
            GaussianRegimeModel().label_map()

    def test_insufficient_data_raises(self):
        with pytest.raises(RegimeUnavailable):
            GaussianRegimeModel(n_states=3).fit(np.zeros((2, 4)))

    def test_separation_wide_gap_is_large(self):
        # Bull/Bear means are far apart relative to their variance -> a large
        # separation score (clearly distinguishable states).
        model = GaussianRegimeModel(n_states=3)
        means = np.array([[1.0, 0, 0, 0], [-1.0, 0, 0, 0], [0.0, 0, 0, 0]])
        covars = np.tile(np.eye(4) * 0.01, (3, 1, 1))
        sep = model._build_separation(means, covars)
        assert sep[BULL] > 5.0
        assert sep[BEAR] > 5.0

    def test_separation_thin_gap_is_small(self):
        # Bull/Chop/Bear means are nearly identical relative to their
        # variance -> a small separation score (label-switching risk).
        model = GaussianRegimeModel(n_states=3)
        means = np.array([[0.001, 0, 0, 0], [-0.001, 0, 0, 0], [0.0, 0, 0, 0]])
        covars = np.tile(np.eye(4) * 1.0, (3, 1, 1))
        sep = model._build_separation(means, covars)
        assert sep[BULL] < 0.01
        assert sep[BEAR] < 0.01

    def test_separation_populates_regime_state(self):
        feats = compute_regime_features(make_synthetic_prices(["AAPL"], n_days=400))
        model = GaussianRegimeModel(n_states=3).fit(feats.values)
        state = model.classify(feats.values)
        assert state.separation >= 0.0
        assert state.separation == model.separation(state.label)

    def test_separation_unknown_label_is_inf(self):
        model = GaussianRegimeModel(n_states=3)
        assert model.separation(CHOP) == float("inf")


# ---------------------------------------------------------------------------
# Per-symbol strategy
# ---------------------------------------------------------------------------


class TestRegimeHMMStrategy:
    def test_registered(self):
        assert "regime_hmm" in list_strategies()

    def test_generates_valid_signals(self):
        view = _view(_store())
        signals = get("regime_hmm")().generate(view)
        assert len(signals) > 0
        for sig in signals:
            assert sig.strategy == "regime_hmm"
            assert isinstance(sig.score, float)
            assert 0.0 <= sig.confidence <= 1.0
            assert sig.meta["regime"] in {BULL, CHOP, BEAR}
            assert len(sig.meta["posterior"]) == sig.meta["n_states"]
            # Sign convention: Bull positive, Bear negative.
            if sig.meta["regime"] == BULL:
                assert sig.score > 0
            elif sig.meta["regime"] == BEAR:
                assert sig.score < 0

    def test_empty_universe(self):
        store = _store()
        empty = PitViewAdapter(store, _view(store).asof, [])
        assert get("regime_hmm")().generate(empty) == []

    def test_deterministic(self):
        view = _view(_store())
        a = get("regime_hmm")().generate(view)
        b = get("regime_hmm")().generate(view)
        assert {s.symbol: s.meta["regime"] for s in a} == {
            s.symbol: s.meta["regime"] for s in b
        }
        assert {s.symbol: round(s.score, 9) for s in a} == {
            s.symbol: round(s.score, 9) for s in b
        }

    def test_no_look_ahead(self):
        """The signal at ``asof`` must not depend on data after ``asof``."""
        full = _store(n_days=520)
        asof = pd.Timestamp("2023-06-01")

        # A truncated store containing only rows on/before asof.
        df = make_synthetic_prices(SYMBOLS, n_days=520)
        truncated = PointInTimeDataStore()
        truncated.load(prices=df[df["date"] <= asof].copy())

        sig_full = {
            s.symbol: (s.meta["regime"], round(s.score, 9))
            for s in get("regime_hmm")().generate(PitViewAdapter(full, asof, SYMBOLS))
        }
        sig_trunc = {
            s.symbol: (s.meta["regime"], round(s.score, 9))
            for s in get("regime_hmm")().generate(
                PitViewAdapter(truncated, asof, SYMBOLS)
            )
        }
        assert sig_full == sig_trunc

    def test_graceful_degradation_when_hmm_unavailable(self, monkeypatch):
        def boom(self, X):
            raise RegimeUnavailable("simulated missing hmmlearn")

        monkeypatch.setattr(GaussianRegimeModel, "fit", boom)
        assert get("regime_hmm")().generate(_view(_store())) == []

    def test_low_separation_damps_bull_bear_signal(self, monkeypatch):
        """A thin-margin (low-separation) label should be damped toward
        neutral rather than traded at full confidence, but keep its sign."""

        def fake_classify(self, X):
            return RegimeState(
                label=BULL, confidence=0.9, state_idx=0, posterior=[0.9, 0.05, 0.05],
                separation=0.05,  # far below default min_state_separation=0.5
            )

        monkeypatch.setattr(GaussianRegimeModel, "classify", fake_classify)
        strat = get("regime_hmm")(
            {"min_state_separation": 0.5, "separation_damping_floor": 0.15}
        )
        signals = strat.generate(_view(_store()))
        assert signals
        for sig in signals:
            assert sig.meta["separation"] == 0.05
            assert sig.meta["separation_damping"] == pytest.approx(0.15)
            # Damped floor (0.15) applied to confidence (0.9), sign preserved.
            assert sig.score == pytest.approx(0.15 * 0.9)

    def test_high_separation_is_not_damped(self, monkeypatch):
        def fake_classify(self, X):
            return RegimeState(
                label=BEAR, confidence=0.8, state_idx=2, posterior=[0.1, 0.1, 0.8],
                separation=10.0,  # well above threshold
            )

        monkeypatch.setattr(GaussianRegimeModel, "classify", fake_classify)
        strat = get("regime_hmm")({"min_state_separation": 0.5})
        signals = strat.generate(_view(_store()))
        assert signals
        for sig in signals:
            assert sig.meta["separation_damping"] == pytest.approx(1.0)
            assert sig.score == pytest.approx(-0.8)

    def test_min_state_separation_disabled_is_noop(self, monkeypatch):
        def fake_classify(self, X):
            return RegimeState(
                label=BULL, confidence=0.7, state_idx=0, posterior=[0.7, 0.2, 0.1],
                separation=0.0,
            )

        monkeypatch.setattr(GaussianRegimeModel, "classify", fake_classify)
        strat = get("regime_hmm")({"min_state_separation": 0.0})
        signals = strat.generate(_view(_store()))
        assert signals
        for sig in signals:
            assert sig.score == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# RiskAgent market-regime exposure overlay
# ---------------------------------------------------------------------------


class TestRegimeOverlay:
    def _proposal(self):
        return TradeProposal(
            asof=datetime(2023, 6, 1), targets={"AAPL": 0.02, "MSFT": -0.02}
        )

    def test_disabled_by_default_is_noop(self):
        agent = RiskAgent(config={})
        ctx = AgentContext(now=datetime(2023, 6, 1))
        decision = agent.run(ctx, proposal=self._proposal(), portfolio=None)
        assert decision.adjusted_targets == {"AAPL": 0.02, "MSFT": -0.02}

    @pytest.mark.parametrize(
        "label,expected_scale",
        [(BULL, 1.5), (BEAR, 0.5), (CHOP, 0.25)],
    )
    def test_overlay_scales_gross_by_regime(self, label, expected_scale):
        agent = RiskAgent(config={"regime_overlay": {"enabled": True}})
        ctx = AgentContext(now=datetime(2023, 6, 1))
        regime = RegimeState(label=label, confidence=1.0, state_idx=0, posterior=[1.0])
        decision = agent.run(
            ctx, proposal=self._proposal(), portfolio=None, regime_state=regime
        )
        assert decision.adjusted_targets["AAPL"] == pytest.approx(0.02 * expected_scale)
        assert decision.adjusted_targets["MSFT"] == pytest.approx(-0.02 * expected_scale)
        assert any("Regime overlay" in a for a in decision.actions)

    def test_overlay_blends_by_confidence(self):
        # At 50% confidence a Bear (factor 0.5) scales by 1 + (0.5-1)*0.5 = 0.75.
        agent = RiskAgent(config={"regime_overlay": {"enabled": True}})
        ctx = AgentContext(now=datetime(2023, 6, 1))
        regime = RegimeState(label=BEAR, confidence=0.5, state_idx=1, posterior=[0.5])
        decision = agent.run(
            ctx, proposal=self._proposal(), portfolio=None, regime_state=regime
        )
        assert decision.adjusted_targets["AAPL"] == pytest.approx(0.02 * 0.75)

    def test_overlay_live_detection_from_pit_view(self):
        store = _store(n_days=600)
        ctx = AgentContext(now=_view(store).asof, pit_view=_view(store))
        agent = RiskAgent(config={"regime_overlay": {"enabled": True}})
        decision = agent.run(ctx, proposal=self._proposal(), portfolio=None)
        assert decision.approved
        assert any("Regime overlay" in a for a in decision.actions)

    def test_overlay_damps_low_separation_label(self):
        """Mirrors the per-symbol regime_hmm strategy's guard against
        label-switching: a high-confidence read of a label that's only
        noise-level distinct from its neighbour (thin separation) must be
        pulled back toward a no-op instead of applying the full playbook
        factor at full strength."""
        agent = RiskAgent(config={"regime_overlay": {"enabled": True}})
        ctx = AgentContext(now=datetime(2023, 6, 1))
        # Full confidence, but separation (0.1) sits far below the 0.5
        # threshold -> damping = max(0.15, 0.1/0.5) = 0.2.
        # effective = 1 + (0.5-1)*1.0*0.2 = 0.9 (vs 0.5 undamped).
        regime = RegimeState(
            label=BEAR, confidence=1.0, state_idx=1, posterior=[1.0], separation=0.1,
        )
        decision = agent.run(
            ctx, proposal=self._proposal(), portfolio=None, regime_state=regime
        )
        assert decision.adjusted_targets["AAPL"] == pytest.approx(0.02 * 0.9)
        assert any("label-switching risk" in a for a in decision.actions)

    def test_overlay_full_separation_undamped(self):
        """A well-separated label (at/above threshold) applies the full
        playbook factor, unchanged from before this damping existed."""
        agent = RiskAgent(config={"regime_overlay": {"enabled": True}})
        ctx = AgentContext(now=datetime(2023, 6, 1))
        regime = RegimeState(
            label=BEAR, confidence=1.0, state_idx=1, posterior=[1.0], separation=0.5,
        )
        decision = agent.run(
            ctx, proposal=self._proposal(), portfolio=None, regime_state=regime
        )
        assert decision.adjusted_targets["AAPL"] == pytest.approx(0.02 * 0.5)
        assert not any("label-switching risk" in a for a in decision.actions)

    def test_overlay_separation_damping_configurable(self):
        agent = RiskAgent(
            config={
                "regime_overlay": {
                    "enabled": True,
                    "min_state_separation": 1.0,
                    "separation_damping_floor": 0.5,
                }
            }
        )
        ctx = AgentContext(now=datetime(2023, 6, 1))
        # separation=0.1 / threshold=1.0 -> raw ratio 0.1, floored to 0.5.
        # effective = 1 + (0.5-1)*1.0*0.5 = 0.75.
        regime = RegimeState(
            label=BEAR, confidence=1.0, state_idx=1, posterior=[1.0], separation=0.1,
        )
        decision = agent.run(
            ctx, proposal=self._proposal(), portfolio=None, regime_state=regime
        )
        assert decision.adjusted_targets["AAPL"] == pytest.approx(0.02 * 0.75)


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


class TestWiring:
    def test_regime_hmm_wired_into_technical_analyst(self):
        from firm.runtime import build_orchestrator

        orch = build_orchestrator({"strategies": ["regime_hmm", "momentum"]})
        tech = orch.analysts[0]
        names = {s.name for s in tech.strategies}
        assert "regime_hmm" in names

    def test_overlay_config_reaches_risk_agent(self):
        from firm.runtime import build_orchestrator

        orch = build_orchestrator(
            {"strategies": ["regime_hmm"], "regime_overlay": {"enabled": True}}
        )
        assert orch.risk.regime_overlay_enabled is True
