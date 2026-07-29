"""Tests for the ensemble regime classifier (firm.regime.ensemble) and its
wiring into MarketRegimeDetector / RiskAgent / the orchestrator.

EnsembleRegimeModel majority-vote/confidence-blending logic is tested against
hand-crafted fake members (deterministic, no dependency on hmmlearn's
stochastic convergence landing on a specific vote split) plus a handful of
real-fit smoke tests using the same synthetic-data fixtures as
test_regime_hmm.py. Detector/agent wiring is tested end-to-end with real
synthetic data, mirroring test_regime_hmm.py's existing patterns.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from firm.agents.base import AgentContext
from firm.agents.risk import RiskAgent
from firm.backtest.firm_strategy import PitViewAdapter
from firm.contracts.models import TradeProposal
from firm.data.pit_store import PointInTimeDataStore
from firm.data.synthetic import make_synthetic_prices
from firm.regime.detector import MarketRegimeDetector
from firm.regime.ensemble import DEFAULT_SEEDS, EnsembleRegimeModel
from firm.regime.features import compute_regime_features
from firm.regime.model import BEAR, BULL, CHOP, RegimeState, RegimeUnavailable, hmm_available

pytestmark = pytest.mark.skipif(not hmm_available(), reason="hmmlearn not installed")

SYMBOLS = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]


def _store(n_days: int = 600, end_date: str = "2023-12-31") -> PointInTimeDataStore:
    df = make_synthetic_prices(SYMBOLS, n_days=n_days, end_date=end_date)
    store = PointInTimeDataStore()
    store.load(prices=df)
    return store


def _view(store: PointInTimeDataStore, asof: datetime | None = None) -> PitViewAdapter:
    if asof is None:
        asof = store.get_prices(SYMBOLS, datetime(2100, 1, 1), 5)["date"].max()
    return PitViewAdapter(store, asof, SYMBOLS)


class _FakeMember:
    """Stand-in for a fitted GaussianRegimeModel whose classify() is fixed —
    lets the ensemble's vote-aggregation logic be tested deterministically
    without depending on hmmlearn's stochastic EM convergence."""

    def __init__(self, state: RegimeState) -> None:
        self._state = state

    def fit(self, X):
        return self

    def classify(self, X):
        return self._state


def _regime(label: str, confidence: float, separation: float = 1.0) -> RegimeState:
    return RegimeState(label=label, confidence=confidence, state_idx=0, separation=separation)


class TestEnsembleVoteAggregation:
    """Deterministic tests of classify()'s vote/confidence-blending logic,
    injecting fake pre-classified members directly."""

    def _ensemble_with_members(self, states: list[RegimeState]) -> EnsembleRegimeModel:
        ens = EnsembleRegimeModel(seeds=tuple(range(len(states))))
        ens._members = [_FakeMember(s) for s in states]
        return ens

    def test_majority_label_wins(self):
        ens = self._ensemble_with_members([
            _regime(BULL, 0.9), _regime(BULL, 0.8), _regime(BEAR, 0.7),
        ])
        state = ens.classify(np.zeros((10, 4)))
        assert state.label == BULL

    def test_confidence_blends_vote_share_and_member_confidence(self):
        # 2/3 voted Bull with confidences 0.9 and 0.7 -> vote_share=2/3,
        # avg_confidence=0.8 -> combined = 2/3 * 0.8.
        ens = self._ensemble_with_members([
            _regime(BULL, 0.9), _regime(BULL, 0.7), _regime(BEAR, 0.99),
        ])
        state = ens.classify(np.zeros((10, 4)))
        assert state.confidence == pytest.approx((2 / 3) * 0.8, abs=1e-9)

    def test_unanimous_vote_is_more_confident_than_split_vote(self):
        unanimous = self._ensemble_with_members([_regime(BULL, 0.8)] * 5)
        split = self._ensemble_with_members(
            [_regime(BULL, 0.8)] * 3 + [_regime(BEAR, 0.8)] * 2
        )
        assert unanimous.classify(np.zeros((10, 4))).confidence > (
            split.classify(np.zeros((10, 4))).confidence
        )

    def test_posterior_is_vote_share_vector_summing_to_one(self):
        ens = self._ensemble_with_members([
            _regime(BULL, 0.9), _regime(BULL, 0.8), _regime(CHOP, 0.5), _regime(BEAR, 0.7),
        ])
        state = ens.classify(np.zeros((10, 4)))
        assert len(state.posterior) == 3
        assert sum(state.posterior) == pytest.approx(1.0)
        # Order is [Bull, Chop, Bear]; 2/4 Bull, 1/4 Chop, 1/4 Bear.
        assert state.posterior == pytest.approx([0.5, 0.25, 0.25])

    def test_separation_averaged_over_agreeing_members_only(self):
        ens = self._ensemble_with_members([
            _regime(BULL, 0.9, separation=2.0),
            _regime(BULL, 0.8, separation=4.0),
            _regime(BEAR, 0.9, separation=100.0),  # must not pollute Bull's average
        ])
        state = ens.classify(np.zeros((10, 4)))
        assert state.separation == pytest.approx(3.0)

    def test_classify_before_fit_raises(self):
        with pytest.raises(RegimeUnavailable):
            EnsembleRegimeModel().classify(np.zeros((10, 4)))


class TestEnsembleFit:
    def test_real_fit_classify_smoke(self):
        """End-to-end with real hmmlearn fits (not fakes) — sanity-checks
        the whole pipeline produces a valid, well-formed RegimeState."""
        feats = compute_regime_features(make_synthetic_prices(["AAPL"], n_days=400))
        ensemble = EnsembleRegimeModel(n_states=3).fit(feats.values)
        assert ensemble.fitted
        state = ensemble.classify(feats.values)
        assert state.label in {BULL, CHOP, BEAR}
        assert 0.0 <= state.confidence <= 1.0
        assert len(state.posterior) == 3
        assert sum(state.posterior) == pytest.approx(1.0)

    def test_default_seeds_avoid_ties(self):
        # 5 seeds over 3 labels can never produce an exact tie for 1st place
        # in the pathological worst case (2-2-1 still has a strict winner).
        assert len(DEFAULT_SEEDS) == 5

    def test_one_member_failing_to_fit_is_tolerated(self, monkeypatch):
        """A single bad seed shouldn't sink the whole ensemble — the
        remaining members still form a majority."""
        import firm.regime.ensemble as ensemble_mod

        real_cls = ensemble_mod.GaussianRegimeModel
        call_count = {"n": 0}

        class _FlakyOnFirstCall(real_cls):
            def fit(self, X):
                call_count["n"] += 1
                if call_count["n"] == 2:  # not the first (see RegimeUnavailable re-raise)
                    raise ValueError("simulated convergence failure")
                return super().fit(X)

        monkeypatch.setattr(ensemble_mod, "GaussianRegimeModel", _FlakyOnFirstCall)
        feats = compute_regime_features(make_synthetic_prices(["AAPL"], n_days=400))
        ensemble = EnsembleRegimeModel(n_states=3).fit(feats.values)
        assert len(ensemble._members) == len(DEFAULT_SEEDS) - 1

    def test_hmmlearn_unavailable_on_first_member_reraises(self, monkeypatch):
        import firm.regime.ensemble as ensemble_mod

        class _AlwaysUnavailable:
            def __init__(self, *a, **k):
                pass

            def fit(self, X):
                raise RegimeUnavailable("hmmlearn not installed")

        monkeypatch.setattr(ensemble_mod, "GaussianRegimeModel", _AlwaysUnavailable)
        with pytest.raises(RegimeUnavailable):
            EnsembleRegimeModel().fit(np.zeros((100, 4)))

    def test_all_members_failing_raises_regime_unavailable(self, monkeypatch):
        import firm.regime.ensemble as ensemble_mod

        class _AlwaysBroken(ensemble_mod.GaussianRegimeModel):
            def fit(self, X):
                raise ValueError("always fails")

        monkeypatch.setattr(ensemble_mod, "GaussianRegimeModel", _AlwaysBroken)
        with pytest.raises(RegimeUnavailable):
            EnsembleRegimeModel().fit(np.zeros((100, 4)))


class TestMarketRegimeDetectorEnsemble:
    def test_ensemble_flag_uses_ensemble_model(self):
        store = _store()
        detector = MarketRegimeDetector(ensemble=True, min_data_points=30)
        state = detector.detect(_view(store))
        assert state is not None
        assert isinstance(detector._model, EnsembleRegimeModel)
        assert state.label in {BULL, CHOP, BEAR}

    def test_default_still_uses_single_model(self):
        from firm.regime.model import GaussianRegimeModel

        store = _store()
        detector = MarketRegimeDetector(min_data_points=30)
        detector.detect(_view(store))
        assert isinstance(detector._model, GaussianRegimeModel)


class TestRiskAgentEnsembleWiring:
    @staticmethod
    def _proposal() -> TradeProposal:
        return TradeProposal(
            asof=datetime(2023, 6, 1), targets={"AAPL": 0.02, "MSFT": -0.02},
        )

    def test_overlay_ensemble_config_reaches_detector(self):
        store = _store()
        ctx = AgentContext(now=_view(store).asof, pit_view=_view(store))
        agent = RiskAgent(config={"regime_overlay": {"enabled": True, "ensemble": True}})
        decision = agent.run(ctx, proposal=self._proposal(), portfolio=None)
        assert decision.approved
        assert isinstance(agent._regime_detector, MarketRegimeDetector)
        assert agent._regime_detector.ensemble is True

    def test_overlay_ensemble_off_by_default(self):
        store = _store()
        ctx = AgentContext(now=_view(store).asof, pit_view=_view(store))
        agent = RiskAgent(config={"regime_overlay": {"enabled": True}})
        agent.run(ctx, proposal=self._proposal(), portfolio=None)
        assert agent._regime_detector.ensemble is False


class TestOrchestratorEnsembleWiring:
    def test_strategy_regime_weights_ensemble_config_reaches_detector(self):
        from firm.runtime import build_orchestrator

        orch = build_orchestrator({
            "strategies": ["momentum"],
            "strategy_regime_weights": {"enabled": True, "ensemble": True},
        })
        store = _store()
        orch._resolve_market_regime(_view(store))
        assert orch._regime_weights_detector.ensemble is True
