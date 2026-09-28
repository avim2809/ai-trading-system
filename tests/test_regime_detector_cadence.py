"""MarketRegimeDetector.retrain_frequency must count new daily bars, not detect() calls.

Live runs ~7 cycles per trading day on the same completed-bar panel, and the
sleeved orchestrator calls RiskAgent once per sleeve per cycle -- counting
calls made `retrain_frequency: 5` ("refit weekly" in config/live.yaml) refit
at least daily on IBKR and several times per cycle on Alpaca (found
2026-09-28, docs/optimal_combination_fix_2026_09.md §6).
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from firm.backtest.firm_strategy import PitViewAdapter
from firm.data.pit_store import PointInTimeDataStore
from firm.data.synthetic import make_synthetic_prices
from firm.regime.detector import MarketRegimeDetector
from firm.regime.model import hmm_available

pytestmark = pytest.mark.skipif(not hmm_available(), reason="hmmlearn not installed")

SYMBOLS = ["AAPL", "MSFT", "GOOG"]


@pytest.fixture(scope="module")
def store() -> PointInTimeDataStore:
    s = PointInTimeDataStore()
    s.load(prices=make_synthetic_prices(SYMBOLS, n_days=500, end_date="2023-12-29"))
    return s


def _dates(store: PointInTimeDataStore) -> list[pd.Timestamp]:
    df = store.get_prices(SYMBOLS, datetime(2100, 1, 1), 1000)
    return sorted(pd.to_datetime(df["date"]).unique())


def _count_fits(monkeypatch) -> list[int]:
    """Count real HMM fits (patches the model class, so it works on any detector version)."""
    import firm.regime.detector as det_mod

    calls: list[int] = []
    orig = det_mod.GaussianRegimeModel.fit

    def wrapped(self, X):
        calls.append(1)
        return orig(self, X)

    monkeypatch.setattr(det_mod.GaussianRegimeModel, "fit", wrapped)
    return calls


def test_repeated_calls_on_same_bar_do_not_refit(store, monkeypatch):
    det = MarketRegimeDetector(retrain_frequency=5, min_data_points=30)
    fits = _count_fits(monkeypatch)
    view = PitViewAdapter(store, _dates(store)[-1], SYMBOLS)
    for _ in range(40):  # ~6 live days' worth of cycles, all on one bar
        det.detect(view)
    assert len(fits) == 1


def test_refits_every_n_new_bars(store, monkeypatch):
    det = MarketRegimeDetector(retrain_frequency=5, min_data_points=30)
    fits = _count_fits(monkeypatch)
    days = _dates(store)[-20:]
    for d in days:
        view = PitViewAdapter(store, d, SYMBOLS)
        for _ in range(7):  # 7 intraday cycles per day
            det.detect(view)
    # first fit + one every 5 new bars over the remaining 19 days
    assert len(fits) == 1 + 19 // 5


def test_daily_cadence_unchanged(store, monkeypatch):
    """One call per bar (the backtest) keeps the exact previous refit schedule."""
    det = MarketRegimeDetector(retrain_frequency=5, min_data_points=30)
    fits = _count_fits(monkeypatch)
    for d in _dates(store)[-20:]:
        det.detect(PitViewAdapter(store, d, SYMBOLS))
    assert len(fits) == 1 + 19 // 5
