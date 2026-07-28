"""Tests for Sharadar ETL normalizers and suggest_regime_weights helpers."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str, rel_path: str):
    path = _REPO / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestSharadarEtl:
    def test_normalize_prices_sep_columns(self):
        etl = _load_module("etl_sharadar", "scripts/etl_sharadar_to_cache.py")
        df = pd.DataFrame({
            "ticker": ["aapl"],
            "date": ["2020-01-02"],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "closeadj": [1.4],
            "volume": [1000],
        })
        out = etl.normalize_sharadar_prices(df)
        assert list(out.columns) == [
            "date", "symbol", "open", "high", "low", "close", "volume", "adj_close",
        ]
        assert out.iloc[0]["symbol"] == "AAPL"
        assert float(out.iloc[0]["adj_close"]) == 1.4

    def test_normalize_fundamentals_sf1_columns(self):
        etl = _load_module("etl_sharadar", "scripts/etl_sharadar_to_cache.py")
        df = pd.DataFrame({
            "ticker": ["MSFT"],
            "dimension": ["MRY"],
            "datekey": ["2020-03-31"],
            "marketcap": [1e12],
            "pe": [30.0],
            "pb": [10.0],
            "roe": [0.4],
            "revenue": [1e11],
            "netinc": [1e10],
            "eps": [5.0],
        })
        out = etl.normalize_sharadar_fundamentals(df)
        assert "symbol" in out.columns and out.iloc[0]["symbol"] == "MSFT"
        assert float(out.iloc[0]["market_cap"]) == 1e12


class TestSuggestRegimeWeights:
    def test_suggest_multipliers_boosts_high_sharpe_strategy(self):
        suggest = _load_module("suggest_rw", "scripts/suggest_strategy_regime_weights.py")
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        regime = pd.Series(["Bull"] * 10, index=dates)
        strategy_returns = {
            "winner": pd.Series([0.01] * 10, index=dates),
            "loser": pd.Series([-0.01] * 10, index=dates),
        }
        weights = suggest._suggest_multipliers(
            strategy_returns, regime, scale=0.5, floor=0.5, cap=1.5,
        )
        assert weights["Bull"]["winner"] > 1.0
        assert weights["Bull"]["loser"] < 1.0

    def test_soften_weights_json_roundtrip(self, tmp_path):
        """Helper for damped calibration runs (scale multipliers toward 1.0)."""
        raw = {
            "strategy_regime_weights": {
                "weights": {"Bull": {"momentum": 1.3, "trend": 0.7}},
            },
        }
        strength = 0.33
        weights = raw["strategy_regime_weights"]["weights"]
        for regime, strat_map in weights.items():
            for strat, mult in list(strat_map.items()):
                strat_map[strat] = round(1.0 + (mult - 1.0) * strength, 3)
        assert weights["Bull"]["momentum"] == pytest.approx(1.099)
        assert weights["Bull"]["trend"] == pytest.approx(0.901)
