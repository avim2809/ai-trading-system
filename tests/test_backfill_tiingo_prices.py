"""Tests for scripts/backfill_tiingo_prices.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from firm.data.cache import ParquetCache
from firm.data.schemas import PRICE_COLS

_REPO = Path(__file__).resolve().parents[1]


def _load_backfill():
    path = _REPO / "scripts" / "backfill_tiingo_prices.py"
    spec = importlib.util.spec_from_file_location("backfill_tiingo", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


class TestLoadLiveUniverseSymbols:
    def test_reads_universe_from_live_yaml(self, tmp_path):
        backfill = _load_backfill()
        cfg = tmp_path / "live.yaml"
        cfg.write_text(
            yaml.dump({"universe": {"symbols": ["aapl", "MSFT", "spy"]}}),
            encoding="utf-8",
        )
        assert backfill.load_live_universe_symbols(cfg) == ["AAPL", "MSFT", "SPY"]


class TestBackfillSymbols:
    def test_merges_per_symbol_into_combined_prices(self, tmp_path):
        backfill = _load_backfill()
        cache_dir = str(tmp_path)

        def _fake_prices(symbols, start, end):
            sym = symbols[0]
            return pd.DataFrame({
                "date": [pd.Timestamp("2020-01-02")],
                "symbol": [sym],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.0],
                "volume": [100.0],
                "adj_close": [1.0],
            })[PRICE_COLS]

        mock_provider = MagicMock()
        mock_provider.get_prices.side_effect = _fake_prices

        with patch.object(backfill, "TiingoProvider", return_value=mock_provider), \
             patch.object(backfill, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(cache_dir=cache_dir)
            result = backfill.backfill_symbols(
                ["AAPL", "MSFT"],
                start="2020-01-01",
                end="2020-12-31",
                cache_dir=cache_dir,
                min_interval_sec=0,
                dry_run=False,
            )

        assert result["fetched"] == 2
        assert result["failed"] == []
        combined = ParquetCache(cache_dir).get("combined/prices")
        assert set(combined["symbol"]) == {"AAPL", "MSFT"}

    def test_dry_run_does_not_write(self, tmp_path):
        backfill = _load_backfill()
        cache_dir = str(tmp_path)
        with patch.object(backfill, "get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(cache_dir=cache_dir)
            result = backfill.backfill_symbols(
                ["AAPL"],
                start="2020-01-01",
                end="2020-12-31",
                cache_dir=cache_dir,
                min_interval_sec=0,
                dry_run=True,
            )
        assert result["fetched"] == 0
        assert ParquetCache(cache_dir).get("combined/prices") is None
