"""Tests for fundamentals cache helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from firm.data.fundamentals_cache import (
    hours_since_refresh,
    merge_with_cached_fundamentals,
    save_refresh_meta,
    symbols_missing_fundamentals,
)


def test_symbols_missing_fundamentals():
    cached = pd.DataFrame({"symbol": ["AAPL", "MSFT"]})
    assert symbols_missing_fundamentals(["AAPL", "GOOG"], cached) == ["GOOG"]


def test_merge_with_cached_fundamentals():
    cached = pd.DataFrame({"date": ["2024-01-01"], "symbol": ["MSFT"], "pe_ratio": [30.0]})
    live = pd.DataFrame({"date": ["2024-01-01"], "symbol": ["AAPL"], "pe_ratio": [25.0]})
    merged = merge_with_cached_fundamentals(live, cached)
    assert set(merged["symbol"]) == {"AAPL", "MSFT"}


def test_refresh_meta_roundtrip(tmp_path: Path):
    save_refresh_meta(tmp_path, symbols=["AAPL"], row_count=5, symbol_count=1)
    age = hours_since_refresh(tmp_path)
    assert age is not None
    assert age < 1.0
    meta_path = tmp_path / "combined/fundamentals_refresh.json"
    assert meta_path.exists()
    data = json.loads(meta_path.read_text())
    assert data["row_count"] == 5
