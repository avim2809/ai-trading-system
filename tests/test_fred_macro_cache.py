"""Tests for the FRED macro cache round-trip (PART 3 Phase 4 of the
remediation plan) -- ``macro_bundle_to_long_frame``/``macro_bundle_from_cache``
and ``firm.runtime.load_macro``'s cache-vs-live-fetch preference.

Before this, every backtest with FRED_API_KEY set fetched macro data live,
over the network, on every single run (see runtime.py's macro-loading
block) -- these tests lock in that the cached path now takes priority and
that the round-trip is lossless.
"""

from __future__ import annotations

import pandas as pd

from firm.data.providers.fred import macro_bundle_from_cache, macro_bundle_to_long_frame


def test_round_trip_preserves_values_and_dates():
    bundle = {
        "T10Y2Y": pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "T10Y2Y": [-0.3, -0.28]}),
        "VIXCLS": pd.DataFrame({"date": ["2024-01-02", "2024-01-03"], "VIXCLS": [13.1, 12.9]}),
    }
    long_frame = macro_bundle_to_long_frame(bundle)
    assert set(long_frame["symbol"].unique()) == {"T10Y2Y", "VIXCLS"}
    assert len(long_frame) == 4

    restored = macro_bundle_from_cache(long_frame)
    assert set(restored.keys()) == {"T10Y2Y", "VIXCLS"}
    assert list(restored["T10Y2Y"]["T10Y2Y"]) == [-0.3, -0.28]
    assert list(restored["VIXCLS"]["VIXCLS"]) == [13.1, 12.9]
    assert restored["T10Y2Y"].columns.tolist() == ["date", "T10Y2Y"]


def test_empty_series_are_skipped_by_to_long_frame():
    bundle = {"T10Y2Y": pd.DataFrame({"date": ["2024-01-02"], "T10Y2Y": [-0.3]}), "EMPTY": pd.DataFrame()}
    long_frame = macro_bundle_to_long_frame(bundle)
    assert "EMPTY" not in set(long_frame["symbol"].unique())


def test_to_long_frame_empty_bundle_returns_empty_frame_with_right_columns():
    long_frame = macro_bundle_to_long_frame({})
    assert long_frame.empty
    assert list(long_frame.columns) == ["date", "symbol", "value"]


def test_from_cache_none_returns_empty_dict():
    assert macro_bundle_from_cache(None) == {}


def test_from_cache_empty_frame_returns_empty_dict():
    assert macro_bundle_from_cache(pd.DataFrame(columns=["date", "symbol", "value"])) == {}


def test_load_macro_prefers_cache_over_live_fetch(monkeypatch):
    from unittest.mock import MagicMock

    from firm import runtime

    cached_long = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]), "symbol": ["T10Y2Y"], "value": [-0.3],
    })
    fake_cache = MagicMock()
    fake_cache.get.return_value = cached_long
    # ParquetCache is imported *inside* load_macro (deferred import, matching
    # the rest of this module's optional-dependency style), so patch it at
    # its source module -- the local import resolves the current attribute
    # on firm.data.cache at call time, not a runtime.ParquetCache that never
    # exists.
    monkeypatch.setattr("firm.data.cache.ParquetCache", lambda *a, **k: fake_cache)

    from firm.config import get_settings

    bundle = runtime.load_macro(get_settings())
    assert bundle is not None
    assert "T10Y2Y" in bundle
    assert list(bundle["T10Y2Y"]["T10Y2Y"]) == [-0.3]


def test_load_macro_returns_none_when_cache_key_absent(monkeypatch):
    from unittest.mock import MagicMock

    from firm import runtime

    fake_cache = MagicMock()
    fake_cache.get.return_value = None
    monkeypatch.setattr("firm.data.cache.ParquetCache", lambda *a, **k: fake_cache)

    from firm.config import get_settings

    assert runtime.load_macro(get_settings()) is None
