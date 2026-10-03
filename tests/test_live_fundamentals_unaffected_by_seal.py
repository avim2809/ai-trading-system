"""The seal must be invisible to live processes (guards default to None).

The live fundamentals path (``data_feed`` -> ``runtime.load_fundamentals`` ->
``PointInTimeDataStore.get_fundamentals``) swallows errors at debug level, so a guard
that raised there would silently drop live fundamentals. Only ``firm.research.seal``
installs the guard, and live modules never import it.
"""

from __future__ import annotations

import datetime as dt
import types

import pandas as pd

from firm import runtime
from firm.data import pit_store
from firm.data.cache import ParquetCache
from firm.data.pit_store import PointInTimeDataStore


def test_live_fundamentals_path_works_past_seal_date(tmp_path, monkeypatch):
    monkeypatch.setenv("FIRM_RESEARCH", "1")  # no env var may change behaviour
    monkeypatch.setenv("HOLDOUT_UNSEAL_TOKEN", "")
    assert pit_store._ACCESS_GUARD is None and runtime._ACCESS_GUARD is None

    post_seal = pd.Timestamp("2026-10-05")
    fundamentals = pd.DataFrame({"date": [post_seal], "symbol": ["AAPL"], "pe": [28.0]})
    prices = pd.DataFrame({"date": [post_seal], "symbol": ["AAPL"], "close": [200.0]})
    cache_dir = tmp_path / "cache"
    cache = ParquetCache(cache_dir)
    cache.put("combined/fundamentals", fundamentals)
    cache.put("combined/prices", prices)

    settings = types.SimpleNamespace(data=types.SimpleNamespace(cache_dir=str(cache_dir)))
    loaded = runtime.load_fundamentals(settings)
    assert loaded is not None and len(loaded) == 1
    assert len(runtime.load_prices(settings)) == 1

    store = PointInTimeDataStore()
    store.load(prices=runtime.load_prices(settings), fundamentals=loaded)
    asof = dt.datetime(2026, 10, 6, 9, 30)
    assert len(store.get_fundamentals(["AAPL"], asof)) == 1
    assert len(store.get_prices(["AAPL"], asof)) == 1
    assert store.get_universe(asof) == ["AAPL"]
