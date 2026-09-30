"""Core + BTC-trend sleeves through the real Allocator (no engine, no broker).

Pins the reconciliation of the two sleeves' contracts: the BTC sleeve trades
only at its weekly review, on an on/off flip or a within-sleeve gap above its
pre-registered 0.10 band, never on the daily NAV-level drift trigger; its
orders are fractional GTC; bad data holds rather than sells.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from firm.allocation.allocator import Allocator
from firm.allocation.btc_trend import BtcTrendSleeve
from firm.allocation.sleeves import StaticSleeve

NAV = 100_000.0


def _btc_uptrend(end: str = "2026-01-07", n: int = 200, vol: float = 0.02) -> pd.Series:
    idx = pd.date_range(end=end, periods=n, freq="D")
    rng = np.random.default_rng(0)
    rets = rng.normal(0.004, vol, n)            # clearly trending up
    return pd.Series(40_000 * np.cumprod(1 + rets), index=idx)


def _alloc() -> tuple[Allocator, BtcTrendSleeve]:
    btc = BtcTrendSleeve(weight=0.08)
    core = StaticSleeve("core", 0.92, {"SPY": 0.6, "IEF": 0.4}, rebalance="monthly")
    return Allocator([core, btc], band_abs=0.02), btc


def _plan(alloc, asof, positions, btc_hist, last_btc, last_core=datetime(2026, 1, 2)):
    prices = {"SPY": 500.0, "IEF": 100.0, "BTC/USD": float(btc_hist.iloc[-1])}
    hist = {"SPY": pd.Series([500.0] * 300, index=pd.date_range(end=asof, periods=300)),
            "IEF": pd.Series([100.0] * 300, index=pd.date_range(end=asof, periods=300)),
            "BTC/USD": btc_hist}
    return alloc.plan(asof, NAV, positions, prices, hist,
                      {"core": last_core, "btc_trend": last_btc})


def _btc_orders(plan):
    return [o for o in plan.orders if o["symbol"] == "BTC/USD"]


def test_btc_review_buys_fractional_gtc_when_on_and_flat():
    alloc, btc = _alloc()
    hist = _btc_uptrend()
    target_in = btc.target_weights(datetime(2026, 1, 7), {"BTC/USD": hist})["BTC/USD"]
    assert target_in > 0
    # Wednesday 2026-01-07, last review before Sunday 01-04 -> due.
    plan = _plan(alloc, datetime(2026, 1, 7, 15), {"SPY": 55_200, "IEF": 36_800}, hist,
                 last_btc=datetime(2025, 12, 29))
    orders = _btc_orders(plan)
    assert len(orders) == 1 and orders[0]["side"] == "buy"
    assert orders[0]["fractional"] is True and orders[0]["time_in_force"] == "gtc"
    assert orders[0]["quantity"] != int(orders[0]["quantity"])
    assert orders[0]["notional"] == pytest.approx(0.08 * target_in * NAV, rel=0.01)


def test_btc_not_traded_midweek_even_if_drifted_past_nav_band():
    alloc, btc = _alloc()
    hist = _btc_uptrend()
    # Reviewed this week already (last_btc after Sunday 01-04); held 12% of NAV
    # vs a target <= 8%: the NAV-level 2% band would fire, drift_check=False must not.
    plan = _plan(alloc, datetime(2026, 1, 7, 15), {"SPY": 55_200, "IEF": 36_800, "BTC/USD": 12_000},
                 hist, last_btc=datetime(2026, 1, 5))
    assert _btc_orders(plan) == []
    assert "BTC/USD" not in plan.drift_symbols


def test_btc_review_holds_within_sleeve_band_trades_beyond_it():
    alloc, btc = _alloc()
    hist = _btc_uptrend()
    target_in = btc.target_weights(datetime(2026, 1, 7), {"BTC/USD": hist})["BTC/USD"]
    target_mv = 0.08 * target_in * NAV
    base = {"SPY": 55_200, "IEF": 36_800}
    near = _plan(alloc, datetime(2026, 1, 7, 15), {**base, "BTC/USD": target_mv + 0.05 * 0.08 * NAV},
                 hist, last_btc=datetime(2025, 12, 29))
    assert _btc_orders(near) == []                      # gap 0.05 of sleeve <= 0.10 band
    far = _plan(alloc, datetime(2026, 1, 7, 15), {**base, "BTC/USD": target_mv + 0.2 * 0.08 * NAV},
                hist, last_btc=datetime(2025, 12, 29))
    assert [o["side"] for o in _btc_orders(far)] == ["sell"]   # gap 0.20 > 0.10


def test_btc_trend_off_at_review_sells_everything():
    alloc, _ = _alloc()
    idx = pd.date_range(end="2026-01-07", periods=200, freq="D")
    falling = pd.Series(np.linspace(90_000, 40_000, 200), index=idx)
    plan = _plan(alloc, datetime(2026, 1, 7, 15), {"SPY": 55_200, "IEF": 36_800, "BTC/USD": 5_000},
                 falling, last_btc=datetime(2025, 12, 29))
    orders = _btc_orders(plan)
    assert len(orders) == 1 and orders[0]["side"] == "sell"


def test_btc_stale_data_holds_position_instead_of_selling():
    alloc, _ = _alloc()
    stale = _btc_uptrend(end="2025-12-20")
    plan = _plan(alloc, datetime(2026, 1, 7, 15), {"SPY": 55_200, "IEF": 36_800, "BTC/USD": 5_000},
                 stale, last_btc=datetime(2025, 12, 29))
    assert _btc_orders(plan) == []
    assert any("btc_trend" in e for e in plan.errors)


def test_core_still_uses_nav_drift_band():
    alloc, _ = _alloc()
    hist = _btc_uptrend()
    # Core not due (rebalanced this month); SPY 3 points over target -> drift trade.
    plan = _plan(alloc, datetime(2026, 1, 7, 15), {"SPY": 58_200, "IEF": 36_800}, hist,
                 last_btc=datetime(2026, 1, 5))
    assert "SPY" in plan.drift_symbols
    assert any(o["symbol"] == "SPY" and o["side"] == "sell" for o in plan.orders)
