"""Synthetic OHLCV data generator via geometric Brownian motion.

Provides realistic-looking price series for UI demos and testing without
requiring paid API keys or cached market data.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "V", "JNJ",
]


def make_synthetic_prices(
    symbols: list[str] | None = None,
    n_days: int = 504,
    end_date: str = "2023-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data via geometric Brownian motion.

    Returns a DataFrame with columns:
    ``date, symbol, open, high, low, close, volume, adj_close``
    """
    if symbols is None:
        symbols = list(DEFAULT_SYMBOLS)

    rng = np.random.RandomState(seed)
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    dates = pd.bdate_range(end=end_dt, periods=n_days)

    rows: list[dict] = []
    for sym_idx, sym in enumerate(symbols):
        price = 100.0 + sym_idx * 20
        mu = 0.0005 + rng.uniform(-0.0003, 0.0003)
        sigma = 0.015 + rng.uniform(0, 0.01)
        for d in dates:
            ret = rng.normal(mu, sigma)
            price *= 1 + ret
            high = price * (1 + abs(rng.normal(0, 0.005)))
            low = price * (1 - abs(rng.normal(0, 0.005)))
            rows.append({
                "date": d,
                "symbol": sym,
                "open": round(price * (1 + rng.normal(0, 0.002)), 4),
                "high": round(high, 4),
                "low": round(low, 4),
                "close": round(price, 4),
                "volume": int(rng.uniform(1e6, 1e7)),
                "adj_close": round(price, 4),
            })

    return pd.DataFrame(rows)
