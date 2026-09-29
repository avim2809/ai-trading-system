"""Custom backtrader data feeds backed by the ParquetCache / PIT store.

Converts cached pandas DataFrames (multi-symbol OHLCV) into individual
Backtrader ``PandasData`` feeds, one per symbol.
"""

from __future__ import annotations

import logging

import backtrader as bt
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


class AdjustedPandasData(bt.feeds.PandasData):
    """Extended PandasData with an additional adjusted-close line."""

    lines = ("adj_close",)
    params = (
        ("datetime", None),  # use the DataFrame index
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
        ("adj_close", "adj_close"),
    )


def dataframe_to_feed(
    df: pd.DataFrame,
    symbol: str,
    **kwargs,
) -> AdjustedPandasData:
    """Convert a single-symbol slice of a multi-symbol DataFrame to a feed.

    The input DataFrame must have columns ``symbol``, ``date``, and at
    least ``open``, ``high``, ``low``, ``close``, ``volume``.  If
    ``adj_close`` is missing the ``close`` column is used as a fallback.
    """
    sym_df = df.loc[df["symbol"] == symbol].copy()
    if sym_df.empty:
        raise ValueError(f"No data found for symbol {symbol!r}")

    sym_df["date"] = pd.to_datetime(sym_df["date"])
    sym_df = sym_df.sort_values("date").set_index("date")

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(sym_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for {symbol}: {missing}")

    if "adj_close" not in sym_df.columns:
        sym_df["adj_close"] = sym_df["close"]

    sym_df = sym_df[["open", "high", "low", "close", "volume", "adj_close"]]
    sym_df = sym_df.dropna(subset=["open", "high", "low", "close"])
    sym_df = _total_return_adjust(sym_df, symbol)

    return AdjustedPandasData(dataname=sym_df, **kwargs)


def _total_return_adjust(sym_df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Rescale OHLC (and volume inversely) by ``adj_close / close``.

    The broker fills and marks positions at the feed's ``close``. The cache's
    ``close`` is the raw, split-unadjusted print, so every stock split showed
    up as a fake one-day P&L jump on any position held through it: a long
    NVDA position "lost" 90% on 2024-06-10 (10:1 split), a short AMZN one
    "gained" 95% on 2022-06-06. Found 2026-09-29: the live-config backtest's
    worst day, -6.96% on a 0.4%-daily-vol book, was the NVDA split, and 8
    such days fall inside 2020-2026 (see docs/edge_search_verdict_2026_09.md).
    Raw closes also ignored dividends entirely. Scaling each bar by the
    cache's split+dividend adjustment factor makes day-over-day returns equal
    total returns (what a real holder earns). Order sizing, costs (all
    percentage-based) and NAV stay consistent because every backtest
    consumer of prices reads this same feed. Volume is divided by the same
    factor so dollar volume is unchanged.

    Rows with a missing or non-positive ratio take the nearest valid factor;
    a symbol with no valid ratio at all is left raw.
    """
    ratio = sym_df["adj_close"] / sym_df["close"]
    ratio = ratio.where(np.isfinite(ratio) & (ratio > 0))
    if ratio.notna().sum() == 0:
        log.warning("%s: no valid adj_close/close ratio; feed left unadjusted", symbol)
        return sym_df
    ratio = ratio.ffill().bfill()
    out = sym_df.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = sym_df[col] * ratio
    out["volume"] = sym_df["volume"] / ratio
    out["adj_close"] = out["close"]
    jumps = (ratio.pct_change().abs() > 0.2).sum()
    if jumps:
        log.debug("%s: total-return adjusted feed (%d split-sized factor changes)", symbol, int(jumps))
    return out


def total_return_adjust_panel(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Multi-symbol version of :func:`_total_return_adjust` for the backtest's
    whole price panel (what ``PitView.prices()`` serves strategies).

    Live providers already serve adjusted OHLC (Alpaca ``adjustment=ALL``, IBKR
    TRADES bars), while the cache's OHLC is raw, so without this a strategy
    reading ``high``/``low`` next to ``adj_close`` (``volatility_breakout``'s
    true range) saw a different, split-contaminated series in backtests than
    live. After adjustment ``close == adj_close`` and every OHLC ratio is
    continuous across splits.
    """
    if prices_df is None or prices_df.empty or "adj_close" not in prices_df.columns:
        return prices_df
    parts = []
    for sym, g in prices_df.groupby("symbol", sort=False):
        g = g.sort_values("date")
        adjusted = _total_return_adjust(g.set_index("date"), str(sym))
        parts.append(adjusted.reset_index())
    out = pd.concat(parts, ignore_index=True)
    return out[prices_df.columns]


def load_feeds(
    prices_df: pd.DataFrame,
    symbols: list[str],
) -> dict[str, AdjustedPandasData]:
    """Create one feed per symbol from a multi-symbol DataFrame.

    Returns a dict mapping ``symbol -> AdjustedPandasData``.  Symbols
    with no data in *prices_df* are silently skipped with a warning.
    """
    feeds: dict[str, AdjustedPandasData] = {}
    for sym in symbols:
        try:
            feeds[sym] = dataframe_to_feed(prices_df, sym)
        except ValueError:
            log.warning("Skipping symbol %s: no data in prices DataFrame", sym)
    return feeds
