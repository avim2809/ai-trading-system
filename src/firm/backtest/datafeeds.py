"""Custom backtrader data feeds backed by the ParquetCache / PIT store.

Converts cached pandas DataFrames (multi-symbol OHLCV) into individual
Backtrader ``PandasData`` feeds, one per symbol.
"""

from __future__ import annotations

import logging

import backtrader as bt
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

    return AdjustedPandasData(dataname=sym_df, **kwargs)


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
