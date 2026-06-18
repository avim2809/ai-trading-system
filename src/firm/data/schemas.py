"""Canonical column schemas for all data types.

Every provider must map its raw API response columns to these names before
returning DataFrames.  Downstream code can rely on these constants for
column access without magic strings.
"""

PRICE_COLS: list[str] = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adj_close",
]

FUNDAMENTAL_COLS: list[str] = [
    "date",
    "symbol",
    "market_cap",
    "pe_ratio",
    "pb_ratio",
    "roe",
    "debt_to_equity",
    "revenue",
    "net_income",
    "eps",
    "dividend_yield",
]

SENTIMENT_COLS: list[str] = [
    "date",
    "symbol",
    "sentiment_score",
    "news_volume",
    "source",
    "headline",
]

CORPORATE_ACTION_COLS: list[str] = [
    "date",
    "symbol",
    "action_type",
    "value",
    "description",
]

# Universe membership window columns (used by UniverseResolver for
# survivorship-aware point-in-time index membership).
COL_INDEX = "index"
COL_SYMBOL = "symbol"
COL_ADDED_DATE = "added_date"
COL_REMOVED_DATE = "removed_date"

UNIVERSE_COLUMNS: list[str] = [
    COL_INDEX,
    COL_SYMBOL,
    COL_ADDED_DATE,
    COL_REMOVED_DATE,
]
