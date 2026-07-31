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

# Analyst rating-consensus snapshot (raw counts, not a derived score — let
# strategies decide how to weight strong_buy vs buy etc.). "date" is the
# consensus snapshot date (FMP's grades-historical is monthly; a live
# scraper's date is its fetch date). See firm.data.providers.fmp.FMPProvider
# .get_analyst_ratings and firm.strategies.investing_analyst_ratings.
ANALYST_RATINGS_COLS: list[str] = [
    "date",
    "symbol",
    "strong_buy",
    "buy",
    "hold",
    "sell",
    "strong_sell",
]

# Danelfin AI-driven composite scores (1-10 scale per component). See
# firm.data.providers.danelfin.DanelfinProvider.get_ai_scores and
# firm.strategies.danelfin_ai_score. "fundamental_score" can be null (missing
# for some historical dates per Danelfin's own data) — callers must handle
# missing values, not assume every column is always populated.
AI_SCORE_COLS: list[str] = [
    "date",
    "symbol",
    "ai_score",
    "fundamental_score",
    "technical_score",
    "sentiment_score",
    "low_risk_score",
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
