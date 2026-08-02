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

# Danelfin's /v3/* latest-snapshot signals (trading-parameters +
# price-forecast + performance track record combined into one row per
# symbol per fetch). Unlike AI_SCORE_COLS/ANALYST_RATINGS_COLS, these have
# NO historical time series at all (Danelfin's own docs: "always returns
# latest snapshot, no historical dates") — "date" here is always the fetch
# date, and there is no cache-backed history to backtest against, ever.
# A strategy reading this capability will always see an empty frame in
# backtests (no historical data source exists to populate one) and only
# ever sees real data in live cycles. See
# firm.data.providers.danelfin.DanelfinProvider.get_live_signals and
# firm.strategies.danelfin_live_signals.
LIVE_SIGNAL_COLS: list[str] = [
    "date",
    "symbol",
    "tp_signal",             # trading-parameters buy/hold/sell
    "tp_entry_price",
    "tp_stop_loss_pct",
    "tp_take_profit_pct",
    "pf_median_return_3m",   # price-forecast median 3-month return (decimal)
    "pf_q05_return_3m",
    "pf_q95_return_3m",
    "perf_win_rate_3m",      # historical win-rate for this ticker's buy signal
    "perf_alpha_win_rate_3m",
    # Added 2026-08-02 (Danelfin deepening plan Phase 6) — richer
    # confidence weighting in danelfin_live_signals, blending win-rate
    # across horizons instead of 3m alone, plus a genuine alpha-vs-
    # benchmark adjustment (win_rate alone doesn't distinguish "beats a
    # falling market" from "beats a rising one").
    "perf_win_rate_1m",
    "perf_win_rate_6m",
    "perf_win_rate_1y",
    "perf_avg_alpha_3m",
]

# Danelfin's /v3/beststocks — their own curated Top-25 "Best Stocks" list.
# Same no-history caveat as LIVE_SIGNAL_COLS: latest snapshot only, "date" is
# always the fetch date. See
# firm.data.providers.danelfin.DanelfinProvider.get_best_stocks and
# firm.strategies.danelfin_best_stocks_signal.
BEST_STOCKS_COLS: list[str] = [
    "date",
    "symbol",
    "rank",              # 1 (highest conviction) .. 25
    "ai_score",
    "ai_score_change",
    "fundamental_score",
    "technical_score",
    "sentiment_score",
    "low_risk_score",
    "perf_ytd",
    "sector",
    "country",
]

# A broad cross-sectional POPULATION snapshot (many symbols across many
# sectors on a single date), NOT limited to this project's own fixed
# universe — deliberately a different shape from AI_SCORE_COLS (which is
# one row per universe symbol per date, a time series). This is what
# firm.strategies.danelfin_market_percentile ranks each universe symbol
# against, to answer "is this actually a top ai_score relative to the
# whole market" rather than "top score relative to an arbitrarily chosen
# ~25-name universe." Backed by
# firm.data.providers.danelfin.DanelfinProvider.get_historical_sector_scores
# (genuinely historical bulk /ranking mode) called once per sector and
# concatenated — see firm.data.danelfin_market_percentile for the fetch
# helper. Real cost note: a full population snapshot for one date costs
# ~11 sectors x ~6 low_risk values (+pagination) = 66+ Danelfin API calls —
# NOT cheap to fetch on every rebalance date, unlike AI_SCORE_COLS which is
# served from a pre-populated cache.
MARKET_PERCENTILE_COLS: list[str] = [
    "date",
    "symbol",
    "sector",
    "ai_score",
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
