"""Shared runtime helpers used by both the CLI and the API.

Refactored from ``firm.scripts.run_backtest`` so that the same wiring
logic is available without going through argparse.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from firm.backtest.engine import BacktestEngine
from firm.config import Settings
from firm.data.pit_store import PointInTimeDataStore

log = logging.getLogger(__name__)


def _maybe_wrap(agent, role_key: str, llm_cls_path: str, agent_modes: dict, config: dict, llm_config: dict):
    """Conditionally replace *agent* with its LLM-enhanced variant.

    Uses deferred import by dotted path so the system works without the
    ``llm`` extra installed.
    """
    mode = agent_modes.get(role_key, "quant")
    if mode not in ("llm_enhanced", "llm_only"):
        return agent
    try:
        module_path, cls_name = llm_cls_path.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        llm_cls = getattr(mod, cls_name)
        return llm_cls(config=config, llm_config=llm_config)
    except Exception:
        log.warning("Could not load LLM agent %s – falling back to quant", llm_cls_path)
        return agent


# Analyst category → strategy names (mirrors the wiring in tests/test_e2e.py).
# Any registered strategy not listed here (e.g. ml_prediction, gann, stat_arb,
# regime_hmm) falls through to the technical analyst.
_FUND_STRATEGIES = {"multi_factor"}
_SENT_STRATEGIES = {"sentiment", "event_driven"}


def _build_categorized_strategies(config: dict) -> dict[str, list]:
    """Instantiate strategies from *config* and bucket them by analyst.

    ``config["strategies"]`` is a list of registered strategy names; when it is
    absent or empty we default to **all** registered strategies (matching the
    intent of the API/UI strategy selector and the test wiring).  Optional
    per-strategy params come from ``config["strategy_params"]``.
    """
    from firm.strategies import get, list_strategies

    names = config.get("strategies") or list_strategies()
    params_map: dict = config.get("strategy_params", {}) or {}

    instances = []
    for name in names:
        try:
            instances.append(get(name)(params_map.get(name)))
        except Exception:
            log.warning("Could not instantiate strategy %r — skipping", name, exc_info=True)

    fund = [s for s in instances if s.name in _FUND_STRATEGIES]
    sent = [s for s in instances if s.name in _SENT_STRATEGIES]
    tech = [s for s in instances if s not in fund and s not in sent]
    return {"technical": tech, "fundamental": fund, "sentiment": sent}


def build_orchestrator(config: dict):
    """Construct the full agent pipeline from *config*.

    Imports are deferred so the module loads even in minimal environments.
    Reads ``agent_modes`` and ``llm_config`` from *config* to optionally
    wrap quant agents with their LLM-enhanced variants, and ``strategies``
    to wire alpha strategies into the analysts.
    """
    from firm.agents.orchestrator import Orchestrator

    from firm.agents.analysts.fundamental import FundamentalAnalyst
    from firm.agents.analysts.technical import TechnicalAnalyst
    from firm.agents.analysts.sentiment import SentimentAnalyst
    from firm.agents.research.bull import BullResearcher
    from firm.agents.research.bear import BearResearcher
    from firm.agents.research.debate import DebateAgent
    from firm.agents.trader import TraderAgent
    from firm.agents.risk import RiskAgent
    from firm.agents.execution import ExecutionAgent

    agent_modes: dict = config.get("agent_modes", {})
    llm_config: dict = dict(config.get("llm_config") or {})
    if not llm_config:
        try:
            from firm.llm.config import provider_config
            llm_config = provider_config()
        except Exception:
            llm_config = {}
    bt_policy = config.get("backtest_policy")
    if bt_policy in ("cache_only", "live_calls", "disabled"):
        llm_config.setdefault("enhancement", {})["policy"] = bt_policy

    strats = _build_categorized_strategies(config)

    tech = TechnicalAnalyst(strategies=strats["technical"], config=config)
    fund = FundamentalAnalyst(strategies=strats["fundamental"], config=config)
    sent = SentimentAnalyst(strategies=strats["sentiment"], config=config)
    bull = BullResearcher(config=config)
    bear = BearResearcher(config=config)
    debate = DebateAgent(config=config)
    trader = TraderAgent(config=config)
    risk = RiskAgent(config=config)

    _LLM_MAP = {
        "technical_analyst": ("firm.agents.llm.technical_analyst_llm.LLMTechnicalAnalyst", tech),
        "fundamental_analyst": ("firm.agents.llm.fundamental_analyst_llm.LLMFundamentalAnalyst", fund),
        "sentiment_analyst": ("firm.agents.llm.sentiment_analyst_llm.LLMSentimentAnalyst", sent),
        "bull_researcher": ("firm.agents.llm.bull_researcher_llm.LLMBullResearcher", bull),
        "bear_researcher": ("firm.agents.llm.bear_researcher_llm.LLMBearResearcher", bear),
        "debate": ("firm.agents.llm.debate_llm.LLMDebateAgent", debate),
        "trader": ("firm.agents.llm.trader_llm.LLMTraderAgent", trader),
        "risk": ("firm.agents.llm.risk_llm.LLMRiskAgent", risk),
    }

    for role_key, (cls_path, _) in _LLM_MAP.items():
        wrapped = _maybe_wrap(
            _LLM_MAP[role_key][1], role_key, cls_path, agent_modes, config, llm_config,
        )
        if role_key == "technical_analyst":
            tech = wrapped
        elif role_key == "fundamental_analyst":
            fund = wrapped
        elif role_key == "sentiment_analyst":
            sent = wrapped
        elif role_key == "bull_researcher":
            bull = wrapped
        elif role_key == "bear_researcher":
            bear = wrapped
        elif role_key == "debate":
            debate = wrapped
        elif role_key == "trader":
            trader = wrapped
        elif role_key == "risk":
            risk = wrapped

    # _maybe_wrap may have swapped in LLM-enhanced analysts constructed without
    # strategies; re-attach the categorized lists so the wrap path keeps them.
    tech.strategies = strats["technical"]
    fund.strategies = strats["fundamental"]
    sent.strategies = strats["sentiment"]

    analysts = [tech, fund, sent]

    return Orchestrator(
        analysts=analysts,
        bull=bull,
        bear=bear,
        debate=debate,
        trader=trader,
        risk=risk,
        execution=ExecutionAgent(config=config),
        config=config,
    )


def load_prices(settings: Settings) -> pd.DataFrame:
    """Load cached price data.

    Primary source: ``firm.data.cache.ParquetCache``'s ``"combined/prices"``
    key — the hashed-filename cache `fetch-data` actually writes to. Falls
    back to a plain ``prices.parquet``/``prices.csv`` file in the cache dir
    for data placed there manually (e.g. hand-copied from another source),
    which is what this function used to assume `fetch-data` produced —
    it never did, so `data_source="cache"` never had any real data to read
    after following the documented `fetch-data` workflow.
    """
    from firm.data.cache import ParquetCache

    cache = ParquetCache(settings.data.cache_dir)
    df = cache.get("combined/prices")
    if df is not None and not df.empty:
        return df

    cache_dir = Path(settings.data.cache_dir)
    prices_path = cache_dir / "prices.parquet"
    if prices_path.exists():
        return pd.read_parquet(prices_path)
    csv_path = cache_dir / "prices.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"No cached price data found. Run fetch-data first. "
        f"Looked in ParquetCache key 'combined/prices' and {prices_path}, {csv_path}"
    )


def load_fundamentals(settings: Settings) -> pd.DataFrame | None:
    """Load cached fundamental panels when ``fetch-data`` wrote them.

    Primary key: ``combined/fundamentals`` (ParquetCache). Falls back to
    ``fundamentals`` for older cache layouts. Returns ``None`` when absent so
    callers can degrade gracefully (mirrors optional FRED macro loading).
    """
    from firm.data.cache import ParquetCache

    cache = ParquetCache(settings.data.cache_dir)
    for key in ("combined/fundamentals", "fundamentals"):
        df = cache.get(key)
        if df is not None and not df.empty:
            return _expand_fundamental_symbol_aliases(df)
    return None


def load_sentiment(settings: Settings) -> pd.DataFrame | None:
    """Load cached news-sentiment panels when ``fetch-data`` wrote them.

    Primary key: ``combined/sentiment`` (ParquetCache) — the same key
    ``fetch-data`` and the live ``sentiment_cache`` module populate. Returns
    ``None`` when absent so callers degrade gracefully; without this, the
    ``sentiment`` strategy always emits an empty signal list in backtests
    (``PitView.sentiment()`` has nothing to aggregate), silently diverging
    from live where the same strategy is fully active.
    """
    from firm.data.cache import ParquetCache

    cache = ParquetCache(settings.data.cache_dir)
    df = cache.get("combined/sentiment")
    if df is not None and not df.empty:
        return df
    return None


def load_analyst_ratings(settings: Settings) -> pd.DataFrame | None:
    """Load cached analyst-ratings-consensus panels when ``fetch-data`` wrote them.

    Primary key: ``combined/analyst_ratings`` (ParquetCache) — same layout as
    ``combined/fundamentals``/``combined/sentiment``. Returns ``None`` when
    absent so callers degrade gracefully (the ``investing_analyst_ratings``
    strategy simply emits no signals, same as ``sentiment`` does today).
    """
    from firm.data.cache import ParquetCache

    cache = ParquetCache(settings.data.cache_dir)
    df = cache.get("combined/analyst_ratings")
    if df is not None and not df.empty:
        return df
    return None


def load_ai_scores(settings: Settings) -> pd.DataFrame | None:
    """Load cached Danelfin AI-score panels when ``fetch-data`` wrote them.

    Primary key: ``combined/ai_scores`` (ParquetCache) — same layout as
    ``combined/analyst_ratings``. Returns ``None`` when absent so callers
    degrade gracefully (the ``danelfin_ai_score`` strategy simply emits no
    signals).
    """
    from firm.data.cache import ParquetCache

    cache = ParquetCache(settings.data.cache_dir)
    df = cache.get("combined/ai_scores")
    if df is not None and not df.empty:
        return df
    return None


def load_market_percentile(settings: Settings) -> pd.DataFrame | None:
    """Load a cached cross-sectional ai_score population snapshot panel.

    Primary key: ``combined/market_percentile`` (ParquetCache). Unlike
    every other ``load_*`` helper here, this cache is NOT populated by the
    regular ``fetch-data`` pipeline — a full population snapshot costs
    ~66+ Danelfin API calls per date (see
    firm.data.danelfin_market_percentile's docstring), so populating it is
    a deliberate, bounded one-off step (see
    scripts/fetch_market_percentile_calibration_data.py) rather than
    something refreshed automatically. Returns ``None`` when absent so
    callers degrade gracefully (the ``danelfin_market_percentile``
    strategy simply emits no signals).
    """
    from firm.data.cache import ParquetCache

    cache = ParquetCache(settings.data.cache_dir)
    df = cache.get("combined/market_percentile")
    if df is not None and not df.empty:
        return df
    return None


def load_universe_membership(settings: Settings) -> pd.DataFrame | None:
    """Load historical index-membership windows for survivorship-aware backtests.

    Schema: :data:`firm.data.schemas.UNIVERSE_COLUMNS` (``index``, ``symbol``,
    ``added_date``, ``removed_date``). Primary source: ParquetCache key
    ``combined/universe_membership`` (mirrors ``fetch-data``'s prices/fundamentals
    layout). Falls back to a plain ``universe_membership.csv`` in the cache dir
    for hand-placed data. Returns ``None`` when absent so callers degrade to
    :meth:`firm.data.universe.UniverseResolver.from_static` — see
    ``build_universe_resolver``.
    """
    from firm.data.cache import ParquetCache

    cache = ParquetCache(settings.data.cache_dir)
    df = cache.get("combined/universe_membership")
    if df is not None and not df.empty:
        return df

    csv_path = Path(settings.data.cache_dir) / "universe_membership.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def build_universe_resolver(settings: Settings, fallback_symbols: list[str]):
    """Build a :class:`~firm.data.universe.UniverseResolver`, preferring real data.

    Thin wrapper combining ``load_universe_membership`` with
    :func:`firm.data.universe.build_resolver` so callers (CLI, API job runner,
    live engine) share one code path instead of duplicating the cache lookup +
    fallback logic.
    """
    from firm.data.universe import build_resolver

    try:
        membership = load_universe_membership(settings)
    except Exception:
        log.warning("Universe membership cache load failed — using static fallback", exc_info=True)
        membership = None
    return build_resolver(membership, fallback_symbols)


_FUNDAMENTALS_SYMBOL_ALIASES: dict[str, str] = {"GOOG": "GOOGL"}


def _expand_fundamental_symbol_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Duplicate rows for tickers that share fundamentals with a cached listing."""
    if df.empty or "symbol" not in df.columns:
        return df
    have = set(df["symbol"].astype(str).str.upper())
    extras: list[pd.DataFrame] = []
    for target, source in _FUNDAMENTALS_SYMBOL_ALIASES.items():
        if target in have or source not in have:
            continue
        copy = df[df["symbol"].astype(str).str.upper() == source].copy()
        copy["symbol"] = target
        extras.append(copy)
        log.debug("Expanded fundamentals alias %s -> %s (%d rows)", source, target, len(copy))
    if not extras:
        return df
    merged = pd.concat([df, *extras], ignore_index=True)
    if "date" in merged.columns:
        merged["date"] = pd.to_datetime(merged["date"])
        merged = merged.sort_values(["symbol", "date"]).drop_duplicates(
            subset=["symbol", "date"],
            keep="last",
        )
    return merged


def run_backtest_from_config(
    config: dict,
    prices_df: pd.DataFrame,
    universe: list[str],
) -> tuple[BacktestEngine, "BacktestReport"]:  # noqa: F821
    """Setup + run + report in one call.  Used by both CLI and API."""
    # Fundamentals/sentiment: same ParquetCache keys as ``fetch-data`` so
    # multi_factor / event_driven / sentiment backtests match live when
    # FMP/Massive/news data is cached. Loaded *before* pit_store.load() and
    # passed in a single call — a second load(fundamentals=...) call without
    # `prices` would raise (prices has no default and each call fully
    # replaces prior state), silently crashing any real dataset that had
    # cached fundamentals.
    fund_df = None
    try:
        from firm.config import get_settings

        fund_df = load_fundamentals(get_settings())
        if fund_df is not None:
            log.info(
                "Loaded fundamentals cache: %d rows, %d symbols",
                len(fund_df), fund_df["symbol"].nunique(),
            )
        else:
            log.debug(
                "No cached fundamentals in backtest; fundamental strategies "
                "use degraded logic (see multi_factor / event_driven)"
            )
    except Exception:
        log.warning("Fundamentals cache load failed — continuing price-only", exc_info=True)

    sentiment_df = None
    try:
        from firm.config import get_settings

        sentiment_df = load_sentiment(get_settings())
        if sentiment_df is not None:
            log.info(
                "Loaded sentiment cache: %d rows, %d symbols",
                len(sentiment_df), sentiment_df["symbol"].nunique(),
            )
        else:
            log.debug(
                "No cached sentiment in backtest; the sentiment strategy "
                "will emit no signals (see firm.strategies.sentiment)"
            )
    except Exception:
        log.warning("Sentiment cache load failed — sentiment strategy will be inactive", exc_info=True)

    estimates_df = None
    try:
        from firm.config import get_settings

        estimates_df = load_analyst_ratings(get_settings())
        if estimates_df is not None:
            log.info(
                "Loaded analyst-ratings cache: %d rows, %d symbols",
                len(estimates_df), estimates_df["symbol"].nunique(),
            )
        else:
            log.debug(
                "No cached analyst ratings in backtest; the "
                "investing_analyst_ratings strategy will emit no signals"
            )
    except Exception:
        log.warning("Analyst-ratings cache load failed — that strategy will be inactive", exc_info=True)

    ai_scores_df = None
    try:
        from firm.config import get_settings

        ai_scores_df = load_ai_scores(get_settings())
        if ai_scores_df is not None:
            log.info(
                "Loaded AI-scores cache: %d rows, %d symbols",
                len(ai_scores_df), ai_scores_df["symbol"].nunique(),
            )
        else:
            log.debug(
                "No cached AI scores in backtest; the danelfin_ai_score "
                "strategy will emit no signals"
            )
    except Exception:
        log.warning("AI-scores cache load failed — that strategy will be inactive", exc_info=True)

    market_percentile_df = None
    try:
        from firm.config import get_settings

        market_percentile_df = load_market_percentile(get_settings())
        if market_percentile_df is not None:
            log.info(
                "Loaded market-percentile cache: %d rows across %d date(s)",
                len(market_percentile_df), market_percentile_df["date"].nunique(),
            )
        else:
            log.debug(
                "No cached market-percentile pool in backtest; the "
                "danelfin_market_percentile strategy will emit no signals"
            )
    except Exception:
        log.warning("Market-percentile cache load failed — that strategy will be inactive", exc_info=True)

    pit_store = PointInTimeDataStore()
    pit_store.load(
        prices=prices_df, fundamentals=fund_df, sentiment=sentiment_df,
        estimates=estimates_df, ai_scores=ai_scores_df,
        market_percentile=market_percentile_df,
    )

    # Optionally load FRED macro data — gracefully skipped when key is absent.
    fred_api_key = config.get("fred_api_key", "")
    if not fred_api_key:
        try:
            from firm.config import get_settings
            fred_api_key = get_settings().fred_api_key
        except Exception:
            log.warning("fred_key_lookup_failed — macro data will be skipped", exc_info=True)
    if fred_api_key:
        try:
            from firm.data.providers.fred import fetch_macro_bundle
            start = config.get("start_date", "2010-01-01")
            end = config.get("end_date", "2030-12-31")
            macro_bundle = fetch_macro_bundle(fred_api_key, start, end)
            if macro_bundle:
                pit_store.load_macro(macro_bundle)
        except Exception:
            log.warning("FRED macro load failed — no macro features", exc_info=True)

    orchestrator = build_orchestrator(config)

    # Build a memory log when llm_config is present so backtest reflections
    # are stored — agents learn from backtest outcomes just like live ones.
    memory = None
    llm_config = config.get("llm_config", {})
    if llm_config:
        try:
            from firm.agents.memory import TradingMemoryLog
            memory = TradingMemoryLog(config)
        except Exception:
            log.debug("Memory log init failed — skipping", exc_info=True)

    bt_fields = {
        "start_date", "end_date", "initial_capital",
        "commission_pct", "slippage_pct", "rebalance_frequency",
    }
    bt_config = {k: v for k, v in config.items() if k in bt_fields}

    engine = BacktestEngine(bt_config)
    engine.setup(prices_df, pit_store, orchestrator, universe,
                 memory=memory, llm_config=llm_config or None)
    engine.run()
    report = engine.generate_report()
    return engine, report
