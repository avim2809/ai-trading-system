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
    llm_config: dict = config.get("llm_config", {})

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
    """Load cached price data from the Parquet/CSV cache directory."""
    cache_dir = Path(settings.data.cache_dir)
    prices_path = cache_dir / "prices.parquet"
    if prices_path.exists():
        return pd.read_parquet(prices_path)
    csv_path = cache_dir / "prices.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(
        f"No cached price data found. Run fetch-data first. "
        f"Looked in: {prices_path}, {csv_path}"
    )


def run_backtest_from_config(
    config: dict,
    prices_df: pd.DataFrame,
    universe: list[str],
) -> tuple[BacktestEngine, "BacktestReport"]:  # noqa: F821
    """Setup + run + report in one call.  Used by both CLI and API."""
    pit_store = PointInTimeDataStore()
    pit_store.load(prices=prices_df)

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
