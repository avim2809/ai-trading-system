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


def build_orchestrator(config: dict):
    """Construct the full agent pipeline from *config*.

    Imports are deferred so the module loads even in minimal environments.
    Reads ``agent_modes`` and ``llm_config`` from *config* to optionally
    wrap quant agents with their LLM-enhanced variants.
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

    tech = TechnicalAnalyst(config=config)
    fund = FundamentalAnalyst(config=config)
    sent = SentimentAnalyst(config=config)
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

    orchestrator = build_orchestrator(config)

    bt_fields = {
        "start_date", "end_date", "initial_capital",
        "commission_pct", "slippage_pct", "rebalance_frequency",
    }
    bt_config = {k: v for k, v in config.items() if k in bt_fields}

    engine = BacktestEngine(bt_config)
    engine.setup(prices_df, pit_store, orchestrator, universe)
    engine.run()
    report = engine.generate_report()
    return engine, report
