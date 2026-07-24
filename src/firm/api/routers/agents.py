"""Agent pipeline single-step endpoint for the inspector UI."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter

from firm.api.schemas import StepRequest
from firm.api.serializers import serialize_blackboard

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/agents/step")
def agent_step(req: StepRequest):
    """Run one orchestrator step and return the serialized blackboard."""
    from firm.data.synthetic import make_synthetic_prices
    from firm.data.pit_store import PointInTimeDataStore
    from firm.backtest.firm_strategy import PitViewAdapter
    from firm.portfolio.state import PortfolioState
    from firm.runtime import build_orchestrator

    if req.data_source == "synthetic":
        prices_df = make_synthetic_prices(
            symbols=req.symbols,
            end_date=req.asof_date,
            seed=req.seed,
        )
    else:
        from firm.runtime import load_prices
        from firm.config import get_settings
        prices_df = load_prices(get_settings())

    pit_store = PointInTimeDataStore()
    pit_store.load(prices=prices_df)

    asof = datetime.fromisoformat(req.asof_date)
    pit_view = PitViewAdapter(pit_store, asof, req.symbols)

    prices_map: dict[str, float] = {}
    for sym in req.symbols:
        pdf = pit_store.get_prices([sym], asof, lookback_days=1)
        if not pdf.empty:
            prices_map[sym] = float(pdf.iloc[-1]["close"])

    portfolio = PortfolioState(initial_capital=10_000_000)

    config: dict = {
        "strategies": req.strategies,
        "strategy_params": req.strategy_params,
    }
    try:
        from firm.llm.config import load_llm_config, provider_config
        llm_yaml = load_llm_config()
        config["agent_modes"] = llm_yaml.get("agent_modes", {})
        config["llm_config"] = provider_config()
        config["backtest_policy"] = llm_yaml.get("backtest_policy", "cache_only")
    except Exception:
        log.warning("Could not load LLM config for agent step; using quant agents", exc_info=True)
    orchestrator = build_orchestrator(config)

    context = {
        "pit_view": pit_view,
        "portfolio": portfolio,
        "prices": prices_map,
    }

    _orders, blackboard = orchestrator.step(context)
    return serialize_blackboard(blackboard)
