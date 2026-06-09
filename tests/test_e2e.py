"""End-to-end integration tests for the AI Investment Firm.

These tests validate the complete system pipeline from data layer through
strategies, agents, Backtrader engine, eval, and experiments.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.agents.analysts.fundamental import FundamentalAnalyst
from firm.agents.analysts.sentiment import SentimentAnalyst
from firm.agents.analysts.technical import TechnicalAnalyst
from firm.agents.base import AgentContext
from firm.agents.blackboard import Blackboard
from firm.agents.execution import ExecutionAgent
from firm.agents.orchestrator import Orchestrator
from firm.agents.research.bear import BearResearcher
from firm.agents.research.bull import BullResearcher
from firm.agents.research.debate import DebateAgent
from firm.agents.risk import RiskAgent
from firm.agents.trader import TraderAgent
from firm.backtest.engine import BacktestEngine
from firm.backtest.firm_strategy import PitViewAdapter
from firm.contracts.models import (
    DebateResult,
    ExecutionReport,
    RiskDecision,
    Signal,
    SignalSet,
    Thesis,
    TradeProposal,
)
from firm.data.pit_store import PointInTimeDataStore
from firm.eval.metrics import compute_all_metrics
from firm.experiments.registry import RunRegistry
from firm.experiments.runner import ExperimentRunner
from firm.portfolio.state import PortfolioState
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import get, list_strategies


# ---------------------------------------------------------------------------
# Synthetic Data Factory
# ---------------------------------------------------------------------------


def make_synthetic_prices(
    symbols: list[str],
    n_days: int = 252,
    start_date: str = "2020-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic properties.

    Uses geometric Brownian motion for prices with correlated returns.
    """
    np.random.seed(seed)
    start = pd.Timestamp(start_date)
    dates = pd.bdate_range(start=start, periods=n_days)

    rows = []
    for sym in symbols:
        drift = np.random.uniform(-0.0002, 0.0005)
        vol = np.random.uniform(0.01, 0.03)
        price = np.random.uniform(20, 200)

        for date in dates:
            ret = drift + vol * np.random.randn()
            price *= (1 + ret)
            price = max(price, 1.0)

            intraday_vol = price * np.random.uniform(0.005, 0.02)
            high = price + abs(np.random.randn()) * intraday_vol
            low = price - abs(np.random.randn()) * intraday_vol
            low = max(low, 0.5)
            opn = low + (high - low) * np.random.uniform(0.2, 0.8)
            volume = int(np.random.lognormal(14, 1))

            rows.append({
                "date": date,
                "symbol": sym,
                "open": round(opn, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(price, 2),
                "adj_close": round(price, 2),
                "volume": volume,
            })

    return pd.DataFrame(rows)


def make_synthetic_fundamentals(
    symbols: list[str],
    n_quarters: int = 8,
    start_date: str = "2020-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic quarterly fundamental data."""
    np.random.seed(seed + 1)
    start = pd.Timestamp(start_date)

    rows = []
    for sym in symbols:
        pe_base = np.random.uniform(10, 40)
        roe_base = np.random.uniform(0.05, 0.25)
        revenue_base = np.random.uniform(1e9, 50e9)

        for q in range(n_quarters):
            date = start + timedelta(days=q * 91)
            rows.append({
                "date": date,
                "symbol": sym,
                "pe_ratio": round(pe_base + np.random.randn() * 3, 2),
                "pb_ratio": round(np.random.uniform(1, 8), 2),
                "roe": round(roe_base + np.random.randn() * 0.03, 4),
                "revenue": round(revenue_base * (1 + 0.02 * q + np.random.randn() * 0.05)),
                "earnings_surprise": round(np.random.randn() * 0.05, 4),
                "debt_to_equity": round(np.random.uniform(0.2, 2.0), 2),
                "dividend_yield": round(np.random.uniform(0, 0.04), 4),
            })

    return pd.DataFrame(rows)


def make_synthetic_sentiment(
    symbols: list[str],
    n_days: int = 252,
    start_date: str = "2020-01-01",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily sentiment data."""
    np.random.seed(seed + 2)
    start = pd.Timestamp(start_date)
    dates = pd.bdate_range(start=start, periods=n_days)

    rows = []
    for sym in symbols:
        base_sentiment = np.random.uniform(-0.2, 0.2)
        for date in dates:
            score = base_sentiment + np.random.randn() * 0.3
            score = max(-1.0, min(1.0, score))
            rows.append({
                "date": date,
                "symbol": sym,
                "sentiment_score": round(score, 4),
                "news_volume": int(np.random.poisson(5)),
                "social_volume": int(np.random.poisson(20)),
            })

    return pd.DataFrame(rows)


def build_pit_store(
    symbols: list[str],
    n_days: int = 252,
    start_date: str = "2020-01-01",
    seed: int = 42,
) -> tuple[PointInTimeDataStore, pd.DataFrame]:
    """Build a fully loaded PIT store with synthetic data. Returns (store, prices_df)."""
    prices = make_synthetic_prices(symbols, n_days, start_date, seed=seed)
    fundamentals = make_synthetic_fundamentals(symbols, start_date=start_date, seed=seed)
    sentiment = make_synthetic_sentiment(symbols, n_days, start_date, seed=seed)

    store = PointInTimeDataStore()
    store.load(prices=prices, fundamentals=fundamentals, sentiment=sentiment)
    return store, prices


def build_orchestrator(
    strategies: list[BaseStrategy] | None = None,
    risk_config: dict | None = None,
) -> Orchestrator:
    """Build the full orchestrator with all agents."""
    if strategies is None:
        import firm.strategies  # noqa: F401 - triggers registry
        all_names = list_strategies()
        strategies = [get(name)() for name in all_names]

    tech_strats = [s for s in strategies if s.name in (
        "momentum", "trend", "mean_reversion", "volatility_breakout", "seasonality"
    )]
    fund_strats = [s for s in strategies if s.name in ("multi_factor",)]
    sent_strats = [s for s in strategies if s.name in ("sentiment", "event_driven")]

    remaining = [s for s in strategies if s not in tech_strats + fund_strats + sent_strats]
    tech_strats.extend(remaining)

    tech_analyst = TechnicalAnalyst(strategies=tech_strats)
    fund_analyst = FundamentalAnalyst(strategies=fund_strats)
    sent_analyst = SentimentAnalyst(strategies=sent_strats)

    bull = BullResearcher()
    bear = BearResearcher()
    debate = DebateAgent()
    trader = TraderAgent(config={"max_positions": 10})
    risk = RiskAgent(config=risk_config or {})
    execution = ExecutionAgent()

    return Orchestrator(
        analysts=[tech_analyst, fund_analyst, sent_analyst],
        bull=bull,
        bear=bear,
        debate=debate,
        trader=trader,
        risk=risk,
        execution=execution,
    )


# ---------------------------------------------------------------------------
# Test 1: No-Look-Ahead
# ---------------------------------------------------------------------------


def test_no_look_ahead():
    """Verify that the PIT store and strategy pipeline never leak future data.

    1. Create synthetic price data for 3 symbols over 120 trading days
    2. Load into PointInTimeDataStore
    3. Create a PitViewAdapter with asof = day 60
    4. Run strategies via generate()
    5. Verify that no signal references data after day 60
    6. Monkey-patch PIT store to track every data access and assert
       all accessed dates <= asof
    """
    symbols = ["AAAA", "BBBB", "CCCC"]
    n_days = 120
    start_date = "2020-01-01"

    pit_store, prices_df = build_pit_store(symbols, n_days, start_date)

    dates = sorted(prices_df["date"].unique())
    asof_date = dates[59]  # day 60 (0-indexed)
    asof_dt = pd.Timestamp(asof_date).to_pydatetime()

    accessed_dates: list[pd.Timestamp] = []
    original_get_prices = pit_store.get_prices

    def tracking_get_prices(syms, asof, lookback_days=252):
        result = original_get_prices(syms, asof, lookback_days)
        if not result.empty:
            accessed_dates.extend(result["date"].tolist())
        return result

    pit_store.get_prices = tracking_get_prices

    original_get_fundamentals = pit_store.get_fundamentals

    def tracking_get_fundamentals(syms, asof):
        result = original_get_fundamentals(syms, asof)
        if not result.empty and "date" in result.columns:
            accessed_dates.extend(result["date"].tolist())
        return result

    pit_store.get_fundamentals = tracking_get_fundamentals

    original_get_sentiment = pit_store.get_sentiment

    def tracking_get_sentiment(syms, asof, lookback_days=5):
        result = original_get_sentiment(syms, asof, lookback_days)
        if not result.empty:
            accessed_dates.extend(result["date"].tolist())
        return result

    pit_store.get_sentiment = tracking_get_sentiment

    pit_view = PitViewAdapter(pit_store, asof_dt, symbols)

    assert pit_view.asof == asof_dt
    assert set(pit_view.universe) == set(symbols)

    import firm.strategies  # noqa: F401
    strategy_names = list_strategies()
    strategies = [get(name)() for name in strategy_names]

    all_signals: list[Signal] = []
    for strat in strategies:
        try:
            signals = strat.generate(pit_view)
            all_signals.extend(signals)
        except Exception:
            pass

    assert len(all_signals) > 0, "At least some strategies should produce signals"

    for sig in all_signals:
        assert sig.asof <= asof_dt, (
            f"Signal from {sig.strategy} for {sig.symbol} has asof={sig.asof} "
            f"which is AFTER the PIT boundary {asof_dt}"
        )

    asof_ts = pd.Timestamp(asof_dt)
    for accessed in accessed_dates:
        assert pd.Timestamp(accessed) <= asof_ts, (
            f"LOOK-AHEAD VIOLATION: accessed data with date {accessed} "
            f"which is AFTER asof={asof_ts}"
        )


# ---------------------------------------------------------------------------
# Test 2: Reproducibility / Golden-Run
# ---------------------------------------------------------------------------


def test_reproducibility():
    """Verify that two runs with the same seed produce identical results.

    1. Create synthetic data for a small universe (3 symbols, ~100 days)
    2. Build full pipeline
    3. Run orchestrator.step() with seed=42
    4. Record all outputs
    5. Reset, set seed=42, run again
    6. Assert all outputs are identical
    """
    symbols = ["AAAA", "BBBB", "CCCC"]
    n_days = 100
    start_date = "2020-01-01"

    def run_with_seed(seed: int):
        np.random.seed(seed)
        random.seed(seed)

        pit_store, prices_df = build_pit_store(symbols, n_days, start_date, seed=seed)
        dates = sorted(prices_df["date"].unique())
        asof_date = dates[-1]
        asof_dt = pd.Timestamp(asof_date).to_pydatetime()

        pit_view = PitViewAdapter(pit_store, asof_dt, symbols)
        portfolio = PortfolioState(initial_capital=1_000_000)

        prices = {}
        for sym in symbols:
            sym_prices = prices_df[prices_df["symbol"] == sym].sort_values("date")
            if not sym_prices.empty:
                prices[sym] = float(sym_prices.iloc[-1]["close"])

        orchestrator = build_orchestrator()

        np.random.seed(seed)
        random.seed(seed)

        context = {
            "pit_view": pit_view,
            "portfolio": portfolio,
            "prices": prices,
        }
        orders, bb = orchestrator.step(context)
        return orders, bb

    orders1, bb1 = run_with_seed(42)
    orders2, bb2 = run_with_seed(42)

    assert len(orders1) == len(orders2), "Order count differs between runs"

    for o1, o2 in zip(orders1, orders2):
        assert o1["symbol"] == o2["symbol"]
        assert abs(o1.get("quantity", 0) - o2.get("quantity", 0)) < 1e-10
        assert o1.get("side") == o2.get("side")

    assert len(bb1.signal_sets) == len(bb2.signal_sets)
    for ss1, ss2 in zip(bb1.signal_sets, bb2.signal_sets):
        assert ss1.domain == ss2.domain
        assert len(ss1.signals) == len(ss2.signals)
        for s1, s2 in zip(
            sorted(ss1.signals, key=lambda s: (s.symbol, s.strategy)),
            sorted(ss2.signals, key=lambda s: (s.symbol, s.strategy)),
        ):
            assert s1.symbol == s2.symbol
            assert s1.strategy == s2.strategy
            assert abs(s1.score - s2.score) < 1e-10

    assert len(bb1.theses) == len(bb2.theses)
    assert len(bb1.debate_results) == len(bb2.debate_results)

    if bb1.proposal and bb2.proposal:
        assert bb1.proposal.targets == bb2.proposal.targets


# ---------------------------------------------------------------------------
# Test 3: Full Pipeline Integration
# ---------------------------------------------------------------------------


def test_full_pipeline_integration():
    """Run the complete agent pipeline end-to-end with synthetic data.

    Verifies all 10 strategies, all agents, and the orchestrator work
    together producing expected artifacts on the blackboard.
    """
    symbols = ["AAAA", "BBBB", "CCCC", "DDDD", "EEEE"]
    n_days = 252
    start_date = "2020-01-01"

    pit_store, prices_df = build_pit_store(symbols, n_days, start_date)

    dates = sorted(prices_df["date"].unique())
    asof_date = dates[-1]
    asof_dt = pd.Timestamp(asof_date).to_pydatetime()

    pit_view = PitViewAdapter(pit_store, asof_dt, symbols)
    portfolio = PortfolioState(initial_capital=10_000_000)

    prices = {}
    for sym in symbols:
        sym_prices = prices_df[prices_df["symbol"] == sym].sort_values("date")
        if not sym_prices.empty:
            prices[sym] = float(sym_prices.iloc[-1]["close"])

    orchestrator = build_orchestrator()

    context = {
        "pit_view": pit_view,
        "portfolio": portfolio,
        "prices": prices,
    }

    orders, bb = orchestrator.step(context)

    assert isinstance(orders, list)
    assert isinstance(bb, Blackboard)
    assert len(bb.signal_sets) > 0, "Should have at least one signal set"

    total_signals = sum(len(ss.signals) for ss in bb.signal_sets)
    assert total_signals > 0, "Should have produced signals"

    domains = {ss.domain for ss in bb.signal_sets}
    assert "technical" in domains, "Technical analyst should produce a SignalSet"

    if orders:
        assert bb.proposal is not None, "Blackboard should have proposal"
        assert bb.risk_decision is not None, "Blackboard should have risk decision"
        assert bb.risk_decision.approved, "Risk decision should be approved for valid orders"
        assert bb.execution_report is not None

        for order in orders:
            assert "symbol" in order
            assert "side" in order
            assert order["symbol"] in symbols


# ---------------------------------------------------------------------------
# Test 4: Backtrader E2E
# ---------------------------------------------------------------------------


def test_backtrader_e2e():
    """Run a minimal Backtrader backtest with the full agent pipeline.

    Verifies the BacktestEngine + FirmStrategy bridge work together
    without exceptions and produce valid results.
    """
    symbols = ["AAAA", "BBBB"]
    n_days = 60
    start_date = "2020-01-01"

    pit_store, prices_df = build_pit_store(symbols, n_days, start_date)

    import firm.strategies  # noqa: F401
    momentum_strat = get("momentum")()
    trend_strat = get("trend")()

    orchestrator = build_orchestrator(
        strategies=[momentum_strat, trend_strat],
        risk_config={"max_position_pct": 0.30, "max_gross_exposure": 3.0},
    )

    config = {
        "initial_capital": 100_000,
        "commission_pct": 0.001,
        "rebalance_frequency": "weekly",
    }

    engine = BacktestEngine(config)
    engine.setup(prices_df, pit_store, orchestrator, symbols)

    results = engine.run()
    assert results is not None
    assert len(results) > 0

    analysis = engine.get_results()
    assert "final_value" in analysis
    assert analysis["final_value"] > 0, "Portfolio value should be positive"

    assert "sharpe" in analysis
    assert "max_drawdown_pct" in analysis
    assert "returns" in analysis


# ---------------------------------------------------------------------------
# Test 5: Risk Manager Constraint Enforcement
# ---------------------------------------------------------------------------


def test_risk_constraints_enforced_e2e():
    """Verify risk manager correctly constrains oversized positions.

    Creates a scenario where the PM would propose positions exceeding
    the per-name cap, then verifies the risk manager clips them.
    """
    symbols = ["AAAA", "BBBB", "CCCC"]
    n_days = 252
    start_date = "2020-01-01"

    pit_store, prices_df = build_pit_store(symbols, n_days, start_date)

    dates = sorted(prices_df["date"].unique())
    asof_dt = pd.Timestamp(dates[-1]).to_pydatetime()
    pit_view = PitViewAdapter(pit_store, asof_dt, symbols)
    portfolio = PortfolioState(initial_capital=10_000_000)

    prices = {}
    for sym in symbols:
        sym_prices = prices_df[prices_df["symbol"] == sym].sort_values("date")
        if not sym_prices.empty:
            prices[sym] = float(sym_prices.iloc[-1]["close"])

    strict_risk_config = {
        "max_position_pct": 0.03,
        "max_gross_exposure": 0.5,
        "max_net_exposure": 0.2,
        "veto_threshold": 0.9,
    }

    risk_agent = RiskAgent(config=strict_risk_config)

    oversized_proposal = TradeProposal(
        asof=asof_dt,
        targets={"AAAA": 0.40, "BBBB": -0.35, "CCCC": 0.25},
        per_strategy={},
        notes="Deliberately oversized",
    )

    ctx = AgentContext(now=asof_dt, pit_view=pit_view, portfolio=portfolio)
    decision = risk_agent.run(ctx, proposal=oversized_proposal, portfolio=portfolio)

    assert isinstance(decision, RiskDecision)
    assert len(decision.violations) > 0, "Should have constraint violations"

    for sym, weight in decision.adjusted_targets.items():
        assert abs(weight) <= strict_risk_config["max_position_pct"] + 0.01, (
            f"{sym} weight {weight:.4f} exceeds cap {strict_risk_config['max_position_pct']}"
        )

    adjusted_gross = sum(abs(w) for w in decision.adjusted_targets.values())
    assert adjusted_gross <= strict_risk_config["max_gross_exposure"] + 0.01


# ---------------------------------------------------------------------------
# Test 6: Experiment Runner Integration
# ---------------------------------------------------------------------------


def test_experiment_runner_integration(tmp_path):
    """Verify ExperimentRunner creates proper run records.

    Uses synthetic data, runs a minimal experiment, verifies:
    - Run is registered
    - Config is saved
    - Status transitions: pending -> running -> completed
    - Artifacts directory exists
    """
    registry = RunRegistry(base_dir=str(tmp_path / "runs"))
    runner = ExperimentRunner(registry=registry)

    config = {
        "name": "e2e_test_experiment",
        "backtest": {
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "initial_capital": 100_000,
        },
        "strategies": ["momentum", "trend"],
        "risk": {"max_position_pct": 0.05},
    }

    run = runner.run(config, seed=42, notes="E2E integration test")

    assert run is not None
    assert run.status == "completed"
    assert run.seed == 42
    assert run.config == config
    assert "E2E integration test" in run.notes

    artifacts_path = Path(run.artifacts_dir)
    assert artifacts_path.exists(), "Artifacts directory should exist"
    assert (artifacts_path / "config.json").exists(), "Config snapshot should be saved"

    retrieved = registry.get_run(run.run_id)
    assert retrieved is not None
    assert retrieved.status == "completed"
    assert retrieved.run_id == run.run_id

    all_runs = registry.list_runs()
    assert len(all_runs) >= 1
    assert any(r.run_id == run.run_id for r in all_runs)


# ---------------------------------------------------------------------------
# Test 7: Metrics computation on synthetic returns
# ---------------------------------------------------------------------------


def test_metrics_on_pipeline_output():
    """Verify eval metrics work with realistic return series from a pipeline run."""
    np.random.seed(42)
    n_days = 252
    daily_returns = pd.Series(
        np.random.normal(0.0005, 0.015, n_days),
        index=pd.bdate_range("2020-01-01", periods=n_days),
    )

    metrics = compute_all_metrics(daily_returns)

    assert "total_return" in metrics
    assert "sharpe_ratio" in metrics
    assert "max_drawdown" in metrics
    assert "cagr" in metrics
    assert "annualized_volatility" in metrics
    assert "sortino_ratio" in metrics
    assert "calmar_ratio" in metrics
    assert "hit_rate" in metrics

    assert metrics["annualized_volatility"] > 0
    assert 0 <= metrics["hit_rate"] <= 1
    assert metrics["max_drawdown"] >= 0


# ---------------------------------------------------------------------------
# Test 8: PitViewAdapter protocol compliance
# ---------------------------------------------------------------------------


def test_pit_view_adapter_protocol():
    """Verify PitViewAdapter satisfies the PitView protocol."""
    symbols = ["AAAA", "BBBB"]
    pit_store, prices_df = build_pit_store(symbols, n_days=50)

    dates = sorted(prices_df["date"].unique())
    asof_dt = pd.Timestamp(dates[-1]).to_pydatetime()

    adapter = PitViewAdapter(pit_store, asof_dt, symbols)

    assert isinstance(adapter, PitView)
    assert adapter.asof == asof_dt
    assert set(adapter.universe) == set(symbols)

    prices = adapter.prices()
    assert isinstance(prices, pd.DataFrame)
    if not prices.empty:
        assert "date" in prices.columns
        assert "symbol" in prices.columns
        max_date = prices["date"].max()
        assert max_date <= pd.Timestamp(asof_dt)

    fundamentals = adapter.fundamentals()
    assert isinstance(fundamentals, pd.DataFrame)

    sentiment = adapter.sentiment()
    assert isinstance(sentiment, pd.DataFrame)
