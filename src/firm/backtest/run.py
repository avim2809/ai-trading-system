"""Reusable backtest execution from a flat config dict.

Extracted from the API job manager so the same prices → PIT store →
orchestrator → engine → report flow can be driven by the background job
runner, the CLI, and the walk-forward experiment harness without
duplicating the wiring.
"""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from firm.backtest.engine import BacktestEngine
from firm.data.pit_store import PointInTimeDataStore
from firm.eval.reports import BacktestReport
from firm.runtime import build_orchestrator

log = logging.getLogger(__name__)

# Backtest-section keys consumed by BacktestEngine.
_BT_FIELDS = frozenset({
    "start_date", "end_date", "initial_capital",
    "commission_pct", "slippage_pct", "rebalance_frequency",
})


def execute_backtest(config: dict) -> BacktestReport:
    """Run a single backtest from a flat *config* and return its report.

    ``config`` keys: ``data_source`` (``"synthetic"`` or a real provider),
    ``start_date``/``end_date``, ``universe_symbols``, ``strategies``,
    ``seed``, ``warmup_days`` (real-data only, default 365), plus the
    engine fields in :data:`_BT_FIELDS`. Mirrors the wiring previously
    inline in ``firm.api.jobs.JobManager``.
    """
    data_source = config.get("data_source", "synthetic")
    start_date = config.get("start_date", "2020-01-01")
    end_date = config.get("end_date", "2023-12-31")
    warmup_days = config.get("warmup_days", 365)

    if data_source == "synthetic":
        from firm.data.synthetic import DEFAULT_SYMBOLS, make_synthetic_prices

        symbols = config.get("universe_symbols") or list(DEFAULT_SYMBOLS)
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        span_days = (end_dt - start_dt).days
        n_days = int(span_days * 5 / 7) + 252
        prices_df = make_synthetic_prices(
            symbols=symbols,
            n_days=n_days,
            end_date=end_date,
            seed=config.get("seed", 42),
        )
    else:
        from firm.config import get_settings
        from firm.runtime import load_prices

        prices_df = load_prices(get_settings())
        symbols = config.get("universe_symbols") or []

        # Unlike the synthetic branch (which generates exactly the requested
        # span), cached/real data is loaded in full regardless of what's
        # asked for — a walk-forward run's 5 folds each request a different
        # start_date/end_date, but without this filter every fold ran on the
        # *entire* cached history and produced byte-identical results,
        # silently defeating the whole point of walk-forward validation.
        #
        # Loading exactly [start_date, end_date] and nothing more created a
        # second, subtler bug: strategies with a real lookback requirement
        # (regime_hmm needs 252 days to train its HMM, momentum's 12-month
        # factor needs 252, gann needs 120+) got starved for a large chunk
        # of every ~99-trading-day fold — some generated zero signals until
        # well into the fold, others silently degraded to a much shorter,
        # different lookback than what they were designed for. Loading an
        # extra `warmup_days` of history *before* start_date gives those
        # strategies real data to work with; FirmStrategy still won't
        # place a single trade before start_date (see its own start_date
        # gate), so this can't leak into the reported performance — it only
        # gives long-lookback strategies a fair, correctly-warmed-up start.
        load_start = pd.Timestamp(start_date) - pd.Timedelta(days=warmup_days)
        dates = pd.to_datetime(prices_df["date"])
        mask = (dates >= load_start) & (dates <= pd.Timestamp(end_date))
        prices_df = prices_df[mask]

    pit_store = PointInTimeDataStore()
    fund_df = None
    if data_source != "synthetic":
        try:
            from firm.config import get_settings
            from firm.runtime import load_fundamentals

            fund_df = load_fundamentals(get_settings())
        except Exception:
            log.warning(
                "Fundamentals cache load failed — continuing price-only",
                exc_info=True,
            )

    pit_store.load(prices=prices_df, fundamentals=fund_df)
    if fund_df is not None and not fund_df.empty:
        log.info(
            "Loaded fundamentals cache: %d rows, %d symbols",
            len(fund_df), fund_df["symbol"].nunique(),
        )
    elif data_source != "synthetic":
        log.debug(
            "No cached fundamentals in backtest; fundamental strategies "
            "use degraded logic (see multi_factor / event_driven)"
        )

    universe = symbols or pit_store.get_universe(datetime.fromisoformat(start_date))

    orchestrator = build_orchestrator(config)

    bt_config = {k: v for k, v in config.items() if k in _BT_FIELDS}
    engine = BacktestEngine(bt_config)
    engine.setup(prices_df, pit_store, orchestrator, universe)
    engine.run()
    report = engine.generate_report()

    if data_source != "synthetic":
        # Trading/snapshots already can't start before start_date (the
        # engine-level gate), but the raw return series still runs from
        # whatever the first loaded bar is — trim it back to the actual
        # evaluation window before it reaches any metric computation.
        start_ts = pd.Timestamp(start_date)
        if not report.returns.empty:
            report.returns = report.returns[report.returns.index >= start_ts]
        if not report.benchmark_returns.empty:
            report.benchmark_returns = report.benchmark_returns[report.benchmark_returns.index >= start_ts]

    return report


def build_equity_data(report: BacktestReport) -> dict:
    """Extract equity curve + drawdown series from a report for the UI."""
    data: dict = {"dates": [], "values": [], "drawdown": []}
    if not report.snapshots:
        return data
    data["dates"] = [s.asof.isoformat() for s in report.snapshots]
    data["values"] = [s.nav for s in report.snapshots]
    peak = 0.0
    for nav in data["values"]:
        peak = max(peak, nav)
        dd = (nav - peak) / peak if peak > 0 else 0.0
        data["drawdown"].append(round(dd, 6))
    return data
