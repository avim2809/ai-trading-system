"""Background job manager for backtest execution.

Runs backtests in a daemon thread (serialised via a lock since
Backtrader/Cerebro is single-threaded and CPU-heavy), updating the
RunRegistry with status transitions and writing result artefacts.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from firm.experiments.registry import RunRegistry

log = logging.getLogger(__name__)


class JobManager:
    """Manages background backtest execution."""

    def __init__(self, registry: RunRegistry) -> None:
        self.registry = registry
        self._lock = threading.Lock()
        self._threads_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def launch(self, run_id: str, config: dict) -> None:
        """Start a backtest in a background daemon thread.

        Backtests are serialised by ``self._lock`` (Cerebro is single-threaded
        and CPU-heavy), so a launch while another run is in progress will wait.
        We mark such runs ``queued`` up front so they are not invisibly stuck
        in ``pending`` with no indication they are waiting.
        """
        if self._lock.locked():
            try:
                self.registry.update_run(run_id, status="queued")
            except Exception:
                log.debug("Could not mark run %s queued", run_id, exc_info=True)
        t = threading.Thread(target=self._run, args=(run_id, config), daemon=True)
        with self._threads_lock:
            self._threads = [x for x in self._threads if x.is_alive()]
            self._threads.append(t)
        t.start()

    def _run(self, run_id: str, config: dict) -> None:
        with self._lock:
            try:
                self.registry.update_run(run_id, status="running")

                from firm.data.synthetic import make_synthetic_prices, DEFAULT_SYMBOLS
                from firm.runtime import build_orchestrator

                data_source = config.get("data_source", "synthetic")
                start_date = config.get("start_date", "2020-01-01")
                end_date = config.get("end_date", "2023-12-31")

                if data_source == "synthetic":
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
                    from firm.runtime import load_prices
                    from firm.config import get_settings
                    settings = get_settings()
                    prices_df = load_prices(settings)
                    symbols = config.get("universe_symbols") or []

                from firm.data.pit_store import PointInTimeDataStore
                pit_store = PointInTimeDataStore()
                pit_store.load(prices=prices_df)

                universe = symbols or pit_store.get_universe(
                    datetime.fromisoformat(start_date)
                )

                orchestrator = build_orchestrator(config)

                from firm.backtest.engine import BacktestEngine

                bt_fields = {
                    "start_date", "end_date", "initial_capital",
                    "commission_pct", "slippage_pct", "rebalance_frequency",
                }
                bt_config = {k: v for k, v in config.items() if k in bt_fields}

                engine = BacktestEngine(bt_config)
                engine.setup(prices_df, pit_store, orchestrator, universe)
                engine.run()

                report = engine.generate_report()

                artifacts_dir = self.registry.get_run(run_id).artifacts_dir
                art = Path(artifacts_dir)
                art.mkdir(parents=True, exist_ok=True)

                report.save(str(art / "report.json"))

                report_dict = report.to_dict()
                metrics = report_dict.get("portfolio", {})

                equity_data = self._build_equity_data(report)
                (art / "equity.json").write_text(
                    json.dumps(equity_data, indent=2, default=str),
                    encoding="utf-8",
                )

                self.registry.update_run(
                    run_id,
                    status="completed",
                    end_time=datetime.now(),
                    metrics=metrics,
                )
                log.info("Run %s completed", run_id)

            except Exception as e:
                log.error("Run %s failed: %s", run_id, e, exc_info=True)
                self.registry.update_run(
                    run_id,
                    status="failed",
                    end_time=datetime.now(),
                    notes=str(e),
                )

    @staticmethod
    def _build_equity_data(report) -> dict:
        """Extract equity curve and drawdown from a BacktestReport."""
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
