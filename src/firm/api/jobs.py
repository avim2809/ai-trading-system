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

    def run_walk_forward_sync(
        self,
        config: dict,
        n_splits: int = 5,
        train_pct: float = 0.7,
        seed: int = 42,
        param_grid: list[dict] | None = None,
        selection_metric: str = "sharpe_ratio",
    ) -> dict:
        """Run a walk-forward analysis to completion and return its summary.

        Each fold is registered as a normal run (so it appears in the
        dashboard) and executed under ``self._lock`` to serialise Cerebro.
        With a ``param_grid`` of >=2 candidates, every fold additionally runs
        each candidate on its train window first to select the one that runs
        on the test window — this multiplies total runtime by roughly
        ``len(param_grid)`` per fold, since this endpoint blocks until the
        whole analysis (all folds x all candidates) completes.
        Returns ``{"fold_ids", "aggregate"}``.
        """
        from firm.experiments.runner import ExperimentRunner

        with self._lock:
            runner = ExperimentRunner(registry=self.registry)
            runs = runner.run_walk_forward(
                config,
                n_splits=n_splits,
                train_pct=train_pct,
                seed=seed,
                param_grid=param_grid,
                selection_metric=selection_metric,
            )
            aggregate = runner.aggregate_walk_forward(runs)
        return {"fold_ids": [r.run_id for r in runs], "aggregate": aggregate}

    def _run(self, run_id: str, config: dict) -> None:
        with self._lock:
            try:
                self.registry.update_run(run_id, status="running")

                from firm.backtest.run import execute_backtest

                report = execute_backtest(config)

                artifacts_dir = self.registry.get_run(run_id).artifacts_dir
                art = Path(artifacts_dir)
                art.mkdir(parents=True, exist_ok=True)

                report.save(str(art / "report.json"))
                report.save_trades(str(art / "trades.parquet"))

                report_dict = report.to_dict()
                metrics = report_dict.get("portfolio", {})

                from firm.backtest.run import build_equity_data

                equity_data = build_equity_data(report)
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
