"""Experiment runner – execute parameter sweeps and collect results.

Provides reproducible, parameterized backtest execution with seeded
randomness, walk-forward analysis, and in-sample/OOS splitting.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from firm.experiments.registry import ExperimentRun, RunRegistry


class ExperimentRunner:
    """Runs reproducible, parameterized backtest experiments."""

    def __init__(self, registry: RunRegistry | None = None):
        self.registry = registry or RunRegistry()

    def load_experiment_config(self, config_path: str) -> dict:
        """Load experiment config from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Experiment config not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            raise ValueError(f"Empty experiment config: {path}")
        return data

    def set_seeds(self, seed: int) -> None:
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def run(
        self, config: dict, seed: int = 42, notes: str = ""
    ) -> ExperimentRun:
        """Execute a single experiment run.

        Steps:
        1. Register run in registry
        2. Set seeds
        3. Configure and run backtest engine
        4. Compute metrics
        5. Save results to artifacts_dir
        6. Update registry with metrics and status
        """
        run = self.registry.create_run(config, seed=seed, notes=notes)
        self.set_seeds(seed)

        try:
            self.registry.update_run(run.run_id, status="running")
            results = self._execute_backtest(config, run)
            self.registry.update_run(
                run.run_id,
                status="completed",
                end_time=datetime.now(),
                metrics=results.get("metrics", {}),
            )
        except Exception as e:
            self.registry.update_run(
                run.run_id,
                status="failed",
                end_time=datetime.now(),
                notes=f"Error: {e!s}",
            )
            raise

        return self.registry.get_run(run.run_id)

    def _execute_backtest(self, config: dict, run: ExperimentRun) -> dict:
        """Internal: configure and run the backtest engine.

        Once BacktestEngine is fully wired (Phase 3 engine task),
        this method will instantiate and run it. For now it saves
        the config and returns an empty metrics dict so the runner
        interface is exercisable end-to-end.
        """
        artifacts = Path(run.artifacts_dir)
        results: dict[str, Any] = {"metrics": {}}

        results_path = artifacts / "results.json"
        results_path.write_text(
            json.dumps(results, indent=2, default=str), encoding="utf-8"
        )
        return results

    def run_walk_forward(
        self,
        config: dict,
        n_splits: int = 5,
        train_pct: float = 0.7,
        seed: int = 42,
    ) -> list[ExperimentRun]:
        """Run walk-forward analysis.

        Splits the date range into n_splits windows.
        For each window: train on train_pct, test on remaining.
        Returns list of ExperimentRun for each fold.
        """
        backtest_cfg = config.get("backtest", {})
        start_date = backtest_cfg.get("start_date", "2018-01-01")
        end_date = backtest_cfg.get("end_date", "2023-12-31")

        splits = self._compute_walk_forward_splits(
            start_date, end_date, n_splits, train_pct
        )

        runs: list[ExperimentRun] = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(
            splits
        ):
            fold_config = {
                **config,
                "backtest": {
                    **backtest_cfg,
                    "start_date": test_start,
                    "end_date": test_end,
                },
                "_walk_forward": {
                    "fold": i,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                },
            }
            fold_run = self.run(
                fold_config, seed=seed, notes=f"walk-forward fold {i}"
            )
            runs.append(fold_run)

        return runs

    def run_in_sample_oos(
        self, config: dict, split_date: str, seed: int = 42
    ) -> tuple[ExperimentRun, ExperimentRun]:
        """Run in-sample and out-of-sample backtests.

        Returns (in_sample_run, oos_run).
        """
        backtest_cfg = config.get("backtest", {})

        is_config = {
            **config,
            "backtest": {**backtest_cfg, "end_date": split_date},
        }
        oos_config = {
            **config,
            "backtest": {**backtest_cfg, "start_date": split_date},
        }

        is_run = self.run(is_config, seed=seed, notes="in-sample")
        oos_run = self.run(oos_config, seed=seed, notes="out-of-sample")

        return is_run, oos_run

    @staticmethod
    def _compute_walk_forward_splits(
        start_date: str,
        end_date: str,
        n_splits: int,
        train_pct: float,
    ) -> list[tuple[str, str, str, str]]:
        """Compute walk-forward date splits.

        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        from datetime import timedelta

        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end - start).days

        window_days = total_days // n_splits
        splits: list[tuple[str, str, str, str]] = []

        for i in range(n_splits):
            window_start = start + timedelta(days=i * window_days)
            window_end = window_start + timedelta(days=window_days)
            if window_end > end:
                window_end = end

            train_days = int(window_days * train_pct)
            train_start = window_start
            train_end = window_start + timedelta(days=train_days)
            test_start = train_end + timedelta(days=1)
            test_end = window_end

            splits.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))

        return splits
