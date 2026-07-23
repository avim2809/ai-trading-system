"""Experiment runner – execute parameter sweeps and collect results.

Provides reproducible, parameterized backtest execution with seeded
randomness, walk-forward analysis, and in-sample/OOS splitting.
"""

from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from firm.experiments.registry import ExperimentRun, RunRegistry

log = logging.getLogger(__name__)


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

    @staticmethod
    def _flatten_config(config: dict) -> dict:
        """Translate a nested experiment config to the flat config that
        :func:`firm.backtest.run.execute_backtest` consumes.

        Experiment configs nest backtest params under ``backtest`` and the
        strategy list under ``strategies.enabled``; the engine path expects
        those at the top level.
        """
        flat: dict[str, Any] = {}
        bt = config.get("backtest", {}) or {}
        flat.update(bt)

        strategies = config.get("strategies")
        if isinstance(strategies, dict):
            flat["strategies"] = strategies.get("enabled")
        elif isinstance(strategies, list):
            flat["strategies"] = strategies

        flat["seed"] = config.get("seed", 42)
        flat["data_source"] = config.get("data_source", "synthetic")

        # Apply the risk block first so explicit top-level keys (e.g. an
        # overridden regime_overlay) take precedence over risk defaults.
        risk = config.get("risk")
        if isinstance(risk, dict):
            flat.update(risk)

        for key in (
            "strategy_params", "regime_overlay", "agent_modes",
            "llm_config", "universe_symbols",
            "allocation_method", "kelly_fraction", "signal_combination",
        ):
            if key in config:
                flat[key] = config[key]
        return flat

    def _execute_backtest(self, config: dict, run: ExperimentRun) -> dict:
        """Run the backtest engine for *run* and persist its artifacts.

        Writes ``report.json``, ``equity.json`` and ``results.json`` into the
        run's artifacts dir (matching a normal API-launched run, so folds show
        up in the dashboard) and returns a ``{"metrics": {...}}`` dict combining
        portfolio and benchmark-relative metrics.
        """
        from firm.backtest.run import build_equity_data, execute_backtest

        artifacts = Path(run.artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)

        report = execute_backtest(self._flatten_config(config))
        report.save(str(artifacts / "report.json"))
        report.save_trades(str(artifacts / "trades.parquet"))
        (artifacts / "equity.json").write_text(
            json.dumps(build_equity_data(report), indent=2, default=str),
            encoding="utf-8",
        )

        report_dict = report.to_dict()
        metrics: dict[str, Any] = dict(report_dict.get("portfolio", {}))
        metrics.update(report_dict.get("benchmark", {}))

        results: dict[str, Any] = {"metrics": metrics}
        (artifacts / "results.json").write_text(
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
    def aggregate_walk_forward(runs: list[ExperimentRun]) -> dict[str, Any]:
        """Summarize out-of-sample metrics across walk-forward folds.

        Returns ``{"n_folds", "fold_ids", "metrics": {name: {mean, std,
        min, max, values}}}`` over the completed folds. Aggregating the
        per-fold OOS metrics (not a single full-window fit) is what makes
        walk-forward an honest overfitting check.
        """
        completed = [r for r in runs if r.status == "completed" and r.metrics]
        fold_ids = [r.run_id for r in runs]
        if not completed:
            return {"n_folds": 0, "fold_ids": fold_ids, "metrics": {}}

        metric_names: set[str] = set()
        for r in completed:
            metric_names.update(r.metrics.keys())

        agg: dict[str, Any] = {}
        for name in sorted(metric_names):
            vals = [
                float(r.metrics[name])
                for r in completed
                if isinstance(r.metrics.get(name), (int, float))
            ]
            if not vals:
                continue
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / len(vals)
            agg[name] = {
                "mean": mean,
                "std": var ** 0.5,
                "min": min(vals),
                "max": max(vals),
                "values": vals,
            }

        result: dict[str, Any] = {
            "n_folds": len(completed),
            "fold_ids": fold_ids,
            "metrics": agg,
        }

        # Formal overfitting read (PBO / Deflated Sharpe) across the OOS folds.
        overfitting = ExperimentRunner._walk_forward_overfitting(completed)
        if overfitting:
            result["overfitting"] = overfitting
        return result

    @staticmethod
    def _walk_forward_overfitting(runs: list[ExperimentRun]) -> dict[str, Any]:
        """Load each fold's OOS returns and run the Bailey/LdP overfitting checks.

        Reads ``equity.json`` from each fold's artifacts dir, converts the NAV
        curve to per-period returns, and delegates to
        :func:`firm.eval.overfitting.walk_forward_overfitting`. Degrades to an
        empty dict if the equity artifacts are missing or too short.
        """
        from firm.eval.overfitting import walk_forward_overfitting

        fold_returns: list[list[float]] = []
        for r in runs:
            equity_path = Path(r.artifacts_dir) / "equity.json"
            if not equity_path.exists():
                log.debug(
                    "overfitting: fold %s has no equity.json; skipping",
                    getattr(r, "run_id", "?"),
                )
                continue
            try:
                data = json.loads(equity_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning(
                    "overfitting: could not read %s (%s); skipping fold",
                    equity_path, exc,
                )
                continue
            values = [float(v) for v in data.get("values", []) if v]
            if len(values) < 3:
                continue
            rets = [
                values[i] / values[i - 1] - 1.0
                for i in range(1, len(values))
                if values[i - 1] > 0
            ]
            if len(rets) >= 2:
                fold_returns.append(rets)

        if len(fold_returns) < 2:
            return {}
        return walk_forward_overfitting([np.asarray(f) for f in fold_returns])

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
