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
            "strategy_params", "regime_overlay", "seasonality_overlay",
            "macro_overlay", "agent_modes",
            "llm_config", "universe_symbols",
            "allocation_method", "kelly_fraction", "signal_combination",
            "strategy_circuit_breaker",
            "strategy_regime_weights",
            "conviction_smoothing_enabled",
            "conviction_smoothing_halflife_days",
            "zscore_demean",
            # See run_walk_forward_pbo_audit.py's _build_config comment:
            # surfaced as top-level keys specifically so a param_grid
            # candidate can override just these two fields without a
            # shallow top-level merge clobbering the whole nested
            # "backtest" sub-dict they normally live in.
            "rebalance_band_pct",
            "rebalance_fraction",
            # allocation_method == "joint_optimizer" knobs (see
            # firm.agents.trader.TraderAgent.__init__ / firm.portfolio.
            # optimizer). Not nested under "risk"/"backtest" in any existing
            # config, so without this explicit entry a param_grid candidate
            # overriding one of these would be silently dropped by this
            # allowlist -- the same bug class hit twice already this session
            # (conviction_smoothing_enabled, rebalance_band_pct/fraction).
            "optimizer_cost_aversion",
            "optimizer_target_avg_vol",
            "optimizer_ridge_frac",
            "optimizer_holding_horizon_days",
            "optimizer_cov_lookback_days",
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
        # avg_turnover/total_turnover/rebalance_count — needed to validate a
        # construction change (no-trade bands, turnover-aware sizing, ...)
        # against the metric it's actually meant to move, with the same
        # per-fold mean/std/min/max treatment aggregate_walk_forward gives
        # every other metric here.
        metrics.update(report_dict.get("turnover", {}))

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
        param_grid: list[dict[str, Any]] | None = None,
        selection_metric: str = "sharpe_ratio",
        embargo_days: int = 1,
    ) -> list[ExperimentRun]:
        """Run walk-forward analysis.

        Splits the date range into n_splits windows (train on train_pct,
        test on the remainder). When *param_grid* names two or more candidate
        config overrides, each fold performs genuine train->select->test
        optimization: every candidate is backtested over that fold's **train**
        window, the one with the best in-sample *selection_metric* (default
        ``sharpe_ratio``) is picked, and *that* candidate — not the input
        *config* verbatim — is what actually runs on the **test** window.
        Without a grid (the default), the train window is never touched and
        each fold simply backtests *config* unchanged over the test window,
        exactly as before — a plain sequential out-of-sample replay, not an
        optimization (there is nothing to optimize with a single candidate).

        Every fold with a genuine multi-candidate selection gets a
        ``walk_forward_selection.json`` written into its artifacts dir
        (candidates tried, the winner, and each candidate's train-window
        per-period returns) — :meth:`_walk_forward_overfitting` reads these
        to compute PBO/DSR from real competing trials instead of the old
        heuristic of treating sequential OOS folds as pseudo-trials.

        ``embargo_days`` (default 1, matching prior behaviour) is the
        calendar-day gap enforced between each fold's train and test windows
        — see :meth:`_compute_walk_forward_splits`.

        Returns list of ExperimentRun for each fold (the *test*-window run).
        """
        backtest_cfg = config.get("backtest", {})
        start_date = backtest_cfg.get("start_date", "2018-01-01")
        end_date = backtest_cfg.get("end_date", "2023-12-31")

        splits = self._compute_walk_forward_splits(
            start_date, end_date, n_splits, train_pct, embargo_days=embargo_days
        )
        candidates = param_grid or [{}]
        log.info(
            "run_walk_forward: %d folds x %d candidate(s) (selection_metric=%s)",
            len(splits), len(candidates), selection_metric,
        )

        runs: list[ExperimentRun] = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(
            splits
        ):
            selected_override: dict[str, Any] = candidates[0]
            selection: dict[str, Any] | None = None
            if len(candidates) > 1:
                selected_override, selection = self._select_candidate_on_train(
                    config, candidates, train_start, train_end, selection_metric,
                )

            fold_config = {
                **self._merge_override(config, selected_override),
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
                    **(
                        {"selected_candidate": selection["selected_index"]}
                        if selection
                        else {}
                    ),
                },
            }
            notes = f"walk-forward fold {i}"
            if selection is not None:
                notes += (
                    f" (candidate {selection['selected_index']}/{len(candidates)}"
                    " selected on train)"
                )
            fold_run = self.run(fold_config, seed=seed, notes=notes)
            if selection is not None:
                self._save_walk_forward_selection(fold_run, selection)
            runs.append(fold_run)

        return runs

    @staticmethod
    def _merge_override(base: dict, override: dict) -> dict:
        """Shallow-merge *override* onto *base*, deep-merging ``strategy_params``
        one level (per-strategy dicts) so a grid candidate can override a
        single strategy's single param without clobbering every other
        strategy's settings or that strategy's other params.
        """
        if not override:
            return dict(base)
        merged = {**base, **override}
        base_sp = base.get("strategy_params")
        override_sp = override.get("strategy_params")
        if isinstance(base_sp, dict) and isinstance(override_sp, dict):
            sp = {**base_sp}
            for strat, params in override_sp.items():
                if isinstance(params, dict) and isinstance(sp.get(strat), dict):
                    sp[strat] = {**sp[strat], **params}
                else:
                    sp[strat] = params
            merged["strategy_params"] = sp
        return merged

    def _select_candidate_on_train(
        self,
        config: dict,
        candidates: list[dict[str, Any]],
        train_start: str,
        train_end: str,
        selection_metric: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Backtest every candidate override over the train window and return
        ``(winning_override, selection_record)``.

        ``selection_record`` captures every candidate's train-window metric
        value and per-period returns — the latter is what lets
        :meth:`_walk_forward_overfitting` compute genuine trial-based
        PBO/DSR instead of misusing OOS folds as pseudo-trials.
        """
        from firm.backtest.run import execute_backtest

        backtest_cfg = config.get("backtest", {})
        trials: list[dict[str, Any]] = []
        for idx, override in enumerate(candidates):
            train_config = self._flatten_config({
                **self._merge_override(config, override),
                "backtest": {
                    **backtest_cfg, "start_date": train_start, "end_date": train_end,
                },
            })
            metric_value: float | None = None
            returns: list[float] = []
            try:
                report = execute_backtest(train_config)
                metric_value = report.portfolio_summary().get(selection_metric)
                if not report.returns.empty:
                    returns = [float(v) for v in report.returns.tolist()]
            except Exception:
                log.warning(
                    "walk-forward: candidate %d/%d failed on train window "
                    "%s..%s; scoring as unusable",
                    idx, len(candidates), train_start, train_end, exc_info=True,
                )
            trials.append({
                "index": idx,
                "override": override,
                "metric_value": metric_value,
                "train_returns": returns,
            })
            log.debug(
                "walk-forward train candidate %d/%d %s=%s (train %s..%s)",
                idx, len(candidates), selection_metric, metric_value,
                train_start, train_end,
            )

        def _score(t: dict[str, Any]) -> float:
            v = t["metric_value"]
            return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else float("-inf")

        best = max(trials, key=_score)
        if _score(best) == float("-inf"):
            log.warning(
                "walk-forward: every candidate failed or produced no usable "
                "%s on train window %s..%s; defaulting to candidate 0 rather "
                "than silently picking an arbitrary one",
                selection_metric, train_start, train_end,
            )
            best = trials[0]
        log.info(
            "walk-forward: selected candidate %d/%d (train %s=%s) for train "
            "window %s..%s, to run on the test window",
            best["index"], len(candidates), selection_metric,
            best["metric_value"], train_start, train_end,
        )

        selection = {
            "selection_metric": selection_metric,
            "selected_index": best["index"],
            "train_start": train_start,
            "train_end": train_end,
            "trials": [
                {
                    "index": t["index"],
                    "override": t["override"],
                    "metric_value": t["metric_value"],
                }
                for t in trials
            ],
            "train_returns_by_trial": [t["train_returns"] for t in trials],
        }
        return best["override"], selection

    @staticmethod
    def _save_walk_forward_selection(run: ExperimentRun, selection: dict[str, Any]) -> None:
        path = Path(run.artifacts_dir) / "walk_forward_selection.json"
        path.write_text(json.dumps(selection, indent=2, default=str), encoding="utf-8")

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
    def aggregate_walk_forward(
        runs: list[ExperimentRun], embargo_pct: float = 0.0
    ) -> dict[str, Any]:
        """Summarize out-of-sample metrics across walk-forward folds.

        Returns ``{"n_folds", "fold_ids", "metrics": {name: {mean, std,
        min, max, values}}}`` over the completed folds. Aggregating the
        per-fold OOS metrics (not a single full-window fit) is what makes
        walk-forward an honest overfitting check.

        ``embargo_pct`` (default 0.0 = original behaviour) is forwarded to
        the per-fold PBO computation — see
        :func:`firm.eval.overfitting.cscv_pbo`.
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
        overfitting = ExperimentRunner._walk_forward_overfitting(
            completed, embargo_pct=embargo_pct
        )
        if overfitting:
            result["overfitting"] = overfitting
        return result

    @staticmethod
    def _walk_forward_overfitting(
        runs: list[ExperimentRun], embargo_pct: float = 0.0
    ) -> dict[str, Any]:
        """Load each fold's OOS returns and run the Bailey/LdP overfitting checks.

        Reads ``equity.json`` from each fold's artifacts dir, converts the NAV
        curve to per-period returns, and delegates to
        :func:`firm.eval.overfitting.walk_forward_overfitting` for the pooled
        OOS PSR. When a fold also has a ``walk_forward_selection.json`` (i.e.
        :meth:`run_walk_forward` was given a real ``param_grid``), that fold's
        genuine per-candidate train-window returns are passed through too, so
        PBO/DSR are computed from real competing trials rather than treating
        sequential OOS folds as pseudo-trials. Degrades to an empty dict if
        the equity artifacts are missing or too short. ``embargo_pct`` is
        forwarded to the underlying CSCV/PBO computation.
        """
        from firm.eval.overfitting import walk_forward_overfitting

        fold_returns: list[list[float]] = []
        fold_trial_returns: list[list[list[float]]] = []
        for r in runs:
            art_dir = Path(r.artifacts_dir)
            equity_path = art_dir / "equity.json"
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
            if len(rets) < 2:
                continue
            fold_returns.append(rets)

            selection_path = art_dir / "walk_forward_selection.json"
            if not selection_path.exists():
                continue
            try:
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning(
                    "overfitting: could not read %s (%s); fold's train "
                    "trials omitted from PBO/DSR",
                    selection_path, exc,
                )
                continue
            trial_returns = selection.get("train_returns_by_trial") or []
            if trial_returns:
                fold_trial_returns.append(trial_returns)

        if len(fold_returns) < 2:
            return {}
        return walk_forward_overfitting(
            [np.asarray(f) for f in fold_returns],
            fold_trial_returns=(
                [[np.asarray(t) for t in fold] for fold in fold_trial_returns]
                or None
            ),
            embargo_pct=embargo_pct,
        )

    @staticmethod
    def _compute_walk_forward_splits(
        start_date: str,
        end_date: str,
        n_splits: int,
        train_pct: float,
        embargo_days: int = 1,
    ) -> list[tuple[str, str, str, str]]:
        """Compute walk-forward date splits.

        ``embargo_days`` (default 1, matching the previous hard-coded
        behaviour exactly) is the calendar-day gap enforced between each
        fold's train and test windows — an embargo against serial-correlation
        leakage across the train/test boundary (López de Prado): the test
        window never starts the day immediately after training ends. ``0``
        reproduces a back-to-back split with no gap.

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
            test_start = train_end + timedelta(days=max(embargo_days, 0))
            test_end = window_end

            splits.append((
                train_start.strftime("%Y-%m-%d"),
                train_end.strftime("%Y-%m-%d"),
                test_start.strftime("%Y-%m-%d"),
                test_end.strftime("%Y-%m-%d"),
            ))

        return splits
