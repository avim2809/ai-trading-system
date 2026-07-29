"""Tests for the experiment management system (registry + runner)."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from firm.experiments.registry import ExperimentRun, RunRegistry
from firm.experiments.runner import ExperimentRunner


@pytest.fixture
def tmp_runs_dir(tmp_path):
    """Provide a temporary directory for run artifacts."""
    return str(tmp_path / "runs")


@pytest.fixture
def registry(tmp_runs_dir):
    """Create a RunRegistry backed by a temp directory."""
    return RunRegistry(base_dir=tmp_runs_dir)


@pytest.fixture
def sample_config():
    """Minimal experiment config for testing."""
    return {
        "name": "test_experiment",
        "backtest": {
            "start_date": "2020-01-01",
            "end_date": "2022-12-31",
            "initial_capital": 1_000_000,
        },
        "strategies": {"enabled": ["momentum"]},
        "seed": 42,
    }


@pytest.fixture
def experiment_yaml(tmp_path, sample_config):
    """Write a sample experiment YAML and return its path."""
    yaml_path = tmp_path / "test_experiment.yaml"
    yaml_path.write_text(yaml.dump(sample_config), encoding="utf-8")
    return str(yaml_path)


# ---------------------------------------------------------------------------
# RunRegistry tests
# ---------------------------------------------------------------------------


class TestRunRegistry:
    def test_create_run(self, registry, sample_config):
        run = registry.create_run(sample_config, seed=123, notes="test run")

        assert run.status == "pending"
        assert run.seed == 123
        assert run.notes == "test run"
        assert run.config == sample_config
        assert run.config_hash == RunRegistry.config_hash(sample_config)
        assert Path(run.artifacts_dir).exists()

        config_snapshot = Path(run.artifacts_dir) / "config.json"
        assert config_snapshot.exists()
        saved = json.loads(config_snapshot.read_text(encoding="utf-8"))
        assert saved == sample_config

    def test_get_run(self, registry, sample_config):
        run = registry.create_run(sample_config)
        retrieved = registry.get_run(run.run_id)

        assert retrieved is not None
        assert retrieved.run_id == run.run_id
        assert retrieved.config == sample_config

    def test_get_run_not_found(self, registry):
        assert registry.get_run("nonexistent_id") is None

    def test_list_runs_all(self, registry, sample_config):
        registry.create_run(sample_config)
        time.sleep(0.01)
        registry.create_run(sample_config, notes="second")

        runs = registry.list_runs()
        assert len(runs) == 2

    def test_list_runs_by_status(self, registry, sample_config):
        run1 = registry.create_run(sample_config)
        run2 = registry.create_run(sample_config)

        registry.update_run(run1.run_id, status="completed")

        completed = registry.list_runs(status="completed")
        pending = registry.list_runs(status="pending")

        assert len(completed) == 1
        assert completed[0].run_id == run1.run_id
        assert len(pending) == 1
        assert pending[0].run_id == run2.run_id

    def test_update_run(self, registry, sample_config):
        run = registry.create_run(sample_config)
        metrics = {"sharpe_ratio": 1.5, "max_drawdown": 0.12}
        registry.update_run(run.run_id, status="completed", metrics=metrics)

        updated = registry.get_run(run.run_id)
        assert updated.status == "completed"
        assert updated.metrics == metrics

    def test_update_run_not_found(self, registry):
        with pytest.raises(KeyError):
            registry.update_run("bad_id", status="failed")

    def test_update_run_invalid_field(self, registry, sample_config):
        run = registry.create_run(sample_config)
        with pytest.raises(AttributeError):
            registry.update_run(run.run_id, nonexistent_field="value")

    def test_compare_runs(self, registry, sample_config):
        run1 = registry.create_run(sample_config)
        time.sleep(0.01)
        run2 = registry.create_run(sample_config)

        registry.update_run(
            run1.run_id,
            metrics={"sharpe_ratio": 1.2, "max_drawdown": 0.15},
        )
        registry.update_run(
            run2.run_id,
            metrics={"sharpe_ratio": 0.9, "max_drawdown": 0.20},
        )

        comparison = registry.compare_runs([run1.run_id, run2.run_id])

        assert "sharpe_ratio" in comparison
        assert "max_drawdown" in comparison
        assert comparison["sharpe_ratio"][run1.run_id] == 1.2
        assert comparison["sharpe_ratio"][run2.run_id] == 0.9
        assert comparison["max_drawdown"][run1.run_id] == 0.15
        assert comparison["max_drawdown"][run2.run_id] == 0.20

    def test_config_hash_determinism(self, sample_config):
        h1 = RunRegistry.config_hash(sample_config)
        h2 = RunRegistry.config_hash(sample_config)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_config_hash_differs_for_different_configs(self, sample_config):
        h1 = RunRegistry.config_hash(sample_config)
        modified = {**sample_config, "seed": 99}
        h2 = RunRegistry.config_hash(modified)
        assert h1 != h2

    def test_config_hash_key_order_invariant(self):
        c1 = {"a": 1, "b": 2, "c": 3}
        c2 = {"c": 3, "a": 1, "b": 2}
        assert RunRegistry.config_hash(c1) == RunRegistry.config_hash(c2)

    def test_persistence(self, tmp_runs_dir, sample_config):
        reg1 = RunRegistry(base_dir=tmp_runs_dir)
        run = reg1.create_run(sample_config, seed=7, notes="persist test")
        reg1.update_run(
            run.run_id,
            status="completed",
            metrics={"sharpe_ratio": 2.0},
        )

        reg2 = RunRegistry(base_dir=tmp_runs_dir)
        reloaded = reg2.get_run(run.run_id)

        assert reloaded is not None
        assert reloaded.status == "completed"
        assert reloaded.metrics == {"sharpe_ratio": 2.0}
        assert reloaded.seed == 7
        assert reloaded.notes == "persist test"
        assert reloaded.config == sample_config

    def test_run_id_format(self, registry, sample_config):
        run = registry.create_run(sample_config)
        parts = run.run_id.split("_")
        assert len(parts) == 4  # YYYYMMDD_HHMMSS_microseconds_hash8
        assert len(parts[3]) == 8

    def test_experiment_run_to_dict_roundtrip(self, registry, sample_config):
        run = registry.create_run(sample_config)
        d = run.to_dict()
        restored = ExperimentRun.from_dict(d)

        assert restored.run_id == run.run_id
        assert restored.config == run.config
        assert restored.start_time == run.start_time
        assert restored.seed == run.seed

    def test_clear_all_removes_runs_and_artifacts(self, registry, sample_config):
        run1 = registry.create_run(sample_config)
        run2 = registry.create_run(sample_config, notes="second")
        art_dir1 = Path(run1.artifacts_dir)
        art_dir2 = Path(run2.artifacts_dir)
        assert art_dir1.exists()
        assert art_dir2.exists()

        count = registry.clear_all()

        assert count == 2
        assert registry.list_runs() == []
        assert not art_dir1.exists()
        assert not art_dir2.exists()

    def test_clear_all_empty_registry_returns_zero(self, registry):
        assert registry.clear_all() == 0

    def test_clear_all_persists_across_reload(self, tmp_runs_dir, sample_config):
        registry = RunRegistry(base_dir=tmp_runs_dir)
        registry.create_run(sample_config)
        registry.clear_all()

        reloaded = RunRegistry(base_dir=tmp_runs_dir)
        assert reloaded.list_runs() == []


# ---------------------------------------------------------------------------
# ExperimentRunner tests
# ---------------------------------------------------------------------------


class TestExperimentRunner:
    def test_set_seeds_deterministic(self, tmp_runs_dir):
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))

        runner.set_seeds(42)
        vals1 = [random.random() for _ in range(5)]
        np_vals1 = np.random.rand(5).tolist()

        runner.set_seeds(42)
        vals2 = [random.random() for _ in range(5)]
        np_vals2 = np.random.rand(5).tolist()

        assert vals1 == vals2
        assert np_vals1 == np_vals2

    def test_set_seeds_different_seed_different_output(self, tmp_runs_dir):
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))

        runner.set_seeds(42)
        vals1 = np.random.rand(5).tolist()

        runner.set_seeds(99)
        vals2 = np.random.rand(5).tolist()

        assert vals1 != vals2

    def test_load_experiment_config(self, tmp_runs_dir, experiment_yaml, sample_config):
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))
        loaded = runner.load_experiment_config(experiment_yaml)

        assert loaded["name"] == sample_config["name"]
        assert loaded["backtest"] == sample_config["backtest"]
        assert loaded["strategies"] == sample_config["strategies"]

    def test_load_experiment_config_missing_file(self, tmp_runs_dir):
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))
        with pytest.raises(FileNotFoundError):
            runner.load_experiment_config("/nonexistent/path.yaml")

    def test_run_creates_completed_run(self, tmp_runs_dir, sample_config):
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))
        result = runner.run(sample_config, seed=42, notes="basic run")

        assert result.status == "completed"
        assert result.notes == "basic run"
        assert result.end_time is not None
        assert Path(result.artifacts_dir).exists()

    def test_run_saves_results_artifact(self, tmp_runs_dir, sample_config):
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))
        result = runner.run(sample_config)

        results_path = Path(result.artifacts_dir) / "results.json"
        assert results_path.exists()

    def test_walk_forward_split_logic(self):
        splits = ExperimentRunner._compute_walk_forward_splits(
            "2020-01-01", "2022-12-31", n_splits=3, train_pct=0.7
        )

        assert len(splits) == 3
        for train_start, train_end, test_start, test_end in splits:
            assert train_start < train_end
            assert train_end < test_start
            assert test_start <= test_end

    def test_walk_forward_split_embargo_days(self):
        """``embargo_days`` (default 1, matching prior hard-coded behaviour)
        controls the calendar gap between train_end and test_start."""
        default_splits = ExperimentRunner._compute_walk_forward_splits(
            "2020-01-01", "2022-12-31", n_splits=3, train_pct=0.7
        )
        explicit_default_splits = ExperimentRunner._compute_walk_forward_splits(
            "2020-01-01", "2022-12-31", n_splits=3, train_pct=0.7, embargo_days=1
        )
        assert default_splits == explicit_default_splits

        no_gap_splits = ExperimentRunner._compute_walk_forward_splits(
            "2020-01-01", "2022-12-31", n_splits=3, train_pct=0.7, embargo_days=0
        )
        for (_, train_end, test_start, _), (_, d_train_end, d_test_start, _) in zip(
            no_gap_splits, default_splits
        ):
            assert train_end == d_train_end
            assert test_start == train_end  # no embargo -> back-to-back
            assert d_test_start > test_start  # default (1 day) leaves a gap

        wide_splits = ExperimentRunner._compute_walk_forward_splits(
            "2020-01-01", "2022-12-31", n_splits=3, train_pct=0.7, embargo_days=5
        )
        for (_, train_end, test_start, test_end) in wide_splits:
            assert train_end < test_start <= test_end

    def test_walk_forward_runs(self, tmp_runs_dir, sample_config):
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))
        runs = runner.run_walk_forward(sample_config, n_splits=3, train_pct=0.7)

        assert len(runs) == 3
        for run in runs:
            assert run.status == "completed"
            assert "walk-forward fold" in run.notes
            # Real engine execution now produces metrics (no longer a stub).
            assert run.metrics
            assert "sharpe_ratio" in run.metrics
            assert (Path(run.artifacts_dir) / "report.json").exists()

    def test_aggregate_walk_forward(self, tmp_runs_dir, sample_config):
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))
        runs = runner.run_walk_forward(sample_config, n_splits=3, train_pct=0.7)

        agg = runner.aggregate_walk_forward(runs)
        assert agg["n_folds"] == 3
        assert len(agg["fold_ids"]) == 3
        assert "sharpe_ratio" in agg["metrics"]
        sharpe = agg["metrics"]["sharpe_ratio"]
        assert {"mean", "std", "min", "max", "values"} <= set(sharpe)
        assert len(sharpe["values"]) == 3
        assert sharpe["min"] <= sharpe["mean"] <= sharpe["max"]

    def test_walk_forward_no_grid_skips_train_and_matches_legacy_shape(
        self, tmp_runs_dir, sample_config
    ):
        """Default (no param_grid) behaviour is unchanged: no
        walk_forward_selection.json, notes don't mention candidate selection."""
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))
        runs = runner.run_walk_forward(sample_config, n_splits=2, train_pct=0.7)

        assert len(runs) == 2
        for run in runs:
            assert run.status == "completed"
            assert "selected on train" not in run.notes
            assert not (Path(run.artifacts_dir) / "walk_forward_selection.json").exists()

    def test_walk_forward_with_param_grid_selects_on_train(
        self, tmp_runs_dir, sample_config
    ):
        """A real param_grid makes each fold genuinely optimize: every
        candidate is backtested on the train window and the winner (not the
        base config) is what's actually run on the test window."""
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))
        param_grid = [
            {"strategy_params": {"momentum": {"lookback_months": 6}}},
            {"strategy_params": {"momentum": {"lookback_months": 12}}},
        ]
        runs = runner.run_walk_forward(
            sample_config, n_splits=2, train_pct=0.7, param_grid=param_grid
        )

        assert len(runs) == 2
        for run in runs:
            assert run.status == "completed"
            assert "selected on train" in run.notes

            selection_path = Path(run.artifacts_dir) / "walk_forward_selection.json"
            assert selection_path.exists()
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            assert selection["selection_metric"] == "sharpe_ratio"
            assert selection["selected_index"] in (0, 1)
            assert len(selection["trials"]) == 2
            assert len(selection["train_returns_by_trial"]) == 2

            # The fold's config must reflect the *winning* candidate's
            # override, not just the base config verbatim.
            winner = param_grid[selection["selected_index"]]
            assert (
                run.config["strategy_params"]["momentum"]["lookback_months"]
                == winner["strategy_params"]["momentum"]["lookback_months"]
            )

    def test_aggregate_walk_forward_with_param_grid_uses_genuine_trials(
        self, tmp_runs_dir, sample_config
    ):
        """With a real grid, the overfitting block is computed from genuine
        per-fold competing candidates (persisted train_returns_by_trial),
        not the old folds-as-pseudo-trials heuristic."""
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))
        param_grid = [
            {"strategy_params": {"momentum": {"lookback_months": 6}}},
            {"strategy_params": {"momentum": {"lookback_months": 12}}},
            {"strategy_params": {"momentum": {"lookback_months": 9}}},
        ]
        runs = runner.run_walk_forward(
            sample_config, n_splits=2, train_pct=0.7, param_grid=param_grid
        )
        agg = runner.aggregate_walk_forward(runs)

        assert "overfitting" in agg
        of = agg["overfitting"]
        assert "deflated_sharpe" in of
        assert "probabilistic_sharpe" in of
        # Enough train-window rows/candidates should yield a genuine PBO.
        if "pbo" in of:
            assert 0.0 <= of["pbo"] <= 1.0
            assert of["pbo_n_folds"] >= 1

    def test_aggregate_walk_forward_empty(self):
        agg = ExperimentRunner.aggregate_walk_forward([])
        assert agg["n_folds"] == 0
        assert agg["metrics"] == {}

    def test_aggregate_walk_forward_includes_overfitting(self, tmp_path):
        """Folds with equity artifacts get a PBO/DSR overfitting block."""
        from datetime import datetime

        runs = []
        for i in range(5):
            art = tmp_path / f"fold{i}"
            art.mkdir()
            nav = [100_000.0]
            for _ in range(40):
                nav.append(nav[-1] * (1 + 0.001))  # steady OOS drift
            (art / "equity.json").write_text(
                json.dumps({"dates": [], "values": nav}), encoding="utf-8"
            )
            runs.append(
                ExperimentRun(
                    run_id=f"r{i}",
                    config={},
                    config_hash="x",
                    start_time=datetime.now(),
                    status="completed",
                    metrics={"sharpe_ratio": 1.0},
                    artifacts_dir=str(art),
                )
            )

        agg = ExperimentRunner.aggregate_walk_forward(runs)
        assert "overfitting" in agg
        of = agg["overfitting"]
        assert of["n_folds"] == 5
        assert "deflated_sharpe" in of
        assert of["deflated_sharpe"] <= of["probabilistic_sharpe"] + 1e-9

    def test_aggregate_walk_forward_reads_selection_json_for_pbo(self, tmp_path):
        """A fold with a walk_forward_selection.json (genuine multi-candidate
        train trials) contributes a real PBO; one without contributes only to
        the pooled OOS PSR/DSR baseline."""
        from datetime import datetime

        runs = []
        for i in range(4):
            art = tmp_path / f"fold{i}"
            art.mkdir()
            nav = [100_000.0]
            for _ in range(40):
                nav.append(nav[-1] * (1 + 0.001))
            (art / "equity.json").write_text(
                json.dumps({"dates": [], "values": nav}), encoding="utf-8"
            )
            if i < 2:
                rng = np.random.default_rng(i)
                (art / "walk_forward_selection.json").write_text(
                    json.dumps({
                        "selection_metric": "sharpe_ratio",
                        "selected_index": 0,
                        "trials": [],
                        "train_returns_by_trial": [
                            rng.normal(0.0005, 0.01, size=30).tolist()
                            for _ in range(3)
                        ],
                    }),
                    encoding="utf-8",
                )
            runs.append(
                ExperimentRun(
                    run_id=f"r{i}",
                    config={},
                    config_hash="x",
                    start_time=datetime.now(),
                    status="completed",
                    metrics={"sharpe_ratio": 1.0},
                    artifacts_dir=str(art),
                )
            )

        agg = ExperimentRunner.aggregate_walk_forward(runs)
        of = agg["overfitting"]
        assert "pbo" in of
        assert of["pbo_n_folds"] == 2

    def test_in_sample_oos_config_splitting(self, tmp_runs_dir, sample_config):
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))
        is_run, oos_run = runner.run_in_sample_oos(
            sample_config, split_date="2021-06-30"
        )

        assert is_run.status == "completed"
        assert oos_run.status == "completed"
        assert is_run.notes == "in-sample"
        assert oos_run.notes == "out-of-sample"

        assert is_run.config["backtest"]["end_date"] == "2021-06-30"
        assert oos_run.config["backtest"]["start_date"] == "2021-06-30"

    def test_in_sample_oos_preserves_other_config(self, tmp_runs_dir, sample_config):
        runner = ExperimentRunner(registry=RunRegistry(base_dir=tmp_runs_dir))
        is_run, oos_run = runner.run_in_sample_oos(
            sample_config, split_date="2021-06-30"
        )

        assert is_run.config["strategies"] == sample_config["strategies"]
        assert oos_run.config["strategies"] == sample_config["strategies"]
        assert is_run.config["name"] == sample_config["name"]
        assert oos_run.config["name"] == sample_config["name"]
