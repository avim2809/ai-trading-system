"""Experiment registry – track and reproduce past experiment configurations.

Manages versioned experiment runs with config snapshots, metrics,
and artifacts directories for full reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentRun:
    """Record of a single experiment run."""

    run_id: str
    config: dict[str, Any]
    config_hash: str
    start_time: datetime
    end_time: datetime | None = None
    status: str = "pending"
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts_dir: str = ""
    notes: str = ""
    seed: int = 42

    def to_dict(self) -> dict:
        data = asdict(self)
        data["start_time"] = self.start_time.isoformat()
        data["end_time"] = self.end_time.isoformat() if self.end_time else None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentRun:
        data = dict(data)
        data["start_time"] = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            data["end_time"] = datetime.fromisoformat(data["end_time"])
        else:
            data["end_time"] = None
        return cls(**data)


class RunRegistry:
    """Manages versioned experiment runs."""

    def __init__(self, base_dir: str = "runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._registry_file = self.base_dir / "registry.json"
        self._runs: list[ExperimentRun] = []
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if self._registry_file.exists():
            raw = json.loads(self._registry_file.read_text(encoding="utf-8"))
            self._runs = [ExperimentRun.from_dict(r) for r in raw]
        else:
            self._runs = []

    def _save(self) -> None:
        """Persist registry to disk."""
        data = [run.to_dict() for run in self._runs]
        self._registry_file.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    def create_run(
        self, config: dict, seed: int = 42, notes: str = ""
    ) -> ExperimentRun:
        """Create and register a new run. Returns ExperimentRun with artifacts_dir set."""
        now = datetime.now()
        cfg_hash = self.config_hash(config)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S_%f")
        run_id = f"{timestamp_str}_{cfg_hash[:8]}"

        artifacts_dir = self.base_dir / run_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        config_snapshot_path = artifacts_dir / "config.json"
        config_snapshot_path.write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8"
        )

        run = ExperimentRun(
            run_id=run_id,
            config=config,
            config_hash=cfg_hash,
            start_time=now,
            status="pending",
            artifacts_dir=str(artifacts_dir),
            notes=notes,
            seed=seed,
        )
        self._runs.append(run)
        self._save()
        return run

    def update_run(self, run_id: str, **kwargs: Any) -> None:
        """Update run fields (status, metrics, end_time, etc.)."""
        run = self.get_run(run_id)
        if run is None:
            raise KeyError(f"Run {run_id!r} not found in registry")
        for key, value in kwargs.items():
            if not hasattr(run, key):
                raise AttributeError(
                    f"ExperimentRun has no attribute {key!r}"
                )
            setattr(run, key, value)
        self._save()

    def get_run(self, run_id: str) -> ExperimentRun | None:
        """Retrieve a run by ID."""
        for run in self._runs:
            if run.run_id == run_id:
                return run
        return None

    def list_runs(self, status: str | None = None) -> list[ExperimentRun]:
        """List runs, optionally filtered by status."""
        if status is None:
            return list(self._runs)
        return [r for r in self._runs if r.status == status]

    def compare_runs(self, run_ids: list[str]) -> dict:
        """Compare metrics across runs. Returns {metric: {run_id: value}}."""
        runs = [self.get_run(rid) for rid in run_ids]
        runs = [r for r in runs if r is not None]

        all_metrics: set[str] = set()
        for r in runs:
            all_metrics.update(r.metrics.keys())

        result: dict[str, dict[str, float]] = {}
        for metric in sorted(all_metrics):
            result[metric] = {}
            for r in runs:
                result[metric][r.run_id] = r.metrics.get(metric, float("nan"))
        return result

    @staticmethod
    def config_hash(config: dict) -> str:
        """Deterministic hash of config for reproducibility tracking."""
        return hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest()
