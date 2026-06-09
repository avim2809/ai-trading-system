"""Tests for the FastAPI backend.

Uses FastAPI's TestClient (built on httpx) to exercise every endpoint
with synthetic data, requiring no API keys or cached market files.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from firm.api.app import create_app

# Reset the runs router singletons before each test module import so
# tests don't leak state between runs.
import firm.api.routers.runs as _runs_mod


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path):
    """Give each test a fresh RunRegistry backed by a temp directory."""
    from firm.experiments.registry import RunRegistry
    from firm.api.jobs import JobManager

    registry = RunRegistry(base_dir=str(tmp_path / "runs"))
    _runs_mod._registry = registry
    _runs_mod._job_manager = JobManager(registry)
    yield
    _runs_mod._registry = None
    _runs_mod._job_manager = None


@pytest.fixture()
def client():
    app = create_app()
    return TestClient(app)


# ------------------------------------------------------------------
# Meta endpoints
# ------------------------------------------------------------------

class TestMeta:
    def test_health(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_strategies(self, client):
        r = client.get("/api/strategies")
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, list)
        names = [s["name"] for s in data]
        assert "momentum" in names
        assert "trend" in names

    def test_config_defaults(self, client):
        r = client.get("/api/config/defaults")
        assert r.status_code == 200
        data = r.json()
        assert "universe" in data
        assert "backtest" in data
        assert "risk" in data
        assert "initial_capital" in data["backtest"]


# ------------------------------------------------------------------
# Runs endpoints
# ------------------------------------------------------------------

class TestRuns:
    def test_list_runs_empty(self, client):
        r = client.get("/api/runs")
        assert r.status_code == 200
        assert r.json() == []

    def _launch_and_wait(self, client) -> str:
        """Launch a synthetic backtest and poll until completed."""
        r = client.post("/api/runs", json={
            "strategies": ["momentum", "trend"],
            "data_source": "synthetic",
            "start_date": "2022-01-01",
            "end_date": "2023-12-31",
            "seed": 42,
            "notes": "api-test",
        })
        assert r.status_code == 200
        run_id = r.json()["run_id"]
        assert run_id

        terminal_statuses = {"completed", "failed"}
        status = "pending"
        for _ in range(120):
            time.sleep(0.5)
            detail = client.get(f"/api/runs/{run_id}").json()
            status = detail["status"]
            if status in terminal_statuses:
                break

        assert status == "completed", f"Run ended with status={status}, notes={detail.get('notes')}"
        assert detail["metrics"]
        return run_id

    def test_launch_and_poll(self, client):
        self._launch_and_wait(client)

    def test_report_after_completion(self, client):
        run_id = self._launch_and_wait(client)
        r = client.get(f"/api/runs/{run_id}/report")
        assert r.status_code == 200
        data = r.json()
        assert "portfolio" in data

    def test_equity_after_completion(self, client):
        run_id = self._launch_and_wait(client)
        r = client.get(f"/api/runs/{run_id}/equity")
        assert r.status_code == 200
        data = r.json()
        assert "dates" in data
        assert "values" in data
        assert "drawdown" in data

    def test_get_run_not_found(self, client):
        r = client.get("/api/runs/nonexistent")
        assert r.status_code == 404

    def test_compare(self, client):
        r1 = client.post("/api/runs", json={
            "strategies": ["momentum"],
            "data_source": "synthetic",
            "seed": 1,
        })
        r2 = client.post("/api/runs", json={
            "strategies": ["trend"],
            "data_source": "synthetic",
            "seed": 2,
        })
        id1 = r1.json()["run_id"]
        id2 = r2.json()["run_id"]

        for _ in range(120):
            time.sleep(0.5)
            s1 = client.get(f"/api/runs/{id1}").json()["status"]
            s2 = client.get(f"/api/runs/{id2}").json()["status"]
            if s1 in {"completed", "failed"} and s2 in {"completed", "failed"}:
                break

        r = client.post("/api/runs/compare", json={"run_ids": [id1, id2]})
        assert r.status_code == 200


# ------------------------------------------------------------------
# Agent step endpoint
# ------------------------------------------------------------------

class TestAgentStep:
    def test_agent_step(self, client):
        r = client.post("/api/agents/step", json={
            "strategies": ["momentum", "trend"],
            "symbols": ["AAPL", "MSFT", "GOOG", "AMZN", "META"],
            "asof_date": "2023-06-15",
            "data_source": "synthetic",
            "seed": 42,
        })
        assert r.status_code == 200
        data = r.json()
        assert "asof" in data
        assert "signal_sets" in data
        assert "theses" in data
        assert "debate_results" in data
        assert isinstance(data["signal_sets"], list)
        assert len(data["signal_sets"]) > 0
