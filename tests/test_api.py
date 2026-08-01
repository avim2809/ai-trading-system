"""Tests for the FastAPI backend.

Uses FastAPI's TestClient (built on httpx) to exercise every endpoint
with synthetic data, requiring no API keys or cached market files.
"""

from __future__ import annotations

import json
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

    def test_health_no_live_engine_reports_broker_not_applicable(self, client):
        body = client.get("/api/health").json()
        assert body["broker"]["live_engine_running"] is False
        assert body["broker"]["connected"] is None
        assert body["broker"]["type"] is None


class TestSectorMapHardFail:
    """Missing risk.sector_map must refuse to start live trading, not just
    warn per cycle — see RiskAgent.run()'s "Sector concentration cap NOT
    enforced" warning, which is silent enough to go unnoticed for weeks in
    an unattended live deployment.
    """

    @pytest.fixture(autouse=True)
    def _mock_broker(self, monkeypatch, tmp_path):
        import firm.api.routers.live as live_mod
        from tests.test_brokers import MockBroker

        monkeypatch.setattr(live_mod, "_create_broker", lambda broker_type: MockBroker())
        monkeypatch.setattr(live_mod, "_APPROVALS_PATH", str(tmp_path / "approvals.json"))
        monkeypatch.setattr(live_mod, "_TRADE_HISTORY_ORDERS_PATH", str(tmp_path / "order_history.json"))
        monkeypatch.setattr(live_mod, "_TRADE_HISTORY_CYCLES_PATH", str(tmp_path / "cycle_history.json"))
        monkeypatch.setattr(live_mod, "_KILL_SWITCH_STATE_PATH", str(tmp_path / "kill_switch_state.json"))
        monkeypatch.setattr(live_mod, "_STATE_DB_PATH", str(tmp_path / "live_state.db"))
        monkeypatch.setattr(live_mod, "_MEMORY_LOG_PATH", str(tmp_path / "decisions.jsonl"))

    def test_start_rejected_when_sector_map_overridden_to_empty(self, client):
        resp = client.post("/api/live/start", json={
            "broker": "alpaca_paper",
            "schedule": "hourly",
            "risk_overrides": {"sector_map": {}},
        })
        assert resp.status_code == 400, resp.text
        assert "sector_map" in resp.json()["detail"]
        assert client.get("/api/live/status").json()["state"] == "stopped"

    def test_start_succeeds_with_sector_map_from_live_yaml(self, client):
        # config/live.yaml ships a populated sector_map for its default
        # universe, so a normal start (no overrides) must not regress.
        resp = client.post("/api/live/start", json={
            "broker": "alpaca_paper",
            "schedule": "hourly",
        })
        assert resp.status_code == 200, resp.text
        client.post("/api/live/stop")


class TestHealthBrokerConnectivity:
    @pytest.fixture(autouse=True)
    def _mock_broker(self, monkeypatch, tmp_path):
        import firm.api.routers.live as live_mod
        from tests.test_brokers import MockBroker

        monkeypatch.setattr(live_mod, "_create_broker", lambda broker_type: MockBroker())
        monkeypatch.setattr(live_mod, "_APPROVALS_PATH", str(tmp_path / "approvals.json"))
        monkeypatch.setattr(live_mod, "_TRADE_HISTORY_ORDERS_PATH", str(tmp_path / "order_history.json"))
        monkeypatch.setattr(live_mod, "_TRADE_HISTORY_CYCLES_PATH", str(tmp_path / "cycle_history.json"))
        monkeypatch.setattr(live_mod, "_KILL_SWITCH_STATE_PATH", str(tmp_path / "kill_switch_state.json"))
        monkeypatch.setattr(live_mod, "_STATE_DB_PATH", str(tmp_path / "live_state.db"))
        monkeypatch.setattr(live_mod, "_MEMORY_LOG_PATH", str(tmp_path / "decisions.jsonl"))

    def test_health_reflects_ibkr_connectivity_when_running(self, client):
        client.post("/api/live/start", json={"broker": "ibkr_paper", "schedule": "hourly"})

        body = client.get("/api/health").json()
        assert body["status"] == "ok"  # process itself stays healthy either way
        assert body["broker"]["live_engine_running"] is True
        assert body["broker"]["type"] == "ibkr_paper"
        assert body["broker"]["connected"] is True

        engine = client.app.state.live_engine
        engine._broker._connected = False
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["broker"]["connected"] is False

        client.post("/api/live/stop")


# ------------------------------------------------------------------
# Server bind address (security: default to loopback-only, reachable
# only through a reverse proxy that adds TLS + auth)
# ------------------------------------------------------------------

class TestServerBindAddress:
    def test_defaults_to_loopback(self, monkeypatch):
        import firm.api.app as app_mod
        import uvicorn

        monkeypatch.delenv("FIRM_API_HOST", raising=False)
        captured = {}
        monkeypatch.setattr(uvicorn, "run", lambda app, host, port: captured.update(host=host, port=port))
        app_mod.run()
        assert captured["host"] == "127.0.0.1"
        assert captured["port"] == 8000

    def test_respects_env_override(self, monkeypatch):
        import firm.api.app as app_mod
        import uvicorn

        monkeypatch.setenv("FIRM_API_HOST", "0.0.0.0")
        captured = {}
        monkeypatch.setattr(uvicorn, "run", lambda app, host, port: captured.update(host=host, port=port))
        app_mod.run()
        assert captured["host"] == "0.0.0.0"

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

        # A synthetic 2-strategy/2-year backtest completes in ~5s when the
        # machine is otherwise idle, but JobManager serialises all runs
        # behind one lock and this suite's other tests (walk-forward folds,
        # HMM fits, etc.) can leave it CPU-starved when the full suite runs
        # under parallel/concurrent load — 120 * 0.5s = 60s was measured to
        # be too tight under that contention (pre-existing flake, tracked as
        # fix-preexisting-test-failures). 240 * 0.5s = 120s gives real
        # headroom while still returning immediately once the run finishes.
        terminal_statuses = {"completed", "failed"}
        status = "pending"
        for _ in range(240):
            time.sleep(0.5)
            detail = client.get(f"/api/runs/{run_id}").json()
            status = detail["status"]
            if status in terminal_statuses:
                break

        assert status == "completed", f"Run ended with status={status}, notes={detail.get('notes')}"
        assert detail["metrics"]
        return run_id

    def test_invalid_run_request_returns_422(self, client):
        """Regression: bad input is rejected up front, not run as a failed job."""
        # Inverted date range.
        r = client.post("/api/runs", json={
            "start_date": "2023-12-31", "end_date": "2020-01-01",
        })
        assert r.status_code == 422
        # Negative capital.
        r = client.post("/api/runs", json={"initial_capital": -5})
        assert r.status_code == 422
        # Malformed date.
        r = client.post("/api/runs", json={"start_date": "not-a-date"})
        assert r.status_code == 422

    def test_launch_and_poll(self, client):
        self._launch_and_wait(client)

    def test_cost_overrides_reach_the_stored_run_config(self, client):
        """Regression: spread_pct/short_borrow_annual_pct must flow from
        RunRequest through to the config actually handed to the backtest
        engine, not just live in the request schema."""
        r = client.post("/api/runs", json={
            "strategies": ["momentum"],
            "data_source": "synthetic",
            "spread_pct": 0.0009,
            "short_borrow_annual_pct": 0.05,
        })
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["config"]["spread_pct"] == 0.0009
        assert detail["config"]["short_borrow_annual_pct"] == 0.05

    def test_cost_overrides_default_to_settings_yaml_values(self, client):
        r = client.post("/api/runs", json={
            "strategies": ["momentum"], "data_source": "synthetic",
        })
        assert r.status_code == 200
        run_id = r.json()["run_id"]

        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["config"]["spread_pct"] == 0.0002
        assert detail["config"]["short_borrow_annual_pct"] == 0.003

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

    def test_walk_forward(self, client):
        r = client.post("/api/runs/walk_forward", json={
            "strategies": ["momentum"],
            "data_source": "synthetic",
            "start_date": "2022-01-01",
            "end_date": "2023-12-31",
            "seed": 42,
            "n_splits": 3,
            "train_pct": 0.7,
        })
        assert r.status_code == 200, r.text
        data = r.json()
        assert len(data["fold_ids"]) == 3
        assert data["aggregate"]["n_folds"] == 3
        assert "sharpe_ratio" in data["aggregate"]["metrics"]
        # Folds are surfaced as normal runs in the dashboard.
        listed = {run["run_id"] for run in client.get("/api/runs").json()}
        assert set(data["fold_ids"]) <= listed

    def test_walk_forward_rejects_bad_splits(self, client):
        r = client.post("/api/runs/walk_forward", json={"n_splits": 1})
        assert r.status_code == 422

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

    def test_clear_runs(self, client):
        self._launch_and_wait(client)
        self._launch_and_wait(client)
        assert len(client.get("/api/runs").json()) == 2

        r = client.delete("/api/runs")
        assert r.status_code == 200
        assert r.json() == {"cleared": 2}
        assert client.get("/api/runs").json() == []

    def test_clear_runs_when_empty(self, client):
        r = client.delete("/api/runs")
        assert r.status_code == 200
        assert r.json() == {"cleared": 0}


# ------------------------------------------------------------------
# Observability: Prometheus /metrics endpoint (Tier A)
# ------------------------------------------------------------------

class TestMetrics:
    def test_metrics_endpoint_exposes_prometheus(self, client):
        pytest.importorskip("prometheus_fastapi_instrumentator")
        # Generate some traffic so request metrics are populated.
        client.get("/api/meta/strategies")
        r = client.get("/metrics")
        assert r.status_code == 200
        body = r.text
        # Prometheus exposition format + the instrumentator's HTTP metric.
        assert "# HELP" in body
        assert "http_request" in body


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


# ------------------------------------------------------------------
# Live log tailing (frontend log monitor)
# ------------------------------------------------------------------

class TestLogsTail:
    @pytest.fixture(autouse=True)
    def _isolate_log_file(self, tmp_path, monkeypatch):
        import firm.api.routers.logs as logs_mod

        log_file = tmp_path / "api.log"
        monkeypatch.setattr(logs_mod, "_LOG_FILE", log_file)
        self.log_file = log_file

    def test_tail_missing_file_returns_empty(self, client):
        r = client.get("/api/logs/tail")
        assert r.status_code == 200
        data = r.json()
        assert data == {"lines": [], "next_offset": 0, "reset": False}

    def test_tail_reads_new_lines_and_advances_offset(self, client):
        self.log_file.write_text(
            '{"ts": "2026-01-01T00:00:00+00:00", "level": "INFO", "logger": "firm.x", "msg": "hello"}\n'
        )
        r = client.get("/api/logs/tail")
        assert r.status_code == 200
        data = r.json()
        assert len(data["lines"]) == 1
        assert data["lines"][0]["msg"] == "hello"
        assert data["reset"] is False
        assert data["next_offset"] > 0

        # No new content since the returned offset -> nothing new.
        r2 = client.get(f"/api/logs/tail?offset={data['next_offset']}")
        assert r2.json()["lines"] == []

        # Appending a second line surfaces only the new one.
        with self.log_file.open("a") as f:
            f.write('{"ts": "2026-01-01T00:00:01+00:00", "level": "WARNING", "logger": "firm.y", "msg": "world"}\n')
        r3 = client.get(f"/api/logs/tail?offset={data['next_offset']}")
        data3 = r3.json()
        assert len(data3["lines"]) == 1
        assert data3["lines"][0]["msg"] == "world"
        assert data3["lines"][0]["level"] == "WARNING"

    def test_tail_handles_malformed_line(self, client):
        self.log_file.write_text("not json at all\n")
        r = client.get("/api/logs/tail")
        data = r.json()
        assert len(data["lines"]) == 1
        assert data["lines"][0]["level"] == "RAW"
        assert data["lines"][0]["msg"] == "not json at all"

    def test_tail_offset_past_end_resets(self, client):
        self.log_file.write_text('{"ts": null, "level": "INFO", "logger": "x", "msg": "a"}\n')
        r = client.get("/api/logs/tail?offset=999999")
        data = r.json()
        assert data["reset"] is True
        assert len(data["lines"]) == 1


# ------------------------------------------------------------------
# Decision memory API (frontend Decisions page)
# ------------------------------------------------------------------

class TestMemoryDecisionsAPI:
    @pytest.fixture(autouse=True)
    def _isolate_memory_path(self, tmp_path, monkeypatch):
        import firm.agents.memory as memory_mod
        monkeypatch.setattr(memory_mod, "_DEFAULT_PATH", tmp_path / "decisions.jsonl")
        self.memory_path = tmp_path / "decisions.jsonl"

    def test_no_decisions_yet(self, client):
        r = client.get("/api/memory/decisions")
        assert r.status_code == 200
        assert r.json() == []

    def test_lists_decisions_newest_first(self, client):
        from firm.agents.memory import TradingMemoryLog

        log = TradingMemoryLog()
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.05}, nav_at_decision=100_000)
        log.store_decision(date="2026-01-02", proposal_weights={"MSFT": 0.05}, nav_at_decision=101_000)

        r = client.get("/api/memory/decisions")
        data = r.json()
        assert len(data) == 2
        assert data[0]["date"] == "2026-01-02"
        assert data[1]["date"] == "2026-01-01"
        assert data[0]["nav_at_decision"] == 101_000

    def test_limit_param(self, client):
        from firm.agents.memory import TradingMemoryLog

        log = TradingMemoryLog()
        for i in range(5):
            log.store_decision(date=f"2026-01-0{i+1}", proposal_weights={"AAPL": 0.05})

        r = client.get("/api/memory/decisions?limit=2")
        assert len(r.json()) == 2

    def test_lessons_empty(self, client):
        r = client.get("/api/memory/lessons")
        assert r.status_code == 200
        assert r.json() == {
            "total": 0,
            "counts": {"correct": 0, "incorrect": 0, "partial": 0, "unknown": 0},
            "recent_lessons": [],
        }

    def test_lessons_aggregates_reflected_decisions(self, client):
        from unittest.mock import MagicMock

        from firm.agents.memory import TradingMemoryLog

        log = TradingMemoryLog()
        log.store_decision(date="2026-01-01", proposal_weights={"AAPL": 0.05})
        llm = MagicMock()
        llm.chat_json.return_value = {
            "verdict": "correct", "what_worked": "x", "what_failed": "",
            "lesson": "size up on confirmed regime",
        }
        log.reflect(date="2026-01-01", raw_return=0.02, benchmark_return=0.01, llm_service=llm)

        r = client.get("/api/memory/lessons")
        data = r.json()
        assert data["total"] == 1
        assert data["counts"]["correct"] == 1
        assert data["recent_lessons"] == ["size up on confirmed regime"]


# ------------------------------------------------------------------
# Live config/start round-trip (regression: strategies/risk/schedule
# used to be silently dropped by PUT /live/config, and /live/start
# never threaded strategies/risk through at all)
# ------------------------------------------------------------------

class TestLiveConfigRoundTrip:
    @pytest.fixture(autouse=True)
    def _mock_broker(self, monkeypatch, tmp_path):
        import firm.api.routers.live as live_mod
        from tests.test_brokers import MockBroker

        monkeypatch.setattr(live_mod, "_create_broker", lambda broker_type: MockBroker())
        # Never let tests read/write the real production approvals file.
        monkeypatch.setattr(live_mod, "_APPROVALS_PATH", str(tmp_path / "approvals.json"))
        monkeypatch.setattr(live_mod, "_TRADE_HISTORY_ORDERS_PATH", str(tmp_path / "order_history.json"))
        monkeypatch.setattr(live_mod, "_TRADE_HISTORY_CYCLES_PATH", str(tmp_path / "cycle_history.json"))
        monkeypatch.setattr(live_mod, "_KILL_SWITCH_STATE_PATH", str(tmp_path / "kill_switch_state.json"))
        monkeypatch.setattr(live_mod, "_STATE_DB_PATH", str(tmp_path / "live_state.db"))
        monkeypatch.setattr(live_mod, "_MEMORY_LOG_PATH", str(tmp_path / "decisions.jsonl"))

    def test_start_applies_strategies_and_risk(self, client):
        resp = client.post("/api/live/start", json={
            "broker": "alpaca_paper",
            "schedule": "hourly",
            "symbols": ["AAPL", "MSFT"],
            "strategies": ["momentum"],
            "kill_switch_drawdown": 0.2,
            "max_daily_trades": 10,
            "max_daily_turnover": 0.3,
        })
        assert resp.status_code == 200, resp.text

        cfg = client.get("/api/live/config").json()
        assert cfg["broker"] == "alpaca_paper"
        assert cfg["strategies"]["enabled"] == ["momentum"]
        assert cfg["risk"]["kill_switch_drawdown"] == 0.2
        assert cfg["risk"]["max_daily_trades"] == 10
        assert cfg["risk"]["max_daily_turnover"] == 0.3
        assert cfg["universe"]["symbols"] == ["AAPL", "MSFT"]

        status = client.get("/api/live/status").json()
        assert status["active_strategies"] == ["momentum"]
        assert status["broker"] == "alpaca_paper"

    def test_start_applies_deep_risk_overrides(self, client):
        resp = client.post("/api/live/start", json={
            "broker": "alpaca_paper",
            "schedule": "hourly",
            "risk_overrides": {"max_position_pct": 0.05, "max_gross_exposure": 1.5},
        })
        assert resp.status_code == 200, resp.text

        engine = client.app.state.live_engine
        assert engine._config["max_position_pct"] == 0.05
        assert engine._config["max_gross_exposure"] == 1.5

        client.post("/api/live/stop")

        client.post("/api/live/stop")

    def test_put_config_updates_strategies_autoapprove_risk_universe(self, client):
        client.post("/api/live/start", json={"broker": "alpaca_paper", "schedule": "hourly"})

        resp = client.put("/api/live/config", json={
            "approval_mode": "full_auto",
            "strategies": {"enabled": ["trend", "momentum"], "auto_approve": ["trend"]},
            "risk": {"kill_switch_drawdown": 0.15, "max_daily_trades": 5, "max_daily_turnover": 0.25},
            "universe": {"symbols": ["GOOG"]},
        })
        assert resp.status_code == 200, resp.text

        cfg = client.get("/api/live/config").json()
        assert set(cfg["strategies"]["enabled"]) == {"trend", "momentum"}
        assert cfg["strategies"]["auto_approve"] == ["trend"]
        assert cfg["risk"]["kill_switch_drawdown"] == 0.15
        assert cfg["risk"]["max_daily_trades"] == 5
        assert cfg["risk"]["max_daily_turnover"] == 0.25
        assert cfg["universe"]["symbols"] == ["GOOG"]

        client.post("/api/live/stop")

    def test_put_config_schedule_restarts_scheduler(self, client):
        client.post("/api/live/start", json={"broker": "alpaca_paper", "schedule": "hourly"})
        resp = client.put("/api/live/config", json={"schedule": "market_close"})
        assert resp.status_code == 200, resp.text

        cfg = client.get("/api/live/config").json()
        assert cfg["schedule"] == "market_close"

        client.post("/api/live/stop")

    def test_trigger_skips_when_market_closed_unless_forced(self, client, monkeypatch):
        import pandas as pd
        from unittest.mock import MagicMock
        import firm.data.providers.fallback as fallback_mod
        import firm.live.engine as engine_mod

        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], None)
        monkeypatch.setattr(engine_mod, "build_orchestrator", lambda config: mock_orch)

        # /live/start builds a real FallbackProvider, which would otherwise
        # make real network calls to Massive/Tiingo/etc. on force=true.
        mock_provider = MagicMock()
        mock_provider.get_prices.return_value = pd.DataFrame()
        mock_provider.get_fundamentals.return_value = pd.DataFrame()
        mock_provider.get_news_sentiment.return_value = pd.DataFrame()
        monkeypatch.setattr(fallback_mod, "FallbackProvider", lambda *a, **k: mock_provider)

        client.post("/api/live/start", json={"broker": "alpaca_paper", "schedule": "hourly"})
        engine = client.app.state.live_engine
        engine._broker._market_open = False

        resp = client.post("/api/live/trigger?sync=true")
        assert resp.json()["skipped"] is True
        mock_orch.step.assert_not_called()

        resp = client.post("/api/live/trigger?sync=true&force=true")
        assert resp.json()["skipped"] is False

        client.post("/api/live/stop")


# ------------------------------------------------------------------
# Clearing approvals / cycle (order) history
# ------------------------------------------------------------------

class TestLiveClearEndpoints:
    @pytest.fixture(autouse=True)
    def _mock_broker(self, monkeypatch, tmp_path):
        import firm.api.routers.live as live_mod
        from tests.test_brokers import MockBroker

        monkeypatch.setattr(live_mod, "_create_broker", lambda broker_type: MockBroker())
        # Never let tests read/write the real production approvals file.
        monkeypatch.setattr(live_mod, "_APPROVALS_PATH", str(tmp_path / "approvals.json"))
        monkeypatch.setattr(live_mod, "_TRADE_HISTORY_ORDERS_PATH", str(tmp_path / "order_history.json"))
        monkeypatch.setattr(live_mod, "_TRADE_HISTORY_CYCLES_PATH", str(tmp_path / "cycle_history.json"))
        monkeypatch.setattr(live_mod, "_KILL_SWITCH_STATE_PATH", str(tmp_path / "kill_switch_state.json"))
        monkeypatch.setattr(live_mod, "_STATE_DB_PATH", str(tmp_path / "live_state.db"))
        monkeypatch.setattr(live_mod, "_MEMORY_LOG_PATH", str(tmp_path / "decisions.jsonl"))

    def _mock_cycle_deps(self, monkeypatch):
        """run_cycle() otherwise builds a real orchestrator + FallbackProvider
        that make real network calls — mock both like test_trigger_* does."""
        import pandas as pd
        from unittest.mock import MagicMock
        import firm.data.providers.fallback as fallback_mod
        import firm.live.engine as engine_mod

        mock_orch = MagicMock()
        mock_orch.step.return_value = ([], None)
        monkeypatch.setattr(engine_mod, "build_orchestrator", lambda config: mock_orch)

        mock_provider = MagicMock()
        mock_provider.get_prices.return_value = pd.DataFrame()
        mock_provider.get_fundamentals.return_value = pd.DataFrame()
        mock_provider.get_news_sentiment.return_value = pd.DataFrame()
        monkeypatch.setattr(fallback_mod, "FallbackProvider", lambda *a, **k: mock_provider)

    def test_clear_cycles_wipes_history(self, client, monkeypatch):
        self._mock_cycle_deps(monkeypatch)
        client.post("/api/live/start", json={"broker": "alpaca_paper", "schedule": "hourly"})
        engine = client.app.state.live_engine
        engine.run_cycle(force=True)
        engine.run_cycle(force=True)
        assert len(client.get("/api/live/cycles").json()) == 2

        r = client.delete("/api/live/cycles")
        assert r.status_code == 200
        assert r.json() == {"cleared": 2}
        assert client.get("/api/live/cycles").json() == []
        assert client.get("/api/live/orders").json() == []

        client.post("/api/live/stop")

    def test_clear_cycles_no_engine_returns_zero(self, client):
        r = client.delete("/api/live/cycles")
        assert r.status_code == 200
        assert r.json() == {"cleared": 0}

    def test_clear_approvals_wipes_queue(self, client):
        client.post("/api/live/start", json={"broker": "alpaca_paper", "schedule": "hourly"})
        queue = client.app.state.approval_queue
        queue.add(orders=[{"symbol": "AAPL", "side": "buy", "quantity": 1}], blackboard=None)

        assert len(client.get("/api/live/approvals").json()) == 1

        r = client.delete("/api/live/approvals")
        assert r.status_code == 200
        assert r.json() == {"cleared": 1}
        assert client.get("/api/live/approvals").json() == []

        client.post("/api/live/stop")

    def test_clear_approvals_no_queue_returns_zero(self, client):
        r = client.delete("/api/live/approvals")
        assert r.status_code == 200
        assert r.json() == {"cleared": 0}

    def test_list_approvals_includes_resolved_history(self, client):
        client.post("/api/live/start", json={"broker": "alpaca_paper", "schedule": "hourly"})
        queue = client.app.state.approval_queue
        aid = queue.add(orders=[{"symbol": "AAPL", "side": "buy", "quantity": 1}], blackboard=None)
        queue.reject(aid, "test reject")

        resp = client.get("/api/live/approvals")
        assert resp.status_code == 200
        rows = resp.json()
        assert len(rows) == 1
        assert rows[0]["status"] == "rejected"

        client.post("/api/live/stop")

    def test_orders_persist_after_stop(self, client, monkeypatch):
        from tests.test_brokers import MockBroker

        self._mock_cycle_deps(monkeypatch)
        mock_broker = MockBroker()
        import firm.api.routers.live as live_mod

        monkeypatch.setattr(live_mod, "_create_broker", lambda broker_type: mock_broker)

        client.post("/api/live/start", json={
            "broker": "alpaca_paper",
            "schedule": "hourly",
            "approval_mode": "full_auto",
            "initial_capital": 100_000,
        })
        engine = client.app.state.live_engine
        mock_orch = engine._orchestrator

        mock_orch.step.return_value = (
            [{"symbol": "AAPL", "side": "buy", "quantity": 1, "strategy": "momentum"}],
            None,
        )
        engine.run_cycle(force=True)

        client.post("/api/live/stop")

        orders = client.get("/api/live/orders").json()
        assert len(orders) >= 1
        assert orders[0]["symbol"] == "AAPL"


class TestKillSwitchResetEndpoint:
    @pytest.fixture(autouse=True)
    def _mock_broker(self, monkeypatch, tmp_path):
        import firm.api.routers.live as live_mod
        from tests.test_brokers import MockBroker

        monkeypatch.setattr(live_mod, "_create_broker", lambda broker_type: MockBroker(initial_cash=50_000))
        monkeypatch.setattr(live_mod, "_APPROVALS_PATH", str(tmp_path / "approvals.json"))
        monkeypatch.setattr(live_mod, "_TRADE_HISTORY_ORDERS_PATH", str(tmp_path / "order_history.json"))
        monkeypatch.setattr(live_mod, "_TRADE_HISTORY_CYCLES_PATH", str(tmp_path / "cycle_history.json"))
        monkeypatch.setattr(live_mod, "_KILL_SWITCH_STATE_PATH", str(tmp_path / "kill_switch_state.json"))
        monkeypatch.setattr(live_mod, "_STATE_DB_PATH", str(tmp_path / "live_state.db"))
        monkeypatch.setattr(live_mod, "_MEMORY_LOG_PATH", str(tmp_path / "decisions.jsonl"))

    def _mock_cycle_deps(self, monkeypatch):
        from unittest.mock import MagicMock

        import pandas as pd

        import firm.data.providers.fallback as fallback_mod
        import firm.live.engine as engine_mod

        mock_orch = MagicMock()
        mock_orch.step.return_value = (
            [{"symbol": "AAPL", "side": "buy", "quantity": 1, "strategy": "momentum"}],
            None,
        )
        monkeypatch.setattr(engine_mod, "build_orchestrator", lambda config: mock_orch)

        mock_provider = MagicMock()
        mock_provider.get_prices.return_value = pd.DataFrame()
        mock_provider.get_fundamentals.return_value = pd.DataFrame()
        mock_provider.get_news_sentiment.return_value = pd.DataFrame()
        monkeypatch.setattr(fallback_mod, "FallbackProvider", lambda *a, **k: mock_provider)

    def test_reset_no_engine_returns_400(self, client):
        resp = client.post("/api/live/kill-switch/reset")
        assert resp.status_code == 400

    def test_reset_when_not_halted_is_a_noop(self, client, monkeypatch):
        self._mock_cycle_deps(monkeypatch)
        client.post("/api/live/start", json={
            "broker": "alpaca_paper", "schedule": "hourly",
            "approval_mode": "full_auto", "initial_capital": 100_000,
        })
        resp = client.post("/api/live/kill-switch/reset")
        assert resp.status_code == 200
        body = resp.json()
        assert body["reset"] is False
        assert body["halted"] is False
        client.post("/api/live/stop")

    def test_reset_after_trip_rearms_and_persists(self, client, monkeypatch, tmp_path):
        self._mock_cycle_deps(monkeypatch)
        client.post("/api/live/start", json={
            "broker": "alpaca_paper", "schedule": "hourly",
            "approval_mode": "full_auto", "initial_capital": 100_000,
            "kill_switch_drawdown": 0.1,
        })
        engine = client.app.state.live_engine
        engine.run_cycle(force=True)
        assert engine.halted is True
        assert client.get("/api/live/alerts").json()["halted"] is True

        resp = client.post("/api/live/kill-switch/reset")
        assert resp.status_code == 200
        body = resp.json()
        assert body["reset"] is True
        assert body["halted"] is False
        assert client.get("/api/live/alerts").json()["halted"] is False

        state_path = tmp_path / "kill_switch_state.json"
        assert state_path.exists()
        assert json.loads(state_path.read_text())["halted"] is False

        client.post("/api/live/stop")
