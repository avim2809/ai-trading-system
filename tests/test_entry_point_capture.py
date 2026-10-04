"""P1-12: every backtest entry point leaves exactly one capture record (inbox line or ledger row)."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from firm.backtest import _capture_state as cs
from firm.backtest.run import execute_backtest

SRC = Path(__file__).resolve().parents[1] / "src"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

CFG = {
    "data_source": "synthetic", "start_date": "2020-01-01", "end_date": "2020-06-30",
    "universe_symbols": ["AAPL", "MSFT"], "strategies": ["momentum"], "seed": 7,
    "initial_capital": 100000,
}


@pytest.fixture(autouse=True)
def _reset_warned():
    cs._warned.clear()
    yield
    cs._warned.clear()


def _inbox_lines(root: Path | None = None) -> list[dict]:
    root = root or Path(os.environ["FIRM_RESEARCH_LEDGER_ROOT"])
    out: list[dict] = []
    for f in sorted((root / "inbox").glob("*.jsonl")):
        out += [json.loads(x) for x in f.read_text().splitlines() if x]
    return out


def _child_env(tmp_path: Path) -> dict:
    root = tmp_path / "ledger"
    (root / "inbox").mkdir(parents=True, exist_ok=True)
    (root / "returns").mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PYTEST_CURRENT_TEST", "FIRM_AUTO_START_LIVE")}
    env.update(PYTHONPATH=str(SRC), FIRM_DATA_DIR=str(tmp_path / "data"), FIRM_RESEARCH_LEDGER_ROOT=str(root),
               PYTHONDONTWRITEBYTECODE="1")
    return env


def _git_repo(path: Path, dirty: bool) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    run = lambda *a: subprocess.run(["git", "-c", "safe.directory=*", *a], cwd=path, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    (path / "a.txt").write_text("x")
    run("add", "a.txt")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    if dirty:
        (path / "a.txt").write_text("changed")
    return path


def test_execute_backtest_synthetic_records_one_unregistered_row(tmp_path):
    repo = _git_repo(tmp_path / "repo", dirty=False)
    from firm.research import capture, ledger

    with patch.dict(os.environ, {"FIRM_LEDGER_REPO_DIR": str(repo)}), \
         patch.object(cs, "LEDGER_SINK", capture.record_unregistered):
        report = execute_backtest(dict(CFG))
    df = ledger.trials()
    assert len(df) == 1
    row = df.iloc[0]
    assert row["mode"] == "unregistered" and row["status"] == "completed"
    assert row["family"] == "unregistered:execute_backtest" and row["seed"] == 7
    assert row["preregistration_id"] is None
    assert row["returns_path"] and row["n_obs"] == len(report.returns.dropna())
    assert _inbox_lines() == []  # armed path does not use the inbox


def test_experiment_runner_records_once_not_twice(tmp_path):
    from firm.experiments.registry import RunRegistry
    from firm.experiments.runner import ExperimentRunner

    runner = ExperimentRunner(registry=RunRegistry(base_dir=str(tmp_path / "runs")))
    cfg = {"name": "t", "backtest": {k: CFG[k] for k in ("start_date", "end_date", "initial_capital")},
           "strategies": {"enabled": ["momentum"]}, "universe_symbols": ["AAPL", "MSFT"], "seed": 42}
    runner.run(cfg, seed=42)
    lines = _inbox_lines()
    assert [x["entry"] for x in lines] == ["ExperimentRunner.run"]


def test_walk_forward_rows_n_folds_plus_n_times_k_candidates(tmp_path):
    from firm.experiments.registry import RunRegistry
    from firm.experiments.runner import ExperimentRunner

    runner = ExperimentRunner(registry=RunRegistry(base_dir=str(tmp_path / "runs")))
    cfg = {"name": "wf", "backtest": {"start_date": "2020-01-01", "end_date": "2021-12-31", "initial_capital": 100000},
           "strategies": {"enabled": ["momentum"]}, "universe_symbols": ["AAPL", "MSFT"], "seed": 42}
    n_folds, k = 2, 2
    runner.run_walk_forward(cfg, n_splits=n_folds, train_pct=0.7,
                            param_grid=[{"tag": "a"}, {"tag": "b"}])
    lines = _inbox_lines()
    folds = [x for x in lines if x["entry"] == "ExperimentRunner.run"]
    trains = [x for x in lines if x["entry"] == "execute_backtest"]
    assert len(folds) == n_folds and all("_walk_forward" in x["config"] for x in folds)
    assert len(trains) == n_folds * k
    assert len(lines) == n_folds + n_folds * k  # nothing deduped away


def test_importing_api_app_does_not_mark_api_process(tmp_path):
    code = (
        "import firm.api.app as a\n"
        "from firm.backtest import _capture_state as cs\n"
        "assert cs.IN_API_PROCESS is False\n"
        "from fastapi.testclient import TestClient\n"
        "with TestClient(a.app):\n"
        "    assert cs.IN_API_PROCESS is True\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=_child_env(tmp_path), cwd=tmp_path,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


_UNARMED_CHILD = """
import sys
{pre}
from firm.backtest.run import execute_backtest
execute_backtest({cfg!r})
assert 'firm.research' not in sys.modules, 'firm.research imported by the unarmed hook'
"""


def test_unarmed_backtest_never_imports_firm_research(tmp_path):
    env = _child_env(tmp_path)
    r = subprocess.run([sys.executable, "-c", _UNARMED_CHILD.format(pre="", cfg=CFG)], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    lines = _inbox_lines(tmp_path / "ledger")
    assert len(lines) == 1 and lines[0]["entry"] == "execute_backtest"
    assert lines[0]["tag"].startswith("proc-") and lines[0]["source"] == "process"
    assert not (tmp_path / "ledger" / "trials.jsonl").exists()


def test_api_mode_writes_inbox_line_and_does_not_import_firm_research(tmp_path):
    env = _child_env(tmp_path)
    env["FIRM_API_PORT"] = "8123"
    pre = "from firm.backtest import _capture_state as cs; cs.mark_api_process()"
    r = subprocess.run([sys.executable, "-c", _UNARMED_CHILD.format(pre=pre, cfg=CFG)], env=env, cwd=tmp_path,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    f = tmp_path / "ledger" / "inbox" / "api-8123.jsonl"
    assert len(f.read_text().splitlines()) == 1
    assert json.loads(f.read_text())["source"] == "api"


def test_dirty_tree_capture_still_writes_row(tmp_path):
    repo = _git_repo(tmp_path / "repo", dirty=True)
    from firm.research import capture, ledger

    with patch.dict(os.environ, {"FIRM_LEDGER_REPO_DIR": str(repo)}), \
         patch.object(cs, "LEDGER_SINK", capture.record_unregistered):
        execute_backtest(dict(CFG))
    df = ledger.trials()
    assert len(df) == 1 and df.iloc[0]["code_commit"].endswith("+dirty")
    assert df.iloc[0]["mode"] == "unregistered"


def test_missing_inbox_warns_once_and_backtest_unaffected(tmp_path, caplog):
    with patch.dict(os.environ, {"FIRM_RESEARCH_LEDGER_ROOT": str(tmp_path / "nope")}), \
         caplog.at_level(logging.WARNING, logger=cs.log.name):
        r1 = execute_backtest(dict(CFG))
        r2 = execute_backtest(dict(CFG))
    assert not r1.returns.empty and r1.portfolio_summary() == r2.portfolio_summary()
    warns = [x for x in caplog.records if "inbox" in x.getMessage() and x.levelno == logging.WARNING]
    assert len(warns) == 1
    assert not (tmp_path / "nope").exists()  # the hook never creates directories


def test_run_backtest_console_script_captured(tmp_path):
    from firm.config import Settings
    from firm.scripts import run_backtest as rb

    dates = pd.date_range("2020-01-01", "2020-06-01", freq="B")
    prices = pd.DataFrame({"date": dates, "symbol": "AAPL", "open": 1.0, "high": 1.0, "low": 1.0,
                           "close": 1.0, "volume": 1.0, "adj_close": 1.0})
    settings = Settings()
    settings.data.cache_dir = str(tmp_path)
    settings.backtest.start_date, settings.backtest.end_date = "2020-03-01", "2020-06-01"

    class Eng:
        def __init__(self, c): ...
        def setup(self, *a): ...
        def run(self): ...
        def generate_report(self):
            rep = MagicMock()
            rep.to_text.return_value = ""
            return rep

    with patch.object(sys, "argv", ["run-backtest", "--config", "config/settings.yaml",
                                    "--output-dir", str(tmp_path / "runs")]), \
         patch("firm.scripts.run_backtest.get_settings", return_value=settings), \
         patch("firm.scripts.run_backtest.load_prices", return_value=prices), \
         patch("firm.scripts.run_backtest.load_fundamentals", return_value=None), \
         patch("firm.scripts.run_backtest.build_orchestrator", return_value=MagicMock()), \
         patch("firm.backtest.engine.BacktestEngine", Eng), patch("builtins.print"):
        rb.main()
    assert [x["entry"] for x in _inbox_lines()] == ["run_backtest.main"]


def test_sync_ingests_inbox_exactly_once(tmp_path):
    repo = _git_repo(tmp_path / "repo", dirty=False)
    root = Path(os.environ["FIRM_RESEARCH_LEDGER_ROOT"])
    for _ in range(3):
        execute_backtest({**CFG, "end_date": "2020-03-31"})
    assert len(_inbox_lines()) == 3
    import importlib.util

    spec = importlib.util.spec_from_file_location("sync_ledger_mirror", SCRIPTS / "sync_ledger_mirror.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from firm.research import ledger

    with patch.dict(os.environ, {"FIRM_LEDGER_REPO_DIR": str(repo)}):
        assert mod.ingest_inbox() == 3
        assert mod.ingest_inbox() == 0
    df = ledger.trials(mode="unregistered")
    assert len(df) == 3 and sorted(df["source_entry_index"].astype(int)) == [0, 1, 2]
    assert set(df["source_file"]) == {f"inbox/{p.name}" for p in (root / "inbox").glob("*.jsonl")}
    assert ledger.verify_chain().ok


def test_failed_backtest_recorded_and_raises():
    with pytest.raises(Exception) as ei:
        execute_backtest({**CFG, "start_date": "not-a-date"})
    lines = _inbox_lines()
    assert len(lines) == 1 and lines[0]["status"] == "failed"
    assert lines[0]["error"] == repr(ei.value)


def test_capture_failure_never_breaks_backtest(tmp_path, caplog):
    from firm.research import capture

    def boom(*a, **k):
        raise RuntimeError("ledger down")

    repo = _git_repo(tmp_path / "repo", dirty=False)
    with patch.dict(os.environ, {"FIRM_LEDGER_REPO_DIR": str(repo)}), \
         patch.object(cs, "LEDGER_SINK", boom), caplog.at_level(logging.WARNING):
        report = execute_backtest(dict(CFG))
    assert not report.returns.empty
    assert any("ledger sink failed" in x.getMessage() for x in caplog.records)
    del capture


def test_nested_entry_points_record_only_outermost():
    with cs.capture_run("outer", {"a": 1}, 1) as o:
        with cs.capture_run("inner", {"b": 2}, 2):
            pass
        del o
    assert [x["entry"] for x in _inbox_lines()] == ["outer"]
    assert cs._DEPTH.get() == 0


def test_default_behaviour_bit_identical():
    with_hook = execute_backtest(dict(CFG))
    from firm.backtest import run as run_mod

    no_hook = run_mod._execute_backtest_impl(dict(CFG))
    pd.testing.assert_series_equal(with_hook.returns, no_hook.returns)
    assert with_hook.portfolio_summary() == no_hook.portfolio_summary()
