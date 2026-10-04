"""Concurrency / crash behaviour of the host ledger (P1-01)."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import signal
import time

from firm.research import ledger as L


def _writer(root: str, n: int, tag: str, big: int = 0) -> None:
    os.environ[L.LEDGER_ROOT_ENV] = root
    for i in range(n):
        cfg = {"tag": tag, "i": i, "pad": "x" * big}
        rec = L.TrialRecord(
            trial_id=os.urandom(16).hex(), family="f", mode="legacy", config=cfg,
            config_hash=L.config_hash(cfg), code_commit="", data_snapshot_id=None, seed=None,
            start=None, end=None, returns_path=None, gross_sharpe=None, net_sharpe=None,
            periods_per_year=None, sharpe_conversion=None, n_obs=None, skew=None, kurt=None,
            preregistration_id=None, touched_holdout=False, status="completed", error=None,
        )
        L.record_trial(rec)


def test_two_process_chain_cannot_fork(tmp_path, monkeypatch):
    root = tmp_path / "led"
    root.mkdir()
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(root))
    ctx = mp.get_context("spawn")
    ps = [ctx.Process(target=_writer, args=(str(root), 200, t)) for t in ("a", "b")]
    for p in ps:
        p.start()
    for p in ps:
        p.join(timeout=300)
        assert p.exitcode == 0
    rep = L.verify_chain()
    assert rep.ok and rep.n_rows == 400
    rows = [json.loads(x) for x in (root / "trials.jsonl").read_text().splitlines()]
    assert [r["seq"] for r in rows] == list(range(400))
    assert all(rows[i]["prev_hash"] == rows[i - 1]["row_hash"] for i in range(1, 400))
    tags = [r["row"]["config"]["tag"] for r in rows]
    assert tags.count("a") == 200 and tags.count("b") == 200


def test_kill_midwrite(tmp_path, monkeypatch):
    root = tmp_path / "led"
    root.mkdir()
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(root))
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_writer, args=(str(root), 100000, "k", 20000))
    p.start()
    f = root / "trials.jsonl"
    deadline = time.time() + 60
    while time.time() < deadline and not (f.exists() and f.stat().st_size > 2_000_000):
        time.sleep(0.05)
    os.kill(p.pid, signal.SIGKILL)
    p.join()
    rep = L.verify_chain()
    assert rep.ok and rep.n_rows > 0
    if rep.partial_tail:
        import pytest

        with pytest.raises(L.LedgerCorruptError):
            _writer(str(root), 1, "after")
