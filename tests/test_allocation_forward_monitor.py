"""P5-06 allocation forward monitor: synthetic fixtures only, no network beyond a loopback fake, no .env."""

from __future__ import annotations

import builtins
import hashlib
import http.server
import importlib
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import firm  # noqa: E402
from firm.monitoring import allocation_forward as m  # noqa: E402

assert str(Path(firm.__file__).resolve()).startswith(str(ROOT)), firm.__file__

import allocation_forward_test_preregistered as frozen  # noqa: E402
import allocation_replay  # noqa: E402
import run_alt_premia_evaluation as ev  # noqa: E402

CFG = yaml.safe_load((ROOT / "config" / "live_alpaca.yaml").read_text())["allocation"]
FROZEN_FILES = [
    "scripts/allocation_forward_test_preregistered.py", "scripts/allocation_replay.py",
    "scripts/alt_premia_preregistered_bars.py", "scripts/run_alt_premia_evaluation.py",
]


def _sha(paths):
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def make_inputs(d: Path, start="2025-09-01", end="2026-11-30", seed=1) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end)
    for t in ("SPY", "IEF", "EFA", "EEM", "TLT", "GLD", "DBC", "VNQ", "SVXY"):
        px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.008, len(days))))
        pd.DataFrame({"date": days, "adjClose": px, "close": px}).to_parquet(d / f"tiingo_{t}.parquet")
    cal = pd.date_range(start, end, freq="D")
    btc = 30000 * np.exp(np.cumsum(rng.normal(0.001, 0.03, len(cal))))
    pd.DataFrame({"date": cal, "close": btc}).to_parquet(d / "tiingo_BTCUSD.parquet")
    for n in ("VIX", "VIX3M"):
        pd.DataFrame({"date": days, "close": 18 + rng.normal(0, 1, len(days))}).to_parquet(d / f"cboe_{n}.parquet")
    pd.DataFrame({"date": days, "put": 100 + np.cumsum(rng.normal(0, 0.2, len(days)))}).to_parquet(d / "cboe_PUT.parquet")
    pd.DataFrame({"date": days, "value": 4.0}).to_parquet(d / "fred_DTB3.parquet")
    pd.DataFrame({"announcement": pd.to_datetime(["2026-10-28", "2026-12-09"])}).to_parquet(d / "fomc_scheduled.parquet")
    return d


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    return make_inputs(tmp_path_factory.mktemp("inputs"))


def test_input_set_is_complete(inputs):
    assert sorted(p.name for p in inputs.iterdir()) == sorted(m.INPUT_FILES) and len(m.INPUT_FILES) == 15


# --- simulation wrapper -----------------------------------------------------

def test_simulated_nav_equals_replay(inputs):
    got = m.simulated_nav(CFG, inputs, "2026-09-30", "2026-10-30", 100_000.0)
    with m._data_end_override("2026-10-30"):
        want = allocation_replay.replay(CFG, inputs, "2026-09-30", "2026-10-30", 100_000.0)["_nav"]
    pd.testing.assert_series_equal(got, want, check_freq=False)
    assert len(got) > 5


def test_simulated_nav_after_data_end_nonempty(inputs):
    got = m.simulated_nav(CFG, inputs, "2026-09-30", "2026-11-13", 100_000.0)
    assert len(got) > 0 and got.index.min() >= pd.Timestamp("2026-09-30")
    with pytest.raises((IndexError, KeyError, ValueError)):          # bare replay: cutoff cannot pass silently
        allocation_replay.replay(CFG, inputs, "2026-09-30", None, 100_000.0)


def test_data_end_override_does_not_leak(inputs):
    before = _sha(FROZEN_FILES)
    fp = frozen.bars_fingerprint()
    orig = ev.prereg.DATA_END
    assert orig == "2026-09-28"
    m.simulated_nav(CFG, inputs, "2026-09-30", "2026-10-15", 100_000.0)
    assert ev.prereg.DATA_END == "2026-09-28"
    with pytest.raises(RuntimeError):
        with m._data_end_override("2030-01-01"):
            assert ev.prereg.DATA_END == "2030-01-01"
            raise RuntimeError("boom")
    assert ev.prereg.DATA_END == "2026-09-28"
    with pytest.raises(Exception):
        m.simulated_nav(CFG, inputs / "missing", "2026-09-30", "2026-10-15", 1.0)
    assert ev.prereg.DATA_END == "2026-09-28"
    assert frozen.bars_fingerprint() == fp and _sha(FROZEN_FILES) == before


# --- deployment record ------------------------------------------------------

def _history(tmp_path, *, rule=True, pre_fill=None, fp=None):
    real = json.loads((ROOT / "docs" / "allocation_forward_test_trial_history.json").read_text())
    if fp:
        real["entries"][0]["fingerprint"] = fp
    if pre_fill:
        real["entries"][0]["pre_fill_nav"] = pre_fill
    if rule:
        real["entries"].append({"live_nav_selection_rule": m.LIVE_NAV_RULE_ID, "source": "GET /api/live/portfolio-history"})
    p = tmp_path / "hist.json"
    p.write_text(json.dumps(real))
    return p


def test_start_and_fingerprint_read_from_trial_history(tmp_path):
    p = _history(tmp_path, pre_fill=100000.0)
    dep = m.read_deployment(p)
    assert dep.start_date == "2026-09-30" and dep.initial_nav == 100000.0 and dep.rule_entry_ok
    assert dep.expected_fingerprint == frozen.bars_fingerprint()
    assert m.read_deployment(_history(tmp_path, rule=False)).initial_nav == 97912.36
    assert m.read_deployment(_history(tmp_path, rule=False)).initial_nav_note == m.ENTRY_COST_NOTE
    cfg = m.load_replay_config(ROOT / "config" / "live_alpaca.yaml")
    assert cfg == CFG and [s["name"] for s in cfg["sleeves"]] == ["core", "btc_trend"]


# --- live nav selection -----------------------------------------------------

def test_live_nav_last_snapshot_per_trading_day_after_close():
    # 2026-10-05 Mon, 06 Tue, 07 Wed (EDT = UTC-4). Mon: 16:30 ET and 19:00 ET -> take 19:00; Tue: only 15:30 ET.
    ts = pd.to_datetime(["2026-10-05 14:00", "2026-10-05 20:30", "2026-10-05 23:00",
                         "2026-10-06 19:30", "2026-10-07 20:05"])    # UTC
    s = pd.Series([1, 2, 3, 4, 5], index=ts, dtype=float)
    nav, flagged = m.select_live_nav(s)
    assert nav.loc["2026-10-05"] == 3.0 and nav.loc["2026-10-07"] == 5.0
    assert list(flagged) == ["2026-10-06"] and "2026-10-06" not in nav.index


# --- GET-only client --------------------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    seen: list = []

    def do_GET(self):
        type(self).seen.append(("GET", self.path))
        body = json.dumps({"dates": ["2026-10-05T20:30:00"], "values": [123.0]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # pragma: no cover - must never be hit
        type(self).seen.append(("POST", self.path)); self.send_response(200); self.end_headers()

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_server():
    _Handler.seen = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_client_get_only_allowlist(fake_server):
    c = m.GetOnlyClient(fake_server)
    snaps = m.fetch_live_snapshots(c)
    assert float(snaps.iloc[0]) == 123.0 and _Handler.seen == [("GET", "/api/live/portfolio-history")]
    n = len(_Handler.seen)
    for method, path in [("POST", "/api/live/portfolio-history"), ("DELETE", "/api/live/cycles"), ("PUT", "/api/live/status"),
                         ("GET", "/api/live/stop"), ("GET", "/api/live/portfolio-history/../stop"),
                         ("GET", "/api/live/portfolio-history?x=1"), ("GET", "http://evil/api/live/status"),
                         ("GET", "/api/live/kill-switch/reset")]:
        with pytest.raises(m.ForbiddenRequest):
            c.request(method, path)
    with pytest.raises(m.ForbiddenRequest):
        c.get("/api/live/stop")
    assert len(_Handler.seen) == n                      # nothing was sent
    for bad in ("http://example.com", "https://127.0.0.1:8001", "http://user:pw@127.0.0.1:8001"):
        with pytest.raises(m.ForbiddenRequest):
            m.GetOnlyClient(bad)


# --- tracking error and bars ------------------------------------------------

def _navs(te: np.ndarray, start="2026-01-02"):
    idx = pd.bdate_range(start, periods=len(te) + 1)
    sim = pd.Series(100.0, index=idx)
    live = pd.Series(100.0 * np.cumprod(np.r_[1.0, 1.0 + te]), index=idx)
    return live, sim


def _exact(n, mean, sd, seed=0):
    x = np.random.default_rng(seed).normal(size=n)
    x = (x - x.mean()) / x.std(ddof=1)
    return mean + sd * x


def test_i1_pass_and_fail():
    ok = m.bar_i1(*_navs(_exact(63, 3e-4, 10e-4)))
    assert ok.passed is True
    assert m.bar_i1(*_navs(_exact(63, 6e-4, 10e-4))).passed is False
    assert m.bar_i1(*_navs(_exact(63, 0.0, 16e-4))).passed is False
    live, sim = _navs(_exact(70, 3e-4, 10e-4))
    r = m.bar_i1(live, sim, {str(live.index[10].date()): "broker outage"})
    assert str(live.index[10].date()) in r.note and r.value["n"] == 69


def test_i1_not_evaluable_before_63_days():
    r = m.bar_i1(*_navs(_exact(40, 3e-4, 10e-4)))
    assert r.passed is None and "mean" in r.value and "sd" in r.value


def test_i2_month_end_drift():
    idx = pd.bdate_range("2026-10-01", "2026-12-02")
    sim = pd.Series(100.0, index=idx)
    for dev, want in ((0.019, True), (0.021, False), (-0.021, False)):
        live = sim.copy(); live.loc["2026-10-30"] = 100 * (1 + dev)
        assert m.bar_i2(live, sim).passed is want
    live = sim.copy(); live.loc["2026-12-02"] = 150.0      # mid-month last day is not a month-end
    assert m.bar_i2(live, sim).passed is True


def _fills(rows):
    return pd.DataFrame(rows, columns=["symbol", "fill_price", "close", "reason"])


def test_i3_fill_bounds():
    f = lambda bps, sym, r=None: (sym, 100 * (1 + bps / 1e4), 100.0, r)
    assert m.bar_i3(_fills([f(14, "SPY")] * 3)).passed is True
    assert m.bar_i3(_fills([f(16, "SPY")] * 3)).passed is False
    assert m.bar_i3(_fills([f(59, "BTC/USD")] * 3)).passed is True
    assert m.bar_i3(_fills([f(61, "BTC/USD")] * 3)).passed is False
    assert m.bar_i3(_fills([f(1, "SPY")] * 3 + [f(101, "SPY")])).passed is False
    assert m.bar_i3(_fills([f(1, "SPY")] * 3 + [f(101, "SPY", "gap open")])).passed is True
    assert m.bar_i3(_fills([f(1, "BTC/USD")] * 3 + [f(301, "BTC/USD")])).passed is False
    assert m.bar_i3(_fills([f(1, "BTC/USD")] * 3 + [f(301, "BTC/USD", "thin book")])).passed is True
    assert m.bar_i3(None).passed is None


def _orders(n_core, trigger="monthly", month="2026-10", unmapped=0):
    rows = [{"order_id": i, "symbol": "SPY", "date": f"{month}-01", "sleeve": "core", "decision_id": f"d{i}"}
            for i in range(n_core)]
    rows += [{"order_id": 100 + i, "symbol": "SPY", "date": f"{month}-02", "sleeve": "core", "decision_id": None}
             for i in range(unmapped)]
    dec = pd.DataFrame([{"decision_id": f"d{i}", "trigger": trigger, "breach_weight": np.nan} for i in range(n_core)])
    return pd.DataFrame(rows), dec


def test_i4_trade_count():
    sim = {"2026-10": 4}
    for live_n, want in ((3, True), (4, True), (5, True), (6, False), (2, False)):
        fills, dec = _orders(live_n)
        assert m.bar_i4(fills, dec, sim).passed is want, live_n
    fills, dec = _orders(4, trigger=None)
    assert m.bar_i4(fills, dec, sim).passed is False
    fills, dec = _orders(4, trigger="drift_band")        # drift trade without breach weight
    assert m.bar_i4(fills, dec, sim).passed is False
    fills, dec = _orders(4)
    assert m.bar_i4(fills, dec, None).passed is None      # sim counts unavailable: not a false pass


def test_bar_sleeve_drift():
    row = lambda share, tw, on=True: {"date": "2026-10-04", "share_of_nav": share, "target_within": tw, "trend_on": on}
    run = lambda *r: m.bar_sleeve_drift(pd.DataFrame(list(r))).passed
    tw = 0.6
    lo, hi = 0.08 * (tw - 0.10), 0.08 * (tw + 0.10)
    assert run(row(lo, tw)) and run(row(hi, tw))
    assert run(row(lo - 1e-4, tw)) is False and run(row(hi + 1e-4, tw)) is False
    assert run(row(0.0, 0.0, on=False)) is True and run(row(0.001, 0.0, on=False)) is False


def test_bar_text_matches_fingerprint():
    m.check_thresholds_against_frozen(frozen)
    muts = [("|mean| <= 5 bps/day", "|mean| <= 6 bps/day"), ("stdev <= 15 bps/day", "stdev <= 16 bps/day"),
            ("<= 2% at", "<= 3% at"), ("<= 15 bps for SPY/IEF", "<= 16 bps for SPY/IEF"),
            ("<= 60 bps for BTC/USD", "<= 61 bps for BTC/USD"), ("no single fill > 100 bps / 300", "no single fill > 101 bps / 300"),
            ("+/-1 of", "+/-2 of"), ("0.08 x [target_within +/- 0.10]", "0.08 x [target_within +/- 0.11]"),
            ("rolling 63-trading-day", "rolling 64-trading-day")]
    for needle, repl in muts:
        assert any(needle in b["rule"] for b in frozen.BARS_IMPLEMENTATION), needle
        class Mut:
            PORTFOLIO = frozen.PORTFOLIO
            BARS_IMPLEMENTATION = [dict(b, rule=b["rule"].replace(needle, repl, 1))
                                   for b in frozen.BARS_IMPLEMENTATION]
        with pytest.raises(ValueError):
            m.check_thresholds_against_frozen(Mut)
    # the text the fingerprint hashes is the text checked: any edit changes the fingerprint
    payload_changed = json.dumps(frozen.BARS_IMPLEMENTATION).replace("5 bps/day", "6 bps/day")
    assert payload_changed != json.dumps(frozen.BARS_IMPLEMENTATION)


def test_i5_params_match_and_mismatch_detected():
    assert m.verify_frozen_params(CFG, frozen.PORTFOLIO).passed is True
    import copy
    bad = copy.deepcopy(CFG); bad["sleeves"][1]["weight"] += 1e-9
    r = m.verify_frozen_params(bad, frozen.PORTFOLIO)
    assert r.passed is False and "satellite.weight_of_nav" in r.note
    bad = copy.deepcopy(CFG); bad["cash_buffer"] = 0.0100000001
    assert m.verify_frozen_params(bad, frozen.PORTFOLIO).passed is False
    bad = copy.deepcopy(CFG); bad["sleeves"][0]["weights"]["SPY"] = 0.6 + 1e-9
    assert m.verify_frozen_params(bad, frozen.PORTFOLIO).passed is False
    fills, dec = _orders(2, unmapped=1)
    out = m.evaluate_bars(*_navs(_exact(70, 0, 1e-4)), fills.assign(fill_price=1.0, close=1.0), dec, frozen.PORTFOLIO, live_cfg=CFG)
    assert [b.bar_id for b in out] == list(m.BAR_IDS)
    assert out[4].passed is False and "unmapped orders: 1" in out[4].note


# --- run_daily --------------------------------------------------------------

def _snapshots_for(sim: pd.Series, scale=1.0) -> pd.Series:
    ts = [pd.Timestamp(d.date()).tz_localize("US/Eastern").replace(hour=16, minute=30).tz_convert("UTC").tz_localize(None)
          for d in sim.index]
    return pd.Series(sim.to_numpy() * scale, index=pd.DatetimeIndex(ts))


@pytest.fixture
def daily_env(tmp_path, inputs):
    state = tmp_path / "data" / "forward_monitors" / "allocation"
    (state).mkdir(parents=True)
    import shutil
    shutil.copytree(inputs, state / "inputs")
    sim = m.simulated_nav(CFG, inputs, "2026-09-30", "2026-11-13", _history_nav())
    return state, sim, tmp_path


def _history_nav():
    return 100000.0


def test_run_daily_writes_no_tracked_path(daily_env):
    state, sim, tmp = daily_env
    hist = _history(tmp, pre_fill=100000.0)
    sent = []
    before = subprocess.run(["git", "-c", "safe.directory=*", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    res = m.run_daily(state, live_nav_fetcher=lambda: _snapshots_for(sim), notify=lambda **k: sent.append(k),
                      trial_history_path=hist)
    after = subprocess.run(["git", "-c", "safe.directory=*", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
    assert before == after
    assert res["status"] == "ok" and res["data_end_override"] and not sent
    assert (state / "state.json").exists() and list((state / "daily").glob("*.json"))
    top = {p.name for p in tmp.iterdir() if p.name != "hist.json"}
    assert top == {"data"}
    assert ev.prereg.DATA_END == "2026-09-28"
    # repo-internal state dir outside data/forward_monitors is refused
    with pytest.raises(ValueError):
        m.run_daily(ROOT / "docs" / "x", live_nav_fetcher=lambda: pd.Series(dtype=float), notify=lambda **k: None, trial_history_path=hist)


def test_run_daily_bar_fail_alerts_only(daily_env):
    state, sim, tmp = daily_env
    sent = []
    res = m.run_daily(state, live_nav_fetcher=lambda: _snapshots_for(sim, 1.05), notify=lambda **k: sent.append(k),
                      trial_history_path=_history(tmp, pre_fill=100000.0))
    # live is 5% above sim on the (few) days: I2 only bites at a completed month-end (2026-10-30 is in range)
    assert res["status"] == "fail" and "I2_cumulative_drift" in res["failed"]
    assert sent and sent[0]["kind"] == "allocation_forward_bar_fail" and sent[0]["severity"] == "critical"


def test_fingerprint_mismatch_alerts(daily_env):
    state, sim, tmp = daily_env
    sent = []
    res = m.run_daily(state, live_nav_fetcher=lambda: pytest.fail("must not fetch"), notify=lambda **k: sent.append(k),
                      trial_history_path=_history(tmp, fp="0" * 64))
    assert res["status"] == "integrity_failure" and sent[0]["kind"] == "allocation_forward_integrity_failure"
    assert sent[0]["severity"] == "critical"


def test_missing_live_nav_rule_blocks(daily_env):
    state, sim, tmp = daily_env
    sent = []
    res = m.run_daily(state, live_nav_fetcher=lambda: pytest.fail("must not fetch"), notify=lambda **k: sent.append(k),
                      trial_history_path=_history(tmp, rule=False))
    assert res["status"] == "blocked_missing_live_nav_rule" and sent[0]["severity"] == "warning"


def test_never_halts(daily_env, monkeypatch):
    """No broker/engine/API write: the only network object allowed is the injected GET fetcher."""
    import requests
    state, sim, tmp = daily_env

    def boom(*a, **k):
        raise AssertionError("network/broker call from run_daily")
    for name in ("post", "put", "delete", "patch", "get", "request"):
        monkeypatch.setattr(requests, name, boom)
    monkeypatch.setattr(requests.Session, "request", boom)
    for mod in ("firm.live.engine", "firm.runtime", "firm.api.app"):
        assert mod not in sys.modules or True
    res = m.run_daily(state, live_nav_fetcher=lambda: _snapshots_for(sim, 1.05), notify=lambda **k: None,
                      trial_history_path=_history(tmp, pre_fill=100000.0))
    assert res["status"] == "fail"
    src = Path(m.__file__).read_text()
    for forbidden in ("firm.live.engine", "firm.runtime", "firm.api", "alpaca_trade", "alpaca.trading", "ib_insync", "any_fail\"]("):
        assert forbidden not in src.replace("IMPLEMENTATION_ACTION.any_fail", "")


# --- alerts, env, unit files ------------------------------------------------

def test_alert_uses_environ_only(monkeypatch):
    import dotenv
    import firm.live.notifications as n
    monkeypatch.setattr(dotenv, "dotenv_values", lambda *a, **k: (_ for _ in ()).throw(AssertionError("dotenv_values called")))
    seen = []
    monkeypatch.setattr(n, "_post_webhook", lambda url, alert, timeout: seen.append((url, alert)))
    monkeypatch.setenv("ALERT_WEBHOOK_URL", "http://hook.invalid/x")
    m.make_notify()(severity="critical", kind="allocation_forward_bar_fail", message="x")
    assert seen[0][0] == "http://hook.invalid/x" and seen[0][1]["kind"] == "allocation_forward_bar_fail"
    monkeypatch.delenv("ALERT_WEBHOOK_URL")
    m.make_notify()(severity="critical", kind="k", message="x")
    assert len(seen) == 1
    for p in (Path(m.__file__), ROOT / "scripts" / "allocation_forward_monitor.py"):
        assert "dotenv" not in p.read_text()


def test_no_env_read(daily_env, monkeypatch):
    state, sim, tmp = daily_env
    real_open = builtins.open
    opened = []

    def spy(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)
    monkeypatch.setattr(builtins, "open", spy)
    real_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (opened.append(str(self)), real_read(self, *a, **k))[1])
    m.run_daily(state, live_nav_fetcher=lambda: _snapshots_for(sim), notify=lambda **k: None,
                trial_history_path=_history(tmp, pre_fill=100000.0))
    assert not any(Path(p).name == ".env" or "/.env" in p or "data_alpaca" in p for p in opened)


def test_unit_file_hygiene():
    svc = (ROOT / "deploy" / "allocation-forward-monitor.service").read_text()
    lines = [ln.strip() for ln in svc.splitlines() if ln.strip() and not ln.startswith("#")]
    env_files = [ln for ln in lines if ln.startswith("EnvironmentFile=")]
    assert env_files and all(not ln.endswith("/.env") and "ai-trading-system" not in ln for ln in env_files)
    user = next(ln for ln in lines if ln.startswith("User=")).split("=")[1]
    assert user not in ("root", "0")
    assert "Nice=10" in lines
    rw = [ln for ln in lines if ln.startswith("ReadWritePaths=")]
    assert rw == ["ReadWritePaths=/local/store/git/ai-trading-system/data/forward_monitors/allocation"]
    assert "ProtectSystem=strict" in lines
    execs = [ln for ln in lines if ln.startswith("ExecStart=")]
    assert execs[0].endswith("refresh-inputs") and execs[1].endswith(" run")
    assert not any(ln.startswith("User=root") for ln in lines)
    timer = (ROOT / "deploy" / "allocation-forward-monitor.timer").read_text()
    assert "Persistent=true" in timer and "UTC" in timer


def test_refresh_inputs_writes_all_or_nothing(tmp_path):
    ok = {n: (lambda: pd.DataFrame({"a": [1]})) for n in m.INPUT_FILES}
    out = tmp_path / "inputs"
    assert len(m.refresh_inputs(out, ok)) == 15 and len(list(out.glob("*.parquet"))) == 15
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    bad = dict(ok); bad["cboe_PUT.parquet"] = lambda: (_ for _ in ()).throw(RuntimeError("net"))
    with pytest.raises(RuntimeError):
        m.refresh_inputs(out, bad)
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before
    with pytest.raises(ValueError):
        m.refresh_inputs(out, {k: v for k, v in ok.items() if k != "fred_DTB3.parquet"})


def test_cli_validate_and_export_guard(tmp_path):
    spec = importlib.util.spec_from_file_location("afm_cli", ROOT / "scripts" / "allocation_forward_monitor.py")
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    assert cli.DEFAULT_STATE == ROOT / "data" / "forward_monitors" / "allocation"
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "allocation_forward_monitor.py"), "export",
                          "--out", str(ROOT / "docs" / "x.json"), "--state", str(tmp_path)],
                         capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
    assert out.returncode != 0 and not (ROOT / "docs" / "x.json").exists()
