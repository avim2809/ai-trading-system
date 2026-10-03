"""Forward-data seal: fail-closed loader, access guards, default-off in live processes.

Synthetic fixtures only (CI has no real data). CODEOWNERS-protected (P0-04): if a test
here looks wrong, report it; do not change it to make code pass.
"""

from __future__ import annotations

import ast
import datetime as dt
import sys
import types

import pandas as pd
import pytest
import yaml

from tests.integrity._fresh_import import REPO_ROOT, repo_data_listing, run_fresh

SEAL = dt.date(2026, 10, 1)
LAST_USABLE = dt.date(2026, 9, 30)
SEALED_DT = dt.datetime(2026, 10, 1)
USABLE_DT = dt.datetime(2026, 9, 30)

REQUIRED_DENY = {
    "data/research/s2_forward",
    "data/forward_monitors",
    "data/cache",
    "data_alpaca",
    "research/monitoring_sealed",
    "docs/s2_forward_snapshot.json",
    "data/logs",
    "data/live_state.db",
    "data/cycle_history.json",
    "data/order_history.json",
    "data/execution_audit.jsonl",
    "data/memory",
    "data/llm_cache*.db",
}


def _real_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "config" / "research_freeze.yaml").read_text())


@pytest.fixture
def slots(monkeypatch):
    """Both guard hooks reset to None (restored after the test), live modules hidden."""
    from firm import runtime
    from firm.data import pit_store
    from firm.research import seal

    for name in ("firm.live.engine", "firm.api.app"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(pit_store, "_ACCESS_GUARD", None)
    monkeypatch.setattr(runtime, "_ACCESS_GUARD", None)
    seal._load_config.cache_clear()
    yield pit_store, runtime, seal
    seal._load_config.cache_clear()


@pytest.fixture
def guards(slots):
    pit_store, runtime, seal = slots
    seal.install_guards()
    return seal


@pytest.fixture
def tmp_freeze(slots, monkeypatch, tmp_path):
    """Point the seal at a temp config whose roots live under tmp_path."""
    _, _, seal = slots
    allowed = tmp_path / "allowed"
    denied = tmp_path / "denied"
    allowed.mkdir()
    denied.mkdir()
    cfg = {
        "seal_date": SEAL,
        "burned_through": LAST_USABLE,
        "unseal_token_sha256": None,
        "allow_roots": [str(allowed)],
        "deny_paths": [str(denied), str(allowed / "inner_denied"), str(allowed / "secret*.parquet")],
        "exempt_monitors": [],
        "unseal_log": [],
        "sealed_instruments": [],
    }
    cfg_path = tmp_path / "research_freeze.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(seal, "_CONFIG_PATH", cfg_path)
    seal._load_config.cache_clear()
    return types.SimpleNamespace(seal=seal, allowed=allowed, denied=denied, path=cfg_path, cfg=cfg)


def _frame(*days: dt.date, symbol: str = "X") -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime(list(days)), "symbol": symbol, "close": 1.0})


# --------------------------------------------------------------------------- config / seal


def test_config_has_exact_seal_and_lists():
    cfg = _real_config()
    assert cfg["seal_date"] == SEAL
    assert cfg["burned_through"] == LAST_USABLE
    assert cfg["unseal_token_sha256"] is None  # owner fills; the agent never sees the preimage
    assert cfg["unseal_log"] == []
    assert REQUIRED_DENY <= set(cfg["deny_paths"])
    assert {"data/research/eodhd", "data/research/fred", "data/research/insider"} <= set(cfg["allow_roots"])
    assert not set(cfg["allow_roots"]) & set(cfg["deny_paths"])


def test_dates_come_from_config(slots):
    _, _, seal = slots
    assert seal.seal_date() == SEAL
    assert seal.max_research_date() == LAST_USABLE
    assert seal.max_research_date() == seal.seal_date() - dt.timedelta(days=1)


def test_incoherent_config_fails_closed(tmp_freeze):
    cfg = dict(tmp_freeze.cfg, burned_through=dt.date(2026, 9, 1))
    tmp_freeze.path.write_text(yaml.safe_dump(cfg))
    tmp_freeze.seal._load_config.cache_clear()
    with pytest.raises(tmp_freeze.seal.HoldoutAccessError):
        tmp_freeze.seal.max_research_date()


def test_missing_config_fails_closed(tmp_freeze, tmp_path):
    tmp_freeze.seal._CONFIG_PATH = tmp_path / "does_not_exist.yaml"
    tmp_freeze.seal._load_config.cache_clear()
    with pytest.raises(tmp_freeze.seal.HoldoutAccessError):
        tmp_freeze.seal.seal_date()


@pytest.mark.parametrize(
    "value",
    [dt.date(2026, 9, 30), dt.datetime(2026, 9, 30, 23, 59), pd.Timestamp("2026-09-30"), "2026-09-30"],
)
def test_check_asof_accepts_last_usable_day(slots, value):
    _, _, seal = slots
    seal.check_asof(value, what="t")


@pytest.mark.parametrize(
    "value",
    [dt.date(2026, 10, 1), dt.datetime(2026, 10, 1, 0, 0), pd.Timestamp("2026-10-01"), "2027-01-01"],
)
def test_check_asof_rejects_sealed_days(slots, value):
    _, _, seal = slots
    with pytest.raises(seal.HoldoutAccessError):
        seal.check_asof(value, what="t")


def test_check_frame(slots):
    _, _, seal = slots
    seal.check_frame(_frame(dt.date(2026, 9, 29), LAST_USABLE), what="ok")
    seal.check_frame(pd.DataFrame(), what="empty")
    with pytest.raises(seal.HoldoutAccessError):
        seal.check_frame(_frame(LAST_USABLE, SEAL), what="post-seal row")
    # DatetimeIndex frames are checked too
    idx = pd.DataFrame({"v": [1, 2]}, index=pd.to_datetime([LAST_USABLE, SEAL]))
    with pytest.raises(seal.HoldoutAccessError):
        seal.check_frame(idx, what="index")
    # fail closed when no date information can be found
    with pytest.raises(seal.HoldoutAccessError):
        seal.check_frame(pd.DataFrame({"v": [1]}), what="undated")


# --------------------------------------------------------------------------- store getters


def test_getter_past_seal_raises(guards):
    from firm.data.pit_store import PointInTimeDataStore

    store = PointInTimeDataStore()
    with pytest.raises(guards.HoldoutAccessError):
        store.get_prices(["X"], asof=SEALED_DT)
    assert store.get_prices(["X"], asof=USABLE_DT).empty  # empty store, but no raise


def test_union_end_past_seal_raises(guards):
    from firm.data.pit_store import PointInTimeDataStore

    store = PointInTimeDataStore()
    with pytest.raises(guards.HoldoutAccessError):
        store.get_universe_union(dt.datetime(2026, 9, 1), SEALED_DT)
    assert store.get_universe_union(dt.datetime(2026, 9, 1), USABLE_DT) == []


def test_load_frame_with_post_seal_rows_raises(guards):
    from firm.data.pit_store import PointInTimeDataStore

    store = PointInTimeDataStore()
    with pytest.raises(guards.HoldoutAccessError):
        store.load(_frame(LAST_USABLE, SEAL))
    with pytest.raises(guards.HoldoutAccessError):
        store.load(_frame(LAST_USABLE), fundamentals=_frame(SEAL))
    with pytest.raises(guards.HoldoutAccessError):
        store.load_macro({"T10Y2Y": pd.DataFrame({"date": [SEAL], "T10Y2Y": [0.1]})})
    store.load(_frame(dt.date(2026, 9, 29), LAST_USABLE))  # pre-seal frames load fine
    store.load_macro({"T10Y2Y": pd.DataFrame({"date": [LAST_USABLE], "T10Y2Y": [0.1]})})


def test_env_var_ignored(guards, monkeypatch):
    from firm.data.pit_store import PointInTimeDataStore

    monkeypatch.setenv("HOLDOUT_UNSEAL_TOKEN", "anything")
    monkeypatch.setenv("FIRM_RESEARCH", "0")
    with pytest.raises(guards.HoldoutAccessError):
        PointInTimeDataStore().get_prices(["X"], asof=SEALED_DT)


_GETTERS = {
    "get_prices": lambda s, d: s.get_prices(["X"], d),
    "get_fundamentals": lambda s, d: s.get_fundamentals(["X"], d),
    "get_estimates": lambda s, d: s.get_estimates(["X"], d),
    "get_ai_scores": lambda s, d: s.get_ai_scores(["X"], d),
    "get_market_percentile_pool": lambda s, d: s.get_market_percentile_pool(d),
    "get_sentiment": lambda s, d: s.get_sentiment(["X"], d),
    "get_macro": lambda s, d: s.get_macro("T10Y2Y", d),
    "get_universe": lambda s, d: s.get_universe(d),
    "get_universe_union": lambda s, d: s.get_universe_union(d - dt.timedelta(days=5), d),
}


@pytest.mark.parametrize("name", sorted(_GETTERS))
def test_every_guarded_getter_raises_past_seal(guards, name):
    from firm.data.pit_store import PointInTimeDataStore

    store = PointInTimeDataStore()
    call = _GETTERS[name]
    with pytest.raises(guards.HoldoutAccessError):
        call(store, dt.datetime.combine(guards.seal_date(), dt.time()))
    call(store, dt.datetime.combine(guards.max_research_date(), dt.time()))  # returns


def test_guarded_getter_set_is_complete():
    """The parametrised table above must cover every asof/start-end getter."""
    tree = ast.parse((REPO_ROOT / "src/firm/data/pit_store.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PointInTimeDataStore")
    covered = {
        n.name
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name.startswith("get_")
        and {a.arg for a in n.args.args} & {"asof", "start", "end"}
    }
    assert covered == set(_GETTERS)


# --------------------------------------------------------------------------- runtime loaders

_PRICE_LIKE = lambda d: pd.DataFrame({"date": [pd.Timestamp(d)], "symbol": ["X"], "close": [1.0]})  # noqa: E731
_LOADERS = {
    "load_prices": ("combined/prices", _PRICE_LIKE),
    "load_fundamentals": ("combined/fundamentals", _PRICE_LIKE),
    "load_macro": (
        "combined/macro",
        lambda d: pd.DataFrame({"date": [pd.Timestamp(d)], "symbol": ["T10Y2Y"], "value": [0.1]}),
    ),
    "load_sentiment": ("combined/sentiment", _PRICE_LIKE),
    "load_analyst_ratings": ("combined/analyst_ratings", _PRICE_LIKE),
    "load_ai_scores": ("combined/ai_scores", _PRICE_LIKE),
    "load_market_percentile": ("combined/market_percentile", _PRICE_LIKE),
    "load_universe_membership": (
        "combined/universe_membership",
        lambda d: pd.DataFrame(
            {
                "index": ["sp500"],
                "symbol": ["X"],
                "added_date": [pd.Timestamp(d)],
                "removed_date": [pd.NaT],
            }
        ),
    ),
}


@pytest.mark.parametrize("name", sorted(_LOADERS))
def test_every_runtime_loader_raises_past_seal(guards, monkeypatch, tmp_path, name):
    from firm import runtime
    from firm.data.cache import ParquetCache

    key, build = _LOADERS[name]
    current: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr(ParquetCache, "get", lambda self, k: current.get(k))
    settings = types.SimpleNamespace(data=types.SimpleNamespace(cache_dir=str(tmp_path / "cache")))
    loader = getattr(runtime, name)

    current[key] = build(guards.seal_date())
    with pytest.raises(guards.HoldoutAccessError):
        loader(settings)
    current[key] = build(guards.max_research_date())
    assert loader(settings) is not None


def test_runtime_loaders_set_is_complete():
    tree = ast.parse((REPO_ROOT / "src/firm/runtime.py").read_text())
    loaders = {n.name for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("load_")}
    assert loaders == set(_LOADERS)


def test_runtime_loader_refuses_denied_cache_dir(guards):
    from firm import runtime

    settings = types.SimpleNamespace(data=types.SimpleNamespace(cache_dir="data/cache"))
    with pytest.raises(guards.HoldoutAccessError):
        runtime.load_prices(settings)
    assert not (REPO_ROOT / "data" / "cache" / "combined").exists()


# --------------------------------------------------------------------------- data_access


def test_read_parquet_enforces_path_and_asof(tmp_freeze, monkeypatch):
    from firm.research import data_access

    seal = tmp_freeze.seal
    good = tmp_freeze.allowed / "spy.parquet"
    _frame(dt.date(2026, 9, 28), LAST_USABLE, SEAL, dt.date(2026, 10, 5)).to_parquet(good)

    out = data_access.read_parquet(good, asof=dt.date(2026, 9, 29))
    assert out["date"].max() == pd.Timestamp("2026-09-28") and len(out) == 1
    out = data_access.read_parquet(good, asof=seal.max_research_date())
    assert out["date"].max() == pd.Timestamp(LAST_USABLE) and len(out) == 2  # rows after asof dropped
    with pytest.raises(seal.HoldoutAccessError):
        data_access.read_parquet(good, asof=seal.seal_date())

    opened: list[object] = []
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: opened.append(a) or pd.DataFrame())
    outside = tmp_freeze.denied / "x.parquet"
    in_deny_under_allow = tmp_freeze.allowed / "inner_denied" / "x.parquet"
    glob_denied = tmp_freeze.allowed / "secret_q.parquet"
    for bad in (outside, in_deny_under_allow, glob_denied, tmp_freeze.allowed.parent / "elsewhere.parquet"):
        with pytest.raises(seal.HoldoutAccessError):
            data_access.read_parquet(bad, asof=dt.date(2026, 9, 1))
    assert opened == []  # refused before the file was opened


def test_read_parquet_undated_file_fails_closed(tmp_freeze):
    from firm.research import data_access

    p = tmp_freeze.allowed / "undated.parquet"
    pd.DataFrame({"v": [1, 2]}).to_parquet(p)
    with pytest.raises(tmp_freeze.seal.HoldoutAccessError):
        data_access.read_parquet(p, asof=dt.date(2026, 9, 1))


def test_data_access_denies_monitor_dirs(slots):
    from firm.research import data_access

    _, _, seal = slots
    for entry in _real_config()["deny_paths"]:
        target = REPO_ROOT / entry.replace("*", "x")
        with pytest.raises(seal.HoldoutAccessError):
            data_access.assert_path_allowed(target)
        with pytest.raises(seal.HoldoutAccessError):  # and repo-relative spelling
            data_access.assert_path_allowed(entry.replace("*", "x"))


def test_data_access_is_allow_list(slots):
    from firm.research import data_access

    _, _, seal = slots
    for refused in ("data/live_state.db", "data/logs/api.log", "data/cycle_history.json", "data/llm_cache_ab_llm.db",
                    "data/anything_new.parquet", "src/firm/runtime.py", "/etc/passwd"):
        with pytest.raises(seal.HoldoutAccessError):
            data_access.assert_path_allowed(refused)
    for permitted in ("data/research/eodhd/etfs_full/SPY.parquet", "data/research/fred/T10Y2Y.parquet",
                      "data/research/insider/x.csv"):
        data_access.assert_path_allowed(permitted)


def test_symlink_cannot_escape_the_allow_list(tmp_freeze):
    from firm.research import data_access

    target = tmp_freeze.denied / "secret.parquet"
    _frame(dt.date(2026, 9, 1)).to_parquet(target)
    link = tmp_freeze.allowed / "link.parquet"
    link.symlink_to(target)
    with pytest.raises(tmp_freeze.seal.HoldoutAccessError):
        data_access.assert_path_allowed(link)


def test_load_panel(tmp_freeze):
    from firm.research import data_access

    seal = tmp_freeze.seal
    panel = tmp_freeze.allowed / "etfs"
    panel.mkdir()
    _frame(dt.date(2026, 9, 25), dt.date(2026, 9, 30), SEAL, symbol="SPY").to_parquet(panel / "SPY.parquet")
    _frame(dt.date(2026, 9, 25), dt.date(2026, 9, 30), symbol="IEF").to_parquet(panel / "IEF.parquet")

    out = data_access.load_panel("etfs", start="2026-09-26", end="2026-09-30")
    assert set(out["symbol"]) == {"SPY", "IEF"}
    assert out["date"].min() == pd.Timestamp("2026-09-30") and out["date"].max() == pd.Timestamp("2026-09-30")
    only = data_access.load_panel("etfs", start="2026-09-01", end="2026-09-30", symbols=["SPY"])
    assert set(only["symbol"]) == {"SPY"} and len(only) == 2

    with pytest.raises(seal.HoldoutAccessError):
        data_access.load_panel("etfs", start="2026-09-01", end="2026-10-01")
    for bad_kind in ("../denied", "etfs/../../denied", "", "missing_kind"):
        with pytest.raises(seal.HoldoutAccessError):
            data_access.load_panel(bad_kind, start="2026-09-01", end="2026-09-30")


def test_data_access_first_use_installs_guards(tmp_freeze):
    from firm import runtime
    from firm.data import pit_store
    from firm.research import data_access

    p = tmp_freeze.allowed / "a.parquet"
    _frame(dt.date(2026, 9, 1)).to_parquet(p)
    assert pit_store._ACCESS_GUARD is None and runtime._ACCESS_GUARD is None
    data_access.read_parquet(p, asof=dt.date(2026, 9, 2))
    assert pit_store._ACCESS_GUARD is not None and runtime._ACCESS_GUARD is not None


# --------------------------------------------------------------------------- install / default-off


def test_install_guards_is_idempotent(slots):
    pit_store, runtime, seal = slots
    seal.install_guards()
    first = (pit_store._ACCESS_GUARD, runtime._ACCESS_GUARD)
    seal.install_guards()
    assert (pit_store._ACCESS_GUARD, runtime._ACCESS_GUARD) == first
    assert all(g is not None for g in first)


@pytest.mark.parametrize("live_module", ["firm.live.engine", "firm.api.app"])
def test_install_guards_refuses_in_live_process(slots, monkeypatch, live_module):
    pit_store, runtime, seal = slots
    monkeypatch.setitem(sys.modules, live_module, types.ModuleType(live_module))
    with pytest.raises(seal.HoldoutAccessError):
        seal.install_guards()
    assert pit_store._ACCESS_GUARD is None and runtime._ACCESS_GUARD is None


def test_guard_default_is_none_on_fresh_import(tmp_path):
    before = repo_data_listing()
    code = (
        "import sys\n"
        "import firm.api.app\n"
        "from firm.data import pit_store\n"
        "from firm import runtime\n"
        "assert pit_store._ACCESS_GUARD is None and runtime._ACCESS_GUARD is None\n"
        "assert not any(m == 'firm.research' or m.startswith('firm.research.') for m in sys.modules)\n"
        "print('GUARDS_OFF')\n"
    )
    out = run_fresh(code, tmp_path)
    assert out.returncode == 0, out.stderr
    assert "GUARDS_OFF" in out.stdout.splitlines()
    assert repo_data_listing() == before  # nothing created under the repo's data/


# --------------------------------------------------------------------------- static guard coverage


def _first_statement(fn: ast.FunctionDef) -> ast.stmt:
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant):
        body = body[1:]  # docstring
    return body[0]


def _is_guard_check(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.If) and "_ACCESS_GUARD" in ast.dump(stmt.test)


def test_all_getters_begin_with_guard():
    pit_tree = ast.parse((REPO_ROOT / "src/firm/data/pit_store.py").read_text())
    cls = next(n for n in pit_tree.body if isinstance(n, ast.ClassDef) and n.name == "PointInTimeDataStore")
    missing = []
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name.startswith("_"):
            continue
        args = {a.arg for a in fn.args.args}
        if fn.name.startswith("get_") and args & {"asof", "start", "end"}:
            if not _is_guard_check(_first_statement(fn)):
                missing.append(f"PointInTimeDataStore.{fn.name}")
        if fn.name in ("load", "load_macro") and "_ACCESS_GUARD" not in ast.dump(fn):
            missing.append(f"PointInTimeDataStore.{fn.name}")

    rt_tree = ast.parse((REPO_ROOT / "src/firm/runtime.py").read_text())
    for fn in rt_tree.body:
        if isinstance(fn, ast.FunctionDef) and fn.name.startswith("load_"):
            if not _is_guard_check(_first_statement(fn)):
                missing.append(f"runtime.{fn.name}")
    assert not missing, f"getters without a leading _ACCESS_GUARD check: {missing}"
    # each module defines its own hook (self-contained: no cross-module symbol needed by live code)
    for tree, label in ((pit_tree, "pit_store"), (rt_tree, "runtime")):
        names = {
            t.id
            for n in tree.body
            if isinstance(n, (ast.Assign, ast.AnnAssign))
            for t in ([n.target] if isinstance(n, ast.AnnAssign) else n.targets)
            if isinstance(t, ast.Name)
        }
        assert "_ACCESS_GUARD" in names, f"{label} must define _ACCESS_GUARD itself"


def test_live_modules_do_not_import_firm_research():
    live_files = [
        *sorted((REPO_ROOT / "src/firm/live").glob("*.py")),
        *sorted((REPO_ROOT / "src/firm/api").rglob("*.py")),
        REPO_ROOT / "src/firm/runtime.py",
        REPO_ROOT / "src/firm/data/pit_store.py",
    ]
    offenders = []
    for path in live_files:
        for node in ast.walk(ast.parse(path.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names]
            if any(n == "firm.research" or n.startswith("firm.research.") for n in names):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, offenders
