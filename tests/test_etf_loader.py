"""ETF point-in-time loader + hashed manifests (P2-02). Synthetic fixtures in tmp_path only."""

from __future__ import annotations

import datetime as dt
import types

import numpy as np
import pandas as pd
import pytest
import yaml

from firm.research.seal import HoldoutAccessError

SEAL = dt.date(2026, 10, 1)
LAST = dt.date(2026, 9, 30)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Temp freeze config whose allow root is tmp_path/eodhd; synthetic store under it."""
    from firm import runtime
    from firm.data import pit_store
    from firm.research import seal

    for name in ("firm.live.engine", "firm.api.app"):
        monkeypatch.delitem(__import__("sys").modules, name, raising=False)
    monkeypatch.setattr(pit_store, "_ACCESS_GUARD", None)
    monkeypatch.setattr(runtime, "_ACCESS_GUARD", None)
    root = tmp_path / "eodhd"
    (root / "etfs_full").mkdir(parents=True)
    (root / "forex").mkdir()
    macro = tmp_path / "il_macro"
    macro.mkdir()
    cfg = {
        "seal_date": SEAL, "burned_through": LAST, "unseal_token_sha256": None,
        "allow_roots": [str(root), str(macro)], "deny_paths": [str(tmp_path / "denied")],
        "exempt_monitors": [], "unseal_log": [], "sealed_instruments": [],
    }
    p = tmp_path / "research_freeze.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(seal, "_CONFIG_PATH", p)
    seal._load_config.cache_clear()
    yield types.SimpleNamespace(root=root, macro=macro, tmp=tmp_path)
    seal._load_config.cache_clear()


def _bars(dates, start=100.0, seed=0, vol=0.004):
    rng = np.random.default_rng(seed)
    px = start * np.cumprod(1 + rng.normal(0.0003, vol, len(dates)))
    return pd.DataFrame({
        "date": pd.to_datetime(dates), "open": px, "high": px * 1.002, "low": px * 0.998,
        "close": px, "adjusted_close": px, "volume": 1_000_000.0,
    })


def _dates(start="2025-10-01", n=200):
    return pd.bdate_range(start, periods=n)


def _write(env, sym, df):
    df.to_parquet(env.root / "etfs_full" / f"{sym}.parquet", index=False)


def _universe(env, syms):
    p = env.tmp / "universe.yaml"
    p.write_text(yaml.safe_dump({"instruments": [{"symbol": s} for s in syms]}))
    return p


def _setup(env, syms=("SPY", "AAA"), n=200):
    d = _dates(n=n)
    d = d[d <= pd.Timestamp(LAST)]
    for i, s in enumerate(syms):
        _write(env, s, _bars(d, seed=i))
    return d, _universe(env, syms)


def _load(env, uni, asof, **kw):
    from firm.data.etf_loader import load_etf_universe

    return load_etf_universe(uni, asof=asof, data_root=env.root, **kw)


def test_no_lookahead(env):
    d, uni = _setup(env)
    asof = d[100].date()
    a = _load(env, uni, asof)["AAA"].bars
    assert a.index.max() <= pd.Timestamp(asof)
    df = pd.read_parquet(env.root / "etfs_full" / "AAA.parquet")
    df.loc[df.index[150:], ["open", "high", "low", "close", "adjusted_close"]] *= 1.07
    _write(env, "AAA", df)
    b = _load(env, uni, asof)["AAA"].bars
    pd.testing.assert_frame_equal(a, b)


def test_dates_monotonic_unique(env):
    d, uni = _setup(env)
    df = pd.read_parquet(env.root / "etfs_full" / "AAA.parquet")
    df = pd.concat([df, df.iloc[[10, 11]]]).sample(frac=1, random_state=1)
    _write(env, "AAA", df)
    bars = _load(env, uni, d[-1].date())["AAA"].bars
    assert bars.index.is_monotonic_increasing and bars.index.is_unique
    assert {"open", "high", "low", "close", "adjusted_close", "volume", "segment"} <= set(bars.columns)


def _spike_fixture(env, spike_pos=100, revert_after=5):
    d = _dates(n=220)
    _write(env, "SPY", _bars(d, seed=0))
    df = _bars(d, seed=1)
    cols = ["open", "high", "low", "close", "adjusted_close"]
    base = df.loc[spike_pos - 1, "close"]
    for k in range(spike_pos, spike_pos + revert_after):  # elevated run, then reverts
        df.loc[k, cols] = base * 3.0
    _write(env, "AAA", df)
    return d, _universe(env, ["SPY", "AAA"]), spike_pos, revert_after


@pytest.mark.parametrize("seed", range(6))
def test_pit_asof_property(env, seed):
    """load(asof) == load(later)[:asof] when asof is > 5 bars past any spike (loop in place of hypothesis)."""
    d, uni, sp, ra = _spike_fixture(env)
    rng = np.random.default_rng(seed)
    first = sp + ra + 6  # more than 5 bars past the last spiked bar
    for _ in range(5):
        i = int(rng.integers(first, len(d) - 10))
        j = int(rng.integers(i + 1, len(d)))
        early = _load(env, uni, d[i].date())["AAA"].bars
        late = _load(env, uni, d[j].date())["AAA"].bars
        pd.testing.assert_frame_equal(early, late.loc[: d[i]])


def test_spike_lookahead_not_used(env):
    # spike (+200%) starts at position p; reverts at p+4. asof = p+1 cannot see the reversal, the bar is kept.
    d, uni, p, _ra = _spike_fixture(env, spike_pos=100, revert_after=4)
    asof_early = d[p + 1].date()
    asof_late = d[p + 8].date()
    early = _load(env, uni, asof_early)["AAA"].bars
    late = _load(env, uni, asof_late)["AAA"].bars
    assert d[p] in early.index  # kept: reversal not yet visible
    assert d[p] not in late.index  # dropped once the reversal is visible
    assert d[p + 1] not in late.index


def test_delisted_truncated_and_reported(env):
    from firm.data.etf_loader import load_etf_universe_with_report

    d, uni = _setup(env, ("SPY", "LIVE"))
    dd = d[:120]
    _write(env, "DEAD", _bars(dd, seed=9))
    pd.DataFrame({"Code": ["DEAD", "GONE", "STK"], "Type": ["ETF", "ETF", "Common Stock"]}).to_parquet(
        env.root / "symbols_delisted.parquet", index=False)
    out, rep = load_etf_universe_with_report(uni, asof=d[-1].date(), data_root=env.root, include_delisted=True)
    assert out["DEAD"].delisted_on == dd[-1].date()
    assert out["DEAD"].bars.index.max() == dd[-1]
    assert out["LIVE"].delisted_on is None
    assert rep["missing"] == ["GONE"]  # delisted ETF with no history: reported, not skipped
    assert "STK" not in out
    out2 = _load(env, uni, d[-1].date(), include_delisted=False)
    assert "DEAD" not in out2


def test_total_return_uses_adjusted_close_within_segments(env):
    from firm.data.etf_loader import total_return

    d, uni = _setup(env)
    s = _load(env, uni, d[-1].date())["AAA"]
    r = total_return(s)
    adj = s.bars["adjusted_close"]
    seg = s.bars["segment"]
    exp = adj.groupby(seg).pct_change()
    pd.testing.assert_series_equal(r, exp, check_names=False)
    assert np.isnan(r.iloc[0])


def test_manifest_roundtrip_and_tamper(env):
    from firm.data.manifest import build_manifest, snapshot_id, verify_manifest, write_manifest

    d, _ = _setup(env)
    paths = sorted((env.root / "etfs_full").glob("*.parquet"))
    m = build_manifest(paths, root=env.root, dataset="eodhd_etfs_full", licence="vendor")
    assert list(m.columns) == ["dataset", "licence", "path", "bytes", "sha256", "rows", "first_date", "last_date"]
    assert m["rows"].iloc[0] == len(d)
    sid = snapshot_id(m)
    assert sid == snapshot_id(m.sample(frac=1, random_state=3))  # order independent
    out = env.tmp / "m.json"
    write_manifest(m, out)
    assert verify_manifest(out, env.root) == []
    target = env.root / "etfs_full" / "AAA.parquet"
    raw = bytearray(target.read_bytes())
    raw[-1] ^= 0x01
    target.write_bytes(bytes(raw))
    assert verify_manifest(out, env.root) == ["etfs_full/AAA.parquet"]
    m2 = build_manifest(paths, root=env.root, dataset="eodhd_etfs_full", licence="vendor", meta=False)
    assert snapshot_id(m2) != sid


@pytest.mark.parametrize("asof,ok", [(dt.date(2026, 10, 1), False), (dt.date(2026, 10, 2), False), (dt.date(2026, 9, 30), True)])
def test_seal_blocks_post_seal_asof(env, asof, ok):
    d = pd.bdate_range("2026-08-03", "2026-10-09")  # fixture deliberately holds post-seal rows
    _write(env, "SPY", _bars(d, seed=0))
    _write(env, "AAA", _bars(d, seed=1))
    uni = _universe(env, ["SPY", "AAA"])
    if not ok:
        with pytest.raises(HoldoutAccessError):
            _load(env, uni, asof)
        return
    bars = _load(env, uni, asof)["AAA"].bars
    assert bars.index.max() <= pd.Timestamp(LAST)


def test_uses_cleaning_v3(env):
    d = _dates(n=120)
    _write(env, "SPY", _bars(d, seed=0))
    df = _bars(d, seed=1)
    df.loc[50, ["open", "high", "low", "close", "adjusted_close"]] = 999999.9999  # v3-only sentinel rule
    _write(env, "AAA", df)
    s = _load(env, _universe(env, ["SPY", "AAA"]), d[-1].date())["AAA"]
    assert d[50] not in s.bars.index
    assert s.clean_report["sentinel"] == 1


def _cpi(env, with_publish=True):
    obs = pd.to_datetime(["2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01"])
    df = pd.DataFrame({"date": obs, "value": np.linspace(100, 103, len(obs))})
    if with_publish:  # published on the 15th of the following month
        df["publish_date"] = (obs + pd.offsets.MonthBegin(1) + pd.Timedelta(days=14))
    df.to_parquet(env.macro / "il_cpi.parquet", index=False)


def test_cpi_publication_lag(env):
    from firm.data.etf_loader import il_cpi

    _cpi(env)
    # July obs is published 2026-08-15; the August obs on 2026-09-15.
    s = il_cpi(dt.date(2026, 8, 14), il_macro_root=env.macro)
    assert s.index.max() == pd.Timestamp("2026-06-01")
    s = il_cpi(dt.date(2026, 8, 15), il_macro_root=env.macro)
    assert s.index.max() == pd.Timestamp("2026-07-01")
    assert il_cpi(dt.date(2026, 9, 30), il_macro_root=env.macro).index.max() == pd.Timestamp("2026-08-01")
    for bad in (dt.date(2026, 10, 1), dt.date(2026, 10, 2)):
        with pytest.raises(HoldoutAccessError):
            il_cpi(bad, il_macro_root=env.macro)


def test_cpi_default_lag_when_publish_unknown(env):
    from firm.data.etf_loader import CPI_DEFAULT_LAG_DAYS, il_cpi

    assert CPI_DEFAULT_LAG_DAYS == 45
    _cpi(env, with_publish=False)
    # obs 2026-07-01 (month-end 07-31) + 45d = 2026-09-14
    assert il_cpi(dt.date(2026, 9, 13), il_macro_root=env.macro).index.max() == pd.Timestamp("2026-06-01")
    assert il_cpi(dt.date(2026, 9, 14), il_macro_root=env.macro).index.max() == pd.Timestamp("2026-07-01")


def test_usd_ils_seal_and_sources(env):
    from firm.data.etf_loader import usd_ils

    d = pd.bdate_range("2026-08-03", "2026-10-09")
    fx = pd.DataFrame({"date": d, "close": 3.6 + 0.001 * np.arange(len(d))})
    fx.to_parquet(env.macro / "usd_ils.parquet", index=False)
    fx.assign(close=fx["close"] * 1.0001).to_parquet(env.root / "forex" / "USDILS.parquet", index=False)
    s = usd_ils(dt.date(2026, 9, 30), il_macro_root=env.macro)
    assert s.index.max() <= pd.Timestamp(LAST)
    s2 = usd_ils(dt.date(2026, 9, 30), source="eodhd", data_root=env.root)
    assert s2.iloc[-1] == pytest.approx(s.iloc[-1] * 1.0001)
    for bad in (dt.date(2026, 10, 1), dt.date(2026, 10, 2)):
        with pytest.raises(HoldoutAccessError):
            usd_ils(bad, il_macro_root=env.macro)


def test_load_dividends_pit_and_unadjusted(env):
    from firm.data.etf_loader import load_dividends

    (env.root / "corporate_actions" / "dividends").mkdir(parents=True)
    pd.DataFrame({"date": pd.to_datetime(["2026-03-20", "2026-09-20", "2026-10-20"]), "value": [0.4, 0.5, 0.6],
                  "unadjustedValue": [0.41, 0.51, 0.61]}).to_parquet(env.root / "corporate_actions" / "dividends" / "AAA.parquet", index=False)
    out = load_dividends(["AAA", "NOPE"], asof=LAST, data_root=env.root)
    assert list(out["amount"]) == [0.41, 0.51]  # post-seal dividend dropped; unadjusted value used; missing file skipped
    with pytest.raises(HoldoutAccessError):
        load_dividends(["AAA"], asof=SEAL, data_root=env.root)
