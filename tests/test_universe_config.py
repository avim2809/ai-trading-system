"""P2-01 schema tests for config/universe_{etf,futures}.yaml."""
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ETF = yaml.safe_load((ROOT / "config" / "universe_etf.yaml").read_text())
FUT = yaml.safe_load((ROOT / "config" / "universe_futures.yaml").read_text())
FROZEN_DRAFT_MARKETS = ["ES", "NQ", "RTY", "FDAX", "ZF", "ZN", "ZB", "FGBL", "FGBS", "6E", "6J", "6B", "6A", "6C", "CL", "GC", "HG", "ZC", "ZS"]
DATA = ROOT / "data" / "research" / "eodhd" / "etfs_full"


def _d(v):
    return v if isinstance(v, dt.date) else dt.date.fromisoformat(str(v))


def test_etf_count_and_unique_symbols():
    syms = [i["symbol"] for i in ETF["instruments"]]
    assert 15 <= len(syms) <= 25
    assert len(set(syms)) == len(syms)


def test_asset_classes_and_cell_cap():
    cells = ETF["asset_classes"]
    assert len(cells) == 10
    counts = {c: 0 for c in cells}
    for i in ETF["instruments"]:
        assert i["asset_class"] in cells
        assert i["symbol"] in cells[i["asset_class"]]
        counts[i["asset_class"]] += 1
    assert max(counts.values()) <= 3
    assert {s for v in cells.values() for s in v} == {i["symbol"] for i in ETF["instruments"]}


def test_date_ordering():
    for i in ETF["instruments"]:
        assert _d(i["inception_date"]) < _d(i["first_trade_date"]) <= _d(i["full_vol_window_date"]), i["symbol"]
        assert i["min_return_days_to_enter"] == 256
        assert i["ucits_variant"] is None
        assert _d(i["full_vol_window_date"]) <= _d(ETF["asof"])


def test_no_mutual_fund_proxy_is_a_member():
    members = {i["symbol"] for i in ETF["instruments"]}
    for p in ETF["mutual_fund_proxies"]:
        assert p["tradable"] is False and p["symbol"] not in members


def _scan(symbol: str) -> set[str]:
    pat = re.compile(r"""["']%s["']""" % re.escape(symbol))
    return {p.stem for p in (ROOT / "scripts").glob("*_preregistered*.py") if pat.search(p.read_text())}


def test_prior_exposure_matches_frozen_prereg_scan():
    for i in ETF["instruments"]:
        assert _scan(i["symbol"]) <= set(i["prior_exposure"]), i["symbol"]


def test_every_symbol_has_data_file():
    if not DATA.is_dir():
        pytest.skip("data directory absent (CI)")
    for i in ETF["instruments"]:
        assert (DATA / f"{i['symbol']}.parquet").exists(), i["symbol"]


def test_futures_draft_covers_frozen_markets():
    assert FUT["status"] == "draft"
    by = {i["symbol"]: i for i in FUT["instruments"]}
    for m in FROZEN_DRAFT_MARKETS:
        assert m in by and "dropped_reason" in by[m]
        assert by[m]["margin_unverified"] is True
