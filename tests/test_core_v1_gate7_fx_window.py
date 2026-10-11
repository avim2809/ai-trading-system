"""Gate-7 FX-covered sub-window (owner decision 2026-10-11, option 1). Synthetic data only; tmp ledger (autouse fixture); no real data.

USD/ILS can begin after the frozen evaluation window. The after-tax benchmark and the gate-7 comparison are then judged on the FX-covered
sub-window only; the candidate's ILS NAVs must be truncated identically; every other gate keeps the full window.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_core_v1_evaluation as E

from firm.reporting import after_tax as AT
from firm.reporting import gate_report as GR
from firm.research import core_v1_pipeline as P

GATES = yaml.safe_load((ROOT / "config" / "gates.yaml").read_text())
SYMS = ("SPY", "IEF", "GLD")
N_DAYS = 900


@pytest.fixture(scope="module")
def panel():
    return P.synthetic_panel(seed=5, n_days=N_DAYS, symbols=SYMS, start="2010-01-04")


def _late_fx(panel, k: int) -> pd.Series:
    return panel.fx.iloc[k:]


def _stub_sel(panel) -> SimpleNamespace:
    """A candidate that holds an equal-weight SPY/IEF/GLD book re-set monthly (adjusted units), no engine costs."""
    idx = panel.close.index
    pos = pd.DataFrame(0.0, index=idx, columns=list(SYMS))
    month = pd.Series(idx.year * 100 + idx.month, index=idx)
    for s in SYMS:
        first = panel.close[s].groupby(month).transform("first")
        pos[s] = 30_000.0 / first
    rets = pd.Series(0.0003, index=idx) + panel.ret[list(SYMS)].mean(axis=1).fillna(0.0) * 0.5
    return SimpleNamespace(engine=SimpleNamespace(positions=pos, trades=pd.DataFrame(), returns=rets, excess_returns=rets))


def _run(panel, fx_start_i: int | None, eval_start_i: int = 260, min_days: int = 100):
    pn = P.Panel(**{**panel.__dict__, "fx": panel.fx if fx_start_i is None else _late_fx(panel, fx_start_i)})
    eval_index = pn.close.index[eval_start_i:]
    cfg = AT.load_tax_config(ROOT / "config" / "tax_il.yaml")
    ctx = P.LedgerCtx(mode="exploratory", family="core_v1_test", snapshot_id="synthetic", seed=1)
    out = E.run_benchmark(_stub_sel(pn), pn, None, None, GATES, ctx, cfg, P.CM.load_cost_config(), eval_index, 200, 7, False, min_fx_window_days=min_days)
    return pn, eval_index, out


# ---- the window helper -------------------------------------------------------------------------------------------------------------------
def test_window_is_derived_from_the_fx_series_not_hard_coded(panel):
    pn = P.Panel(**{**panel.__dict__, "fx": _late_fx(panel, 400)})
    ei = pn.close.index[260:]
    w = E.gate7_window(pn, ei, min_days=100)
    assert w["fx_first_date"] == str(pn.fx.index[0].date()) and w["fx_truncated"] is True
    assert w["window_start"] == w["fx_first_date"] and w["window_end"] == str(ei[-1].date())
    assert w["full_window_start"] == str(ei[0].date()) and w["n_days"] == int((ei >= pn.fx.index[0]).sum())
    assert "FX-covered sub-window" in w["deviation"] and "USD/ILS" in w["deviation"]


def test_fx_covering_the_whole_window_is_not_truncated(panel):
    ei = panel.close.index[260:]
    w = E.gate7_window(panel, ei, min_days=100)
    assert w["fx_truncated"] is False and w["window_start"] == w["full_window_start"] and w["n_days"] == len(ei) and w["deviation"] is None


def test_window_leading_nan_fx_is_ignored_for_the_start(panel):
    fx = panel.fx.copy()
    fx.iloc[:350] = np.nan
    pn = P.Panel(**{**panel.__dict__, "fx": fx})
    assert E.gate7_window(pn, pn.close.index[260:], min_days=100)["fx_first_date"] == str(fx.index[350].date())


def test_too_short_fx_window_fails_closed_naming_the_dates(panel):
    pn = P.Panel(**{**panel.__dict__, "fx": _late_fx(panel, N_DAYS - 50)})
    with pytest.raises(E.StartupError) as ei:
        E.gate7_window(pn, pn.close.index[260:], min_days=1260)
    msg = str(ei.value)
    assert str(pn.fx.index[0].date()) in msg and str(pn.close.index[-1].date()) in msg and "1260" in msg


def test_default_minimum_is_five_years_of_trading_days():
    assert E.MIN_FX_WINDOW_DAYS == 1260


def test_missing_fx_fails_closed(panel):
    pn = P.Panel(**{**panel.__dict__, "fx": None})
    with pytest.raises(E.StartupError):
        E.gate7_window(pn, pn.close.index[260:], min_days=100)


# ---- run_benchmark: candidate and benchmark cover identical dates ------------------------------------------------------------------------
def test_truncated_benchmark_runs_and_records_the_window(panel):
    pn, ei, out = _run(panel, 500)
    fx0 = pn.fx.index[0]
    assert out["fx_truncated"] is True and out["fx_first_date"] == str(fx0.date())
    assert out["window_start"] == str(fx0.date()) and out["window_end"] == str(ei[-1].date()) and out["full_window_start"] == str(ei[0].date())
    assert out["n_days"] == int((ei >= fx0).sum())
    assert "FX-covered sub-window" in out["deviation"]


def test_candidate_and_benchmark_navs_have_identical_dates(panel):
    pn, ei, _ = _run(panel, 500)
    cfg = E.no_cpi_config(AT.load_tax_config(ROOT / "config" / "tax_il.yaml"))
    fx0 = pn.fx.index[0]
    navs = E.candidate_navs(_stub_sel(pn), pn, cfg, ei, entry=fx0)
    assert all(v.index[0] >= fx0 for v in navs.values()) and navs["after_tax"].notna().all()
    px = pn.raw_close.loc[fx0:, ["SPY", "IEF"]]
    assert navs["after_tax"].index.equals(px.index)


def test_mde_uses_the_truncated_length(panel):
    _, _, full = _run(panel, None)
    _, _, short = _run(panel, 500)
    assert full["fx_truncated"] is False and short["n_days"] < full["n_days"]
    assert short["gap_standard_error"] > full["gap_standard_error"]          # fewer days, wider interval
    assert short["minimum_detectable_sharpe_gap"] > full["minimum_detectable_sharpe_gap"]
    assert short["minimum_detectable_sharpe_gap"] == short["margin"]


def test_core_only_100_and_pre_tax_numbers_keep_the_full_window(panel):
    _, _, full = _run(panel, None)
    _, _, short = _run(panel, 500)
    assert short["core_only_100"] == full["core_only_100"]


def test_run_benchmark_fails_closed_when_the_subwindow_is_too_short(panel):
    with pytest.raises(E.StartupError):
        _run(panel, N_DAYS - 40, min_days=1260)


def test_pinned_annual_variant_and_monthly_sensitivity_unchanged(panel):
    _, _, out = _run(panel, 500)
    assert out["gate_variant"] == "annual" and out["sensitivity_variant"] == "monthly"


# ---- the report states the deviation -----------------------------------------------------------------------------------------------------
def test_gate7_outcome_and_report_state_the_truncation(panel):
    _, _, out = _run(panel, 500)
    o = GR._g7(out, GATES["g_research"])
    assert "FX-covered sub-window" in o.reason and out["window_start"] in o.reason
    md = GR.render_report([o], "C", {"deviations": [out["deviation"]]}, {})
    assert md.count("FX-covered sub-window") >= 2
