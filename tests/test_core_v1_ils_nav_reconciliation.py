"""Batch 23 audit of the candidate's ILS NAV (core_v1 gate 7). Synthetic data only; no ledger, no real data.

Invariant: with zero cost, zero tax and rf=None, the candidate's GROSS ILS NAV from ``candidate_navs`` equals the engine's USD equity
(``INITIAL * cumprod(1 + engine.returns)``) times the USD/ILS level path, to numerical tolerance, with and without a late FX entry.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_core_v1_evaluation as E

from firm.backtest import vector_engine as VE
from firm.costs.model import CostBreakdown
from firm.reporting import after_tax as AT
from firm.research import core_v1_pipeline as P

SYMS = ["AAA", "BBB", "CCC"]
N = 700
CAP = 100_000.0


def _consistent_panel(seed: int = 3, drift: float = 0.0004, yld: float = 0.006) -> P.Panel:
    """Raw prices with drift, quarterly dividends, adjusted close = provider total-return convention: adj_t/adj_{t-1} = (P_t + D_t)/P_{t-1}."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-04", periods=N)
    raw = pd.DataFrame(index=idx, columns=SYMS, dtype=float)
    adj = raw.copy()
    rows = []
    ex = {d for d in idx[(idx.month % 3 == 0) & (idx.day >= 15) & (idx.day <= 19)]}
    ex = {d for d in ex if d != idx[0]}
    ex = set(pd.Series(sorted(ex)).groupby(pd.Series(sorted(ex)).dt.to_period("M")).first())
    for j, s in enumerate(SYMS):
        r = drift * (1 + j) + rng.normal(0, 0.008, N)
        pr = 50.0 * (1 + j) * np.cumprod(1 + r)
        a = np.empty(N)
        a[0] = pr[0]
        for t in range(1, N):
            dv = yld * pr[t - 1] if idx[t] in ex else 0.0
            if dv:
                rows.append({"date": idx[t], "symbol": s, "amount": dv})
            a[t] = a[t - 1] * (pr[t] + dv) / pr[t - 1]
        raw[s], adj[s] = pr, a
    ret = adj.pct_change()
    vol = pd.DataFrame(1e6, index=idx, columns=SYMS)
    fx = pd.Series(4.0 * np.cumprod(1 + rng.normal(-0.0002, 0.003, N)), index=idx)
    return P._finish_panel(SYMS, adj, raw, vol, ret, "synthetic", pd.DataFrame(rows), fx, "synthetic")


def _engine(panel: P.Panel) -> SimpleNamespace:
    idx = panel.close.index
    month = pd.Series(idx.year * 100 + idx.month, index=idx)
    tg = pd.DataFrame(0.0, index=idx, columns=SYMS)
    for j, s in enumerate(SYMS):
        first = panel.close[s].groupby(month).transform("first")
        sign = 1.0 if j != 1 else 0.6
        tg[s] = sign * (60_000.0 + 20_000.0 * j) / first      # ~1.9x gross: levered, as the real book is (negative cash)
    tg.iloc[:5] = 0.0
    zero = lambda *a, **k: CostBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    eng = VE.run_vector_backtest(panel.close, tg, dict.fromkeys(SYMS, 1.0), zero, VE.EngineConfig(initial_capital=CAP, allow_short=False),
                                 adv=panel.adv, vol_pct=panel.ret.fillna(0.01).abs() * 0 + 0.01,
                                 ledger_ctx=VE.LedgerContext(family="core_v1_engine", mode="exploratory", data_snapshot_id="synthetic", seed=1,
                                                             config={}))
    return SimpleNamespace(engine=eng)


@pytest.mark.parametrize("entry_i", [None, 300])
def test_gross_ils_nav_reproduces_engine_return_times_fx(entry_i):
    pn = _consistent_panel()
    sel = _engine(pn)
    cfg = AT.load_tax_config(ROOT / "config" / "tax_il.yaml")
    idx = pn.close.index
    entry = None if entry_i is None else idx[entry_i]
    nav = E.candidate_navs(sel, pn, cfg, idx[260:], entry=entry)["gross"]
    eq = CAP * (1.0 + sel.engine.returns).cumprod()
    start = nav.index[0]
    want = eq.loc[nav.index] * pn.fx.loc[nav.index]
    assert nav.index.equals(eq.loc[start:].index)
    rel = (nav / want - 1.0)
    # entry mode starts at the engine equity of the entry day: levels match from there, so the comparison is on the whole path
    assert rel.abs().max() < 1e-6, f"gross ILS NAV deviates from engine equity x FX: max rel err {rel.abs().max():.3e}, end {rel.iloc[-1]:.3e}"
