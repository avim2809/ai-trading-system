"""Tests for firm.validation.stress_suite (P3-07). Synthetic data only."""

from __future__ import annotations

import inspect
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from firm.validation import stress_suite
from firm.validation.stress_suite import (
    StressPeriod,
    load_stress_periods,
    run_stress_suite,
    suite_summary,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = [
    ("GFC", date(2008, 9, 1), date(2009, 3, 31)),
    ("Flash crash / euro", date(2010, 4, 15), date(2010, 7, 15)),
    ("US downgrade", date(2011, 7, 15), date(2011, 10, 15)),
    ("Volmageddon", date(2018, 1, 26), date(2018, 4, 30)),
    ("Q4 2018", date(2018, 10, 1), date(2018, 12, 31)),
    ("COVID", date(2020, 2, 15), date(2020, 4, 30)),
    ("2022 inflation", date(2022, 1, 1), date(2022, 10, 31)),
    ("Regional banks", date(2023, 3, 1), date(2023, 3, 31)),
    ("Yen-carry", date(2024, 7, 15), date(2024, 8, 31)),
    ("Liberation Day", date(2025, 3, 15), date(2025, 5, 31)),
]


def _series(vals, start="2020-01-06"):
    return pd.Series(vals, index=pd.bdate_range(start, periods=len(vals)))


def _period(s, n):
    idx = pd.bdate_range(s, periods=n)
    return StressPeriod("p", idx[0].date(), idx[-1].date())


def _run(r, period, ref=0.1, positions=None, **kw):
    kw.setdefault("min_active_fraction", 0.5)
    return run_stress_suite(r, positions, [period], target_vol=0.1, reference_max_dd=ref, **kw)[0]


def test_config_has_ten_periods_exact_dates():
    got = [(p.name, p.start, p.end) for p in load_stress_periods(str(ROOT / "config/stress_periods.yaml"))]
    assert got == EXPECTED


def test_all_periods_before_seal():
    seal = yaml.safe_load((ROOT / "config/research_freeze.yaml").read_text())["seal_date"]
    for p in load_stress_periods(str(ROOT / "config/stress_periods.yaml")):
        assert p.end < seal


def test_hand_computed_episode():
    r = _series([0.10, -0.20, 0.05, -0.10, 0.02])
    res = _run(r, _period("2020-01-06", 5))
    eq = [1.1, 0.88, 0.924, 0.8316, 0.848232]
    assert res.total_return == pytest.approx(eq[-1] - 1, abs=1e-12)
    assert res.max_drawdown == pytest.approx(1 - 0.8316 / 1.1, abs=1e-12)
    assert res.worst_day == pytest.approx(-0.20, abs=1e-12)
    vol = np.std([0.10, -0.20, 0.05, -0.10, 0.02], ddof=1) * np.sqrt(256)
    assert res.realised_vol == pytest.approx(vol, abs=1e-12)
    assert res.vol_ratio == pytest.approx(vol / 0.1, abs=1e-12)
    assert res.n_days == 5 and res.status == "ok"


def test_drawdown_starts_at_episode_start():
    r = _series([-0.5, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01])
    res = _run(r, _period("2020-01-08", 5))
    assert res.max_drawdown == 0.0


def test_breach_flag_at_1p5x():
    # single 0.15 loss day vs reference 0.1 -> exactly 1.5x
    r = _series([0.0, -0.15, 0.0, 0.0, 0.0])
    p = _period("2020-01-06", 5)
    assert _run(r, p, ref=0.1).breach is False
    assert _run(_series([0.0, -0.1501, 0.0, 0.0, 0.0]), p, ref=0.1).breach is True


def test_reference_callable_gets_n_days():
    seen = []
    r = _series([0.0, -0.2, 0.0, 0.0, 0.0, 0.0])
    _run(r, _period("2020-01-06", 6), ref=lambda n: seen.append(n) or 0.1)
    assert seen == [6]


def _pos(n_days, starts):
    idx = pd.bdate_range("2020-01-06", periods=n_days)
    df = pd.DataFrame(1.0, index=idx, columns=list(starts))
    for c, k in starts.items():
        df.loc[idx[:k], c] = np.nan
    return df


def test_active_instruments_counted():
    n = 20
    pos = _pos(n, {"A": 0, "B": 6, "C": 12})
    r = _series([0.001] * n)
    early = _run(r, _period("2020-01-06", 5), positions=pos, min_active_fraction=0.0)
    late = StressPeriod("l", pos.index[12].date(), pos.index[17].date())
    later = _run(r, late, positions=pos, min_active_fraction=0.0)
    assert (early.n_active_instruments, early.active_instruments) == (1, ["A"])
    assert later.n_active_instruments == 3
    assert _run(r, _period("2020-01-06", 5)).n_active_instruments == -1


def test_short_window_reported_not_dropped():
    r = _series([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    res = run_stress_suite(r, None, [_period("2020-01-06", 3)], 0.1, 0.1, min_active_fraction=0.5)
    assert len(res) == 1 and res[0].status == "unusable" and res[0].n_days == 3 and not res[0].breach


def test_empty_window_reported():
    r = _series([0.01] * 10)
    res = run_stress_suite(r, None, [StressPeriod("x", date(2008, 9, 1), date(2009, 3, 31))], 0.1, 0.1,
                           min_active_fraction=0.5)
    assert res[0].n_days == 0 and res[0].status == "unusable"


def test_unusable_and_low_coverage_counted():
    n = 20
    pos = _pos(n, {"A": 0, "B": 15, "C": 15, "D": 15})
    r = _series([0.001] * n)
    periods = [_period("2020-01-06", 3), _period("2020-01-06", 10), StressPeriod("z", date(2019, 1, 1), date(2019, 2, 1))]
    res = run_stress_suite(r, pos, periods, 0.1, 0.1, min_active_fraction=0.5)
    s = suite_summary(res)
    assert s["n_unusable"] == 2 and s["n_low_coverage"] == 1 and s["all_ok"] is False
    assert [x.status for x in res] == ["unusable", "low_coverage", "unusable"]


def test_not_applicable_when_no_active():
    n = 10
    pos = _pos(n, {"A": 10})
    res = _run(_series([0.001] * n), _period("2020-01-06", 8), positions=pos)
    assert res.status == "not_applicable"
    assert suite_summary([res])["n_not_applicable"] == 1


def test_summary_worst_and_breach():
    r = _series([0.0, -0.2, 0.0, 0.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0])
    ps = [StressPeriod("a", r.index[0].date(), r.index[4].date()), StressPeriod("b", r.index[5].date(), r.index[9].date())]
    s = suite_summary(run_stress_suite(r, None, ps, 0.1, 0.1, min_active_fraction=0.5))
    assert s["worst_episode"] == "a" and s["any_breach"] and s["n_breached"] == 1 and s["all_ok"]


def test_reference_required():
    r = _series([0.0] * 6)
    with pytest.raises(TypeError):
        run_stress_suite(r, None, [_period("2020-01-06", 5)], target_vol=0.1, min_active_fraction=0.5)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        run_stress_suite(r, None, [_period("2020-01-06", 5)], 0.1, 0.1)  # type: ignore[call-arg]


def test_no_tuning_api():
    for name, fn in inspect.getmembers(stress_suite, inspect.isfunction):
        if name.startswith("_") or fn.__module__ != stress_suite.__name__:
            continue
        for pname, prm in inspect.signature(fn).parameters.items():
            ann = str(prm.annotation)
            if pname == "reference_max_dd":  # the only callable: a pre-committed reference by episode length
                continue
            assert "Callable" not in ann, (name, pname)
            assert "grid" not in pname.lower() and "param" not in pname.lower(), (name, pname)


def test_import_light():
    code = (
        "import sys, firm.validation.stress_suite;"
        "bad=[m for m in sys.modules if m.split('.')[:2] in (['firm','live'],['firm','api'],['firm','runtime'])];"
        "print('BAD:'+','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin"}).stdout
    assert out.strip() == "BAD:"
