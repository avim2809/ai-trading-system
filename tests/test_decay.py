"""P5-03: CUSUM decay monitor."""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from firm.monitoring.decay import (
    CusumConfig,
    CusumState,
    calibrate_h,
    calibrate_h_detailed,
    calibrate_h_gate,
    cusum_run,
    cusum_step,
    siegmund_arl,
    simulate_run_lengths,
    t_test_36m,
)
from firm.validation.sharpe_stats import SharpeUnitsError

ARL0_SMALL = 250
SEED_CAL = 1001  # calibration seed A
SEED_VER = 2002  # independent verification seed B
CFG0 = CusumConfig(mu0=0.05, mu1=0.0, sigma=1.0, h=0.0)  # placeholder h: calibrate_h ignores it


@functools.cache
def _cal_small():
    return calibrate_h_detailed(CFG0, arl0_periods=ARL0_SMALL, n_sims=3000, seed=SEED_CAL)


def _h_small() -> float:
    return _cal_small().h


def _with_h(cfg: CusumConfig, h: float) -> CusumConfig:
    return CusumConfig(mu0=cfg.mu0, mu1=cfg.mu1, sigma=cfg.sigma, h=h, k=cfg.k)


def test_in_control_arl_within_tolerance():
    cfg = _with_h(CFG0, _h_small())
    rl = simulate_run_lengths(cfg, n_sims=3000, seed=SEED_VER)
    assert rl.mean() == pytest.approx(ARL0_SMALL, rel=0.20)
    # calibration on a different seed lands near the target on its own draws
    detail = _cal_small()
    assert detail.seed == SEED_CAL
    assert detail.arl_achieved == pytest.approx(ARL0_SMALL, rel=0.05)
    assert detail.arl_se > 0


def test_detection_delay_shift_to_sr_minus_0_3():
    """mu0 = haircut SR 0.25, mu1 = SR -0.3 (annual), per-period standardised, ARL0 = 1260."""
    per = math.sqrt(252)
    cfg0 = CusumConfig(mu0=0.25 / per, mu1=-0.3 / per, sigma=1.0, h=0.0)
    h = calibrate_h(cfg0, arl0_periods=1260, n_sims=2000, seed=SEED_CAL)
    cfg = _with_h(cfg0, h)
    rl = simulate_run_lengths(cfg, n_sims=2000, seed=SEED_VER, true_mean=cfg.mu1)
    delay = float(rl.mean())
    print(
        f"\nmean detection delay {delay:.0f} periods ({delay / 252:.2f} y), "
        f"median {np.median(rl):.0f}, h={h:.3f}"
    )
    ref = siegmund_arl(cfg, true_mean=cfg.mu1)  # independent analytic reference
    assert delay == pytest.approx(ref, rel=0.20)
    assert 2.0 <= delay / 252 <= 3.2  # source expectation: about 2.5 years (mean ~ 625 days)


def test_detects_large_shift_fast():
    cfg = _with_h(CFG0, _h_small())
    rl = simulate_run_lengths(cfg, n_sims=2000, seed=SEED_VER, true_mean=cfg.mu0 - 2 * cfg.sigma)
    assert np.median(rl) < 0.2 * ARL0_SMALL


def test_in_control_alarm_rate():
    cfg = _with_h(CFG0, _h_small())
    rl = simulate_run_lengths(cfg, n_sims=3000, seed=SEED_VER + 1)
    for n in (250, 500, 750):
        frac = float((rl <= n).mean())
        assert frac == pytest.approx(1 - math.exp(-n / ARL0_SMALL), abs=0.06)


def test_default_k_uses_sigma():
    cfg = CusumConfig(mu0=0.10, mu1=0.0, sigma=2.0, h=5.0)
    assert cfg.k_eff == pytest.approx((0.10 - 0.0) / (2 * 2.0))
    assert CusumConfig(mu0=0.10, mu1=0.0, sigma=2.0, h=5.0, k=0.3).k_eff == 0.3
    # sigma enters the standardised increment: z = (x - mu0) / sigma
    st = cusum_step(CusumState(0.0, 0, False, None), -2.0, cfg)
    assert st.s == pytest.approx(max(0.0, (2.0 + 0.10) / 2.0 - cfg.k_eff))


def test_t_test_36m_rolling_one_sided():
    rng = np.random.default_rng(5)
    old = rng.normal(0.05, 0.01, 40)  # must be ignored
    low = np.concatenate([old, rng.normal(-0.01, 0.02, 36)])
    high = np.concatenate([old * -5, rng.normal(0.03, 0.02, 36)])
    t_lo, p_lo = t_test_36m(low, 0.0)
    t_hi, p_hi = t_test_36m(high, 0.0)
    assert t_lo < 0 < t_hi
    assert p_lo < 0.05 and p_hi > 0.95
    # uses the last 36 only
    t_last, p_last = t_test_36m(low[-36:], 0.0)
    assert (t_lo, p_lo) == (t_last, p_last)
    x = low[-36:]
    assert t_last == pytest.approx((x.mean() - 0.0) / (x.std(ddof=1) / 6.0))
    with pytest.raises(ValueError):
        t_test_36m(low[-35:], 0.0)


def test_cusum_step_equals_cusum_run():
    cfg = _with_h(CFG0, 3.0)
    x = np.random.default_rng(3).normal(-0.3, 1.0, 400)
    st = CusumState(0.0, 0, False, None)
    for v in x:
        st = cusum_step(st, float(v), cfg)
    assert st == cusum_run(x, cfg)
    assert st.n == 400 and st.alarm and st.first_alarm_index is not None


def test_reset_policy():
    cfg = _with_h(CFG0, 2.0)
    st = cusum_run(np.full(200, -1.0), cfg)
    assert st.alarm and st.first_alarm_index is not None
    first = st.first_alarm_index
    for _ in range(500):  # strongly favourable data does not clear the alarm
        st = cusum_step(st, 5.0, cfg)
    assert st.alarm and st.first_alarm_index == first
    assert st.s == 0.0  # statistic itself may decay; the flag may not


def test_unit_guard():
    with pytest.raises(SharpeUnitsError):
        CusumConfig(mu0=1.5, mu1=-0.5, sigma=1.0, h=3.0)  # annualised Sharpe as per-period mu
    with pytest.raises(SharpeUnitsError):
        CusumConfig(mu0=0.01, mu1=-3.0, sigma=1.0, h=3.0)
    with pytest.raises(ValueError):
        CusumConfig(mu0=0.01, mu1=0.02, sigma=1.0, h=3.0)  # not a decay alternative
    with pytest.raises(ValueError):
        CusumConfig(mu0=0.02, mu1=0.0, sigma=0.0, h=3.0)


def test_deterministic_given_seed():
    a = calibrate_h(CFG0, arl0_periods=100, n_sims=500, seed=7)
    b = calibrate_h(CFG0, arl0_periods=100, n_sims=500, seed=7)
    c = calibrate_h(CFG0, arl0_periods=100, n_sims=500, seed=8)
    assert a == b and a != c
    cfg = _with_h(CFG0, a)
    r1 = simulate_run_lengths(cfg, n_sims=200, seed=1)
    assert np.array_equal(r1, simulate_run_lengths(cfg, n_sims=200, seed=1))


def test_bootstrap_variant_calibrates_to_target():
    rng = np.random.default_rng(11)
    sample = rng.standard_t(3, 3000) * 0.5  # fat-tailed in-control sample (raw units, mean ~0)
    d = calibrate_h_detailed(CFG0, arl0_periods=100, n_sims=1500, seed=SEED_CAL, sample=sample)
    assert d.method == "stationary_bootstrap"
    assert d.arl_achieved == pytest.approx(100, rel=0.08)
    cfg = _with_h(CFG0, d.h)
    rl = simulate_run_lengths(cfg, n_sims=1500, seed=SEED_VER, sample=sample)
    assert rl.mean() == pytest.approx(100, rel=0.20)
    gate = calibrate_h_gate(CFG0, arl0_periods=100, n_sims=1500, seed=SEED_CAL, sample=sample)
    normal = calibrate_h_detailed(CFG0, arl0_periods=100, n_sims=1500, seed=SEED_CAL)
    assert gate.h == max(d.h, normal.h)
