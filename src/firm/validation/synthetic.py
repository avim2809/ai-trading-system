"""Synthetic GARCH(1,1)-t validation harness for the statistics pipeline (credibility plan P1-08).

Bollerslev (1986) GARCH, Bollerslev (1987) Student-t errors. Research-only; never imported by a live
module (``tests/test_live_import_isolation.py``). Every scenario parameter (K, T, SR, GARCH, B, n_sims,
seeds, alpha) comes from the loaded :class:`Scenario` spec, which :func:`load_acceptance` builds from the
frozen ``p1_08_acceptance`` block of ``config/gates.yaml``. Acceptance bars are READ from the same block by
:func:`evaluate`; none is written in this file. The harness may not be changed to make a module pass.

Per-sim seeds are ``base_seed + sim_index``; results are bit-reproducible for a given spec.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.signal import lfilter

from firm.validation.bootstrap import politis_white_block_length, stationary_bootstrap_indices
from firm.validation.cv import CombinatorialPurgedCV
from firm.validation.multiple_testing import reality_check, romano_wolf, spa_test
from firm.validation.pbo import pbo as cscv_pbo
from firm.validation.pbo import sharpe_cols
from firm.validation.sharpe_stats import dsr as dsr_fn
from firm.validation.sharpe_stats import length_adjusted_var_sr, moments, psr

log = logging.getLogger(__name__)

PERIODS_PER_YEAR = 252
FREEZE_DRAFT = Path("plan/drafts/P1-08/scenario_freeze.yaml")
SEED_STRIDE = 1000  # base_seed = master_seed + SEED_STRIDE * scenario_index (frozen in scenario_freeze.yaml)
ORACLE_SEED_OFFSET = 10_000_000  # oracle null sims use base_seed + this + i (disjoint from every scenario)
PILOT_SEED = 20261004  # fixed pilot stream used to calibrate the trend strength in expectation
PILOT_N = 100_000


# ------------------------------------------------------------------ generator


@dataclass(frozen=True)
class GarchTParams:
    omega: float = 5e-6
    alpha: float = 0.05
    beta: float = 0.90
    nu: float = 6.0
    mu: float = 0.0  # unconditional daily vol sqrt(5e-6/0.05) = 1%

    def unconditional_vol(self) -> float:
        return math.sqrt(self.omega / (1.0 - self.alpha - self.beta))

    def finite_fourth_moment(self) -> bool:
        """``beta^2 + 2*alpha*beta + kappa*alpha^2 < 1`` with ``kappa = 3(nu-2)/(nu-4)`` (needs nu > 4)."""
        if self.nu <= 4.0:
            return False
        kappa = 3.0 * (self.nu - 2.0) / (self.nu - 4.0)
        return self.beta**2 + 2.0 * self.alpha * self.beta + kappa * self.alpha**2 < 1.0

    def validate(self) -> None:
        if not self.omega > 0.0 or self.alpha < 0.0 or self.beta < 0.0:
            raise ValueError("GARCH needs omega > 0, alpha >= 0, beta >= 0")
        if self.alpha + self.beta >= 1.0:
            raise ValueError(f"GARCH not stationary: alpha+beta={self.alpha + self.beta} >= 1")
        if not self.nu > 2.0:
            raise ValueError(f"Student-t needs nu > 2 for finite variance, got {self.nu}")


# Documented stress scenario: INFINITE fourth moment (0.81+0.144+9*0.0064 = 1.0116), unconditional vol ~0.71%.
STRESS_PARAMS = GarchTParams(omega=1e-6, alpha=0.08, beta=0.90, nu=5.0)


def standardised_t(shape, nu: float, rng: np.random.Generator) -> np.ndarray:
    """Student-t(nu) draws scaled to unit variance: ``t * sqrt((nu-2)/nu)``."""
    return rng.standard_t(nu, size=shape) * math.sqrt((nu - 2.0) / nu)


def _garch_matrix(n: int, k: int, params: GarchTParams, rng: np.random.Generator, burn: int, corr: float) -> np.ndarray:
    """``n x k`` GARCH-t returns; innovations are ``sqrt(corr)*common + sqrt(1-corr)*idio`` (unit variance)."""
    params.validate()
    if not 0.0 <= corr < 1.0:
        raise ValueError("corr must be in [0, 1)")
    total = n + burn
    z = standardised_t((total, k), params.nu, rng)
    if corr > 0.0 and k > 1:
        common = standardised_t((total, 1), params.nu, rng)
        z = math.sqrt(corr) * common + math.sqrt(1.0 - corr) * z
    s2 = np.full(k, params.omega / (1.0 - params.alpha - params.beta))
    out = np.empty((total, k))
    for t in range(total):
        e = np.sqrt(s2) * z[t]
        out[t] = params.mu + e
        s2 = params.omega + params.alpha * e * e + params.beta * s2
    return out[burn:]


def simulate_garch_t(n: int, params: GarchTParams, seed: int, burn: int = 500) -> np.ndarray:
    """``r_t = mu + sigma_t z_t``; ``sigma_t^2 = omega + alpha (r_{t-1}-mu)^2 + beta sigma_{t-1}^2``; z unit-variance t."""
    rng = np.random.default_rng(seed)
    return _garch_matrix(n, 1, params, rng, burn, 0.0)[:, 0]


def inject_drift(r: np.ndarray, target_annual_sr: float, periods_per_year: int = PERIODS_PER_YEAR) -> np.ndarray:
    """Constant mean so that buy-and-hold has the target annualised Sharpe."""
    r = np.asarray(r, dtype=float)
    return r + target_annual_sr / math.sqrt(periods_per_year) * float(r.std())


def _trend_path(r: np.ndarray, phi: float, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Recursive trend injection: ``x_t = r_t + phi*s_{t-1}``, ``s_{t-1} = sign(sum x_{t-h..t-1})``."""
    n = len(r)
    x = np.empty(n)
    s = np.zeros(n)
    run = 0.0
    for t in range(n):
        if t > h:
            run -= x[t - h - 1]  # window is exactly x[t-h .. t-1]
        if t >= h:
            s[t] = 1.0 if run > 0.0 else -1.0
        x[t] = r[t] + phi * s[t]
        run += x[t]
    return x, s


def _trend_sr(r: np.ndarray, phi: float, h: int, ppy: int) -> float:
    x, s = _trend_path(r, phi, h)
    st = (s * x)[h:]
    return float(st.mean() / st.std(ddof=1) * math.sqrt(ppy))


@lru_cache(maxsize=64)
def _trend_phi_unit(params: GarchTParams, h: int, target: float, ppy: int) -> float:
    """phi per unit daily vol, by bisection on a long pilot so the rule hits ``target`` in EXPECTATION."""
    pilot = simulate_garch_t(PILOT_N, params, PILOT_SEED)
    pilot = pilot / float(pilot.std())
    lo, hi = 0.0, 1.0
    while _trend_sr(pilot, hi, h, ppy) < target and hi < 1e3:
        hi *= 2.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if _trend_sr(pilot, mid, h, ppy) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def inject_trend(
    r: np.ndarray,
    target_annual_sr: float,
    horizon_days: int,
    seed: int = 0,
    periods_per_year: int = PERIODS_PER_YEAR,
    params: GarchTParams | None = None,
) -> np.ndarray:
    """Add ``phi*s_{t-1}``, ``s`` = sign of the past ``horizon_days`` return (21..252), so that the rule
    "long when the past return is positive, else short" has the target annualised Sharpe in expectation.
    ``phi`` is solved by bisection on a long fixed-seed pilot of the same GARCH-t (scaled to ``r``'s vol)."""
    r = np.asarray(r, dtype=float)
    if not 1 <= horizon_days < len(r):
        raise ValueError("horizon_days must be in [1, len(r))")
    p = params or GarchTParams()
    phi = _trend_phi_unit(p, int(horizon_days), float(target_annual_sr), int(periods_per_year)) * float(r.std())
    return _trend_path(r, phi, int(horizon_days))[0]


def inject_carry(
    r: np.ndarray, target_annual_sr: float, seed: int, periods_per_year: int = PERIODS_PER_YEAR
) -> np.ndarray:
    """Add a slowly varying predictable mean (AR(1), half-life ~60 days, sd = half its level)."""
    r = np.asarray(r, dtype=float)
    level = target_annual_sr / math.sqrt(periods_per_year) * float(r.std())
    a = 0.5 ** (1.0 / 60.0)
    eps = np.random.default_rng(seed).standard_normal(len(r))
    x = lfilter([math.sqrt(1.0 - a * a)], [1.0, -a], eps)  # unit-variance AR(1)
    return r + level * (1.0 + 0.5 * x)


def make_universe(
    n_assets: int, n_days: int, seed: int, corr: float = 0.3, signal: dict | None = None,
    params: GarchTParams | None = None,
) -> np.ndarray:
    """``T x K`` equicorrelated GARCH-t returns; ``signal`` = ``{"kind", "annual_sr", "index", "horizon_days"}``
    injects drift / trend / carry into column ``index`` (the true strategy)."""
    p = params or GarchTParams()
    rng = np.random.default_rng(seed)
    U = _garch_matrix(n_days, n_assets, p, rng, 500, corr)
    if signal:
        j = int(signal.get("index", 0))
        kind, sr = signal["kind"], float(signal["annual_sr"])
        if kind == "drift":
            U[:, j] = inject_drift(U[:, j], sr)
        elif kind == "trend":
            U[:, j] = inject_trend(U[:, j], sr, int(signal.get("horizon_days", 126)), seed + 1, params=p)
        elif kind == "carry":
            U[:, j] = inject_carry(U[:, j], sr, seed + 1)
        else:
            raise ValueError(f"unknown signal kind {kind!r}")
    return U


# ------------------------------------------------------------------ scenarios


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: Literal["null", "drift", "trend", "carry"]
    k_trials: int
    n_days: int
    annual_sr: float
    garch: GarchTParams
    B: int
    n_sims: int
    base_seed: int
    alpha: float
    # extensions (all from the frozen block / harness structure, never tuned)
    corr: float = 0.3
    ar_rho: float = 0.0
    S: int = 16
    horizon_days: int = 126
    randomise_signal: bool = True
    tests: tuple[str, ...] = field(default_factory=tuple)
    oracle: bool = False
    cpcv: bool = False
    ledger: bool = False
    iid_control: bool = False
    non_binding: bool = False  # reported only, never a pass bar


def wilson(k: int, n: int, alpha: float) -> dict:
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    p = k / n if n else float("nan")
    if not n:
        return {"k": k, "n": n, "rate": p, "lo": float("nan"), "hi": float("nan")}
    den = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"k": int(k), "n": int(n), "rate": float(p), "lo": float(max(0.0, c - h)), "hi": float(min(1.0, c + h))}


def _block_hash(block: dict) -> str:
    return hashlib.sha256(json.dumps(block, sort_keys=True, default=str).encode()).hexdigest()


def load_acceptance(gates_path: Path = Path("config/gates.yaml")) -> dict:
    """``{"scenarios": [Scenario], "bars": {...}, "scenarios_sha256": str, ...}`` from the frozen block."""
    gates_path = Path(gates_path)
    block = yaml.safe_load(gates_path.read_text())["p1_08_acceptance"]
    sc = block["scenarios"]
    freeze_source = str(gates_path)
    corr, master = sc.get("cross_correlation"), sc.get("master_seed")
    if isinstance(corr, dict) or isinstance(master, dict):
        fz_path = Path(__file__).resolve().parents[3] / FREEZE_DRAFT
        fz = yaml.safe_load(fz_path.read_text())
        corr = fz["cross_correlation"] if isinstance(corr, dict) else corr
        master = fz["master_seed"] if isinstance(master, dict) else master
        freeze_source = f"{fz_path} (gates.yaml defers these two values to P1-08)"
    g = sc["garch"]
    garch = GarchTParams(omega=g["omega"], alpha=g["alpha"], beta=g["beta"], nu=g["nu"], mu=g["mu"])
    T, B = int(sc["T"]), int(sc["bootstrap_B"])
    size, power, pbo_b = block["size"], block["power"], block["pbo"]
    n_sims, alpha, K = int(size["n_sims"]), float(size["alpha"]), int(size["K"])
    sd = pbo_b["strong_drift"]
    single, multi, rep = power["single_test"], power["multiple_testing"], power["reported_only"]
    ppy = int(sc["periods_per_year"])
    if ppy != PERIODS_PER_YEAR:
        raise ValueError("harness generators assume the frozen periods_per_year")
    T_single = int(single["years"]) * ppy
    base = {"garch": garch, "B": B, "alpha": alpha, "corr": float(corr)}
    bt = ("rc", "spa", "rw", "dsr")
    # (name, kind, K, T, SR, sims, extra) in fixed order: seed = master + SEED_STRIDE * index
    # The K=10 / SR 1.5 reported scenario is defined by the ticket text, not by gates.yaml (reported only).
    rows = [
        ("null_size", "null", K, T, 0.0, n_sims, {"tests": bt + ("psr0",)}),
        ("null_pbo", "null", K, T, 0.0, int(pbo_b["noise_matrices"]), {"tests": ("pbo",), "S": int(sd["S"])}),
        ("power_single", "drift", int(single["K"]), T_single, float(single["annualised_sr"]), n_sims,
         {"tests": ("t", "psr")}),
        ("power_multi", "drift", int(multi["K"]), T, float(multi["true_strategy_annualised_sr"]), n_sims,
         {"tests": bt, "oracle": True}),
        ("strong_drift", "drift", int(sd["N"]), int(sd["T"]), float(sd["annualised_sr"]), n_sims,
         {"tests": ("pbo", "dsr"), "S": int(sd["S"]), "oracle": True}),
        ("reported_sr1_k50", "drift", int(rep["K"]), T, float(rep["true_strategy_annualised_sr"]), n_sims,
         {"tests": bt + ("pbo",), "oracle": True, "cpcv": True, "non_binding": True, "S": int(sd["S"])}),
        ("reported_k10_sr15", "drift", 10, T, 1.5, n_sims, {"tests": bt, "oracle": True, "non_binding": True}),
        ("trend_k50", "trend", int(multi["K"]), T, float(multi["true_strategy_annualised_sr"]), n_sims,
         {"tests": bt, "oracle": True, "non_binding": True}),
        ("carry_k50", "carry", int(multi["K"]), T, float(multi["true_strategy_annualised_sr"]), n_sims,
         {"tests": bt, "oracle": True, "non_binding": True}),
        ("autocorr_null", "null", K, T, 0.0, n_sims, {"tests": ("rc", "spa", "rw"), "ar_rho": 0.3, "iid_control": True,
                                                          "non_binding": True}),
        ("stress_null", "null", K, T, 0.0, n_sims, {"tests": bt, "non_binding": True, "garch": STRESS_PARAMS}),
        ("plumbing_ledger", "null", K, T, 0.0, int(sd["min_seeds"]), {"tests": ("rc",), "ledger": True, "non_binding": True}),
    ]
    scenarios = []
    for i, (name, kind, k, t, sr, ns, extra) in enumerate(rows):
        kw = {**base, **extra}
        scenarios.append(Scenario(name=name, kind=kind, k_trials=k, n_days=t, annual_sr=sr, n_sims=ns,
                                  base_seed=int(master) + SEED_STRIDE * i, **kw))
    bars = {k: v for k, v in block.items() if k != "scenarios"}
    return {"scenarios": scenarios, "bars": bars, "scenarios_sha256": _block_hash(sc),
            "freeze_source": freeze_source, "master_seed": int(master), "cross_correlation": float(corr),
            "gates_path": str(gates_path)}


# ------------------------------------------------------------------ one simulation


def _signal_for(s: Scenario, i: int) -> dict | None:
    if s.kind == "null":
        return None
    idx = i % s.k_trials if s.randomise_signal else 0
    return {"kind": s.kind, "annual_sr": s.annual_sr, "index": idx, "horizon_days": s.horizon_days}


def _universe(s: Scenario, seed: int, i: int) -> tuple[np.ndarray, int]:
    sig = _signal_for(s, i)
    R = make_universe(s.k_trials, s.n_days, seed, s.corr, sig, s.garch)
    if s.ar_rho:
        R = lfilter([1.0], [1.0, -s.ar_rho], R, axis=0)
    return R, (sig["index"] if sig else -1)


def _ledger_roundtrip(s: Scenario, i: int, R: np.ndarray, p_mem: float, bl: float, idx: np.ndarray) -> bool:
    """Route one simulated universe through the REAL ledger (throwaway root): record_trial x K, trials(),
    gate_n, then RC on the reloaded returns; must equal the in-memory result."""
    from firm.research import ledger as L
    from firm.validation.effective_trials import gate_n

    root = os.environ.get(L.LEDGER_ROOT_ENV)
    if not root:
        raise RuntimeError(f"{L.LEDGER_ROOT_ENV} must point at a throwaway dir; refusing the host ledger")
    sub = Path(root) / f"{s.name}_{i:04d}"
    (sub / "returns").mkdir(parents=True, exist_ok=True)
    os.environ[L.LEDGER_ROOT_ENV] = str(sub)
    try:
        dates = pd.bdate_range("2000-01-03", periods=R.shape[0])
        ids = []
        for k in range(R.shape[1]):
            cfg = {"synthetic": s.name, "sim": i, "k": k}
            tid = f"syn{i:04d}_{k:03d}"
            sr, sk, ku, n = moments(R[:, k])
            rec = L.TrialRecord(
                trial_id=tid, family=f"synthetic_{s.name}", mode="exploratory", config=cfg,
                config_hash=L.config_hash(cfg), code_commit="", data_snapshot_id=None, seed=s.base_seed + i,
                start=str(dates[0].date()), end=str(dates[-1].date()), returns_path=None, gross_sharpe=sr,
                net_sharpe=sr, periods_per_year=PERIODS_PER_YEAR, sharpe_conversion="per_period", n_obs=n,
                skew=sk, kurt=ku, preregistration_id=None, touched_holdout=False, status="completed", error=None,
            )
            L.record_trial(rec, returns=pd.Series(R[:, k], index=dates))
            ids.append(tid)
        tr = L.trials()
        back = pd.DataFrame({r["trial_id"]: pd.read_parquet(sub / r["returns_path"])["returns"]
                             for _, r in tr.iterrows()})
        tc = gate_n(tr, back)
        same = reality_check(back[ids].to_numpy(), B=idx.shape[0], block_len=bl, seed=0, _indices=idx).p_value == p_mem
        return bool(len(tr) == R.shape[1] and tc.gate_n == R.shape[1] and same and L.verify_chain().ok)
    finally:
        os.environ[L.LEDGER_ROOT_ENV] = root


def _cpcv_positive_paths(R: np.ndarray) -> int:
    T = R.shape[0]
    dates = pd.bdate_range("2000-01-03", periods=T)
    cv = CombinatorialPurgedCV(10, 2, label_end=pd.Series(dates, index=dates), embargo_pct=0.01)
    preds = {}
    for sid, (train, test) in enumerate(cv.split()):
        star = int(np.argmax(sharpe_cols(R[train])))  # selection rule applied inside each split
        preds[sid] = R[test, star]
    paths = cv.backtest_paths(preds)
    return int(sum(1 for p in paths if p.mean() > 0.0))


def _sim(args: tuple[Scenario, int]) -> dict:
    s, i = args
    for name in ("firm.validation.multiple_testing", "firm.validation.sharpe_stats", "firm.validation.effective_trials"):
        logging.getLogger(name).setLevel(logging.ERROR)
    seed = s.base_seed + i
    R, sig = _universe(s, seed, i)
    T, K = R.shape
    out: dict = {"sig": sig}
    srs = sharpe_cols(R)
    best = int(np.argmax(srs))
    out["best_is_true"] = bool(best == sig) if sig >= 0 else None
    out["max_t"] = float(np.max(srs) * math.sqrt(T))
    out["max_sr"] = float(np.max(srs))
    if "t" in s.tests or "psr" in s.tests:
        sr, sk, ku, n = moments(R[:, 0])
        tcrit = float(stats.t.ppf(1.0 - s.alpha, T - 1))
        out["t"] = bool(sr * math.sqrt(T) > tcrit)
        out["psr"] = bool(psr(sr, 0.0, n, sk, ku) >= 1.0 - s.alpha)
    if "psr0" in s.tests:
        sr, sk, ku, n = moments(R[:, 0])
        out["psr0"] = float(psr(sr, 0.0, n, sk, ku))
    if "dsr" in s.tests:
        sr, sk, ku, n = moments(R[:, best])
        var_sr = length_adjusted_var_sr(srs, T, T)
        out["dsr"] = float(dsr_fn(sr, n, sk, ku, K, var_sr))
    bl = idx = None
    if {"rc", "spa", "rw"} & set(s.tests):
        bl = float(politis_white_block_length(R))
        idx = stationary_bootstrap_indices(T, s.B, bl, seed)
        rc = reality_check(R, B=s.B, block_len=bl, seed=seed, _indices=idx)
        out["rc_p"] = rc.p_value
        if "spa" in s.tests:
            out["spa_p"] = spa_test(R, B=s.B, block_len=bl, seed=seed, _indices=idx).p_consistent
        if "rw" in s.tests:
            rw = romano_wolf(R, alpha=s.alpha, B=s.B, block_len=bl, seed=seed, _indices=idx)
            rej = rw["reject"].to_numpy()
            out["rw_any"] = bool(rej.any())
            out["rw_true"] = bool(rej[sig]) if sig >= 0 else None
        if s.iid_control:
            idx1 = stationary_bootstrap_indices(T, s.B, 1.0, seed)
            out["rc_p_iid"] = reality_check(R, B=s.B, block_len=1.0, seed=seed, _indices=idx1).p_value
        if s.ledger:
            out["plumbing_ok"] = _ledger_roundtrip(s, i, R, rc.p_value, bl, idx)
    if "pbo" in s.tests:
        out["pbo"] = float(cscv_pbo(R, S=s.S).pbo)
    if s.cpcv:
        out["cpcv_pos_paths"] = _cpcv_positive_paths(R)
    return out


def _oracle_sim(args: tuple[Scenario, int]) -> tuple[float, float]:
    s, i = args
    R = make_universe(s.k_trials, s.n_days, s.base_seed + ORACLE_SEED_OFFSET + i, s.corr, None, s.garch)
    if s.ar_rho:
        R = lfilter([1.0], [1.0, -s.ar_rho], R, axis=0)
    srs = sharpe_cols(R)
    return float(srs.max() * math.sqrt(R.shape[0])), float(srs.max())


def _map(fn, items: list, workers: int) -> list:
    if workers <= 1:
        return [fn(x) for x in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(fn, items, chunksize=1))


# ------------------------------------------------------------------ aggregation


def _rate(vals: list, alpha: float) -> dict:
    v = [x for x in vals if x is not None]
    return wilson(int(sum(bool(x) for x in v)), len(v), alpha)


def _aggregate(s: Scenario, raw: list[dict], null_oracle: list[tuple[float, float]] | None) -> dict:
    a = s.alpha
    col = lambda k: [r[k] for r in raw if k in r]
    rates: dict = {}
    if "rc" in s.tests:
        null = s.kind == "null"
        rates["reality_check"] = _rate([p < a for p in col("rc_p")], a)
        if "spa" in s.tests:
            rates["spa"] = _rate([p < a for p in col("spa_p")], a)
        if "rw" in s.tests:
            rates["romano_wolf_fwer" if null else "romano_wolf"] = _rate(col("rw_any" if null else "rw_true"), a)
            if not null:
                rates["romano_wolf_any"] = _rate(col("rw_any"), a)
        if s.iid_control:
            rates["reality_check_iid_control"] = _rate([p < a for p in col("rc_p_iid")], a)
    if "dsr" in s.tests:
        rates["dsr_false_pass" if s.kind == "null" else "dsr_pass"] = _rate([d >= 1.0 - a for d in col("dsr")], a)
    if "t" in s.tests:
        rates["t_test"] = _rate(col("t"), a)
        rates["psr"] = _rate(col("psr"), a)
    if s.ledger:
        rates["plumbing_ok"] = _rate(col("plumbing_ok"), a)
    out: dict = {"spec": asdict(s), "n_sims": len(raw), "rates": rates, "raw": raw}
    if "psr0" in s.tests:
        out["psr0_ks_p"] = float(stats.kstest(col("psr0"), "uniform").pvalue)
    if "pbo" in s.tests:
        v = np.array(col("pbo"))
        z = float(stats.norm.ppf(1.0 - a / 2.0))
        se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float("nan")
        out.update(pbo_mean=float(v.mean()), pbo_ci=[float(v.mean() - z * se), float(v.mean() + z * se)], n=len(v))
    if s.cpcv:
        c = col("cpcv_pos_paths")
        out["cpcv_pos_paths_mean"] = float(np.mean(c))
        out["cpcv_pos_paths_ge_7_of_9"] = _rate([x >= 7 for x in c], a)
    if null_oracle is not None:
        mt = np.array([x[0] for x in null_oracle])
        ms = np.array([x[1] for x in null_oracle])
        crit_t, crit_s = float(np.quantile(mt, 1.0 - a, method="higher")), float(np.quantile(ms, 1.0 - a, method="higher"))
        out["oracle"] = {
            "maxt_crit": crit_t, "maxsr_crit": crit_s, "n_null": len(null_oracle),
            "maxt_power": _rate([x >= crit_t for x in col("max_t")], a),
            "maxsr_pass": _rate([x >= crit_s for x in col("max_sr")], a),
        }
        if s.k_trials > 0 and s.kind != "null":
            out["analytic_bonferroni_power"] = _bonferroni_power(s)
    if s.k_trials > 1 and s.kind in ("drift", "trend", "carry"):
        bt = _rate(col("best_is_true"), a)
        out["best_is_true"] = bt
    return out


def _bonferroni_power(s: Scenario) -> float:
    """Normal-approx power of a one-sided Bonferroni max-t test: ``1 - Phi(z_{1-a/K} - SR*sqrt(T/ppy))``."""
    t_mean = s.annual_sr * math.sqrt(s.n_days / PERIODS_PER_YEAR)
    return float(1.0 - stats.norm.cdf(stats.norm.ppf(1.0 - s.alpha / s.k_trials) - t_mean))


def run_scenario(s: Scenario, workers: int = 2) -> dict:
    """Run one scenario purely from its spec; returns rates (Wilson intervals) per statistic."""
    s.garch.validate()
    raw = _map(_sim, [(s, i) for i in range(s.n_sims)], workers)
    null_oracle = None
    if s.oracle:
        null_oracle = _map(_oracle_sim, [(s, i) for i in range(s.n_sims)], workers)
    return _aggregate(s, raw, null_oracle)


# ------------------------------------------------------------------ verdict


def _fmt(d: dict) -> str:
    return f"{d['rate']:.3f} (95% CI {d['lo']:.3f}-{d['hi']:.3f}, n={d['n']})"


def _bar_checks(results: dict, acc: dict) -> list[tuple[str, bool, bool, str]]:
    """(label, point_pass, ci_pass, detail) for every frozen bar; ci_pass False with point_pass True = marginal."""
    bars = acc["bars"]
    chk: list[tuple[str, bool, bool, str]] = []
    size = bars["size"]
    mx = size["max_rejection_rate"]
    ns = results["null_size"]
    for stat in size["applies_separately_to"]:
        d = ns["rates"][stat]
        chk.append((f"size {stat}", d["rate"] <= mx, d["hi"] <= mx, f"{_fmt(d)} vs max {mx}"))
    d = ns["rates"]["dsr_false_pass"]
    mxd = bars["dsr"]["null_false_pass_max"]
    chk.append(("dsr null false-pass", d["rate"] <= mxd, d["hi"] <= mxd, f"{_fmt(d)} vs max {mxd}"))
    kp = bars["dsr"]["two_sided_calibration"]["psr0_single_null_uniform_ks_p_min"]
    chk.append(("psr0 uniform (KS)", ns["psr0_ks_p"] >= kp, ns["psr0_ks_p"] >= kp, f"KS p={ns['psr0_ks_p']:.4f} vs min {kp}"))
    lo_r, hi_r = bars["pbo"]["noise_mean_pbo_range"]
    npb = results["null_pbo"]
    m, ci = npb["pbo_mean"], npb["pbo_ci"]
    chk.append(("pbo noise mean", lo_r <= m <= hi_r, lo_r <= ci[0] and ci[1] <= hi_r,
                f"mean {m:.3f} CI {ci[0]:.3f}-{ci[1]:.3f} vs [{lo_r}, {hi_r}]"))
    sdb = bars["pbo"]["strong_drift"]
    sd = results["strong_drift"]
    m, ci = sd["pbo_mean"], sd["pbo_ci"]
    chk.append(("pbo strong-drift mean", m < sdb["mean_pbo_max_exclusive"] and sd["n"] >= sdb["min_seeds"],
                ci[1] < sdb["mean_pbo_max_exclusive"], (f"mean {m:.4f} CI {ci[0]:.4f}-{ci[1]:.4f}, n={sd['n']} vs < "
                f"{sdb['mean_pbo_max_exclusive']}")))
    if bars["dsr"]["two_sided_calibration"]["strong_drift_pass_rate_within_mc_error_of_oracle"]:
        a, o = sd["rates"]["dsr_pass"], sd["oracle"]["maxsr_pass"]
        half = math.hypot(a["hi"] - a["lo"], o["hi"] - o["lo"]) / 2.0  # combined Wilson half-widths
        diff = abs(a["rate"] - o["rate"])
        chk.append(("dsr strong-drift vs oracle", diff <= half, diff <= half,
                    f"DSR {_fmt(a)} vs oracle {_fmt(o)}; |diff|={diff:.3f} vs MC half-width {half:.3f}"))
    ps = bars["power"]["single_test"]["min_power"]
    for stat in ("t_test", "psr"):
        d = results["power_single"]["rates"][stat]
        chk.append((f"power single {stat}", d["rate"] >= ps, d["lo"] >= ps, f"{_fmt(d)} vs min {ps}"))
    mt = bars["power"]["multiple_testing"]
    pm = results["power_multi"]
    orc = pm["oracle"]["maxt_power"]
    for stat in ("reality_check", "spa", "romano_wolf"):
        d = pm["rates"][stat]
        ok = d["rate"] >= mt["min_power"] or orc["rate"] - d["rate"] <= mt["or_within_of_oracle_max_t"]
        ci_ok = d["lo"] >= mt["min_power"] or orc["rate"] - d["rate"] <= mt["or_within_of_oracle_max_t"]
        chk.append((f"power multiple {stat}", ok, ci_ok,
                    f"{_fmt(d)} vs min {mt['min_power']} or oracle {_fmt(orc)} - {mt['or_within_of_oracle_max_t']}"))
    return chk


def evaluate(results: dict, acceptance: dict) -> list[str]:
    """Failures (empty = pass): every measured rate against the bar READ from gates.yaml (point estimates)."""
    fails = [f"{lab}: {det}" for lab, ok, _, det in _bar_checks(results, acceptance) if not ok]
    plumb = results.get("plumbing_ledger", {}).get("rates", {}).get("plumbing_ok")
    if plumb is not None and plumb["k"] != plumb["n"]:
        fails.append(f"ledger plumbing: {plumb['n'] - plumb['k']} of {plumb['n']} round trips disagreed")
    return fails


def find_marginal(results: dict, acceptance: dict) -> list[str]:
    """Bars met on the point estimate whose Monte-Carlo interval is not: listed for the owner, not silently passed."""
    return [f"{lab}: {det}" for lab, ok, ci_ok, det in _bar_checks(results, acceptance) if ok and not ci_ok]


# ------------------------------------------------------------------ report


def write_report(results: dict, failures: list[str], out_dir: Path, *, meta: dict | None = None,
                 marginal: list[str] | None = None, acceptance: dict | None = None) -> Path:
    """Dated markdown (+ machine-readable JSON) report; returns the markdown path."""
    meta = meta or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = meta.get("date_utc", "undated")
    quick = bool(meta.get("quick"))
    suffix = "_quick" if quick else ""
    md_path = out_dir / f"synthetic_{stamp}{suffix}.md"
    js_path = out_dir / f"synthetic_{stamp}{suffix}.json"
    verdict = "PASS" if not failures else "FAIL"
    L = [f"# Synthetic GARCH-t validation of the statistics pipeline (P1-08), {stamp}", ""]
    if quick:
        L += ["**NON-EVIDENCE: `--quick` smoke run (reduced sims/B/K/T); never an H1 result.**", ""]
    L += [f"**Verdict: {verdict}**" + (" (marginal items listed below)" if marginal else ""), ""]
    L += ["## Provenance", ""] + [f"- {k}: {v}" for k, v in meta.items()] + [""]
    if acceptance:
        L += [f"- scenarios block sha256: `{acceptance['scenarios_sha256']}`",
              f"- cross_correlation / master_seed source: {acceptance['freeze_source']}", ""]
    L += ["## Failures", ""] + ([f"- {f}" for f in failures] or ["- none"]) + [""]
    L += ["## Marginal (point estimate meets the bar, Monte-Carlo interval does not)", ""]
    L += [f"- {m}" for m in (marginal or [])] or ["- none"]
    L += ["", "## Scenario results", ""]
    for name, r in results.items():
        sp = r["spec"]
        L += [f"### {name}{' (reported only, not a pass bar)' if sp.get('non_binding') else ''}", "",
              (f"kind={sp['kind']} K={sp['k_trials']} T={sp['n_days']} SR={sp['annual_sr']} B={sp['B']} sims={sp['n_sims']} "
              f"base_seed={sp['base_seed']} corr={sp['corr']} ar_rho={sp['ar_rho']} garch={sp['garch']}"), ""]
        for k, d in r["rates"].items():
            L.append(f"- {k}: {_fmt(d)}")
        for k in ("psr0_ks_p", "pbo_mean", "pbo_ci", "cpcv_pos_paths_mean", "analytic_bonferroni_power"):
            if k in r:
                L.append(f"- {k}: {r[k]}")
        if "oracle" in r:
            o = r["oracle"]
            L.append(f"- oracle max-t power: {_fmt(o['maxt_power'])} (crit {o['maxt_crit']:.3f}); "
                     f"oracle max-SR pass: {_fmt(o['maxsr_pass'])}")
        if "cpcv_pos_paths_ge_7_of_9" in r:
            L.append(f"- CPCV >= 7 of 9 paths positive: {_fmt(r['cpcv_pos_paths_ge_7_of_9'])}")
        L.append("")
    sha = (acceptance or {}).get("gates_sha256") or meta.get("gates_yaml_sha256", "unknown")
    L += ["---", f"Harness thresholds were read from gates.yaml frozen at {sha}; none were changed after viewing results.", ""]
    md_path.write_text("\n".join(L))
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "raw"} for k, v in results.items()}
    js_path.write_text(json.dumps({"meta": meta, "verdict": verdict, "failures": failures, "marginal": marginal or [],
                                   "results": slim}, indent=1, default=str, sort_keys=True))
    (out_dir / f"synthetic_{stamp}{suffix}.raw.json").write_text(
        json.dumps({k: v["raw"] for k, v in results.items()}, default=str, sort_keys=True))
    return md_path


__all__ = [
    "STRESS_PARAMS",
    "GarchTParams",
    "Scenario",
    "evaluate",
    "find_marginal",
    "inject_carry",
    "inject_drift",
    "inject_trend",
    "load_acceptance",
    "make_universe",
    "replace",
    "run_scenario",
    "simulate_garch_t",
    "standardised_t",
    "wilson",
    "write_report",
]
