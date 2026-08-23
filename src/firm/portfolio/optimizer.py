"""Joint constrained portfolio optimizer.

Replaces the historical ``TraderAgent`` L1-normalize-to-full-investment
sizing + ``RiskAgent``'s 9 independent sequential clip/scale passes with a
single mean-variance-with-transaction-costs QP, solved jointly. See PART 2 of
``docs/remediation_progress.md`` / the session plan for the full diagnosis
and design rationale; summary of what this fixes:

1. Gross exposure/book size now scales with aggregate conviction strength
   (via the risk-aversion term) instead of always being forced to
   ``sum(|w|) == 1``.
2. All structural constraints (position/gross/net/liquidity caps, sector,
   vol) are enforced *jointly* in one solve instead of via sequential passes
   that can undo each other.
3. Transaction costs (the same commission/slippage/spread/sqrt-market-impact
   model ``ExecutionAgent``/``_liquidity.py`` already use) are native to the
   objective, producing an endogenous no-trade region and partial
   rebalancing — instead of being bolted on downstream in ``ExecutionAgent``
   (``rebalance_band_pct``/``rebalance_fraction``).

This module is deliberately pure and side-effect-free: no ``Agent``
subclass, no I/O, no LLM calls. ``TraderAgent`` (``src/firm/agents/
trader.py``, ``allocation_method == "joint_optimizer"``) is the only caller.

Determinism: every solve here must be run single-threaded and on inputs
built from a *sorted* symbol list, or the same inputs can produce
last-bit-different floats across runs (multi-threaded BLAS/solver reduction
order is not associative) -- this pipeline's ``test_reproducibility`` test
compares two seeded runs' target weights, so nondeterminism here is a real
regression, not a cosmetic concern.

Never raises and never hangs: this runs inside a live trading cycle. Every
public function degrades gracefully (documented per-function) and the
top-level :func:`solve_portfolio` enforces its own wall-clock budget
(``_SOLVE_TIME_BUDGET_SECONDS``, far under the orchestrator's per-stage
timeout) since a hung C-extension solve does not respond to the
orchestrator's thread-pool cancellation.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Self-imposed wall-clock budget for one solve attempt. Far below
# orchestrator_stage_timeout_seconds (default 300s) -- that timeout is a
# last-resort net that must never actually fire; this budget is what's
# supposed to trip first, every time.
_SOLVE_TIME_BUDGET_SECONDS = 5.0

# Covariance-usability thresholds (see estimate_covariance's docstring).
_MIN_COV_OBS = 60
_MAX_COND_NUMBER = 1e10

# IC (information coefficient) calibration defaults -- see estimate_ic.
_IC_PRIOR_DEFAULT = 0.03
_IC_CAP_DEFAULT = 0.15
_IC_SHRINKAGE_N0_DEFAULT = 90
_IC_IR_REF_DEFAULT = 1.0
_IC_IR_CAP_DEFAULT = 2.0

_TRADING_DAYS_PER_YEAR = 252


@dataclass
class OptimizerConstraints:
    """Structural constraints fed to the QP as hard/soft bounds.

    Field names/defaults deliberately mirror ``RiskAgent``'s existing
    ``RiskConfig`` fields (``max_position_pct``, ``max_gross_exposure``,
    ``max_net_exposure``, ``max_sector_pct``, ``vol_target``,
    ``max_participation_pct``) so a caller can build one of these directly
    from the same config dict.
    """

    max_position_pct: float = 0.05
    max_gross_exposure: float = 1.5
    max_net_exposure: float = 0.50
    max_sector_pct: float = 0.25
    vol_target: float = 0.12  # annualized; the soft ceiling (sigma_max)
    max_participation_pct: float = 0.10
    sector_map: dict[str, str] = field(default_factory=dict)
    # symbol -> trailing average daily dollar volume; used for the (hard)
    # liquidity/trade-size bound |w_i - w0_i| <= max_participation_pct *
    # adv_dollars_i / nav. A symbol missing from this dict gets no liquidity
    # bound (fails open, matching RiskAgent._cap_liquidity's existing
    # posture: a missing/thin data provider must degrade the cap, not block
    # a cycle).
    adv_dollars: dict[str, float] = field(default_factory=dict)
    nav: float = 1.0
    # Pre-solve risk-budget scalars (see RiskAgent's drawdown-breaker/
    # regime-overlay, which now scale the *budget* before the solve instead
    # of the *solved book* after it). 1.0 = no scaling, unchanged.
    gross_budget_scale: float = 1.0
    vol_budget_scale: float = 1.0


@dataclass
class CostParams:
    """Transaction-cost model -- must match ``_liquidity.py``/
    ``ExecutionAgent``'s cost model so the risk cap, the optimizer, and the
    execution cost estimate all agree on one cost/ADV definition."""

    commission_pct: float = 0.0005
    slippage_pct: float = 0.0005
    spread_pct: float = 0.0002
    market_impact_coefficient: float = 0.005
    # kappa: cost-aversion multiplier. 1.0 = costs enter the objective in
    # true return units (the "honest" trade-off); raise for extra turnover
    # damping.
    cost_aversion: float = 1.0


@dataclass
class RiskAversionParams:
    """Controls how much the book "breathes" with aggregate conviction --
    see module docstring point 1. Do not hand-pick a bare risk-aversion
    number (its natural units are unintuitive); calibrate it from a target
    average book vol instead."""

    # Below vol_target (the hard-ish ceiling in OptimizerConstraints) so
    # there's headroom for the book to run hotter on genuinely strong days.
    target_avg_vol: float = 0.085  # annualized
    # If set, used directly (mainly for tests). If None, solve_portfolio
    # derives it from target_avg_vol + the current alpha/covariance so the
    # book's ex-ante vol matches target_avg_vol on average. Recalibrate this
    # slowly (e.g. monthly) in a live deployment -- refitting every bar
    # defeats the "breathing" property.
    risk_aversion: float | None = None
    # Ridge/robustness term size, as a fraction of risk_aversion -- standard
    # mean-variance error-maximization defense (Michaud). Not decoration;
    # do not drop this "to simplify" the objective.
    ridge_frac: float = 0.075
    # Expected number of days a new position will be held before the signal
    # that motivated it decays/gets re-evaluated. `alpha` (from
    # compute_alpha) is a *single-day* expected return, but transaction
    # costs are a *one-time* cost of establishing the trade -- comparing
    # them directly (holding_horizon_days=1) makes almost every trade look
    # unprofitable even with real signal, because a one-day alpha can't
    # amortize a cost paid once (confirmed empirically: with
    # holding_horizon_days=1 the optimizer holds current weights almost
    # unconditionally). The objective scales alpha's contribution by this
    # horizon (cost stays as the true one-time cost) so the trade-off
    # reflects "is this signal worth trading on given how long we expect to
    # benefit from it," not "does one day's alpha alone repay the cost."
    # Default of 5 is a conservative starting assumption (roughly matches
    # TraderAgent's existing conviction-EMA halflife of ~3 days) --
    # calibrate from realized signal persistence when data allows.
    holding_horizon_days: float = 5.0


@dataclass
class SolveResult:
    """Everything :func:`solve_portfolio` returns, beyond the bare weights."""

    targets: dict[str, float]
    status: str  # "optimal", "diagonal_cov_fallback", "closed_form_fallback", "hold_fallback"
    solve_seconds: float
    risk_aversion_used: float | None = None
    notes: str = ""


def estimate_covariance(
    returns_by_symbol: dict[str, "Any"],
    symbols: list[str],
    lookback_days: int = _TRADING_DAYS_PER_YEAR,
) -> np.ndarray | None:
    """Ledoit-Wolf-shrunk daily covariance matrix over *symbols* (sorted
    order), or ``None`` if unusable.

    ``returns_by_symbol`` maps symbol -> a pandas Series (or anything
    ``pd.DataFrame`` accepts as a column) of daily returns, already aligned
    to a common date index by the caller (e.g. via ``pit_view.prices``).
    Ledoit-Wolf's shrinkage intensity is chosen analytically -- no manual
    tuning, and the result is always PSD, which is what makes this safe to
    run unattended (unlike raw sample covariance on ~25 names with a few
    hundred observations, which is the classic error-maximization trap).

    "Unusable" (returns ``None``, caller should degrade to a diagonal
    covariance -- see :func:`diagonal_covariance`) is precisely:
    fewer than ``_MIN_COV_OBS`` valid overlapping observations, any NaN/inf
    surviving the shrinkage, or a condition number above ``_MAX_COND_NUMBER``.
    """
    import pandas as pd

    cols = {s: returns_by_symbol[s] for s in symbols if s in returns_by_symbol}
    if len(cols) < 2:
        return None
    frame = pd.DataFrame(cols).tail(lookback_days).dropna(how="any")
    if len(frame) < _MIN_COV_OBS:
        log.debug(
            "estimate_covariance: only %d overlapping obs (<%d) -- unusable",
            len(frame), _MIN_COV_OBS,
        )
        return None

    try:
        from sklearn.covariance import LedoitWolf

        lw = LedoitWolf().fit(frame.to_numpy())
        cov = lw.covariance_
    except Exception:
        log.warning("estimate_covariance: LedoitWolf fit failed", exc_info=True)
        return None

    # Re-expand to the full, sorted `symbols` order (symbols missing history
    # get a zero row/col here; the caller's degradation cascade is expected
    # to catch the resulting singularity via the condition-number check
    # below, or the caller should have filtered them out already).
    n = len(symbols)
    full = np.zeros((n, n))
    idx = {s: i for i, s in enumerate(symbols)}
    present = list(frame.columns)
    sub_idx = [idx[s] for s in present]
    for i, si in enumerate(sub_idx):
        for j, sj in enumerate(sub_idx):
            full[si, sj] = cov[i, j]

    if not np.all(np.isfinite(full)):
        log.warning("estimate_covariance: non-finite values after shrinkage -- unusable")
        return None
    # Diagonal-only rows (missing history) make the matrix singular by
    # construction when compared against the rest -- give them a small
    # positive variance so cond() is well-defined rather than exploding.
    missing = [s for s in symbols if s not in present]
    if missing:
        fallback_var = float(np.mean(np.diag(cov))) if len(cov) else 1e-4
        for s in missing:
            full[idx[s], idx[s]] = fallback_var

    try:
        cond = np.linalg.cond(full)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(cond) or cond > _MAX_COND_NUMBER:
        log.debug("estimate_covariance: cond=%s exceeds %s -- unusable", cond, _MAX_COND_NUMBER)
        return None
    return full


def diagonal_covariance(vols: dict[str, float], symbols: list[str]) -> np.ndarray:
    """Diagonal fallback covariance from per-symbol annualized vol estimates.

    Always well-conditioned by construction -- the degradation target when
    :func:`estimate_covariance` reports the shrunk sample covariance is
    unusable. Missing symbols get a conservative default daily variance.
    """
    n = len(symbols)
    default_daily_var = (0.30 ** 2) / _TRADING_DAYS_PER_YEAR  # ~30% ann. vol default
    diag = np.full(n, default_daily_var)
    for i, s in enumerate(symbols):
        v = vols.get(s)
        if v and v > 0:
            diag[i] = (v ** 2) / _TRADING_DAYS_PER_YEAR
    return np.diag(diag)


def estimate_ic(
    trailing_returns: "Any | None",
    ic_prior: float = _IC_PRIOR_DEFAULT,
    ic_cap: float = _IC_CAP_DEFAULT,
    ir_ref: float = _IC_IR_REF_DEFAULT,
    ir_cap: float = _IC_IR_CAP_DEFAULT,
    n0: int = _IC_SHRINKAGE_N0_DEFAULT,
) -> float:
    """Shrunk "effective IC" (information coefficient) for the combined
    signal, proxied from its own trailing realized return series (Path B --
    see the module/plan docstring: no signal+forward-return history store
    exists yet for a genuine cross-sectional rank-IC, so this proxies via
    the Fundamental-Law-of-Active-Management relationship IR = IC*sqrt(BR)
    instead, treating the mapping as a bounded *trust multiplier* rather
    than a literal IC to avoid overstating effective breadth on a small,
    correlated mega-cap universe).

    ``trailing_returns`` is a realized daily return series for the combined
    book (or ``None``/empty -- cold start). A persistently negative realized
    IR shrinks the result toward (never below) 0 -- it must never flip the
    sign of a downstream alpha estimate, only its magnitude. ``ir_ref``/
    ``ir_cap`` are on an **annualized**-IR scale (e.g. ``ir_ref=1.0`` means
    "full-strength trust at an annualized IR of 1.0"); the realized daily
    mean/std ratio is annualized (``* sqrt(252)``) internally before being
    compared against them.

    Always returns a finite value in ``[0, ic_cap]``. Cold start (fewer than
    ``n0`` observations, or no data at all) returns exactly ``ic_prior``.
    """
    if trailing_returns is None:
        return ic_prior
    try:
        arr = np.asarray(trailing_returns, dtype=float)
        arr = arr[np.isfinite(arr)]
    except Exception:
        return ic_prior
    n = len(arr)
    if n < 2:
        return ic_prior

    std = float(arr.std(ddof=1))
    if std <= 0 or not np.isfinite(std):
        return ic_prior
    # mean/std of *daily* returns is on a tiny scale (a genuinely excellent
    # annualized IR of 2.0 is only ~0.126 here, since dividing by sqrt(252)
    # is exactly what annualizing an IR does) -- ir_ref/ir_cap are meant as
    # annualized-IR-scale thresholds (a "full-strength trust at IR==1.0,
    # capped at IR==2.0" story), so the daily ratio must be annualized
    # before comparing against them. Confirmed via the first real walk-
    # forward+PBO run of this module: without this, trust was chronically
    # ~0.05-0.15 even for a genuinely decent book, starving alpha of
    # magnitude and producing a near-flat, under-traded portfolio that
    # looked "safe" (low vol, low turnover) but materially underperformed
    # the benchmark -- not a real no-edge finding, a units bug.
    ir_annualized = (float(arr.mean()) / std) * (_TRADING_DAYS_PER_YEAR ** 0.5)
    trust = np.clip(ir_annualized / ir_ref, 0.0, ir_cap) if ir_ref > 0 else 0.0
    sample_ic = ic_prior * trust

    # Shrink toward the prior early in a run (n0-style shrinkage, same
    # spirit as conviction-EMA/IC calibration elsewhere in this codebase).
    phi = n / (n + n0) if n0 > 0 else 1.0
    ic_eff = phi * sample_ic + (1.0 - phi) * ic_prior
    return float(np.clip(ic_eff, 0.0, ic_cap))


def compute_alpha(
    net_convictions: dict[str, float],
    vols: dict[str, float],
    ic_eff: float,
    conviction_scale: float = 3.0,
) -> dict[str, float]:
    """Expected (daily) return per symbol: ``alpha_i = ic_eff * vol_i * s_i``.

    ``s_i = conviction_scale * net_convictions[i]`` inverts the ``/3.0`` cap
    the debate stage applies (see ``research/bull.py``/``bear.py``) so the
    combined standardized score is recovered for ``|s_i| < conviction_scale``
    (mildly conservative in the saturated tails -- see the module docstring
    for why surfacing the pre-cap z-score is a later-increment refinement,
    not required for v1). ``vols`` should be the same per-symbol vol used to
    build the risk-term covariance (``sqrt(diag(Sigma))``) -- using a
    different source here would make the sizing internally inconsistent.
    Symbols missing a vol estimate get ``alpha=0`` (no sizing basis) rather
    than a guessed value.
    """
    alpha: dict[str, float] = {}
    for sym, conv in net_convictions.items():
        vol = vols.get(sym)
        if vol is None or vol <= 0 or not np.isfinite(vol):
            alpha[sym] = 0.0
            continue
        # vols are annualized; alpha is a *daily* expected return, matching
        # the daily covariance -- convert here so callers always pass
        # annualized vol (the convention used everywhere else in this repo,
        # e.g. RiskAgent.vol_target, TraderAgent._estimate_vol).
        daily_vol = vol / (_TRADING_DAYS_PER_YEAR ** 0.5)
        alpha[sym] = ic_eff * daily_vol * (conviction_scale * conv)
    return alpha


def _transaction_cost_terms(
    symbols: list[str],
    cost: CostParams,
    constraints: OptimizerConstraints,
) -> tuple[float, np.ndarray]:
    """Returns ``(c_lin, c_imp)`` -- the flat linear cost rate (same for
    every symbol) and a per-symbol sqrt-impact coefficient array, in the
    exact units ``|Delta w_i|`` and ``|Delta w_i|^1.5`` need to be multiplied
    by to land in *return* units (fraction of NAV), matching
    ``_liquidity.py``'s ``market_impact_pct``:
    impact_cost_as_nav_frac_i = coefficient * sqrt(NAV/ADV_i) * |Delta w_i|^1.5.
    """
    c_lin = cost.commission_pct + cost.slippage_pct + cost.spread_pct
    n = len(symbols)
    c_imp = np.zeros(n)
    nav = constraints.nav if constraints.nav > 0 else 1.0
    for i, s in enumerate(symbols):
        adv = constraints.adv_dollars.get(s)
        if adv and adv > 0 and cost.market_impact_coefficient > 0:
            c_imp[i] = cost.market_impact_coefficient * (nav / adv) ** 0.5
    return c_lin, c_imp


def _closed_form_fallback(
    alpha: np.ndarray,
    cov: np.ndarray,
    risk_aversion: float,
    constraints: OptimizerConstraints,
) -> np.ndarray:
    """Unconstrained mean-variance solution ``w = (1/lambda) * Sigma^-1 *
    alpha``, then sequentially clipped to box -> gross -> net. This is
    deliberately the *old* sequential-clip logic, kept only as the
    last-resort rung before holding current weights -- see
    :func:`solve_portfolio`'s degradation cascade."""
    try:
        w = np.linalg.solve(cov, alpha) / max(risk_aversion, 1e-12)
    except np.linalg.LinAlgError:
        w = np.zeros_like(alpha)

    c = constraints.max_position_pct
    w = np.clip(w, -c, c)
    gross = np.abs(w).sum()
    cap_g = constraints.max_gross_exposure * constraints.gross_budget_scale
    if gross > cap_g and gross > 0:
        w = w * (cap_g / gross)
    net = w.sum()
    if abs(net) > constraints.max_net_exposure and net != 0:
        excess = abs(net) - constraints.max_net_exposure
        sign = 1.0 if net > 0 else -1.0
        same_side = w * sign > 0
        total_same = np.abs(w[same_side]).sum()
        if total_same > 0:
            reduction = excess * (np.abs(w[same_side]) / total_same) * sign
            w[same_side] -= reduction
    return w


def solve_portfolio(
    alpha: dict[str, float],
    current_weights: dict[str, float],
    symbols: list[str],
    cov: np.ndarray | None,
    constraints: OptimizerConstraints,
    cost: CostParams,
    risk: RiskAversionParams,
) -> SolveResult:
    """Solve ``max_w  alpha.w - (lambda/2) w'Sigma w - kappa*TC(w-w0)``
    subject to the constraints in *constraints*, and return target weights.

    *symbols* must be a **sorted, deterministic** list -- see the module
    docstring on determinism. *cov* must already be a valid (PSD, finite,
    well-conditioned) covariance in *symbols* order, e.g. from
    :func:`estimate_covariance` with a :func:`diagonal_covariance` fallback
    already applied by the caller if the former returned ``None`` -- if
    *cov* is ``None`` here, this function builds its own diagonal fallback
    from unit variance so it still never raises.

    Never raises. Always returns a finite, hard-cap-compliant
    ``SolveResult`` -- see the module docstring's degradation cascade:
    primary cvxpy solve -> closed-form fallback -> hold current weights.
    Self-bounds its own wall-clock budget (``_SOLVE_TIME_BUDGET_SECONDS``)
    independently of the caller's timeout, since a hung C-extension solve
    would not respond to external cancellation.
    """
    t0 = time.monotonic()
    n = len(symbols)
    if n == 0:
        return SolveResult(targets={}, status="hold_fallback", solve_seconds=0.0, notes="empty universe")

    a = np.array([alpha.get(s, 0.0) for s in symbols], dtype=float)
    w0 = np.array([current_weights.get(s, 0.0) for s in symbols], dtype=float)
    if not np.all(np.isfinite(a)):
        log.warning("solve_portfolio: non-finite alpha values -- holding current weights")
        return SolveResult(
            targets=dict(zip(symbols, w0)), status="hold_fallback",
            solve_seconds=time.monotonic() - t0, notes="non-finite alpha",
        )
    if np.allclose(a, 0.0):
        # No signal at all -- nothing to size against; hold rather than
        # invent a book from pure risk-minimization noise.
        return SolveResult(
            targets=dict(zip(symbols, w0)), status="hold_fallback",
            solve_seconds=time.monotonic() - t0, notes="all-zero alpha (no signal)",
        )
    # Scale by the assumed holding horizon (see RiskAversionParams.
    # holding_horizon_days' docstring) -- lambda is calibrated FROM this
    # scaled alpha to hit target_avg_vol below, so the risk/return-implied
    # book *shape* stays anchored to target_avg_vol regardless of the
    # horizon assumption (confirmed: a joint rescale of alpha and a
    # vol-calibrated lambda leaves the unconstrained optimum unchanged) --
    # but the horizon scaling *does* correctly change how that book compares
    # against the one-time transaction cost below, which does not rescale.
    a = a * max(risk.holding_horizon_days, 1e-9)

    if cov is None or cov.shape != (n, n) or not np.all(np.isfinite(cov)):
        cov = np.eye(n) * ((0.30 ** 2) / _TRADING_DAYS_PER_YEAR)
        cov_status_note = "diagonal_cov_fallback"
    else:
        cov_status_note = "optimal"

    lam = risk.risk_aversion
    if lam is None:
        # Derive lambda so ex-ante vol at the unconstrained optimum matches
        # target_avg_vol: vol ~= (1/lambda) * sqrt(alpha' Sigma^-1 alpha).
        try:
            quad = float(a @ np.linalg.solve(cov, a))
            quad = max(quad, 1e-12)
            sigma_target_daily = risk.target_avg_vol / (_TRADING_DAYS_PER_YEAR ** 0.5)
            lam = (quad ** 0.5) / max(sigma_target_daily, 1e-12)
        except np.linalg.LinAlgError:
            lam = 50.0  # empirically-reasonable order-of-magnitude default
    lam = max(lam, 1e-6)
    lam_ridge = lam * risk.ridge_frac

    c_lin, c_imp = _transaction_cost_terms(symbols, cost, constraints)
    sigma_max_daily = constraints.vol_target * constraints.vol_budget_scale / (_TRADING_DAYS_PER_YEAR ** 0.5)
    gross_cap = constraints.max_gross_exposure * constraints.gross_budget_scale

    status = "optimal"
    notes = cov_status_note
    w_solved: np.ndarray | None = None

    if time.monotonic() - t0 < _SOLVE_TIME_BUDGET_SECONDS:
        w_solved = _solve_qp(
            a, w0, cov, lam, lam_ridge, c_lin, c_imp,
            constraints, gross_cap, sigma_max_daily, symbols,
            deadline=t0 + _SOLVE_TIME_BUDGET_SECONDS,
        )
        if w_solved is None:
            status = "closed_form_fallback"
            notes = "cvxpy solve failed/infeasible/timed out"

    if w_solved is None:
        w_solved = _closed_form_fallback(a, cov, lam, constraints)
        # Final sanitizer -- must never emit non-finite/None/unbounded.
        if not np.all(np.isfinite(w_solved)):
            log.error("solve_portfolio: closed-form fallback produced non-finite weights -- holding")
            return SolveResult(
                targets=dict(zip(symbols, w0)), status="hold_fallback",
                solve_seconds=time.monotonic() - t0, notes="fallback produced non-finite weights",
            )

    targets = {s: float(w_solved[i]) for i, s in enumerate(symbols) if abs(w_solved[i]) > 1e-10}
    return SolveResult(
        targets=targets, status=status, solve_seconds=time.monotonic() - t0,
        risk_aversion_used=lam, notes=notes,
    )


def _solve_qp(
    a: np.ndarray,
    w0: np.ndarray,
    cov: np.ndarray,
    lam: float,
    lam_ridge: float,
    c_lin: float,
    c_imp: np.ndarray,
    constraints: OptimizerConstraints,
    gross_cap: float,
    sigma_max_daily: float,
    symbols: list[str],
    deadline: float,
) -> np.ndarray | None:
    """The actual cvxpy solve. Returns ``None`` on any failure/infeasibility/
    timeout so :func:`solve_portfolio` can fall back -- never raises."""
    try:
        import cvxpy as cp
    except Exception:
        log.error("cvxpy unavailable -- falling back to closed-form solve", exc_info=True)
        return None

    n = len(symbols)
    try:
        w = cp.Variable(n)
        dw = w - w0

        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Not PD (can happen at the diagonal-fallback boundary with a
            # zero variance) -- nudge the diagonal, matches standard
            # jitter practice for near-singular covariance.
            L = np.linalg.cholesky(cov + np.eye(n) * 1e-10)

        risk_term = 0.5 * lam * cp.sum_squares(L.T @ w)
        # sum_squares(w-w0) is in raw-weight units, not the covariance's
        # variance units the risk term above is scaled in (variance ~1e-4
        # for daily equity returns) -- multiplying by bare lam_ridge alone
        # would make the ridge penalty ~1e4x too strong relative to the risk
        # term for the same nominal "fraction of lambda", confirmed
        # empirically (an early version of this function barely traded at
        # all with ridge_frac=0.075 even at zero transaction cost). Scale by
        # the covariance's average variance so ridge_frac is genuinely
        # comparable to the risk term's strength, as intended.
        avg_var = float(np.mean(np.diag(cov)))
        ridge_term = 0.5 * lam_ridge * avg_var * cp.sum_squares(w - w0)
        cost_term = c_lin * cp.norm1(dw)
        if np.any(c_imp > 0):
            cost_term = cost_term + cp.sum(cp.multiply(c_imp, cp.power(cp.abs(dw), 1.5)))

        objective = cp.Maximize(a @ w - risk_term - ridge_term - cost_term)

        cons = [
            w <= constraints.max_position_pct,
            w >= -constraints.max_position_pct,
            cp.norm1(w) <= gross_cap,
            cp.sum(w) <= constraints.max_net_exposure,
            cp.sum(w) >= -constraints.max_net_exposure,
        ]

        nav = constraints.nav if constraints.nav > 0 else 1.0
        for i, s in enumerate(symbols):
            adv = constraints.adv_dollars.get(s)
            if adv and adv > 0 and constraints.max_participation_pct > 0:
                trade_cap = constraints.max_participation_pct * adv / nav
                cons.append(cp.abs(dw[i]) <= trade_cap)

        # Soft vol constraint (slack-penalized so w=w0 stays feasible even
        # if realized vol has spiked since the covariance was estimated).
        xi_vol = cp.Variable(nonneg=True)
        cons.append(cp.norm(L.T @ w, 2) <= sigma_max_daily + xi_vol)
        slack_penalty = 1e4 * xi_vol

        # Soft per-sector cap, same slack treatment.
        slack_terms = [slack_penalty]
        if constraints.sector_map:
            sectors: dict[str, list[int]] = {}
            for i, s in enumerate(symbols):
                sec = constraints.sector_map.get(s)
                if sec:
                    sectors.setdefault(sec, []).append(i)
            for sec, idxs in sectors.items():
                if len(idxs) < 2:
                    continue
                xi_sec = cp.Variable(nonneg=True)
                cons.append(cp.norm1(w[idxs]) <= constraints.max_sector_pct + xi_sec)
                slack_terms.append(1e4 * xi_sec)

        objective = cp.Maximize(a @ w - risk_term - ridge_term - cost_term - cp.sum(slack_terms))
        problem = cp.Problem(objective, cons)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        # Single-threaded for determinism -- see module docstring. cvxpy
        # raises a UserWarning on "optimal_inaccurate" -- a status this
        # function already treats as acceptable (subject to the harder
        # post-solve constraint re-check below), so the warning is pure log
        # noise here, not a signal this call site needs to react to.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            problem.solve(solver=cp.CLARABEL, verbose=False)

        if problem.status not in ("optimal", "optimal_inaccurate"):
            log.warning("solve_portfolio: cvxpy status=%s -- falling back", problem.status)
            return None
        w_val = w.value
        if w_val is None or not np.all(np.isfinite(w_val)):
            return None
        # Post-solve sanity re-check on the *hard* constraints (defense in
        # depth -- an "optimal_inaccurate" status can very slightly violate).
        eps = 1e-4
        if (
            np.any(np.abs(w_val) > constraints.max_position_pct + eps)
            or np.abs(w_val).sum() > gross_cap + eps
            or abs(w_val.sum()) > constraints.max_net_exposure + eps
        ):
            log.warning("solve_portfolio: solved book fails hard-constraint re-check -- falling back")
            return None
        return w_val
    except Exception:
        log.warning("solve_portfolio: cvxpy solve raised -- falling back", exc_info=True)
        return None
