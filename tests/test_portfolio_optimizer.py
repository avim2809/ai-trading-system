"""Tests for the joint constrained portfolio optimizer
(src/firm/portfolio/optimizer.py) -- see PART 2 of docs/remediation_progress.md
for the full design rationale.
"""

from __future__ import annotations

import numpy as np
import pytest

from firm.portfolio.optimizer import (
    CostParams,
    OptimizerConstraints,
    RiskAversionParams,
    compute_alpha,
    diagonal_covariance,
    estimate_covariance,
    estimate_ic,
    solve_portfolio,
)


def _toy_cov(symbols, seed=42, scale=0.01):
    rng = np.random.default_rng(seed)
    n = len(symbols)
    a = rng.standard_normal((n, n)) * scale
    return a @ a.T + np.eye(n) * 1e-5


class TestSolvePortfolioDegradationCascade:
    """Never raise, never emit non-finite/None/unbounded, never hang -- this
    runs unattended inside a live cycle. Each rung of the documented
    cascade (primary solve -> diagonal cov -> closed-form -> hold) gets its
    own test."""

    def test_all_zero_alpha_holds_current_weights_exactly(self):
        symbols = ["A", "B", "C"]
        w0 = {"A": 0.02, "B": -0.01, "C": 0.0}
        result = solve_portfolio(
            {s: 0.0 for s in symbols}, w0, symbols, np.eye(3) * 1e-4,
            OptimizerConstraints(), CostParams(), RiskAversionParams(),
        )
        assert result.status == "hold_fallback"
        assert result.targets == w0

    def test_non_finite_alpha_holds_current_weights(self):
        symbols = ["A", "B", "C"]
        w0 = {"A": 0.02, "B": -0.01, "C": 0.0}
        result = solve_portfolio(
            {"A": float("nan"), "B": 0.001, "C": 0.0}, w0, symbols,
            np.eye(3) * 1e-4, OptimizerConstraints(), CostParams(), RiskAversionParams(),
        )
        assert result.status == "hold_fallback"
        assert result.targets == w0

    def test_empty_universe_returns_empty_without_raising(self):
        result = solve_portfolio({}, {}, [], None, OptimizerConstraints(), CostParams(), RiskAversionParams())
        assert result.targets == {}
        assert result.status == "hold_fallback"

    def test_cov_none_degrades_to_diagonal_fallback_and_still_solves(self):
        symbols = ["A", "B", "C"]
        alpha = {"A": 0.002, "B": -0.001, "C": 0.0005}
        w0 = {"A": 0.0, "B": 0.0, "C": 0.0}
        result = solve_portfolio(
            alpha, w0, symbols, None, OptimizerConstraints(), CostParams(), RiskAversionParams(),
        )
        assert result.status == "optimal"
        assert result.notes == "diagonal_cov_fallback"
        assert all(np.isfinite(v) for v in result.targets.values())

    def test_mismatched_cov_shape_degrades_to_diagonal_fallback(self):
        """A caller-supplied cov for the wrong symbol count must not crash
        the solve -- degrade exactly like cov=None."""
        symbols = ["A", "B", "C"]
        alpha = {"A": 0.002, "B": -0.001, "C": 0.0005}
        w0 = {"A": 0.0, "B": 0.0, "C": 0.0}
        result = solve_portfolio(
            alpha, w0, symbols, np.eye(5), OptimizerConstraints(), CostParams(), RiskAversionParams(),
        )
        assert result.notes == "diagonal_cov_fallback"
        assert all(np.isfinite(v) for v in result.targets.values())

    def test_never_raises_on_pathological_inputs(self):
        """A grab-bag of degenerate inputs that must all degrade cleanly,
        never raise, never hang."""
        symbols = ["A", "B", "C"]
        w0 = {"A": 0.01, "B": 0.0, "C": -0.01}
        pathological_covs = [
            np.zeros((3, 3)),
            np.full((3, 3), np.inf),
            -np.eye(3),  # negative variance, nonsensical but finite
        ]
        for cov in pathological_covs:
            result = solve_portfolio(
                {"A": 0.001, "B": -0.0005, "C": 0.0002}, w0, symbols, cov,
                OptimizerConstraints(), CostParams(), RiskAversionParams(),
            )
            assert all(np.isfinite(v) for v in result.targets.values()), cov


class TestSolvePortfolioHardConstraints:
    """The whole "safe to run unattended" guarantee rests on box/gross/net
    being genuinely hard -- verify the solved book actually respects them,
    not just that the solve returns a status."""

    def test_position_cap_respected(self):
        symbols = [f"S{i}" for i in range(8)]
        rng = np.random.default_rng(1)
        alpha = {s: float(rng.standard_normal() * 0.01) for s in symbols}
        w0 = {s: 0.0 for s in symbols}
        constraints = OptimizerConstraints(max_position_pct=0.05, nav=1_000_000)
        result = solve_portfolio(
            alpha, w0, symbols, _toy_cov(symbols), constraints, CostParams(), RiskAversionParams(),
        )
        for v in result.targets.values():
            assert abs(v) <= 0.05 + 1e-6

    def test_gross_exposure_cap_respected(self):
        symbols = [f"S{i}" for i in range(10)]
        rng = np.random.default_rng(2)
        alpha = {s: float(abs(rng.standard_normal()) * 0.01) for s in symbols}  # all-long, strong signal
        w0 = {s: 0.0 for s in symbols}
        constraints = OptimizerConstraints(max_gross_exposure=0.5, max_position_pct=0.5, nav=1_000_000)
        result = solve_portfolio(
            alpha, w0, symbols, _toy_cov(symbols), constraints, CostParams(), RiskAversionParams(),
        )
        gross = sum(abs(v) for v in result.targets.values())
        assert gross <= 0.5 + 1e-4

    def test_net_exposure_cap_respected(self):
        symbols = [f"S{i}" for i in range(8)]
        rng = np.random.default_rng(3)
        alpha = {s: float(abs(rng.standard_normal()) * 0.01) for s in symbols}  # all-long
        w0 = {s: 0.0 for s in symbols}
        constraints = OptimizerConstraints(max_net_exposure=0.3, max_position_pct=0.5, max_gross_exposure=2.0, nav=1_000_000)
        result = solve_portfolio(
            alpha, w0, symbols, _toy_cov(symbols), constraints, CostParams(), RiskAversionParams(),
        )
        net = sum(result.targets.values())
        assert abs(net) <= 0.3 + 1e-4

    def test_liquidity_trade_cap_respected(self):
        """|Delta w_i| must never exceed max_participation_pct * ADV/NAV,
        even when the signal wants a much bigger move."""
        symbols = ["A", "B"]
        alpha = {"A": 0.01, "B": -0.01}
        w0 = {"A": 0.0, "B": 0.0}
        nav = 1_000_000.0
        constraints = OptimizerConstraints(
            max_position_pct=0.5, max_gross_exposure=2.0, max_net_exposure=2.0,
            max_participation_pct=0.10, adv_dollars={"A": 100_000.0, "B": 100_000.0}, nav=nav,
        )
        result = solve_portfolio(
            alpha, w0, symbols, _toy_cov(symbols), constraints, CostParams(), RiskAversionParams(),
        )
        trade_cap = 0.10 * 100_000.0 / nav  # = 0.01
        for s in symbols:
            assert abs(result.targets.get(s, 0.0) - w0[s]) <= trade_cap + 1e-6


class TestSolvePortfolioSensibleBehavior:
    """Directional/economic sanity checks -- not just "doesn't crash."""

    def test_hold_current_weights_is_always_feasible_and_cheap(self):
        """w=w0 must never be rejected by the hard constraints (the
        no-trade-region/feasibility guarantee the design relies on) --
        confirmed by checking a solve where costs are enormous relative to
        alpha lands very close to w0."""
        symbols = ["A", "B", "C"]
        w0 = {"A": 0.03, "B": -0.02, "C": 0.01}
        alpha = {"A": 0.0001, "B": -0.00005, "C": 0.00002}  # tiny signal
        huge_cost = CostParams(commission_pct=0.05, slippage_pct=0.05, spread_pct=0.05)
        result = solve_portfolio(
            alpha, w0, symbols, _toy_cov(symbols), OptimizerConstraints(nav=1_000_000),
            huge_cost, RiskAversionParams(),
        )
        for s in symbols:
            assert abs(result.targets.get(s, 0.0) - w0[s]) < 0.01

    def test_higher_cost_aversion_reduces_turnover(self):
        symbols = ["A", "B", "C", "D"]
        rng = np.random.default_rng(5)
        alpha = {s: float(rng.standard_normal() * 0.002) for s in symbols}
        w0 = {s: 0.0 for s in symbols}
        cov = _toy_cov(symbols, seed=5)
        constraints = OptimizerConstraints(nav=1_000_000, adv_dollars={s: 5_000_000 for s in symbols})

        low_cost = CostParams(cost_aversion=1.0, commission_pct=0.0002, slippage_pct=0.0001, spread_pct=0.0001)
        high_cost = CostParams(cost_aversion=1.0, commission_pct=0.01, slippage_pct=0.01, spread_pct=0.01)
        r_low = solve_portfolio(alpha, w0, symbols, cov, constraints, low_cost, RiskAversionParams())
        r_high = solve_portfolio(alpha, w0, symbols, cov, constraints, high_cost, RiskAversionParams())

        turnover_low = sum(abs(v) for v in r_low.targets.values())
        turnover_high = sum(abs(v) for v in r_high.targets.values())
        assert turnover_high < turnover_low

    def test_stronger_conviction_produces_larger_book_at_fixed_target_vol(self):
        """Root cause 2 fix: gross exposure should scale with aggregate
        signal strength, not be forced to a constant -- unlike the old
        L1-normalize-to-1 behavior this replaces."""
        symbols = ["A", "B", "C", "D", "E"]
        cov = _toy_cov(symbols, seed=9)
        w0 = {s: 0.0 for s in symbols}
        constraints = OptimizerConstraints(nav=1_000_000, max_position_pct=0.5, max_gross_exposure=3.0)
        cost = CostParams(commission_pct=0.0, slippage_pct=0.0, spread_pct=0.0)  # isolate the vol/alpha effect
        risk = RiskAversionParams(ridge_frac=0.0)

        weak = {s: 0.0002 * (1 if i % 2 == 0 else -1) for i, s in enumerate(symbols)}
        strong = {s: 0.002 * (1 if i % 2 == 0 else -1) for i, s in enumerate(symbols)}
        r_weak = solve_portfolio(weak, w0, symbols, cov, constraints, cost, risk)
        r_strong = solve_portfolio(strong, w0, symbols, cov, constraints, cost, risk)

        gross_weak = sum(abs(v) for v in r_weak.targets.values())
        gross_strong = sum(abs(v) for v in r_strong.targets.values())
        assert gross_strong > gross_weak

    def test_holding_horizon_of_one_day_barely_trades_confirmed_regression(self):
        """Regression coverage for a real bug found while building this: with
        holding_horizon_days=1, a single day's alpha essentially never repays
        a one-time transaction cost, so the optimizer held current weights
        almost unconditionally even with real signal. holding_horizon_days>1
        (the default, 5) must trade meaningfully more."""
        symbols = ["A", "B", "C"]
        cov = _toy_cov(symbols, seed=11)
        alpha = {"A": 0.001, "B": 0.0008, "C": -0.0005}
        w0 = {"A": 0.05, "B": 0.0, "C": 0.03}
        constraints = OptimizerConstraints(nav=1_000_000, adv_dollars={s: 5_000_000 for s in symbols})
        cost = CostParams()

        r_one_day = solve_portfolio(alpha, w0, symbols, cov, constraints, cost, RiskAversionParams(holding_horizon_days=1.0))
        r_five_day = solve_portfolio(alpha, w0, symbols, cov, constraints, cost, RiskAversionParams(holding_horizon_days=5.0))

        turnover_one_day = sum(abs(r_one_day.targets.get(s, 0.0) - w0[s]) for s in symbols)
        turnover_five_day = sum(abs(r_five_day.targets.get(s, 0.0) - w0[s]) for s in symbols)
        assert turnover_one_day < 0.001  # essentially no trading
        assert turnover_five_day > turnover_one_day


class TestSolvePortfolioDeterminism:
    """test_e2e.py::test_reproducibility requires two seeded orchestrator
    runs to produce identical proposal.targets -- confirmed empirically that
    a single-threaded Clarabel solve on identical inputs is bit-for-bit
    reproducible, so this holds to exact equality, not just a tolerance."""

    def test_repeated_solves_on_identical_inputs_are_bit_identical(self):
        symbols = sorted(["AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "JPM"])
        rng = np.random.default_rng(7)
        alpha = {s: float(rng.standard_normal() * 0.001) for s in symbols}
        w0 = {s: float(rng.uniform(-0.03, 0.03)) for s in symbols}
        cov = _toy_cov(symbols, seed=7)
        constraints = OptimizerConstraints(nav=1_000_000, adv_dollars={s: 5_000_000 for s in symbols})
        cost = CostParams()
        risk = RiskAversionParams()

        results = [
            solve_portfolio(alpha, w0, symbols, cov, constraints, cost, risk).targets
            for _ in range(3)
        ]
        assert results[0] == results[1] == results[2]


class TestEstimateIC:
    def test_cold_start_returns_prior_exactly(self):
        assert estimate_ic(None) == pytest.approx(0.03)
        assert estimate_ic(np.array([0.001])) == pytest.approx(0.03)  # n=1, too thin

    def test_never_negative_and_never_exceeds_cap(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            returns = rng.standard_normal(200) * 0.01 + rng.uniform(-0.01, 0.01)
            ic = estimate_ic(returns, ic_cap=0.15)
            assert 0.0 <= ic <= 0.15

    def test_persistently_negative_ir_shrinks_toward_zero_not_negative(self):
        """A reliably negative IR clips its own `sample_ic` term to exactly
        0 (never negative -- the sign-flip guarantee), but `ic_eff` blends
        that with `(1-phi)*ic_prior` (n0-style shrinkage), so for finite n
        it lands strictly between 0 and ic_prior, not at exactly 0."""
        rng = np.random.default_rng(2)
        bad_returns = -np.abs(rng.standard_normal(200)) * 0.01 - 0.001  # reliably negative
        ic = estimate_ic(bad_returns, ic_prior=0.03, n0=90)
        assert 0.0 <= ic < 0.03

    def test_realistic_good_track_record_builds_trust_above_the_prior(self):
        """Regression for a real units bug found in the first walk-forward+
        PBO gate run of this module: `ir_daily` (mean/std of *daily*
        returns) was compared directly against `ir_ref`/`ir_cap`, which are
        annualized-IR-scale thresholds -- dividing by sqrt(252) is exactly
        what annualizing does, so a genuinely strong annualized Sharpe of
        ~1.5 has ir_daily ~= 0.09, chronically far below ir_ref=1.0. Under
        the bug, a strategy with a real, strong realized track record ended
        up LESS trusted than the cold-start prior (ic_eff < ic_prior) --
        backwards. A realistic ~1% daily vol book with an annualized Sharpe
        of ~1.5 must build trust *above* the prior, not below it."""
        rng = np.random.default_rng(11)
        daily_vol = 0.01
        target_annualized_sharpe = 1.5
        daily_mean = target_annualized_sharpe * daily_vol / (252 ** 0.5)
        returns = rng.standard_normal(300) * daily_vol + daily_mean
        ic = estimate_ic(returns, ic_prior=0.03, ir_ref=1.0, ir_cap=2.0, n0=90)
        assert ic > 0.03

    def test_more_negative_evidence_shrinks_further_toward_zero(self):
        rng = np.random.default_rng(2)
        short_bad = -np.abs(rng.standard_normal(100)) * 0.01 - 0.001
        long_bad = -np.abs(rng.standard_normal(2000)) * 0.01 - 0.001
        ic_short = estimate_ic(short_bad, ic_prior=0.03, n0=90)
        ic_long = estimate_ic(long_bad, ic_prior=0.03, n0=90)
        assert ic_long < ic_short  # more observations of "reliably bad" -> closer to 0

    def test_strong_positive_ir_approaches_the_cap(self):
        rng = np.random.default_rng(3)
        good_returns = np.abs(rng.standard_normal(500)) * 0.001 + 0.002  # reliably positive, low noise
        ic = estimate_ic(good_returns, ic_cap=0.15, n0=90)
        assert ic > 0.05  # meaningfully above the prior


class TestComputeAlpha:
    def test_zero_vol_symbol_gets_zero_alpha_not_nan(self):
        alpha = compute_alpha({"A": 0.5, "B": 0.5}, {"A": 0.2, "B": 0.0}, ic_eff=0.03)
        assert alpha["B"] == 0.0
        assert alpha["A"] != 0.0

    def test_missing_vol_gets_zero_alpha(self):
        alpha = compute_alpha({"A": 0.5}, {}, ic_eff=0.03)
        assert alpha["A"] == 0.0

    def test_sign_matches_conviction_sign(self):
        alpha = compute_alpha({"A": 0.5, "B": -0.5}, {"A": 0.2, "B": 0.2}, ic_eff=0.03)
        assert alpha["A"] > 0
        assert alpha["B"] < 0

    def test_higher_ic_scales_magnitude_up(self):
        low = compute_alpha({"A": 0.5}, {"A": 0.2}, ic_eff=0.01)
        high = compute_alpha({"A": 0.5}, {"A": 0.2}, ic_eff=0.10)
        assert abs(high["A"]) > abs(low["A"])


class TestEstimateCovariance:
    def test_too_few_observations_returns_none(self):
        import pandas as pd

        rng = np.random.default_rng(1)
        returns = {s: pd.Series(rng.standard_normal(30) * 0.01) for s in ["A", "B", "C"]}
        assert estimate_covariance(returns, ["A", "B", "C"]) is None

    def test_enough_history_returns_a_finite_well_conditioned_matrix(self):
        import pandas as pd

        rng = np.random.default_rng(1)
        n_obs = 300
        returns = {s: pd.Series(rng.standard_normal(n_obs) * 0.01) for s in ["A", "B", "C"]}
        cov = estimate_covariance(returns, ["A", "B", "C"])
        assert cov is not None
        assert cov.shape == (3, 3)
        assert np.all(np.isfinite(cov))
        assert np.linalg.cond(cov) < 1e10

    def test_missing_symbol_history_does_not_crash(self):
        import pandas as pd

        rng = np.random.default_rng(1)
        returns = {s: pd.Series(rng.standard_normal(300) * 0.01) for s in ["A", "B"]}
        cov = estimate_covariance(returns, ["A", "B", "C"])  # C has no history
        # Either None (degraded, acceptable) or a finite well-conditioned
        # matrix (the fallback-variance-for-missing-symbols path) -- must
        # not raise either way.
        if cov is not None:
            assert np.all(np.isfinite(cov))


class TestDiagonalCovariance:
    def test_uses_provided_vols(self):
        cov = diagonal_covariance({"A": 0.20, "B": 0.40}, ["A", "B"])
        assert cov[0, 0] < cov[1, 1]  # B has 2x the vol -> 4x the variance
        assert cov[0, 1] == 0.0  # diagonal only

    def test_missing_symbol_gets_a_conservative_default_not_zero(self):
        cov = diagonal_covariance({}, ["A"])
        assert cov[0, 0] > 0
