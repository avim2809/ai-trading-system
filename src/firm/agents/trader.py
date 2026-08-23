"""Trader / portfolio manager agent.

Converts ``DebateResult`` conviction scores into target portfolio weights
via a configurable allocation method (conviction-weighted, risk-parity, or
equal-weight), respecting per-strategy capital budgets and a maximum
number of positions.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np

from firm.agents.base import Agent, AgentContext
from firm.contracts.models import DebateResult, TradeProposal
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

log = logging.getLogger(__name__)


class TraderAgent(Agent):
    """Produces a TradeProposal from debate-synthesised convictions."""

    role = "trader"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="trader", config=config)
        cfg = config or {}
        self.allocation_method: str = cfg.get("allocation_method", "conviction_weighted")
        self.max_positions: int = cfg.get("max_positions", 20)
        self.strategy_budgets: dict[str, float] = cfg.get("strategy_budgets", {})
        # Fraction of full Kelly to bet (default half-Kelly, the standard
        # uncertainty haircut). Only used when allocation_method == "kelly".
        self.kelly_fraction: float = float(cfg.get("kelly_fraction", 0.5))
        # Confirmed live (2026-08-03 through 2026-08-07): with max_positions
        # (20) sitting at the universe size, the *average* position lands
        # exactly on risk.max_position_pct (0.05) by construction — so daily
        # noise in the unsmoothed, freshly-z-scored conviction pushes ~half
        # the book above/below cap (and can flip sign) every single cycle,
        # producing 60-95% daily turnover against a nominal 25% cap. EMA
        # smoothing damps that cycle-to-cycle noise; off by default since it
        # changes live/backtest trading behavior and every other
        # risk-bearing toggle in this file is explicit opt-in.
        self.conviction_smoothing_enabled: bool = bool(
            cfg.get("conviction_smoothing_enabled", False)
        )
        halflife_days = float(cfg.get("conviction_smoothing_halflife_days", 3.0))
        # EMA weight on today's new reading, derived from the half-life so
        # the config knob stays interpretable ("days until a shock's
        # influence halves") instead of a bare, unintuitive alpha.
        self._conviction_smoothing_alpha: float = (
            1.0 - 0.5 ** (1.0 / halflife_days) if halflife_days > 0 else 1.0
        )
        self._conviction_ema: dict[str, float] = {}

        # allocation_method == "joint_optimizer" config. Field names for the
        # structural constraints deliberately match RiskAgent's own config
        # keys (max_position_pct, max_gross_exposure, ...) and the shared
        # cost-model keys BacktestConfig/_liquidity.py/ExecutionAgent already
        # use (commission_pct, ...) -- both agents are constructed from the
        # same flat config dict (see runtime.build_orchestrator), so this
        # reads values already present for RiskAgent/the cost model rather
        # than inventing parallel ones. See src/firm/portfolio/optimizer.py
        # and PART 2 of the remediation plan for the full design.
        self._optimizer_constraints = OptimizerConstraints(
            max_position_pct=cfg.get("max_position_pct", 0.05),
            max_gross_exposure=cfg.get("max_gross_exposure", 2.0),
            max_net_exposure=cfg.get("max_net_exposure", 0.5),
            max_sector_pct=cfg.get("max_sector_pct", 0.25),
            vol_target=cfg.get("vol_target", 0.15),
            # RiskAgent's own max_participation_pct defaults to None (its
            # liquidity cap is opt-in); the optimizer treats a liquidity
            # bound as a structural hard constraint always worth having, so
            # it falls back to a sensible default instead of inheriting
            # "off".
            max_participation_pct=cfg.get("max_participation_pct") or 0.10,
            sector_map=cfg.get("sector_map", {}) or {},
        )
        self._optimizer_cost = CostParams(
            commission_pct=cfg.get("commission_pct", 0.001),
            slippage_pct=cfg.get("slippage_pct", 0.0005),
            spread_pct=cfg.get("spread_pct", 0.0002),
            market_impact_coefficient=cfg.get("market_impact_coefficient", 0.0),
            cost_aversion=float(cfg.get("optimizer_cost_aversion", 1.0)),
        )
        self._optimizer_risk = RiskAversionParams(
            target_avg_vol=float(cfg.get("optimizer_target_avg_vol", 0.085)),
            ridge_frac=float(cfg.get("optimizer_ridge_frac", 0.075)),
            holding_horizon_days=float(cfg.get("optimizer_holding_horizon_days", 5.0)),
        )
        self._optimizer_adv_lookback_days: int = int(cfg.get("adv_lookback_days", 20))
        self._optimizer_cov_lookback_days: int = int(cfg.get("optimizer_cov_lookback_days", 252))
        # Rolling NAV history for the joint_optimizer's Path-B IC proxy --
        # see _update_and_estimate_ic's docstring for why this is
        # self-maintained rather than read from ctx.portfolio.history.
        self._book_nav_history: list[float] = []
        self._BOOK_NAV_HISTORY_MAXLEN: int = 400

    def get_state(self) -> dict[str, Any]:
        """Conviction-EMA memory + joint_optimizer's rolling NAV history for
        cross-restart persistence (see
        ``firm.live.state_store.LiveStateStore.save_trader_state``)."""
        return {
            "conviction_ema": dict(self._conviction_ema),
            "book_nav_history": list(self._book_nav_history),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._conviction_ema = dict(state.get("conviction_ema") or {})
        self._book_nav_history = list(state.get("book_nav_history") or [])

    def _smooth_convictions(
        self, results: list[DebateResult]
    ) -> list[DebateResult]:
        """Blend each symbol's fresh conviction with its running EMA.

        A symbol with no prior EMA (new to the universe, or its first
        appearance after a gap) starts at full strength rather than being
        damped toward zero — there is no "yesterday" to blend with yet.
        """
        alpha = self._conviction_smoothing_alpha
        smoothed = []
        for r in results:
            prev = self._conviction_ema.get(r.symbol)
            ema = r.net_conviction if prev is None else (
                alpha * r.net_conviction + (1.0 - alpha) * prev
            )
            self._conviction_ema[r.symbol] = ema
            smoothed.append(replace(r, net_conviction=ema))
        return smoothed

    def run(self, ctx: AgentContext, **inputs: Any) -> TradeProposal:
        debate_results: list[DebateResult] = inputs.get("debate_results", [])
        blackboard = inputs.get("blackboard")

        if self.conviction_smoothing_enabled:
            debate_results = self._smooth_convictions(debate_results)

        ranked = sorted(debate_results, key=lambda r: abs(r.net_conviction), reverse=True)
        selected = [r for r in ranked if abs(r.net_conviction) > 1e-8][: self.max_positions]

        if self.allocation_method == "equal_weight":
            targets = self._equal_weight(selected)
        elif self.allocation_method == "risk_parity":
            targets = self._risk_parity(selected, ctx)
        elif self.allocation_method == "kelly":
            targets = self._kelly(selected, ctx)
        elif self.allocation_method == "joint_optimizer":
            targets = self._joint_optimizer(selected, ctx, inputs.get("prices") or {})
        else:
            targets = self._conviction_weighted(selected)

        log.debug(
            "TraderAgent allocation=%s selected=%d/%d -> %d target positions",
            self.allocation_method, len(selected), len(debate_results), len(targets),
        )

        per_strategy = self._attribute_to_strategies(targets, blackboard)

        return TradeProposal(
            asof=ctx.now,
            targets=targets,
            per_strategy=per_strategy,
            notes=(
                f"Method: {self.allocation_method}, "
                f"{len(selected)}/{len(debate_results)} positions selected"
            ),
        )

    # ------------------------------------------------------------------
    # allocation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _conviction_weighted(results: list[DebateResult]) -> dict[str, float]:
        total = sum(abs(r.net_conviction) for r in results)
        if total == 0:
            return {}
        return {r.symbol: r.net_conviction / total for r in results}

    @staticmethod
    def _equal_weight(results: list[DebateResult]) -> dict[str, float]:
        if not results:
            return {}
        w = 1.0 / len(results)
        return {r.symbol: w if r.net_conviction >= 0 else -w for r in results}

    def _risk_parity(
        self, results: list[DebateResult], ctx: AgentContext
    ) -> dict[str, float]:
        """Inverse-volatility weighting (signed by conviction).

        Each name is weighted ∝ 1/vol using a realized-vol estimate from the
        PitView, so high-vol names get smaller allocations.  Falls back to
        equal weight only when no vol estimates are available.
        """
        if not results:
            return {}
        pit_view = getattr(ctx, "pit_view", None)
        inv_vol: dict[str, float] = {}
        for r in results:
            vol = self._estimate_vol(pit_view, r.symbol)
            if vol and vol > 0:
                inv_vol[r.symbol] = 1.0 / vol

        total = sum(inv_vol.values())
        if total <= 0:
            # No usable vol data – degrade to equal weight.
            w = 1.0 / len(results)
            return {r.symbol: w if r.net_conviction >= 0 else -w for r in results}

        targets: dict[str, float] = {}
        for r in results:
            mag = inv_vol.get(r.symbol, 0.0) / total
            targets[r.symbol] = mag if r.net_conviction >= 0 else -mag
        return targets

    def _kelly(
        self, results: list[DebateResult], ctx: AgentContext
    ) -> dict[str, float]:
        """Fractional-Kelly allocation from each name's return history.

        For every selected name we estimate the win probability ``p`` and the
        win/loss payoff ratio ``b`` from its realized daily returns in the
        PitView, compute the classical Kelly fraction ``f = (p*b - q)/b``,
        apply :attr:`kelly_fraction` (default half-Kelly), and sign it by the
        conviction direction. Negative-edge names get zero weight. The surviving
        magnitudes are L1-normalised so ``sum |w| = 1``. Falls back to
        conviction weighting when no name has a positive edge (thin history).
        """
        if not results:
            return {}
        pit_view = getattr(ctx, "pit_view", None)

        magnitudes: dict[str, float] = {}
        signs: dict[str, float] = {}
        for r in results:
            edge = self._kelly_edge(pit_view, r.symbol)
            if edge is None or edge <= 0:
                continue
            magnitudes[r.symbol] = edge * self.kelly_fraction
            signs[r.symbol] = 1.0 if r.net_conviction >= 0 else -1.0

        total = sum(magnitudes.values())
        if total <= 0:
            # No positive-edge names — degrade to conviction weighting.
            log.debug(
                "Kelly: no positive-edge names among %d candidates "
                "(thin history); falling back to conviction weighting",
                len(results),
            )
            return self._conviction_weighted(results)
        log.debug(
            "Kelly: %d/%d names had positive edge (fraction=%.2f)",
            len(magnitudes), len(results), self.kelly_fraction,
        )

        return {sym: signs[sym] * mag / total for sym, mag in magnitudes.items()}

    @classmethod
    def _kelly_edge(
        cls, pit_view: Any, symbol: str, lookback: int = 252
    ) -> float | None:
        """Full-Kelly fraction from *symbol*'s realized daily returns, or None.

        ``f = (p*b - q)/b`` where ``p`` is the empirical win rate and ``b`` is
        the average-win / average-loss payoff ratio. Returns None when history
        is too thin or degenerate. May be negative (negative edge).
        """
        returns = cls._symbol_returns(pit_view, symbol, lookback)
        if returns is None or len(returns) < 20:
            return None
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        if len(wins) == 0 or len(losses) == 0:
            return None
        p = float(len(wins) / len(returns))
        avg_win = float(wins.mean())
        avg_loss = float(-losses.mean())
        if avg_loss <= 0:
            return None
        b = avg_win / avg_loss
        if b <= 0:
            return None
        q = 1.0 - p
        return (p * b - q) / b

    @staticmethod
    def _symbol_returns(pit_view: Any, symbol: str, lookback: int):
        """Daily return Series for *symbol* from PitView prices, or None."""
        if pit_view is None:
            return None
        try:
            df = pit_view.prices(symbols=[symbol], lookback_days=lookback)
        except Exception:
            log.debug("Could not load prices for %s return series", symbol, exc_info=True)
            return None
        if df is None or df.empty:
            return None
        col = "adj_close" if "adj_close" in df.columns else "close"
        if col not in df.columns:
            return None
        prices = df.sort_values("date")[col]
        returns = prices.pct_change().dropna()
        return returns if len(returns) >= 2 else None

    @staticmethod
    def _estimate_vol(pit_view: Any, symbol: str, lookback: int = 63) -> float | None:
        """Annualized realized vol for *symbol* from PitView prices, or None."""
        if pit_view is None:
            return None
        try:
            df = pit_view.prices(symbols=[symbol], lookback_days=lookback)
        except Exception:
            log.debug("Could not load prices for %s vol estimate", symbol, exc_info=True)
            return None
        if df is None or df.empty:
            return None
        col = "adj_close" if "adj_close" in df.columns else "close"
        if col not in df.columns:
            return None
        prices = df.sort_values("date")[col]
        returns = prices.pct_change().dropna()
        if len(returns) < 2:
            return None
        vol = float(returns.std() * (252 ** 0.5))
        return vol if vol > 0 else None

    def _update_and_estimate_ic(self, nav: float | None) -> float:
        """Path-B realized-IR IC proxy (see ``firm.portfolio.optimizer.
        estimate_ic``), fed from a rolling NAV history TraderAgent
        maintains itself as instance state.

        Deliberately NOT ``ctx.portfolio.history``: ``PortfolioState.
        record_snapshot()`` (the only thing that appends to ``.history``)
        is only ever called from the live path
        (``firm.live.portfolio_sync``) -- see the explanatory comment in
        ``firm.backtest.engine`` at the equity-curve fallback. Reading
        ``.history`` here would make this Path-B trust-building mechanism
        permanently inert (pinned at ``ic_prior``) in every backtest,
        confirmed by the first walk-forward+PBO gate run of this module
        producing bit-identical results before and after an unrelated
        `estimate_ic` units fix, because the fixed code path was never
        actually reached. Self-maintained history instead runs off
        ``ctx.portfolio.nav`` (already available every cycle in both
        backtest and live), so Path-B is genuinely exercised in both.

        Uses only NAV values recorded in *prior* cycles for the estimate --
        this cycle's own nav is appended only afterward, so there is no
        look-ahead into today's own return.
        """
        returns = None
        if len(self._book_nav_history) >= 2:
            arr = np.asarray(self._book_nav_history, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                rets = arr[1:] / arr[:-1] - 1.0
            rets = rets[np.isfinite(rets)]
            if len(rets) >= 2:
                returns = rets
        ic_eff = estimate_ic(returns)
        if nav is not None and np.isfinite(nav) and nav > 0:
            self._book_nav_history.append(float(nav))
            if len(self._book_nav_history) > self._BOOK_NAV_HISTORY_MAXLEN:
                del self._book_nav_history[: -self._BOOK_NAV_HISTORY_MAXLEN]
        return ic_eff

    def _gather_market_data(
        self, pit_view: Any, symbols: list[str],
    ) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
        """Returns ``(returns_by_symbol, vols, adv_dollars)`` for every
        symbol in *symbols*, from a **single** batched ``pit_view.prices()``
        call.

        Confirmed by a real point-in-time backtest (Q1 2024, 25-name
        universe): calling ``_symbol_returns``/``_estimate_vol``/
        ``estimate_adv_dollars`` independently per symbol (each issuing its
        own single-symbol ``pit_view.prices()`` call — ~3 calls x ~20
        symbols = ~60 calls/cycle, each re-scanning the whole cached price
        panel) pushed a 3-month/61-cycle backtest well past 300s, vs. ~90s
        for the pre-joint_optimizer baseline. One batched multi-symbol call
        plus local pandas grouping removes that ~60x redundancy.
        """
        if pit_view is None or not symbols:
            return {}, {}, {}
        lookback = max(self._optimizer_cov_lookback_days + 60, self._optimizer_adv_lookback_days)
        try:
            df = pit_view.prices(symbols=symbols, lookback_days=lookback)
        except Exception:
            log.debug("joint_optimizer: batched price fetch failed", exc_info=True)
            return {}, {}, {}
        if df is None or df.empty:
            return {}, {}, {}
        price_col = "adj_close" if "adj_close" in df.columns else "close"
        if price_col not in df.columns or "symbol" not in df.columns:
            return {}, {}, {}

        returns_by_symbol: dict[str, Any] = {}
        vols: dict[str, float] = {}
        adv_dollars: dict[str, float] = {}
        has_volume = "volume" in df.columns
        for sym, g in df.groupby("symbol"):
            g = g.sort_values("date")
            rets = g[price_col].pct_change().dropna()
            if len(rets) >= 2:
                returns_by_symbol[sym] = rets
                vol = float(rets.std() * (252 ** 0.5))
                if vol > 0:
                    vols[sym] = vol
            if has_volume:
                adv_window = g.tail(self._optimizer_adv_lookback_days)
                adv = (adv_window["volume"] * adv_window[price_col]).mean()
                if adv == adv and adv > 0:  # NaN-safe
                    adv_dollars[sym] = float(adv)
        return returns_by_symbol, vols, adv_dollars

    def _joint_optimizer(
        self,
        results: list[DebateResult],
        ctx: AgentContext,
        prices: dict[str, float],
    ) -> dict[str, float]:
        """``allocation_method == "joint_optimizer"``: convictions -> target
        weights via the mean-variance-with-costs QP in
        ``firm.portfolio.optimizer`` instead of L1-normalizing to full
        investment. See PART 2 of the remediation plan for the full
        diagnosis/design; this method's only job is to gather this cycle's
        inputs (alpha, covariance, current weights, liquidity/NAV) from
        ``ctx``/``prices`` and hand them to the pure solver, which never
        raises and always degrades gracefully on its own.
        """
        if not results:
            return {}
        pit_view = getattr(ctx, "pit_view", None)
        portfolio = getattr(ctx, "portfolio", None)
        symbols = sorted(r.symbol for r in results)
        net_convictions = {r.symbol: r.net_conviction for r in results}

        returns_by_symbol, fallback_vols, adv_dollars = self._gather_market_data(pit_view, symbols)

        cov = estimate_covariance(returns_by_symbol, symbols, self._optimizer_cov_lookback_days)
        if cov is None:
            cov = diagonal_covariance(fallback_vols, symbols)
            log.debug(
                "joint_optimizer: Ledoit-Wolf covariance unusable (thin/degenerate "
                "history) -- degraded to diagonal covariance from realized vol"
            )

        # alpha's vol input must come from the *same* covariance used in the
        # risk term (see compute_alpha's docstring) -- derive it from cov's
        # own diagonal rather than fallback_vols directly, so alpha and risk
        # stay internally consistent regardless of which cov source won.
        vols_for_alpha = {}
        for i, s in enumerate(symbols):
            daily_var = float(cov[i, i])
            if daily_var > 0 and np.isfinite(daily_var):
                vols_for_alpha[s] = (daily_var * 252) ** 0.5

        current_weights: dict[str, float] = {}
        if portfolio is not None and hasattr(portfolio, "get_weights"):
            try:
                current_weights = portfolio.get_weights(prices)
            except Exception:
                log.debug("joint_optimizer: could not read current weights", exc_info=True)

        # NAV must be read *after* get_weights(prices) above -- that call is
        # what marks portfolio._last_prices for today, so .nav below
        # reflects today's actual marks rather than a stale prior cycle's.
        nav = getattr(portfolio, "nav", None) if portfolio is not None else None
        ic_eff = self._update_and_estimate_ic(nav)
        alpha = compute_alpha(net_convictions, vols_for_alpha, ic_eff)

        constraints = replace(
            self._optimizer_constraints,
            adv_dollars=adv_dollars,
            nav=float(nav) if nav and nav > 0 else 1.0,
        )

        result = solve_portfolio(
            alpha, current_weights, symbols, cov, constraints,
            self._optimizer_cost, self._optimizer_risk,
        )
        if result.status != "optimal":
            log.warning(
                "joint_optimizer: solve degraded to status=%s (%s) -- %d symbols, "
                "ic_eff=%.4f, solve_seconds=%.3f",
                result.status, result.notes, len(symbols), ic_eff, result.solve_seconds,
            )
        else:
            log.debug(
                "joint_optimizer: solved %d symbols in %.3fs (ic_eff=%.4f, "
                "lambda=%s, gross=%.3f, net=%.3f)",
                len(symbols), result.solve_seconds, ic_eff, result.risk_aversion_used,
                sum(abs(w) for w in result.targets.values()),
                sum(result.targets.values()),
            )
        return result.targets

    def _attribute_to_strategies(
        self,
        targets: dict[str, float],
        blackboard: Any,
    ) -> dict[str, dict[str, float]]:
        """Build a per-strategy weight attribution from blackboard signals."""
        if blackboard is None:
            return {}

        per_strategy: dict[str, dict[str, float]] = {}

        for sym, weight in targets.items():
            signals = blackboard.get_signals_by_symbol(sym)
            if not signals:
                per_strategy.setdefault("unattributed", {})[sym] = weight
                continue

            total_score = sum(abs(s.score) for s in signals)
            for sig in signals:
                frac = abs(sig.score) / total_score if total_score > 0 else 1.0 / len(signals)
                bucket = per_strategy.setdefault(sig.strategy, {})
                bucket[sym] = bucket.get(sym, 0.0) + weight * frac

        # NOTE: per_strategy is an *attribution* of the final ``targets`` and
        # must sum (per symbol) back to them.  Strategy budget caps are a
        # portfolio-construction concern and are intentionally NOT applied here
        # — scaling one bucket in isolation would break that invariant and
        # leave per_strategy inconsistent with the weights actually traded.
        return per_strategy
