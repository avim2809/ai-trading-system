"""Execution agent.

Takes an approved ``RiskDecision``, diffs its adjusted target weights
against current portfolio holdings, and produces an order list plus
turnover and cost estimates in an ``ExecutionReport``.
"""

from __future__ import annotations

import logging
from typing import Any

from firm.agents._liquidity import estimate_adv_dollars, market_impact_pct
from firm.agents.base import Agent, AgentContext
from firm.contracts.models import ExecutionReport, RiskDecision

log = logging.getLogger(__name__)


class ExecutionAgent(Agent):
    """Translates approved target weights into executable orders."""

    role = "execution"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="execution", config=config)
        cfg = config or {}
        self.commission_pct: float = cfg.get("commission_pct", 0.001)
        self.slippage_pct: float = cfg.get("slippage_pct", 0.0005)
        # Bid-ask spread cost — approximates the cost of crossing the quoted
        # spread, on top of commission/slippage. Short-borrow fees are NOT
        # estimated here: IBKR charges/reports real borrow costs against the
        # live account directly (see config/live.yaml costs: block).
        self.spread_pct: float = cfg.get("spread_pct", 0.0)
        # Size/volume-aware market-impact term, on top of the flat commission
        # + slippage + spread rates above. Those are flat percentages of
        # notional regardless of order size relative to the name's trading
        # volume — realistic for a small trade, but understate cost for a
        # large one and overstate it for a tiny one. This adds a
        # participation-rate-scaled term (see firm.agents._liquidity.
        # sqrt_impact_pct) using the same ADV lookback/definition as
        # RiskAgent's liquidity cap (``adv_lookback_days``, shared top-level
        # config key). 0.0 (Python-level default) disables it entirely;
        # config/live.yaml and config/settings.yaml opt in with a
        # conservative calibration.
        self.market_impact_coefficient: float = cfg.get("market_impact_coefficient", 0.0)
        # Optional linear-below/sqrt-above crossover (None = pure sqrt law,
        # unchanged default). See firm.agents._liquidity.market_impact_pct.
        self.market_impact_crossover: float | None = cfg.get(
            "market_impact_crossover_participation"
        )
        self.adv_lookback_days: int = int(cfg.get("adv_lookback_days", 20))
        # No-trade / rebalance band: |target_w - current_w| must exceed this
        # fraction of NAV before an order is generated at all. 0.0 (default)
        # preserves prior behavior exactly -- every existing test and any
        # caller that hasn't opted in sees identical output. Confirmed live:
        # with no band at all, the z-scored/L1-normalized construction
        # pipeline re-derives a fresh target for every name each cycle, so
        # even noise-level drift (a few basis points) generated a real order
        # -- one contributor to the 60-95%/day turnover this system's own
        # docs/live.yaml comments already diagnosed. This is expected to be
        # the highest-leverage single turnover fix pending a live-faithful
        # backtest confirmation: it doesn't change *what* the strategies/
        # risk stack decides, only whether a decision small enough to be
        # noise gets acted on.
        self.rebalance_band_pct: float = float(cfg.get("rebalance_band_pct", 0.0))
        # Turnover-aware sizing: trade only this fraction of the gap to
        # target each cycle (1.0 = full rebalance, unchanged prior
        # behavior). Complementary to rebalance_band_pct above -- the band
        # decides whether a deviation is worth trading AT ALL, this decides
        # how much of a real (above-band) deviation to close in one cycle.
        # Since TraderAgent re-derives target weights fresh every cycle from
        # the day's z-scored/L1-normalized conviction, partial rebalancing
        # lets the position drift toward target over several cycles instead
        # of snapping fully each time signal noise moves the target —
        # directly reduces the per-cycle trade size on every above-band
        # deviation, not just the sub-band ones the rebalance band already
        # filters out.
        self.rebalance_fraction: float = float(cfg.get("rebalance_fraction", 1.0))

    def run(self, ctx: AgentContext, **inputs: Any) -> ExecutionReport:
        decision: RiskDecision = inputs["decision"]
        portfolio = inputs.get("portfolio")
        prices: dict[str, float] = inputs.get("prices", {})
        per_strategy: dict[str, dict[str, float]] = inputs.get("per_strategy", {})
        attribution = inputs.get("attribution")

        target_weights = decision.adjusted_targets
        symbol_strategy = self._dominant_strategy_by_symbol(per_strategy)
        # This cycle's per-strategy attribution only covers symbols present in
        # the new target weights — a symbol being closed out entirely (held
        # today, absent from `targets`) has no entry here even though it was
        # opened by a specific strategy. Fall back to whichever strategy
        # currently holds the largest position in that symbol so exit trades
        # stay traceable instead of collapsing into "composite".
        if attribution is not None:
            held_strategy = attribution.dominant_strategy_by_symbol()
            missing = 0
            for sym in held_strategy:
                if sym not in symbol_strategy:
                    symbol_strategy[sym] = held_strategy[sym]
                    missing += 1
            if missing:
                log.debug(
                    "Attributed %d closing/held symbol(s) to strategy via "
                    "held-position fallback", missing,
                )

        if portfolio is not None and prices:
            current_weights = portfolio.get_weights(prices)
            nav = portfolio.cash + sum(
                shares * prices.get(sym, 0.0)
                for sym, shares in portfolio.holdings.items()
            )
        else:
            current_weights = {}
            nav = ctx.config.get("initial_capital", 10_000_000)

        orders: list[dict[str, Any]] = []
        turnover = 0.0

        all_symbols = sorted(set(target_weights) | set(current_weights))
        for sym in all_symbols:
            target_w = target_weights.get(sym, 0.0)
            current_w = current_weights.get(sym, 0.0)
            diff_w = target_w - current_w

            if abs(diff_w) < 1e-6:
                continue

            # No-trade band: a deviation smaller than this is treated as
            # noise, not a real rebalance signal -- skip it entirely (no
            # order, no cost estimate, no turnover contribution). A target
            # of exactly 0.0 with a small leftover current_w is deliberately
            # tolerated here too (not force-closed), matching standard
            # drift-band rebalancing practice: liquidating dust costs more
            # in commission/spread than leaving it is worth.
            if self.rebalance_band_pct > 0 and abs(diff_w) < self.rebalance_band_pct:
                continue

            price = prices.get(sym, 0.0)
            if price <= 0:
                log.warning("No price for %s – skipping order", sym)
                continue

            # Turnover-aware sizing: close only a fraction of the (already
            # above-band) gap this cycle. 1.0 (default) is a no-op --
            # byte-identical behavior to before this knob existed.
            if self.rebalance_fraction < 1.0:
                diff_w *= self.rebalance_fraction

            dollar_amount = diff_w * nav
            quantity = abs(dollar_amount / price)
            side = "buy" if dollar_amount > 0 else "sell"

            if sym not in symbol_strategy:
                log.warning(
                    "No strategy attribution for %s order in %s (no current-cycle "
                    "signal and no held position) — falling back to 'composite'; "
                    "P&L for this trade will not be traceable to a strategy",
                    side, sym,
                )

            # Pre-trade transaction-cost estimate per order, using the same
            # commission + slippage model as the backtest. Surfacing it per
            # order (not just in aggregate) lets the live approval/routing
            # layer weigh expected cost before sending each order.
            notional = abs(dollar_amount)
            est_commission = notional * self.commission_pct
            est_slippage = notional * self.slippage_pct
            est_spread = notional * self.spread_pct
            est_impact = self._estimate_impact_cost(ctx, sym, notional)

            orders.append(
                {
                    "symbol": sym,
                    "side": side,
                    # Signed share count: the canonical field consumed by
                    # PortfolioState.update and PerformanceAttribution.
                    # ``quantity`` stays absolute for broker order requests.
                    "shares": quantity if side == "buy" else -quantity,
                    "quantity": quantity,
                    "notional": notional,
                    "price": price,
                    "strategy": symbol_strategy.get(sym, "composite"),
                    "est_commission": est_commission,
                    "est_slippage": est_slippage,
                    "est_spread": est_spread,
                    "est_impact": est_impact,
                    "est_cost": est_commission + est_slippage + est_spread + est_impact,
                }
            )
            turnover += abs(diff_w)

        # Aggregate cost is the sum of the per-order estimates (identical to the
        # previous total_notional * (commission + slippage) formulation).
        costs = sum(o["est_cost"] for o in orders)

        return ExecutionReport(fills=orders, turnover=turnover, costs=costs)

    def _estimate_impact_cost(self, ctx: AgentContext, symbol: str, notional: float) -> float:
        """Size/volume-aware market-impact cost estimate for one order.

        Returns ``0.0`` (no-op, matching pre-existing flat-pct-only
        behaviour) whenever the model is disabled (``market_impact_coefficient
        <= 0``), ``ctx.pit_view`` isn't wired up, or ADV data isn't
        available — never raises, since a missing/thin data provider must
        degrade the cost estimate, not block order generation.
        """
        if self.market_impact_coefficient <= 0 or notional <= 0:
            return 0.0
        pit_view = getattr(ctx, "pit_view", None)
        if pit_view is None:
            return 0.0
        adv_dollars = estimate_adv_dollars(pit_view, symbol, self.adv_lookback_days)
        if not adv_dollars:
            return 0.0
        participation = notional / adv_dollars
        impact_pct = market_impact_pct(
            participation, self.market_impact_coefficient, self.market_impact_crossover
        )
        return notional * impact_pct

    @staticmethod
    def _dominant_strategy_by_symbol(
        per_strategy: dict[str, dict[str, float]],
    ) -> dict[str, str]:
        """Map each symbol to the strategy that contributed the most weight.

        Used to tag each order with a real originating strategy so the live
        engine's per-strategy approval routing is meaningful.  Symbols absent
        from the attribution fall back to ``"composite"`` at the call site.
        """
        best: dict[str, tuple[float, str]] = {}
        for strat, weights in (per_strategy or {}).items():
            for sym, weight in weights.items():
                contribution = abs(weight)
                if sym not in best or contribution > best[sym][0]:
                    best[sym] = (contribution, strat)
        return {sym: strat for sym, (_, strat) in best.items()}
