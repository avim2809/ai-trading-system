"""Execution agent.

Takes an approved ``RiskDecision``, diffs its adjusted target weights
against current portfolio holdings, and produces an order list plus
turnover and cost estimates in an ``ExecutionReport``.

Optionally (config key ``protective_orders``, off by default — see
``ExecutionAgent.__init__``) also submits broker-side stop/trailing-stop
orders directly, as a side channel outside the fills/turnover/cost
accounting above, when a configured strategy opens or increases a
position. This only fires if a ``broker`` object is passed into ``run()``
via ``inputs`` — today's live orchestrator/engine call sites don't do that
(wiring that through was out of scope for this change, which was
deliberately confined to brokers/base.py, brokers/alpaca.py, brokers/ibkr.py
and this file), so merging this is a no-op for the running system until a
caller supplies both the config and the broker.
"""

from __future__ import annotations

import logging
from typing import Any

from firm.agents._liquidity import estimate_adv_dollars, market_impact_pct
from firm.agents.base import Agent, AgentContext
from firm.brokers.base import BrokerError, OrderRequest
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

        # Full-close floor for a target_w == 0 symbol (2026-09-20). NOT the
        # same exemption already tried and reverted for current_w == 0 above
        # (that regression was about skipping the band on *opening* a new
        # position, which this doesn't touch) -- this is about *closing* an
        # existing one all the way to flat. Left alone, a target-0 position
        # is genuinely subject to asymptotic stranding: rebalance_band_pct
        # tolerates any remainder under the (5% blended / ~0.3-2.5% sleeved)
        # band forever, and even above the band, rebalance_fraction geometric
        # decay (0.7/cycle) only asymptotes toward zero, never reaching it --
        # a real, cited failure mode in professional portfolio construction
        # (Gårleanu-Pedersen "aim in front of target" never fully arrives;
        # the standard fix is a semicontinuous/threshold-holding constraint:
        # a position is either flat or above a floor, never parked
        # indefinitely just under one). Deliberately a SEPARATE, MUCH SMALLER
        # threshold than rebalance_band_pct itself, not a blanket bypass of
        # it: the codebase's own 3-window A/B (see rebalance_band_pct's own
        # comment above) found unconditionally forcing full closes at the
        # *band's* width regressed Sharpe 3.45->0.80 and 8x'd turnover by
        # fighting the band's real job of suppressing noise-level churn. This
        # only forces a full, band/fraction-bypassing close once the
        # remainder is smaller than close_dust_fraction of the *band itself*
        # (default 20% of it) -- small enough that true single-cycle dust
        # (what the reverted experiment was actually about) is still left
        # alone, but a materially-sized position that would otherwise sit
        # just under the band forever now gets swept once it decays that far.
        self.close_dust_fraction: float = float(cfg.get("close_dust_fraction", 0.2))

        # Broker-resident protective stops (off by default). Keyed by
        # strategy name, mirroring the spirit of RiskAgent's
        # ``stop_loss_overlay`` config (also opt-in, also per-strategy) but
        # shaped as a plain dict of per-strategy settings rather than an
        # enabled-flag + strategies-list pair, e.g.:
        #   {"mean_reversion": {"stop_loss_pct": 0.07},
        #    "stat_arb": {"trailing_stop_pct": 0.05}}
        # RiskAgent's overlay is a portfolio-level, cycle-gated backstop that
        # zeroes a held symbol's *target weight* once per scheduled cycle;
        # this is a complementary, faster, broker-resident layer that
        # protects the position between cycles (a broker-side stop can
        # trigger intracycle, long before the next scheduled decision).
        # {} (default) submits no protective orders at all -- byte-identical
        # behavior to before this existed. Only fires when a ``broker`` is
        # also supplied to run() (see _maybe_submit_protective_orders).
        self.protective_orders_cfg: dict[str, dict[str, Any]] = cfg.get("protective_orders", {}) or {}

        # Extended-hours limit-price tolerance. Both IBKR and Alpaca only
        # actually honor an "outside regular trading hours" flag on a LIMIT
        # order -- a market order with the flag set is queued until the next
        # regular session instead of filling now (see
        # IBKRBroker._submit_order_once_unlocked's Warning-2109 log and
        # AlpacaBroker._extended_hours_kwarg). This agent otherwise only
        # emits market orders, so without this substitution
        # config/live.yaml's extended_hours_trading feature would run on
        # schedule but produce no real early-execution benefit. When
        # Orchestrator threads
        # ``extended_hours=True`` into run() below (only for a cycle
        # gate-verified premarket/afterhours -- see Orchestrator.step's
        # cycle_type handling), this switches that cycle's orders to
        # marketable limit orders instead: buy up to
        # ``price * (1 + tolerance)``, sell down to
        # ``price * (1 - tolerance)``. Wide enough to have a real chance of
        # filling against thin extended-hours liquidity (a limit pegged
        # exactly at the last regular-session price would rarely execute at
        # all once the market gaps), narrow enough to still cap slippage --
        # a limit order with no meaningful bound defeats the point of using
        # one. 0.5% default is a starting, not empirically-tuned, buffer;
        # tune per-symbol volatility if live fills show it's mis-sized.
        # 0.0 tolerance would peg the limit at the last price exactly (valid
        # but likely to go unfilled). Regular-hours cycles never read this
        # -- ``extended_hours`` defaults False and this knob has zero effect
        # on today's market-order behavior.
        self.extended_hours_limit_tolerance_pct: float = float(
            cfg.get("extended_hours_limit_tolerance_pct", 0.005)
        )

    def run(self, ctx: AgentContext, **inputs: Any) -> ExecutionReport:
        decision: RiskDecision = inputs["decision"]
        portfolio = inputs.get("portfolio")
        prices: dict[str, float] = inputs.get("prices", {})
        per_strategy: dict[str, dict[str, float]] = inputs.get("per_strategy", {})
        attribution = inputs.get("attribution")
        # Optional broker handle for the protective-stop side channel below.
        # Not wired up by the live engine/orchestrator today (out of scope
        # for this change — see ExecutionAgent module docstring) so passing
        # nothing here is exactly today's behavior; a caller that does supply
        # one only sees protective orders fire if protective_orders_cfg is
        # also non-empty.
        broker = inputs.get("broker")
        # Set by Orchestrator only for a cycle gate-verified to genuinely be
        # running inside a configured premarket/afterhours window (see
        # Orchestrator.step / _step_impl) -- never a blanket config toggle
        # applied regardless of when this cycle actually runs. False (the
        # default, and every existing caller/test that hasn't been updated
        # to pass it) reproduces prior behavior byte-for-byte: plain market
        # orders, no limit_price.
        extended_hours: bool = bool(inputs.get("extended_hours", False))

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
            #
            # An exemption for current_w == 0 (never held) was tried and
            # reverted (PART 3 of the remediation plan): reasonable in
            # theory (a brand-new position's target is signal, not drift),
            # but a 3-window A/B against the already-shipped, live-validated
            # band showed a real regression -- turnover roughly 8x higher
            # and Sharpe dropped from 3.45 to 0.80 on the 2024-Q1 window
            # alone. In practice, a large share of what the band suppresses
            # *is* small new-position churn as names rotate in/out of the
            # top-N conviction ranking every cycle, not just drift on
            # existing positions -- the "obviously correct" fix broke the
            # band's actual job. Left as originally shipped; the known
            # side effect (a multiplicative RiskAgent overlay stacking with
            # regime_overlay can shrink every target below the band and
            # lock the book at 0% invested) is a real, understood
            # limitation of composing overlays, not fixed here.
            # Full-close floor (see close_dust_fraction's field docstring):
            # a target of exactly 0 with a remaining position bigger than
            # this much smaller threshold closes in full, bypassing BOTH the
            # band above and rebalance_fraction below -- otherwise it falls
            # through to the same band/fraction handling as every other
            # case, including staying genuinely-dust-tolerant below this
            # smaller floor.
            forced_full_close = False
            if target_w == 0.0 and current_w != 0.0:
                close_floor = self.rebalance_band_pct * self.close_dust_fraction
                if abs(current_w) >= close_floor:
                    forced_full_close = True

            if not forced_full_close:
                if self.rebalance_band_pct > 0 and abs(diff_w) < self.rebalance_band_pct:
                    continue

            price = prices.get(sym, 0.0)
            if price <= 0:
                log.warning("No price for %s – skipping order", sym)
                continue

            # Turnover-aware sizing: close only a fraction of the (already
            # above-band) gap this cycle. 1.0 (default) is a no-op --
            # byte-identical behavior to before this knob existed. Skipped
            # entirely for a forced full close -- the whole point is to
            # reach flat in one shot, not decay toward it.
            if self.rebalance_fraction < 1.0 and not forced_full_close:
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

            order: dict[str, Any] = {
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
            if extended_hours:
                # See extended_hours_limit_tolerance_pct's __init__ docstring
                # -- both live brokers only honor outside-RTH on a limit
                # order, so this is the one substitution that makes
                # config/live.yaml's extended_hours_trading feature do
                # anything real. LiveTradingEngine._execute_orders reads
                # these two keys straight off the fill dict (falling back to
                # "market"/None for every order that doesn't set them), so
                # no other plumbing needs to change.
                order["order_type"] = "limit"
                order["limit_price"] = self._extended_hours_limit_price(price, side)
                log.info(
                    "Extended-hours cycle: %s %s as LIMIT @ %.4f (last price %.4f, "
                    "tolerance %.3f%%)",
                    side, sym, order["limit_price"], price,
                    self.extended_hours_limit_tolerance_pct * 100.0,
                )

            orders.append(order)
            turnover += abs(diff_w)

            if self.protective_orders_cfg and broker is not None:
                self._maybe_submit_protective_order(
                    broker=broker,
                    strategy=symbol_strategy.get(sym, "composite"),
                    symbol=sym,
                    current_w=current_w,
                    target_w=target_w,
                    price=price,
                    nav=nav,
                )

        # Aggregate cost is the sum of the per-order estimates (identical to the
        # previous total_notional * (commission + slippage) formulation).
        costs = sum(o["est_cost"] for o in orders)

        return ExecutionReport(fills=orders, turnover=turnover, costs=costs)

    # ------------------------------------------------------------------
    # Broker-side protective stops (opt-in)
    # ------------------------------------------------------------------

    @staticmethod
    def _is_fresh_open(current_w: float, target_w: float) -> bool:
        """True only when *symbol* is moving from genuinely flat to a new
        position -- the sole case this submits a protective order for.

        Deliberately narrower than "opens or increases": this class has no
        broker order-ID tracking, so it cannot cancel a previously-submitted
        protective stop before placing another one. Firing again on every
        cycle that *increases* an already-open position (the original
        design) would leave a separate resting stop order at the broker for
        each cycle's incremental add, stacked on top of the still-live
        stop(s) from earlier cycles -- e.g. three 07-cycle increases into the
        same long leaves three resting sell-stops; if the price ever trades
        through the stop level, all three fire together and the position
        flips net short by roughly the sum of the earlier tranches, not just
        flat. Restricting to flat -> open avoids that entirely: once a
        symbol has a protective order, this returns False for it on every
        later cycle until the position is fully closed (target_w == 0,
        which zeroes current_w for the next cycle) and genuinely reopened.
        The known cost: neither a same-direction increase nor a same-cycle
        sign flip gets a *new* stop sized to the larger/flipped position --
        only the original tranche is covered by the first stop. Sizing or
        replacing the resting stop as a position changes would need this
        agent to track and cancel its own previously-submitted broker order
        IDs, which is out of scope here.
        """
        return current_w == 0 and target_w != 0

    def _maybe_submit_protective_order(
        self,
        *,
        broker: Any,
        strategy: str,
        symbol: str,
        current_w: float,
        target_w: float,
        price: float,
        nav: float,
    ) -> None:
        """Submit a broker-side stop/trailing-stop covering *symbol*'s new
        position, if *strategy* opts in via ``protective_orders``.

        No-op (no config for this strategy, not a fresh open, or a bad
        price/nav) means nothing is submitted -- default behavior is
        unchanged. A submission failure is logged and swallowed rather than
        raised: a protective order is a best-effort safety net on top of the
        primary order, not a condition the primary rebalance should fail on.
        """
        cfg = self.protective_orders_cfg.get(strategy)
        if not cfg:
            return
        if not self._is_fresh_open(current_w, target_w):
            return
        if price <= 0 or nav <= 0:
            return

        # current_w == 0 here (see _is_fresh_open), so target_w is the whole
        # new position, not just an incremental slice.
        protective_qty = int(round(abs(target_w) * nav / price))
        if protective_qty <= 0:
            return

        is_long = target_w > 0
        protective_side = "sell" if is_long else "buy"

        trailing_stop_pct = cfg.get("trailing_stop_pct")
        stop_loss_pct = cfg.get("stop_loss_pct")

        order_kwargs: dict[str, Any] = dict(
            symbol=symbol,
            side=protective_side,
            quantity=protective_qty,
            strategy=strategy,
            client_order_id=f"protective-{strategy}-{symbol}-{protective_side}",
        )
        if trailing_stop_pct is not None:
            order_kwargs["order_type"] = "trailing_stop"
            order_kwargs["trail_percent"] = float(trailing_stop_pct) * 100.0
        elif stop_loss_pct is not None:
            order_kwargs["order_type"] = "stop"
            raw_stop_price = (
                price * (1 - float(stop_loss_pct)) if is_long else price * (1 + float(stop_loss_pct))
            )
            # Brokers reject sub-penny prices for stocks trading above $1
            # (Alpaca: "sub-penny increment does not fulfill minimum pricing
            # criteria") -- confirmed live, every protective stop submitted
            # since this feature was enabled failed 100% of the time because
            # price * (1 +/- pct) is an arbitrary float. Sub-$1 names allow
            # sub-penny (tick = $0.0001); round accordingly.
            decimals = 2 if raw_stop_price >= 1.0 else 4
            order_kwargs["stop_price"] = round(raw_stop_price, decimals)
        else:
            log.debug(
                "protective_orders configured for %s but neither stop_loss_pct "
                "nor trailing_stop_pct is set — skipping %s", strategy, symbol,
            )
            return

        req = OrderRequest(**order_kwargs)
        try:
            status = broker.submit_order(req)
            log.info(
                "Protective %s order submitted for %s (%s, qty=%d) -> %s",
                req.order_type, symbol, strategy, protective_qty, status.status,
            )
        except BrokerError:
            log.warning(
                "Protective %s order submission failed for %s (%s, qty=%d)",
                req.order_type, symbol, strategy, protective_qty, exc_info=True,
            )

    def _extended_hours_limit_price(self, price: float, side: str) -> float:
        """Marketable limit price for an extended-hours order: buy up to
        ``price * (1 + tolerance)``, sell down to ``price * (1 - tolerance)``.

        See ``extended_hours_limit_tolerance_pct``'s ``__init__`` docstring
        for the "not too tight, not too loose" rationale. Rounded to cents
        (or 4dp under $1, same threshold IBKR/Alpaca enforce for stocks --
        see the sub-penny rounding precedent in
        ``_maybe_submit_protective_order``) since brokers reject sub-penny
        limit prices above $1.
        """
        tolerance = self.extended_hours_limit_tolerance_pct
        raw = price * (1 + tolerance) if side == "buy" else price * (1 - tolerance)
        decimals = 2 if raw >= 1.0 else 4
        return round(raw, decimals)

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
