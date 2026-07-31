"""NAV ledger for the Danelfin Best-Stocks paper arm — synthetic or
broker-executed.

A JSON-persisted equal-weight NAV tracker: hold 25 symbols at target equal
dollar weight, mark to market daily, and apply Danelfin's own stated
rebalance cadence (quarterly replace / annual reweight) on schedule.

Two execution modes, both maintained by this same class:
  - ``full_rebalance``/``quarterly_replace``/``annual_rebalance`` — the
    original synthetic mode: hypothetical fractional shares, no broker
    involved at all (see best_stocks_arm.py's module docstring for why
    this was the original, more conservative design).
  - ``rebalance_via_broker`` — real IBKR paper orders, whole shares, real
    fills. Added after the user explicitly asked to hook this arm into
    real trade execution, sharing the main engine's own IBKR paper
    account (a real collision risk — see
    firm.live.best_stocks_execution's module docstring for the guard
    this depends on). Uses a SEPARATE state file
    (``data/best_stocks_ledger_live.json`` by convention) from the
    synthetic ledger, so the two remain independently comparable rather
    than one silently overwriting the other's history.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from firm.brokers.ibkr import IBKRBroker

log = logging.getLogger("firm.live.best_stocks_ledger")

QUARTERLY_REPLACE_DAYS = 91
ANNUAL_REBALANCE_DAYS = 365


@dataclass
class BestStocksLedger:
    initial_capital: float = 100_000.0
    cash: float = 0.0
    holdings: dict[str, float] = field(default_factory=dict)  # symbol -> shares
    selection_meta: list[dict] = field(default_factory=list)  # last selection snapshot
    nav_history: list[dict] = field(default_factory=list)  # [{date, nav}]
    last_full_rebalance: str | None = None  # ISO date
    last_quarterly_replace: str | None = None  # ISO date

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "BestStocksLedger":
        p = Path(path)
        if not p.exists():
            return cls()
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # valuation
    # ------------------------------------------------------------------

    def nav(self, prices: dict[str, float]) -> float:
        holdings_value = sum(
            shares * prices[sym] for sym, shares in self.holdings.items() if sym in prices
        )
        missing = [sym for sym in self.holdings if sym not in prices]
        if missing:
            log.warning("best_stocks_missing_prices symbols=%s (excluded from NAV)", missing)
        return self.cash + holdings_value

    def mark_to_market(self, asof: datetime, prices: dict[str, float]) -> float:
        current_nav = self.nav(prices)
        self.nav_history.append({"date": asof.isoformat(), "nav": current_nav})
        return current_nav

    # ------------------------------------------------------------------
    # rebalancing
    # ------------------------------------------------------------------

    def full_rebalance(
        self, asof: datetime, target_selection: list[dict], prices: dict[str, float]
    ) -> None:
        """Liquidate everything and redistribute the current NAV equally
        across ``target_selection``'s symbols (fractional shares — this is
        a synthetic ledger, not a real broker order book)."""
        current_nav = self.nav(prices) if self.holdings or self.cash else self.initial_capital
        symbols = [row["symbol"] for row in target_selection]
        priced_symbols = [s for s in symbols if s in prices and prices[s] > 0]
        missing = set(symbols) - set(priced_symbols)
        if missing:
            log.warning("best_stocks_rebalance_missing_prices symbols=%s (skipped)", sorted(missing))
        if not priced_symbols:
            log.warning("best_stocks_rebalance_no_priced_symbols — leaving ledger unchanged")
            return
        per_symbol_dollars = current_nav / len(priced_symbols)
        self.holdings = {sym: per_symbol_dollars / prices[sym] for sym in priced_symbols}
        self.cash = current_nav - sum(
            shares * prices[sym] for sym, shares in self.holdings.items()
        )
        self.selection_meta = [row for row in target_selection if row["symbol"] in priced_symbols]
        self.last_full_rebalance = asof.date().isoformat()
        # A full rebalance also resets the quarterly-replace clock — both
        # cadences start counting from the same event (the initial
        # portfolio construction, or a later annual rebalance).
        self.last_quarterly_replace = asof.date().isoformat()
        log.info(
            "best_stocks_full_rebalance asof=%s nav=%.2f n_holdings=%d",
            asof.date(), current_nav, len(self.holdings),
        )

    def quarterly_replace(
        self, asof: datetime, fresh_selection: list[dict], prices: dict[str, float]
    ) -> None:
        """Replace holdings that no longer appear in a freshly re-run
        selection (i.e. no longer meet the buy/low-risk/volume/top-sector
        criteria); keep still-qualifying holdings at their current share
        count, redistribute freed-up dollars equally across the newly
        added names. This is a defensible-but-not-uniquely-specified
        reading of Danelfin's stated "replace stocks that no longer meet
        the criteria" rule — no single symbol-level diff was published."""
        current_nav = self.nav(prices)
        fresh_symbols = {row["symbol"] for row in fresh_selection}
        held_symbols = set(self.holdings)
        dropped = held_symbols - fresh_symbols
        added_candidates = [s for s in fresh_symbols - held_symbols if s in prices and prices[s] > 0]

        if not dropped and not added_candidates:
            log.info("best_stocks_quarterly_replace asof=%s: no changes needed", asof.date())
            self.last_quarterly_replace = asof.date().isoformat()
            return

        freed_dollars = sum(
            self.holdings[sym] * prices[sym] for sym in dropped if sym in prices
        )
        for sym in dropped:
            del self.holdings[sym]

        n_new = min(len(added_candidates), len(dropped)) or len(added_candidates)
        new_symbols = added_candidates[:n_new] if n_new else []
        if new_symbols:
            per_symbol_dollars = freed_dollars / len(new_symbols)
            for sym in new_symbols:
                self.holdings[sym] = per_symbol_dollars / prices[sym]
        else:
            self.cash += freed_dollars

        self.selection_meta = [row for row in fresh_selection if row["symbol"] in self.holdings]
        self.last_quarterly_replace = asof.date().isoformat()
        log.info(
            "best_stocks_quarterly_replace asof=%s dropped=%s added=%s n_holdings=%d",
            asof.date(), sorted(dropped), sorted(new_symbols), len(self.holdings),
        )
        _ = current_nav  # kept for logging parity / future use; nav unaffected by a swap

    def annual_rebalance(self, asof: datetime, prices: dict[str, float]) -> None:
        """Reset current holdings back to equal dollar weighting WITHOUT
        changing which symbols are held (that's quarterly_replace's job)."""
        if not self.holdings:
            return
        current_nav = self.nav(prices)
        priced_symbols = [s for s in self.holdings if s in prices and prices[s] > 0]
        if not priced_symbols:
            log.warning("best_stocks_annual_rebalance_no_priced_symbols — leaving ledger unchanged")
            return
        per_symbol_dollars = current_nav / len(priced_symbols)
        self.holdings = {sym: per_symbol_dollars / prices[sym] for sym in priced_symbols}
        self.cash = current_nav - sum(
            shares * prices[sym] for sym, shares in self.holdings.items()
        )
        self.last_full_rebalance = asof.date().isoformat()
        log.info("best_stocks_annual_rebalance asof=%s nav=%.2f", asof.date(), current_nav)

    # ------------------------------------------------------------------
    # real broker execution (whole shares, real fills)
    # ------------------------------------------------------------------

    def rebalance_via_broker(
        self,
        asof: datetime,
        broker: "IBKRBroker",
        rebalance_kind: str,
        target_selection: list[dict] | None = None,
    ) -> None:
        """Place real IBKR paper orders (whole shares) to bring holdings
        toward a target allocation, instead of the synthetic
        full_rebalance/quarterly_replace/annual_rebalance's hypothetical
        fractional-share math. See this module's docstring and
        firm.live.best_stocks_execution's module docstring for the
        shared-account collision-guard rationale — re-checked HERE, fresh,
        even though callers are also expected to exclude colliding symbols
        at selection time (defense in depth: the main engine's universe
        can change between selection and execution).

        rebalance_kind:
          "full"      — initial construction; target_selection required.
                        Every symbol not in the fresh selection is sold to 0.
          "quarterly" — re-run selection; only symbols that dropped out are
                        sold, only newly-added symbols are bought (using the
                        dollars freed by the sells) — still-qualifying
                        holdings are left untouched, matching Danelfin's
                        "replace what no longer qualifies" wording more
                        literally than a full reweight would.
                        target_selection required (freshly re-run).
          "annual"    — re-equal-weight CURRENT holdings only, no symbol
                        changes; target_selection ignored.
        """
        from firm.brokers.base import BrokerError, OrderRequest
        from firm.live.best_stocks_execution import main_engine_excluded_symbols

        excluded = main_engine_excluded_symbols()
        held = set(self.holdings)

        if rebalance_kind in ("full", "quarterly"):
            if not target_selection:
                log.warning(
                    "best_stocks_rebalance_via_broker_no_selection kind=%s — skipping", rebalance_kind,
                )
                return
            tradable = [row for row in target_selection if row["symbol"] not in excluded]
            skipped = [row["symbol"] for row in target_selection if row["symbol"] in excluded]
            if skipped:
                log.warning(
                    "best_stocks_collision_guard_skipped kind=%s symbols=%s (main engine universe)",
                    rebalance_kind, sorted(skipped),
                )
            fresh_symbols = {row["symbol"] for row in tradable}
            held_collisions: set[str] = set()
        elif rebalance_kind == "annual":
            held_collisions = held & excluded
            if held_collisions:
                log.error(
                    "best_stocks_collision_guard_triggered symbols=%s currently held but now "
                    "also in the main engine's universe — left untouched (not sold) this rebalance",
                    sorted(held_collisions),
                )
            fresh_symbols = held - held_collisions
        else:
            raise ValueError(f"unknown rebalance_kind: {rebalance_kind!r}")

        if not fresh_symbols and not held_collisions:
            log.warning(
                "best_stocks_rebalance_via_broker_no_tradable_symbols kind=%s — leaving ledger unchanged",
                rebalance_kind,
            )
            return

        all_symbols = fresh_symbols | held | held_collisions
        try:
            prices = broker.get_current_prices(list(all_symbols))
        except BrokerError:
            log.error("best_stocks_price_fetch_failed — aborting rebalance", exc_info=True)
            return

        current_nav = self.nav(prices)
        # Untouched positions (collision-guard holdouts) always keep their
        # current share count — never defaulted to 0, which would instead
        # generate an unwanted full-liquidation sell order for them.
        target_shares: dict[str, int] = {s: int(round(self.holdings[s])) for s in held_collisions}

        if rebalance_kind == "quarterly":
            keep = held & fresh_symbols
            dropped = held - fresh_symbols - held_collisions
            added = fresh_symbols - held
            for s in keep:
                target_shares[s] = int(round(self.holdings[s]))  # unchanged — no order generated
            for s in dropped:
                target_shares[s] = 0
            priced_added = [s for s in added if prices.get(s, 0) > 0]
            if priced_added:
                freed_dollars = sum(self.holdings.get(s, 0.0) * prices.get(s, 0.0) for s in dropped)
                per_symbol_dollars = freed_dollars / len(priced_added)
                for s in priced_added:
                    target_shares[s] = int(per_symbol_dollars // prices[s])
        else:  # full or annual — equal-weight across the whole (guard-adjusted) target set
            priced_targets = [s for s in fresh_symbols if prices.get(s, 0) > 0]
            if not priced_targets:
                log.warning("best_stocks_rebalance_via_broker_no_priced_targets — leaving ledger unchanged")
                return
            # Collision-guard holdouts (annual only) keep their current value
            # untouched and are excluded from the reweight pool entirely, so
            # the remaining NAV split across priced_targets doesn't silently
            # double-count their (unresized) value.
            held_collisions_value = sum(
                self.holdings.get(s, 0.0) * prices.get(s, 0.0) for s in held_collisions
            )
            per_symbol_dollars = (current_nav - held_collisions_value) / len(priced_targets)
            for s in priced_targets:
                target_shares[s] = int(per_symbol_dollars // prices[s])

        for symbol in all_symbols:
            target = target_shares.get(symbol, 0)
            current = int(round(self.holdings.get(symbol, 0.0)))
            delta = target - current
            if delta == 0:
                continue
            side = "buy" if delta > 0 else "sell"
            order = OrderRequest(
                symbol=symbol, side=side, quantity=abs(delta), order_type="market",
                strategy="danelfin_best_stocks",
                client_order_id=f"beststocks-{symbol}-{asof.strftime('%Y%m%d')}-{side}",
            )
            try:
                status = broker.submit_order(order)
            except BrokerError:
                log.error(
                    "best_stocks_order_failed symbol=%s side=%s qty=%d", symbol, side, abs(delta), exc_info=True,
                )
                continue
            filled = status.filled_quantity or 0.0
            if filled <= 0:
                log.warning(
                    "best_stocks_order_unfilled symbol=%s side=%s status=%s", symbol, side, status.status,
                )
                continue
            signed = filled if side == "buy" else -filled
            new_qty = self.holdings.get(symbol, 0.0) + signed
            if abs(new_qty) < 1e-9:
                self.holdings.pop(symbol, None)
            else:
                self.holdings[symbol] = new_qty
            fill_price = status.avg_fill_price or prices.get(symbol, 0.0)
            self.cash -= signed * fill_price
            log.info(
                "best_stocks_order_filled symbol=%s side=%s qty=%.0f price=%.2f",
                symbol, side, filled, fill_price,
            )

        self.selection_meta = [
            row for row in (target_selection or []) if row["symbol"] in self.holdings
        ] or self.selection_meta

        if rebalance_kind == "full":
            self.last_full_rebalance = asof.date().isoformat()
            self.last_quarterly_replace = asof.date().isoformat()
        elif rebalance_kind == "quarterly":
            self.last_quarterly_replace = asof.date().isoformat()
        elif rebalance_kind == "annual":
            self.last_full_rebalance = asof.date().isoformat()

    # ------------------------------------------------------------------
    # scheduling
    # ------------------------------------------------------------------

    def due_for_quarterly_replace(self, asof: datetime) -> bool:
        if self.last_quarterly_replace is None:
            return False  # first-ever rebalance is a full_rebalance, not a replace
        elapsed = (asof.date() - datetime.fromisoformat(self.last_quarterly_replace).date()).days
        return elapsed >= QUARTERLY_REPLACE_DAYS

    def due_for_annual_rebalance(self, asof: datetime) -> bool:
        if self.last_full_rebalance is None:
            return False
        elapsed = (asof.date() - datetime.fromisoformat(self.last_full_rebalance).date()).days
        return elapsed >= ANNUAL_REBALANCE_DAYS
