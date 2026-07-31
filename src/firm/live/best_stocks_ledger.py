"""Synthetic mark-to-market ledger for the Danelfin Best-Stocks paper arm.

Deliberately NOT a broker-connected engine (see best_stocks_arm.py's module
docstring for why) — just a JSON-persisted equal-weight NAV tracker: hold
25 symbols at target equal dollar weight, mark to market daily, and apply
Danelfin's own stated rebalance cadence (quarterly replace / annual
reweight) on schedule.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

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
