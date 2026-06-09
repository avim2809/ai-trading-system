"""Live portfolio state: holdings, cash, NAV, and per-strategy sub-ledgers.

The backtest engine mutates a single :class:`PortfolioState` instance each
bar; strategies and agents only observe it through read-only snapshots.
"""

from __future__ import annotations

from datetime import datetime

from firm.contracts.models import PortfolioSnapshot


class PortfolioState:
    """Tracks portfolio holdings, cash, NAV, and per-strategy sub-ledgers."""

    def __init__(self, initial_capital: float = 10_000_000):
        self.cash: float = initial_capital
        self.holdings: dict[str, float] = {}  # symbol -> shares
        self._strategy_ledger: dict[str, dict[str, float]] = {}  # strategy -> {symbol: pnl}
        self._history: list[PortfolioSnapshot] = []

    @property
    def nav(self) -> float:
        """Net asset value (cash + mark-to-market holdings).

        Requires :meth:`get_weights` or :meth:`update` to have been called
        with current prices so holdings are meaningful.  Without prices the
        best estimate is cash alone.
        """
        return self.cash + sum(self.holdings.values())

    def get_weights(self, prices: dict[str, float]) -> dict[str, float]:
        """Return symbol -> weight (market-value / NAV)."""
        total = self.cash + sum(
            shares * prices.get(sym, 0.0) for sym, shares in self.holdings.items()
        )
        if total == 0:
            return {}
        return {
            sym: (shares * prices.get(sym, 0.0)) / total
            for sym, shares in self.holdings.items()
        }

    def update(self, fills: list[dict], prices: dict[str, float]) -> None:
        """Apply a list of fills and reprice the book.

        Each fill dict should contain ``{"symbol": str, "shares": float,
        "price": float, "strategy": str}``.
        """
        for fill in fills:
            sym = fill["symbol"]
            shares = fill["shares"]
            price = fill["price"]
            strategy = fill.get("strategy", "_default")

            self.holdings[sym] = self.holdings.get(sym, 0.0) + shares
            self.cash -= shares * price

            if strategy not in self._strategy_ledger:
                self._strategy_ledger[strategy] = {}
            ledger = self._strategy_ledger[strategy]
            ledger[sym] = ledger.get(sym, 0.0) - shares * price

        self.holdings = {s: q for s, q in self.holdings.items() if q != 0}

    def record_snapshot(
        self,
        asof: datetime,
        prices: dict[str, float],
    ) -> PortfolioSnapshot:
        """Create and store an immutable snapshot of the current state."""
        weights = self.get_weights(prices)
        total_nav = self.cash + sum(
            shares * prices.get(sym, 0.0) for sym, shares in self.holdings.items()
        )
        snap = PortfolioSnapshot(
            asof=asof,
            holdings=dict(self.holdings),
            weights=weights,
            cash=self.cash,
            nav=total_nav,
            per_strategy_pnl={
                strat: sum(pnls.values())
                for strat, pnls in self._strategy_ledger.items()
            },
        )
        self._history.append(snap)
        return snap

    def get_strategy_pnl(self, strategy: str) -> float:
        """Cumulative PnL attributed to a strategy."""
        return sum(self._strategy_ledger.get(strategy, {}).values())

    @property
    def history(self) -> list[PortfolioSnapshot]:
        return list(self._history)
