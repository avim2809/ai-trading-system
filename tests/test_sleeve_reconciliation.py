"""Tests for sleeve_reconciliation.apply_realized_fill.

See src/firm/live/sleeve_reconciliation.py's module docstring for the
design: apportion a symbol's real broker fill pro-rata across whichever
sleeves contributed a decided quantity for it this cycle, correcting each
sleeve's earlier hypothetical (decision-time) credit.
"""

from __future__ import annotations

from firm.live.sleeve_reconciliation import apply_realized_fill
from firm.portfolio.state import PortfolioState


def _sleeve(initial_capital: float, holdings: dict[str, float]) -> PortfolioState:
    p = PortfolioState(initial_capital=initial_capital)
    p.holdings = dict(holdings)
    return p


def _decisions(symbol: str, per_strategy_shares: dict[str, float]) -> dict:
    return {
        strategy: {"status": "approved", "fills": [{"symbol": symbol, "shares": shares, "price": 100.0}]}
        for strategy, shares in per_strategy_shares.items()
    }


class TestSingleSleeveDegenerateCase:
    """One sleeve holding a symbol: pro-rata scaling must be exact -- this
    is provably safe even under the general multi-sleeve ambiguity, since
    it degenerates to blended mode's own self-heal for that one sleeve."""

    def test_full_fill_applies_no_correction(self):
        a = _sleeve(100_000, {"AAPL": 10.0})
        decisions = _decisions("AAPL", {"momentum": 10.0})
        applied = apply_realized_fill({"momentum": a}, decisions, "AAPL", real_filled_qty=10.0, avg_fill_price=100.0)
        assert applied == {}
        assert a.holdings["AAPL"] == 10.0

    def test_full_cancellation_reverses_entire_hypothetical_credit(self):
        a = _sleeve(100_000, {"AAPL": 10.0})
        decisions = _decisions("AAPL", {"momentum": 10.0})
        applied = apply_realized_fill({"momentum": a}, decisions, "AAPL", real_filled_qty=0.0, avg_fill_price=100.0)
        assert applied == {"momentum": -10.0}
        assert "AAPL" not in a.holdings  # fully zeroed out and pruned
        assert a.cash == 100_000 + 10.0 * 100.0  # cash credited back

    def test_partial_fill_scales_holding_and_cash_together(self):
        a = _sleeve(100_000, {"AAPL": 10.0})
        decisions = _decisions("AAPL", {"momentum": 10.0})
        applied = apply_realized_fill({"momentum": a}, decisions, "AAPL", real_filled_qty=6.0, avg_fill_price=100.0)
        assert applied == {"momentum": -4.0}
        assert a.holdings["AAPL"] == 6.0
        assert a.cash == 100_000 + 4.0 * 100.0


class TestCashCorrectionUsesDecisionPriceNotAvgFillPrice:
    """Regression: a cancelled order reports avg_fill_price=0.0 (no fill
    happened at all) -- the cash reversal for the un-filled portion must use
    the ORIGINAL decision-time price, not 0.0, or the reversal silently
    vanishes."""

    def test_full_cancellation_with_zero_avg_fill_price_still_reverses_cash(self):
        a = _sleeve(100_000, {"AAPL": 10.0})
        a.cash -= 10.0 * 250.0  # mimics the original hypothetical debit at decision price 250
        decisions = {
            "momentum": {"status": "approved", "fills": [{"symbol": "AAPL", "shares": 10.0, "price": 250.0}]},
        }
        # Real broker: order never filled at all -- Alpaca reports
        # avg_fill_price=0.0 for a fully cancelled/unfilled order.
        applied = apply_realized_fill({"momentum": a}, decisions, "AAPL", real_filled_qty=0.0, avg_fill_price=0.0)
        assert applied == {"momentum": -10.0}
        assert "AAPL" not in a.holdings
        assert a.cash == 100_000  # fully reversed back to original, not stuck at 100_000 - 2500

    def test_partial_fill_at_different_real_price_than_decided(self):
        a = _sleeve(100_000, {"AAPL": 10.0})
        a.cash -= 10.0 * 250.0  # decided at 250/share
        decisions = {
            "momentum": {"status": "approved", "fills": [{"symbol": "AAPL", "shares": 10.0, "price": 250.0}]},
        }
        # Real fill: only 4 shares actually filled, at a real price of 248.
        applied = apply_realized_fill({"momentum": a}, decisions, "AAPL", real_filled_qty=4.0, avg_fill_price=248.0)
        assert applied == {"momentum": -6.0}
        assert a.holdings["AAPL"] == 4.0
        # cash = 100_000 - 2500 (original debit) + (2500 - 4*248 credited back)
        assert a.cash == 100_000 - 4.0 * 248.0


class TestMultiSleevePartialFillSplit:
    """Two sleeves with opposing signs contribute to one net order --
    pro-rata scaling must preserve each sleeve's direction, not just split
    magnitude blindly."""

    def test_opposing_signs_split_proportionally(self):
        # momentum wants +10 (buy), mean_reversion wants -4 (sell) -> net
        # decided = +6. Real broker only filled +3 (ratio 0.5).
        buyer = _sleeve(100_000, {"AAPL": 10.0})
        seller = _sleeve(100_000, {"AAPL": -4.0})
        decisions = _decisions("AAPL", {"momentum": 10.0, "mean_reversion": -4.0})
        applied = apply_realized_fill(
            {"momentum": buyer, "mean_reversion": seller},
            decisions, "AAPL", real_filled_qty=3.0, avg_fill_price=100.0,
        )
        # ratio = 3/6 = 0.5 -- each contributor scaled by the same ratio
        assert applied["momentum"] == -5.0    # 10 -> 5, correction -5
        assert applied["mean_reversion"] == 2.0  # -4 -> -2, correction +2
        assert buyer.holdings["AAPL"] == 5.0
        assert seller.holdings["AAPL"] == -2.0

    def test_zero_net_decision_qty_is_a_no_op_not_a_zero_division(self):
        # Two sleeves exactly cancel out (+5 and -5) -- net decided is 0,
        # so nothing about this symbol was actually a shared decision to
        # apportion; must not raise ZeroDivisionError.
        a = _sleeve(100_000, {"AAPL": 5.0})
        b = _sleeve(100_000, {"AAPL": -5.0})
        decisions = _decisions("AAPL", {"momentum": 5.0, "mean_reversion": -5.0})
        applied = apply_realized_fill(
            {"momentum": a, "mean_reversion": b}, decisions, "AAPL", real_filled_qty=0.0, avg_fill_price=100.0,
        )
        assert applied == {}
        assert a.holdings["AAPL"] == 5.0  # untouched
        assert b.holdings["AAPL"] == -5.0  # untouched


class TestNoContributingSleeve:
    def test_symbol_not_decided_this_cycle_is_a_no_op(self):
        a = _sleeve(100_000, {"MSFT": 3.0})
        decisions = _decisions("AAPL", {"momentum": 10.0})  # AAPL decided, not MSFT
        applied = apply_realized_fill({"momentum": a}, decisions, "MSFT", real_filled_qty=5.0, avg_fill_price=100.0)
        assert applied == {}
        assert a.holdings["MSFT"] == 3.0  # untouched

    def test_strategy_not_in_sleeve_portfolios_is_skipped_not_raised(self):
        decisions = _decisions("AAPL", {"orphaned_strategy": 10.0})
        applied = apply_realized_fill({}, decisions, "AAPL", real_filled_qty=5.0, avg_fill_price=100.0)
        assert applied == {}
