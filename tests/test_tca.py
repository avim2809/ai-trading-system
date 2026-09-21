"""Tests for firm.eval.tca: realized-vs-modeled execution cost.

Covers the calculation logic directly against the persisted order-record
shape ``LiveTradingEngine._status_to_dict`` produces (decision-time
``price``/``est_*`` merged with the broker's ``avg_fill_price``/``status``)
-- see test_live_engine.py's TestExecuteOrdersTCA for the plumbing that
produces this shape from a live cycle, and test_api.py's
TestLiveTCAEndpoint for the endpoint wiring.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from firm.eval.tca import aggregate_tca, compute_tca_record


def _order(
    *,
    side: str = "buy",
    price: float = 100.0,
    avg_fill_price: float = 100.0,
    filled_quantity: float = 10.0,
    status: str = "filled",
    strategy: str = "momentum",
    est_commission: float = 0.0,
    est_slippage: float = 0.0,
    est_spread: float = 0.0,
    est_impact: float = 0.0,
    notional: float | None = None,
    timestamp: str | None = None,
) -> dict:
    notional = notional if notional is not None else price * filled_quantity
    return {
        "order_id": "o1",
        "symbol": "AAPL",
        "side": side,
        "strategy": strategy,
        "status": status,
        "filled_quantity": filled_quantity,
        "avg_fill_price": avg_fill_price,
        "price": price,
        "notional": notional,
        "est_commission": est_commission,
        "est_slippage": est_slippage,
        "est_spread": est_spread,
        "est_impact": est_impact,
        "est_cost": est_commission + est_slippage + est_spread + est_impact,
        "timestamp": timestamp or "2026-09-22T10:00:00",
    }


class TestComputeTcaRecord:
    def test_buy_filled_worse_than_decision_is_a_positive_cost(self):
        """Bought at 150 when the decision was made at 149 -- paid more,
        so this must read as a positive (adverse) realized cost."""
        record = compute_tca_record(_order(side="buy", price=149.0, avg_fill_price=150.0, filled_quantity=10.0))
        assert record["computable"] is True
        assert record["realized_slippage_bps"] == (1.0 / 149.0) * 10_000
        assert record["realized_cost_dollars"] == 10.0  # $1/share * 10 shares

    def test_sell_filled_worse_than_decision_is_a_positive_cost(self):
        """Sold at 149 when the decision was made at 150 -- received less,
        so this is also an adverse (positive) realized cost, mirroring the
        buy case in the opposite price direction."""
        record = compute_tca_record(_order(side="sell", price=150.0, avg_fill_price=149.0, filled_quantity=10.0))
        assert record["computable"] is True
        assert record["realized_slippage_bps"] == (1.0 / 150.0) * 10_000
        assert record["realized_cost_dollars"] == 10.0

    def test_price_improvement_is_negative(self):
        """A buy that filled *below* the decision price saved money -- the
        realized cost must come back negative, not clamped to zero."""
        record = compute_tca_record(_order(side="buy", price=150.0, avg_fill_price=149.0, filled_quantity=10.0))
        assert record["computable"] is True
        assert record["realized_slippage_bps"] < 0
        assert record["realized_cost_dollars"] == -10.0

    def test_exact_fill_at_decision_price_is_zero_not_unknown(self):
        record = compute_tca_record(_order(price=100.0, avg_fill_price=100.0))
        assert record["computable"] is True
        assert record["realized_slippage_bps"] == 0.0

    def test_modeled_cost_excludes_commission(self):
        """Commission is a flat fee, not a price deviation -- it must not
        be folded into modeled_cost_bps (which is compared against a
        realized *price*-based slippage), even though it's part of the
        order's total est_cost."""
        record = compute_tca_record(_order(
            price=100.0, avg_fill_price=100.0, filled_quantity=10.0, notional=1000.0,
            est_commission=5.0, est_slippage=1.0, est_spread=0.5, est_impact=0.5,
        ))
        # (1.0 + 0.5 + 0.5) / 1000 * 10_000 = 20 bps -- commission excluded.
        assert record["modeled_cost_bps"] == 20.0
        assert record["est_cost"] == 7.0  # still reports the full modeled total elsewhere

    def test_cost_error_is_realized_minus_modeled(self):
        record = compute_tca_record(_order(
            price=149.0, avg_fill_price=150.0, filled_quantity=10.0, notional=1490.0,
            est_slippage=0.75, est_spread=0.3, est_impact=0.1,
        ))
        expected_realized = (1.0 / 149.0) * 10_000
        expected_modeled = (0.75 + 0.3 + 0.1) / 1490.0 * 10_000
        assert record["cost_error_bps"] == expected_realized - expected_modeled

    def test_pending_order_is_not_computable(self):
        """A broker response that hasn't settled past 'pending' has no real
        fill price yet -- must not be silently treated as zero slippage."""
        record = compute_tca_record(_order(status="pending", avg_fill_price=0.0))
        assert record["computable"] is False
        assert record["realized_slippage_bps"] is None
        assert record["realized_cost_dollars"] is None

    def test_zero_avg_fill_price_on_a_filled_status_is_not_computable(self):
        """Defensive: a 'filled' status with avg_fill_price still 0.0 (e.g.
        a broker quirk, or a record captured before the fill settled) must
        not compute a division-by-zero-adjacent bogus slippage number."""
        record = compute_tca_record(_order(status="filled", avg_fill_price=0.0))
        assert record["computable"] is False

    def test_missing_decision_price_is_not_computable(self):
        """A record from before decision-time context was merged in (or
        one that went through a code path that doesn't merge it, e.g. the
        approval-queue path) must degrade to 'unknown', not crash or
        assume price == 0."""
        order = _order()
        del order["price"]
        record = compute_tca_record(order)
        assert record["computable"] is False

    def test_partial_fill_is_computable_on_the_filled_portion(self):
        record = compute_tca_record(_order(status="partial", filled_quantity=4.0, avg_fill_price=101.0, price=100.0))
        assert record["computable"] is True
        assert record["realized_cost_dollars"] == 4.0  # $1/share * 4 filled shares


class TestAggregateTca:
    def test_empty_input(self):
        agg = aggregate_tca([])
        assert agg["n_orders"] == 0
        assert agg["n_computable"] == 0
        assert agg["realized_slippage_bps"] == {"mean": None, "median": None, "n": 0}

    def test_unknown_fill_counted_but_excluded_from_stats(self):
        orders = [
            _order(price=100.0, avg_fill_price=101.0, strategy="momentum"),
            _order(status="pending", avg_fill_price=0.0, strategy="momentum"),
        ]
        agg = aggregate_tca(orders)
        assert agg["n_orders"] == 2
        assert agg["n_computable"] == 1
        assert agg["n_unknown_fill"] == 1
        assert agg["realized_slippage_bps"]["n"] == 1

    def test_mean_and_median_match_hand_computed_values(self):
        # Three buys, decision price 100, filled at 101/102/103 -> realized
        # slippage bps = 100, 200, 300 respectively (1/100, 2/100, 3/100 * 1e4).
        orders = [
            _order(price=100.0, avg_fill_price=101.0),
            _order(price=100.0, avg_fill_price=102.0),
            _order(price=100.0, avg_fill_price=103.0),
        ]
        agg = aggregate_tca(orders)
        stats = agg["realized_slippage_bps"]
        assert stats["n"] == 3
        assert stats["mean"] == 200.0
        assert stats["median"] == 200.0

    def test_group_by_strategy_isolates_each_strategys_orders(self):
        orders = [
            _order(price=100.0, avg_fill_price=101.0, strategy="momentum"),
            _order(price=100.0, avg_fill_price=110.0, strategy="trend"),
        ]
        agg = aggregate_tca(orders, group_by="strategy")
        by_strategy = agg["by_strategy"]
        assert by_strategy["momentum"]["n"] == 1
        assert by_strategy["trend"]["n"] == 1
        assert by_strategy["momentum"]["realized_slippage_bps"]["mean"] == 100.0
        assert by_strategy["trend"]["realized_slippage_bps"]["mean"] == 1000.0

    def test_since_filters_out_older_records(self):
        now = datetime(2026, 9, 22, 12, 0, 0)
        old = _order(price=100.0, avg_fill_price=200.0, timestamp=(now - timedelta(days=10)).isoformat())
        recent = _order(price=100.0, avg_fill_price=101.0, timestamp=(now - timedelta(hours=1)).isoformat())
        agg = aggregate_tca([old, recent], since=now - timedelta(days=1))
        assert agg["n_orders"] == 1
        assert agg["realized_slippage_bps"]["mean"] == 100.0

    def test_record_with_unparseable_timestamp_is_kept_not_dropped(self):
        order = _order(price=100.0, avg_fill_price=101.0, timestamp="not-a-date")
        agg = aggregate_tca([order], since=datetime(2026, 9, 22))
        assert agg["n_orders"] == 1

    def test_total_est_commission_sums_across_all_records_including_unknown_fill(self):
        orders = [
            _order(est_commission=1.5),
            _order(status="pending", avg_fill_price=0.0, est_commission=2.5),
        ]
        agg = aggregate_tca(orders)
        assert agg["total_est_commission"] == 4.0
