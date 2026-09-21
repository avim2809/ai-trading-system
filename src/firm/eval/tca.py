"""Transaction cost analysis: realized (broker fill) vs. modeled (pre-trade
estimate) execution cost.

``firm.agents.execution.ExecutionAgent`` already estimates commission,
slippage, spread, and market-impact cost *before* an order is submitted
(``est_commission``/``est_slippage``/``est_spread``/``est_impact``/
``est_cost``, all in dollars, alongside the decision-time ``price`` the
estimate was computed against). Those fields are merged onto the same
persisted order record the broker's actual fill lands on (see
``LiveTradingEngine._status_to_dict``'s ``order`` parameter and
``firm.live.trade_history.TradeHistoryStore``), so this module is a pure
function over that one record shape — no separate store, no live state.

Only the *price-based* cost components (slippage/spread/impact) are
comparable to a realized fill price: commission is a flat fee applied
regardless of execution quality, so it has no "realized" counterpart to
compare against and is reported as-is rather than folded into the
comparison.
"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import Any

_PRICE_BASED_EST_KEYS = ("est_slippage", "est_spread", "est_impact")


def compute_tca_record(entry: dict[str, Any]) -> dict[str, Any]:
    """Derive realized-vs-modeled cost for one persisted order record.

    ``entry`` is one row from ``TradeHistoryStore.list_orders()`` (or the
    same shape from ``CycleResult.order_statuses``): broker fields
    (``avg_fill_price``, ``filled_quantity``, ``status``) plus the
    decision-time fields merged in at submission time (``price``,
    ``notional``, ``est_commission``, ``est_slippage``, ``est_spread``,
    ``est_impact``, ``est_cost``).

    Returns a record with ``computable: False`` (and every cost field
    ``None``) rather than raising or silently reporting zero, whenever a
    real fill price isn't actually known yet — e.g. a market order whose
    broker response hasn't settled past "pending" (``avg_fill_price`` is
    still ``0.0``), or a pre-execution-context record from before this
    field existed. Callers must check the flag rather than treat a
    present-but-zero ``realized_slippage_bps`` as "no slippage".
    """
    decision_price = _to_float(entry.get("price"))
    fill_price = _to_float(entry.get("avg_fill_price"))
    filled_qty = _to_float(entry.get("filled_quantity"))
    status = entry.get("status")

    computable = (
        status in ("filled", "partial")
        and decision_price is not None and decision_price > 0
        and fill_price is not None and fill_price > 0
        and filled_qty is not None and filled_qty > 0
    )

    record: dict[str, Any] = {
        "order_id": entry.get("order_id"),
        "symbol": entry.get("symbol"),
        "side": entry.get("side"),
        "strategy": entry.get("strategy"),
        "timestamp": entry.get("timestamp"),
        "computable": computable,
        "decision_price": decision_price,
        "fill_price": fill_price,
        "filled_quantity": filled_qty,
        "est_commission": _to_float(entry.get("est_commission")),
        "est_cost": _to_float(entry.get("est_cost")),
        "realized_slippage_bps": None,
        "realized_cost_dollars": None,
        "modeled_cost_bps": None,
        "cost_error_bps": None,
    }
    if not computable:
        return record

    # Adverse direction depends on side: a buy is worse the higher the fill
    # lands above the decision price; a sell is worse the lower it lands
    # below it. Positive == cost, negative == price improvement.
    adverse = (
        (fill_price - decision_price) if entry.get("side") == "buy"
        else (decision_price - fill_price)
    )
    decision_notional = _to_float(entry.get("notional")) or (decision_price * filled_qty)
    modeled_market_cost = sum(
        _to_float(entry.get(key)) or 0.0 for key in _PRICE_BASED_EST_KEYS
    )

    realized_slippage_bps = adverse / decision_price * 10_000
    modeled_cost_bps = (
        modeled_market_cost / decision_notional * 10_000 if decision_notional > 0 else None
    )

    record["realized_slippage_bps"] = realized_slippage_bps
    record["realized_cost_dollars"] = adverse * filled_qty
    record["modeled_cost_bps"] = modeled_cost_bps
    record["cost_error_bps"] = (
        realized_slippage_bps - modeled_cost_bps if modeled_cost_bps is not None else None
    )
    return record


def aggregate_tca(
    entries: list[dict[str, Any]],
    *,
    group_by: str = "strategy",
    since: datetime | None = None,
) -> dict[str, Any]:
    """Aggregate realized-vs-modeled cost across persisted order records.

    ``since`` filters to records whose ``timestamp`` parses to on/after
    that cutoff (naive UTC, matching ``firm.time_utils.utcnow``); a record
    with a missing/unparseable timestamp is kept rather than dropped, so a
    malformed timestamp can't silently shrink the window.

    Records that aren't ``computable`` (see :func:`compute_tca_record`)
    are counted in ``n_orders``/``n_unknown_fill`` but excluded from every
    bps statistic — the honest answer when a broker fill's real price
    isn't known is "not counted", not "zero slippage".
    """
    if since is not None:
        kept = []
        for e in entries:
            ts = _parse_ts(e.get("timestamp"))
            if ts is None or ts >= since:
                kept.append(e)
        entries = kept

    records = [compute_tca_record(e) for e in entries]
    computable = [r for r in records if r["computable"]]

    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in computable:
        key = str(r.get(group_by) or "unknown")
        by_group.setdefault(key, []).append(r)

    return {
        "n_orders": len(records),
        "n_computable": len(computable),
        "n_unknown_fill": len(records) - len(computable),
        "realized_slippage_bps": _bps_stats(_field(computable, "realized_slippage_bps")),
        "modeled_cost_bps": _bps_stats(_field(computable, "modeled_cost_bps")),
        "cost_error_bps": _bps_stats(_field(computable, "cost_error_bps")),
        "total_est_commission": sum(_field(records, "est_commission")),
        f"by_{group_by}": {
            key: _group_stats(group_records) for key, group_records in by_group.items()
        },
    }


def _group_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(records),
        "realized_slippage_bps": _bps_stats(_field(records, "realized_slippage_bps")),
        "modeled_cost_bps": _bps_stats(_field(records, "modeled_cost_bps")),
        "cost_error_bps": _bps_stats(_field(records, "cost_error_bps")),
    }


def _field(records: list[dict[str, Any]], key: str) -> list[float]:
    return [r[key] for r in records if r.get(key) is not None]


def _bps_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "n": 0}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "n": len(values),
    }


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
