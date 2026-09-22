#!/usr/bin/env python
"""Decompose sleeved-mode reconciliation drift into known, explainable causes.

Read-only diagnostic. Does not correct anything, does not touch the broker
or any live state -- see docs/claude-memory/project_vps_migration_sep22.md
and /root/.claude/plans/compressed-mapping-frost.md for the full incident
this was built for: the Alpaca (:8001, sleeved) instance's
GET /api/live/reconciliation showed a growing, unexplained mismatch. Sleeve
books (Orchestrator._sleeve_portfolios) are credited from a virtual,
hypothetical execution pass at decision time -- assumed 100% fill at the
decision-time price/quantity -- never from what the broker actually did.

Uses only order_history.json (now trustworthy after the AlpacaBroker
_map_order fix landed the same day -- previously every record showed
status="pending" regardless of real fill state). Per symbol: how much of
"submitted quantity" never actually filled (cancelled/rejected -- a clean,
reliable single-source bucket), versus what's left over (residual --
genuinely needs a symbol-by-symbol look, not further auto-decomposed here).

execution_audit.jsonl was tried as a second cross-reference source (to
split out integer-share rounding and guard-rejected-before-submission
quantity) and deliberately dropped: it logs the live-submission gate check,
which fires more often than once per real trading decision (confirmed
live: 23 separate AAPL audit records on a single day, when at most ~7
hourly cycles could have run that day) -- summing it produces a
plausible-looking but wrong number (an early version of this script
reported "AAPL rounding: -600.55 shares", which is impossible for a
rounding effect). Don't resurrect that cross-reference without first
understanding why the gate fires that many times per symbol per day.

Usage:
    python scripts/diagnose_sleeve_drift.py --data-dir data_alpaca
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_order_history(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "order_history.json"
    if not path.exists():
        return []
    with path.open() as f:
        return json.load(f)


def _signed_qty(order: dict[str, Any], qty_field: str = "filled_quantity") -> float:
    side = str(order.get("side", "")).lower().rsplit(".", 1)[-1]
    qty = float(order.get(qty_field, 0.0) or 0.0)
    return qty if side == "buy" else -qty


def diagnose(data_dir: Path) -> None:
    orders = _load_order_history(data_dir)

    broker_total: dict[str, float] = defaultdict(float)  # real signed filled qty
    decided_total: dict[str, float] = defaultdict(float)  # signed submitted (post-rounding) qty
    cancelled_or_rejected: dict[str, float] = defaultdict(float)
    order_count: dict[str, int] = defaultdict(int)

    for o in orders:
        symbol = o.get("symbol")
        if not symbol:
            continue
        status = o.get("status")
        order_count[symbol] += 1
        broker_total[symbol] += _signed_qty(o, "filled_quantity")
        decided_total[symbol] += _signed_qty(o, "quantity")
        if status in ("cancelled", "rejected"):
            cancelled_or_rejected[symbol] += _signed_qty(o, "quantity")

    symbols = sorted(set(broker_total) | set(decided_total))

    print(f"{'symbol':<8}{'#orders':>9}{'broker_fill':>14}{'submitted':>12}"
          f"{'cancelled':>12}{'residual':>12}")
    for sym in symbols:
        broker = broker_total.get(sym, 0.0)
        decided = decided_total.get(sym, 0.0)
        cancelled = cancelled_or_rejected.get(sym, 0.0)
        residual = (decided - broker) - cancelled
        print(f"{sym:<8}{order_count.get(sym, 0):>9}{broker:>14.2f}{decided:>12.2f}"
              f"{cancelled:>12.2f}{residual:>12.2f}")

    print()
    print("residual = (submitted - broker_fill) - cancelled: what's left after")
    print("accounting for orders that never filled at all. Not zero on most")
    print("symbols is expected (partial fills, still-open orders) -- look for")
    print("symbols where |residual| is large relative to #orders/broker_fill,")
    print("not for a globally-zero total.")
    print()
    print("This does NOT explain drift caused by the sleeve-crediting design")
    print("gap itself (virtual fills never reconciled per-sleeve) -- see")
    print("Phase 3 of the plan for that. This script only checks whether")
    print("order_history.json's own submitted-vs-filled numbers are sane.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data_alpaca", help="instance data dir (default: data_alpaca)")
    args = parser.parse_args()
    diagnose(Path(args.data_dir))


if __name__ == "__main__":
    sys.exit(main())
