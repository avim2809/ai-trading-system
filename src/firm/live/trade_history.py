"""Persistent live order and cycle history for the dashboard.

In-memory engine state is lost on restart; this module appends submitted
orders and cycle summaries to JSON files under ``data/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_ORDERS_PATH = Path("data/order_history.json")
_DEFAULT_CYCLES_PATH = Path("data/cycle_history.json")


class TradeHistoryStore:
    """Append-only JSON persistence for live orders and cycle summaries."""

    def __init__(
        self,
        orders_path: str | Path | None = None,
        cycles_path: str | Path | None = None,
    ) -> None:
        self._orders_path = Path(orders_path or _DEFAULT_ORDERS_PATH)
        self._cycles_path = Path(cycles_path or _DEFAULT_CYCLES_PATH)
        self._orders: list[dict[str, Any]] = []
        self._cycles: list[dict[str, Any]] = []
        self._load()

    def record_orders(
        self,
        orders: list[dict[str, Any]],
        *,
        cycle_id: int | None = None,
        source: str = "cycle",
        approval_id: str | None = None,
    ) -> None:
        if not orders:
            return
        for row in orders:
            entry = dict(row)
            entry.setdefault("source", source)
            if cycle_id is not None:
                entry["cycle_id"] = cycle_id
            if approval_id is not None:
                entry["approval_id"] = approval_id
            self._orders.append(entry)
        self._save_orders()
        log.debug(
            "Recorded %d order(s) to trade history (source=%s)", len(orders), source,
        )

    def update_order_status(
        self,
        order_id: str,
        *,
        status: str,
        filled_quantity: float,
        avg_fill_price: float,
    ) -> bool:
        """Correct a previously recorded order's status/fill fields in place.

        Records are otherwise append-only, written once at submission time —
        this is the one exception, used by reconciliation against the
        broker's true post-submission status (see
        ``firm.live.order_reconciliation``). Returns True if a matching
        record was found and updated.
        """
        updated = False
        for entry in self._orders:
            if entry.get("order_id") == order_id:
                entry["status"] = status
                entry["filled_quantity"] = filled_quantity
                entry["avg_fill_price"] = avg_fill_price
                updated = True
        if updated:
            self._save_orders()
        return updated

    def record_cycle(self, summary: dict[str, Any]) -> None:
        self._cycles.append(dict(summary))
        self._save_cycles()
        log.debug("Recorded cycle %s to trade history", summary.get("cycle_id"))

    def list_orders(self, limit: int = 500) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return list(reversed(self._orders[-limit:]))

    def list_cycles(self, limit: int = 50) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return list(reversed(self._cycles[-limit:]))

    def clear_orders(self) -> int:
        count = len(self._orders)
        self._orders = []
        self._save_orders()
        return count

    def clear_cycles(self) -> int:
        count = len(self._cycles)
        self._cycles = []
        self._save_cycles()
        return count

    def clear_all(self) -> dict[str, int]:
        orders = self.clear_orders()
        cycles = self.clear_cycles()
        return {"orders": orders, "cycles": cycles}

    def _load(self) -> None:
        self._orders = self._read_list(self._orders_path, "order")
        self._cycles = self._read_list(self._cycles_path, "cycle")

    def _read_list(self, path: Path, label: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                log.warning("Trade history %s file is not a list — ignoring", path)
                return []
            log.info("Loaded %d %s history record(s) from %s", len(data), label, path)
            return [row for row in data if isinstance(row, dict)]
        except Exception:
            log.warning("Failed to load %s history from %s", label, path, exc_info=True)
            return []

    def _save_orders(self) -> None:
        self._write_list(self._orders_path, self._orders)

    def _save_cycles(self) -> None:
        self._write_list(self._cycles_path, self._cycles)

    @staticmethod
    def _write_list(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
