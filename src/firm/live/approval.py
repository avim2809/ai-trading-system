"""Approval queue for semi-auto live trading.

Pending trade proposals wait here until a human approves or rejects
them via the API.  Stale approvals auto-expire.  State is persisted
to a JSON file so nothing is lost across restarts.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from firm.api.serializers import serialize_blackboard
from firm.brokers.base import Broker, OrderRequest, OrderStatus
from firm.time_utils import utcnow

log = logging.getLogger(__name__)

_DEFAULT_EXPIRY_MINUTES = 60


@dataclass
class PendingApproval:
    """One pending trade proposal awaiting human approval."""

    approval_id: str
    created_at: datetime
    expires_at: datetime
    orders: list[dict[str, Any]]
    blackboard_snapshot: dict[str, Any]
    status: Literal["pending", "approved", "rejected", "expired"] = "pending"
    reject_reason: str = ""
    strategy: str = ""

    def is_expired(self) -> bool:
        return self.status == "pending" and utcnow() > self.expires_at


class ApprovalQueue:
    """In-memory approval queue with JSON file persistence."""

    def __init__(
        self,
        broker: Broker | None = None,
        persist_path: str | Path | None = None,
        expiry_minutes: int = _DEFAULT_EXPIRY_MINUTES,
    ) -> None:
        self._broker = broker
        self._persist_path = Path(persist_path) if persist_path else None
        self._expiry_minutes = expiry_minutes
        self._queue: list[PendingApproval] = []
        if self._persist_path and self._persist_path.exists():
            self._load()

    def set_broker(self, broker: Broker) -> None:
        self._broker = broker

    def add(
        self,
        orders: list[dict[str, Any]],
        blackboard: Any,
        strategy: str = "",
    ) -> str:
        """Enqueue orders for approval.  Returns the approval_id."""
        now = utcnow()
        approval = PendingApproval(
            approval_id=uuid.uuid4().hex[:12],
            created_at=now,
            expires_at=now + timedelta(minutes=self._expiry_minutes),
            orders=orders,
            blackboard_snapshot=serialize_blackboard(blackboard) if hasattr(blackboard, "asof") else {},
            strategy=strategy,
        )
        self._queue.append(approval)
        self._save()
        log.info("Queued approval %s (%d orders)", approval.approval_id, len(orders))
        return approval.approval_id

    def get_pending(self) -> list[PendingApproval]:
        self.expire_stale()
        return [a for a in self._queue if a.status == "pending"]

    def get_all(self) -> list[PendingApproval]:
        self.expire_stale()
        return list(self._queue)

    def get_by_id(self, approval_id: str) -> PendingApproval | None:
        for a in self._queue:
            if a.approval_id == approval_id:
                return a
        return None

    def resolve_id(self, approval_id: str) -> str:
        """Resolve a full or unique-prefix approval id."""
        if self.get_by_id(approval_id) is not None:
            return approval_id
        matches = [
            a.approval_id for a in self._queue if a.approval_id.startswith(approval_id)
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Approval id {approval_id!r} is ambiguous ({len(matches)} matches)"
            )
        raise ValueError(f"Approval {approval_id} not found")

    def clear(self) -> int:
        """Wipe every approval record (pending and historical) and its
        persisted file. Returns the number removed."""
        count = len(self._queue)
        self._queue = []
        self._save()
        return count

    def approve(self, approval_id: str) -> list[OrderStatus]:
        """Approve and execute the pending orders."""
        approval_id = self.resolve_id(approval_id)
        approval = self.get_by_id(approval_id)
        if approval is None:
            raise ValueError(f"Approval {approval_id} not found")
        if approval.status != "pending":
            raise ValueError(f"Approval {approval_id} is {approval.status}, not pending")
        if approval.is_expired():
            approval.status = "expired"
            self._save()
            raise ValueError(f"Approval {approval_id} has expired")
        if self._broker is None:
            raise RuntimeError("No broker configured on ApprovalQueue")

        results: list[OrderStatus] = []
        for order_dict in approval.orders:
            raw_qty = float(order_dict.get("quantity", abs(order_dict.get("shares", 0))))
            share_qty = int(round(abs(raw_qty)))
            if share_qty <= 0:
                log.debug(
                    "Skipping dust approval order %s %s (raw qty %.4f rounds to 0 shares)",
                    order_dict.get("side"), order_dict.get("symbol"), raw_qty,
                )
                continue
            req = OrderRequest(
                symbol=order_dict["symbol"],
                side=order_dict["side"],
                quantity=share_qty,
                order_type=order_dict.get("order_type", "market"),
                limit_price=order_dict.get("limit_price"),
                strategy=order_dict.get("strategy", ""),
                client_order_id=f"appr-{approval_id}-{order_dict['symbol']}-{order_dict['side']}",
            )
            try:
                status = self._broker.submit_order(req)
                results.append(status)
            except Exception:
                log.error("Failed to submit order for %s", req.symbol, exc_info=True)

        approval.status = "approved"
        self._save()
        log.info("Approved %s – submitted %d orders", approval_id, len(results))
        return results

    def reject(self, approval_id: str, reason: str = "") -> None:
        approval = self.get_by_id(approval_id)
        if approval is None:
            raise ValueError(f"Approval {approval_id} not found")
        if approval.status != "pending":
            raise ValueError(f"Approval {approval_id} is {approval.status}, not pending")
        approval.status = "rejected"
        approval.reject_reason = reason
        self._save()
        log.info("Rejected %s: %s", approval_id, reason)

    def expire_stale(self) -> int:
        """Auto-expire overdue approvals.  Returns count expired."""
        count = 0
        for a in self._queue:
            if a.is_expired():
                a.status = "expired"
                count += 1
        if count:
            self._save()
            log.info("Expired %d stale approvals", count)
        return count

    def _save(self) -> None:
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for a in self._queue:
            d = {
                "approval_id": a.approval_id,
                "created_at": a.created_at.isoformat(),
                "expires_at": a.expires_at.isoformat(),
                "orders": a.orders,
                "blackboard_snapshot": a.blackboard_snapshot,
                "status": a.status,
                "reject_reason": a.reject_reason,
                "strategy": a.strategy,
            }
            data.append(d)
        self._persist_path.write_text(json.dumps(data, indent=2, default=str))

    def _load(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        try:
            data = json.loads(self._persist_path.read_text())
            for d in data:
                self._queue.append(
                    PendingApproval(
                        approval_id=d["approval_id"],
                        created_at=datetime.fromisoformat(d["created_at"]),
                        expires_at=datetime.fromisoformat(d["expires_at"]),
                        orders=d["orders"],
                        blackboard_snapshot=d.get("blackboard_snapshot", {}),
                        status=d.get("status", "pending"),
                        reject_reason=d.get("reject_reason", ""),
                        strategy=d.get("strategy", ""),
                    )
                )
            log.info("Loaded %d approvals from %s", len(self._queue), self._persist_path)
        except Exception:
            log.warning("Failed to load approval queue", exc_info=True)
