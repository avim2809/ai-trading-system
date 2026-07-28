"""Execution-safety gate — the hard lock between the agent and a real broker.

No order reaches a *live* broker unless two independent locks are open at once:

1. ``FIRM_ALLOW_TRADING=1`` in the process environment — a human has to set it
   in the service env; the agent cannot flip it mid-run.
2. A per-call ``live=True`` request against a broker that is actually live.

Everything else routes to paper. On top of that, :func:`guard_order` enforces a
:class:`RiskProfile` of hard caps (risk-per-trade, daily loss, min-stop ATR,
notional, symbol allowlist) and requires an exact typed confirmation token for
the live path. Every decision is appended to an immutable audit JSONL.

Ported from the external trading-suite ``execution-safety`` skill and adapted to
firm's live engine: the environment flag is namespaced ``FIRM_ALLOW_TRADING``
and the engine calls :func:`guard_live_submission` as a systemd-friendly hard
gate that sits *on top of* the existing approval queue. It also runs every
order through :func:`guard_order` (with ``live=False`` — actual live/paper
routing is :func:`guard_live_submission`'s job, not duplicated here) so the
:class:`RiskProfile` symbol-allowlist and max-position-notional caps act as a
final, independently-auditable check on the broker-bound order, one level
below the RiskAgent's portfolio-level weight caps. Standard library only.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

# The one environment flag that arms live trading. Namespaced to firm so it
# never collides with an unrelated ALLOW_TRADING in the environment.
ALLOW_TRADING_ENV = "FIRM_ALLOW_TRADING"

# Broker type strings that route to real capital (see routers/live._create_broker).
LIVE_BROKERS = frozenset({"ibkr", "ibkr_live", "alpaca_live"})

# Immutable decision log. Override with FIRM_EXECUTION_AUDIT.
_DEFAULT_AUDIT = Path("data") / "execution_audit.jsonl"


def audit_path() -> Path:
    return Path(os.environ.get("FIRM_EXECUTION_AUDIT", str(_DEFAULT_AUDIT)))


def trading_armed() -> bool:
    """True only when ``FIRM_ALLOW_TRADING=1`` is set in the environment."""
    return os.environ.get(ALLOW_TRADING_ENV) == "1"


def is_live_broker(broker_type: str | None) -> bool:
    """True when *broker_type* routes to a real (non-paper) broker."""
    return (broker_type or "").lower() in LIVE_BROKERS


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class RiskProfile:
    """Hard limits the gate enforces; breach any one and the order is blocked.

    All percentages are of account equity unless noted. An empty
    ``symbol_allowlist`` fails closed (nothing allowed) — the safe default for a
    misconfigured profile.
    """

    account_equity: float = 100_000.0
    max_risk_per_trade_pct: float = 1.0
    max_daily_loss_pct: float = 3.0
    current_open_risk_pct: float = 0.0
    max_position_notional: float = 50_000.0
    min_stop_atr_mult: float = 1.0
    symbol_allowlist: list[str] = field(default_factory=list)
    # A per-trade protective stop is the CLI/discretionary-trading skill's
    # original risk-sizing primitive; the live engine instead rebalances to
    # portfolio target weights (no per-order stop concept) and manages risk
    # via RiskAgent position/sector/correlation/liquidity caps plus the
    # engine's drawdown kill-switch. Set False there so "no stop attached"
    # doesn't block every single rebalancing order; CLI/manual callers keep
    # the stricter default.
    require_stop: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "RiskProfile":
        data = json.loads(Path(path).read_text())
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))


@dataclass
class Order:
    """A proposed order. ``risk_amount`` is the cash at risk if the stop is hit;
    when omitted it is derived from ``|price - stop| * qty``."""

    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    order_type: str = "market"  # "market" | "limit"
    price: float = 0.0
    stop: Optional[float] = None
    atr: Optional[float] = None
    risk_amount: Optional[float] = None

    def notional(self) -> float:
        return abs(self.price * self.qty)

    def derived_risk_amount(self) -> Optional[float]:
        if self.risk_amount is not None:
            return self.risk_amount
        if self.stop is not None and self.price:
            return abs(self.price - self.stop) * self.qty
        return None

    def confirmation_token(self) -> str:
        """Exact phrase a human must echo to authorise a live order.

        Deterministic, e.g. ``'CONFIRM SELL 100 SPY @ market'``.
        """
        qty = int(self.qty) if float(self.qty).is_integer() else self.qty
        venue = "market" if self.order_type == "market" else f"limit {self.price:g}"
        return f"CONFIRM {self.side.upper()} {qty} {self.symbol.upper()} @ {venue}"


# --------------------------------------------------------------------------- #
# Audit log
# --------------------------------------------------------------------------- #
def audit_decision(record: dict[str, Any], path: Path | None = None) -> str:
    """Append a decision record to the audit JSONL. Returns the audit id."""
    path = path or audit_path()
    audit_id = f"aud_{uuid.uuid4().hex[:12]}"
    record = {"audit_id": audit_id, "ts": time.time(), **record}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    log.debug("execution audit %s written to %s", audit_id, path)
    return audit_id


# --------------------------------------------------------------------------- #
# Hard risk limits
# --------------------------------------------------------------------------- #
def check_risk_limits(order: Order, profile: RiskProfile) -> list[str]:
    """Return a list of breach reasons. Empty list = passes every hard limit."""
    breaches: list[str] = []

    if order.symbol.upper() not in {s.upper() for s in profile.symbol_allowlist}:
        breaches.append(
            f"{order.symbol} is not on the allowlist {profile.symbol_allowlist}."
        )

    risk_amt = order.derived_risk_amount()
    if order.stop is None:
        if profile.require_stop:
            breaches.append("No protective stop attached.")
    elif order.atr is not None and order.price:
        stop_dist = abs(order.price - order.stop)
        min_dist = profile.min_stop_atr_mult * order.atr
        if stop_dist < min_dist:
            breaches.append(
                f"Stop is {stop_dist:.4g} away ({stop_dist / order.atr:.2f}x ATR); "
                f"minimum is {profile.min_stop_atr_mult:g}x ATR ({min_dist:.4g})."
            )

    if risk_amt is not None and profile.account_equity > 0:
        risk_pct = 100.0 * risk_amt / profile.account_equity
        if risk_pct > profile.max_risk_per_trade_pct:
            breaches.append(
                f"Risk {risk_pct:.2f}% of equity exceeds the "
                f"{profile.max_risk_per_trade_pct:g}% per-trade cap."
            )
        projected = profile.current_open_risk_pct + risk_pct
        if projected > profile.max_daily_loss_pct:
            breaches.append(
                f"Projected open risk {projected:.2f}% exceeds the "
                f"{profile.max_daily_loss_pct:g}% daily stop."
            )

    if order.notional() > profile.max_position_notional:
        breaches.append(
            f"Notional {order.notional():.2f} exceeds the max position "
            f"{profile.max_position_notional:.2f}."
        )

    return breaches


# --------------------------------------------------------------------------- #
# The gate (standalone / testable)
# --------------------------------------------------------------------------- #
def guard_order(
    order: Order,
    profile: RiskProfile,
    *,
    live: bool = False,
    confirmation: Optional[str] = None,
    audit: Path | None = None,
) -> dict[str, Any]:
    """Run an order through the full safety pipeline and return the decision.

    Pipeline (first failure wins): hard risk limits → mode/env lock → typed
    confirmation → route. Returns
    ``{"routed": "live"|"paper"|"blocked", "reason", "audit_id", ...}``. No real
    order is ever transmitted here — routing "live" only means the order cleared
    every gate; the caller hands a cleared order to the actual broker.
    """
    env_armed = trading_armed()
    wants_live = bool(live)
    live_mode = wants_live and env_armed

    base = {
        "symbol": order.symbol,
        "side": order.side,
        "qty": order.qty,
        "order_type": order.order_type,
        "requested_live": wants_live,
        "env_armed": env_armed,
        "live_mode": live_mode,
    }

    def finish(routed: str, reason: str, extra: dict | None = None) -> dict[str, Any]:
        record = {"decision": routed, "reason": reason, **base, **(extra or {})}
        audit_id = audit_decision(record, audit)
        _lvl = logging.WARNING if routed == "blocked" else (
            logging.INFO if routed == "live" else logging.DEBUG
        )
        log.log(
            _lvl,
            "guard_order[%s]: routed=%s (audit=%s) — %s",
            order.symbol, routed, audit_id, reason,
        )
        return {"routed": routed, "reason": reason, "audit_id": audit_id, **(extra or {})}

    breaches = check_risk_limits(order, profile)
    if breaches:
        return finish("blocked", "; ".join(breaches), {"breaches": breaches})

    if live_mode:
        expected = order.confirmation_token()
        if confirmation != expected:
            return finish(
                "blocked",
                f"Live order requires exact confirmation '{expected}'. "
                f"Got {confirmation!r}.",
                {"expected_confirmation": expected},
            )
        return finish("live", "Cleared for live routing.", {})

    if wants_live and not env_armed:
        reason = (
            f"Live requested but {ALLOW_TRADING_ENV}=1 is not set — routed to "
            "paper. Set it in the service env to arm live trading."
        )
    else:
        reason = "Routed to paper (default mode)."
    return finish("paper", reason, {})


# --------------------------------------------------------------------------- #
# Engine-facing gate
# --------------------------------------------------------------------------- #
def guard_live_submission(
    broker_type: str | None,
    order: dict[str, Any],
    *,
    cycle_id: int = 0,
    audit: Path | None = None,
) -> dict[str, Any]:
    """Lightweight env-lock gate used by the live engine before broker submit.

    Live brokers require ``FIRM_ALLOW_TRADING=1``; without it the order is
    blocked (never silently downgraded) and the decision is audited. Paper
    brokers always pass. This is a hard gate on top of the human approval queue,
    not a replacement for it.

    Returns ``{"allowed": bool, "reason": str, "audit_id": str}``.
    """
    live = is_live_broker(broker_type)
    armed = trading_armed()
    allowed = (not live) or armed

    record = {
        "kind": "live_submission_gate",
        "broker_type": broker_type,
        "symbol": order.get("symbol"),
        "side": order.get("side"),
        "quantity": order.get("quantity", order.get("shares")),
        "strategy": order.get("strategy"),
        "cycle_id": cycle_id,
        "live_broker": live,
        "env_armed": armed,
        "allowed": allowed,
    }
    if allowed:
        reason = (
            "paper broker — no live lock required"
            if not live
            else f"live broker armed ({ALLOW_TRADING_ENV}=1)"
        )
    else:
        reason = (
            f"BLOCKED: live broker {broker_type!r} requires {ALLOW_TRADING_ENV}=1 "
            "in the service environment; order not submitted."
        )
    record["reason"] = reason
    audit_id = audit_decision(record, audit)
    if not allowed:
        log.warning(
            "guard_live_submission BLOCKED %s %s (broker=%s, cycle=%d, audit=%s): %s",
            record.get("side"), record.get("symbol"), broker_type,
            cycle_id, audit_id, reason,
        )
    elif live:
        log.info(
            "guard_live_submission ALLOWED live %s %s (broker=%s, cycle=%d, audit=%s)",
            record.get("side"), record.get("symbol"), broker_type, cycle_id, audit_id,
        )
    else:
        log.debug(
            "guard_live_submission paper %s %s (broker=%s, cycle=%d, audit=%s)",
            record.get("side"), record.get("symbol"), broker_type, cycle_id, audit_id,
        )
    return {"allowed": allowed, "reason": reason, "audit_id": audit_id}
