"""Combine sleeve targets into one long-only book and plan the orders to get there.

The allocator is deliberately pure: it takes the broker's real positions,
NAV and prices as inputs and returns an :class:`AllocationPlan` (targets,
actual weights, which sleeves rebalance, and order dicts in the exact shape
``LiveTradingEngine._execute_orders`` consumes). It never talks to a broker
or a data provider itself, so it is fully unit-testable and the engine keeps
sole ownership of submission, safety gates and persistence.

Trading rules (see ``plan``):

* A symbol trades when any sleeve that owns it is due for its scheduled
  rebalance, OR its actual weight has drifted more than ``band_abs`` from
  its combined target (checked every run).
* Sells are always emitted before buys.
* Never short: a sell never exceeds the current long quantity, targets are
  clamped at >= 0. Never lever: projected gross exposure after the plan is
  capped at ``max_gross`` x NAV by scaling buys down.
* Symbols held at the broker that no sleeve targets are left alone (logged)
  unless ``liquidate_unmanaged`` is set.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from firm.allocation.sleeves import Sleeve, SleeveConfigError

log = logging.getLogger(__name__)

UNMANAGED_STRATEGY = "allocation_unmanaged"
_FRACTIONAL_DECIMALS = 6
_EPS = 1e-9


@dataclass
class AllocationPlan:
    """Result of one :meth:`Allocator.plan` call (JSON-serializable via to_dict)."""

    asof: str
    nav: float
    targets: dict[str, float] = field(default_factory=dict)
    actual_weights: dict[str, float] = field(default_factory=dict)
    sleeve_targets: dict[str, dict[str, float]] = field(default_factory=dict)
    sleeve_weights: dict[str, float] = field(default_factory=dict)
    due_sleeves: list[str] = field(default_factory=list)
    # Sleeves the plan brings back to target this run (due and fully
    # plannable). The engine records last_rebalance for these only after
    # their orders actually submit.
    rebalanced_sleeves: list[str] = field(default_factory=list)
    drift_symbols: list[str] = field(default_factory=list)
    unmanaged: dict[str, float] = field(default_factory=dict)
    orders: list[dict[str, Any]] = field(default_factory=list)
    # symbol -> sleeves that contributed to it (order attribution)
    symbol_sleeves: dict[str, list[str]] = field(default_factory=dict)
    gross_before: float = 0.0
    gross_after: float = 0.0
    buy_scale: float = 1.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _order_notional(order: dict[str, Any]) -> float:
    return abs(float(order["quantity"]) * float(order["price"]))


class Allocator:
    """Plan orders that move the broker book toward the combined sleeve targets."""

    def __init__(
        self,
        sleeves: list[Sleeve],
        band_abs: float = 0.02,
        min_order_notional: float = 100.0,
        liquidate_unmanaged: bool = False,
        max_gross: float = 1.0,
        cash_buffer: float = 0.0,
    ) -> None:
        if not sleeves:
            raise SleeveConfigError("Allocator needs at least one sleeve")
        names = [s.name for s in sleeves]
        if len(set(names)) != len(names):
            raise SleeveConfigError(f"Duplicate sleeve names: {names}")
        total = sum(float(s.weight) for s in sleeves)
        if total > 1.0 + _EPS:
            raise SleeveConfigError(f"Total sleeve weight {total:.4f} exceeds 1.0")
        if not 0.0 < max_gross <= 1.0:
            raise SleeveConfigError(f"max_gross must be in (0, 1], got {max_gross}")
        if band_abs < 0:
            raise SleeveConfigError(f"band_abs must be >= 0, got {band_abs}")
        self.sleeves = list(sleeves)
        self.band_abs = float(band_abs)
        self.min_order_notional = float(min_order_notional)
        self.liquidate_unmanaged = bool(liquidate_unmanaged)
        self.max_gross = float(max_gross)
        # Fraction of NAV always kept in cash: every combined target is
        # scaled by (1 - cash_buffer), so sleeve proportions are exact while
        # non-marginable buys (Alpaca crypto needs settled cash) and
        # whole-share rounding always have headroom.
        if not 0.0 <= float(cash_buffer) < 0.5:
            raise SleeveConfigError(f"cash_buffer must be in [0, 0.5), got {cash_buffer}")
        self.cash_buffer = float(cash_buffer)

    def symbols(self) -> list[str]:
        """Union of every sleeve's declared symbols (stable order)."""
        out: list[str] = []
        for s in self.sleeves:
            for sym in s.symbols():
                if sym not in out:
                    out.append(sym)
        return out

    def fractional_symbols(self) -> set[str]:
        return {
            sym for s in self.sleeves if getattr(s, "fractional", False) for sym in s.symbols()
        }

    def symbol_aliases(self) -> dict[str, str]:
        """Broker-symbol -> sleeve-symbol aliases (e.g. Alpaca reports a
        ``BTC/USD`` crypto position as ``BTCUSD``). Without this the
        allocator would see its own crypto holding as unmanaged."""
        return {sym.replace("/", ""): sym for sym in self.symbols() if "/" in sym}

    def normalize_symbols(self, mapping: dict[str, float] | None) -> dict[str, float]:
        if not mapping:
            return {}
        aliases = self.symbol_aliases()
        out: dict[str, float] = {}
        for sym, val in mapping.items():
            key = aliases.get(sym, sym)
            out[key] = out.get(key, 0.0) + float(val)
        return out

    # ------------------------------------------------------------------

    def _sleeve_targets(
        self, asof: datetime, history: dict[str, pd.Series], plan: AllocationPlan,
    ) -> tuple[dict[str, dict[str, float]], set[str]]:
        """Within-sleeve weights per sleeve (validated), plus the symbols of
        sleeves that failed to produce targets (frozen this run)."""
        result: dict[str, dict[str, float]] = {}
        frozen: set[str] = set()
        for sleeve in self.sleeves:
            try:
                raw = sleeve.target_weights(asof, history) or {}
            except Exception as exc:
                msg = f"sleeve {sleeve.name}: target_weights failed ({exc}); holding its symbols"
                log.exception("Allocation %s", msg)
                plan.errors.append(msg)
                frozen.update(sleeve.symbols())
                continue
            weights: dict[str, float] = {}
            for sym, w in raw.items():
                w = float(w)
                if not math.isfinite(w):
                    msg = f"sleeve {sleeve.name}: non-finite weight for {sym}; holding its symbols"
                    log.error("Allocation %s", msg)
                    plan.errors.append(msg)
                    frozen.update(sleeve.symbols())
                    weights = {}
                    break
                if w < 0:
                    log.warning(
                        "Allocation sleeve %s returned negative weight %.4f for %s "
                        "-- clamped to 0 (long-only)", sleeve.name, w, sym,
                    )
                    w = 0.0
                if sym not in sleeve.symbols():
                    log.warning(
                        "Allocation sleeve %s targets %s, which is not in its declared "
                        "symbols() %s", sleeve.name, sym, sleeve.symbols(),
                    )
                weights[sym] = w
            else:
                total = sum(weights.values())
                if total > 1.0 + _EPS:
                    log.warning(
                        "Allocation sleeve %s weights sum to %.4f > 1 -- scaling to 1 (no leverage)",
                        sleeve.name, total,
                    )
                    weights = {s: w / total for s, w in weights.items()}
                for sym in sleeve.symbols():
                    weights.setdefault(sym, 0.0)
                result[sleeve.name] = weights
        return result, frozen

    def plan(
        self,
        asof: datetime,
        nav: float,
        positions: dict[str, float],
        prices: dict[str, float],
        history: dict[str, pd.Series],
        last_rebalance_by_sleeve: dict[str, datetime | None],
        quantities: dict[str, float] | None = None,
    ) -> AllocationPlan:
        """Build the plan for one run.

        ``positions`` = broker market value per symbol (signed; a short is
        negative), ``quantities`` = broker share/coin quantity per symbol
        (optional; used for exact full liquidations, otherwise derived from
        market value / price), ``prices`` = current mark per symbol.
        """
        plan = AllocationPlan(asof=asof.isoformat(), nav=float(nav))
        plan.sleeve_weights = {s.name: float(s.weight) for s in self.sleeves}
        positions = self.normalize_symbols(positions)
        quantities = self.normalize_symbols(quantities) if quantities else {}
        prices = {**prices, **{k: v for k, v in self.normalize_symbols(prices).items() if k not in prices}}
        if nav <= 0 or not math.isfinite(nav):
            msg = f"non-positive NAV {nav}; no orders planned"
            log.error("Allocation %s", msg)
            plan.errors.append(msg)
            return plan

        plan.actual_weights = {
            sym: round(mv / nav, 6) for sym, mv in positions.items() if abs(mv) > _EPS
        }
        plan.gross_before = round(sum(abs(mv) for mv in positions.values()) / nav, 6)

        due: dict[str, bool] = {}
        for sleeve in self.sleeves:
            try:
                due[sleeve.name] = bool(
                    sleeve.is_rebalance_due(asof, last_rebalance_by_sleeve.get(sleeve.name))
                )
            except Exception as exc:
                msg = f"sleeve {sleeve.name}: is_rebalance_due failed ({exc}); treated as not due"
                log.exception("Allocation %s", msg)
                plan.errors.append(msg)
                due[sleeve.name] = False
        plan.due_sleeves = [n for n, d in due.items() if d]

        sleeve_targets, frozen = self._sleeve_targets(asof, history, plan)
        plan.sleeve_targets = {n: {s: round(w, 6) for s, w in t.items()} for n, t in sleeve_targets.items()}

        combined: dict[str, float] = {}
        owners: dict[str, list[str]] = {}
        for sleeve in self.sleeves:
            if sleeve.name not in sleeve_targets:
                continue
            for sym, w in sleeve_targets[sleeve.name].items():
                combined[sym] = combined.get(sym, 0.0) + sleeve.weight * w * (1.0 - self.cash_buffer)
                owners.setdefault(sym, []).append(sleeve.name)
        # A symbol shared with a sleeve that failed this run can't have a
        # trustworthy combined target -- hold it untouched.
        for sym in frozen:
            combined.pop(sym, None)
        plan.targets = {s: round(w, 6) for s, w in combined.items()}
        plan.symbol_sleeves = {s: list(o) for s, o in owners.items() if s in combined}

        managed = set(combined) | frozen
        fractional_syms = {
            sym for s in self.sleeves if getattr(s, "fractional", False)
            for sym in sleeve_targets.get(s.name, {})
        }
        tif_by_symbol: dict[str, str] = {}
        for s in self.sleeves:
            tif = getattr(s, "time_in_force", None)
            if tif:
                for sym in sleeve_targets.get(s.name, {}):
                    tif_by_symbol.setdefault(sym, tif)

        unplannable_sleeves: set[str] = set()
        sells: list[dict[str, Any]] = []
        buys: list[dict[str, Any]] = []

        def _qty_toward_zero(q: float, fractional: bool) -> float:
            if fractional:
                scale = 10 ** _FRACTIONAL_DECIMALS
                return math.floor(abs(q) * scale) / scale
            return float(math.floor(abs(q) + _EPS))

        def _current_qty(sym: str, price: float) -> float:
            if sym in quantities:
                return float(quantities[sym])
            return positions.get(sym, 0.0) / price if price > 0 else 0.0

        def _make_order(sym: str, side: str, qty: float, price: float, strategy: str,
                        fractional: bool) -> dict[str, Any]:
            order: dict[str, Any] = {
                "symbol": sym,
                "side": side,
                "quantity": qty,
                "price": price,
                "strategy": strategy,
                "order_type": "market",
                "notional": round(qty * price, 2),
                "shares": qty if side == "buy" else -qty,
            }
            if fractional:
                order["fractional"] = True
            if sym in tif_by_symbol:
                order["time_in_force"] = tif_by_symbol[sym]
            return order

        for sym in sorted(combined):
            target_w = combined[sym]
            actual_mv = positions.get(sym, 0.0)
            actual_w = actual_mv / nav
            sleeves_for_sym = owners.get(sym, [])
            is_due = any(due.get(n, False) for n in sleeves_for_sym)
            owner_objs = [s for s in self.sleeves if s.name in sleeves_for_sym]
            # A sleeve can opt out of the daily NAV-level drift trigger
            # (``drift_check = False``) when its own pre-registered rule
            # trades only at its review, e.g. the weekly BTC trend rule.
            drift_enabled = any(getattr(s, "drift_check", True) for s in owner_objs)
            drifted = drift_enabled and abs(actual_w - target_w) > self.band_abs + _EPS
            if drifted:
                plan.drift_symbols.append(sym)
            if not (is_due or drifted):
                log.debug(
                    "Allocation %s: within band (actual %.4f vs target %.4f, band %.4f) and not due",
                    sym, actual_w, target_w, self.band_abs,
                )
                continue
            if is_due and not drifted and len(owner_objs) == 1:
                # Sleeve-level trade band, in units of the sleeve's own
                # capital (``band_within``): at a review, trade only on an
                # on/off flip or when the within-sleeve gap exceeds the band
                # -- the pre-registered rule's own "trade if flips or
                # |target - held| > band".
                owner = owner_objs[0]
                band_within = getattr(owner, "band_within", None)
                sleeve_capital = owner.weight * (1.0 - self.cash_buffer)
                if band_within is not None and sleeve_capital > 0:
                    flip = (target_w > _EPS) != (actual_w > _EPS)
                    gap_within = abs(target_w - actual_w) / sleeve_capital
                    if not flip and gap_within <= float(band_within) + _EPS:
                        log.info(
                            "Allocation %s: sleeve %s review, no flip and within-sleeve gap %.4f "
                            "<= band %.4f; holding", sym, owner.name, gap_within, band_within,
                        )
                        continue
            price = float(prices.get(sym) or 0.0)
            if price <= 0 or not math.isfinite(price):
                msg = f"{sym}: no usable price; cannot trade it this run"
                log.error("Allocation %s", msg)
                plan.errors.append(msg)
                unplannable_sleeves.update(sleeves_for_sym)
                continue
            fractional = sym in fractional_syms
            strategy = max(
                sleeves_for_sym,
                key=lambda n: next(s.weight for s in self.sleeves if s.name == n)
                * sleeve_targets[n].get(sym, 0.0),
            ) if sleeves_for_sym else UNMANAGED_STRATEGY
            cur_qty = _current_qty(sym, price)
            target_mv = max(0.0, target_w) * nav
            delta_mv = target_mv - actual_mv
            if target_w <= _EPS and cur_qty > 0:
                # Full exit: sell the exact held quantity, not a rounded estimate.
                sells.append(_make_order(sym, "sell", abs(cur_qty), price, strategy, fractional))
                continue
            if delta_mv < 0:
                if cur_qty <= 0:
                    continue  # never sell into a short
                qty = min(_qty_toward_zero(delta_mv / price, fractional), cur_qty)
                if qty <= 0 or qty * price < self.min_order_notional:
                    log.debug("Allocation %s: sell %.6f below min notional -- skipped", sym, qty)
                    continue
                sells.append(_make_order(sym, "sell", qty, price, strategy, fractional))
            elif delta_mv > 0:
                qty = _qty_toward_zero(delta_mv / price, fractional)
                if qty <= 0 or qty * price < self.min_order_notional:
                    log.debug("Allocation %s: buy %.6f below min notional -- skipped", sym, qty)
                    continue
                buys.append(_make_order(sym, "buy", qty, price, strategy, fractional))

        for sym, mv in positions.items():
            if sym in managed or abs(mv) <= _EPS:
                continue
            plan.unmanaged[sym] = round(mv / nav, 6)
            if not self.liquidate_unmanaged:
                log.info(
                    "Allocation: leaving unmanaged position %s (%.2f%% of NAV) untouched "
                    "(liquidate_unmanaged=false)", sym, 100 * mv / nav,
                )
                continue
            price = float(prices.get(sym) or 0.0)
            cur_qty = float(quantities.get(sym, mv / price if price > 0 else 0.0))
            if price <= 0 or cur_qty == 0:
                msg = f"{sym}: unmanaged position has no usable price/quantity; not liquidated"
                log.error("Allocation %s", msg)
                plan.errors.append(msg)
                continue
            # A short is closed with a buy-to-cover (reduces exposure, never
            # opens a new short); a long with a plain sell.
            side = "sell" if cur_qty > 0 else "buy"
            log.warning(
                "Allocation: liquidating unmanaged position %s (%s %.6f, %.2f%% of NAV)",
                sym, side, abs(cur_qty), 100 * mv / nav,
            )
            sells.append(_make_order(sym, side, abs(cur_qty), price, UNMANAGED_STRATEGY,
                                     fractional=abs(cur_qty) != math.floor(abs(cur_qty))))

        # Projected gross after the plan; scale buys down if it would exceed
        # max_gross x NAV (unmanaged positions left in place count too).
        projected = dict(positions)
        for o in sells + buys:
            signed = o["quantity"] * o["price"] * (1 if o["side"] == "buy" else -1)
            projected[o["symbol"]] = projected.get(o["symbol"], 0.0) + signed
        gross_after = sum(abs(v) for v in projected.values())
        limit = self.max_gross * nav
        if gross_after > limit + 1e-6 and buys:
            buy_notional = sum(_order_notional(o) for o in buys)
            excess = gross_after - limit
            scale = max(0.0, (buy_notional - excess) / buy_notional) if buy_notional > 0 else 0.0
            # Expected, small overshoot: positions sitting overweight inside
            # their drift band consume the cash a due buy would need (e.g. a
            # weekly BTC buy mid-month); buys wait for the next rebalance.
            # Only a larger overshoot is worth a warning.
            level = logging.INFO if excess <= self.band_abs * nav + 1e-6 else logging.WARNING
            log.log(
                level,
                "Allocation: projected gross %.2f%% exceeds cap %.2f%% -- scaling buys by %.4f",
                100 * gross_after / nav, 100 * self.max_gross, scale,
            )
            plan.buy_scale = round(scale, 6)
            scaled: list[dict[str, Any]] = []
            for o in buys:
                qty = _qty_toward_zero(o["quantity"] * scale, bool(o.get("fractional")))
                if qty <= 0 or qty * o["price"] < self.min_order_notional:
                    continue
                scaled.append(_make_order(o["symbol"], "buy", qty, o["price"], o["strategy"],
                                          bool(o.get("fractional"))))
            buys = scaled
            projected = dict(positions)
            for o in sells + buys:
                signed = o["quantity"] * o["price"] * (1 if o["side"] == "buy" else -1)
                projected[o["symbol"]] = projected.get(o["symbol"], 0.0) + signed
            gross_after = sum(abs(v) for v in projected.values())
        plan.gross_after = round(gross_after / nav, 6)

        sells.sort(key=_order_notional, reverse=True)
        buys.sort(key=_order_notional, reverse=True)
        plan.orders = sells + buys
        plan.rebalanced_sleeves = [
            n for n in plan.due_sleeves
            if n in sleeve_targets and n not in unplannable_sleeves
        ]
        log.info(
            "Allocation plan: nav=%.2f due=%s drift=%s orders=%d (sells=%d buys=%d) "
            "gross %.4f -> %.4f unmanaged=%d errors=%d",
            nav, plan.due_sleeves, plan.drift_symbols, len(plan.orders), len(sells), len(buys),
            plan.gross_before, plan.gross_after, len(plan.unmanaged), len(plan.errors),
        )
        return plan


def build_allocator(allocation_cfg: dict[str, Any]) -> Allocator:
    """Build sleeves + allocator from an ``allocation`` config block."""
    from firm.allocation.sleeves import build_sleeves

    sleeves = build_sleeves(allocation_cfg)
    return Allocator(
        sleeves,
        band_abs=float(allocation_cfg.get("band_abs", 0.02)),
        min_order_notional=float(allocation_cfg.get("min_order_notional", 100.0)),
        liquidate_unmanaged=bool(allocation_cfg.get("liquidate_unmanaged", False)),
        max_gross=float(allocation_cfg.get("max_gross", 1.0)),
        cash_buffer=float(allocation_cfg.get("cash_buffer", 0.0)),
    )
