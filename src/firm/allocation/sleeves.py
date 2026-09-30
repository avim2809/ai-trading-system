"""Allocation sleeves: independent slices of NAV with their own target weights.

A :class:`Sleeve` controls a fixed fraction (``weight``) of total NAV and
decides, on its own schedule, how that fraction is spread across its
symbols (long-only, the rest of the sleeve sits in cash). The
:class:`~firm.allocation.allocator.Allocator` combines every sleeve's
targets into one book and turns the difference against the broker's real
positions into orders.

Sleeve types are pluggable through :func:`build_sleeves`: each entry of the
``allocation.sleeves`` config list carries a ``type:`` key. ``static`` is
built in; other types are imported lazily from their own module, so a
missing optional module fails loudly at engine start instead of silently
dropping that sleeve's capital into cash.
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd

from firm.allocation.calendar import is_us_trading_day, to_market_date

log = logging.getLogger(__name__)

REBALANCE_FREQUENCIES = ("monthly", "weekly", "daily")


class SleeveConfigError(ValueError):
    """Raised when the allocation config cannot be turned into sleeves."""


class Sleeve(ABC):
    """One independently-scheduled slice of the allocation portfolio.

    Subclasses set ``name`` and ``weight`` (fraction of total NAV this sleeve
    controls). Two optional class/instance attributes shape the orders the
    allocator emits for this sleeve's symbols:

    * ``fractional`` -- ``True`` lets orders carry fractional quantities
      (crypto). Default ``False``: whole shares only, same as the pipeline.
    * ``time_in_force`` -- e.g. ``"gtc"`` for Alpaca crypto, which rejects
      ``day`` orders. ``None`` keeps the broker adapter's default (``day``).
    """

    name: str
    weight: float
    fractional: bool = False
    time_in_force: str | None = None

    @abstractmethod
    def target_weights(self, asof: datetime, history: dict[str, pd.Series]) -> dict[str, float]:
        """Weights WITHIN the sleeve (sum <= 1, long-only, rest is cash).

        ``history`` = daily closes (adjusted) per symbol, completed bars
        only, up to ``asof``.
        """

    @abstractmethod
    def is_rebalance_due(self, asof: datetime, last_rebalance: datetime | None) -> bool:
        """True when this sleeve should trade back to its targets now."""

    def symbols(self) -> list[str]:
        """Every symbol this sleeve may hold (used for history + allowlist)."""
        raise NotImplementedError(
            f"Sleeve {getattr(self, 'name', type(self).__name__)!r} must implement symbols()"
        )


def is_calendar_rebalance_due(
    frequency: str,
    asof: datetime,
    last_rebalance: datetime | None,
    tz: str = "US/Eastern",
) -> bool:
    """Shared calendar rule for monthly/weekly/daily sleeves.

    Only ever due on a US trading day (``asof`` in US/Eastern). ``monthly``
    is due on the first trading day of a calendar month that differs from
    ``last_rebalance``'s -- and, if that day was missed (outage, halted
    engine), on every later trading day of the month until it succeeds, so a
    missed first day never skips a whole month. ``weekly`` compares ISO
    weeks the same way; ``daily`` compares dates. ``last_rebalance=None``
    (never rebalanced) is always due on a trading day.
    """
    today = to_market_date(asof, tz)
    if not is_us_trading_day(today):
        return False
    if last_rebalance is None:
        return True
    last = to_market_date(last_rebalance, tz)
    if frequency == "monthly":
        return (today.year, today.month) != (last.year, last.month)
    if frequency == "weekly":
        return today.isocalendar()[:2] != last.isocalendar()[:2]
    if frequency == "daily":
        return today != last
    raise SleeveConfigError(f"Unknown rebalance frequency {frequency!r}")


class StaticSleeve(Sleeve):
    """Fixed within-sleeve weights (e.g. 60% SPY / 40% IEF), calendar rebalanced."""

    def __init__(
        self,
        name: str,
        weight: float,
        weights: dict[str, float],
        rebalance: str = "monthly",
    ) -> None:
        if rebalance not in REBALANCE_FREQUENCIES:
            raise SleeveConfigError(
                f"Sleeve {name!r}: rebalance must be one of {REBALANCE_FREQUENCIES}, got {rebalance!r}"
            )
        if not weights:
            raise SleeveConfigError(f"Sleeve {name!r}: weights must not be empty")
        cleaned = {str(s): float(w) for s, w in weights.items()}
        if any(w < 0 for w in cleaned.values()):
            raise SleeveConfigError(f"Sleeve {name!r}: negative weights are not allowed (long-only)")
        total = sum(cleaned.values())
        if total > 1.0 + 1e-9:
            raise SleeveConfigError(
                f"Sleeve {name!r}: within-sleeve weights sum to {total:.4f} > 1.0"
            )
        self.name = name
        self.weight = float(weight)
        self.rebalance = rebalance
        self._weights = cleaned

    def target_weights(self, asof: datetime, history: dict[str, pd.Series]) -> dict[str, float]:
        return dict(self._weights)

    def is_rebalance_due(self, asof: datetime, last_rebalance: datetime | None) -> bool:
        return is_calendar_rebalance_due(self.rebalance, asof, last_rebalance)

    def symbols(self) -> list[str]:
        return list(self._weights)

    def __repr__(self) -> str:
        return (
            f"StaticSleeve(name={self.name!r}, weight={self.weight}, "
            f"weights={self._weights}, rebalance={self.rebalance!r})"
        )


# type -> "module:attribute". Imported lazily so an optional sleeve module
# (built separately, possibly not present yet) never breaks import of this
# package -- only a config that actually asks for it.
SLEEVE_REGISTRY: dict[str, str] = {
    "static": "firm.allocation.sleeves:StaticSleeve",
    "btc_trend": "firm.allocation.btc_trend:BtcTrendSleeve",
}


def _resolve_sleeve_class(type_name: str) -> type:
    target = SLEEVE_REGISTRY.get(type_name)
    if target is None:
        raise SleeveConfigError(
            f"Unknown sleeve type {type_name!r}; known types: {sorted(SLEEVE_REGISTRY)}"
        )
    module_name, _, attr = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        log.error(
            "Allocation sleeve type %r needs module %s, which could not be imported (%s) "
            "-- refusing to start rather than leaving that sleeve's capital in cash",
            type_name, module_name, exc,
        )
        raise SleeveConfigError(
            f"Sleeve type {type_name!r}: module {module_name} is not available ({exc})"
        ) from exc
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        log.error("Allocation sleeve module %s has no class %s", module_name, attr)
        raise SleeveConfigError(
            f"Sleeve type {type_name!r}: {module_name} has no attribute {attr}"
        ) from exc


def build_sleeves(config: dict[str, Any]) -> list[Sleeve]:
    """Build sleeves from an ``allocation`` config block.

    Accepts either the ``allocation`` block itself (``{"sleeves": [...]}``)
    or the list directly. Each sleeve entry needs ``name``, ``type`` and
    ``weight``; every other key is passed to the sleeve class as a keyword
    argument (``weights``/``rebalance`` for ``static``). Validates unique
    names, weights in [0, 1] and a total sleeve weight <= 1.0 (the
    remainder is cash). Raises :class:`SleeveConfigError` on any problem --
    callers (engine start) should let it propagate.
    """
    entries = config.get("sleeves") if isinstance(config, dict) else config
    if not entries:
        raise SleeveConfigError("allocation.sleeves is empty -- nothing to allocate")
    sleeves: list[Sleeve] = []
    seen: set[str] = set()
    for i, raw in enumerate(entries):
        if not isinstance(raw, dict):
            raise SleeveConfigError(f"allocation.sleeves[{i}] must be a mapping, got {type(raw).__name__}")
        entry = dict(raw)
        type_name = str(entry.pop("type", "static"))
        name = entry.pop("name", None)
        if not name:
            raise SleeveConfigError(f"allocation.sleeves[{i}] is missing 'name'")
        if name in seen:
            raise SleeveConfigError(f"Duplicate sleeve name {name!r}")
        if "weight" not in entry:
            raise SleeveConfigError(f"Sleeve {name!r} is missing 'weight'")
        weight = float(entry.pop("weight"))
        if not 0.0 <= weight <= 1.0:
            raise SleeveConfigError(f"Sleeve {name!r}: weight {weight} outside [0, 1]")
        cls = _resolve_sleeve_class(type_name)
        try:
            sleeve = cls(name=name, weight=weight, **entry)
        except SleeveConfigError:
            raise
        except TypeError as exc:
            raise SleeveConfigError(f"Sleeve {name!r} ({type_name}): bad parameters ({exc})") from exc
        # Fail at build time, not on the first trading cycle, if a sleeve
        # can't even say which symbols it trades.
        syms = sleeve.symbols()
        if not syms:
            raise SleeveConfigError(f"Sleeve {name!r} declares no symbols")
        seen.add(name)
        sleeves.append(sleeve)
        log.info(
            "Allocation sleeve built: %s (type=%s, weight=%.4f, symbols=%s, fractional=%s)",
            name, type_name, weight, syms, bool(getattr(sleeve, "fractional", False)),
        )
    total = sum(s.weight for s in sleeves)
    if total > 1.0 + 1e-9:
        raise SleeveConfigError(f"Total sleeve weight {total:.4f} exceeds 1.0")
    log.info("Allocation: %d sleeve(s), total weight %.4f (cash remainder %.4f)",
             len(sleeves), total, max(0.0, 1.0 - total))
    return sleeves
