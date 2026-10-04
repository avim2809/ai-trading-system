"""Strategy registry – discover and instantiate strategies by name.

Usage::

    from firm.strategies.registry import register, get

    @register("momentum")
    class MomentumStrategy(BaseStrategy): ...

    cls = get("momentum")
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from firm.strategies.base import BaseStrategy

_REGISTRY: dict[str, type[BaseStrategy]] = {}


def register(name: str):
    """Class decorator that registers a strategy under *name*."""

    def wrapper(cls: type[BaseStrategy]) -> type[BaseStrategy]:
        _REGISTRY[name] = cls
        return cls

    return wrapper


def get(name: str) -> type[BaseStrategy]:
    """Look up a registered strategy class by name."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name]


def list_strategies() -> list[str]:
    """Return names of all registered strategies."""
    return list(_REGISTRY)


# ---------------------------------------------------------------------------
# Lifecycle status sidecar (P0-03).  Advisory metadata only: list_strategies()
# and get() are unchanged.  See docs/DEPRECATIONS.md for evidence per row.
# ---------------------------------------------------------------------------


class StrategyStatus(str, Enum):
    ACTIVE = "active"
    LEGACY_LIVE = "legacy_live"  # still enabled on the IBKR control; warn-only
    ARCHIVED = "archived"
    ARCHIVED_BENCHMARKS = "archived_benchmarks"
    DECOMMISSIONED = "decommissioned"


class StrategyNotAllocatableError(RuntimeError):
    """Raised by research/backtest entry points for archived/decommissioned names."""


_S = StrategyStatus

# Names absent from this table are ACTIVE (e.g. dummy strategies in tests).
_STATUS: dict[str, StrategyStatus] = {
    # Enabled on the IBKR :8000 pipeline (kept as the blended control, OD-03).
    "momentum": _S.LEGACY_LIVE,
    "trend": _S.LEGACY_LIVE,
    "mean_reversion": _S.LEGACY_LIVE,
    "stat_arb": _S.LEGACY_LIVE,
    "multi_factor": _S.LEGACY_LIVE,
    "sentiment": _S.LEGACY_LIVE,
    "event_driven": _S.LEGACY_LIVE,
    "volatility_breakout": _S.LEGACY_LIVE,
    "seasonality": _S.LEGACY_LIVE,
    "regime_hmm": _S.LEGACY_LIVE,
    "pattern_recognition": _S.LEGACY_LIVE,
    # Already off everywhere.
    "gann": _S.DECOMMISSIONED,
    "ml_prediction": _S.ARCHIVED,
    "danelfin_ai_score": _S.ARCHIVED,
    "danelfin_best_stocks_signal": _S.ARCHIVED,
    "danelfin_live_signals": _S.ARCHIVED,
    "danelfin_market_percentile": _S.ARCHIVED,
    "insider_cluster": _S.ARCHIVED,
    "investing_analyst_ratings": _S.ARCHIVED,
}

# Recorded intent for LEGACY_LIVE names; takes effect only via P0-07 (OD-03).
_TARGET_STATUS: dict[str, StrategyStatus] = {
    "seasonality": _S.DECOMMISSIONED,
    "pattern_recognition": _S.DECOMMISSIONED,
    "mean_reversion": _S.ARCHIVED,
    "event_driven": _S.ARCHIVED,
    "sentiment": _S.ARCHIVED,
    "regime_hmm": _S.ARCHIVED,
    "volatility_breakout": _S.ARCHIVED,
    "stat_arb": _S.ARCHIVED,
    "momentum": _S.ARCHIVED_BENCHMARKS,
    "trend": _S.ARCHIVED_BENCHMARKS,
    "multi_factor": _S.ARCHIVED_BENCHMARKS,
}

_ALLOCATABLE = (_S.ACTIVE, _S.LEGACY_LIVE)


def status(name: str) -> StrategyStatus:
    """Lifecycle status of *name*; unknown names are ACTIVE."""
    return _STATUS.get(name, _S.ACTIVE)


def target_status(name: str) -> StrategyStatus | None:
    """Recorded retirement target for a LEGACY_LIVE name, else None."""
    return _TARGET_STATUS.get(name)


def list_allocatable() -> list[str]:
    """Registered names whose status is ACTIVE or LEGACY_LIVE."""
    return [n for n in _REGISTRY if status(n) in _ALLOCATABLE]


def require_allocatable(name: str, *, allow_archived: bool = False) -> None:
    """Refuse archived/decommissioned strategies (research/backtest entry points only)."""
    st = status(name)
    if st in _ALLOCATABLE or allow_archived:
        return
    raise StrategyNotAllocatableError(
        f"Strategy '{name}' has status '{st.value}' and may not be used in new research "
        "(see docs/DEPRECATIONS.md); pass allow_archived=True for legacy replays."
    )
