"""Allocation portfolio (``strategy_mode: allocation``): passive core + satellite sleeves.

See :mod:`firm.allocation.sleeves` (the :class:`Sleeve` interface, the
built-in :class:`StaticSleeve` and the pluggable :func:`build_sleeves`
registry) and :mod:`firm.allocation.allocator` (combining sleeve targets
into one long-only book and planning the orders). The live-engine wiring
lives in ``firm.live.engine.LiveTradingEngine._run_allocation_cycle``.
"""

from firm.allocation.allocator import AllocationPlan, Allocator, build_allocator
from firm.allocation.sleeves import (
    SLEEVE_REGISTRY,
    Sleeve,
    SleeveConfigError,
    StaticSleeve,
    build_sleeves,
    is_calendar_rebalance_due,
)

__all__ = [
    "SLEEVE_REGISTRY",
    "AllocationPlan",
    "Allocator",
    "Sleeve",
    "SleeveConfigError",
    "StaticSleeve",
    "build_allocator",
    "build_sleeves",
    "is_calendar_rebalance_due",
]
