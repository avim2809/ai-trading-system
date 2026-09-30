"""Shared ``Sleeve`` interface for capital-sleeve allocations.

Minimal stub, created ahead of the allocation-engine work being built in
parallel on this same interface (see ``src/firm/allocation/btc_trend.py``'s
``BtcTrendSleeve`` for a concrete implementation). This file intentionally
contains nothing beyond the agreed ABC below -- the integrator should
replace/extend it with the real allocation-engine plumbing (sleeve registry,
NAV bookkeeping, order generation, etc.) without needing to touch
``BtcTrendSleeve`` or its tests, which only depend on this exact interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class Sleeve(ABC):
    """A single satellite capital sleeve: controls ``weight`` fraction of
    total NAV, allocated across its own ``symbols()`` (long-only, remainder
    in cash)."""

    name: str
    weight: float  # fraction of total NAV the sleeve controls

    @abstractmethod
    def target_weights(self, asof: datetime, history: dict[str, pd.Series]) -> dict[str, float]:
        """Weights WITHIN the sleeve (sum <= 1, long-only; remainder is cash).

        ``history`` = daily closes per symbol, completed bars only, up to asof.
        """

    @abstractmethod
    def is_rebalance_due(self, asof: datetime, last_rebalance: datetime | None) -> bool:
        ...

    def symbols(self) -> list[str]:
        ...
