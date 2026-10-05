"""Lifecycle-gated wrapper around an allocation ``Sleeve`` (ticket P5-01; dry-run only, not wired into the live allocator).

``LifecycleGatedSleeve`` returns no targets unless ``target_weight_allowed`` says the family may carry weight on this account
type. It is NOT registered in the live ``SLEEVE_REGISTRY`` (that is P6-01); the H5 test builds ``Allocator([...])`` directly.
"Sleeve" here is the repo's capital-bucket class; lifecycle states apply to strategy families.

The wrapper forwards every attribute the Allocator reads via ``getattr`` (``name``, ``weight``, ``fractional``,
``time_in_force``, ``drift_check``, ``band_within``) so a gated sleeve plans exactly like the bare one; an attribute the inner
sleeve lacks stays absent (``__getattr__`` raises ``AttributeError``), preserving the Allocator's ``getattr`` defaults.

Fail-closed: an unknown family, an unreadable registry or override store yields ``{}`` and a logged error. Note that the Allocator
zero-fills every declared symbol of a sleeve that returns ``{}``, so a gated-out family is planned DOWN to zero on its symbols
(it sells existing holdings at its next due/drift trigger); pre-PAPER families hold nothing, so this only matters for
PROBATION+zero_weight, DECOMMISSIONED and live-account PAPER cases.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from firm.allocation.sleeves import Sleeve
from firm.lifecycle.decommission import zero_weight_active
from firm.lifecycle.state_machine import load_registry, target_weight_allowed

log = logging.getLogger(__name__)

__all__ = ["LifecycleGatedSleeve"]


class LifecycleGatedSleeve(Sleeve):
    def __init__(
        self,
        inner: Sleeve,
        family: str,
        registry_path: Path,
        overrides_path: Path,
        account_type: Literal["paper", "live"],
        now_fn: Callable[[], datetime],
    ) -> None:
        self._inner = inner
        self.family = family
        self._registry_path = Path(registry_path)
        self._overrides_path = Path(overrides_path)
        self.account_type = account_type
        self._now_fn = now_fn

    # Forwarded explicitly because the ABC defines class-level defaults that would otherwise shadow the inner values.
    @property
    def name(self) -> str:  # type: ignore[override]
        return self._inner.name

    @property
    def weight(self) -> float:  # type: ignore[override]
        return self._inner.weight

    @property
    def fractional(self) -> bool:  # type: ignore[override]
        return self._inner.fractional

    @property
    def time_in_force(self) -> str | None:  # type: ignore[override]
        return self._inner.time_in_force

    def __getattr__(self, item: str) -> Any:
        if item.startswith("__") or item == "_inner":
            raise AttributeError(item)
        return getattr(self._inner, item)  # drift_check, band_within, ... (AttributeError if absent)

    def _allowed(self) -> bool:
        try:
            reg = load_registry(self._registry_path)
            rec = reg.get(self.family)
            zero = zero_weight_active(self._overrides_path, reg.approvals_dir, self.family)
        except Exception:
            log.exception("lifecycle gate for %s failed closed (registry/override unreadable)", self.family)
            return False
        ok = target_weight_allowed(rec, account_type=self.account_type, zero_weight=zero)
        if not ok:
            log.warning(
                "lifecycle gate: %s in state %s (zero_weight=%s, account=%s) gets no target weight",
                self.family,
                rec.state.value,
                zero,
                self.account_type,
            )
        return ok

    def target_weights(self, asof: datetime, history: dict[str, pd.Series]) -> dict[str, float]:
        if not self._allowed():
            return {}
        return self._inner.target_weights(asof, history)

    def is_rebalance_due(self, asof: datetime, last_rebalance: datetime | None) -> bool:
        return self._inner.is_rebalance_due(asof, last_rebalance)

    def symbols(self) -> list[str]:
        return self._inner.symbols()
