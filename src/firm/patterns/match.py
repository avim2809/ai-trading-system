"""The detection result shared by every pattern-rule module.

Rule functions (in :mod:`firm.patterns.rules`) populate the *detection*
fields (pattern/direction/pivots/entry/stop/target/fit_quality/
geometry_tolerance_used) from pure price geometry. :mod:`firm.patterns.scanner`
then fills in the *scoring* fields (volume_ratio, duration_bars,
follow_through_atr, quality_score, score_breakdown) uniformly across all
pattern types via :func:`dataclasses.replace`, so every rule module is scored
by the same yardstick instead of inventing its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from firm.patterns.extrema import Pivot


@dataclass(frozen=True)
class PatternMatch:
    pattern: str  # e.g. "head_shoulders_top", "cup_handle"
    direction: str  # "long" | "short"
    pivots: tuple[Pivot, ...]  # structural extrema, oldest first
    confirm_index: int  # bar index (into the scanned window) of the breakout close
    entry: float  # breakout/confirmation level
    stop: float  # structural stop (pre-ATR-floor; scanner may widen it)
    target: float  # measured-move target
    fit_quality: float  # 0-1: R^2 of the neckline/trendline/poly fit
    geometry_tolerance_used: float  # 0-1: 1.0 = perfectly on-tolerance, 0 = at the tolerance limit
    meta: dict[str, Any] = field(default_factory=dict)

    # --- filled in by scanner.py after detection ---
    volume_ratio: float = float("nan")
    duration_bars: int = 0
    follow_through_atr: float = 0.0
    risk_reward: float = 0.0
    quality_score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        return self.confirm_index >= 0
