"""Adaptive per-sleeve capital reweighting -- the recommendation math.

``capital_allocation_mode: "sleeved"`` (config/live_alpaca.yaml) splits
capital across strategies via ``strategy_capital_weights`` -- a fixed
split set once at cutover and never revisited, so a sleeve that has
quietly underperformed for weeks keeps exactly the same capital as one
that has been compounding well. This module computes what that split
*should* become given each sleeve's recent risk-adjusted performance.

Deliberately does no Sharpe/Sortino math of its own -- it consumes
``Orchestrator.get_sleeve_metrics()``, which already computes those
consistently with the rest of the system (``firm.eval.metrics``, the same
functions ``GET /api/live/attribution`` and every backtest report use).
Reusing that dict rather than recomputing ratios here means this module's
numbers can never quietly disagree with what the rest of the system
already reports for the same sleeve.

Pure functions only -- nothing here touches a running engine or mutates
capital. See ``firm.live.capital_reallocation_job`` for the read-only
scheduled check and ``Orchestrator.apply_capital_reallocation`` for the
explicit, human-triggered action that actually moves capital.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Sortino (not Sharpe) by default: it already penalizes downside deviation
# only, which is a more direct read of "how painful was this sleeve's bad
# stretch" than Sharpe's symmetric variance -- appropriate for a mechanism
# whose whole point is reacting to underperformance.
DEFAULT_METRIC = "sortino_ratio"
# No sleeve loses more than 97% of its capital from one bad stretch, and
# none can capture more than 35% of the book from one good one -- both
# deliberately conservative so a single noisy window can only nudge the
# split, never swing it to an extreme. Calibrate before ever enabling live.
DEFAULT_FLOOR_PCT = 0.03
DEFAULT_CAP_PCT = 0.35
# A sleeve with fewer than 30 daily observations keeps its current weight
# untouched -- a Sharpe/Sortino computed from a handful of days is mostly
# noise, and reweighting on noise is worse than not reweighting at all.
DEFAULT_MIN_TRACK_DAYS = 30
# How aggressively a sleeve's weight moves per standard deviation of
# relative performance -- 0 reproduces the current split exactly (a
# sanity/no-op check), 1.0 would swing a +-1 sigma sleeve's share by 100%
# of its starting share before floor/cap clip it back down.
DEFAULT_TILT_STRENGTH = 0.25

_WATERFILL_MAX_ITER = 50


def compute_reallocated_weights(
    sleeve_metrics: dict[str, dict[str, float]],
    current_weights: dict[str, float],
    *,
    metric: str = DEFAULT_METRIC,
    floor_pct: float = DEFAULT_FLOOR_PCT,
    cap_pct: float = DEFAULT_CAP_PCT,
    min_track_days: int = DEFAULT_MIN_TRACK_DAYS,
    tilt_strength: float = DEFAULT_TILT_STRENGTH,
) -> dict[str, dict[str, Any]]:
    """Propose a new capital split, tilted toward better trailing *metric*.

    *sleeve_metrics* is ``Orchestrator.get_sleeve_metrics()``'s output;
    *current_weights* is ``Orchestrator.sleeve_capital_weights()``'s
    output (every sleeve's current target share, summing to ~1.0).

    Strategies with fewer than *min_track_days* observations (or missing
    *metric* entirely, e.g. never traded) are frozen at their current
    weight -- untouched, not reweighted to zero. Every other ("eligible")
    strategy's weight is:

    1. Its current share of the *adjustable* budget (1.0 minus whatever
       the frozen strategies already hold).
    2. Tilted by its z-score on *metric* against its eligible peers:
       ``share * (1 + tilt_strength * z)`` -- relative to the peer group,
       not an absolute threshold, so the mechanism reacts to one sleeve
       genuinely out/under-performing its peers rather than to whether
       markets were good or bad this month for everyone at once.
    3. Clipped to ``[floor_pct, cap_pct]`` and renormalized to exactly
       fill the adjustable budget via iterative water-filling (see
       :func:`_waterfill`) -- a single clip-then-scale pass can push a
       newly-boosted strategy back over its cap.

    Returns ``{strategy: {"current_weight", "new_weight", "score",
    "n_days", "reason", ["z_score"]}}`` -- deliberately more than just the
    weight map (see :func:`weights_only` for that), since the per-strategy
    rationale is exactly what a human needs to see before trusting this
    to move real capital.
    """
    strategies = list(current_weights)
    if not strategies:
        return {}

    result: dict[str, dict[str, Any]] = {}
    eligible: list[str] = []
    for strategy in strategies:
        m = sleeve_metrics.get(strategy) or {}
        n_days = int(m.get("n_days", 0))
        base_weight = float(current_weights.get(strategy, 0.0))
        if n_days < min_track_days or metric not in m:
            result[strategy] = {
                "current_weight": base_weight,
                "new_weight": base_weight,
                "score": m.get(metric),
                "n_days": n_days,
                "reason": "insufficient_history",
            }
        else:
            eligible.append(strategy)
            result[strategy] = {
                "current_weight": base_weight,
                "score": float(m[metric]),
                "n_days": n_days,
                "reason": "reweighted",
            }

    frozen_total = sum(result[s]["current_weight"] for s in strategies if s not in eligible)
    adjustable_budget = max(0.0, 1.0 - frozen_total)

    if not eligible or adjustable_budget <= 1e-9:
        # Nothing eligible, or the frozen (insufficient-history) sleeves
        # already account for the whole book -- leave every eligible
        # strategy's weight exactly where it is rather than reallocate a
        # near-zero or negative budget.
        for s in eligible:
            result[s]["new_weight"] = result[s]["current_weight"]
            result[s]["reason"] = "no_adjustable_budget"
        return _finalize(result, strategies)

    eligible_current_total = sum(result[s]["current_weight"] for s in eligible)
    if eligible_current_total > 1e-9:
        base_shares = {s: result[s]["current_weight"] / eligible_current_total for s in eligible}
    else:
        base_shares = {s: 1.0 / len(eligible) for s in eligible}

    scores = [result[s]["score"] for s in eligible]
    mean_score = sum(scores) / len(scores)
    variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
    std_score = variance**0.5

    tilted: dict[str, float] = {}
    for s in eligible:
        z = (result[s]["score"] - mean_score) / std_score if std_score > 1e-9 else 0.0
        result[s]["z_score"] = z
        # Floored well above zero (not just >=0): a large negative tilt
        # must shrink a sleeve's share toward the water-fill's floor
        # clamp, never toward a literal zero input weight that would make
        # the proportional water-fill step below divide-by-zero-adjacent.
        tilted[s] = max(base_shares[s] * (1.0 + tilt_strength * z), 1e-9)

    new_weights = _waterfill(tilted, adjustable_budget, floor_pct, cap_pct)
    for s in eligible:
        result[s]["new_weight"] = new_weights[s]

    return _finalize(result, strategies)


def _waterfill(
    tilted: dict[str, float],
    budget: float,
    floor_pct: float,
    cap_pct: float,
) -> dict[str, float]:
    """Distribute *budget* proportional to *tilted*, clipped to
    ``[floor_pct, cap_pct]`` absolute share of the whole book.

    Standard bounded-proportional-allocation technique: repeatedly pin
    whichever strategies' proportional share would breach a bound at that
    bound, then redistribute the remaining budget proportionally among
    the still-unpinned strategies, until nothing new gets pinned. A
    single clip-then-renormalize pass can't do this in one shot -- pinning
    one strategy at its cap changes every other strategy's proportional
    share of what's left, which can push a previously-fine strategy over
    its own bound too.
    """
    n = len(tilted)
    if floor_pct * n > budget + 1e-9 or cap_pct * n < budget - 1e-9:
        # floor/cap can't simultaneously hold for this many eligible
        # sleeves and this budget -- equal split is the only allocation
        # that doesn't privilege an arbitrary tie-break among them.
        log.warning(
            "capital_reallocation: floor=%.3f/cap=%.3f infeasible for %d eligible "
            "sleeve(s) and budget %.4f -- falling back to an equal split",
            floor_pct, cap_pct, n, budget,
        )
        return {s: budget / n for s in tilted}

    weights = dict(tilted)
    pinned: dict[str, float] = {}
    remaining = set(tilted)

    for _ in range(_WATERFILL_MAX_ITER):
        if not remaining:
            break
        free_budget = budget - sum(pinned.values())
        total_free = sum(weights[s] for s in remaining)
        if total_free <= 1e-12 or free_budget <= 1e-12:
            share = free_budget / len(remaining)
            for s in remaining:
                pinned[s] = share
            remaining.clear()
            break

        newly_pinned = []
        for s in remaining:
            proportional = free_budget * weights[s] / total_free
            if proportional < floor_pct:
                pinned[s] = floor_pct
                newly_pinned.append(s)
            elif proportional > cap_pct:
                pinned[s] = cap_pct
                newly_pinned.append(s)
        if not newly_pinned:
            for s in remaining:
                pinned[s] = free_budget * weights[s] / total_free
            remaining.clear()
            break
        remaining.difference_update(newly_pinned)
    else:
        # Shouldn't happen (each iteration pins at least one more
        # strategy, so this terminates within `n` passes) -- split
        # whatever's left equally rather than loop forever.
        if remaining:
            free_budget = budget - sum(pinned.values())
            share = free_budget / len(remaining)
            for s in remaining:
                pinned[s] = share

    return pinned


def _finalize(
    result: dict[str, dict[str, Any]], strategies: list[str],
) -> dict[str, dict[str, Any]]:
    """Defensively renormalize so ``new_weight`` sums to exactly 1.0.

    The water-fill above already targets the adjustable budget exactly,
    but this feeds real capital math downstream (``Orchestrator.
    apply_capital_reallocation``) -- worth a final exact-sum guarantee
    rather than trusting accumulated float error to stay negligible.
    """
    total = sum(result[s]["new_weight"] for s in strategies)
    if total > 1e-9 and abs(total - 1.0) > 1e-9:
        for s in strategies:
            result[s]["new_weight"] = result[s]["new_weight"] / total
    return result


def weights_only(result: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Extract just the ``{strategy: new_weight}`` map from a recommendation."""
    return {s: info["new_weight"] for s, info in result.items()}
