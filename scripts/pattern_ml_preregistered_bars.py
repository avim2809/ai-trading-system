#!/usr/bin/env python
"""Pre-registered pass/fail bars for the Workstream D honest ML evaluation.

``docs/pattern_recognition_plan.md`` Workstream D ("Two things must be true
for this to be an honest test rather than another re-roll"): the numbers
below must be fixed BEFORE the real walk-forward run executes, and never
edited afterward to make a result pass. This module exists ONLY to hold
those numbers plus a fingerprint of them, so any report produced by
``scripts/validate_pattern_ml_workstream_d.py`` can show, verifiably, that
the bars it was checked against predate the run.

DO NOT EDIT ``BARS`` OR ``PREREGISTERED_AT`` BELOW AFTER A REAL (non-dry-run)
evaluation has executed against them. If a genuinely new hypothesis emerges
(not "the same test with different numbers until it passes" -- see the
plan's own "On pre-commit to accepting the result" section for the
distinction), the correct move is a NEW dated bars module / a new commit
with a new ``PREREGISTERED_AT``, with the old one and its result left
intact in git history for anyone to compare -- never a silent edit to this
file's committed content.

One-shot enforcement, concretely:
  1. This file is committed on its own, before the real run, so its git
     blob/commit timestamp is independently checkable (``git log --
     scripts/pattern_ml_preregistered_bars.py``) against the run's own
     reported wall-clock time.
  2. ``bars_fingerprint()`` hashes the frozen content below; the harness
     embeds that hash in every report it writes, so a report can be
     checked against the bars file that was live at commit time.
  3. The harness appends (never overwrites) every invocation -- dry-run or
     real -- to ``docs/pattern_ml_workstream_d_run_log.json``, so a second
     "real" attempt against the same bars/model artifacts is visible in the
     ledger rather than silently replacing the first report.
"""

from __future__ import annotations

import hashlib
import json

# Frozen the moment this file was first written (UTC, ISO-8601) -- the
# instant these bars became "pre-registered" per the plan's own language.
# NEVER change this value on an edit to BARS; see the module docstring.
PREREGISTERED_AT = "2026-09-27T19:10:00Z"

# Each bar name maps to (comparison, threshold, description). ``comparison``
# is one of "gt" (strictly greater than threshold passes), "lt" (strictly
# less than), or "bool" (threshold ignored; the value itself must be True).
BARS: dict[str, dict] = {
    "executable_expectancy_r_positive": {
        "comparison": "gt",
        "threshold": 0.0,
        "description": (
            "Executable expectancy (mean R/trade), priced at a realistic "
            "next-bar-open fill with risk = |fill - stop| (NOT |entry - "
            "stop|), must be > 0."
        ),
    },
    "expectancy_ci_excludes_zero": {
        "comparison": "bool",
        "threshold": True,
        "description": (
            "The symbol-block bootstrap 95% CI on that expectancy must "
            "exclude zero (ci_low > 0)."
        ),
    },
    "expectancy_beats_placebo": {
        "comparison": "bool",
        "threshold": True,
        "description": (
            "The executable expectancy must beat the direction-flipped "
            "placebo (mirrored barriers, flipped direction, same subset). "
            "If the placebo does as well or better, this is a FAIL "
            "regardless of the raw expectancy number."
        ),
    },
    "deflated_sharpe_min": {
        "comparison": "gt",
        "threshold": 0.95,
        "description": (
            "Deflated Sharpe Ratio > 0.95, deflated by the CUMULATIVE "
            "cross-session trial count (this run's grid PLUS every prior "
            "session's trials against this same ML layer -- see "
            "docs/pattern_ml_trial_history.json)."
        ),
    },
    "pbo_max": {
        "comparison": "lt",
        "threshold": 0.50,
        "description": (
            "Probability of Backtest Overfitting (CSCV) < 0.50, computed "
            "over a candidate grid of >= 5 genuinely distinct candidates "
            "(a 2-candidate grid degenerates to a coin flip -- lambda = "
            "+-log(2) -- and is not admissible evidence for this bar)."
        ),
    },
    "pbo_min_candidates": {
        "comparison": "gt",
        "threshold": 4,
        "description": (
            "The PBO grid must have >= 5 candidates (threshold is 4, "
            "checked as strictly-greater) for the PBO number itself to be "
            "admissible -- see cscv_pbo's N=2 degeneracy note above."
        ),
    },
    "fold_consistency_min": {
        "comparison": "gt",
        "threshold": 0.749,
        "description": (
            "Fold consistency >= 75% -- matches "
            "scripts/validate_pattern_cnn_walkforward.py's own "
            "CONSISTENT_MAJORITY_THRESHOLD / derive_recommendation bar, "
            "reused (not reimplemented) by this harness. Checked as "
            "strictly-greater than 0.749 to admit an exact 0.75 while "
            "avoiding float-equality comparison."
        ),
    },
}


# --------------------------------------------------------------------------
# Pre-registered TEST DESIGN (not just the pass/fail thresholds above). All
# of this must be fixed before the real run too -- picking whichever
# candidate/threshold happens to look best after seeing results would be
# exactly the "re-roll until it passes" this module exists to prevent.
# --------------------------------------------------------------------------

# The ONE meta-confidence threshold whose executable-expectancy result
# actually determines the pass/fail bars above (executable_expectancy_r_positive
# / expectancy_ci_excludes_zero / expectancy_beats_placebo). Chosen a priori
# as a moderate, round value -- not selected after looking at any threshold
# sweep's numbers. 0.50 is the model's own natural decision boundary (P(act)
# > no-act) so it is the least arbitrary choice available, and the least
# generous to the "find a profitable high-confidence sliver" hypothesis
# (a stricter threshold trivially has fewer, more cherry-picked trades).
TARGET_META_CONFIDENCE_THRESHOLD = 0.50

# Reported alongside the target for transparency/diagnostics ONLY -- never
# used to decide pass/fail. Seeing "0.50 fails but 0.75 would have passed"
# is honest context, not grounds to retroactively promote 0.75 to the target.
DIAGNOSTIC_META_CONFIDENCE_THRESHOLDS = (0.60, 0.70, 0.75)

# Trade-level (point-in-time scan) executable-expectancy check parameters --
# fixed for the same reason the threshold above is: a bootstrap seed or
# timeout horizon could otherwise be quietly varied until a CI happens to
# exclude zero. Matches scripts/measure_pattern_executable_expectancy.py's
# own defaults (--timeout-bars 20, --seed 7) for direct comparability with
# Workstream B's numbers.
TRADE_LEVEL_TIMEOUT_BARS = 20
TRADE_LEVEL_N_BOOT = 2000
TRADE_LEVEL_SEED = 7

# Portfolio-level (isolated single-strategy) walk-forward parameters --
# identical to scripts/validate_pattern_cnn_walkforward.py's own argparse
# defaults, so this run is directly comparable to the three prior sessions
# in docs/pattern_ml_trial_history.json.
WALK_FORWARD_START_DATE = "2018-01-01"
WALK_FORWARD_END_DATE = "2025-12-31"
WALK_FORWARD_N_SPLITS = 8
WALK_FORWARD_TRAIN_PCT = 0.7
WALK_FORWARD_EMBARGO_DAYS = 1
WALK_FORWARD_SELECTION_METRIC = "sharpe_ratio"

# >= 5 genuinely distinct candidates (cscv_pbo degenerates to a coin flip at
# N=2 -- overfitting.py:99-168, lambda = +-log(2)). Candidate 0 is the
# rule-based-only control; the rest independently vary
# xgb_confirmation_enabled, xgb_meta_confirmation_enabled, the meta
# confidence gate (the mechanism that actually lets a classifier "find the
# profitable subset of a zero-mean population" -- see the plan's Workstream
# D framing -- a down-weight-only Kelly input cannot test that hypothesis,
# only a real inclusion/exclusion threshold can), and zscore_demean (a real,
# free axis per Workstream A.3 -- this strategy's cross-sectional
# normalization flag, independent of anything pattern-recognition-specific).
PARAM_GRID: list[dict] = [
    {"strategy_params": {"pattern_recognition": {
        "xgb_confirmation_enabled": False, "xgb_meta_confirmation_enabled": False,
    }}},
    {"strategy_params": {"pattern_recognition": {
        "xgb_confirmation_enabled": True, "xgb_meta_confirmation_enabled": False,
    }}},
    {"strategy_params": {"pattern_recognition": {
        "xgb_confirmation_enabled": False, "xgb_meta_confirmation_enabled": True,
        "xgb_meta_min_confidence": None,
    }}},
    {"strategy_params": {"pattern_recognition": {
        "xgb_confirmation_enabled": False, "xgb_meta_confirmation_enabled": True,
        "xgb_meta_min_confidence": TARGET_META_CONFIDENCE_THRESHOLD,
    }}},
    {"strategy_params": {"pattern_recognition": {
        "xgb_confirmation_enabled": False, "xgb_meta_confirmation_enabled": True,
        "xgb_meta_min_confidence": 0.75,
    }}},
    {
        "zscore_demean": False,
        "strategy_params": {"pattern_recognition": {
            "xgb_confirmation_enabled": False, "xgb_meta_confirmation_enabled": True,
            "xgb_meta_min_confidence": 0.75,
        }},
    },
]


def bars_fingerprint() -> str:
    """SHA-256 of the entire frozen pre-registration (bars, target
    threshold, and test design) -- embed this in every report so a reader
    can verify, byte-for-byte, which frozen design a run's verdict was
    checked against (compare against a fresh call of this function against
    the committed file, or against another report's fingerprint)."""
    payload = json.dumps(
        {
            "PREREGISTERED_AT": PREREGISTERED_AT,
            "BARS": BARS,
            "TARGET_META_CONFIDENCE_THRESHOLD": TARGET_META_CONFIDENCE_THRESHOLD,
            "DIAGNOSTIC_META_CONFIDENCE_THRESHOLDS": list(DIAGNOSTIC_META_CONFIDENCE_THRESHOLDS),
            "WALK_FORWARD_START_DATE": WALK_FORWARD_START_DATE,
            "WALK_FORWARD_END_DATE": WALK_FORWARD_END_DATE,
            "WALK_FORWARD_N_SPLITS": WALK_FORWARD_N_SPLITS,
            "WALK_FORWARD_TRAIN_PCT": WALK_FORWARD_TRAIN_PCT,
            "WALK_FORWARD_EMBARGO_DAYS": WALK_FORWARD_EMBARGO_DAYS,
            "WALK_FORWARD_SELECTION_METRIC": WALK_FORWARD_SELECTION_METRIC,
            "PARAM_GRID": PARAM_GRID,
            "TRADE_LEVEL_TIMEOUT_BARS": TRADE_LEVEL_TIMEOUT_BARS,
            "TRADE_LEVEL_N_BOOT": TRADE_LEVEL_N_BOOT,
            "TRADE_LEVEL_SEED": TRADE_LEVEL_SEED,
        },
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def check_bar(name: str, value) -> bool | None:
    """Evaluate one named bar against a measured ``value``.

    Returns ``None`` (not a bool) when ``value`` is ``None`` or non-finite --
    an un-computable bar must never silently read as a pass.
    """
    if value is None:
        return None
    spec = BARS[name]
    comparison = spec["comparison"]
    threshold = spec["threshold"]
    if comparison == "bool":
        return bool(value) is True
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    if comparison == "gt":
        return v > threshold
    if comparison == "lt":
        return v < threshold
    raise ValueError(f"unknown comparison {comparison!r} for bar {name!r}")


def evaluate_all(measurements: dict) -> dict:
    """Check every bar in ``BARS`` against ``measurements`` (a dict keyed by
    bar name). Returns ``{"bars_fingerprint", "preregistered_at", "per_bar":
    {name: {"value", "passed", "description"}}, "overall_pass"}``.

    ``overall_pass`` is True only when every bar has a non-``None`` passing
    result -- a bar that could not be computed at all (``None``) is treated
    as a FAIL for the overall verdict (fail-closed), not skipped.
    """
    per_bar = {}
    all_pass = True
    for name, spec in BARS.items():
        value = measurements.get(name)
        passed = check_bar(name, value)
        per_bar[name] = {
            "value": value,
            "passed": passed,
            "description": spec["description"],
        }
        if passed is not True:
            all_pass = False
    return {
        "bars_fingerprint": bars_fingerprint(),
        "preregistered_at": PREREGISTERED_AT,
        "per_bar": per_bar,
        "overall_pass": all_pass,
    }
