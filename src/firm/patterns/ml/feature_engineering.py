"""Feature engineering for the pattern-confirmation ML layer (Phase 4).

Turns one :class:`~firm.patterns.match.PatternMatch` (as produced by
:func:`firm.patterns.scanner.scan_symbol`) into a flat ``dict[str, float]``
feature vector suitable for XGBoost (see :mod:`firm.patterns.ml.xgb_classifier`).
Deliberately a pure function of its inputs — no I/O, no global state — so it's
trivially unit-testable and safe to call from a scan loop over thousands of
symbols.

Scope note: this is the "XGBoost confirmation classifier" slice of Phase 4
(docs/pattern_recognition_plan.md section 2's phase table). The CNN/GAF image
validator and PPO RL position sizer from that phase's original aspirational
scope are explicitly out of scope here (heavy new dependencies — torch,
stable-baselines3, gymnasium, pyts — not installed on this box); this module
only feeds the classifier half.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from firm.patterns._indicators import atr14
from firm.patterns.match import PatternMatch

log = logging.getLogger(__name__)

# Keep in sync with docs/pattern_recognition_plan.md section 2 (17 pattern
# names across 4 rule families: reversal/triangle/continuation/cup_handle —
# see tests/test_patterns.py's docstring, which independently confirms the
# same count). Duplicated here rather than imported from
# firm.patterns.rules.* because those modules are read-only for this change
# (see docs/pattern_recognition_plan.md) and don't expose a canonical list of
# names to import in the first place -- each rule function just returns a
# literal string. If a future rule module adds a new pattern name, this map
# must be updated by hand; unrecognized names degrade gracefully (see
# `_family_of` below) rather than raising, so a mismatch here doesn't break
# scanning/scoring, just silently loses one-hot signal for the ML layer until
# fixed.
PATTERN_FAMILY_MAP: dict[str, str] = {
    "head_shoulders_top": "reversal",
    "inverse_head_shoulders": "reversal",
    "double_top": "reversal",
    "double_bottom": "reversal",
    "triple_top": "reversal",
    "triple_bottom": "reversal",
    "ascending_triangle": "triangle",
    "descending_triangle": "triangle",
    "rectangle": "triangle",
    "symmetrical_triangle": "triangle",
    "rising_wedge": "triangle",
    "falling_wedge": "triangle",
    "bull_flag": "continuation",
    "bear_flag": "continuation",
    "pennant": "continuation",
    "cup_handle": "cup_handle",
    "rounding_bottom": "cup_handle",
}
PATTERN_NAMES: tuple[str, ...] = tuple(PATTERN_FAMILY_MAP)
PATTERN_FAMILIES: tuple[str, ...] = ("reversal", "triangle", "continuation", "cup_handle")

# score_breakdown (firm.patterns.scorer.PatternScore.as_dict()) component keys.
_SCORE_BREAKDOWN_KEYS = (
    "geometry", "trendline_fit", "volume_confirmation", "duration", "follow_through", "total",
)


def _family_of(pattern: str) -> str | None:
    family = PATTERN_FAMILY_MAP.get(pattern)
    if family is None:
        log.warning(
            "feature_engineering: unrecognized pattern name %r -- "
            "one-hot pattern/family columns will be all-zero for this row "
            "(PATTERN_FAMILY_MAP in feature_engineering.py may need updating)",
            pattern,
        )
    return family


def _ohlcv_context_features(
    match: PatternMatch, ohlcv: pd.DataFrame | None, momentum_lookback: int
) -> dict[str, float]:
    """Optional features derived from the raw OHLCV window around
    ``confirm_index``, when the caller has it available. Returns a
    zero-filled block (plus ``ohlcv_context_available=0.0``) when ``ohlcv``
    is absent or too short to look back into, so the caller never has to
    special-case a missing OHLCV context.
    """
    zeros = {
        "pre_pattern_return": 0.0,
        "pre_pattern_volatility": 0.0,
        "atr_pct": 0.0,
        "ohlcv_context_available": 0.0,
    }
    if ohlcv is None or len(ohlcv) == 0:
        return zeros
    idx = match.confirm_index
    if idx < 0 or idx >= len(ohlcv):
        return zeros

    close = ohlcv["close"].to_numpy(dtype=float)
    high = ohlcv["high"].to_numpy(dtype=float)
    low = ohlcv["low"].to_numpy(dtype=float)

    start = max(0, idx - momentum_lookback)
    pre_pattern_return = float(close[idx] / close[start] - 1.0) if close[start] else 0.0

    window = close[start : idx + 1]
    if len(window) > 1:
        rets = np.diff(window) / window[:-1]
        pre_pattern_volatility = float(np.std(rets))
    else:
        pre_pattern_volatility = 0.0

    atr = atr14(high[: idx + 1], low[: idx + 1], close[: idx + 1])
    current_atr = float(atr[-1]) if len(atr) and not np.isnan(atr[-1]) else 0.0
    atr_pct = float(current_atr / close[idx]) if close[idx] else 0.0

    return {
        "pre_pattern_return": pre_pattern_return,
        "pre_pattern_volatility": pre_pattern_volatility,
        "atr_pct": atr_pct,
        "ohlcv_context_available": 1.0,
    }


def build_features(
    match: PatternMatch,
    ohlcv: pd.DataFrame | None = None,
    *,
    momentum_lookback: int = 20,
) -> dict[str, float]:
    """Flatten one confirmed :class:`PatternMatch` into an all-numeric,
    all-finite feature dict for XGBoost.

    Includes (see inline groups below): the match's own scalar detection/
    scoring fields, scale-invariant stop/target distances (raw price levels
    aren't comparable across symbols at different price points), the
    unpacked ``score_breakdown`` components, a one-hot pattern-family
    encoding and a one-hot specific-pattern encoding, and — only when
    ``ohlcv`` is supplied — a small block of price-context features computed
    from the ``momentum_lookback`` bars leading into ``confirm_index``.

    ``ohlcv`` should be the same high/low/close/volume frame (ascending by
    date) that was passed to :func:`firm.patterns.scanner.scan_symbol` to
    produce ``match`` — ``confirm_index`` is an offset into that array.
    Passing ``None`` still returns a complete, valid feature dict (with the
    OHLCV-context block zeroed and flagged via
    ``ohlcv_context_available=0.0``), which is what keeps this a pure,
    single-argument-testable function rather than one that requires network/
    disk access to unit test.

    Every value is a plain Python ``float`` (0.0/1.0 for the one-hot/flag
    columns) -- no NaN is ever emitted: a missing ``volume_ratio`` (e.g. not
    enough trailing history — see ``firm.patterns._indicators.volume_ratio``)
    is filled with a neutral 1.0 ("at the trailing average, no information")
    and flagged via a companion ``volume_ratio_missing`` column instead, so
    downstream consumers that don't special-case NaN (e.g. a plain
    ``accuracy_score`` computation, or naive column stats) can't be silently
    corrupted by it. XGBoost itself handles NaN as "missing" natively, but
    that's an orthogonal reason to prefer an explicit flag here: it keeps the
    feature dict self-describing regardless of which model consumes it.
    """
    direction_sign = 1.0 if match.direction == "long" else -1.0
    entry = float(match.entry)
    stop_distance_pct = abs(entry - float(match.stop)) / entry if entry else 0.0
    target_distance_pct = abs(float(match.target) - entry) / entry if entry else 0.0

    volume_ratio = match.volume_ratio
    volume_ratio_missing = 1.0 if volume_ratio != volume_ratio else 0.0  # NaN != NaN
    volume_ratio_filled = 1.0 if volume_ratio_missing else float(volume_ratio)

    features: dict[str, float] = {
        # --- raw PatternMatch scalar fields ---
        "direction_sign": direction_sign,
        "entry": entry,
        "stop": float(match.stop),
        "target": float(match.target),
        "fit_quality": float(match.fit_quality),
        "geometry_tolerance_used": float(match.geometry_tolerance_used),
        "volume_ratio": volume_ratio_filled,
        "volume_ratio_missing": volume_ratio_missing,
        "duration_bars": float(match.duration_bars),
        "follow_through_atr": float(match.follow_through_atr),
        "risk_reward": float(match.risk_reward),
        "quality_score": float(match.quality_score),
        "num_pivots": float(len(match.pivots)),
        "confirm_index": float(match.confirm_index),
        # --- derived, scale-invariant versions of entry/stop/target ---
        "stop_distance_pct": float(stop_distance_pct),
        "target_distance_pct": float(target_distance_pct),
    }

    breakdown = match.score_breakdown or {}
    for key in _SCORE_BREAKDOWN_KEYS:
        features[f"score_{key}"] = float(breakdown.get(key, 0.0))

    family = _family_of(match.pattern)
    for name in PATTERN_FAMILIES:
        features[f"family_{name}"] = 1.0 if name == family else 0.0
    for name in PATTERN_NAMES:
        features[f"pattern_{name}"] = 1.0 if name == match.pattern else 0.0

    features.update(_ohlcv_context_features(match, ohlcv, momentum_lookback))
    return features


def build_feature_frame(
    matches: list[tuple[PatternMatch, pd.DataFrame | None]] | list[PatternMatch],
    *,
    momentum_lookback: int = 20,
) -> pd.DataFrame:
    """Batch convenience: build a feature matrix (one row per match) as a
    :class:`pandas.DataFrame` with a stable column order.

    Accepts either a plain list of matches (no OHLCV context) or a list of
    ``(match, ohlcv)`` pairs (per-match context, e.g. when scanning multiple
    symbols with different OHLCV frames). Returns an empty, columnless
    DataFrame for an empty input rather than raising, so callers can always
    check ``.empty`` uniformly.
    """
    rows: list[dict[str, float]] = []
    for item in matches:
        if isinstance(item, tuple):
            match, ohlcv = item
        else:
            match, ohlcv = item, None
        rows.append(build_features(match, ohlcv, momentum_lookback=momentum_lookback))
    return pd.DataFrame(rows)
