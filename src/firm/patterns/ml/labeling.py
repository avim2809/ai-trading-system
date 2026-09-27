"""Triple-barrier labeling (Lopez de Prado, *Advances in Financial Machine
Learning*, ch. 3) for confirmed chart patterns.

Given a confirmed :class:`~firm.patterns.match.PatternMatch` and the same
OHLCV array it was detected from, walks forward bar-by-bar from
``confirm_index + 1`` and labels the outcome by whichever of the pattern's
own three barriers is touched first:

- ``+1`` -- the *profit-take* barrier (``match.target``) is touched first.
- ``-1`` -- the *stop-loss* barrier (``match.stop``) is touched first.
- ``0``  -- neither is touched before the *vertical* (time) barrier, i.e.
  the pattern neither confirms nor invalidates within the timeout window.

Unlike the classical formulation (which sizes barriers off a rolling
volatility estimate), this reuses the pattern's own already-computed
entry/stop/target levels — they're the actual trade the ``pattern_recognition``
strategy would have taken, so labeling against them directly is more
faithful to "would this specific detection have worked" than re-deriving a
generic volatility-scaled barrier would be.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from firm.patterns.match import PatternMatch

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT_BARS = 20


def label_triple_barrier(
    match: PatternMatch,
    ohlcv: pd.DataFrame,
    *,
    timeout_bars: int = DEFAULT_TIMEOUT_BARS,
) -> int:
    """Label one confirmed pattern's subsequent price path.

    Args:
        match: A confirmed match (``match.confirmed`` -- i.e.
            ``confirm_index >= 0``); raises ``ValueError`` otherwise, since
            an unconfirmed match has no breakout bar to walk forward from
            and silently labeling it would produce a meaningless (and easy
            to accidentally trust) row.
        ohlcv: The *same* high/low/close/volume frame (ascending by date,
            same indexing) that produced ``match`` -- ``confirm_index`` is a
            positional offset into it, not a date lookup.
        timeout_bars: Vertical-barrier horizon in bars after
            ``confirm_index``. Kept as a parameter rather than hardcoded so
            a human can tune it per the training script's CLI; 20 bars
            (~1 trading month) is a reasonable default for the daily-bar
            swing-trade horizon this strategy targets (its own default
            ``horizon`` param is "10d", so 20 gives the trade roughly double
            its nominal horizon to resolve before calling it a timeout).

    Returns:
        ``1`` (target hit first), ``-1`` (stop hit first), or ``0`` (neither
        within ``timeout_bars``, including when there simply isn't enough
        subsequent data to look at).

    Note on same-bar barrier collisions: if a single bar's high/low range
    touches *both* the stop and the target (a large gap or whipsaw bar),
    the stop wins -- the same conservative "assume the worse fill" bias this
    codebase applies elsewhere (e.g. ``firm.live.execution_safety``'s
    fail-closed default) rather than assuming the best-case fill order
    within the bar.
    """
    if not match.confirmed:
        raise ValueError(
            f"label_triple_barrier requires a confirmed match (confirm_index >= 0), "
            f"got confirm_index={match.confirm_index} for pattern={match.pattern!r}"
        )

    n = len(ohlcv)
    start = match.confirm_index + 1
    if start >= n:
        log.debug(
            "label_triple_barrier: no bars after confirm_index=%d (n=%d) for %s -- "
            "labeling as 0 (timeout)",
            match.confirm_index, n, match.pattern,
        )
        return 0

    end = min(n, start + timeout_bars)
    high = ohlcv["high"].to_numpy(dtype=float)[start:end]
    low = ohlcv["low"].to_numpy(dtype=float)[start:end]
    return first_barrier_hit(high, low, direction=match.direction, stop=float(match.stop), target=float(match.target))


def label_triple_barrier_with_exit(
    match: PatternMatch,
    ohlcv: pd.DataFrame,
    *,
    timeout_bars: int = DEFAULT_TIMEOUT_BARS,
) -> tuple[int, int]:
    """Same label as :func:`label_triple_barrier`, plus the absolute bar
    index (a position into ``ohlcv``, same indexing as ``match.confirm_index``)
    where that label was actually determined -- i.e. de Prado's ``t1`` for
    this labeled event (ch. 3/4: the event "occupies"
    ``[match.confirm_index, exit_index]``, not just its single confirm bar).

    Added for :mod:`firm.patterns.ml.sample_weights`'s average-uniqueness
    weighting (Part B item 3, 2026-09-27), which needs each event's real
    span to compute how many OTHER labeled events were concurrently
    "in-flight" at any given bar -- :func:`label_triple_barrier` alone only
    ever exposes the label, not how long it took to resolve, so every event
    would otherwise have to be (wrongly) treated as spanning the full fixed
    ``timeout_bars`` regardless of whether it actually resolved on bar 1 or
    bar 20.

    Returns ``(label, exit_index)``. When there are no bars after
    ``confirm_index`` to look at (mirrors :func:`label_triple_barrier`'s own
    degenerate case), ``exit_index`` is ``match.confirm_index`` itself --
    a zero-length span, the only sensible answer when nothing is actually
    known about what happens next.
    """
    if not match.confirmed:
        raise ValueError(
            f"label_triple_barrier_with_exit requires a confirmed match (confirm_index >= 0), "
            f"got confirm_index={match.confirm_index} for pattern={match.pattern!r}"
        )

    n = len(ohlcv)
    start = match.confirm_index + 1
    if start >= n:
        log.debug(
            "label_triple_barrier_with_exit: no bars after confirm_index=%d (n=%d) for %s -- "
            "labeling as 0 (timeout), exit_index=confirm_index (zero-length span)",
            match.confirm_index, n, match.pattern,
        )
        return 0, match.confirm_index

    end = min(n, start + timeout_bars)
    high = ohlcv["high"].to_numpy(dtype=float)[start:end]
    low = ohlcv["low"].to_numpy(dtype=float)[start:end]
    label, offset = _first_barrier_hit_with_offset(
        high, low, direction=match.direction, stop=float(match.stop), target=float(match.target),
    )
    return label, start + offset


def _first_barrier_hit_with_offset(
    high: np.ndarray,
    low: np.ndarray,
    *,
    direction: str,
    stop: float,
    target: float,
) -> tuple[int, int]:
    """Shared core of :func:`first_barrier_hit`/:func:`label_triple_barrier_with_exit`:
    walk ``high``/``low`` bar-by-bar and return ``(label, offset)``, where
    ``offset`` is the 0-indexed position *within the passed arrays* where
    the label was determined -- the last bar (``len(high) - 1``) when
    neither barrier is touched (the vertical/timeout barrier, or simply
    running out of array). Never returns an offset past the array's own
    bounds, and never called on an empty array (see both public callers'
    own empty-input handling).
    """
    is_long = direction == "long"
    for i, (bar_high, bar_low) in enumerate(zip(high, low)):
        if is_long:
            hit_stop = bar_low <= stop
            hit_target = bar_high >= target
        else:
            hit_stop = bar_high >= stop
            hit_target = bar_low <= target
        if hit_stop:
            return -1, i
        if hit_target:
            return 1, i
    return 0, len(high) - 1


def first_barrier_hit(
    high: np.ndarray,
    low: np.ndarray,
    *,
    direction: str,
    stop: float,
    target: float,
) -> int:
    """Shared inner loop of :func:`label_triple_barrier`: walk ``high``/
    ``low`` bar-by-bar and return the first barrier touched (``1`` target,
    ``-1`` stop — stop wins on a same-bar tie, ``0`` if neither). Factored
    out so a live outcome-tracking job can apply the *same* barrier logic
    against freshly-fetched, date-aligned price data for an already-persisted
    historical match, without a second hand-written copy of it (a
    ``PatternMatch`` + its original scan-time array aren't available for a
    row read back out of storage days/weeks later).
    """
    label, _offset = _first_barrier_hit_with_offset(high, low, direction=direction, stop=stop, target=target)
    return label


def label_meta_binary(
    match: PatternMatch,
    ohlcv: pd.DataFrame,
    *,
    timeout_bars: int = DEFAULT_TIMEOUT_BARS,
) -> int:
    """Binary meta-label (Lopez de Prado ch. 3.6): ``1`` ("act" -- this
    detection would have been a genuine win, target hit first) or ``0``
    ("don't act" -- stop hit first, or neither barrier touched before the
    vertical/timeout barrier).

    Why this exists as its OWN function rather than callers thresholding
    :func:`label_triple_barrier`'s ``-1``/``0``/``1`` output inline (Part B
    item 2, 2026-09-27): meta-labeling's whole premise is a SEPARATE
    secondary model answering "should I act on the primary model's call at
    all" (here, ``pattern_recognition``'s own directional signal is the
    primary call), trained on a genuinely different target than "which of
    three things will happen" -- collapsing stop-hit and timeout into the
    same ``0`` class is a deliberate modeling decision (both mean "this
    specific trade wasn't a win"), not an implementation detail a caller
    should have to re-derive correctly every time. Reusing
    :func:`label_triple_barrier`'s barrier-walk (rather than a second,
    separately-hand-written copy) keeps there being exactly one place that
    decides same-bar-collision/timeout/no-remaining-data semantics.

    Args/timeout_bars/raises: identical contract to
    :func:`label_triple_barrier` (see its docstring) -- this is a pure
    relabeling of that function's output, not a different walk.
    """
    return 1 if label_triple_barrier(match, ohlcv, timeout_bars=timeout_bars) == 1 else 0


def label_matches_meta_binary(
    matches: list[PatternMatch],
    ohlcv: pd.DataFrame,
    *,
    timeout_bars: int = DEFAULT_TIMEOUT_BARS,
) -> np.ndarray:
    """Batch convenience mirroring :func:`label_matches`, for the binary
    meta-label (:func:`label_meta_binary`) instead of the 3-class
    triple-barrier label.
    """
    labels = []
    for match in matches:
        if not match.confirmed:
            log.debug("label_matches_meta_binary: skipping unconfirmed match %s", match.pattern)
            continue
        labels.append(label_meta_binary(match, ohlcv, timeout_bars=timeout_bars))
    return np.array(labels, dtype=int)


def label_matches(
    matches: list[PatternMatch],
    ohlcv: pd.DataFrame,
    *,
    timeout_bars: int = DEFAULT_TIMEOUT_BARS,
) -> np.ndarray:
    """Batch convenience: label every *confirmed* match in ``matches``
    against the same ``ohlcv`` frame (e.g. all matches from one
    ``scan_symbol`` call on one symbol). Unconfirmed matches are skipped
    with a debug log rather than raising, since a batch caller scanning many
    symbols shouldn't abort the whole run over one unconfirmed candidate --
    contrast with :func:`label_triple_barrier` itself, which raises, since a
    caller invoking it directly on a single match has presumably already
    decided that match is meant to be labeled.
    """
    labels = []
    for match in matches:
        if not match.confirmed:
            log.debug("label_matches: skipping unconfirmed match %s", match.pattern)
            continue
        labels.append(label_triple_barrier(match, ohlcv, timeout_bars=timeout_bars))
    return np.array(labels, dtype=int)
