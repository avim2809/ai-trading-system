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
    is_long = direction == "long"
    for bar_high, bar_low in zip(high, low):
        if is_long:
            hit_stop = bar_low <= stop
            hit_target = bar_high >= target
        else:
            hit_stop = bar_high >= stop
            hit_target = bar_low <= target
        if hit_stop:
            return -1
        if hit_target:
            return 1
    return 0


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
