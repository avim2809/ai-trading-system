"""Shared breakout-confirmation search used by every pattern-rule module.

A detected geometric structure (neckline, trendline, handle-high, ...) only
becomes a tradeable pattern once price actually closes through the relevant
level — see the "Confirmation" column in the pattern-rule design. This finds
the most recent such close within a lookback window, so a formation that
completed weeks ago but never broke out simply doesn't match.

Also provides :func:`retest_outcome` / :func:`retest_score_modifier`: a
post-breakout "did the level hold on retest" read. Roughly half of genuine
breakouts see price pull back to retest the broken level within a handful of
bars, and continuation odds are meaningfully higher when the level holds
than when it fails — but hard-requiring a retest before acting would forfeit
the ~50% of setups that never retest at all with no clear net benefit. So
this is deliberately a scoring *modifier*, never a hard gate: it is a pure
addition to this module and does not change :func:`find_confirmation` or
its callers in any way.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

Direction = Literal["above", "below"]
RetestOutcome = Literal["held", "failed", "no_retest"]


def find_confirmation(
    close: np.ndarray,
    level_at: Callable[[int], float],
    direction: Direction,
    lookback_bars: int,
    min_index: int,
) -> int:
    """Most recent bar index in the last *lookback_bars* where ``close``
    genuinely **crossed** ``level_at(i)`` in *direction* — i.e. ``close[i]``
    is beyond ``level_at(i)`` *and* ``close[i-1]`` was still on the near side
    of ``level_at(i-1)`` — searching newest-first so the freshest breakout
    (the most recent such crossing event, not merely the most recent bar
    that happens to still be past the level) wins. Returns -1 if none found
    (pattern not yet, or no longer, confirmed).

    Bug history (2026-09-27): this used to be a plain **is-beyond** test —
    ``if close[i] > level: return i`` — with no edge condition at all,
    despite this same docstring always having described it as finding where
    price *crossed* the level. Searching newest-first, that returned
    **today** (``n - 1``) every time price was still past the level, which
    it structurally almost always is for many bars after a real breakout.
    Measured impact: ``bars_since_confirm`` was pinned to ``{0, 1, 2}`` by
    construction, sitting at 0 on ~93% of emissions, and every rule module's
    ``entry = level_at(confirm_index)`` was therefore a price already run
    through (median overshoot +1.72R), not a real, tradeable entry. See
    ``docs/pattern_recognition_plan.md`` (Workstream C) for the full
    measurement. This function now requires an actual transition, so
    ``confirm_index`` becomes a fixed historical point (the real breakout
    bar) rather than drifting forward with every new bar of data — which
    also makes it prefix-invariant / restart-safe for free.

    Never considers bars at or before *min_index* (the pattern's own last
    structural pivot) — a "breakout" before the pattern even finished
    forming isn't one — as a *candidate* confirmation bar ``i``. The one
    exception is the edge check's reference bar ``i - 1``: when ``i ==
    start`` (the oldest bar this function is willing to call a
    confirmation), the *previous* bar — which may be ``min_index`` itself,
    or even earlier when ``lookback_bars`` is the binding constraint instead
    of ``min_index`` — is still read purely to determine whether bar ``i``
    was a genuine transition, not treated as a confirmation candidate in its
    own right. That is the more correct reading of this docstring's own
    "crossed" intent: the pattern's structural pivot is very often sitting
    exactly at the level (e.g. a neckline pivot), so refusing to look one
    bar back would make the very first bar after the pivot structurally
    unconfirmable. The only case truly excluded for lack of information is
    ``i == 0`` (no bar ``-1`` exists at all).
    """
    n = len(close)
    start = max(min_index + 1, n - lookback_bars)
    for i in range(n - 1, start - 1, -1):
        if i - 1 < 0:
            continue  # no prior bar at all -- can't tell if this is a genuine crossing
        level = level_at(i)
        prev_level = level_at(i - 1)
        if direction == "above":
            beyond = close[i] > level
            prev_inside = close[i - 1] <= prev_level
        else:
            beyond = close[i] < level
            prev_inside = close[i - 1] >= prev_level
        if beyond and prev_inside:
            return i
    return -1


def retest_outcome(
    close: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    level_at_confirm: float,
    direction: Direction,
    confirm_index: int,
    *,
    lookback_bars: int = 10,
) -> RetestOutcome:
    """Coarse read of what happened when/if price came back to test the
    just-broken level after ``confirm_index``.

    Deliberately a coarse, OHLCV-only heuristic — not a strict touch-by-touch
    state machine tracking every subsequent cross of the level. It answers
    exactly one question, operationally defined as follows:

    1. **Retest window**: bars ``confirm_index + 1`` .. ``confirm_index +
       lookback_bars`` inclusive, clipped to the end of the arrays. If this
       window is empty (``confirm_index`` is at/past the end of the data),
       returns ``"no_retest"``.
    2. **Did a retest happen?** For an ``"above"`` breakout (price broke
       *up* through the level), a retest is any bar in the window where
       ``low[i] <= level_at_confirm`` (price came back down to touch/cross
       the level again). For a ``"below"`` breakout (broke *down*), a retest
       is any bar where ``high[i] >= level_at_confirm``. If no bar in the
       window retests, returns ``"no_retest"`` — this is the common,
       perfectly fine case (~half of real breakouts never look back), not a
       failure.
    3. **Held vs. failed**, only evaluated once a retest has occurred: look
       at the *last* close in the window (not the retest bar itself, and not
       every bar in between — this is the coarse simplification). For an
       ``"above"`` breakout, ``"held"`` means that last close is still at or
       above ``level_at_confirm`` (price came back, touched the level, and
       resumed/stayed in the breakout direction by window's end); otherwise
       ``"failed"`` (price is back on the wrong side of the level by
       window's end). Mirrored for ``"below"``.

    Bounds: ``lookback_bars <= 0``, ``confirm_index`` out of range, or
    degenerate (empty/too-short) arrays all degrade gracefully to
    ``"no_retest"`` rather than raising.
    """
    n = len(close)
    if n == 0 or confirm_index < 0 or confirm_index >= n:
        log.debug(
            "retest_outcome: confirm_index=%d out of range for array length=%d, returning no_retest",
            confirm_index, n,
        )
        return "no_retest"
    start = confirm_index + 1
    end = min(n, confirm_index + 1 + lookback_bars)  # exclusive
    if start >= end:
        return "no_retest"

    retested = False
    for i in range(start, end):
        if direction == "above" and low[i] <= level_at_confirm:
            retested = True
            break
        if direction == "below" and high[i] >= level_at_confirm:
            retested = True
            break
    if not retested:
        return "no_retest"

    last_close = close[end - 1]
    if direction == "above":
        return "held" if last_close >= level_at_confirm else "failed"
    return "held" if last_close <= level_at_confirm else "failed"


# Small, clearly-bounded magnitude: the integration owner decides exactly
# how (and whether) to fold this into an overall quality_score alongside
# scorer.py's components; this module only provides the clean, tested
# building block.
_RETEST_MODIFIER: dict[str, float] = {"held": 1.0, "failed": -1.0, "no_retest": 0.0}


def retest_score_modifier(outcome: str) -> float:
    """Maps a :func:`retest_outcome` result to a small numeric scorer
    modifier: ``+1.0`` (held), ``-1.0`` (failed), ``0.0`` (no_retest or any
    unrecognized value — logged at ``warning`` since callers should only
    ever pass one of the three literal outcomes from :func:`retest_outcome`).
    """
    modifier = _RETEST_MODIFIER.get(outcome)
    if modifier is None:
        log.warning("retest_score_modifier: unrecognized outcome %r, defaulting to 0.0", outcome)
        return 0.0
    return modifier
