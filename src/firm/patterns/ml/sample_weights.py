"""Sample-uniqueness weighting (Lopez de Prado, *Advances in Financial
Machine Learning*, ch. 4) for overlapping triple-barrier-labeled events.

Why this exists (Part B item 3, 2026-09-27): ``scripts/train_pattern_ml.py``'s
own docstring already calls its rolling-cutoff-every-``step_bars`` sampling
"highly autocorrelated" -- the same underlying pattern is typically still
"the most recent one" across several consecutive cutoffs, so consecutive
training rows are frequently near-duplicates of each other, and any two
events whose outcome-determining windows (``[confirm_index, exit_index]``,
see :func:`firm.patterns.ml.labeling.label_triple_barrier_with_exit`)
overlap in time share information and are not independent draws. Treating
every row as an equally-weighted i.i.d. sample (the status quo before this
module -- ``sample_weight`` had zero hits anywhere in this codebase)
overstates the effective sample size and lets a cluster of overlapping,
correlated events dominate a fit disproportionately.

De Prado's fix: each event's *average uniqueness* is the mean, over every
bar its own span touches, of ``1 / concurrency`` at that bar (how many
labeled events -- including itself -- are simultaneously "in flight" then).
An event with no overlapping neighbors anywhere in its span is fully
unique (weight 1.0); heavy overlap pulls its weight toward 0. Passed as
``sample_weight`` to a classifier's ``fit()``, this down-weights redundant,
clustered events relative to genuinely distinct ones -- it does not drop
any row (unlike de Prado's further sequential-bootstrap resampling, ch.
4.5, which is out of scope here: this module implements the weighting
step alone, since that is what the plan this was scoped against calls for
and what closes the "sample_weight has zero hits" gap; resampling can be
layered on top later if the weighting alone proves insufficient).

Scope note: concurrency is computed **per symbol**, not across the whole
universe. Two events from different symbols that happen to overlap in
calendar time do share some market-wide information too, in principle, but
the concrete autocorrelation problem this codebase's own docs identify is
specifically the *same-symbol* rolling-window overlap quoted above --
cross-symbol concurrency would be a real (if more marginal) refinement,
not implemented here to keep this change focused and testable.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

log = logging.getLogger(__name__)


def average_uniqueness(spans: Sequence[tuple[int, int]]) -> np.ndarray:
    """De Prado ch. 4.2's average-uniqueness weight for each of N events.

    Args:
        spans: One ``(t0, t1)`` integer pair per event -- inclusive bar
            positions the event "occupies" (e.g. ``(match.confirm_index,
            exit_index)`` from
            :func:`firm.patterns.ml.labeling.label_triple_barrier_with_exit`).
            ``t0 <= t1`` for every span; a zero-length event (``t0 == t1``)
            is valid (see that function's own docstring for when this
            happens) and always gets weight ``1.0`` unless another event
            also touches that exact same single bar. All spans must share
            the same integer coordinate space (e.g. bar positions into the
            *same symbol's* OHLCV array) -- comparing spans from different
            symbols/series here would be meaningless (see module docstring).

    Returns:
        A ``(len(spans),)`` float array in ``(0, 1]``, same order as
        ``spans``. Empty input returns an empty array rather than raising.
    """
    if not spans:
        return np.array([], dtype=float)

    t_min = min(t0 for t0, _t1 in spans)
    t_max = max(t1 for _t0, t1 in spans)
    width = t_max - t_min + 1
    concurrency = np.zeros(width, dtype=float)
    for t0, t1 in spans:
        concurrency[t0 - t_min : t1 - t_min + 1] += 1.0

    weights = np.empty(len(spans), dtype=float)
    for i, (t0, t1) in enumerate(spans):
        weights[i] = float(np.mean(1.0 / concurrency[t0 - t_min : t1 - t_min + 1]))
    return weights


def sample_weights_by_group(
    groups: Sequence[str], spans: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Batch convenience: apply :func:`average_uniqueness` independently
    within each distinct value of ``groups`` (e.g. symbol), then reassemble
    the result in the SAME row order as the input -- the shape
    ``scripts/train_pattern_ml.py``'s ``build_dataset`` (one row per
    confirmed match across many symbols, concatenated) actually needs, since
    spans from different symbols are not comparable (see module docstring).

    ``groups``/``spans`` must be the same length (one group label + one span
    per row). Empty input returns an empty array.
    """
    n = len(groups)
    if n != len(spans):
        raise ValueError(f"sample_weights_by_group: groups (n={n}) and spans (n={len(spans)}) must be the same length")
    if n == 0:
        return np.array([], dtype=float)

    weights = np.empty(n, dtype=float)
    groups_arr = np.asarray(groups)
    for group in np.unique(groups_arr):
        idx = np.flatnonzero(groups_arr == group)
        group_spans = [spans[i] for i in idx]
        weights[idx] = average_uniqueness(group_spans)
    return weights
