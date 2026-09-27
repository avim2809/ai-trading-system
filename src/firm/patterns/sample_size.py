"""Per-pattern minimum-sample confidence discount (Part A of the
2026-09-27 false-positive-rate fix).

The retrain in docs/pattern_recognition_plan.md found wildly imbalanced
historical sample counts per pattern name -- ``rising_wedge``: 1,351,
``rectangle``: 752, ... down to ``double_top``: 11, ``bull_flag``: 8,
``bear_flag``: 3, ``cup_handle``: 1, ``rounding_bottom``: 1 -- yet every
pattern name was treated with identical confidence regardless of how much
historical evidence backs it. A quality_score of 85 on a pattern with 1,351
prior confirmed examples and a quality_score of 85 on a pattern with 1
prior example are not equally trustworthy statements, even though the
scorer produces an identical number for both.

This module is a simple, honest empirical-Bayes-style shrinkage: a
pattern's contribution to the final signal is discounted toward 0 (not
"unknown" -- toward "we have essentially no track record for this shape,
treat it skeptically") in proportion to how far its historical sample
count falls short of a pre-registered "reliable" threshold. It is
deliberately NOT a fancier informative prior (e.g. shrinking toward the
pattern *family*'s average rate) -- with counts this skewed (1 to 1,351),
even the family-level estimate for `cup_handle` (n=1 total across both
`cup_handle`/`rounding_bottom` combined) would be built on almost nothing,
so a plain proportional-to-count discount is the more honest choice than a
prior that looks more sophisticated than the data actually supports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_COUNTS_FILENAME = "pattern_sample_counts.json"
DEFAULT_MIN_RELIABLE_SAMPLES = 30

_warned_paths: set[str] = set()


def save_sample_counts(counts: dict[str, int], path: str | Path) -> None:
    """Persist ``{pattern_name: historical_confirmed_count}`` as JSON,
    typically the ``meta["pattern"].value_counts()`` output from whatever
    training run just produced the ML model artifacts this discount
    accompanies (see ``scripts/train_pattern_ml.py``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): int(v) for k, v in counts.items()}, f, indent=2, sort_keys=True)
    log.info("save_sample_counts: wrote %d pattern count(s) to %s", len(counts), path)


def load_sample_counts(path: str | Path) -> dict[str, int] | None:
    """Load counts previously written by :func:`save_sample_counts`.

    Fail-soft: ``None`` (not an exception) if the file is missing or
    corrupt, logged once per path (debug afterward) -- callers should
    treat ``None`` as "no discount information available," same
    "proceed uncalibrated" convention as
    ``firm.patterns.ml.calibration.load_calibration``.
    """
    path = Path(path)
    key = str(path)
    try:
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        return {str(k): int(v) for k, v in data.items()}
    except FileNotFoundError:
        if key not in _warned_paths:
            log.warning("load_sample_counts: no sample-count file at %s -- no discount applied.", path)
            _warned_paths.add(key)
        return None
    except (json.JSONDecodeError, OSError, ValueError, TypeError) as exc:
        if key not in _warned_paths:
            log.warning("load_sample_counts: failed to read/parse %s (%s) -- no discount applied.", path, exc)
            _warned_paths.add(key)
        return None


def confidence_discount(
    pattern: str,
    sample_counts: dict[str, int] | None,
    *,
    min_reliable_samples: int = DEFAULT_MIN_RELIABLE_SAMPLES,
) -> float:
    """Multiplier in ``[0, 1]`` for how much to trust a detection of
    ``pattern``, given its historical sample count.

    ``sample_counts is None`` (no count data available at all -- e.g. no
    training run has produced one yet) returns 1.0 (no discount): this is
    a fail-soft default, not an endorsement -- see module docstring for
    why an uninformed discount would be worse than no discount at all.
    A pattern name absent from a *real* ``sample_counts`` dict is treated
    as a true zero count (maximum discount), not "unknown, skip
    discounting" -- the whole point is that a pattern the training run
    never even saw confirm is exactly the case that most needs it.
    """
    if sample_counts is None:
        return 1.0
    if min_reliable_samples <= 0:
        return 1.0
    count = sample_counts.get(pattern, 0)
    return max(0.0, min(1.0, count / min_reliable_samples))
