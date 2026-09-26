"""Classification-quality metrics -- precision/recall/F1/PR-AUC/confusion
matrix/Brier score/reliability.

Every other module in ``firm.eval`` measures portfolio *returns*
(Sharpe/drawdown/PBO/DSR); nothing anywhere in this repo previously
measured whether a *detector's* positive/negative calls, or a *model's*
probability estimates, are actually any good (see
``scripts/benchmark_pattern_detectors.py``, the first consumer). Thin
wrappers over ``sklearn.metrics`` (a hard dependency already -- no new
install) rather than hand-rolled math, since these are well-understood,
easy-to-get-subtly-wrong computations with no reason to reimplement.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

#: Score bands reliability_diagram_bins groups into by default -- matching
#: scripts/analyze_pattern_scan_outcomes.py's _QUALITY_BANDS boundaries
#: (duplicated, not imported: eval/ is a src/ package, scripts/ is not, and
#: this repo's convention is not to cross-import between them) so a
#: 0-100 quality_score and a 0-1 probability bin identically once scaled.
DEFAULT_RELIABILITY_BINS: tuple[float, ...] = (0.0, 0.6, 0.7, 0.8, 0.9, 1.0)


def binary_classification_report(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    pos_label: int = 1,
) -> dict[str, Any]:
    """Precision/recall/F1/confusion-matrix for one binary decision.

    ``zero_division=0`` throughout (rather than raising or warning): a
    detector that never fires (or never fires correctly) on a small
    benchmark slice is a real, reportable "0", not an error. An empty
    ``y_true``/``y_pred`` (no data at all, distinct from "data but no
    positives") reports ``n=0`` with every metric ``None`` -- "undefined,"
    not a misleading 0.0 that would look like a failing detector.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {
            "precision": None, "recall": None, "f1": None,
            "confusion_matrix": [], "confusion_matrix_labels": [],
            "n": 0, "n_positive": 0,
        }
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()) | {pos_label, 0})
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "precision": float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "n": int(len(y_true)),
        "n_positive": int(np.sum(y_true == pos_label)),
    }


def pr_auc(y_true: Sequence[int] | np.ndarray, y_score: Sequence[float] | np.ndarray) -> float | None:
    """Precision-recall AUC (average precision) for a scored ranking.

    ``None`` (not raised) when ``y_true`` has only one class -- PR-AUC is
    undefined there, and a benchmark slice with zero negatives (or zero
    positives) is a real, expected input shape, not an error condition.
    """
    y_true = np.asarray(y_true)
    if len(set(y_true.tolist())) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def brier_score(y_true: Sequence[int] | np.ndarray, y_prob: Sequence[float] | np.ndarray) -> float:
    """Mean squared error between predicted probabilities and binary
    outcomes -- lower is better, 0.0 is a perfect forecaster, 0.25 is the
    score of an uninformative always-0.5 forecaster on a balanced sample.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_diagram_bins(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    bins: Sequence[float] = DEFAULT_RELIABILITY_BINS,
) -> list[dict[str, Any]]:
    """Bucket predictions by ``y_prob`` and report each bucket's mean
    predicted probability vs. its actual observed positive rate -- a
    well-calibrated model has the two tracking closely in every bucket
    with enough samples; a systematically over- or under-confident one
    shows a persistent gap.

    Buckets with zero samples are still reported (``n=0``,
    ``mean_predicted``/``observed_rate`` both ``None``) rather than
    silently omitted, so a caller can see exactly which score range this
    benchmark slice had no coverage for.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    out: list[dict[str, Any]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < bins[-1] else (y_prob >= lo) & (y_prob <= hi)
        n = int(np.sum(mask))
        out.append({
            "band": f"[{lo:g}, {hi:g}{']' if hi >= bins[-1] else ')'}",
            "n": n,
            "mean_predicted": float(np.mean(y_prob[mask])) if n > 0 else None,
            "observed_rate": float(np.mean(y_true[mask])) if n > 0 else None,
        })
    return out
