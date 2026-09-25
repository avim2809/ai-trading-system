"""Post-hoc probability calibration for the pattern-confirmation ML models.

The CNN/GAF validator (:mod:`firm.patterns.ml.inference`) and the XGBoost
confirmation classifier (:mod:`firm.patterns.ml.xgb_classifier` /
:mod:`firm.patterns.ml.xgb_inference`) are architecturally very different
(deep net vs. boosted trees), so their raw probability outputs are
miscalibrated in different, non-comparable ways: boosted-tree vote-averaging
tends to compress probabilities toward the middle of ``[0, 1]``, while a
cross-entropy-trained deep net tends toward overconfidence at the extremes.
Before either model's score can be ensembled, blended, or thresholded
against the other, each must be independently calibrated to true class
frequencies on held-out data -- that is what this module does.

Two techniques, one per model family (see docs/pattern_recognition_plan.md
for which model uses which):

- **Temperature scaling** (:func:`fit_temperature` / :func:`apply_temperature`)
  for the CNN: a single scalar ``T`` divides the pre-softmax logits before
  re-applying softmax. One parameter is deliberately hard to overfit even on
  a small calibration set, and (unlike a per-class transform) it cannot
  change the model's argmax/ranking -- only how confident it's allowed to be.
- **Platt / sigmoid scaling** (:func:`fit_sigmoid_calibration` /
  :func:`apply_sigmoid_calibration`) for the XGBoost classifier's raw
  per-class probability: a 2-parameter logistic fit,
  ``calibrated = 1 / (1 + exp(a * raw_score + b))`` (the classical Platt
  (1999) parameterization). Isotonic regression is a common alternative but
  needs materially more held-out samples per class to avoid overfitting the
  calibration curve itself; sigmoid scaling is the safer default given this
  system's modest sample sizes, so that is what this module implements.

This module is deliberately generic and has no opinion about *which* model's
score is being calibrated, nor about file layout: it does not hardcode
``data/models/*.calibration.json`` paths internally (see
:func:`save_calibration`/:func:`load_calibration`) -- callers/integration
code decide exactly where a given fitted calibration lives and when to
(re)fit it against a real held-out triple-barrier-labeled dataset. Fitting
itself is pure array math with no I/O, so it is trivially unit-testable
against synthetic data (see ``tests/test_calibration.py``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)

# Logged once per path so a persistently-missing/corrupt calibration sidecar
# logs a single clear WARNING instead of spamming every call site that tries
# to load it (mirrors firm.patterns.ml.inference's _warn_once pattern).
_warned_paths: set[str] = set()


def _warn_once(path: str, message: str, *args: Any) -> None:
    if path not in _warned_paths:
        _warned_paths.add(path)
        log.warning(message, *args)
    else:
        log.debug(message, *args)


# ---------------------------------------------------------------------------
# Temperature scaling (CNN, multiclass)
# ---------------------------------------------------------------------------


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically-stable ``softmax(logits / temperature)``.

    Args:
        logits: ``(n_samples, n_classes)`` raw pre-softmax scores (or a
            single ``(n_classes,)`` row -- broadcasts the same way
            ``np.exp``/axis=-1 reductions do).
        temperature: ``T > 1`` softens (reduces confidence, flattens the
            distribution toward uniform); ``T < 1`` sharpens (increases
            confidence); ``T == 1`` is a no-op (plain softmax).

    Returns:
        Calibrated class probabilities, same shape as ``logits``, each row
        summing to 1.
    """
    if temperature <= 0:
        raise ValueError(f"apply_temperature: temperature must be > 0, got {temperature}")
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / temperature
    shifted = scaled - np.max(scaled, axis=-1, keepdims=True)  # numerically stable
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    label_order: tuple[int, ...] = (-1, 0, 1),
) -> float:
    """Fit a single temperature scalar ``T`` minimizing multiclass negative
    log-likelihood on a held-out ``(logits, labels)`` calibration set.

    Args:
        logits: ``(n_samples, n_classes)`` raw pre-softmax scores, with
            ``n_classes == len(label_order)`` and column ``i`` corresponding
            to ``label_order[i]``.
        labels: ``(n_samples,)`` true class labels, values drawn from
            ``label_order``.
        label_order: The label value each logits column corresponds to, in
            column order. Defaults to the triple-barrier ``(-1, 0, 1)``
            (stop/timeout/target) convention used throughout
            ``firm.patterns.ml``.

    Returns:
        The fitted temperature ``T`` (see :func:`apply_temperature` for its
        interpretation). Found via 1-D bounded optimization
        (``scipy.optimize.minimize_scalar``, ``method="bounded"``) over
        ``T in [0.05, 20.0]`` -- generous enough to cover any realistic
        under/over-confidence while keeping the search well away from the
        ``T -> 0`` numerical singularity.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    n_classes = len(label_order)
    if logits.ndim != 2 or logits.shape[1] != n_classes:
        raise ValueError(
            f"fit_temperature: logits must have shape (n_samples, {n_classes}) to "
            f"match label_order={label_order}, got shape {logits.shape}"
        )
    if logits.shape[0] != len(labels):
        raise ValueError(
            f"fit_temperature: labels length ({len(labels)}) must match "
            f"logits.shape[0] ({logits.shape[0]})"
        )
    if logits.shape[0] == 0:
        raise ValueError("fit_temperature: logits/labels are empty -- nothing to fit")

    label_to_index = {label: i for i, label in enumerate(label_order)}
    try:
        true_idx = np.array([label_to_index[int(v)] for v in labels])
    except KeyError as exc:
        raise ValueError(
            f"fit_temperature: labels contains a value outside label_order={label_order}: {exc}"
        ) from exc

    row_idx = np.arange(logits.shape[0])

    def _nll(temperature: float) -> float:
        probs = apply_temperature(logits, temperature)
        p_true = probs[row_idx, true_idx]
        return float(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))

    result = minimize_scalar(_nll, bounds=(0.05, 20.0), method="bounded")
    temperature = float(result.x)
    log.info(
        "fit_temperature: fitted T=%.4f over %d samples (NLL=%.4f vs. T=1.0 NLL=%.4f)",
        temperature, logits.shape[0], float(result.fun), _nll(1.0),
    )
    return temperature


# ---------------------------------------------------------------------------
# Platt / sigmoid scaling (XGBoost, one-vs-rest binary probability)
# ---------------------------------------------------------------------------


def fit_sigmoid_calibration(raw_scores: np.ndarray, labels_binary: np.ndarray) -> tuple[float, float]:
    """Fit Platt/sigmoid calibration parameters ``(a, b)`` such that
    ``calibrated = 1 / (1 + exp(a * raw_score + b))`` (the classical Platt
    (1999) parameterization) minimizes binary log-loss on a held-out
    ``(raw_scores, labels_binary)`` calibration set.

    Implemented as a 1-feature logistic regression via
    ``sklearn.linear_model.LogisticRegression`` (scikit-learn is already a
    core repo dependency -- see ``pyproject.toml``), fit *unregularized*
    (``C=np.inf``) since the whole point is an unbiased calibration curve,
    not a regularized classifier. sklearn's own convention is
    ``P(y=1|x) = sigmoid(coef * x + intercept)``; converting to this
    module's ``calibrated = 1 / (1 + exp(a*x+b)) = sigmoid(-(a*x+b))``
    parameterization gives ``a = -coef``, ``b = -intercept`` (done below).

    Args:
        raw_scores: ``(n_samples,)`` uncalibrated model outputs (e.g. one
            class's raw XGBoost probability).
        labels_binary: ``(n_samples,)`` ground truth, values in ``{0, 1}``.

    Returns:
        ``(a, b)`` -- pass straight to :func:`apply_sigmoid_calibration`.

    Raises:
        ValueError: if inputs are mismatched in length, empty, or
            ``labels_binary`` doesn't contain both classes (a sigmoid fit is
            meaningless/degenerate with only one class present).
    """
    raw_scores_arr = np.asarray(raw_scores, dtype=np.float64).reshape(-1, 1)
    labels_arr = np.asarray(labels_binary, dtype=np.int64)
    if raw_scores_arr.shape[0] != len(labels_arr):
        raise ValueError(
            "fit_sigmoid_calibration: raw_scores and labels_binary must be the same length "
            f"(got {raw_scores_arr.shape[0]} vs {len(labels_arr)})"
        )
    if len(labels_arr) == 0:
        raise ValueError("fit_sigmoid_calibration: raw_scores/labels_binary are empty -- nothing to fit")
    unique = set(int(v) for v in np.unique(labels_arr))
    if not unique <= {0, 1}:
        raise ValueError(f"fit_sigmoid_calibration: labels_binary must be 0/1-valued, got {sorted(unique)}")
    if len(unique) < 2:
        raise ValueError(
            f"fit_sigmoid_calibration: labels_binary contains only one class ({unique}) -- "
            "cannot fit a meaningful sigmoid calibration from a single-class held-out set"
        )

    clf = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000)
    clf.fit(raw_scores_arr, labels_arr)
    coef = float(clf.coef_[0][0])
    intercept = float(clf.intercept_[0])
    a, b = -coef, -intercept
    log.info(
        "fit_sigmoid_calibration: fitted a=%.4f b=%.4f over %d samples",
        a, b, len(labels_arr),
    )
    return a, b


def apply_sigmoid_calibration(raw_scores: np.ndarray, a: float, b: float) -> np.ndarray:
    """Apply a fitted Platt/sigmoid transform:
    ``calibrated = 1 / (1 + exp(a * raw_score + b))``, computed as
    ``scipy.special.expit(-(a * raw_score + b))`` for numerical stability
    (``expit`` avoids overflow in ``exp`` for large-magnitude arguments).
    """
    raw_scores_arr = np.asarray(raw_scores, dtype=np.float64)
    return expit(-(a * raw_scores_arr + b))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_calibration(params: dict, path: str | Path) -> None:
    """Persist fitted calibration parameters as a plain JSON sidecar.

    ``params`` is written as-is (e.g. ``{"type": "temperature",
    "temperature": 1.34}`` or ``{"type": "sigmoid", "a": ..., "b": ...}``) --
    this module doesn't prescribe the exact schema beyond recommending a
    ``"type"`` discriminator key, since it doesn't know which model/path a
    caller is calibrating. Creates parent directories as needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f)
    log.info("save_calibration: wrote %s calibration to %s", params.get("type", "?"), path)


def load_calibration(path: str | Path) -> dict | None:
    """Load calibration parameters previously written by
    :func:`save_calibration`.

    Fail-soft: returns ``None`` (rather than raising) when the file is
    missing or its JSON is corrupt/unreadable, logging a warning the first
    time (debug afterward) per path -- mirrors
    ``firm.patterns.ml.inference``'s ``_warn_once`` pattern, so a
    persistently-absent calibration sidecar (e.g. calibration simply hasn't
    been fitted yet for a given model) doesn't spam the log every call.
    Callers should treat ``None`` as "proceed uncalibrated."
    """
    path = Path(path)
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        _warn_once(
            str(path),
            "load_calibration: no calibration file at %s -- proceeding uncalibrated.",
            path,
        )
        return None
    except (json.JSONDecodeError, OSError) as exc:
        _warn_once(
            str(path),
            "load_calibration: failed to read/parse calibration file at %s (%s) -- "
            "proceeding uncalibrated.",
            path, exc,
        )
        return None
