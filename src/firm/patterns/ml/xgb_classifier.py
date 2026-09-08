"""Thin XGBoost wrapper for the pattern-confirmation classifier (Phase 4).

Deliberately minimal: :func:`train`, :func:`predict_proba`, :func:`save`, and
:func:`load`.

``xgboost`` is an *optional* dependency (see the ``patterns_ml`` extra in
pyproject.toml) -- deliberately not imported at module scope, so the rest of
``firm.patterns.ml`` (feature engineering, labeling) and this module's own
docstrings/constants stay importable even when it isn't installed. Only
:func:`train` actually needs it (lazily, via :func:`_require_xgboost`), and
raises a clear ``ImportError`` pointing at the extra if it's missing, rather
than a confusing ``ModuleNotFoundError`` deep in a traceback.

Why :func:`train` doesn't just hand back a bare ``xgboost.XGBClassifier``:
the installed xgboost version (3.x) requires ``y`` at fit time to already be
contiguous integers ``0..n_classes-1`` matching exactly what ``np.unique(y)``
finds -- it does **not** auto-encode arbitrary label values (not even
``-1``/``0``/``1``) the way e.g. a plain ``LabelEncoder`` would; passing
labels with a "gap" (verified empirically: even just two present classes
whose *values* aren't literally ``{0, 1}`` raises ``ValueError: Invalid
classes inferred from unique values of y``) fails outright. Triple-barrier
labels are ``-1``/``0``/``1`` (see :mod:`firm.patterns.ml.labeling`), and a
real training call's ``y`` slice (e.g. after a train/test split, or simply a
small/imbalanced sample) can easily miss the middle (``0``, timeout) class
entirely -- so a *fixed* universal encoding (e.g. always ``-1->0, 0->1,
1->2``) is exactly what reintroduces that gap whenever the middle class is
absent. Instead, :func:`train` encodes only the classes actually present in
this specific call to a contiguous ``0..k-1`` range (sorted by original
label value) and returns a small :class:`_FittedPatternModel` bundle
carrying that mapping alongside the fitted booster, so :func:`predict_proba`
can always decode back to the fixed 3-column ``(-1, 0, +1)`` layout
regardless of which subset of classes this particular fit happened to see.

Persistence uses plain ``pickle`` (as scoped for this pass) rather than
XGBoost's own native ``.json``/``.ubj`` format or ONNX export -- pickling
:class:`_FittedPatternModel` works fine since both its fields (the
XGBClassifier and a plain tuple) are themselves picklable. ONNX would be the
natural next step for low-latency serving (a fixed, versioned graph instead
of a pickled Python object tied to this exact xgboost version) but pulls in
another dependency (``onnxruntime``) not justified for this unsupervised
pass -- see the task report for this initiative.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Triple-barrier label values (see firm.patterns.ml.labeling), in the fixed
#: column order every :func:`predict_proba` result follows -- column 0 is
#: "stop hit" (-1), column 1 is "timeout" (0), column 2 is "target hit" (+1).
LABELS: tuple[int, ...] = (-1, 0, 1)
_LABEL_TO_INDEX: dict[int, int] = {label: i for i, label in enumerate(LABELS)}

# Conservative defaults for a shared, resource-constrained (2-core) VPS that
# is also running live paper-trading engines: n_jobs=1 so a human kicking off
# a real training run (scripts/train_pattern_ml.py) doesn't inadvertently
# saturate both cores and starve the live engines. Override via the `params`
# argument for a one-off run on a bigger box.
#
# `objective`/`num_class` are deliberately *not* set here: XGBClassifier
# auto-selects `binary:logistic` or `multi:softprob` (and the matching
# `num_class`) from however many classes a given `train()` call's `y`
# actually contains -- forcing `multi:softprob` unconditionally here breaks
# the (common, e.g. after a train/test split) 2-present-class case, which
# needs `binary:logistic` instead (verified empirically: with an explicit
# `multi:softprob` override and only 2 classes present, XGBoost's C API
# raises `num_class should be greater equal to 1` -- it does not also
# auto-derive `num_class` once the caller has overridden `objective`).
_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": 1,
    "tree_method": "hist",
}


@dataclass
class _FittedPatternModel:
    """A fitted ``xgboost.XGBClassifier`` plus the label encoding actually
    used to fit it (see module docstring for why this indirection exists).
    Opaque to callers -- treat it as the "model" object :func:`train`
    returns and pass it straight through to :func:`predict_proba`/
    :func:`save`.
    """

    booster: Any
    present_labels: tuple[int, ...]  # sorted labels seen at fit time; booster class i <-> present_labels[i]


def _require_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - exercised only when the extra is absent
        raise ImportError(
            "xgboost is required for firm.patterns.ml.xgb_classifier. Install the "
            "optional extra: pip install '.[patterns_ml]' (see pyproject.toml) -- "
            "note this currently pulls a large unused nvidia-nccl-cu13 GPU wheel as "
            "a transitive dependency on Linux; `pip uninstall -y nvidia-nccl-cu13` "
            "afterward is safe (CPU training/inference never touches it)."
        ) from exc
    return xgb


def train(X: pd.DataFrame | np.ndarray, y, *, params: dict[str, Any] | None = None) -> _FittedPatternModel:
    """Fit an XGBoost multi-class classifier over triple-barrier labels.

    Args:
        X: Feature matrix -- a DataFrame (e.g. from
            :func:`firm.patterns.ml.feature_engineering.build_feature_frame`)
            or a 2D array, one row per confirmed pattern match.
        y: Matching sequence of int labels drawn from :data:`LABELS`
            (``-1``/``0``/``1``; see :mod:`firm.patterns.ml.labeling`). Need
            not contain all three values (see module docstring).
        params: Overrides merged over :data:`_DEFAULT_PARAMS` (e.g. to raise
            ``n_jobs`` on a bigger box, or tune ``n_estimators``).

    Returns:
        A :class:`_FittedPatternModel` -- pass it straight to
        :func:`predict_proba`/:func:`save`.
    """
    xgb = _require_xgboost()
    y_arr = np.asarray(y)
    if len(y_arr) == 0:
        raise ValueError("train: y is empty -- nothing to fit")
    unknown = set(int(v) for v in np.unique(y_arr)) - set(LABELS)
    if unknown:
        raise ValueError(f"train: y contains labels outside {LABELS}: {sorted(unknown)}")

    present_labels = tuple(sorted(int(v) for v in np.unique(y_arr)))
    if len(present_labels) < 2:
        log.warning(
            "train: only one distinct label present (%s) across %d rows -- "
            "the fitted model will be degenerate (always predicts that class)",
            present_labels, len(y_arr),
        )
    label_to_compact = {label: i for i, label in enumerate(present_labels)}
    y_compact = np.array([label_to_compact[int(v)] for v in y_arr])

    model_params = {**_DEFAULT_PARAMS, **(params or {})}
    booster = xgb.XGBClassifier(**model_params)
    booster.fit(X, y_compact)
    n_features = X.shape[1] if hasattr(X, "shape") else len(X[0])
    log.info(
        "xgb_classifier.train: fit on %d rows x %d features, labels present=%s",
        len(y_compact), n_features, present_labels,
    )
    return _FittedPatternModel(booster=booster, present_labels=present_labels)


def predict_proba(model: _FittedPatternModel, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Class probabilities, as an ``(n_samples, 3)`` array whose columns are
    *always* ordered ``(-1, 0, +1)`` (matching :data:`LABELS`) regardless of
    which subset of classes ``model`` actually saw at fit time -- a label
    the model never saw during :func:`train` just comes back as an all-zero
    probability column.
    """
    raw = model.booster.predict_proba(X)
    out = np.zeros((raw.shape[0], len(LABELS)))
    for col, label in enumerate(model.present_labels):
        out[:, _LABEL_TO_INDEX[label]] = raw[:, col]
    return out


def predict_label(model: _FittedPatternModel, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Convenience: argmax of :func:`predict_proba`, decoded back to
    :data:`LABELS` values (``-1``/``0``/``1``).
    """
    proba = predict_proba(model, X)
    idx = np.argmax(proba, axis=1)
    return np.array([LABELS[i] for i in idx])


def save(model: _FittedPatternModel, path: str | Path) -> None:
    """Persist a fitted model via plain pickle (see module docstring for why
    not ONNX/native format in this pass). Creates parent directories as
    needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info("xgb_classifier.save: wrote model to %s", path)


def load(path: str | Path) -> _FittedPatternModel:
    """Load a model previously written by :func:`save`."""
    path = Path(path)
    with open(path, "rb") as f:
        model = pickle.load(f)
    log.info("xgb_classifier.load: loaded model from %s", path)
    return model
