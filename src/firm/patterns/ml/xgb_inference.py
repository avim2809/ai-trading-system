"""Live-path ONNX inference for the XGBoost pattern-confirmation classifier
(:mod:`firm.patterns.ml.xgb_classifier`) -- the piece that lets
``firm.strategies.pattern_recognition`` (or any other caller) score a
confirmed chart-pattern match's win/timeout/stop probabilities without
needing ``xgboost`` itself installed.

Why this module exists separately from ``xgb_classifier.py``: that module's
own ``load_onnx``/``predict_proba_onnx`` are documented, in their
docstrings, as "isolated ML environment only" (``.venv-ml``) -- written when
neither ``onnxmltools`` (needed only for *exporting* a booster to ONNX) nor
``onnxruntime`` had a Python 3.14 wheel. That's now stale for
``onnxruntime`` specifically: a genuine ``cp314`` wheel exists today
(confirmed importable in this repo's main ``.venv`` -- ``onnxruntime
1.30.0`` -- while writing this module), exactly the same situation
documented in :mod:`firm.patterns.ml.inference`'s module docstring for the
CNN model. ``onnxmltools``/``xgboost`` themselves are still ``.venv-ml``
only (no export/train capability needed here at all -- see below), so this
module only ever imports ``onnxruntime``, lazily, and only inside
:func:`_load_session`.

This module reads the exact same ``<model>.onnx`` + ``<model>.onnx.labels.json``
sidecar pair :func:`firm.patterns.ml.xgb_classifier.export_onnx` produces
(see that function's docstring for the sidecar's format/purpose) -- it does
not duplicate any export logic, only the *loading*/inference side, wrapped
in this module's own fail-soft, warn-once, lazily-``lru_cache``-d shell so
it matches :mod:`firm.patterns.ml.inference`'s calling convention exactly
(same ``is_available``/``reset_cache`` shape, same "never raises, ``None``
on any failure" contract).

**Expected ONNX graph shape:** the on-disk ``data/models/pattern_xgb.onnx``
artifact (as produced by ``onnxmltools.convert_xgboost`` -- see
``xgb_classifier.export_onnx``) takes one input, ``(n_samples, n_features)``
float32, and produces *two* outputs -- ``label`` (int64, argmax class index)
and ``probabilities`` (float32, ``(n_samples, len(present_labels))``) --
this is sklearn-onnx's standard classifier graph shape (verified against
the real on-disk artifact: input ``[None, 47]``, outputs
``label: [None]`` and ``probabilities: [None, 3]``). Only the second output
is used here; the predicted label is ignored (:func:`score_pattern_confirmation`
returns full probabilities, exactly like ``xgb_classifier.predict_proba``
does, not just an argmax).

**Feature vector contract:** :func:`score_pattern_confirmation` takes an
already-built, already-ordered feature vector -- it does *no* feature
engineering of its own, exactly like ``xgb_classifier.predict_proba`` itself
takes a feature matrix rather than raw pattern data. The authoritative
source of the expected shape/column order is
:func:`firm.patterns.ml.feature_engineering.build_features` (equivalently,
one row of :func:`firm.patterns.ml.feature_engineering.build_feature_frame`'s
output) -- see :func:`score_pattern_confirmation`'s docstring for the exact,
verified 47-column layout the currently-trained on-disk model expects.

**Fail-soft contract:** every public function here is exception-safe. If
``onnxruntime`` isn't installed, the model/sidecar files are missing or
corrupt, or inference raises for any other reason,
:func:`score_pattern_confirmation` logs once (WARNING the first time, DEBUG
afterward, so a persistently missing/broken model doesn't spam the live log
every cycle) and returns ``None`` -- callers must treat ``None`` as "no
ML-based confirmation score available right now," never as an error to
propagate.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: Same fixed label order as xgb_classifier.LABELS / cnn's inference.LABELS:
#: column 0 = "stop hit" (-1), column 1 = "timeout" (0), column 2 =
#: "target hit" (+1).
LABELS: tuple[int, ...] = (-1, 0, 1)
_LABEL_TO_INDEX: dict[int, int] = {label: i for i, label in enumerate(LABELS)}

#: Model artifacts are a shared code-level asset (like pattern_cnn.onnx,
#: pattern_ppo.zip), not per-instance live state -- unlike
#: firm.live.pattern_scan_job's FIRM_DATA_DIR-relative history DB, this path
#: is deliberately *not* prefixed by FIRM_DATA_DIR: both the IBKR (:8000,
#: FIRM_DATA_DIR unset -> "data") and Alpaca (:8001, FIRM_DATA_DIR=
#: "data_alpaca") instances share one checkout and should read the same
#: trained model from the same place.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "data" / "models" / "pattern_xgb.onnx"

# Logged once per (path, reason) so a persistently-missing/broken model logs
# a single clear WARNING instead of spamming every live cycle.
_warned_paths: set[str] = set()


def _warn_once(path: str, message: str, *args: Any) -> None:
    if path not in _warned_paths:
        _warned_paths.add(path)
        log.warning(message, *args)
    else:
        log.debug(message, *args)


@lru_cache(maxsize=4)
def _load_session(model_path: str):
    """Lazy singleton: the ONNX session (+ label sidecar) is constructed at
    most once per distinct ``model_path`` per process, not per call --
    ``lru_cache`` gives us that for free, keyed on the (string) path.
    Returns ``None`` on *any* failure (missing onnxruntime, missing/corrupt
    model file, missing/corrupt sidecar) rather than raising -- see module
    docstring's fail-soft contract.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        _warn_once(
            model_path,
            "pattern XGBoost confirmation scoring unavailable: onnxruntime is not "
            "installed -- no ML-based confirmation score will be produced.",
        )
        return None

    path = Path(model_path)
    if not path.exists():
        _warn_once(
            model_path,
            "pattern XGBoost confirmation scoring unavailable: model file not found "
            "at %s -- no ML-based confirmation score will be produced.",
            path,
        )
        return None

    sidecar = path.with_suffix(path.suffix + ".labels.json")
    try:
        with open(sidecar) as f:
            present_labels = tuple(json.load(f))
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    except Exception:
        _warn_once(
            model_path,
            "pattern XGBoost confirmation scoring unavailable: failed to load model/"
            "sidecar at %s -- no ML-based confirmation score will be produced.",
            path,
        )
        log.debug("XGBoost ONNX model load failure detail", exc_info=True)
        return None

    log.info(
        "pattern_recognition: XGBoost confirmation scoring ACTIVE (onnxruntime session "
        "loaded from %s, labels=%s)", path, present_labels,
    )
    return session, present_labels


def is_available(model_path: str | os.PathLike = DEFAULT_MODEL_PATH) -> bool:
    """Best-effort availability check (e.g. for one summary log line at
    startup) -- triggers (and caches) the same lazy load
    :func:`score_pattern_confirmation` would.
    """
    return _load_session(str(model_path)) is not None


def score_pattern_confirmation(
    features: np.ndarray,
    *,
    model_path: str | os.PathLike = DEFAULT_MODEL_PATH,
    apply_calibration: dict | None = None,
) -> tuple[float, float, float] | None:
    """XGBoost-based pattern-confirmation scoring for ONE pattern match.

    Args:
        features: A 1D feature vector, shape ``(47,)`` for the currently
            trained on-disk ``data/models/pattern_xgb.onnx`` artifact --
            **the exact column order must match**
            :func:`firm.patterns.ml.feature_engineering.build_features`'s
            dict-insertion order (equivalently,
            :func:`~firm.patterns.ml.feature_engineering.build_feature_frame`'s
            DataFrame column order for a single row), which is the
            authoritative source of truth this function defers to rather
            than re-deriving/hardcoding feature names itself. As of this
            writing that order is (verified against
            ``scripts/train_pattern_ml.py``'s ``build_dataset``, which feeds
            ``build_features`` output straight into
            ``xgb_classifier.train`` with no reordering, and against the
            real on-disk model's ``input: [None, 47]`` ONNX graph shape):
            ``direction_sign, entry, stop, target, fit_quality,
            geometry_tolerance_used, volume_ratio, volume_ratio_missing,
            duration_bars, follow_through_atr, risk_reward, quality_score,
            num_pivots, confirm_index, stop_distance_pct,
            target_distance_pct, score_geometry, score_trendline_fit,
            score_volume_confirmation, score_duration, score_follow_through,
            score_total, family_reversal, family_triangle,
            family_continuation, family_cup_handle, pattern_<name>`` (one
            column per entry of
            ``firm.patterns.ml.feature_engineering.PATTERN_NAMES``, in that
            tuple's order) ``, pre_pattern_return, pre_pattern_volatility,
            atr_pct, ohlcv_context_available``. A 2D ``(1, n_features)``
            array is also accepted (squeezed internally) for callers that
            already batch a single row through a DataFrame-derived array.
            **This function pushes all feature-engineering responsibility
            to the caller** -- it performs no pattern/OHLCV processing of
            its own, exactly like ``xgb_classifier.predict_proba`` takes a
            feature matrix rather than raw pattern data (see module
            docstring). A model retrained against a different feature
            schema will silently produce meaningless scores if this
            contract drifts -- there is no way for this function to
            self-detect a column-order mismatch beyond the raw feature
            *count*, which it does not even attempt to validate (accepts
            whatever shape the loaded ONNX graph's input allows).
        model_path: Path to the ``.onnx`` file (its ``.onnx.labels.json``
            sidecar is derived automatically, same convention as
            ``xgb_classifier.export_onnx``/``firm.patterns.ml.inference``).
        apply_calibration: Optional, a dict previously returned by
            :func:`firm.patterns.ml.calibration.load_calibration` (e.g. a
            fitted Platt/sigmoid calibration for this specific model). When
            given and ``apply_calibration["type"] == "sigmoid"``, the raw
            ``p_target`` (target-hit) probability is passed through
            :func:`firm.patterns.ml.calibration.apply_sigmoid_calibration`
            before being returned -- ``p_stop``/``p_timeout`` are left as
            the model's raw output either way (this function calibrates the
            +1/target class only, matching the meta-labeling use case of
            "how much do I trust this specific win-probability", not a full
            joint-distribution recalibration). Any other/unrecognized
            ``"type"`` (e.g. ``"temperature"`` -- meaningful for the CNN's
            pre-softmax logits, but this ONNX graph only ever exposes
            already-softmaxed probabilities, no logits) is logged at debug
            and skipped rather than raising -- calibration application is
            optional and best-effort, never a reason to fail the whole
            score. Defaults to ``None`` (no calibration applied).

    Returns:
        ``(p_stop, p_timeout, p_target)`` -- plain floats in ``[0, 1]``,
        matching the fixed ``(-1, 0, +1)`` :data:`LABELS` column order used
        throughout ``xgb_classifier.py`` -- or ``None`` on any failure
        (``onnxruntime``/model unavailable, malformed ``features``, or any
        unexpected inference-time error). Never raises.
    """
    loaded = _load_session(str(model_path))
    if loaded is None:
        return None
    session, present_labels = loaded

    try:
        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        elif arr.ndim != 2 or arr.shape[0] != 1:
            raise ValueError(
                f"score_pattern_confirmation: expected a single feature row "
                f"(shape (n_features,) or (1, n_features)), got shape {arr.shape}"
            )

        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: arr})
        # sklearn-onnx classifier graphs (onnxmltools.convert_xgboost) emit
        # two outputs: [0] predicted label (unused here), [1] per-class
        # probabilities over `present_labels` (see module docstring).
        proba = np.asarray(outputs[1], dtype=np.float64)[0]

        out = np.zeros(len(LABELS))
        for col, label in enumerate(present_labels):
            out[_LABEL_TO_INDEX[label]] = proba[col]
        p_stop = float(out[_LABEL_TO_INDEX[-1]])
        p_timeout = float(out[_LABEL_TO_INDEX[0]])
        p_target = float(out[_LABEL_TO_INDEX[1]])
    except Exception:
        _warn_once(
            str(model_path) + ":inference_error",
            "pattern XGBoost confirmation scoring failed during inference -- no "
            "ML-based confirmation score will be produced.",
        )
        log.debug("XGBoost ONNX inference failure detail", exc_info=True)
        return None

    if apply_calibration is not None:
        cal_type = apply_calibration.get("type")
        if cal_type == "sigmoid":
            from firm.patterns.ml.calibration import apply_sigmoid_calibration

            try:
                calibrated = apply_sigmoid_calibration(
                    np.array([p_target]), apply_calibration["a"], apply_calibration["b"],
                )
                p_target = float(np.clip(calibrated[0], 0.0, 1.0))
            except Exception:
                log.debug(
                    "score_pattern_confirmation: failed to apply sigmoid calibration "
                    "%s -- returning raw p_target instead", apply_calibration, exc_info=True,
                )
        else:
            log.debug(
                "score_pattern_confirmation: apply_calibration type=%r not applicable to "
                "this ONNX graph's already-softmaxed output -- skipping calibration",
                cal_type,
            )

    return p_stop, p_timeout, p_target


def reset_cache() -> None:
    """Test-only: clear the lazy-singleton session cache and warn-once state
    so a test can exercise :func:`_load_session`'s cold-start path again
    (e.g. after monkeypatching ``onnxruntime`` or the model path between
    cases). Not called anywhere in the live path.
    """
    _load_session.cache_clear()
    _warned_paths.clear()
