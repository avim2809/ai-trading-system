"""Live-path ONNX inference for the CNN/GAF pattern validator
(:mod:`firm.patterns.ml.cnn_validator`) -- the piece that lets
``firm.strategies.pattern_recognition`` use the CNN's learned quality score
instead of the rule-based scanner's own geometric ``quality_score``.

Why this module exists separately from ``cnn_validator.py``: that module
hard-requires ``torch``/``pyts`` (via :func:`_require_torch`/
:func:`_require_pyts`) and is documented, repo-wide, as **isolated
``.venv-ml`` only** -- neither package has a Python 3.14 wheel, and this box
runs 3.14.4 in its main venv. This module instead runs entirely in the main
venv, using only ``onnxruntime`` (a genuine ``cp314`` wheel now exists --
confirmed via ``pip index versions``/``pip download`` on 2026-09-20; the
"no Python 3.14 wheel" note in docs/pattern_recognition_plan.md §7.3/§7.4 is
now stale) plus a hand-rolled, numerically-verified replica of the GAF
encoding step (see :func:`_encode_gasf`) so ``pyts`` itself -- still no 3.14
wheel, and not worth adding as a new production dependency for one array
formula -- never needs to be installed here either.

**GAF replica correctness:** :func:`_encode_gasf` reproduces
``pyts.image.GramianAngularField(method="summation")`` bit-for-bit
(``max abs diff == 0.0`` in a direct numerical comparison run under
``.venv-ml``) for the one case this module needs: ``window_bars ==
image_size`` (the trained model's fixed default, 32 == 32), which makes
pyts's internal PAA step a no-op. :func:`score_pattern_quality` refuses to
guess for any other combination -- see its docstring.

**Fail-soft contract:** every public function here is exception-safe. If
``onnxruntime`` isn't installed, the model/sidecar files are missing or
corrupt, or inference raises for any other reason, :func:`score_pattern_quality`
logs once (at WARNING the first time, DEBUG afterward, so a persistently
missing model doesn't spam the live log every cycle) and returns ``None`` --
callers (``pattern_recognition.py``) must treat ``None`` as "fall back to the
rule-based quality score," never as an error to propagate.
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from firm.patterns.ml.cnn_validator import DEFAULT_IMAGE_SIZE, DEFAULT_WINDOW_BARS, extract_window

log = logging.getLogger(__name__)

#: Same fixed label order as cnn_validator.LABELS / xgb_classifier.LABELS.
LABELS: tuple[int, ...] = (-1, 0, 1)
_LABEL_TO_INDEX: dict[int, int] = {label: i for i, label in enumerate(LABELS)}

#: Model artifacts are a shared code-level asset (like pattern_xgb.pkl,
#: pattern_ppo.zip), not per-instance live state -- unlike
#: firm.live.pattern_scan_job's FIRM_DATA_DIR-relative history DB, this path
#: is deliberately *not* prefixed by FIRM_DATA_DIR: both the IBKR (:8000,
#: FIRM_DATA_DIR unset -> "data") and Alpaca (:8001, FIRM_DATA_DIR=
#: "data_alpaca") instances share one checkout and should read the same
#: trained model from the same place.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "data" / "models" / "pattern_cnn.onnx"

# Logged once per (path, reason) so a persistently-missing/broken model logs
# a single clear WARNING instead of spamming every live cycle.
_warned_paths: set[str] = set()


def _warn_once(path: str, message: str, *args: Any) -> None:
    if path not in _warned_paths:
        _warned_paths.add(path)
        log.warning(message, *args)
    else:
        log.debug(message, *args)


def _encode_gasf(window_0_1: np.ndarray) -> np.ndarray:
    """Gramian Angular Summation Field for an already ``extract_window``-
    normalized (min=0, max=1 within the window) 1D array.

    Reproduces ``pyts.image.GramianAngularField(method="summation")``'s
    default ``sample_range=(-1, 1)`` rescale + GASF formula exactly, for the
    ``window_bars == image_size`` case (PAA is then the identity transform).
    Composing two min-max rescales -- ``extract_window``'s raw-price -> [0, 1]
    followed by pyts's own [0, 1] (well, whatever range it's handed) -> [-1, 1]
    -- onto data whose observed min/max **is** the window's own min/max both
    times collapses to a single direct rescale, i.e. ``2*x - 1``; verified
    numerically against real ``pyts`` output (max abs diff 0.0).
    """
    x = np.clip(2.0 * window_0_1 - 1.0, -1.0, 1.0)
    sin_part = np.sqrt(np.clip(1.0 - x**2, 0.0, 1.0))
    return np.outer(x, x) - np.outer(sin_part, sin_part)


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
            "pattern_recognition CNN scoring unavailable: onnxruntime is not "
            "installed -- falling back to rule-based quality scoring.",
        )
        return None

    path = Path(model_path)
    if not path.exists():
        _warn_once(
            model_path,
            "pattern_recognition CNN scoring unavailable: model file not found "
            "at %s -- falling back to rule-based quality scoring.",
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
            "pattern_recognition CNN scoring unavailable: failed to load model/"
            "sidecar at %s -- falling back to rule-based quality scoring.",
            path,
        )
        log.debug("CNN model load failure detail", exc_info=True)
        return None

    log.info(
        "pattern_recognition: CNN scoring ACTIVE (onnxruntime session loaded from %s, "
        "labels=%s)", path, present_labels,
    )
    return session, present_labels


def is_available(model_path: str | os.PathLike = DEFAULT_MODEL_PATH) -> bool:
    """Best-effort availability check (e.g. for one summary log line in
    ``pattern_recognition.generate()``) -- triggers (and caches) the same
    lazy load :func:`score_pattern_quality` would.
    """
    return _load_session(str(model_path)) is not None


def score_pattern_quality(
    close: np.ndarray,
    confirm_index: int,
    *,
    model_path: str | os.PathLike = DEFAULT_MODEL_PATH,
    window_bars: int = DEFAULT_WINDOW_BARS,
    image_size: int = DEFAULT_IMAGE_SIZE,
    temperature: float = 1.0,
) -> float | None:
    """CNN-based replacement for the rule-based scanner's ``quality_score``.

    Returns a score in ``[0, 1]`` -- ``(P(target hit) - P(stop hit) + 1) / 2``
    from the CNN's 3-class triple-barrier prediction over the GAF-encoded
    price window leading up to ``confirm_index`` -- so ``0.5`` reads as
    "neutral" (model predicts stop and target equally likely) the same way
    the rule-based ``quality_score`` reads as a plain 0-100 confidence, not
    just ``P(target hit)`` alone (which would conflate "this specific setup
    looks good" with "the training set's classes were imbalanced towards
    +1").

    Returns ``None`` -- meaning "fall back to the rule-based quality score"
    -- whenever CNN scoring isn't usable right now: ``onnxruntime``/model
    missing (see :func:`_load_session`), not enough price history before
    ``confirm_index`` for a full window (see
    :func:`~firm.patterns.ml.cnn_validator.extract_window`), a flat window
    (no shape information), ``window_bars != image_size`` (the pure-numpy
    :func:`_encode_gasf` replica only reproduces pyts's PAA step when it's a
    no-op -- rather than silently mis-encoding, this bails out), or any
    unexpected inference-time error. Never raises.

    ``temperature`` (2026-09, default ``1.0`` = no-op, byte-for-byte
    unchanged behavior for every existing caller): divides the model's
    pre-softmax logits before re-applying softmax, via
    :func:`firm.patterns.ml.calibration.apply_temperature` -- see that
    module's docstring for why temperature scaling (rather than Platt/
    sigmoid scaling, used for the XGBoost classifier instead) is the right
    technique for this specific model. Pass the ``"temperature"`` value
    from a calibration dict previously produced by
    :func:`firm.patterns.ml.calibration.fit_temperature` +
    :func:`~firm.patterns.ml.calibration.save_calibration`.
    """
    if window_bars != image_size:
        _warn_once(
            f"{model_path}:window_mismatch",
            "pattern_recognition CNN scoring unavailable: window_bars=%d != "
            "image_size=%d -- the GAF replica used here only supports the "
            "trained default (PAA-as-identity case); falling back to "
            "rule-based quality scoring.",
            window_bars, image_size,
        )
        return None

    loaded = _load_session(str(model_path))
    if loaded is None:
        return None
    session, present_labels = loaded

    window = extract_window(close, confirm_index, window_bars=window_bars)
    if window is None:
        log.debug(
            "score_pattern_quality: not enough history / flat window at "
            "confirm_index=%d (window_bars=%d) -- falling back to rule-based "
            "quality scoring.", confirm_index, window_bars,
        )
        return None

    try:
        gasf = _encode_gasf(window).astype(np.float32)
        input_name = session.get_inputs()[0].name
        (logits,) = session.run(None, {input_name: gasf[None, None, :, :]})
        logits = np.asarray(logits, dtype=np.float64)[0]
        if temperature != 1.0:
            from firm.patterns.ml.calibration import apply_temperature

            probs = apply_temperature(logits[None, :], temperature)[0]
        else:
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
        out = np.zeros(len(LABELS))
        for col, label in enumerate(present_labels):
            out[_LABEL_TO_INDEX[label]] = probs[col]
        proba_target = float(out[_LABEL_TO_INDEX[1]])
        proba_stop = float(out[_LABEL_TO_INDEX[-1]])
        score = (proba_target - proba_stop + 1.0) / 2.0
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        _warn_once(
            str(model_path) + ":inference_error",
            "pattern_recognition CNN scoring failed during inference at "
            "confirm_index=%d -- falling back to rule-based quality scoring.",
            confirm_index,
        )
        log.debug("CNN inference failure detail", exc_info=True)
        return None


def reset_cache() -> None:
    """Test-only: clear the lazy-singleton session cache and warn-once state
    so a test can exercise :func:`_load_session`'s cold-start path again
    (e.g. after monkeypatching ``onnxruntime`` or the model path between
    cases). Not called anywhere in the live path.
    """
    _load_session.cache_clear()
    _warned_paths.clear()
