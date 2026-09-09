"""CNN/GAF chart-pattern image validator (Phase 4b, docs/
pattern_recognition_plan.md §4b) -- **isolated ML environment only**
(``.venv-ml``; needs ``torch``/``pyts``, neither of which has a Python 3.14
wheel yet, see §4a).

Encodes the OHLCV window leading up to a confirmed pattern's breakout as a
Gramian Angular Field (GAF) image -- a standard time-series-to-image
technique (Wang & Oates, 2015) that turns temporal correlation into spatial
correlation an ordinary 2D CNN can learn from -- and trains a small CNN to
predict the same triple-barrier outcome (:mod:`firm.patterns.ml.labeling`)
the XGBoost classifier does, so the two are directly comparable.

Deliberately independent of :mod:`firm.patterns.ml.xgb_classifier`: this
module (like :mod:`firm.patterns.ml.feature_engineering`/``labeling``) has no
hard dependency on ``xgboost``, and the reverse is also true -- neither
requires the other to be installed. This is a standalone research tool, not
wired into any live signal-generation path (see the module-level scope note
in ``scripts/train_cnn_validator.py``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

#: Same fixed label space/column order as xgb_classifier.LABELS, so the two
#: models' outputs are directly comparable.
LABELS: tuple[int, ...] = (-1, 0, 1)
_LABEL_TO_INDEX: dict[int, int] = {label: i for i, label in enumerate(LABELS)}

DEFAULT_IMAGE_SIZE = 32
DEFAULT_WINDOW_BARS = 32


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - exercised only without the isolated env
        raise ImportError(
            "torch is required for firm.patterns.ml.cnn_validator -- run this "
            "under the isolated .venv-ml environment (docs/"
            "pattern_recognition_plan.md §4a), not the main venv."
        ) from exc
    return torch, nn


def _require_pyts():
    try:
        from pyts.image import GramianAngularField
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyts is required for firm.patterns.ml.cnn_validator -- run this "
            "under the isolated .venv-ml environment."
        ) from exc
    return GramianAngularField


def encode_gaf(series: np.ndarray, *, image_size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    """One 1D price series -> one ``(image_size, image_size)`` GAF image.

    ``series`` should already be a fixed-length window (see
    :data:`DEFAULT_WINDOW_BARS`) -- GAF itself doesn't require a fixed
    length, but a CNN's input tensor does, so callers pad/trim before this.
    """
    GramianAngularField = _require_pyts()
    gaf = GramianAngularField(image_size=image_size, method="summation")
    return gaf.fit_transform(series.reshape(1, -1))[0]


def extract_window(close: np.ndarray, confirm_index: int, *, window_bars: int = DEFAULT_WINDOW_BARS) -> np.ndarray | None:
    """The ``window_bars`` closes leading up to and including
    ``confirm_index`` (the breakout bar) -- ``None`` if there isn't enough
    history yet. Normalized to a 0-1 range (min-max over the window) before
    GAF encoding, matching GAF's own expectation that its input already sits
    in a bounded range.
    """
    start = confirm_index - window_bars + 1
    if start < 0:
        return None
    window = close[start : confirm_index + 1].astype(float)
    lo, hi = window.min(), window.max()
    if hi - lo < 1e-9:
        return None  # a flat window carries no shape information
    return (window - lo) / (hi - lo)


class PatternCNN:
    """Thin factory for a small 2D CNN over one-channel GAF images ->
    3-class triple-barrier prediction. A plain function (not a class you'd
    subclass) since the isolated env is the only place ``torch`` exists at
    all -- constructing the actual ``nn.Module`` subclass has to happen
    lazily, inside :func:`build_model`, not at this module's import time.
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError("PatternCNN.build_model() constructs the model -- this class is not instantiated directly")

    @staticmethod
    def build_model(image_size: int = DEFAULT_IMAGE_SIZE):
        torch, nn = _require_torch()

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 16, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(4),
                )
                self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(32 * 4 * 4, 32),
                    nn.ReLU(),
                    nn.Linear(32, len(LABELS)),
                )

            def forward(self, x):
                return self.fc(self.conv(x))

        return _Net()


@dataclass
class _FittedCNN:
    """Bundle a fitted CNN with the label encoding it was trained on --
    same rationale as ``xgb_classifier._FittedPatternModel``: a training
    slice can miss a class, and this lets :func:`predict_proba` always
    decode back to the fixed 3-column ``(-1, 0, +1)`` layout regardless.
    """

    net: Any
    present_labels: tuple[int, ...]
    image_size: int


def train(
    X_images: np.ndarray,
    y,
    *,
    image_size: int = DEFAULT_IMAGE_SIZE,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 0,
) -> _FittedCNN:
    """Train the CNN over a batch of GAF images.

    Args:
        X_images: ``(n, image_size, image_size)`` array (see
            :func:`encode_gaf`).
        y: matching int labels drawn from :data:`LABELS`.
    """
    torch, nn = _require_torch()
    y_arr = np.asarray(y)
    if len(y_arr) == 0:
        raise ValueError("train: y is empty -- nothing to fit")
    unknown = set(int(v) for v in np.unique(y_arr)) - set(LABELS)
    if unknown:
        raise ValueError(f"train: y contains labels outside {LABELS}: {sorted(unknown)}")

    torch.manual_seed(seed)
    present_labels = tuple(sorted(int(v) for v in np.unique(y_arr)))
    if len(present_labels) < 2:
        log.warning(
            "train: only one distinct label present (%s) across %d rows -- "
            "the fitted model will be degenerate", present_labels, len(y_arr),
        )
    label_to_compact = {label: i for i, label in enumerate(present_labels)}
    y_compact = np.array([label_to_compact[int(v)] for v in y_arr])

    net = PatternCNN.build_model(image_size=image_size)
    X_t = torch.tensor(X_images, dtype=torch.float32).unsqueeze(1)  # (n, 1, H, W)
    y_t = torch.tensor(y_compact, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    net.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        log.debug("cnn_validator.train: epoch %d/%d loss=%.4f", epoch + 1, epochs, total_loss / len(dataset))
    net.eval()

    log.info(
        "cnn_validator.train: fit on %d images (%dx%d), labels present=%s",
        len(y_compact), image_size, image_size, present_labels,
    )
    return _FittedCNN(net=net, present_labels=present_labels, image_size=image_size)


def predict_proba(model: _FittedCNN, X_images: np.ndarray) -> np.ndarray:
    """Class probabilities, ``(n, 3)``, columns always ``(-1, 0, +1)``
    regardless of which classes ``model`` actually saw at fit time.
    """
    torch, _nn = _require_torch()
    model.net.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_images, dtype=torch.float32).unsqueeze(1)
        raw = torch.softmax(model.net(X_t), dim=1).numpy()
    out = np.zeros((raw.shape[0], len(LABELS)))
    for col, label in enumerate(model.present_labels):
        out[:, _LABEL_TO_INDEX[label]] = raw[:, col]
    return out


def predict_label(model: _FittedCNN, X_images: np.ndarray) -> np.ndarray:
    proba = predict_proba(model, X_images)
    idx = np.argmax(proba, axis=1)
    return np.array([LABELS[i] for i in idx])


def save(model: _FittedCNN, path: str | Path) -> None:
    """Persist via ``torch.save`` (state_dict + the small metadata needed
    to reconstruct the architecture and label mapping).
    """
    torch, _nn = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.net.state_dict(),
            "present_labels": model.present_labels,
            "image_size": model.image_size,
        },
        path,
    )
    log.info("cnn_validator.save: wrote model to %s", path)


def load(path: str | Path) -> _FittedCNN:
    torch, _nn = _require_torch()
    path = Path(path)
    checkpoint = torch.load(path, weights_only=False)
    net = PatternCNN.build_model(image_size=checkpoint["image_size"])
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    log.info("cnn_validator.load: loaded model from %s", path)
    return _FittedCNN(
        net=net, present_labels=tuple(checkpoint["present_labels"]), image_size=checkpoint["image_size"],
    )


def export_onnx(model: _FittedCNN, path: str | Path) -> None:
    """Export to ONNX via ``torch.onnx.export`` -- unlike XGBoost (see
    ``xgb_classifier.export_onnx``'s feature-naming caveat), a plain
    ``nn.Module`` exports natively with no naming gotchas. Writes the same
    ``<path>.labels.json`` sidecar convention as ``xgb_classifier`` so a
    future shared loader could treat both model types uniformly.
    """
    import json

    torch, _nn = _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 1, model.image_size, model.image_size, dtype=torch.float32)
    torch.onnx.export(
        model.net, dummy, str(path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )
    sidecar = path.with_suffix(path.suffix + ".labels.json")
    with open(sidecar, "w") as f:
        json.dump(list(model.present_labels), f)
    log.info("cnn_validator.export_onnx: wrote %s (+ %s)", path, sidecar)
