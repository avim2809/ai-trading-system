"""ML confirmation layer on top of the rule-based pattern detector (Phase 4
of docs/pattern_recognition_plan.md).

Deliberately scoped to just the XGBoost confirmation classifier -- feature
engineering (:mod:`.feature_engineering`), triple-barrier labeling
(:mod:`.labeling`), and a thin train/predict/persist wrapper
(:mod:`.xgb_classifier`). The CNN/GAF image validator and PPO RL position
sizer from that phase's original aspirational scope are explicitly out of
scope for this pass (heavy new dependencies -- torch, stable-baselines3,
gymnasium, pyts, onnxruntime -- not installed on this shared, resource-
constrained production box); see the task report for this initiative for
why, and treat that as future work, not an oversight.

``xgboost`` itself is an optional dependency (the ``patterns_ml`` extra in
pyproject.toml) -- only :mod:`.xgb_classifier`'s ``train``/``predict_proba``
actually import it, and lazily at call time, so ``feature_engineering`` and
``labeling`` stay usable without it installed.

See ``scripts/train_pattern_ml.py`` for the end-to-end CLI that ties these
together (scan -> features -> triple-barrier labels -> train -> report).
"""

from __future__ import annotations

from firm.patterns.ml.feature_engineering import build_feature_frame, build_features
from firm.patterns.ml.labeling import label_matches, label_triple_barrier

__all__ = [
    "build_features",
    "build_feature_frame",
    "label_triple_barrier",
    "label_matches",
]
