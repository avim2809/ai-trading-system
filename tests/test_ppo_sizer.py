"""Tests for the PPO position-sizing agent (docs/pattern_recognition_plan.md
§4c): firm.patterns.ml.ppo_sizer.

**Run these via the isolated ML environment, not the main venv**::

    .venv-ml/bin/pytest tests/test_ppo_sizer.py -q

stable-baselines3/gymnasium have no Python 3.14 wheels yet (see
docs/pattern_recognition_plan.md §4a for how .venv-ml was built) -- this
file skips everything under the main venv rather than failing collection,
mirroring tests/test_cnn_validator.py's gating.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from firm.patterns.ml import ppo_sizer

try:
    import gymnasium  # noqa: F401
    from stable_baselines3 import PPO  # noqa: F401

    _SB3_STACK_AVAILABLE = True
except ImportError:
    _SB3_STACK_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SB3_STACK_AVAILABLE,
    reason="stable-baselines3/gymnasium not installed -- run via .venv-ml/bin/pytest",
)


def _toy_meta_and_labels(n: int = 60, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(seed)
    meta = pd.DataFrame({
        "direction": rng.choice(["long", "short"], n),
        "quality_score": rng.uniform(0, 100, n),
        "risk_reward": rng.uniform(0.5, 5.0, n),
        "volume_ratio": rng.uniform(0.5, 3.0, n),
        "duration_bars": rng.uniform(5, 100, n),
    })
    labels = rng.choice([-1, 0, 1], n)
    return meta, labels


class TestBuildObservations:
    def test_output_shapes_and_ranges(self):
        meta, labels = _toy_meta_and_labels()
        obs, outcome, scale = ppo_sizer.build_observations(meta, labels)
        assert obs.shape == (len(meta), len(ppo_sizer.FEATURE_KEYS))
        assert outcome.shape == (len(meta),)
        assert scale.shape == (len(meta),)
        # direction_sign (column 2) is -1/+1; the rest are normalized to [0, 1].
        other_cols = np.delete(obs, 2, axis=1)
        assert np.all(other_cols >= -1e-6) and np.all(other_cols <= 1.0 + 1e-6)
        assert set(np.unique(obs[:, 2])) <= {-1.0, 1.0}
        np.testing.assert_array_equal(outcome, labels.astype(float))

    def test_direction_sign_matches_long_short(self):
        meta = pd.DataFrame({
            "direction": ["long", "short"],
            "quality_score": [80.0, 80.0],
            "risk_reward": [2.0, 2.0],
            "volume_ratio": [1.0, 1.0],
            "duration_bars": [10, 10],
        })
        obs, _outcome, _scale = ppo_sizer.build_observations(meta, np.array([1, -1]))
        assert obs[0, 2] == 1.0
        assert obs[1, 2] == -1.0

    def test_reward_scale_uses_risk_reward_on_wins_and_one_otherwise(self):
        meta = pd.DataFrame({
            "direction": ["long", "long", "long"],
            "quality_score": [80.0, 80.0, 80.0],
            "risk_reward": [3.0, 3.0, 3.0],
            "volume_ratio": [1.0, 1.0, 1.0],
            "duration_bars": [10, 10, 10],
        })
        labels = np.array([1, -1, 0])
        _obs, _outcome, scale = ppo_sizer.build_observations(meta, labels)
        assert scale[0] == pytest.approx(3.0)
        assert scale[1] == pytest.approx(1.0)
        assert scale[2] == pytest.approx(1.0)


class TestPatternSizingEnv:
    def test_reset_returns_a_valid_observation(self):
        meta, labels = _toy_meta_and_labels()
        obs, outcome, scale = ppo_sizer.build_observations(meta, labels)
        env = ppo_sizer.PatternSizingEnv.make_env(obs, outcome, scale, seed=0)
        observation, info = env.reset(seed=0)
        assert observation.shape == (len(ppo_sizer.FEATURE_KEYS),)
        assert info == {}

    def test_step_is_single_step_episode_with_bounded_reward(self):
        obs = np.zeros((1, len(ppo_sizer.FEATURE_KEYS)), dtype=np.float32)
        outcome = np.array([1.0])
        scale = np.array([2.0])
        env = ppo_sizer.PatternSizingEnv.make_env(obs, outcome, scale, transaction_cost=0.0, risk_aversion=0.0, seed=0)
        env.reset(seed=0)
        _observation, reward, terminated, truncated, info = env.step(np.array([1.0]))
        assert terminated is True
        assert truncated is False
        assert reward == pytest.approx(2.0)  # size=1 * outcome=1 * scale=2, no cost/penalty
        assert info["outcome"] == 1.0
        assert info["scale"] == 2.0

    def test_risk_aversion_penalizes_large_sizes(self):
        obs = np.zeros((1, len(ppo_sizer.FEATURE_KEYS)), dtype=np.float32)
        outcome = np.array([1.0])
        scale = np.array([1.0])
        env = ppo_sizer.PatternSizingEnv.make_env(obs, outcome, scale, transaction_cost=0.0, risk_aversion=0.5, seed=0)
        env.reset(seed=0)
        _observation, reward, _terminated, _truncated, _info = env.step(np.array([1.0]))
        # pnl=1.0, penalty=0.5*1^2=0.5 -> reward=0.5
        assert reward == pytest.approx(0.5)

    def test_action_clipped_to_valid_range(self):
        obs = np.zeros((1, len(ppo_sizer.FEATURE_KEYS)), dtype=np.float32)
        outcome = np.array([1.0])
        scale = np.array([1.0])
        env = ppo_sizer.PatternSizingEnv.make_env(obs, outcome, scale, transaction_cost=0.0, risk_aversion=0.0, seed=0)
        env.reset(seed=0)
        _observation, reward, _terminated, _truncated, _info = env.step(np.array([5.0]))
        # size clipped to 1.0 before computing pnl
        assert reward == pytest.approx(1.0)


class TestTrainAndPredict:
    def test_train_returns_model_with_bounded_predictions(self):
        meta, labels = _toy_meta_and_labels(seed=1)
        obs, outcome, scale = ppo_sizer.build_observations(meta, labels)
        model = ppo_sizer.train_ppo(obs, outcome, scale, total_timesteps=200, seed=0)
        for i in range(len(obs)):
            size = ppo_sizer.predict_position_size(model, obs[i])
            assert -1.0 <= size <= 1.0

    def test_predict_position_size_is_deterministic(self):
        meta, labels = _toy_meta_and_labels(seed=2)
        obs, outcome, scale = ppo_sizer.build_observations(meta, labels)
        model = ppo_sizer.train_ppo(obs, outcome, scale, total_timesteps=200, seed=0)
        size_a = ppo_sizer.predict_position_size(model, obs[0])
        size_b = ppo_sizer.predict_position_size(model, obs[0])
        assert size_a == size_b

    def test_higher_risk_aversion_learns_smaller_sizes_on_favorable_data(self):
        # Contexts with a strongly favorable, uncorrelated outcome: with
        # risk_aversion large enough, the constrained per-context optimum
        # should pull off the max-size boundary (see the module docstring's
        # rationale + the manual verification in the session that added
        # this parameter).
        rng = np.random.RandomState(3)
        n = 80
        obs = rng.uniform(0, 1, (n, len(ppo_sizer.FEATURE_KEYS))).astype(np.float32)
        outcome = np.ones(n)
        scale = np.ones(n)
        model = ppo_sizer.train_ppo(obs, outcome, scale, total_timesteps=4000, seed=0, risk_aversion=5.0)
        sizes = np.array([ppo_sizer.predict_position_size(model, obs[i]) for i in range(n)])
        assert sizes.mean() < 0.9


class TestSaveAndLoad:
    def test_save_and_load_roundtrip_predicts_identically(self, tmp_path):
        meta, labels = _toy_meta_and_labels(seed=4)
        obs, outcome, scale = ppo_sizer.build_observations(meta, labels)
        model = ppo_sizer.train_ppo(obs, outcome, scale, total_timesteps=200, seed=0)
        path = tmp_path / "model.zip"
        ppo_sizer.save(model, path)
        assert path.exists()
        loaded = ppo_sizer.load(path)
        for i in range(len(obs)):
            assert ppo_sizer.predict_position_size(loaded, obs[i]) == pytest.approx(
                ppo_sizer.predict_position_size(model, obs[i])
            )
