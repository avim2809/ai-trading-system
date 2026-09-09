"""PPO position-sizing agent (Phase 4c, docs/pattern_recognition_plan.md
§4c) -- **isolated ML environment only** (``.venv-ml``; needs
``stable-baselines3``/``gymnasium``, neither of which has a Python 3.14
wheel yet, see §4a).

Learns *how much* to bet on a confirmed pattern (a continuous position-size
action in ``[-1, 1]``) rather than *whether* one exists at all (that's
xgb_classifier/cnn_validator's job) -- reusing the exact same per-match
records (quality_score, risk_reward, direction, ...) and triple-barrier
outcome (:mod:`firm.patterns.ml.labeling`) the other two models train on, so
all three are trained from one dataset, not three independent ones.

Deliberately a *single-step* ("contextual bandit") environment, not a
multi-step portfolio simulation: each episode presents one historical
match's context, the agent picks one position size, and the episode ends
immediately with the reward that size would have realized given what
*actually* happened to that pattern (its real label). A proper sequential
portfolio simulation (capital compounding across many trades over time,
position overlap, etc.) is meaningfully more infrastructure than this pass
scopes -- see the module-level scope note in
``scripts/train_pattern_ppo.py`` for why this stays a standalone research
tool either way.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Observation columns, in this fixed order -- see build_observations.
FEATURE_KEYS: tuple[str, ...] = (
    "quality_score_norm", "risk_reward_norm", "direction_sign", "volume_ratio_norm", "duration_norm",
)
_DEFAULT_TRANSACTION_COST = 0.001  # 10bps, matching the original research sketch's cost term


def _require_gymnasium():
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "gymnasium is required for firm.patterns.ml.ppo_sizer -- run this "
            "under the isolated .venv-ml environment (docs/"
            "pattern_recognition_plan.md §4a), not the main venv."
        ) from exc
    return gym


def _require_sb3():
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "stable-baselines3 is required for firm.patterns.ml.ppo_sizer -- "
            "run this under the isolated .venv-ml environment."
        ) from exc
    return PPO


def build_observations(meta: pd.DataFrame, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From the per-match ``meta`` records (symbol/pattern/direction/
    confirm_index/quality_score/risk_reward/volume_ratio/duration_bars --
    the same shape ``scripts/train_pattern_ml.py``'s ``build_dataset``
    already collects, extended with the two extra fields this needs) and
    matching triple-barrier ``labels``, build the three parallel arrays
    :class:`PatternSizingEnv` needs:

    - ``observations`` ``(n, len(FEATURE_KEYS))``: normalized context.
    - ``directional_outcome`` ``(n,)``: ``+1`` if betting in the pattern's
      own signaled direction would have realized the favorable barrier,
      ``-1`` if the adverse one, ``0`` for a timeout.
    - ``reward_scale`` ``(n,)``: the risk:reward multiple on a win (so
      hitting target on a high-R:R setup pays more than a low one), ``1.0``
      on a loss (the full risked amount) or timeout.
    """
    direction_sign = np.where(meta["direction"].to_numpy() == "long", 1.0, -1.0)
    quality_norm = meta["quality_score"].to_numpy(dtype=float) / 100.0
    rr = meta["risk_reward"].to_numpy(dtype=float)
    rr_norm = np.clip(rr, 0.0, 10.0) / 10.0
    volume_ratio = meta.get("volume_ratio", pd.Series(np.ones(len(meta)))).to_numpy(dtype=float)
    volume_norm = np.clip(np.nan_to_num(volume_ratio, nan=1.0), 0.0, 5.0) / 5.0
    duration = meta.get("duration_bars", pd.Series(np.zeros(len(meta)))).to_numpy(dtype=float)
    duration_norm = np.clip(duration, 0.0, 200.0) / 200.0

    observations = np.column_stack([quality_norm, rr_norm, direction_sign, volume_norm, duration_norm]).astype(np.float32)

    directional_outcome = labels.astype(float)  # labels are already +1/0/-1 relative to the pattern's own direction
    reward_scale = np.where(labels == 1, np.clip(rr, 0.1, 10.0), 1.0)
    return observations, directional_outcome, reward_scale.astype(np.float32)


class PatternSizingEnv:
    """Single-step ("contextual bandit") position-sizing environment --
    built lazily as a real ``gymnasium.Env`` subclass inside
    :func:`make_env`, for the same reason ``cnn_validator.PatternCNN``
    builds its ``nn.Module`` lazily: ``gymnasium`` only exists in the
    isolated environment, so the real base class can't be subclassed at
    this module's import time.
    """

    def __new__(cls, *args, **kwargs):
        raise TypeError("PatternSizingEnv.make_env() constructs the environment -- this class is not instantiated directly")

    @staticmethod
    def make_env(
        observations: np.ndarray,
        directional_outcome: np.ndarray,
        reward_scale: np.ndarray,
        *,
        transaction_cost: float = _DEFAULT_TRANSACTION_COST,
        risk_aversion: float = 1.0,
        seed: int | None = None,
    ):
        gym = _require_gymnasium()
        from gymnasium import spaces

        class _Env(gym.Env):
            metadata: dict[str, Any] = {}

            def __init__(self) -> None:
                super().__init__()
                self._obs = observations
                self._outcome = directional_outcome
                self._scale = reward_scale
                self._cost = transaction_cost
                self._risk_aversion = risk_aversion
                self._rng = np.random.default_rng(seed)
                self._idx = 0
                n_features = observations.shape[1]
                self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
                self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(n_features,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                if seed is not None:
                    self._rng = np.random.default_rng(seed)
                self._idx = int(self._rng.integers(0, len(self._obs)))
                return self._obs[self._idx], {}

            def step(self, action):
                size = float(np.clip(action[0], -1.0, 1.0))
                # Linear P&L term minus transaction cost minus a quadratic
                # risk-aversion penalty on |size| (standard mean-variance /
                # quadratic-utility position sizing) -- without the last
                # term, a purely linear reward makes "always bet max size"
                # optimal whenever the dataset's average outcome is
                # positive (verified empirically: the policy converged to
                # exactly the always-max-size baseline reward with
                # risk_aversion=0). On the real cached-data run, mean
                # outcome*scale was ~1.14, which still clips the per-context
                # optimum to the max-size boundary for risk_aversion values
                # much below ~1.0 (verified: 99.4% of predictions saturated
                # at risk_aversion=0.5) -- collapsing right back to the same
                # degenerate policy. risk_aversion=1.0 (the default) is
                # calibrated against that real run to actually pull the
                # optimum off the boundary and reward sizing *down* on
                # lower-conviction contexts.
                pnl = size * self._outcome[self._idx] * self._scale[self._idx]
                reward = pnl - self._cost * abs(size) - self._risk_aversion * size * size
                obs = self._obs[self._idx]
                info = {"outcome": float(self._outcome[self._idx]), "scale": float(self._scale[self._idx])}
                return obs, float(reward), True, False, info

        return _Env()


def train_ppo(
    observations: np.ndarray,
    directional_outcome: np.ndarray,
    reward_scale: np.ndarray,
    *,
    total_timesteps: int = 20_000,
    seed: int = 0,
    transaction_cost: float = _DEFAULT_TRANSACTION_COST,
    risk_aversion: float = 1.0,
    policy_kwargs: dict[str, Any] | None = None,
):
    """Train a PPO agent on :class:`PatternSizingEnv` built from the given
    dataset. Returns the fitted ``stable_baselines3.PPO`` model directly
    (unlike xgb_classifier/cnn_validator, no extra label-index bookkeeping
    is needed here -- the action space is continuous, not classification).

    ``risk_aversion`` (see :meth:`PatternSizingEnv.make_env`) matters a lot
    here: ``0`` makes "always bet max size" the reward-optimal policy
    whenever the dataset's average outcome is positive, which is not a
    useful sizing lesson. The default ``1.0`` is calibrated against a real
    run over this repo's cached data (mean outcome*scale ~1.14) to actually
    make sizing *down* on lower-conviction contexts pay off -- ``0.5``
    turned out still too weak there (99.4% of predictions saturated at
    max size regardless of context).
    """
    PPO = _require_sb3()
    env = PatternSizingEnv.make_env(
        observations, directional_outcome, reward_scale,
        transaction_cost=transaction_cost, risk_aversion=risk_aversion, seed=seed,
    )
    model = PPO(
        "MlpPolicy", env, seed=seed, verbose=0,
        n_steps=256, batch_size=64, policy_kwargs=policy_kwargs or {},
    )
    model.learn(total_timesteps=total_timesteps)
    log.info(
        "ppo_sizer.train_ppo: trained for %d timesteps on %d contexts",
        total_timesteps, len(observations),
    )
    return model


def predict_position_size(model: Any, observation: np.ndarray) -> float:
    """The agent's chosen position size (``[-1, 1]``) for one observation
    (deterministic -- no exploration noise, matching how a live sizing
    decision would use the trained policy).
    """
    action, _state = model.predict(np.asarray(observation, dtype=np.float32), deterministic=True)
    return float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))


def save(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    log.info("ppo_sizer.save: wrote model to %s", path)


def load(path: str | Path) -> Any:
    """Load a model previously written by :func:`save`. Note: unlike
    xgb_classifier/cnn_validator, there is no ONNX export here -- SB3
    policies don't export as cleanly (the action-sampling wrapper around
    the underlying network isn't a plain forward pass), so consuming a
    saved PPO policy for inference still requires the isolated environment
    (stable-baselines3 itself), not just an ONNX runtime. Documented as a
    known follow-on constraint, not solved in this pass.
    """
    PPO = _require_sb3()
    model = PPO.load(str(path))
    log.info("ppo_sizer.load: loaded model from %s", path)
    return model
