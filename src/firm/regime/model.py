"""Gaussian HMM wrapper for regime detection.

Thin, deterministic wrapper around :class:`hmmlearn.hmm.GaussianHMM` that:

* standardises features (critical for Gaussian-HMM convergence),
* fits via Baum-Welch (EM),
* Laplace-smooths the learned transition matrix (§6.2),
* decodes the current regime via Viterbi and its forward posterior, and
* labels states Bull / Chop / Bear by mean log return (§4.4).

``hmmlearn`` is imported lazily so the wider ``firm`` package still imports in
environments where it is not installed; in that case :meth:`GaussianRegimeModel.fit`
raises :class:`RegimeUnavailable`, which callers catch to degrade gracefully
(mirroring the optional-dependency pattern used for the LLM / broker extras).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from firm.regime.features import apply_laplace_smoothing

#: Canonical regime labels.
BULL = "Bull"
CHOP = "Chop"
BEAR = "Bear"


class RegimeUnavailable(RuntimeError):
    """Raised when an HMM cannot be fitted (e.g. ``hmmlearn`` not installed)."""


def hmm_available() -> bool:
    """Return ``True`` if ``hmmlearn`` can be imported in this environment."""
    try:
        import hmmlearn.hmm  # noqa: F401

        return True
    except Exception:
        return False


@dataclass(frozen=True)
class RegimeState:
    """Decoded regime at a point in time.

    Attributes:
        label: One of ``"Bull"``, ``"Chop"`` or ``"Bear"``.
        confidence: Forward posterior probability of the active state (0–1).
        state_idx: The raw (unlabelled) HMM state index.
        posterior: Posterior probability vector over all states.
    """

    label: str
    confidence: float
    state_idx: int
    posterior: list[float] = field(default_factory=list)


class GaussianRegimeModel:
    """Fit / decode a Gaussian HMM over stationarised market features."""

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        random_state: int = 42,
        laplace_epsilon: float = 1e-6,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.laplace_epsilon = laplace_epsilon
        self._model = None
        self._scaler = None
        self._label_map: dict[int, str] = {}

    @property
    def fitted(self) -> bool:
        return self._model is not None

    def fit(self, X: np.ndarray) -> "GaussianRegimeModel":
        """Standardise *X*, fit the HMM, smooth the transition matrix, label states.

        Args:
            X: Feature matrix of shape ``(T, n_features)`` where column 0 is the
               log return used for labelling.

        Raises:
            RegimeUnavailable: if ``hmmlearn`` is not installed.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except Exception as exc:  # pragma: no cover - exercised via monkeypatch
            raise RegimeUnavailable(
                "hmmlearn is not installed; regime detection is unavailable. "
                "Install it with: pip install hmmlearn"
            ) from exc

        from sklearn.preprocessing import StandardScaler

        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] < self.n_states or X.shape[1] < 1:
            raise RegimeUnavailable(
                f"Insufficient data to fit a {self.n_states}-state HMM: got shape {X.shape}"
            )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        model.fit(X_scaled)

        # §6.2 — guarantee no zero-probability transitions before decoding.
        model.transmat_ = apply_laplace_smoothing(model.transmat_, self.laplace_epsilon)

        self._model = model
        self._scaler = scaler
        self._label_map = self._build_label_map(model.means_)
        return self

    def _build_label_map(self, means: np.ndarray) -> dict[int, str]:
        """Map state indices to Bull / Chop / Bear by mean return (§4.4).

        Standardisation is a monotonic per-feature transform, so ordering
        states by their *scaled* mean return (column 0) matches ordering by
        raw mean return.  Highest → Bull, lowest → Bear, the rest → Chop
        (generalises the 3-state case to any ``n_states``).
        """
        mean_returns = np.asarray(means)[:, 0]
        order = np.argsort(mean_returns)[::-1]  # descending
        label_map: dict[int, str] = {}
        for rank, state_idx in enumerate(order):
            if rank == 0:
                label_map[int(state_idx)] = BULL
            elif rank == len(order) - 1:
                label_map[int(state_idx)] = BEAR
            else:
                label_map[int(state_idx)] = CHOP
        return label_map

    def label_map(self) -> dict[int, str]:
        if not self.fitted:
            raise RegimeUnavailable("Model has not been fitted")
        return dict(self._label_map)

    def _scaled(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RegimeUnavailable("Model has not been fitted")
        return self._scaler.transform(np.asarray(X, dtype=float))

    def current_state(self, X: np.ndarray) -> int:
        """Viterbi-decode *X* and return the most-likely state at the last step."""
        states = self._model.predict(self._scaled(X))
        return int(states[-1])

    def posterior(self, X: np.ndarray) -> np.ndarray:
        """Forward-algorithm posterior P(state | data) at the last step."""
        probs = self._model.predict_proba(self._scaled(X))
        return probs[-1]

    def classify(self, X: np.ndarray) -> RegimeState:
        """Decode the regime at the last observation of *X*.

        Uses the posterior argmax as the active state so the reported
        ``confidence`` (posterior probability) is internally consistent with
        the chosen label.
        """
        post = self.posterior(X)
        state_idx = int(np.argmax(post))
        return RegimeState(
            label=self._label_map.get(state_idx, CHOP),
            confidence=float(post[state_idx]),
            state_idx=state_idx,
            posterior=[float(p) for p in post],
        )
