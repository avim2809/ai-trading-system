"""Ensemble regime classifier: majority vote over several Gaussian-HMM fits.

A single :class:`~firm.regime.model.GaussianRegimeModel`'s Bull/Bear label can
flip between refits when the separation between the extreme states is thin
(see ``GaussianRegimeModel._build_separation``) — the EM optimizer landing in
a different local optimum on a fresh random initialization, not a real change
in market dynamics. 2025-2026 regime-detection research reports materially
better Sharpe and fewer false regime-switch signals from ensemble-HMM voting
than a standalone HMM. :class:`EnsembleRegimeModel` is the cheapest version of
that idea: fit several independent members with different ``random_state``
seeds and combine their classifications by majority vote. It doesn't fix any
single member's instability, but a majority across e.g. 5 independent fits is
far less likely to flip than any one of them.

Exposes the same ``fit(X)``/``classify(X)`` interface as
:class:`~firm.regime.model.GaussianRegimeModel`, so it's a drop-in
replacement wherever a regime model is constructed — see
:class:`~firm.regime.detector.MarketRegimeDetector`'s ``ensemble=`` flag.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from firm.regime.model import (
    BEAR,
    BULL,
    CHOP,
    GaussianRegimeModel,
    RegimeState,
    RegimeUnavailable,
)

log = logging.getLogger(__name__)

_LABEL_ORDER = (BULL, CHOP, BEAR)

#: Default ensemble seeds. Five members is enough for a majority to mean
#: something (ties are impossible) while staying cheap — each member is a
#: full independent Baum-Welch fit.
DEFAULT_SEEDS: tuple[int, ...] = (42, 7, 123, 2024, 99)


class EnsembleRegimeModel:
    """Majority-vote ensemble of :class:`GaussianRegimeModel` members.

    Args:
        n_states: Passed to every member (see ``GaussianRegimeModel``).
        covariance_type: Passed to every member.
        n_iter: Passed to every member.
        seeds: One member is fit per seed. Odd length recommended so a
            3-label majority vote can't tie; ``DEFAULT_SEEDS`` (5) satisfies
            this for the standard Bull/Chop/Bear labelling.
        laplace_epsilon: Passed to every member.
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 100,
        seeds: tuple[int, ...] = DEFAULT_SEEDS,
        laplace_epsilon: float = 1e-6,
    ) -> None:
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.seeds = seeds
        self.laplace_epsilon = laplace_epsilon
        self._members: list[GaussianRegimeModel] = []

    @property
    def fitted(self) -> bool:
        return bool(self._members)

    def fit(self, X: np.ndarray) -> "EnsembleRegimeModel":
        """Fit one :class:`GaussianRegimeModel` per seed.

        A member that fails to fit (numerical/convergence issue, distinct
        from ``hmmlearn`` being entirely uninstalled) is dropped rather than
        aborting the whole ensemble — a majority vote over 3/5 surviving
        members is still meaningful. Re-raises :class:`RegimeUnavailable`
        immediately if the very first member hits it, since that specifically
        means ``hmmlearn`` isn't installed at all and every other seed would
        fail identically — no point retrying.

        Raises:
            RegimeUnavailable: if ``hmmlearn`` is missing, or if every
                member failed to fit for some other reason.
        """
        members: list[GaussianRegimeModel] = []
        last_error: Exception | None = None
        for i, seed in enumerate(self.seeds):
            model = GaussianRegimeModel(
                n_states=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=seed,
                laplace_epsilon=self.laplace_epsilon,
            )
            try:
                model.fit(X)
                members.append(model)
            except RegimeUnavailable:
                if i == 0:
                    raise
                log.debug("Ensemble member seed=%s: hmmlearn unavailable", seed)
                last_error = RegimeUnavailable("hmmlearn unavailable")
            except Exception as exc:
                last_error = exc
                log.debug("Ensemble member seed=%s failed to fit: %s", seed, exc)

        if not members:
            raise RegimeUnavailable(
                f"All {len(self.seeds)} ensemble members failed to fit"
                + (f" (e.g. {last_error!r})" if last_error else "")
            )
        if len(members) < len(self.seeds):
            log.warning(
                "Regime ensemble: only %d/%d members fitted successfully",
                len(members), len(self.seeds),
            )
        self._members = members
        return self

    def classify(self, X: np.ndarray) -> RegimeState:
        """Classify *X* with every member and return the majority-vote label.

        ``confidence`` blends two independent signals so either a noisy
        single-model posterior *or* ensemble disagreement reduces trust in
        the label: ``vote_share(majority_label) * mean(member_confidence)``
        over the members that agreed with the majority.

        ``posterior`` here is **not** a single model's forward posterior
        (raw per-model state indices aren't comparable across independently
        fit members) — it's the vote-share vector ``[P(Bull), P(Chop),
        P(Bear)]`` = fraction of members that voted each label, which sums
        to 1 and is the natural ensemble analogue. No current consumer of
        ``RegimeState`` reads ``posterior`` (only ``label``/``confidence``),
        so this is informational.

        Raises:
            RegimeUnavailable: if the ensemble has not been fitted.
        """
        if not self.fitted:
            raise RegimeUnavailable("Ensemble has not been fitted")

        votes = [m.classify(X) for m in self._members]
        n = len(votes)
        vote_counts = Counter(v.label for v in votes)
        vote_share = {lbl: vote_counts.get(lbl, 0) / n for lbl in _LABEL_ORDER}
        majority_label = max(_LABEL_ORDER, key=lambda lbl: vote_share[lbl])

        agreeing = [v for v in votes if v.label == majority_label]
        avg_confidence = float(np.mean([v.confidence for v in agreeing]))
        finite_seps = [v.separation for v in agreeing if np.isfinite(v.separation)]
        avg_separation = float(np.mean(finite_seps)) if finite_seps else float("inf")

        return RegimeState(
            label=majority_label,
            confidence=vote_share[majority_label] * avg_confidence,
            state_idx=agreeing[0].state_idx,
            posterior=[vote_share[lbl] for lbl in _LABEL_ORDER],
            separation=avg_separation,
        )
