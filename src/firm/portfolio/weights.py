"""Handcrafted instrument weights with an HRP diagnostic (credibility plan P4-01).

Source: plan P4-01; Carver, *Systematic Trading* (handcrafting); Lopez de Prado (2016),
"Building Diversified Portfolios that Outperform Out of Sample".

Handcrafting is a deterministic rule on the group tree, with no data input, so no performance
information can enter the weights:

    w[class] = 1 / n_classes;  w[sub] = w[class] / n_subs_in_class;  w[inst] = w[sub] / n_inst_in_sub

These are risk-share weights; the sizing step (P3-06) converts them to positions by vol targeting.
Groups come from ``config/universe_etf.yaml`` (P2-01) only and are never tuned on returns.

HRP is DIAGNOSTIC ONLY: it is shown next to the handcrafted weights and never feeds sizing or
backtests; switching production weights to HRP is a new pre-registered trial.
  1. correlation distance ``d = sqrt(0.5 * (1 - rho))``
  2. single-linkage clustering, quasi-diagonal (dendrogram leaf) order
  3. recursive bisection with inverse-variance cluster weights; cluster variance ``w' S w``;
     split ``alpha = 1 - V_left / (V_left + V_right)``

The bisection is POSITIONAL (halves the quasi-diagonal order by count, not by dendrogram cluster),
as in the original snippet and in the legacy ``firm.agents.analysts.hrp_signal_weights``, which it
is cross-checked against in tests (test-only import; this module never imports ``firm.agents``).
HRP never inverts the covariance, so singular or rank-deficient PSD matrices are accepted.

Research-only: not imported by ``firm.portfolio.__init__`` or any live module (P0-06 isolation test).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def handcraft_weights(groups: Mapping[str, Mapping[str, Sequence[str]]]) -> pd.Series:
    """groups = {asset_class: {sub_group: [instrument, ...]}}.

    Equal risk share per asset class, equal per sub-group inside it, equal per instrument inside
    that. Returns weights summing to 1.0, indexed by instrument in input order. Deterministic.
    """
    if not groups:
        raise ValueError("groups is empty")
    weights: dict[str, float] = {}
    n_classes = len(groups)
    for cls, subs in groups.items():
        if not subs:
            raise ValueError(f"asset class {cls!r} has no sub-groups")
        for sub, insts in subs.items():
            if isinstance(insts, str) or len(insts) == 0:
                raise ValueError(f"sub-group {cls!r}/{sub!r} must be a non-empty list of instruments")
            for inst in insts:
                if inst in weights:
                    raise ValueError(f"duplicate instrument {inst!r}")
                weights[inst] = 1.0 / n_classes / len(subs) / len(insts)
    return pd.Series(weights, dtype=float, name="handcrafted")


def _validate_cov(cov: pd.DataFrame) -> np.ndarray:
    if cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        raise ValueError("cov must be a non-empty square matrix")
    if list(cov.index) != list(cov.columns):
        raise ValueError("cov index and columns must match")
    a = cov.to_numpy(dtype=float)
    if not np.isfinite(a).all():
        raise ValueError("cov contains NaN or inf")
    scale = float(np.abs(a).max())
    if not np.allclose(a, a.T, rtol=1e-8, atol=1e-12 * max(scale, 1.0)):
        raise ValueError("cov is not symmetric")
    if (np.diag(a) <= 0).any():
        raise ValueError("cov has a non-positive diagonal")
    a = 0.5 * (a + a.T)
    if np.linalg.eigvalsh(a).min() < -1e-8 * scale:
        raise ValueError("cov is not positive semi-definite")
    return a


def _quasi_diag_order(link: np.ndarray, n: int) -> list[int]:
    link = link.astype(int)
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while max(order) >= n:
        expanded: list[int] = []
        for idx in order:
            if idx >= n:
                expanded.extend((int(link[idx - n, 0]), int(link[idx - n, 1])))
            else:
                expanded.append(idx)
        order = expanded
    return order


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov: pd.DataFrame, *, linkage_method: str = "single") -> pd.Series:
    """Lopez de Prado HRP over a covariance matrix of per-period returns. Output sums to 1, all >= 0.

    Diagnostic only; switching production weights to HRP is a new pre-registered trial. Raises
    ``ValueError`` on NaN, asymmetry, a non-positive diagonal or a clearly non-PSD matrix. A
    singular PSD matrix is accepted. With the positional bisection, the textbook block result
    (inverse-variance split of the two blocks) holds only when the two halves of the quasi-diagonal
    order coincide with the blocks, e.g. zero cross-block correlation in equal-size blocks.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    a = _validate_cov(cov)
    names = list(cov.columns)
    n = len(names)
    if n == 1:
        return pd.Series([1.0], index=names, name="hrp")

    sd = np.sqrt(np.diag(a))
    corr = np.clip(a / np.outer(sd, sd), -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    link = linkage(squareform(dist, checks=False), method=linkage_method)
    order = _quasi_diag_order(link, n)

    w = np.ones(n)
    clusters = [list(range(n))]
    while clusters:
        clusters = [
            c[s:e] for c in clusters if len(c) > 1 for s, e in ((0, len(c) // 2), (len(c) // 2, len(c)))
        ]
        for i in range(0, len(clusters) - 1, 2):
            left, right = clusters[i], clusters[i + 1]
            v_l = _cluster_variance(a, [order[p] for p in left])
            v_r = _cluster_variance(a, [order[p] for p in right])
            total = v_l + v_r
            alpha = 1.0 - v_l / total if total > 0 else 0.5
            w[left] *= alpha
            w[right] *= 1.0 - alpha
    out = np.zeros(n)
    out[order] = w
    return pd.Series(out, index=names, name="hrp")


def compare_weights(handcrafted: pd.Series, hrp: pd.Series) -> pd.DataFrame:
    """Columns handcrafted, hrp, diff (= hrp - handcrafted) over the union of instruments. Diagnostic only."""
    df = pd.concat({"handcrafted": handcrafted, "hrp": hrp}, axis=1).fillna(0.0)
    df["diff"] = df["hrp"] - df["handcrafted"]
    return df
