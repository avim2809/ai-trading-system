"""Diversification primitives (ticket P1-03; reused by the P4-02 diversification report).

* ``participation_ratio``: ``(sum lambda)^2 / sum lambda^2`` of a correlation/covariance matrix.
* ``effective_rank``: ``exp(entropy of eigenvalue shares)``; WEIGHTLESS, used for trial counting. It is NOT
  Meucci's ENB; noise eigenvalues add entropy, so it overstates the number of factors.
* ``effective_number_of_bets``: Meucci (2009) principal-portfolio ENB, WEIGHTED (P4-02's H4 must call this with
  the actual asset-class weights, never ``effective_rank``).
* Marchenko-Pastur edge / signal-eigenvalue count, diversification ratio, Fisher-z correlation test, BH.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from firm.eval.overfitting import _norm_cdf
from firm.validation.nulls import benjamini_hochberg

__all__ = [
    "bh_adjust",
    "diversification_ratio",
    "effective_number_of_bets",
    "effective_rank",
    "fisher_z_corr_diff",
    "marchenko_pastur_edge",
    "n_signal_eigenvalues",
    "participation_ratio",
]


def _eigvals(m: np.ndarray) -> np.ndarray:
    a = np.asarray(m, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("expected a square matrix")
    return np.clip(np.linalg.eigvalsh((a + a.T) / 2.0), 0.0, None)


def participation_ratio(corr_or_cov: np.ndarray) -> float:
    lam = _eigvals(corr_or_cov)
    return float(lam.sum() ** 2 / np.sum(lam**2))


def _entropy_exp(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(math.exp(-np.sum(p * np.log(p))))


def effective_rank(cov_or_corr: np.ndarray) -> float:
    lam = _eigvals(cov_or_corr)
    return _entropy_exp(lam / lam.sum())


def effective_number_of_bets(
    cov: np.ndarray, weights: np.ndarray, method: Literal["meucci_pca", "min_torsion"] = "meucci_pca"
) -> float:
    """Meucci (2009): ``p_i = (e_i' w)^2 lambda_i / (w' cov w)``, ``ENB = exp(-sum p_i ln p_i)``."""
    if method == "min_torsion":
        raise NotImplementedError("minimum-torsion ENB is optional in P1-03 and not implemented (not used by any gate)")
    if method != "meucci_pca":
        raise ValueError(f"unknown method {method!r}")
    c = np.asarray(cov, dtype=float)
    w = np.asarray(weights, dtype=float)
    lam, e = np.linalg.eigh((c + c.T) / 2.0)
    lam = np.clip(lam, 0.0, None)
    contrib = (e.T @ w) ** 2 * lam
    tot = contrib.sum()
    if tot <= 0:
        raise ValueError("portfolio variance is zero")
    return _entropy_exp(contrib / tot)


def marchenko_pastur_edge(n_assets: int, n_obs: int, sigma2: float = 1.0) -> float:
    """``lambda_plus = sigma2 * (1 + sqrt(N/T))^2``."""
    return float(sigma2 * (1.0 + math.sqrt(n_assets / n_obs)) ** 2)


def n_signal_eigenvalues(corr: np.ndarray, n_obs: int) -> int:
    lam = np.linalg.eigvalsh((np.asarray(corr, float) + np.asarray(corr, float).T) / 2.0)
    return int(np.sum(lam > marchenko_pastur_edge(len(lam), n_obs)))


def diversification_ratio(weights: np.ndarray, vols: np.ndarray, corr: np.ndarray) -> float:
    w, v = np.asarray(weights, float), np.asarray(vols, float)
    cov = np.outer(v, v) * np.asarray(corr, float)
    return float((w @ v) / math.sqrt(w @ cov @ w))


def fisher_z_corr_diff(r1: float, n1: int, r2: float, n2: int) -> tuple[float, float]:
    """Two independent correlations: ``z = (atanh r1 - atanh r2) / sqrt(1/(n1-3) + 1/(n2-3))``; two-sided p."""
    if n1 <= 3 or n2 <= 3:
        raise ValueError("need n > 3")
    z = (math.atanh(r1) - math.atanh(r2)) / math.sqrt(1.0 / (n1 - 3) + 1.0 / (n2 - 3))
    return float(z), float(min(1.0, 2.0 * (1.0 - _norm_cdf(abs(z)))))


def bh_adjust(p: np.ndarray, q: float = 0.10) -> np.ndarray:
    """Benjamini-Hochberg at level ``q``: boolean accept mask (delegates to P1-07 ``benjamini_hochberg``)."""
    return benjamini_hochberg(np.asarray(p, dtype=float), q)
