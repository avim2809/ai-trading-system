"""Effective number of trials (ONC / effective rank / Marchenko-Pastur) and the single gate-N rule (P1-03).

Methods: Lopez de Prado, *Machine Learning for Asset Managers* (2020) for ONC; eigenvalue-share entropy
(``effective_rank``, called ``enb`` in the API, weightless, NOT Meucci's weighted ENB); Marchenko-Pastur (1967)
signal-eigenvalue count. ``strategy_correlation.py`` does not exist and is not used. Legacy Sharpe-only trials
are never correlated (no return series): they enter ``gate_n`` as raw counts.

ONC here is the simple version (k-means on the distance matrix for k=2..max_k, quality = mean/std of the
silhouette, k=1 special case when every |rho| > ``one_cluster_rho``); the base-clustering refinement is omitted.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from firm.validation.diversification import effective_rank, n_signal_eigenvalues

log = logging.getLogger(__name__)

__all__ = ["TrialCount", "effective_n", "effective_n_all", "gate_n", "onc_clusters", "pairwise_corr"]

ONE_CLUSTER_RHO = 0.95


def pairwise_corr(R: pd.DataFrame, min_overlap: int = 250) -> pd.DataFrame:
    """Pairwise-complete correlation; NaN where fewer than ``min_overlap`` overlapping observations."""
    return R.corr(min_periods=min_overlap)


def _nearest_psd_corr(c: np.ndarray) -> np.ndarray:
    """Clip negative eigenvalues to 0 and renormalise the diagonal; logs the clip magnitude."""
    c = (c + c.T) / 2.0
    lam, e = np.linalg.eigh(c)
    neg = float(-lam[lam < 0].sum())
    if neg > 1e-12:
        log.info("PSD projection of correlation matrix: clipped negative eigenvalue mass %.6g", neg)
    out = (e * np.clip(lam, 0.0, None)) @ e.T
    d = np.sqrt(np.clip(np.diag(out), 1e-18, None))
    out = out / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return (out + out.T) / 2.0


def _prepare(R: pd.DataFrame, min_overlap: int) -> tuple[pd.DataFrame | None, int, int]:
    """(psd corr as DataFrame or None if <=1 live column, n_obs, n_columns_total)."""
    n_total = R.shape[1]
    live = R.loc[:, R.std(skipna=True).fillna(0.0) > 0.0]
    if live.shape[1] <= 1:
        return None, int(R.shape[0]), n_total
    if n_total - live.shape[1]:
        log.info("dropped %d constant column(s): counted raw, redundant", n_total - live.shape[1])
    c = pairwise_corr(live, min_overlap)
    nan = int(c.isna().to_numpy().sum())
    if nan:
        log.warning("%d correlation entries NaN (overlap < %d); treated as 0 (independent: conservative)",
                    nan, min_overlap)
    arr = c.fillna(0.0).to_numpy(dtype=float, copy=True)
    np.fill_diagonal(arr, 1.0)
    n_obs = int(live.notna().sum().median())
    return pd.DataFrame(_nearest_psd_corr(arr), index=live.columns, columns=live.columns), n_obs, n_total


def onc_clusters(corr: pd.DataFrame, max_k: int | None = None, n_init: int = 10, seed: int = 0) -> list[list[str]]:
    n = corr.shape[0]
    names = list(corr.columns)
    if n <= 2:
        return [names] if n < 2 else [[names[0]], [names[1]]]
    a = np.nan_to_num(corr.to_numpy(dtype=float, copy=True), nan=0.0)
    off = np.abs(a[~np.eye(n, dtype=bool)])
    if off.min() > ONE_CLUSTER_RHO:
        return [names]
    dist = np.sqrt(np.clip(0.5 * (1.0 - a), 0.0, None))
    max_k = min(max_k if max_k is not None else max(2, n // 2), n - 1)
    best: tuple[float, float, np.ndarray] | None = None
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(dist)
        if len(set(labels)) < 2:
            continue
        s = silhouette_samples(dist, labels, metric="precomputed")
        sd = float(s.std())
        q = float(s.mean() / sd) if sd > 1e-12 else math.inf
        key = (q, float(s.mean()))
        if best is None or key > best[:2]:
            best = (*key, labels)
    if best is None:
        return [names]
    labels = best[2]
    return [[names[i] for i in np.where(labels == c)[0]] for c in sorted(set(labels))]


def effective_n(
    trial_returns: pd.DataFrame, method: Literal["onc", "enb", "mp_eigen"], min_overlap: int = 250
) -> float:
    if method not in ("onc", "enb", "mp_eigen"):
        raise ValueError(f"unknown method {method!r}")
    if trial_returns.shape[1] == 0:
        return 0.0
    corr, n_obs, _ = _prepare(trial_returns, min_overlap)
    if corr is None:
        return 1.0
    if method == "onc":
        return float(len(onc_clusters(corr)))
    if method == "enb":
        return effective_rank(corr.to_numpy())
    return float(max(1, n_signal_eigenvalues(corr.to_numpy(), n_obs)))


def effective_n_all(trial_returns: pd.DataFrame) -> dict[str, float]:
    return {
        "onc": effective_n(trial_returns, "onc"),
        "enb": effective_n(trial_returns, "enb"),
        "mp_eigen": effective_n(trial_returns, "mp_eigen"),
        "raw": float(trial_returns.shape[1]),
    }


@dataclass(frozen=True)
class TrialCount:
    raw_returns_bearing: int
    effective_onc: float
    effective_enb: float
    effective_mp: float
    legacy_ledgered: int
    legacy_estimate: int
    failed_or_no_returns: int
    unregistered: int
    api: int
    gate_n: int
    sensitivity_ledger_only: int


def _is_api(src: object) -> bool:
    s = str(src or "").replace("\\", "/").rsplit("/", 1)[-1]
    return s.startswith("api-")


def _bucket(r: pd.Series) -> str:
    """Disjoint bucket of one ledger row (priority order documented in :func:`gate_n`)."""
    if r["mode"] == "legacy":
        return "legacy_estimate" if bool(r["count_is_estimate"]) else "legacy_ledgered"
    if r["status"] == "failed":
        return "failed_or_no_returns"
    if r["returns_path"] is not None and not pd.isna(r["returns_path"]):
        return "raw_returns_bearing"
    if _is_api(r.get("source_file")):
        return "api"
    if r["mode"] == "unregistered":
        return "unregistered"
    return "failed_or_no_returns"


def gate_n(ledger_trials: pd.DataFrame, trial_returns: pd.DataFrame) -> TrialCount:
    """The single implementation of the canonical trial count N (OD-09, signed 2026-10-03).

    Every ledger row falls in exactly ONE bucket (sums use ``n_variants``): legacy rows first
    (``legacy_estimate`` if ``count_is_estimate`` else ``legacy_ledgered``); then ``status=failed`` ->
    ``failed_or_no_returns`` (failed trials count toward N, whatever their mode); then rows with a
    ``returns_path`` -> ``raw_returns_bearing`` (counted only here); then API-capture rows (``source_file``
    basename ``api-*``) -> ``api``; ``mode=unregistered`` -> ``unregistered``; everything else (registered or
    exploratory without returns) -> ``failed_or_no_returns``.

    ``gate_n = max(effective_onc, effective_enb, effective_mp, raw_returns_bearing) + legacy_ledgered +
    legacy_estimate + failed_or_no_returns + unregistered + api``. Effective-N estimates are <= the raw count,
    so the max reduces to the raw count: the three methods are REPORTED but can never lower N. Never choose
    the smallest N. ``sensitivity_ledger_only`` excludes the legacy estimates (210 on the census).
    """
    sums = {k: 0 for k in ("raw_returns_bearing", "legacy_ledgered", "legacy_estimate", "failed_or_no_returns",
                           "unregistered", "api")}
    if len(ledger_trials):
        nv = ledger_trials["n_variants"].fillna(1).astype(int)
        for b, n in zip(ledger_trials.apply(_bucket, axis=1), nv, strict=True):
            sums[b] += int(n)
    ons, enb, mp = (
        (effective_n(trial_returns, m) for m in ("onc", "enb", "mp_eigen"))
        if trial_returns.shape[1]
        else (0.0, 0.0, 0.0)
    )
    base = max(ons, enb, mp, float(sums["raw_returns_bearing"]))
    other = sum(v for k, v in sums.items() if k != "raw_returns_bearing")
    gate = int(math.ceil(base - 1e-9)) + other
    sens = gate - sums["legacy_estimate"] - (int(math.ceil(base - 1e-9)) - sums["raw_returns_bearing"])
    return TrialCount(
        raw_returns_bearing=sums["raw_returns_bearing"], effective_onc=float(ons), effective_enb=float(enb),
        effective_mp=float(mp), legacy_ledgered=sums["legacy_ledgered"], legacy_estimate=sums["legacy_estimate"],
        failed_or_no_returns=sums["failed_or_no_returns"], unregistered=sums["unregistered"], api=sums["api"],
        gate_n=gate, sensitivity_ledger_only=sens,
    )
