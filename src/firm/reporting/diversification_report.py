"""Per-family return streams and diversification report: the H4 ENB exit criterion (ticket P4-02; research-only).

"Family" means a signal-family return stream (EWMAC-fast, EWMAC-slow, breakout, optionally carry). Inputs are
ledger-registered trial returns only (per-family daily returns and per-asset-class daily P&L-contribution streams from the
P3-09 engine, each carrying a ``trial_id``); this module reads no monitor or post-seal data.

Gate quantity: ``enb_asset_class``, Meucci (2009) weight-dependent ENB on the covariance of the asset-class P&L streams with
unit weights (they already carry risk size), computed by the single implementation
``firm.validation.diversification.effective_number_of_bets``. The method name and thresholds are read from ``gates.yaml``
(``phase_exits.H4``); nothing numeric about the gate is defined here. Everything else is a diagnostic and never gates.
The report states a miss and the stop rule; it never suggests universe changes.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import yaml

from firm.validation.diversification import (
    calm_stress_corr_test,
    diversification_ratio,
    effective_number_of_bets,
    effective_rank,
    marchenko_pastur_edge,
    n_signal_eigenvalues,
    participation_ratio,
)
from firm.validation.effective_trials import effective_n_all

log = logging.getLogger(__name__)

__all__ = [
    "ENB_METHOD",
    "DiversificationReport",
    "build_report",
    "load_gates",
    "render_markdown",
    "write_report",
]

ENB_METHOD = "meucci_pca_pnl_cov_on_asset_class_pnl"  # the method name frozen in gates.yaml phase_exits.H4.enb_method
MIN_OVERLAP = 250  # same rule as P1-03
MIN_FAMILIES_FOR_FAMILY_ENB = 3


@dataclass(frozen=True)
class DiversificationReport:
    trial_ids: tuple[str, ...]
    corr: pd.DataFrame
    participation_ratio: float
    enb_family: float  # Shannon entropy of correlation eigenvalues; reported diagnostic
    enb_asset_class: (
        float  # GATE quantity: Meucci weight-dependent ENB on asset-class P&L covariance
    )
    enb_asset_class_corr_entropy: float  # diagnostic only
    diversification_ratio: (
        float  # sum(w_i sigma_i) / sqrt(w' Sigma w), unit weights over contribution streams
    )
    rolling_corr: pd.DataFrame  # rolling mean pairwise family corr and rolling asset-class ENB
    stress_corr_flags: tuple[
        tuple[str, str, float], ...
    ]  # family pairs with r_stress > flag threshold
    mp_edge: float
    n_eigs_above_mp: int
    calm_stress: pd.DataFrame
    enb_threshold: float
    enb_pass: bool  # evaluated on enb_asset_class only
    # additions beyond the interface sketch
    instrument_type: str = "etf"
    stress_flag_threshold: float = math.nan
    stress_corr_flags_asset_class: tuple[tuple[str, str, float], ...] = ()
    calm_stress_asset_class: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_families: int = 0
    n_overlap_days: int = 0
    family_enb_informative: bool = False
    rolling_window: int | None = None
    effective_n: Mapping[str, float] = field(default_factory=dict)


def load_gates(path: str | Path = "config/gates.yaml") -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _min_overlap(df: pd.DataFrame) -> int:
    ok = df.notna().to_numpy(dtype=int)
    ov = ok.T @ ok
    n = ov.shape[0]
    return int(ov[~np.eye(n, dtype=bool)].min())


def _check_frame(df: Any, name: str) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.shape[1] < 2:
        raise ValueError(f"{name} is required and needs at least two streams")
    ov = _min_overlap(df)
    if ov < MIN_OVERLAP:
        raise ValueError(f"{name}: pairwise overlap {ov} < {MIN_OVERLAP} days")
    if (df.std(skipna=True).fillna(0.0) <= 0).any():
        raise ValueError(
            f"{name}: constant stream(s) {list(df.columns[df.std(skipna=True).fillna(0.0) <= 0])}"
        )
    return df


def _h4(gates: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        return gates["phase_exits"]["H4"]
    except (KeyError, TypeError) as exc:
        raise ValueError("gates has no phase_exits.H4 block") from exc


def _rolling(fam: pd.DataFrame, ac: pd.DataFrame, window: int) -> pd.DataFrame:
    def run(df: pd.DataFrame, fn) -> pd.Series:
        vals = np.full(len(df), np.nan)
        a = df.to_numpy(dtype=float)
        for e in range(window, len(df) + 1):
            w = a[e - window : e]
            if np.isnan(w).any():
                continue
            try:
                vals[e - 1] = fn(w)
            except ValueError, np.linalg.LinAlgError:
                continue
        return pd.Series(vals, index=df.index)

    def mean_corr(w: np.ndarray) -> float:
        c = np.corrcoef(w, rowvar=False)
        return float(c[np.triu_indices_from(c, k=1)].mean())

    def enb(w: np.ndarray) -> float:
        return effective_number_of_bets(np.cov(w, rowvar=False), np.ones(w.shape[1]))

    index = fam.index.union(ac.index)
    return pd.DataFrame(
        {
            "mean_pairwise_corr_family": run(fam, mean_corr).reindex(index),
            "enb_asset_class": run(ac, enb).reindex(index),
        }
    )


def _flags(cs: pd.DataFrame, threshold: float) -> tuple[tuple[str, str, float], ...]:
    hot = cs[cs["r_stress"] > threshold]
    return tuple((str(r.pair[0]), str(r.pair[1]), float(r.r_stress)) for r in hot.itertuples())


def build_report(
    family_returns: pd.DataFrame,
    asset_class_returns: pd.DataFrame,
    trial_ids: Sequence[str],
    stress_mask: pd.Series,
    gates: Mapping[str, Any],
    *,
    instrument_type: Literal["etf", "futures"] = "etf",
    rolling_window: int | None = None,
) -> DiversificationReport:
    """Build the report. ``rolling_window`` is used only if ``gates`` carries no numeric rolling window (it is deferred in
    ``gates.yaml``); if neither is available the rolling diagnostics are skipped and the report says so."""
    ids = tuple(str(t) for t in trial_ids)
    if not ids or any(not t for t in ids):
        raise ValueError("trial_ids must be a non-empty sequence of ledger trial ids")
    if instrument_type not in ("etf", "futures"):
        raise ValueError("instrument_type must be 'etf' or 'futures'")
    fam = _check_frame(family_returns, "family_returns")
    ac = _check_frame(asset_class_returns, "asset_class_returns")

    h4 = _h4(gates)
    if h4.get("enb_method") != ENB_METHOD:
        raise ValueError(
            f"gates enb_method {h4.get('enb_method')!r} is not the frozen method {ENB_METHOD!r}"
        )
    thr_key = "enb_min_etf" if instrument_type == "etf" else "enb_min_futures"
    enb_threshold = h4.get(thr_key)
    flag_thr = h4.get("stress_correlation_flag_threshold")
    if not isinstance(enb_threshold, (int, float)) or not isinstance(flag_thr, (int, float)):
        raise ValueError(  # noqa: TRY004
            f"gates must provide numeric {thr_key} and stress_correlation_flag_threshold"
        )
    window = h4.get("stress_correlation_rolling_window")
    if not (isinstance(window, int) and not isinstance(window, bool)):
        window = rolling_window
    if window is not None and window < 3:
        raise ValueError("rolling window must be at least 3")

    corr = fam.corr(min_periods=MIN_OVERLAP)
    n_fam = fam.shape[1]
    t_overlap = _min_overlap(fam)
    corr_arr = corr.to_numpy(dtype=float)
    pr = participation_ratio(corr_arr)
    enb_family = effective_rank(corr_arr)
    mp_edge = marchenko_pastur_edge(n_fam, t_overlap)
    n_above = n_signal_eigenvalues(corr_arr, t_overlap)

    sigma = ac.cov(min_periods=MIN_OVERLAP)
    k = ac.shape[1]
    enb_ac = effective_number_of_bets(sigma.to_numpy(dtype=float), np.ones(k))
    enb_ac_corr = effective_rank(ac.corr(min_periods=MIN_OVERLAP).to_numpy(dtype=float))
    div_ratio = diversification_ratio(
        np.ones(k), np.sqrt(np.diag(sigma.to_numpy(dtype=float))), ac.corr().to_numpy(dtype=float)
    )

    cs_fam = calm_stress_corr_test(fam, stress_mask, min_overlap=MIN_OVERLAP)
    cs_ac = calm_stress_corr_test(ac, stress_mask, min_overlap=MIN_OVERLAP)
    eff = effective_n_all(family_returns)
    rolling = _rolling(fam, ac, window) if window is not None else pd.DataFrame()
    enb_pass = bool(enb_ac >= enb_threshold)
    if not enb_pass:
        log.warning(
            "H4 ENB miss: enb_asset_class %.3f < %s; the candidate stops (no post-hoc universe changes)",
            enb_ac,
            enb_threshold,
        )
    log.info(
        "diversification report: trials=%s enb_asset_class=%.3f pass=%s", ids, enb_ac, enb_pass
    )
    return DiversificationReport(
        trial_ids=ids,
        corr=corr,
        participation_ratio=pr,
        enb_family=enb_family,
        enb_asset_class=enb_ac,
        enb_asset_class_corr_entropy=enb_ac_corr,
        diversification_ratio=div_ratio,
        rolling_corr=rolling,
        stress_corr_flags=_flags(cs_fam, float(flag_thr)),
        mp_edge=mp_edge,
        n_eigs_above_mp=n_above,
        calm_stress=cs_fam,
        enb_threshold=float(enb_threshold),
        enb_pass=enb_pass,
        instrument_type=instrument_type,
        stress_flag_threshold=float(flag_thr),
        stress_corr_flags_asset_class=_flags(cs_ac, float(flag_thr)),
        calm_stress_asset_class=cs_ac,
        n_families=n_fam,
        n_overlap_days=t_overlap,
        family_enb_informative=n_fam >= MIN_FAMILIES_FOR_FAMILY_ENB,
        rolling_window=window,
        effective_n=dict(eff),
    )


def _f(x: float) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.3f}"


def _calm_stress_table(cs: pd.DataFrame) -> list[str]:
    lines = [
        "| pair | r_calm | r_stress | z | p | p_adj (BH) | reject |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in cs.itertuples():
        lines.append(
            f"| {r.pair[0]} / {r.pair[1]} | {_f(r.r_calm)} | {_f(r.r_stress)} | {_f(r.z)} | {_f(r.p)} | {_f(r.p_adj)} | {r.reject} |"
        )
    return lines


def render_markdown(rep: DiversificationReport) -> str:
    L: list[str] = ["# Diversification report (H4 ENB exit)", ""]
    L.append("Trial ids: " + ", ".join(rep.trial_ids))
    L.append(
        f"Streams: {rep.n_families} signal families; family overlap {rep.n_overlap_days} days; instrument type {rep.instrument_type}."
    )
    L += ["", "## H4 gate: effective number of bets across asset classes", ""]
    L.append(
        f"- Gate quantity `enb_asset_class` (Meucci weight-dependent ENB on asset-class P&L covariance): **{_f(rep.enb_asset_class)}**"
    )
    L.append(f"- Threshold (from gates.yaml, {rep.instrument_type}): {_f(rep.enb_threshold)}")
    L.append(f"- Result: **{'PASS' if rep.enb_pass else 'MISS'}**")
    if not rep.enb_pass:
        L += [
            "",
            "**STOP: H4 ENB miss. The candidate stops here; no post-hoc universe changes are allowed.**",
        ]
    L += ["", "## Diagnostics (reported, never gating)", ""]
    L.append(
        f"- Asset-class correlation-entropy ENB (weightless): {_f(rep.enb_asset_class_corr_entropy)}"
    )
    L.append(
        f"- Family ENB (correlation-eigenvalue entropy): {_f(rep.enb_family)}, bounded above by the family count {rep.n_families}"
    )
    if not rep.family_enb_informative:
        L.append(
            f"  - Family ENB is uninformative with fewer than {MIN_FAMILIES_FOR_FAMILY_ENB} families."
        )
    L.append(f"- Participation ratio: {_f(rep.participation_ratio)}")
    L.append(
        f"- Diversification ratio (unit weights over asset-class streams): {_f(rep.diversification_ratio)}"
    )
    L.append(
        f"- Marchenko-Pastur upper edge: {_f(rep.mp_edge)}; eigenvalues above it: {rep.n_eigs_above_mp}"
    )
    if rep.effective_n:
        L.append(
            "- Effective number of trials (family streams): "
            + ", ".join(f"{k}={_f(v)}" for k, v in rep.effective_n.items())
        )
    if rep.rolling_window is None:
        L.append(
            "- Rolling diagnostics: not computed (the rolling window is deferred in gates.yaml and was not supplied)."
        )
    else:
        L.append(f"- Rolling window: {rep.rolling_window} days (series in the JSON output).")
    L += ["", "## Stress correlation", ""]
    if rep.stress_corr_flags or rep.stress_corr_flags_asset_class:
        L.append(
            f"**STRESS CORRELATION FLAG: pairs with r_stress above {_f(rep.stress_flag_threshold)}**"
        )
        for a, b, r in rep.stress_corr_flags:
            L.append(f"- family {a} / {b}: r_stress = {_f(r)}")
        for a, b, r in rep.stress_corr_flags_asset_class:
            L.append(f"- asset class {a} / {b}: r_stress = {_f(r)}")
    else:
        L.append(f"No pair exceeds r_stress {_f(rep.stress_flag_threshold)}.")
    L += ["", "### Calm vs stress, family pairs (Fisher z, BH-adjusted)", ""] + _calm_stress_table(
        rep.calm_stress
    )
    L += ["", "### Calm vs stress, asset-class pairs", ""] + _calm_stress_table(
        rep.calm_stress_asset_class
    )
    L += [
        "",
        (
            "Notes: Fisher z assumes i.i.d. observations; volatility clustering makes the test anti-conservative. "
            "The stress mask is an input (stress periods file or pre-registered rule), not defined here."
        ),
        "",
    ]
    return "\n".join(L)


def _jsonable(o: Any) -> Any:
    if isinstance(o, pd.DataFrame):
        return {
            "index": [str(i) for i in o.index],
            "columns": [str(c) for c in o.columns],
            "data": _jsonable(o.to_numpy().tolist()),
        }
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, Mapping):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o if o is None or isinstance(o, str) else str(o)


def write_report(
    rep: DiversificationReport, out_dir: str | Path = "research/reports/core_v1"
) -> dict[str, Path]:
    """Write ``diversification.md`` and ``diversification.json`` (tracked, not ``runs/``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md, js = out / "diversification.md", out / "diversification.json"
    md.write_text(render_markdown(rep), encoding="utf-8")
    fields = {k: getattr(rep, k) for k in rep.__dataclass_fields__}
    js.write_text(json.dumps(_jsonable(fields), indent=2, sort_keys=True), encoding="utf-8")
    log.info("wrote diversification report to %s", out)
    return {"markdown": md, "json": js}
