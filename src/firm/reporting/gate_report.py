"""G-RESEARCH verdict logic and report rendering (ticket P3-08).

Pure functions: no data, no ledger writes. Every threshold is READ from the gates dict (the frozen ``config/gates.yaml``, version 2);
nothing numeric is hard-coded here, so the report writer cannot soften a verdict. Research-only; never imported by a live module.

``results`` schema (per-period Sharpes unless noted; the evaluation driver fills it)::

    dsr:          {sharpe, dsr, n_gate, var_sr, ...}            sharpe = selected config per-period net Sharpe, dsr at the family N
    pbo:          {pbo, prob_oos_loss, selected_median_oos_sharpe, grid_n, effective_grid_n, median_pairwise_corr,
                   family_variant_count}
    cpcv:         {path_sharpes: [..9..]}
    cost_stress:  {net_sharpe: {"1","2","3"}, dsr: {"1","2","3"}}
    stress:       {min_active_fraction (None = not frozen), episodes: [{name, status, max_drawdown, reference_max_dd, ...}]}
    robustness:   {chosen_net_sharpe, perturbations: [{param, direction, net_sharpe}], unassessed: [names], gross_cap_bound_share}
    benchmark:    {sharpe_candidate, sharpe_bm2 (ANNUAL after-tax after-cost, same convention), correlation, margin (minimum
                   detectable Sharpe gap from the power analysis), rationale_claimed}
    mechanism:    {ok, detail}

Status vocabulary: ``pass``; ``fail`` = a POINT failure (Tier D); ``insufficient`` = a confidence/power miss or an input that cannot
support a pass (Tier C). Any gate without a result, or with an unknown status, is Tier D (fail closed, gates ``verdict.combination``).
Not-assessable robustness parameters and an unfrozen ``min_active_fraction`` are reported as ``insufficient`` (documented choice: the
evidence is missing, not adverse), never as ``pass``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from firm.validation.sharpe_stats import length_adjusted_var_sr

__all__ = [
    "IN_SAMPLE_SENTENCE",
    "REQUIRED_TESTS",
    "ProvenanceError",
    "TestOutcome",
    "combine_tier",
    "evaluate_g_research",
    "evaluate_kelly_bound",
    "gate_var_sr",
    "render_report",
    "tier_reason",
    "verify_provenance",
]

REQUIRED_TESTS = tuple(f"G-RESEARCH-{i}" for i in range(1, 9))
_STATUSES = ("pass", "fail", "insufficient")
IN_SAMPLE_SENTENCE = "all data to 2026-09-30 is in-sample; no post-seal data examined"
H4_REASON = "H4 ENB miss"


class ProvenanceError(RuntimeError):
    """A trial id referenced by a report is not a registered, completed ledger row."""


@dataclass
class TestOutcome:
    __test__ = False  # not a pytest class

    test_id: str
    value: dict
    status: str      # "pass" | "fail" | "insufficient"
    reason: str


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _missing(test_id: str, what: str, value: dict | None = None) -> TestOutcome:
    return TestOutcome(test_id, value or {}, "fail", f"missing or unusable result ({what}); fail closed")


# ---------------------------------------------------------------------------------------------------------------------
# helpers shared with the driver
# ---------------------------------------------------------------------------------------------------------------------
def gate_var_sr(srs, n_obs, n_obs_cand: int, *, grid_sharpes) -> float:
    """``var_sr`` per gates ``g_research.dsr.var_sr_rule``: length-adjusted over the family Sharpes, floored at the grid variance.

    ``max(0, var(SR_i) - mean(sampling_var_i)) + sampling_var(T_cand)``, floored at ``var(grid Sharpes)`` (ddof=1). Per-period only.
    """
    grid = np.asarray(grid_sharpes, dtype=float)
    floor = float(np.var(grid, ddof=1)) if len(grid) >= 2 else 0.0
    return length_adjusted_var_sr(srs, n_obs, int(n_obs_cand), floor=floor)


# ---------------------------------------------------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------------------------------------------------
def _g1(res: Mapping | None, g: Mapping) -> TestOutcome:
    tid = "G-RESEARCH-1"
    sr = _num((res or {}).get("sharpe"))
    d = _num((res or {}).get("dsr"))
    if sr is None or d is None:
        return _missing(tid, "sharpe/dsr", dict(res or {}))
    thr = float(g["dsr"]["threshold"])
    val = {**res, "threshold": thr}
    if sr <= 0:
        return TestOutcome(tid, val, "fail", f"point failure: selected per-period net Sharpe {sr:.5f} <= 0")
    if d < thr:
        return TestOutcome(tid, val, "insufficient", f"confidence miss: DSR {d:.4f} < {thr} at family N={res.get('n_gate')}")
    return TestOutcome(tid, val, "pass", f"DSR {d:.4f} >= {thr} at family N={res.get('n_gate')}")


def _pbo_uninformative(res: Mapping, g: Mapping) -> str | None:
    inf = g["pbo"]["informativeness"]
    min_eff = float(inf["min_effective_grid_n"])
    max_corr = float(inf["max_median_pairwise_column_corr"])
    grid_n, eff, corr = _num(res.get("grid_n")), _num(res.get("effective_grid_n")), _num(res.get("median_pairwise_corr"))
    reasons = []
    if grid_n is None or grid_n < min_eff:
        reasons.append(f"grid N {grid_n} < {min_eff:g}")
    if eff is None or eff < min_eff:
        reasons.append(f"effective grid N {eff} < {min_eff:g}")
    if corr is None or corr > max_corr:
        reasons.append(f"median pairwise column correlation {corr} > {max_corr}")
    return "; ".join(reasons) or None


def _g2(res: Mapping | None, g: Mapping) -> TestOutcome:
    tid = "G-RESEARCH-2"
    p, oos_loss = _num((res or {}).get("pbo")), _num((res or {}).get("prob_oos_loss"))
    if p is None or oos_loss is None:
        return _missing(tid, "pbo/prob_oos_loss", dict(res or {}))
    cfg = g["pbo"]
    thr = float(cfg["threshold"])
    extra = cfg.get("threshold_when_variants_exceed") or {}
    n_var = _num(res.get("family_variant_count"))
    if extra and n_var is not None and n_var > float(extra["variant_count_threshold"]):
        thr = float(extra["threshold"])
    val = {**res, "threshold": thr}
    why = _pbo_uninformative(res, g)
    if why:
        return TestOutcome(tid, val, "insufficient", f"PBO uninformative ({why}); never a pass, never a hard fail")
    sel = _num(res.get("selected_median_oos_sharpe"))
    if sel is None:
        return _missing(tid, "selected_median_oos_sharpe", val)
    if (p >= 0.5 and oos_loss >= 0.5) or sel <= 0:
        return TestOutcome(tid, val, "fail", f"point failure: PBO {p:.3f}, prob_oos_loss {oos_loss:.3f}, selected median OOS Sharpe {sel:.5f}")
    if p >= thr:
        return TestOutcome(tid, val, "insufficient", f"confidence miss: PBO {p:.3f} >= {thr}")
    return TestOutcome(tid, val, "pass", f"PBO {p:.3f} < {thr}")


def _g3(res: Mapping | None, g: Mapping) -> TestOutcome:
    tid = "G-RESEARCH-3"
    paths = [_num(x) for x in ((res or {}).get("path_sharpes") or [])]
    if not paths or any(x is None for x in paths):
        return _missing(tid, "path_sharpes", dict(res or {}))
    c = g["cpcv"]
    n_paths, min_pos, min_med = int(c["n_paths"]), int(c["min_positive_paths"]), float(c["median_path_sharpe_min_exclusive"])
    med = float(np.median(paths))
    pos = sum(x > 0 for x in paths)
    val = {**res, "median": med, "n_positive": pos, "n_paths_required": n_paths, "min_positive_paths": min_pos}
    if len(paths) != n_paths:
        return TestOutcome(tid, val, "insufficient", f"{len(paths)} paths, {n_paths} required")
    if not med > min_med:
        return TestOutcome(tid, val, "fail", f"point failure: median CPCV path Sharpe {med:.5f} <= {min_med}")
    if pos < min_pos:
        return TestOutcome(tid, val, "insufficient", f"confidence miss: {pos} of {n_paths} paths positive (< {min_pos})")
    return TestOutcome(tid, val, "pass", f"median path Sharpe {med:.5f}; {pos} of {n_paths} paths positive")


def _g4(res: Mapping | None, g: Mapping) -> TestOutcome:
    tid = "G-RESEARCH-4"
    c = g["cost_stress"]
    key = str(int(c["gate_multiplier"]))
    ns = _num(((res or {}).get("net_sharpe") or {}).get(key))
    d = _num(((res or {}).get("dsr") or {}).get(key))
    if ns is None or d is None:
        return _missing(tid, f"net_sharpe/dsr at {key}x", dict(res or {}))
    dmin, nmin = float(c["dsr_min"]), float(c["net_sharpe_min_exclusive"])
    if not ns > nmin:
        return TestOutcome(tid, dict(res), "fail", f"point failure: net Sharpe {ns:.5f} <= {nmin} at {key}x cost")
    if d < dmin:
        return TestOutcome(tid, dict(res), "insufficient", f"confidence miss: DSR {d:.4f} < {dmin} at {key}x cost")
    return TestOutcome(tid, dict(res), "pass", f"net Sharpe {ns:.5f} > {nmin} and DSR {d:.4f} >= {dmin} at {key}x cost")


def _g5(res: Mapping | None, g: Mapping) -> TestOutcome:
    tid = "G-RESEARCH-5"
    eps = (res or {}).get("episodes")
    if not eps:
        return _missing(tid, "episodes", dict(res or {}))
    mult = float(g["stress"]["max_loss_multiple_of_charter_max_dd"])
    breaches, non_ok = [], []
    for e in eps:
        mdd, ref = _num(e.get("max_drawdown")), _num(e.get("reference_max_dd"))
        if mdd is not None and ref is not None and mdd > mult * ref:
            breaches.append(f"{e.get('name')}: {mdd:.3f} > {mult:g} x {ref:.3f}")
        if e.get("status") != "ok":
            non_ok.append(f"{e.get('name')}={e.get('status')}")
    val = {**res, "multiple": mult, "breaches": breaches, "non_ok": non_ok}
    if breaches:
        return TestOutcome(tid, val, "fail", "point failure: " + "; ".join(breaches))
    if (res or {}).get("min_active_fraction") is None:
        return TestOutcome(tid, val, "insufficient", "min_active_fraction is not frozen in the pre-registration; the suite cannot support a pass")
    if non_ok:
        return TestOutcome(tid, val, "insufficient", "episodes not scored ok: " + ", ".join(non_ok))
    return TestOutcome(tid, val, "pass", f"{len(eps)} episodes within {mult:g} x the bootstrap reference")


def _g6(res: Mapping | None, g: Mapping) -> TestOutcome:
    tid = "G-RESEARCH-6"
    chosen = _num((res or {}).get("chosen_net_sharpe"))
    perts = (res or {}).get("perturbations")
    cap = _num((res or {}).get("gross_cap_bound_share"))
    if chosen is None or perts is None or cap is None:
        return _missing(tid, "chosen_net_sharpe/perturbations/gross_cap_bound_share", dict(res or {}))
    r = g["robustness"]
    floor = float(r["pass_if_perturbed_net_sharpe_at_least"])
    cap_max = float(r["max_fraction_of_days_at_gross_cap"])
    breaches = []
    for p in perts:
        ns = _num(p.get("net_sharpe"))
        if ns is None or chosen <= 0 or ns < floor * chosen:
            breaches.append(f"{p.get('param')} {p.get('direction')}: {p.get('net_sharpe')}")
    val = {**res, "floor_ratio": floor, "n_breaches": len(breaches)}
    if breaches or cap > cap_max:
        why = [f"perturbed net Sharpe below {floor:g} x chosen: " + "; ".join(breaches)] if breaches else []
        if cap > cap_max:
            why.append(f"gross-cap days {cap:.3f} > {cap_max}")
        return TestOutcome(tid, val, "fail", "point failure: " + " | ".join(why))
    un = list(res.get("unassessed") or [])
    if un or not perts:
        return TestOutcome(tid, val, "insufficient", "parameters not assessed: " + (", ".join(un) or "none perturbed"))
    return TestOutcome(tid, val, "pass", f"all {len(perts)} perturbations >= {floor:g} x chosen; gross-cap days {cap:.3f} <= {cap_max}")


def _g7(res: Mapping | None, g: Mapping) -> TestOutcome:
    tid = "G-RESEARCH-7"
    sc, sb, corr = (_num((res or {}).get(k)) for k in ("sharpe_candidate", "sharpe_bm2", "correlation"))
    if sc is None or sb is None or corr is None:
        return _missing(tid, "sharpe_candidate/sharpe_bm2/correlation", dict(res or {}))
    b = g["benchmark"]
    cmax = float(b["correlation_max"])
    margin = _num(res.get("margin"))
    claimed = bool(res.get("rationale_claimed"))
    gap = sc - sb
    low_corr_branch = corr <= cmax and claimed
    val = {**res, "gap": gap, "correlation_max": cmax, "benchmark_variant": b.get("gate_variant")}
    if gap <= 0:
        if low_corr_branch:
            return TestOutcome(tid, val, "pass", f"fails the point estimate (gap {gap:+.3f}) but correlation {corr:.2f} <= {cmax} with the charter rationale")
        return TestOutcome(tid, val, "fail", f"point failure: gap {gap:+.3f} <= 0 vs BM2 and the correlation branch does not apply (corr {corr:.2f})")
    if margin is not None and gap >= margin:
        return TestOutcome(tid, val, "pass", f"beats BM2 by {gap:+.3f} >= margin {margin:.3f}")
    if low_corr_branch:
        return TestOutcome(tid, val, "pass", f"beats BM2 by less than the margin ({gap:+.3f}) but correlation {corr:.2f} <= {cmax} with the charter rationale")
    why = "no power analysis (margin unavailable)" if margin is None else f"margin {margin:.3f}"
    return TestOutcome(tid, val, "insufficient", f"confidence miss: beats BM2 by {gap:+.3f} < {why}; correlation {corr:.2f} > {cmax} or no rationale")


def _g8(res: Mapping | None, g: Mapping) -> TestOutcome:
    tid = "G-RESEARCH-8"
    if not res or "ok" not in res:
        return _missing(tid, "mechanism", dict(res or {}))
    if res["ok"] is True:
        return TestOutcome(tid, dict(res), "pass", str(res.get("detail") or "charter committed before the first core_v1 ledger row"))
    return TestOutcome(tid, dict(res), "fail", f"point failure: charter missing or not timestamped before the first row ({res.get('detail')})")


_EVAL = (("dsr", _g1), ("pbo", _g2), ("cpcv", _g3), ("cost_stress", _g4), ("stress", _g5), ("robustness", _g6),
         ("benchmark", _g7), ("mechanism", _g8))


def evaluate_g_research(results: dict, gates: dict) -> list[TestOutcome]:
    """The eight G-RESEARCH outcomes, thresholds from ``gates['g_research']``. Missing results are failures (fail closed)."""
    g = gates["g_research"]
    return [fn(results.get(key), g) for key, fn in _EVAL]


def evaluate_kelly_bound(inputs: Mapping, gates: dict) -> TestOutcome:
    """``tau <= kelly_fraction x Kelly vol``; ``deflated_sharpe_annual`` is the selected config's DEFLATED annual Sharpe.

    ``gates.charter.kelly_tau_bound`` supplies the rule (fraction 0.5 from the rule text, ``sharpe_haircut``). A breach is Tier D.
    """
    from firm.risk.kelly import check_tau

    tid = "KELLY-TAU-BOUND"
    rule = gates["charter"]["kelly_tau_bound"]
    tau, s = _num(inputs.get("tau")), _num(inputs.get("deflated_sharpe_annual"))
    if tau is None or s is None:
        return _missing(tid, "tau/deflated_sharpe_annual", dict(inputs))
    chk = check_tau(tau, s, float(rule["sharpe_haircut"]), kelly_fraction=0.5)
    val = {"tau": tau, "deflated_sharpe_annual": s, "haircut": float(rule["sharpe_haircut"]), "max_tau": chk.max_tau,
           "planning_sharpe": chk.planning_sharpe}
    if rule.get("enforce") is not True:
        return TestOutcome(tid, val, "pass" if chk.ok else "insufficient", "informational (enforce is not true in gates)")
    if chk.ok:
        return TestOutcome(tid, val, "pass", f"tau {tau:g} <= max tau {chk.max_tau:.4f}")
    return TestOutcome(tid, val, "fail", f"point failure: tau {tau:g} > max tau {chk.max_tau:.4f} (Kelly bound on the deflated Sharpe, 50% haircut)")


# ---------------------------------------------------------------------------------------------------------------------
# tier
# ---------------------------------------------------------------------------------------------------------------------
def combine_tier(outcomes: Sequence[TestOutcome], enb_ok: bool) -> str:
    """"A" only if all eight gates (and any extra outcome, e.g. the Kelly bound) pass and ENB holds; any point failure or ENB miss is
    "D"; otherwise "C". A missing required gate or an unknown status is "D" (fail closed)."""
    if not enb_ok:
        return "D"
    ids = {o.test_id for o in outcomes}
    if any(t not in ids for t in REQUIRED_TESTS):
        return "D"
    if any(o.status not in _STATUSES for o in outcomes):
        return "D"
    if any(o.status == "fail" for o in outcomes):
        return "D"
    if any(o.status == "insufficient" for o in outcomes):
        return "C"
    return "A"


def tier_reason(outcomes: Sequence[TestOutcome], enb_ok: bool) -> str:
    if not enb_ok:
        return H4_REASON
    ids = {o.test_id for o in outcomes}
    parts = [f"{t} missing" for t in REQUIRED_TESTS if t not in ids]
    parts += [f"{o.test_id}: unknown status {o.status!r}" for o in outcomes if o.status not in _STATUSES]
    parts += [f"{o.test_id}: {o.reason}" for o in outcomes if o.status == "fail"]
    if not parts:
        parts = [f"{o.test_id}: {o.reason}" for o in outcomes if o.status == "insufficient"]
    return "; ".join(parts) if parts else "all gates pass"


# ---------------------------------------------------------------------------------------------------------------------
# provenance and rendering
# ---------------------------------------------------------------------------------------------------------------------
def verify_provenance(report_inputs: dict, ledger) -> None:
    """Raise :class:`ProvenanceError` unless every trial id in ``report_inputs['trial_ids']`` is a ``registered``, ``completed`` row.

    ``ledger`` is the frame returned by ``firm.research.ledger.trials()`` (needs ``trial_id``, ``mode``, ``status``).
    """
    ids = [str(t) for t in report_inputs.get("trial_ids") or []]
    if not ids:
        raise ProvenanceError("report references no trial ids")
    rows = {str(r.trial_id): r for r in ledger.itertuples(index=False)}
    absent = [t for t in ids if t not in rows]
    if absent:
        raise ProvenanceError(f"{len(absent)} trial id(s) not in the ledger, e.g. {absent[0]}")
    not_reg = [t for t in ids if rows[t].mode != "registered"]
    if not_reg:
        raise ProvenanceError(f"{len(not_reg)} trial id(s) not registered (mode != registered), e.g. {not_reg[0]}")
    not_done = [t for t in ids if rows[t].status != "completed"]
    if not_done:
        raise ProvenanceError(f"{len(not_done)} trial id(s) not completed, e.g. {not_done[0]}")


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.5g}"
    return str(v)


def render_report(outcomes: Sequence[TestOutcome], tier: str, meta: dict, comparators: dict) -> str:
    """REPORT.md text: tier, every gate row with its values, N and var_sr with sources, comparators and the seal sentences."""
    label = f" ({meta['tier_label']})" if meta.get("tier_label") else ""
    verdict = {"A": "All eight gates pass. Paper-eligible only on the stated N basis.",
               "C": "No point failure, but a confidence or power threshold was missed. Outcome: passive (BM2, or the owner's 92/8 book).",
               "D": "A point failure or an H4 ENB miss. Outcome: passive (BM2, or the owner's 92/8 book); the candidate stops."}[tier]
    L = ["# core_v1 research run: G-RESEARCH report", "", f"**Tier {tier}{label}.** {verdict}", ""]
    if meta.get("tier_reason"):
        L += [f"Reason: {meta['tier_reason']}", ""]
    L += [f"> {IN_SAMPLE_SENTENCE}.", ""]
    L += ["## Provenance", "", f"- data_snapshot_id: `{meta.get('data_snapshot_id')}`", f"- code commit: `{meta.get('code_commit')}`",
          f"- prereg hash: `{meta.get('prereg_hash')}`", f"- gates sha256: `{meta.get('gates_sha256')}`", ""]
    L += ["## Trial count and variance (DSR)", "",
          f"- N (gate, family-N): {meta.get('n_gate')}; raw count in the ledger: {meta.get('n_raw')}; N=463 and N=210 shown in the sensitivities.",
          f"- var_sr: {_fmt(meta.get('var_sr'))} ({meta.get('var_sr_source', 'length-adjusted over the var_sr_family rows, floored at the grid variance')})", ""]
    if "enb" in meta:
        L += ["## H4 diversification", "", f"- ENB (asset-class P&L, Meucci): {_fmt(meta['enb'])} vs minimum {_fmt(meta.get('enb_threshold'))}", ""]
    L += ["## Gates", "", "| test | status | reason |", "|---|---|---|"]
    for o in outcomes:
        L.append(f"| {o.test_id} | **{o.status.upper()}** | {o.reason} |")
    L += ["", "### Values", ""]
    for o in outcomes:
        L += [f"**{o.test_id}** ({o.status})", "", "```", *(f"{k}: {_fmt(v)}" for k, v in o.value.items()), "```", ""]
    if comparators:
        L += ["## Comparators (same engine, same Sharpe convention)", ""]
        for name, vals in comparators.items():
            L.append(f"- {name}: " + ", ".join(f"{k}={_fmt(v)}" for k, v in vals.items()))
        L.append("")
    for key in ("sensitivities", "deviations", "known_leakage"):
        items = meta.get(key)
        if items:
            L += [f"## {key.replace('_', ' ').capitalize()}", "", *[f"- {i}" for i in items], ""]
    L += ["## Decision note", "",
          ("Tier A: proceed to P5-01/P6-03." if tier == "A" else
           "Tier C/D or ENB miss: stop. The outcome is passive (BM2, or the 92/8 book with its active satellite noted); this is a legitimate result."), "",
          f"{IN_SAMPLE_SENTENCE}.", ""]
    return "\n".join(L)
