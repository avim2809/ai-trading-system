"""P3-11 driver: pre-registered estimation of the core_v1 constants (pooled forecast scalars, per-instrument speed filter, FDM, IDM).

ONE run, on real pre-seal data, by the non-root research user, niced, memory-capped and inside the run window. Nothing is chosen here:
tau comes from the approved charter, the instrument weights and their hash from the frozen module ``scripts/core_v1_preregistered.py``
(handcrafted, one group per asset class). Run it ONCE; a re-run is a new trial and may need a new pre-registration.

    cd <worktree> && nice -n 10 ionice -c3 env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=$PWD/src \\
      prlimit --as=2500000000 <research-venv>/bin/python scripts/run_core_v1_constants.py \\
      --manifest research/data_manifests/<etf manifest>.json

Refuses to run (STOP, nothing degraded, no override flag) unless: not root; the sealed paths are unreadable; the ETF store is readable;
``docs/GUARDRAIL_REDTEAM.md`` records a pass; the clock (``date -u``) is inside the run window (weekdays 20:00-04:00 US/Eastern or
Saturday, never a Sunday in UTC); an address-space cap is set (``prlimit --as``; the run ABORTS, it does not degrade, if the cap is hit);
the frozen inputs and the charter verify. Memory: sized like ``deploy/s2-forward-shadow.service`` (MemoryMax=1500M) but as an
address-space cap, 2.5 GB, with single-threaded BLAS.

Steps (fixed order, no branching on results; every step is a ``registered`` ledger trial with ``config['kind']='constant_estimation'``):
 1. load pre-seal total-return and price series through ``firm.data.etf_loader`` (``firm.research.data_access``); record the snapshot id;
 2. vol (P3-01) and the raw EWMAC / breakout series for every pre-registered speed;
 3. scalars: one pooled scalar per rule so that mean |raw * scalar| = 10 on the UNCAPPED, UNFLOORED signed raw forecast;
 4. speed filter per instrument from cost and turnover only (drop a speed whose cost exceeds 1/3 of the charter's expected rule Sharpe);
 5. FDM per instrument from the pooled signed-rule correlation (off-diagonals floored at 0, cap 2.5), group weights split equally over survivors;
 6. IDM from the sub-system returns (one instrument, unit risk) through the P3-09 engine at 1x cost (cap 2.5);
 7. write ``constants.json`` / ``constants.md`` and the DRAFT addendum under ``research/reports/core_v1/`` only (NEVER under
    ``research/preregistration/``: the owner copies the draft in a CODEOWNERS-reviewed commit and P3-08 verifies its hash);
 8. stop.

Documented choices the tickets leave open (see the report): expected rule Sharpe 0.3 = the low end of the charter's stated 0.3-0.6 net range;
equal GROUP weights for the two rule families; reference trade 2% of a 100k account for the cost per trade; Sharpe versus zero (rf=None).
All data to 2026-09-30 is in-sample; the estimation is in-sample by construction.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import core_v1_preregistered as pre
import numpy as np
import pandas as pd
import yaml

import firm.research  # noqa: F401  (ledger / seal entry-point check: scripts/run_*.py must go through firm.research)
from firm.research import core_v1_pipeline as P
from firm.research import run_guards as G
from firm.research import seal
from firm.research.core_v1_pipeline import (
    apply_cost_speed_filter,
    estimate_fdm_rho,
    estimate_idm_H,
    estimate_pooled_scalars,
)
from firm.signals import ewmac as EW

log = logging.getLogger("run_core_v1_constants")

EXPECTED_RULE_SHARPE = 0.3          # low end of the charter's "0.3 to 0.6 net" range (charter section 3); verified against the charter text
EXPECTED_RULE_SHARPE_SOURCE = "research/charters/core_v1.md section 3: 'The plan's own range is 0.3 to 0.6 net' (low end taken)"
OUT_SUBDIR = Path("research") / "reports" / "core_v1"
ADDENDUM_DRAFT = "constants_addendum.draft.yaml"
REQUIRED_KEYS = ("schema", "family", "window", "data_snapshot_id", "gates_sha256", "prereg_fingerprint", "code_commit", "seed", "tau",
                 "instrument_weights", "instrument_weights_sha256", "group_weights", "expected_rule_sharpe", "scalars", "scalar_diagnostics",
                 "speed_filter", "rho", "fdm", "fdm_uncapped", "fdm_cap", "idm", "idm_uncapped", "idm_cap", "H", "trial_ids",
                 "in_sample_declaration", "post_seal_declaration", "meta")


def weights_hash(weights: dict[str, float], order: list[str]) -> str:
    """Same recipe as ``pre.instrument_weights_hash`` (sha256 of the weight vector in the given order, 12 dp), for any symbol list."""
    payload = json.dumps([[s, round(float(weights[s]), 12)] for s in order])
    return hashlib.sha256(payload.encode()).hexdigest()


def git_head(repo: Path) -> str:
    return subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                          check=True, timeout=30).stdout.strip()


def write_constants(path: str | Path, constants: dict, meta: dict) -> str:
    """Write ``constants.json`` (sorted keys, ``meta`` merged in) and return the sha256 of the file. Refuses any path under
    ``research/preregistration/`` (agent-denied; the owner copies the addendum there)."""
    p = Path(path).resolve()
    if "preregistration" in p.parts and "research" in p.parts:
        raise ValueError(f"refusing to write under research/preregistration/: {p}")
    body = P.jsonable({**constants, "meta": meta})
    data = (json.dumps(body, indent=1, sort_keys=True) + "\n").encode()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _sub_panel(panel: P.Panel, s: str) -> P.Panel:
    cut = {"symbols": [s], "close": panel.close[[s]], "raw_close": panel.raw_close[[s]], "ret": panel.ret[[s]], "adv": panel.adv[[s]]}
    return P.Panel(**{**panel.__dict__, **cut})


def run_estimation(panel: P.Panel, *, gates: dict, tau: float, instrument_weights: dict[str, float], window: tuple[str, str], out_dir: Path,
                   ctx: P.LedgerCtx, code_commit: str, prereg_fingerprint: str, gates_sha256: str,
                   expected_rule_sharpe: float = EXPECTED_RULE_SHARPE, max_research_date=None, group_weights: dict | None = None,
                   charter: dict | None = None, repo_dir: Path = ROOT) -> dict:
    """Steps 2-7. ``panel`` is real (main) or synthetic (tests); every step is a ledger trial recorded through ``ctx``."""
    base = P.params_from_gates(gates, tau)
    syms = list(panel.symbols)
    group_weights = dict(group_weights or P.GROUP_WEIGHTS)
    last = panel.close.index.max()
    w_end = min(pd.Timestamp(window[1]), last)
    win = (window[0], str(w_end.date()))
    out_dir.mkdir(parents=True, exist_ok=True)
    defs = P.rule_defs(base, base)

    # step 2: vol and raw forecasts (not a trial: no estimate leaves this step)
    vol = P.vol_annual(panel, base)
    raw = P.raw_forecasts(panel, base, defs, vol)

    # step 3: pooled scalars, one trial per rule
    scalars: dict[str, float] = {}
    for d in defs:
        with P.logged_trial(ctx, "constant_estimation", "scalar", d.rule_id, detail={"window": list(win), "target_mean_abs": P.SCALAR_TARGET_ABS}):
            scalars.update(estimate_pooled_scalars({d.rule_id: {s: raw[d.rule_id][s] for s in syms}}, win, max_research_date))
    diag = P.scalar_diagnostics(raw, scalars, win, base.forecast_cap)
    fcs = P.signed_forecasts(raw, scalars, base)

    # step 4: speed filter per instrument, cost and turnover only
    from firm.costs import model as CM

    cfg, spec = CM.load_cost_config(), CM.spec_from_config(P.COST_SPEC)
    notional = P.REFERENCE_TRADE_FRACTION * P.INITIAL_CAPITAL
    turn, cpt, sigma = {}, {}, {}
    for s in syms:
        f = fcs[s].loc[win[0]:win[1]]
        turn[s] = {r: float(EW.forecast_turnover(f[r].dropna())) for r in f.columns}
        v = vol[s].loc[win[0]:win[1]].dropna()
        sigma[s] = float(v.mean())
        pw = panel.close[s].loc[v.index]
        cpt[s] = P.cost_per_trade_fraction(spec, price=float(pw.median()), adv_units=float(panel.adv[s].loc[v.index].median()),
                                           vol_daily=float((v / np.sqrt(256)).median()), notional=notional, cfg=cfg)
    survivors: dict[str, dict[str, list[int]]] = {}
    for s in syms:
        with P.logged_trial(ctx, "constant_estimation", "speed_filter", s, detail={"expected_rule_sharpe": expected_rule_sharpe}):
            survivors.update(apply_cost_speed_filter({s: turn[s]}, cpt, sigma, expected_rule_sharpe, base.speed_cost_max_fraction))

    # step 5: FDM per instrument (surviving-rule set), full pooled rho stored for any later subset
    rho_full = P.pooled_rho(fcs, win, syms)
    fdm_c, fdm_u, excluded = {}, {}, []
    for s in syms:
        w = P.rule_weights(survivors[s], group_weights)
        with P.logged_trial(ctx, "constant_estimation", "fdm", s, detail={"n_rules": len(w)}):
            if not w:
                excluded.append(s)
                fdm_c[s] = fdm_u[s] = 1.0
                continue
            ids = list(w)
            _, fdm_c[s] = estimate_fdm_rho({q: fcs[q][ids] for q in syms}, win, w, cap=base.fdm_cap)
            fdm_u[s] = P.fdm_from_rho(np.array([w[i] for i in ids]), rho_full.loc[ids, ids].to_numpy(), cap=float("inf"))

    # step 6: IDM from the sub-system returns through the engine at 1x cost (one trial; the engine adds its own exploratory rows)
    sub_rets = {}
    with P.logged_trial(ctx, "constant_estimation", "idm", "all", detail={"n_subsystems": len(syms)}):
        unit = dataclasses.replace(base, gross_cap=None)
        for s in syms:
            if s in excluded:
                continue
            sb = P.ConstantsBundle(scalars=scalars, survivors={s: survivors[s]}, rho=rho_full, idm=1.0, instrument_weights={s: 1.0},
                                   group_weights=group_weights, window=win)
            res = P.run_config(_sub_panel(panel, s), sb, unit, subset_ids=None, ledger_mode=ctx.mode, label=f"idm_subsystem:{s}",
                               family="core_v1_engine", prereg=ctx.prereg, snapshot_id=ctx.snapshot_id, seed=ctx.seed)
            r = res.engine.excess_returns.copy()
            r[vol[s].isna()] = np.nan                    # before the entry gate the sub-system is not trading
            sub_rets[s] = r
        sub_df = pd.DataFrame(sub_rets)
        order = list(sub_df.columns)
        w_vec = np.array([instrument_weights[s] for s in order], dtype=float)
        w_vec = w_vec / w_vec.sum()                      # weights of the active sub-systems (a no-op unless an instrument is excluded)
        H, idm_c = estimate_idm_H(sub_df, w_vec, win, cap=base.idm_cap)
        idm_u = float(P.SZ.idm(w_vec, H, cap=float("inf")))

    ids_by_kind = Counter(k.split(":")[0] for k in ctx.trial_ids)
    constants = {
        "schema": 1, "family": pre.FAMILY, "window": list(win), "entry_gate_days": P.ENTRY_GATE_DAYS, "data_snapshot_id": panel.snapshot_id,
        "data_source": panel.source, "gates_sha256": gates_sha256, "prereg_fingerprint": prereg_fingerprint, "code_commit": code_commit,
        "seed": ctx.seed, "tau": tau, "charter": charter or {}, "instrument_weights": instrument_weights,
        "instrument_weights_sha256": weights_hash(instrument_weights, syms), "instrument_weight_scheme": pre.INSTRUMENT_WEIGHT_SCHEME,
        "group_weights": group_weights, "expected_rule_sharpe": expected_rule_sharpe,
        "expected_rule_sharpe_source": EXPECTED_RULE_SHARPE_SOURCE, "scalars": scalars, "scalar_diagnostics": diag,
        "speed_filter": {"turnover": turn, "cost_per_trade": cpt, "sigma_pct": sigma, "survivors": survivors,
                         "max_cost_fraction": base.speed_cost_max_fraction, "reference_trade_notional_usd": notional,
                         "cost_spec": P.COST_SPEC, "excluded_instruments": excluded},
        "rho": {"rules": list(rho_full.columns), "matrix": rho_full.to_numpy()}, "fdm": fdm_c, "fdm_uncapped": fdm_u,
        "fdm_cap": base.fdm_cap, "idm": idm_c, "idm_uncapped": idm_u, "idm_cap": base.idm_cap,
        "H": {"instruments": order, "matrix": H}, "idm_weights_sha256": weights_hash(dict(zip(order, w_vec, strict=True)), order),
        "trial_ids": dict(ctx.trial_ids), "n_trials_by_kind": dict(ids_by_kind),
        "in_sample_declaration": pre.IN_SAMPLE_DECLARATION, "post_seal_declaration": pre.POST_SEAL_DATA_DECLARATION,
        "known_choices": ["expected rule Sharpe is the low end of the charter's stated range", "equal group weights for ewmac and breakout",
                          "Sharpe versus zero (rf=None); engine prices are adjusted closes", "IDM estimated once on the 'all' speed set"],
    }
    meta = {"generated_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "code_commit": code_commit,
            "ledger_mode": ctx.mode, "n_symbols": len(syms)}
    sha = write_constants(out_dir / "constants.json", constants, meta)
    (out_dir / "constants.md").write_text(render_markdown(constants, sha), encoding="utf-8")
    addendum = {
        "family": pre.FAMILY, "kind": "constants_addendum", "status": "DRAFT_FOR_OWNER_COPY",
        "copy_to": "research/preregistration/core_v1_constants_addendum.yaml (CODEOWNERS-reviewed owner commit; the driver never writes there)",
        "constants_json": "research/reports/core_v1/constants.json", "constants_json_sha256": sha, "code_commit": code_commit,
        "prereg_module_fingerprint": prereg_fingerprint, "gates_sha256": gates_sha256,
        "instrument_weights_sha256": constants["instrument_weights_sha256"], "data_snapshot_id": panel.snapshot_id,
        "window": list(win), "seed": ctx.seed, "ledger_trial_ids": dict(ctx.trial_ids), "n_ledger_rows_by_kind": dict(ids_by_kind),
        "declaration": pre.POST_SEAL_DATA_DECLARATION,
    }
    (out_dir / ADDENDUM_DRAFT).write_text(yaml.safe_dump(P.jsonable(addendum), sort_keys=False), encoding="utf-8")
    log.info("constants.json sha256 %s; %d ledger trials", sha, len(ctx.trial_ids))
    return {"constants": constants, "sha256": sha, "out_dir": out_dir}


def render_markdown(c: dict, sha: str) -> str:
    L = ["# core_v1 constants (P3-11)", "", f"All data to 2026-09-30 is in-sample; {c['post_seal_declaration']}", "",
         f"- constants.json sha256: `{sha}`", f"- window: {c['window'][0]} .. {c['window'][1]}; data snapshot `{c['data_snapshot_id']}`",
         f"- tau (from the charter): {c['tau']}; code commit `{c['code_commit']}`; seed {c['seed']}",
         f"- instrument weights: {c['instrument_weight_scheme']} (sha256 `{c['instrument_weights_sha256']}`)",
         f"- expected rule Sharpe {c['expected_rule_sharpe']} ({c['expected_rule_sharpe_source']})", "",
         "## Pooled scalars (mean |raw*scalar| = 10 on the uncapped raw forecast; the capped mean is reported separately)", "",
         "| rule | scalar | mean abs uncapped | mean abs capped |", "|---|---|---|---|"]
    for r, v in c["scalars"].items():
        d = c["scalar_diagnostics"][r]
        L.append(f"| {r} | {v:.6g} | {d['mean_abs_uncapped']:.6f} | {d['mean_abs_capped']:.4f} |")
    L += ["", "## Surviving speeds per instrument (cost and turnover only)", "", "| instrument | ewmac fast spans | breakout lookbacks | FDM (capped) | FDM (uncapped) |",
          "|---|---|---|---|---|"]
    for s, sv in c["speed_filter"]["survivors"].items():
        L.append(f"| {s} | {sv['ewmac']} | {sv['breakout']} | {c['fdm'][s]:.4f} | {c['fdm_uncapped'][s]:.4f} |")
    L += ["", f"## IDM: {c['idm']:.4f} (uncapped {c['idm_uncapped']:.4f}, cap {c['idm_cap']}); FDM cap {c['fdm_cap']}", "",
          "## Choices left open by the tickets (owner review)", "", *[f"- {k}" for k in c["known_choices"]], ""]
    return "\n".join(L)


def build_ledger_ctx(panel_snapshot: str, fingerprint: str, seed: int) -> P.LedgerCtx:
    """The real run: every step a ``registered`` trial of the approved ``core_v1`` pre-registration (clean committed tree required)."""
    return P.LedgerCtx(mode="registered", prereg=pre.FAMILY, snapshot_id=panel_snapshot, seed=seed, fingerprint=fingerprint,
                       default_params=dict(pre.DEFAULT_CONFIG))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True, help="P2-02 ETF manifest json; its snapshot_id is recorded")
    ap.add_argument("--out", type=Path, default=ROOT / OUT_SUBDIR)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    # guards first, before anything is read; none can be skipped
    G.seal_preflight(ROOT, data_root=G.LIVE_CHECKOUT, euid=lambda: os.geteuid(), access=lambda p, m: os.access(p, m))
    G.assert_in_window(clock=lambda: G.current_utc())
    G.require_memory_cap()
    facts = pre.verify_before_run(ROOT)             # frozen inputs + charter (tau, approval commit); STOP on PreregError
    charter_text = (ROOT / pre.CHARTER_PATH).read_text(encoding="utf-8")
    if "0.3 to 0.6" not in charter_text:
        raise G.PreflightError("charter no longer states the 0.3-0.6 expected Sharpe range the speed filter constant is taken from")
    if "preregistration" in a.out.resolve().parts:
        raise G.PreflightError("--out must not be under research/preregistration/")
    from firm.data.etf_loader import load_dividends, load_etf_universe, usd_ils

    asof = seal.max_research_date()
    seal.check_asof(asof, what="run_core_v1_constants")
    series = load_etf_universe(ROOT / "config" / "universe_etf.yaml", asof=asof, include_delisted=False)
    manifest = json.loads(a.manifest.read_text(encoding="utf-8"))
    panel = P.panel_from_series({s: series[s] for s in pre.UNIVERSE}, snapshot_id=manifest["snapshot_id"],
                                dividends=load_dividends(pre.UNIVERSE, asof=asof), fx=usd_ils(asof, source="eodhd"),
                                source="eodhd etfs_full via firm.data.etf_loader (cleaning v3)")
    import firm.research.prereg as PR

    spec_fp = PR.spec_hash(PR.load_spec(ROOT / "research" / "preregistration" / "20261006_core_v1.yaml"))
    if weights_hash(pre.INSTRUMENT_WEIGHTS, list(panel.symbols)) != pre.INSTRUMENT_WEIGHTS_SHA256:
        raise G.PreflightError("instrument weight vector differs from the frozen hash")
    ctx = build_ledger_ctx(panel.snapshot_id, pre.bars_fingerprint(), pre.SEED)
    gates = yaml.safe_load((ROOT / pre.GATES_FILE).read_text())
    run_estimation(panel, gates=gates, tau=float(facts["tau"]), instrument_weights=pre.INSTRUMENT_WEIGHTS, window=pre.WINDOW, out_dir=a.out,
                   ctx=ctx, code_commit=git_head(ROOT), prereg_fingerprint=pre.bars_fingerprint(), gates_sha256=pre.GATES_SHA256,
                   max_research_date=seal.max_research_date(), charter={**facts, "prereg_yaml_spec_hash": spec_fp})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
