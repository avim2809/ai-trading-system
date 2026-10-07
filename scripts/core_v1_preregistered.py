"""FROZEN pre-registration module for the core_v1 research family (tickets P3-11 constants, P3-08 G-RESEARCH run).

RE-FREEZE (batch 20, 2026-10-06, before any real run). The previous freeze (fingerprint 7826fb03..., 0 real trials, never run) left three
gaps that made a Tier A unreachable by construction: ``min_active_fraction`` (gate 5) and ``vol_ewma_span`` / ``max_vol_scale`` /
``instrument_risk_cap_multiple`` (gate 6) were unfrozen, so gate 5 was ``insufficient`` and 3 of 27 gate-6 parameters were unassessed.
They are frozen below, each WITHOUT any performance input (the source of every value is written next to it). Nothing else changed: weights
scheme, seed, family-N semantics, universe, tau (charter), gates hash, grid and the annual gate-7 variant are exactly as before.

Written WITHOUT any backtest, any data read or any performance input. Mirrors the YAML draft
``plan/drafts/P3-11/core_v1_prereg_DRAFT.yaml`` (the owner commits the YAML under research/preregistration/; this module
records the frozen inputs that the YAML schema has no field for). Do not edit once frozen: a changed fingerprint is a new
pre-registration.

The charter does not exist at freeze time and is deliberately NOT hashed here. ``verify_charter()`` reads the committed charter
at run time and raises ``PreregError`` if it is missing or invalid; the driver (P3-11) and the evaluation (P3-08 step 1) call it
and ``verify_frozen_inputs()`` before doing anything else. Importing this module reads no data and runs nothing.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

STATUS = "FROZEN_PENDING_OWNER_APPROVAL"   # becomes effective when the owner commits the YAML + INDEX entry
FAMILY = "core_v1"
PREREGISTERED_AT = "2026-10-06T20:58:05Z"  # drafting time of the re-freeze, `date -u` (UTC ISO); the owner's approved_at_utc in the YAML is authoritative
PREREG_DRAFT = "plan/drafts/P3-11/core_v1_prereg_DRAFT.yaml"
CHARTER_PATH = "research/charters/core_v1.md"

_HERE_REPO = Path(__file__).resolve().parents[1]
# prereg.recompute_fingerprint imports temp copies of this file with cwd = repo root, so fall back to the cwd.
REPO_DIR = _HERE_REPO if (_HERE_REPO / "config" / "universe_etf.yaml").is_file() else Path.cwd()

# ---------------------------------------------------------------------------------------------------------------------
# Window and seal
# ---------------------------------------------------------------------------------------------------------------------
WINDOW = ("1994-02-02", "2026-09-30")      # earliest first_trade_date .. burned_through; an instrument enters after 256 days
ENTRY_GATE_DAYS = 256
POST_SEAL_DATA_DECLARATION = "No data after 2026-09-30 was examined."
IN_SAMPLE_DECLARATION = (
    "All data up to 2026-09-30 is in-sample (OD-04). Full-window estimation of the pooled scalars, FDM and IDM is in-sample by "
    "construction; CPCV in P3-08 re-estimates them on training groups only where possible and lists the rest as known leakage."
)

# ---------------------------------------------------------------------------------------------------------------------
# Data cleaning (OD-14) and its look-ahead disclosure
# ---------------------------------------------------------------------------------------------------------------------
CLEANING_VERSION = "v3"
CLEANING_LOOKAHEAD_DISCLOSURE = (
    "Full-window cleaning (v2 and v3) looks ahead up to 5 bars when classifying spikes (scripts/eodhd_clean.py). The P2-02 "
    "loader applies clean(raw[date <= asof]), so no bar after asof is used; within the window, a bar's cleaning status may "
    "depend on up to 5 later bars that are themselves before asof."
)

# ---------------------------------------------------------------------------------------------------------------------
# Grid (12 configs; fixed by rationale only, see the draft header). tau, gross cap etc. are NOT dimensions (OD-17).
# ---------------------------------------------------------------------------------------------------------------------
GRID_AXES = {
    "forecast_cap": [15.0, 20.0],
    "buffer_fraction": [0.05, 0.1, 0.2],
    "speed_subset": ["all", "drop_fastest"],
}
MAX_GRID_SIZE = 12
GRID = [dict(zip(GRID_AXES, vals)) for vals in itertools.product(*GRID_AXES.values())]
SPEED_SUBSETS = {
    "all": {"ewmac_fast_spans": [2, 4, 8, 16, 32, 64], "breakout_lookbacks": [20, 40, 80, 160, 320]},
    "drop_fastest": {"ewmac_fast_spans": [8, 16, 32, 64], "breakout_lookbacks": [40, 80, 160, 320]},
}
DEFAULT_CONFIG = {"forecast_cap": 20.0, "buffer_fraction": 0.1, "speed_subset": "all"}   # = gates values, inside the grid

# ---------------------------------------------------------------------------------------------------------------------
# CPCV, seed, selection rule
# ---------------------------------------------------------------------------------------------------------------------
SEED = 20261005                             # arbitrary fixed integer (drafting date); never changed, never searched
EMBARGO_PCT = 0.01                          # = the gates cpcv.embargo_pct
CPCV = {"n_groups": 10, "k_test_groups": 2, "n_paths": 9, "min_positive_paths": 7}
CPCV_SELECTION_RULE = (
    "in each of the 45 splits, re-run on the 8 purged and embargoed TRAIN groups: select the grid config with the highest "
    "training-group net Sharpe (argmax, 1x cost, ties broken by grid order); score the 2 test groups with that config"
)

# ---------------------------------------------------------------------------------------------------------------------
# Frozen input files (sha256 of the committed bytes) and the instrument-weight vector
# ---------------------------------------------------------------------------------------------------------------------
GATES_FILE = "config/gates.yaml"
FILE_HASHES = {
    "config/universe_etf.yaml": "379cb0d4611f46561b1125c72276803ea590ab3de136bcfb3d0ce6a9696d3514",
    "config/stress_periods.yaml": "9b165c652cc7616fc4f441f00350a12a42e9145db9a261f9ab6e2004e9496f20",
    "config/tax_il.yaml": "1badbc38c33a8321929d03d4bafe339b22600e5c312a026bfaa6153683585856",
    GATES_FILE: "bcecaec9ef44163596e27459779a124747d382c8068bf0607d4793df79f71540",
}
GATES_SHA256 = FILE_HASHES[GATES_FILE]

def _load_universe_doc() -> dict:
    """The frozen universe config (its sha256 is in FILE_HASHES; verify_frozen_inputs checks it)."""
    import yaml

    return yaml.safe_load((REPO_DIR / "config" / "universe_etf.yaml").read_text())


_UNIVERSE_DOC = _load_universe_doc()


def _load_benchmark_variants() -> tuple[str, str]:
    """Gate-7 benchmark variant and reported sensitivity, read from the frozen gates file (Amendment 1, 2026-10-06)."""
    import yaml

    doc = yaml.safe_load((REPO_DIR / GATES_FILE).read_text())
    b = next(v["benchmark"] for v in doc.values() if isinstance(v, dict) and isinstance(v.get("benchmark"), dict))
    if b.get("uses_higher_after_tax_sharpe_of_the_two") is not False:
        raise RuntimeError("gates file does not pin the gate-7 benchmark variant ex ante")
    return str(b["gate_variant"]), str(b["sensitivity"])


# gate 7 compares against the pinned variant only; the other variant is a reported sensitivity (never gates)
BENCHMARK_GATE_VARIANT, BENCHMARK_SENSITIVITY_VARIANT = _load_benchmark_variants()
UNIVERSE = [str(i["symbol"]) for i in _UNIVERSE_DOC["instruments"]]   # 14 ETFs, file order; no symbol literals in this module
ASSET_CLASSES = {str(c): [str(s) for s in m] for c, m in _UNIVERSE_DOC["asset_classes"].items()}   # 10 classes

# Handcrafted weights, ONE GROUP PER ASSET CLASS (owner decision 2026-10-05, plan/OWNER_DECISIONS.md addendum "P4-01 handcrafting
# tree"): equal weight across the asset classes of the universe config, equal within each class. Derived with
# firm.portfolio.weights.handcraft_weights (P4-01, merged); no data input. A different vector is a new pre-registration.
# (The first freeze used equal 1/14 because P4-01 was believed unmerged; that was a mistake and is replaced here.)
INSTRUMENT_WEIGHT_SCHEME = "handcrafted_one_group_per_asset_class"


def _handcrafted_weights() -> dict[str, float]:
    from firm.portfolio.weights import handcraft_weights   # pure function; no live import, no data

    w = handcraft_weights({c: {c: m} for c, m in ASSET_CLASSES.items()})
    return {s: float(w[s]) for s in UNIVERSE}


INSTRUMENT_WEIGHTS = _handcrafted_weights()


def instrument_weights_hash(weights: dict[str, float] | None = None) -> str:
    """sha256 of the weight vector in universe order, rounded to 12 dp so float noise cannot change it."""
    w = INSTRUMENT_WEIGHTS if weights is None else weights
    payload = json.dumps([[s, round(float(w[s]), 12)] for s in UNIVERSE])
    return hashlib.sha256(payload.encode()).hexdigest()


INSTRUMENT_WEIGHTS_SHA256 = instrument_weights_hash()

# ---------------------------------------------------------------------------------------------------------------------
# Gate-5 coverage floor and the P4-03 limits layer (frozen by the batch-20 re-freeze; no performance input; owner-confirm items)
# ---------------------------------------------------------------------------------------------------------------------
# MIN_ACTIVE_FRACTION (gate 5, the gates file g_research.stress.min_active_fraction, "frozen in the core_v1 prereg"). The stress suite
# scores an episode "low_coverage" (gate 5 insufficient) when active instruments / universe size at the episode start is below it
# (firm.validation.stress_suite.run_stress_suite). Value 0.5 = a simple majority of the frozen 14-ETF universe (7 of 14 must have passed the
# 256-day entry gate), so that an episode speaks for a diversified book and not for one or two markets. Structural cross-check (tests, from
# universe config metadata only): the two largest asset classes hold 3 + 3 = 6 < 7 instruments, so any 7 active ETFs span at least 3 of the
# 10 classes, the minimum for a 0.40 class risk cap to be feasible (ceil(1 / 0.40) = 3, config/risk.yaml). No other basis exists in the gates
# file, the charter or the plan: the value is a convention, OWNER-CONFIRM. A larger value is stricter (more episodes insufficient).
MIN_ACTIVE_FRACTION = 0.5

# VOL_EWMA_SPAN (gate 6; null in the gates file and in config/risk.yaml, "frozen before P3-11"; P4-03: "slow EWMA span of the realised returns
# of the forecast-sized portfolio"). 252 trading days = one year, the slow counterpart of the 35-day instrument-vol span (Carver's fast span,
# gates vol_span 35) and of the annualisation horizon; a fast value would duplicate vol_span and make the scalar chase noise. Conventional
# round number, no result input, OWNER-CONFIRM (config/risk.yaml cannot carry it: tests/test_risk_limits.py pins it to null until tau is set).
VOL_EWMA_SPAN = 252
# MAX_VOL_SCALE and INSTRUMENT_RISK_CAP_MULTIPLE: the gates robustness_parameters values (1.5 and 2.0 = the source plan's limits, also
# config/risk.yaml max_vol_scale / max_instrument_risk_mult). Read from the frozen gates file, never typed.
# CLASS_RISK_CAP_APPLIED: the 0.40 class risk cap of config/risk.yaml is NOT applied in the evaluation sizing path. It is not one of the 27
# gate-6 parameters, so applying it would add a numeric core parameter that is never perturbed (gate 6: "no exemptions"). 1.0 disables it.
CLASS_RISK_CAP_APPLIED = 1.0
LIMITS_LAYER_RULES = {
    "vol_scale": "clip(tau / sigma_ewma, upper=max_vol_scale) multiplies the capital used for N and the buffer width (w1 = s * raw); NaN "
                 "(fewer than vol_ewma_span observations) = 1.0; sigma <= 0 gives 0",
    "sigma_ewma": "sqrt(256 * EWMA_span(rp^2)), adjust=True, rp_t = sum_i raw_w[t-1,i] * r[t,i] with raw_w the unscaled unbuffered uncapped "
                  "forecast-sized weights (f/10*idm*w_i*tau/vol_i); series starts at the first lagged weight; 256 = the sizing's annualisation",
    "covariance": "zero-mean bias-corrected EWMA covariance, span = vol_ewma_span, x256, NaN return = 0, row t uses returns up to t",
    "caps": "firm.risk.limits.apply_caps after the buffer step: long-only, instrument risk contribution <= instrument_risk_cap_multiple x "
            "handcrafted weight, class cap disabled, gross cap = gross_cap (scaled by gross_cap / 1.0 so a +25% perturbation above 1.0 works); "
            "infeasible caps are skipped as P4-03 documents; the per-day call uses apply_caps(solver='closed_form' (exact quadratic root of the "
            "share equation), max_sweeps=100, tol=1e-6, on_nonconvergence='return' (last down-scaled iterate kept, counted in the diagnostics)), "
            "because the default 50-sweep bisection raises on about a quarter of random 14-instrument cases and costs about 0.2 s per call",
    "scope": "evaluation sizing path only (scripts/run_core_v1_evaluation.py); the P3-11 constants driver and the IDM sub-systems stay unlimited",
}


def _load_limit_values() -> tuple[float, float]:
    """max_vol_scale and instrument_risk_cap_multiple from the frozen gates file (robustness_parameters)."""
    import yaml

    rp = yaml.safe_load((REPO_DIR / GATES_FILE).read_text())["robustness_parameters"]
    return float(rp["max_vol_scale"]["value"]), float(rp["instrument_risk_cap_multiple"]["value"])


MAX_VOL_SCALE, INSTRUMENT_RISK_CAP_MULTIPLE = _load_limit_values()
FROZEN_LIMITS = {"vol_ewma_span": VOL_EWMA_SPAN}   # what firm.research.core_v1_pipeline.params_from_gates(frozen=...) needs

# ---------------------------------------------------------------------------------------------------------------------
# Trial counts. Two different counts, never mixed (the frozen gates file's `n_rule` and `n_counts` are authoritative):
#  * FAMILY N (N_gate, what the DSR deflates for in a family-N verdict): legacy trend/carry trials (standalone trend 1,
#    alt_premia 10, S3 bond/commodity trend 5, S5 crypto momentum 3, futures_trend draft 0 = 19) PLUS the core_v1 GRID of up to
#    12 = 31 (= n_counts.family_provisional; the owner still has to confirm the membership list).
#  * RAW count: every ledger row. The diagnostic rows below (constant_estimation, robustness, cost_stress, stress_suite,
#    benchmark, comparison) are NOT family members: they count toward the RAW count only, never toward the family N.
# ---------------------------------------------------------------------------------------------------------------------
FAMILY_N_LEGACY = 1 + 10 + 5 + 3 + 0
FAMILY_N_GRID_MAX = MAX_GRID_SIZE
FAMILY_N = FAMILY_N_LEGACY + FAMILY_N_GRID_MAX                       # 31
FAMILY_N_ROW_KINDS = ("grid",)                                       # the only core_v1 ledger rows that enter the family N

# Expected DIAGNOSTIC ledger rows per kind (config['kind']); RAW count only (OD-09), not family N. Grid rows (12) are separate.
N_RULES = 6 + 5                              # six EWMAC speeds + five breakout lookbacks
# robustness: one parameter at a time, both directions; each list element is its own parameter (assumption, see report).
# After the batch-20 re-freeze ALL 27 are perturbable (vol_ewma_span, max_vol_scale, instrument_risk_cap_multiple are wired into the sizing path).
N_ROBUSTNESS_PARAMS = 16 + 6 + 5             # 16 scalar entries + 6 EWMAC spans + 5 breakout lookbacks of robustness_parameters
EXPECTED_LEDGER_ROWS = {
    "constant_estimation": N_RULES + len(UNIVERSE) + len(UNIVERSE) + 1,   # 11 scalars + 14 speed filters + 14 FDM sets + 1 IDM = 40
    "robustness": 2 * N_ROBUSTNESS_PARAMS,                                # 54
    "cost_stress": 2,                                                     # 2x and 3x of the selected config (1x is the grid run)
    "stress_suite": 10,                                                   # one per episode in the stress-periods file
    "benchmark": 3,                                                       # BM2 annual (gate), BM2 monthly (sensitivity), core_only_100
    "comparison": 1,                                                      # flat-bps comparison row (never gates)
}
EXPECTED_GRID_ROWS = len(GRID)
DIAGNOSTIC_ROWS_RAW_ONLY = sum(EXPECTED_LEDGER_ROWS.values())        # 110: added to the raw count, never to FAMILY_N
NEW_RAW_ROWS = DIAGNOSTIC_ROWS_RAW_ONLY + EXPECTED_GRID_ROWS         # 122 raw rows from core_v1; 12 of them are family rows


class PreregError(RuntimeError):
    """A frozen input differs from the committed file, or the charter is missing or invalid. STOP, never degrade."""


def bars_fingerprint() -> str:
    payload = json.dumps(
        {"FAMILY": FAMILY, "PREREGISTERED_AT": PREREGISTERED_AT, "WINDOW": WINDOW, "ENTRY_GATE_DAYS": ENTRY_GATE_DAYS,
         "POST_SEAL_DATA_DECLARATION": POST_SEAL_DATA_DECLARATION, "CLEANING_VERSION": CLEANING_VERSION,
         "CLEANING_LOOKAHEAD_DISCLOSURE": CLEANING_LOOKAHEAD_DISCLOSURE, "GRID": GRID, "SPEED_SUBSETS": SPEED_SUBSETS,
         "MAX_GRID_SIZE": MAX_GRID_SIZE, "SEED": SEED, "EMBARGO_PCT": EMBARGO_PCT, "CPCV": CPCV,
         "CPCV_SELECTION_RULE": CPCV_SELECTION_RULE, "FILE_HASHES": FILE_HASHES, "UNIVERSE": UNIVERSE,
         "INSTRUMENT_WEIGHT_SCHEME": INSTRUMENT_WEIGHT_SCHEME, "INSTRUMENT_WEIGHTS_SHA256": INSTRUMENT_WEIGHTS_SHA256,
         "EXPECTED_LEDGER_ROWS": EXPECTED_LEDGER_ROWS, "FAMILY_N": FAMILY_N,
         "FAMILY_N_ROW_KINDS": FAMILY_N_ROW_KINDS, "ASSET_CLASSES": ASSET_CLASSES, "CHARTER_PATH": CHARTER_PATH,
         "BENCHMARK_GATE_VARIANT": BENCHMARK_GATE_VARIANT, "BENCHMARK_SENSITIVITY_VARIANT": BENCHMARK_SENSITIVITY_VARIANT,
         "MIN_ACTIVE_FRACTION": MIN_ACTIVE_FRACTION, "VOL_EWMA_SPAN": VOL_EWMA_SPAN, "MAX_VOL_SCALE": MAX_VOL_SCALE,
         "INSTRUMENT_RISK_CAP_MULTIPLE": INSTRUMENT_RISK_CAP_MULTIPLE, "CLASS_RISK_CAP_APPLIED": CLASS_RISK_CAP_APPLIED,
         "LIMITS_LAYER_RULES": LIMITS_LAYER_RULES, "N_ROBUSTNESS_PARAMS": N_ROBUSTNESS_PARAMS},
        sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def file_sha256(rel: str, repo_dir: Path | None = None) -> str:
    return hashlib.sha256((Path(repo_dir or REPO_DIR) / rel).read_bytes()).hexdigest()


def verify_frozen_inputs(repo_dir: Path | None = None) -> None:
    """Raise PreregError unless every frozen config file still has the recorded sha256. Reads config files only."""
    bad = []
    for rel, want in FILE_HASHES.items():
        try:
            got = file_sha256(rel, repo_dir)
        except OSError as exc:
            bad.append(f"{rel}: unreadable ({exc})")
            continue
        if got != want:
            bad.append(f"{rel}: sha256 {got} != frozen {want}")
    if instrument_weights_hash() != INSTRUMENT_WEIGHTS_SHA256:
        bad.append("instrument weight vector hash mismatch")
    if bad:
        raise PreregError("frozen inputs changed: " + "; ".join(bad))


def _risk_yaml_mismatches(repo_dir: Path | None = None) -> list[str]:
    """config/risk.yaml is not hashed (the owner may later fill tau / vol_ewma_span there); its limit values must still agree with this module."""
    import yaml

    try:
        cfg = yaml.safe_load((Path(repo_dir or REPO_DIR) / "config" / "risk.yaml").read_text()) or {}
    except OSError as exc:
        return [f"config/risk.yaml: unreadable ({exc})"]
    bad = []
    if cfg.get("max_vol_scale") != MAX_VOL_SCALE:
        bad.append(f"risk.yaml max_vol_scale {cfg.get('max_vol_scale')!r} != frozen {MAX_VOL_SCALE}")
    if cfg.get("max_instrument_risk_mult") != INSTRUMENT_RISK_CAP_MULTIPLE:
        bad.append(f"risk.yaml max_instrument_risk_mult {cfg.get('max_instrument_risk_mult')!r} != frozen {INSTRUMENT_RISK_CAP_MULTIPLE}")
    if cfg.get("vol_ewma_span") not in (None, VOL_EWMA_SPAN):
        bad.append(f"risk.yaml vol_ewma_span {cfg.get('vol_ewma_span')!r} != frozen {VOL_EWMA_SPAN}")
    return bad


def verify_charter(repo_dir: Path | None = None, *, ledger=None) -> dict:
    """Read the committed charter at run time; raise PreregError if missing or invalid. Returns its identifying facts.

    Uses firm.research.charter in strict mode (approval fields must be filled) with the charter-before-ledger ordering check
    (``ledger`` may be injected; default reads the host ledger). Also requires family == core_v1 and a gates hash equal to
    this module's. The charter's own hash and ``approved_commit`` are returned for recording, never hard-coded here.
    """
    from firm.research import charter as C   # lazy: firm.research must not be imported by live modules

    repo = Path(repo_dir or REPO_DIR)
    path = repo / CHARTER_PATH
    if not path.is_file():
        raise PreregError(f"charter missing: {CHARTER_PATH} (it must be written and committed by the owner first)")
    try:
        problems = C.validate_charter_file(path, repo_dir=repo, ledger=ledger, strict=True)
    except OSError as exc:
        raise PreregError(f"charter invalid: cannot validate ({exc})") from exc
    try:
        fm, _ = C.parse_charter(path.read_text())
    except C.CharterError as exc:
        raise PreregError(f"charter unparseable: {exc}") from exc
    if fm.get("family") != FAMILY:
        problems.append(f"charter family {fm.get('family')!r} != {FAMILY!r}")
    if fm.get("gates_yaml_sha256") != GATES_SHA256:
        problems.append("charter gates_yaml_sha256 differs from the gates hash frozen in this module")
    if problems:
        raise PreregError("charter invalid: " + "; ".join(problems))
    return {"path": CHARTER_PATH, "sha256": file_sha256(CHARTER_PATH, repo), "approved_commit": fm["approved_commit"],
            "tau": fm["tau"]}


def verify_before_run(repo_dir: Path | None = None, *, ledger=None) -> dict:
    """Single entry point for the drivers: frozen inputs then charter. Raises PreregError; no data is touched."""
    verify_frozen_inputs(repo_dir)
    bad = _risk_yaml_mismatches(repo_dir)
    if bad:
        raise PreregError("frozen limits changed: " + "; ".join(bad))
    return verify_charter(repo_dir, ledger=ledger)


if __name__ == "__main__":
    print("FROZEN fingerprint:", bars_fingerprint())
