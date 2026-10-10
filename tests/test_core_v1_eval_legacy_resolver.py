"""core_v1 evaluation driver: var_sr_family legacy resolution (real docs/ trial-history files, synthetic ledgers) and the re-run bookkeeping.

The first real run died on ``legacy ledger rows ... are missing`` because the host ledger had never been backfilled from docs/. The
frozen docs/*_trial_history.json files are the true source of the legacy per-period Sharpes: the resolver reads them directly and only
cross-checks the ledger rows when they exist. Repo files only; no real data, no host ledger.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import core_v1_preregistered as pre
import run_core_v1_evaluation as E

from firm.research import legacy_adapters as LA

DOCS = ROOT / "docs"
GATES = yaml.safe_load((ROOT / "config" / "gates.yaml").read_text())
NAMES = list(GATES["var_sr_family"])


def _file_value(stem: str, trial: str) -> float:
    doc = json.loads((DOCS / f"{stem}_trial_history.json").read_text())
    (entry,) = doc["entries"]
    return float(entry["trial_daily_sharpes"][entry["trials"].index(trial)])


def _ledger_frame(docs_dir=DOCS) -> pd.DataFrame:
    rows = [dict(r.__dict__) for r in LA.iter_legacy_rows(docs_dir)]
    return pd.DataFrame(rows)


def test_every_var_sr_member_has_a_declared_source_and_the_mapping_files_exist():
    for name in NAMES:
        if name == "core_v1_grid":
            continue
        assert name in E.LEGACY_VAR_SR_SOURCES, name
        assert (ROOT / E.LEGACY_VAR_SR_SOURCES[name][0]).is_file(), name
    assert set(E.LEGACY_VAR_SR_SOURCES) == set(NAMES) - {"core_v1_grid"}


def test_resolver_on_the_real_docs_files_without_any_ledger_rows():
    out, all_legacy = E.legacy_family_sharpes(pd.DataFrame(columns=["mode"]), NAMES, docs_dir=DOCS)
    assert set(out) == set(NAMES) - {"core_v1_grid"}
    assert out["legacy_trend_standalone"] == _file_value("standalone_strategy", "trend")
    assert out["alt_premia_T2"] == _file_value("alt_premia", "T2_cross_asset_trend")
    assert out["alt_premia_C1"] == _file_value("alt_premia", "C1_btc_trend")
    s3 = ["bond_v1", "commodity_v1_primary", "commodity_v2_sensitivity", "combined_v1_primary", "combined_v2_sensitivity"]
    for k, trial in enumerate(s3, start=1):
        assert out[f"eodhd_s3_trial_{k}"] == _file_value("S3", trial)
    assert all(np.isfinite(v) and abs(v) < 0.5 for v in out.values())
    assert len(all_legacy) >= 40 and all(np.isfinite(all_legacy))


def test_resolver_with_a_backfilled_synthetic_ledger_agrees_with_the_files():
    led = _ledger_frame()
    out, _ = E.legacy_family_sharpes(led, NAMES, docs_dir=DOCS)
    out_docs, _ = E.legacy_family_sharpes(pd.DataFrame(columns=["mode"]), NAMES, docs_dir=DOCS)
    assert out == out_docs
    assert E.legacy_ledger_present(led, NAMES)


def test_ledger_row_that_disagrees_with_the_file_fails_closed():
    led = _ledger_frame()
    i = led.index[led["source_file"] == "docs/standalone_strategy_trial_history.json"][0]
    cfg = json.loads(json.dumps(led.at[i, "config"]))
    cfg["trial_daily_sharpes"][cfg["trials"].index("trend")] += 0.01
    led.at[i, "config"] = cfg
    with pytest.raises(E.StartupError, match="legacy_trend_standalone"):
        E.legacy_family_sharpes(led, NAMES, docs_dir=DOCS)


def test_ledger_without_legacy_rows_is_reported_not_fatal():
    assert not E.legacy_ledger_present(pd.DataFrame(columns=["mode", "source_file"]), NAMES)


def test_unknown_member_and_unresolvable_trial_name_fail_closed_naming_the_member(tmp_path):
    with pytest.raises(E.StartupError, match="no legacy source declared.*mystery"):
        E.legacy_family_sharpes(pd.DataFrame(columns=["mode"]), ["mystery"], docs_dir=DOCS)
    d = tmp_path / "docs"
    d.mkdir()
    (d / "alt_premia_trial_history.json").write_text(json.dumps(
        {"entries": [{"trials": ["V1"], "trial_daily_sharpes": [0.1]}]}))
    with pytest.raises(E.StartupError, match="alt_premia_T2"):
        E.legacy_family_sharpes(pd.DataFrame(columns=["mode"]), ["alt_premia_T2"], docs_dir=d)
    with pytest.raises(E.StartupError, match="legacy_trend_standalone"):
        E.legacy_family_sharpes(pd.DataFrame(columns=["mode"]), ["legacy_trend_standalone"], docs_dir=d)


# ---- prior attempts ----------------------------------------------------------------------------------------------------------------
def _row(tid, kind, mode="registered", family="core_v1", ts="2026-10-10T18:30:00+00:00"):
    return {"trial_id": tid, "family": family, "mode": mode, "config": {"kind": kind}, "created_at": ts, "n_variants": 1}


def test_detect_prior_attempts_counts_only_registered_grid_rows_of_the_family():
    led = pd.DataFrame([_row("a", "grid", ts="2026-10-10T18:30:00+00:00"), _row("b", "grid", ts="2026-10-10T18:31:00+00:00"),
                        _row("c", "robustness"), _row("d", "grid", mode="exploratory"), _row("e", "grid", family="core_v1_engine"),
                        _row("f", "grid", mode="legacy")])
    out = E.detect_prior_attempts(led, "core_v1", reason="r")
    assert out["count"] == 2 and out["trial_ids"] == ["a", "b"] and out["reason"] == "r"
    assert out["first_timestamp"] == "2026-10-10T18:30:00+00:00" and out["last_timestamp"] == "2026-10-10T18:31:00+00:00"
    assert E.detect_prior_attempts(led.iloc[0:0], "core_v1", reason="r")["count"] == 0


def test_dsr_keeps_gate_n_at_the_frozen_family_n_and_reports_the_duplicate_inflated_sensitivity():
    rng = np.random.default_rng(0)
    sel = pd.Series(rng.normal(0.0004, 0.01, 1500))
    grid = np.array([0.01, 0.02, 0.015, 0.03])
    legacy = {"a": 0.01, "b": 0.02}
    base = E.build_dsr(sel, grid, legacy, [0.01, 0.02], GATES, pre.FAMILY_N, n_raw=500)
    blk = E.build_dsr(sel, grid, legacy, [0.01, 0.02], GATES, pre.FAMILY_N, n_raw=500, n_prior_attempt=12)
    assert blk["n_gate"] == pre.FAMILY_N == 31 and blk["dsr"] == base["dsr"]
    assert blk["n_family_plus_prior_attempt"] == 43
    assert blk["dsr_family_plus_prior_attempt"] == E.dsr_or_none(blk["sharpe"], blk["n_obs"], blk["skew"], blk["kurt"], 43, blk["var_sr"])
    assert "dsr_family_plus_prior_attempt" not in base
