#!/usr/bin/env python3
"""Print the legacy trial census CSV (P0-05) to stdout.

Ledger-backed rows are read from the tracked ``docs/*_trial_history.json``
files (read-only, never modified); unledgered estimate rows are the
conservative upper estimates fixed in ``docs/legacy_trial_census_2026_10.md``.

    python scripts/seed_legacy_census.py > research/ledger/legacy_backfill.csv
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

COLUMNS = [
    "family", "sub_family", "description", "date_start", "date_end",
    "n_variants", "n_variants_alt_convention", "count_convention",
    "count_is_estimate", "holdout_touched", "trial_daily_sharpes_known",
    "sharpe_frequency", "sharpe_values_per_period", "conversion",
    "source_file", "source_entry_index", "evidence_ref", "notes",
]

SQRT252 = math.sqrt(252)
NO_CONV = "none"


def _fmt(values: list[float]) -> str:
    return ";".join(f"{v:.6g}" for v in values)


def _row(**kw: object) -> dict[str, object]:
    row = {c: "" for c in COLUMNS}
    row.update(kw)
    return row


def _load(name: str) -> dict:
    return json.loads((DOCS / name).read_text())


def _dates(entry: dict) -> tuple[str, str]:
    raw = str(entry.get("date") or entry.get("appended_at") or "")[:10]
    if "/" in raw:  # e.g. "2026-09-26/27"
        head, tail = raw.split("/")
        return head, head[:8] + tail.zfill(2)
    return raw, raw


def ledger_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    comb = _load("combination_trial_history.json")
    for i, e in enumerate(comb["entries"]):
        d0, d1 = _dates(e)
        sh = e.get("oos_sharpes")
        vals = [v / SQRT252 for v in sh.values()] if sh else []
        rows.append(_row(
            family="combination", sub_family="audit", description=e["source"],
            date_start=d0, date_end=d1, n_variants=e["n_trials"],
            n_variants_alt_convention=e["n_trials"],
            count_convention="full candidate grid incl baseline",
            count_is_estimate="false", holdout_touched="yes",
            trial_daily_sharpes_known="yes" if vals else "no",
            sharpe_frequency="annualised" if vals else "",
            sharpe_values_per_period=_fmt(vals),
            conversion="SR_d = SR_a/sqrt(252)" if vals else NO_CONV,
            source_file="docs/combination_trial_history.json",
            source_entry_index=i,
            evidence_ref=f"docs/combination_trial_history.json entries[{i}]",
            notes="oos_sharpes in the file are annualised; converted per-period here" if vals else "Sharpes not retained",
        ))

    pml = _load("pattern_ml_trial_history.json")
    for i, e in enumerate(pml["entries"]):
        d0, d1 = _dates(e)
        rows.append(_row(
            family="pattern_ml", sub_family="walk_forward", description=e["source"],
            date_start=d0, date_end=d1, n_variants=e["n_trials"],
            n_variants_alt_convention=e["n_candidates"],
            count_convention="folds x candidates (alt = candidate configs)",
            count_is_estimate="false", holdout_touched="yes",
            trial_daily_sharpes_known="no", sharpe_frequency="",
            sharpe_values_per_period="", conversion=NO_CONV,
            source_file="docs/pattern_ml_trial_history.json", source_entry_index=i,
            evidence_ref=f"docs/pattern_ml_trial_history.json entries[{i}]",
            notes=f"{e['n_folds']} folds x {e['n_candidates']} candidates; fold-4 test window reused",
        ))

    single = [
        ("standalone_strategy", "standalone_strategy_trial_history.json", "11 strategies standalone eval", "per strategy", "trial_daily_sharpes"),
        ("alt_premia", "alt_premia_trial_history.json", "7 candidates + benchmarks", "per trial", "trial_daily_sharpes"),
        ("insider_cluster", "insider_cluster_trial_history.json", "insider cluster variants", "per trial", "trial_daily_sharpes"),
        ("eodhd_s1", "S1_trial_history.json", "industry ETF momentum", "per trial", "trial_daily_sharpes"),
        ("eodhd_s2", "S2_trial_history.json", "breadth overlay on 60/40", "per trial", "trial_daily_sharpes_cash_excess_governing"),
        ("eodhd_s3", "S3_trial_history.json", "bond/commodity trend", "per trial", "trial_daily_sharpes"),
        ("eodhd_s4", "S4_trial_history.json", "52-week-high", "per trial", "trial_daily_sharpes"),
    ]
    for fam, fname, desc, conv, key in single:
        e = _load(fname)["entries"][0]
        d0, d1 = _dates(e)
        vals = list(e[key])
        note = "S2 uses the cash-excess governing series; bm2-excess series is descriptive only" if fam == "eodhd_s2" else ""
        rows.append(_row(
            family=fam, sub_family="", description=desc, date_start=d0, date_end=d1,
            n_variants=e["n_trials"], n_variants_alt_convention=e["n_trials"],
            count_convention=conv, count_is_estimate="false", holdout_touched="yes",
            trial_daily_sharpes_known="yes", sharpe_frequency="daily",
            sharpe_values_per_period=_fmt(vals), conversion=NO_CONV,
            source_file=f"docs/{fname}", source_entry_index=0,
            evidence_ref=f"docs/{fname} entries[0]", notes=note,
        ))

    s5 = _load("s5_trial_history.json")["entries"][0]
    d0, d1 = _dates(s5)
    rows.append(_row(
        family="eodhd_s5", sub_family="", description="crypto cross-sectional momentum",
        date_start=d0, date_end=d1, n_variants=len(s5["variant_names"]),
        n_variants_alt_convention=len(s5["variant_names"]),
        count_convention="per trial (len variant_names)", count_is_estimate="false",
        holdout_touched="yes", trial_daily_sharpes_known="yes", sharpe_frequency="daily",
        sharpe_values_per_period=_fmt(list(s5["trial_daily_sharpes"])), conversion=NO_CONV,
        source_file="docs/s5_trial_history.json", source_entry_index=0,
        evidence_ref="docs/s5_trial_history.json entries[0]",
        notes="file has no n_trials, no family key and no cumulative_trials; count = len(variant_names)",
    ))

    rows.append(_row(
        family="allocation_forward_test", sub_family="", description="forward test only",
        date_start="", date_end="", n_variants=0, n_variants_alt_convention=0,
        count_convention="forward test, not a backtest family", count_is_estimate="false",
        holdout_touched="no", trial_daily_sharpes_known="no", sharpe_frequency="",
        sharpe_values_per_period="", conversion=NO_CONV,
        source_file="docs/allocation_forward_test_trial_history.json", source_entry_index=0,
        evidence_ref="docs/allocation_forward_test_trial_history.json entries[0]",
        notes="zero-trial shape; loader adapters skip it explicitly",
    ))
    return rows


# (family, description, d0, d1, n, alt, convention, holdout, source_file, evidence_ref, method/notes)
ESTIMATES = [
    ("gann", "IC ablation 10 + followup 3 + swing event 8 + correct_cycles 24 natural + 20 price-derived + multiasset 10 + 42 + 1 (=118 read from archive scripts) + anniversary/squaring grids not read (27 upper estimate)",
     "2026-06-09", "2026-07-26", 145, 8, "config-level (alt = source plan's 8 experiments; lower bound)", "yes",
     "origin/research/gann-archive",
     "git show origin/research/gann-archive:scripts/gann_{ic,followup,swing_event,correct_cycles,multiasset,anniversary,squaring_event}_study.py; docs/gann_research_closeout.md",
     "118 = counted from grids in 5 archive scripts; 27 = upper estimate for anniversary/squaring scripts counted from CLI defaults only (3 cycles x 2 pivot orders x tolerance, 4 thresholds x 2 pivot orders, range projections). Owner must sign ~145 not 8 (OD-09)."),
    ("pre_ledger_ab", "regime-weights v1/v2/v2-soft + HMM on/off + rebalance band/fraction + max_positions + vol target + analyst-ratings/danelfin/percentile on 3 diagnostic windows",
     "2026-06-09", "2026-07-25", 20, "", "estimate", "yes",
     "docs/remediation_progress.md; docs/archive/strategy_regime_weights_calibration.md",
     "docs/remediation_progress.md; docs/archive/strategy_regime_weights_calibration.md",
     "estimate fixed by P0-05 ticket sketch; earliest tuning (6/9..7/25) is in no ledger; overlap with danelfin not de-duplicated"),
    ("strategy_construction", "13 strategies built + 38 defects fixed + 25-name universe chosen with hindsight",
     "2026-06-09", "2026-07-25", 39, "", "13 x 3 placeholder", "unknown (assume yes)",
     "docs/remediation_progress.md; docs/edge_search_verdict_2026_09.md",
     "docs/remediation_progress.md; docs/edge_search_verdict_2026_09.md",
     "13 strategies x 3 placeholder variants each; method fixed by ticket sketch; holdout status unknown, assumed touched"),
    ("pattern_pre_9_25", "ATR detection + quality thresholds + Part A 7 + Part B 6 + retrains + golden benchmark",
     "2026-09-09", "2026-09-24", 20, "", "estimate", "yes (train_pattern_ml.py:292 trailing holdout reused)",
     "docs/pattern_recognition_plan.md; docs/pattern_ml_final_verdict_2026_09.md; docs/pattern_ml_isolated_evaluation_2026_09.md",
     "docs/pattern_recognition_plan.md (git: first 2026-09-09)",
     "estimate fixed by ticket sketch: 7 + 6 enumerated parts plus thresholds/retrains/benchmark; entries on/after 9/25 are in the pattern_ml ledger"),
    ("danelfin", "ai_score + live_signals + best_stocks arm + market_percentile (extra beyond overlap with pre_ledger_ab)",
     "2026-07-31", "2026-08-16", 8, "", "estimate", "yes",
     "docs/remediation_progress.md; docs/danelfin_best_stocks_arm.md",
     "docs/danelfin_best_stocks_arm.md (git: 2026-07-31..2026-08-16)",
     "estimate fixed by ticket sketch; not de-duplicated against pre_ledger_ab"),
    ("llm_configs", "arm A/B + temperature + fallback chain changes (forward, not backtest)",
     "2026-07-29", "2026-09-08", 6, "", "estimate", "no (live-forward)",
     "docs/llm_ab_experiment_log.md", "docs/llm_ab_experiment_log.md (git: 2026-07-29..2026-09-08)",
     "estimate fixed by ticket sketch; counted although forward, because the DSR charge is for configurations tried"),
    ("sleeves_allocation", "sleeved-vs-blended A/B (~2) + allocation build configs (~8)",
     "2026-09-09", "2026-09-30", 10, "", "estimate", "yes",
     "docs/capital_sleeves_plan.md; docs/allocation_portfolio_backtest_2026_09.json",
     "docs/capital_sleeves_plan.md (git: 2026-09-09..2026-09-10); docs/allocation_portfolio_backtest_2026_09.json (2026-09-30)",
     "estimate fixed by ticket sketch: 2 + 8"),
    ("reruns_after_results", "S4 governing run 3 (f0fa1cb ba52b36) + S2 re-evaluation (e95649a 32eca01) + S3/S5 fixes + 9/28 combination re-run",
     "2026-09-28", "2026-10-01", 5, "", "estimate", "yes",
     "git log",
     "git: f0fa1cb 2026-10-01, ba52b36 2026-10-01, e95649a 2026-09-30, 32eca01 2026-09-30",
     "estimate fixed by ticket sketch; commit hashes verified to exist; re-runs after seeing results count as new trials"),
]


def estimate_rows() -> list[dict[str, object]]:
    rows = []
    for fam, desc, d0, d1, n, alt, conv, hold, src, ev, notes in ESTIMATES:
        rows.append(_row(
            family=fam, sub_family="", description=desc, date_start=d0, date_end=d1,
            n_variants=n, n_variants_alt_convention=alt, count_convention=conv,
            count_is_estimate="true", holdout_touched=hold,
            trial_daily_sharpes_known="no", sharpe_frequency="",
            sharpe_values_per_period="", conversion=NO_CONV,
            source_file=src, source_entry_index="", evidence_ref=ev, notes=notes,
        ))
    return rows


def main() -> None:
    w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    for row in ledger_rows() + estimate_rows():
        w.writerow(row)


if __name__ == "__main__":
    main()
