"""Generate the DRAFT ``INDEX.yaml`` for the frozen ``scripts/*_preregistered*.py`` modules (ticket P1-09).

Read-only over scripts/ and docs/. Writes ONLY the path given by ``--out`` (default
``plan/drafts/P1-09/INDEX.yaml``); the owner reviews and commits it as ``research/preregistration/INDEX.yaml``
(CODEOWNERS-protected). Fingerprints are recomputed by import in a subprocess, freeze commits are found by
walking git history, freeze times are git committer times in UTC (never the modules' PREREGISTERED_AT).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import yaml

from firm.research import prereg as P

log = logging.getLogger(__name__)

# module stem -> (family, ledger_path, status, notes)
_META = {
    "allocation_forward_test_preregistered": ("allocation_forward_test", "docs/allocation_forward_test_trial_history.json", "RUN",
        "Forward-test state record (counts 0 trials). Re-frozen before deployment; PREREGISTERED_AT 2026-09-30T10:05:00Z predates its freeze commit."),
    "alt_premia_preregistered_bars": ("alt_premia", "docs/alt_premia_trial_history.json", "RUN",
        "Ledger note says combination=52 (stale; docs/combination_trial_history.json sums to 57)."),
    "combination_preregistered_bars": ("combination", "docs/combination_trial_history.json", "RUN",
        "Trial history records no fingerprint; fingerprint recomputed at the freeze commit. 57 ledgered trials."),
    "eodhd_s1_industry_momentum_preregistered_bars": ("S1", "docs/S1_trial_history.json", "RUN",
        "prior_trials placeholder 206 (each S-family module excluded itself differently)."),
    "eodhd_s2_breadth_overlay_preregistered_bars": ("S2", "docs/S2_trial_history.json", "RUN",
        "prior_trials placeholder 206."),
    "eodhd_s3_bond_commodity_trend_preregistered_bars": ("S3", "docs/S3_trial_history.json", "RUN",
        "prior_trials placeholder 205 (S1/S2/S4 use 206, s5 uses 207). The running total after S5 is 210 and appears in no file."),
    "eodhd_s4_52wk_high_preregistered_bars": ("S4", "docs/S4_trial_history.json", "RUN",
        "prior_trials placeholder 206."),
    "eodhd_s5_crypto_momentum_preregistered_bars": ("s5", "docs/s5_trial_history.json", "RUN",
        "prior_trials placeholder 207. docs/s5_trial_history.json has no family key."),
    "futures_trend_preregistered_bars": ("futures_trend", None, "SUPERSEDED",
        ("superseded-by core_v1 (DRAFT, never run, 0 trials). OD-13 signed 2026-10-03: file stays frozen and untouched; "
         "its market list only seeds universe_futures.yaml as a candidate list chosen without performance input. "
         "STATUS='DRAFT', N_CANDIDATES=2, DSR trials 3.")),
    "insider_cluster_preregistered_bars": ("insider_cluster", "docs/insider_cluster_trial_history.json", "RUN", ""),
    "pattern_ml_preregistered_bars": ("pattern_ml", "docs/pattern_ml_trial_history.json", "RUN",
        "Trial history records no fingerprint; recomputed at the freeze commit. n_trials 104 = folds x candidates (13 configs)."),
    "s2_forward_preregistered": ("s2_forward", None, "RUN",
        "No docs/*_trial_history.json: forward state lives in docs/s2_forward_snapshot.json (sealed, not read)."),
    "standalone_strategy_preregistered_bars": ("standalone_strategy", "docs/standalone_strategy_trial_history.json", "RUN", ""),
}


def build(repo: Path) -> dict:
    entries = []
    for mod in sorted((repo / "scripts").glob("*_preregistered*.py")):
        if mod.stem not in _META:
            raise SystemExit(f"unmapped frozen module {mod.name}: add it to _META")
        family, ledger, status, notes = _META[mod.stem]
        rel = mod.relative_to(repo).as_posix()
        fp = P.recompute_fingerprint(rel, repo)
        commit = P.find_freeze_commit(rel, repo)
        source = "freeze_commit"
        if ledger:
            recorded = P._ledger_fingerprints(repo / ledger)
            if recorded:
                source = "trial_history"
                if fp not in recorded:
                    raise SystemExit(f"{family}: HEAD fingerprint {fp} not in {ledger}")
        entries.append({
            "family": family, "kind": "frozen_module", "module_path": rel, "yaml_path": None,
            "fingerprint": fp, "freeze_commit": commit, "freeze_time_utc": P.freeze_time_utc(commit, repo),
            "ledger_path": ledger, "fingerprint_source": source if ledger or source == "freeze_commit" else "none",
            "status": status, "notes": notes,
        })
        log.info("indexed %s %s", family, fp[:12])
    entries.sort(key=lambda e: e["family"])
    return {"entries": entries}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(P.REPO_DIR))
    ap.add_argument("--out", default="plan/drafts/P1-09/INDEX.yaml")
    a = ap.parse_args()
    repo = Path(a.repo)
    doc = build(repo)
    header = ("# DRAFT for research/preregistration/INDEX.yaml (owner-committed, CODEOWNERS-protected; OD-06, OD-13).\n"
              "# Generated by scripts/build_prereg_index_draft.py. Times are git committer times in UTC.\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(header + yaml.safe_dump(doc, sort_keys=False, width=120))
    problems = P.verify_index(P.load_index(Path(a.out)), repo)
    print(json.dumps({"entries": len(doc["entries"]), "problems": problems}, indent=1))


if __name__ == "__main__":
    main()
