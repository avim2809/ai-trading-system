# P1-09 owner actions

1. Review and commit `INDEX.yaml` as `research/preregistration/INDEX.yaml` and `TEMPLATE.yaml` as
   `research/preregistration/TEMPLATE.yaml` (both protected; the agent cannot write there). `tests/test_prereg.py` reads the real
   path if present, else these drafts. Regenerate with `python scripts/build_prereg_index_draft.py` if scripts/ history changes.
2. Commit `test_all_new_family_yaml_reference_current_gates_hash.py` to `tests/integrity/`.
3. OD-13 (signed 2026-10-03): the `futures_trend` entry is `SUPERSEDED`, notes
   "superseded-by core_v1 (DRAFT, never run, 0 trials)"; `scripts/futures_trend_preregistered_bars.py` is untouched.
4. Index records, does not fix: S-family prior_trials placeholders (206/206/205/206/207), running total 210 in no file, the stale
   alt_premia "combination=52" note, s5 history without family key.
5. Freeze commits are the earliest commit after which `bars_fingerprint()` is stable (found by recomputing across git history),
   times are git committer time in UTC. Several modules' `PREREGISTERED_AT` constants differ from these (e.g. insider_cluster
   17:15Z vs commit 16:57Z), as the host-clock warning predicted.
