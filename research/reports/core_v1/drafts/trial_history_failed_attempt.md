# DRAFT: append-only trial-history text for the failed first core_v1 evaluation attempt

Owner to review and, if adopted, append to the core_v1 trial history (docs/core_v1_trial_history.json is frozen for agents; nothing here edits it).

- **When:** 2026-10-10, 18:25-18:45 UTC (evaluation unit, chain-eval worktree at a76c9a5).
- **What ran:** the pre-registered 12-config grid at 1x cost, each config a registered `core_v1` ledger row (`config.kind = "grid"`), plus its `core_v1_engine` exploratory engine row. 12 registered grid rows now sit in the host ledger.
- **How it stopped:** `StartupError: legacy ledger rows for docs/standalone_strategy_trial_history.json are missing (var_sr_family member legacy_trend_standalone)`, raised by `legacy_family_sharpes` before ANY gate was computed. No DSR, PBO, CPCV, cost, stress, robustness or benchmark result existed; no tier was assigned; no results.json was written. The owner should confirm that nothing was decided from the logged grid Sharpes before the re-run.
- **Cause:** the host ledger had never been backfilled from docs/*_trial_history.json (an owner step), and the driver read the legacy Sharpes only from backfilled ledger rows. The file named in the message exists in docs/; the mapping was right, the ledger rows were absent.
- **Fix (batch21/eval-legacy-fix):** legacy Sharpes are read from the frozen docs files and cross-checked against ledger rows when present. No gate, grid, constant, prereg or charter changed.
- **Counting:** the re-run registers the same 12 configs again (the ledger does not reject duplicate configs). Gate N stays the frozen family N = 31 (grid counted once). The duplicate attempt is reported as a sensitivity: N = 31 + 12 = 43. Raw ledger count includes both attempts. results.json carries `prior_attempts` (count, trial ids, timestamps, reason).
- **Selection risk:** the first attempt's grid Sharpes existed in the ledger returns files before the re-run; the re-run is deterministic on the same data, so the selected config is the same as the first attempt would have chosen. No result-dependent choice was made between attempts.
