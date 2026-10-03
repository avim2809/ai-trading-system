# docs/ index

Start with **`PROJECT_CONTEXT.md`**: architecture, deployment, live config, and a
dated log of every significant change.

**Paths are load-bearing:**
- **Frozen pre-registrations** (`scripts/*_preregistered*.py`) name their ledgers
  and result files by path, and a changed file would break their audit trail.
- **Code and config comments** cite docs by path.

So files named there stay at the top level, even where they look like interim
output.

## Current state (read these first)

| Doc | What it is |
|---|---|
| `PROJECT_CONTEXT.md` | Full reference: architecture, services, config, change log |
| `edge_search_verdict_2026_09.md` | Why the 11-strategy system was replaced: nothing tested beats the benchmarks |
| `allocation_deploy_runbook.md` | How the Alpaca allocation portfolio was deployed, the exact config, and rollback |
| `allocation_forward_test_plan.md` | How the live allocation portfolio is judged (plain language) |
| `s2_forward_test_plan.md` | Shadow forward test of the S2 breadth overlay (running; paper ledger only) |
| `capital_sleeves_plan.md` | Per-strategy capital sleeves (pipeline mode; used by the IBKR/pipeline code path) |
| `REPO_MAP.md` | Verified map of the repo (packages, entry points, engines, data, frozen paths) and the credibility-plan path reconciliation table |
| `../PLAN.md` | Credibility plan (repo root): realistic expectations, gates, phases, ticket index; tickets in `../plan/tickets/`, owner decisions in `../plan/OWNER_DECISIONS.md` |

## Decisions and verdicts (settled, with evidence)

| Doc | Verdict |
|---|---|
| `eodhd_shortlist_verdict_2026_10.md` | EODHD shortlist S1-S5 (industry momentum, breadth overlay, bond/commodity trend, 52-week high, crypto momentum): all Tier C; S2 a near miss |
| `insider_cluster_verdict_2026_09.md` | SEC insider-purchase clusters: Tier C, no edge (median event loses; EODHD bad-bar issues documented) |
| `optimal_combination_fix_2026_09.md` | Signal-combination methods: FAIL (re-measured on the fixed engine, still FAIL) |
| `pattern_ml_final_verdict_2026_09.md` | Pattern-recognition ML layer: stays off |
| `pattern_ml_isolated_evaluation_2026_09.md` | Earlier isolated pattern-ML evaluation (pre-fix; kept, cited by code) |
| `formal_pbo_audit.md` | Six architecture changes, each failed the PBO/DSR gate |
| `portfolio_construction_diagnosis.md` | Confidence vs optimal combination (July; its basis was later refuted) |
| `ensemble_redundancy_audit_2026_09.md` | Ensemble redundancy audit (§2 corrected by the optimal-combination doc) |
| `gann_research_closeout.md` | Gann strategy: permanently disabled |
| `regime_ensemble_scoping.md` | Regime-ensemble scoping |

## Plans and feature docs

`edge_search_plan_2026_09.md` (the approved plan behind the verdict),
`pattern_recognition_plan.md`, `danelfin_best_stocks_arm.md` (decommissioned),
`investing_pro_integration.md`, `llm_ab_test_runbook.md`,
`llm_ab_experiment_log.md`, `remediation_progress.md` (long running log).

## Research and scoping (pending owner decisions)

| Doc | Status |
|---|---|
| `research_findings_beyond_equities_2026_09_30.md` | External research on options, FX, CFDs, futures and more, with corrections |
| `futures_data_vendor_comparison_2026_09.md`, `futures_trend_integration_sketch.md` | Futures trend: needs Norgate data plus 34–56 days of engineering; owner to decide |
| `longer_dataset_options.md` | Survivorship-free data vendors (reference) |
| `research_brief_eodhd_findings.md` | Owner's external shortlist of EODHD-testable ideas (not yet pre-registered) |

## Evaluation records (machine-readable; don't move)

| Family | Ledger (cumulative trials) | Results |
|---|---|---|
| Signal combination | `combination_trial_history.json` | `combination_evaluation_2026_09.json`, `combination_evaluation_2026_09_corrected_engine.json` |
| Pattern ML | `pattern_ml_trial_history.json` | `pattern_ml_workstream_d_run_log.json` |
| Standalone strategies | `standalone_strategy_trial_history.json` | `standalone_strategy_evaluation_2026_09.json` |
| Alternative premia | `alt_premia_trial_history.json` | `alt_premia_evaluation_2026_09.json`, `alt_premia_power_analysis_2026_09.json` |
| Insider clusters | `insider_cluster_trial_history.json` | `insider_cluster_evaluation_2026_09.json` (primary), `insider_cluster_recompute_2026_09.json` (independent recompute + post-hoc) |
| EODHD shortlist S1-S5 | `S1_trial_history.json`, `S2_trial_history.json`, `S3_trial_history.json`, S4/S5 per their pre-registrations | `eodhd_s{1..5}_evaluation_2026_10.json`; shared rules `eodhd_shortlist_protocol_2026_10.md`; S5 data gate `eodhd_s5_crypto_gate_2026_10.md` |
| Allocation forward test | `allocation_forward_test_trial_history.json` | `allocation_replay_2026_09.json` (reference simulation); `allocation_portfolio_backtest_2026_09.json` (first sanity backtest, superseded by the replay, kept because the frozen prereg cites it) |

## External review

`review/README.md` (repo root) is the entry point for a third-party reviewer. It covers:
- every study's pre-registration, freeze commit, harness, tests and artifacts;
- the UTC timeline;
- the SHA-256 manifest of all input data;
- the software environment;
- errata.

The exact input data are in a private bundle described in `review/DATA_BUNDLE.md`.

## Subfolders

| Folder | Contents |
|---|---|
| `prompts/` | Briefs for external research agents (non-equity instruments, EODHD-based ideas) |
| `archive/` | Superseded one-off docs: old RAG research, the LLM look-ahead audit, the regime-weights A/B, the July vendor-decision draft |
| `claude-memory/` | Repo mirror of the agent's persistent memory; keep in sync |
