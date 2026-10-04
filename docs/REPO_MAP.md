# REPO_MAP

Ticket P0-01 deliverable. Drafted read-only on 2026-10-02 from main @ c2bd5cd; verified and corrected on 2026-10-04
against `p0/foundation` @ b4580c9 (counts, line references and `exists`/`extend` paths re-checked with `test -e` and grep).
Purpose: a verified map of what exists in this repo, and a table that maps every path the
credibility plan proposes to the real path. Anything marked "(new)" does not exist today.

Status vocabulary used in the reconciliation table:

- **exists**: the path is real and the plan can use it as is.
- **extend**: the thing exists in some form; the ticket adds to it or wraps it.
- **new**: nothing exists; the ticket creates it (path given is the reconciled one).
- **n/a**: the source path does not apply to this repo (explained in the note).

## 0. Ground rules that shape every mapping

1. **One package: `src/firm/`.** `pyproject.toml` has `[tool.setuptools.packages.find] where = ["src"]`
   and only `src/firm` exists. Top-level `src/validation`, `src/research`, `src/costs` etc. from the
   source plan would need a pyproject change and a reinstall of the live `.venv`. Not allowed.
   All new code goes under `src/firm/<subpackage>/`.
2. **Two live services import from this checkout** (`/local/store/git/ai-trading-system`, branch main):
   IBKR `:8000` (`config/live.yaml`, blended 11-strategy pipeline) and Alpaca `:8001`
   (`config/live_alpaca.yaml`, allocation mode). Both `Restart=always`. A merge to main that touches a
   live-imported module is effectively a deploy. Research agents build in a separate clone,
   `/local/store/research/ai-trading-system`, with worktrees at `<clone>/.claude/worktrees/<ticket>` (PLAN.md section 8; a worktree
   of the live checkout would share its refs). In the live checkout `.claude/worktrees/` exists and is empty and `git worktree list`
   shows only main. Never edit the live checkout.
3. **Frozen paths (never move, rename or edit):** `scripts/*_preregistered*.py`, `docs/*evaluation*.json`,
   `docs/allocation_portfolio_backtest_2026_09.json`, `docs/allocation_replay_2026_09.json`,
   `scripts/eodhd_clean.py`. `docs/*_trial_history.json` are never moved or renamed and are append-only
   (old entries immutable). Source: `docs/README.md` ("Paths are load-bearing").
4. **Tests are flat** (169 `test_*.py` files plus `conftest.py`, `pattern_fixtures.py` and `__init__.py`; no subdirectories today).
5. **Host clock is Asia/Jerusalem (UTC+3).** Take UTC stamps from `date -u` or git, never from logs.
6. `runs/` and `data_alpaca/` are gitignored. `data/` is ignored only for specific subpaths (`cache/`, `research/`, `logs/`, `models/`, `vectordb/`, plus named state files), so any new `data/<dir>` (`forward_monitors/`, `holdout/`) needs an explicit `.gitignore` entry (P0-06). `runs/` holds only 2026-09-25..28 runs.

## 1. Repository top level

| Path | What it is |
|---|---|
| `src/firm/` | The only Python package (see section 2) |
| `scripts/` | 72 entries: CLIs, frozen pre-registrations, evaluation harnesses, ops shell scripts |
| `tests/` | Flat pytest suite (about 3400 tests); `conftest.py` isolates `FIRM_EXECUTION_AUDIT` per test |
| `config/` | `settings.yaml` (backtest defaults, `strategies: []` = all registered), `live.yaml` (IBKR), `live_alpaca.yaml` (Alpaca), `live_alpaca_allocation.example.yaml`, `llm.yaml`, `llm_ab_llm.yaml`, `llm_ab_quant.yaml`, `experiments/`, `strategies/cross_sectional_momentum.yaml` |
| `docs/` | 63 entries (62 plus this file): `PROJECT_CONTEXT.md`, verdicts, evaluation JSONs, trial histories, plans, `claude-memory/`, `archive/`, `prompts/`. Indexed by `docs/README.md` |
| `deploy/` | systemd units and nginx config (section 9) |
| `review/` | Third-party review package (section 8): `README.md`, `ARTIFACT_INDEX.json`, `DATA_BUNDLE.md`, `ENVIRONMENT.json`, `TIMELINE.md`, `data/`, `studies/`, `requirements-lock.txt`, `session_scratch_scripts/` |
| `data/` | Live runtime state for IBKR (`live_state.db`, `kill_switch_state.json`, `execution_audit.jsonl`, `cache/`, `models/`, `vectordb/`, `logs/`) and `data/research/{eodhd,fred,insider,s2_forward}` (licensed vendor data; gitignored). Contains post-2026-09-30 data |
| `data_alpaca/` | Runtime state for the Alpaca instance (gitignored) |
| `runs/` | Gitignored `RunRegistry` output (about 38 dirs, 2026-09-25..28 only) |
| `frontend/` | React UI; API contract in `frontend/src/api/{types,client}.ts` |
| `.claude/` | Live checkout: only `settings.local.json` (gitignored) and an empty `worktrees/`. The research clone has no `.claude/` until its first worktree is created (`.claude/worktrees/<ID>`); no tracked file lives there yet |
| `.github/workflows/ci.yml` | The only workflow (section 9) |
| `AGENTS.md` (23 lines), `CLAUDE.md` (70 lines) | Agent instructions; they have drifted apart. `CLAUDE.md` holds the only "Rules of thumb" block |
| `.cursor/rules/*.mdc` | `frontend`, `ibkr-integration`, `live-trading`, `logging`, `project-context`, `strategies` |
| Absent today (as of b4580c9; `PLAN.md` at repo root, `plan/OWNER_DECISIONS.md` and `plan/tickets/` (56 files) exist and are tracked since fc14f79) | `research/`, `.github/CODEOWNERS`, `.claude/settings.json`, `.claude/hooks/`, `tests/integrity/`, `config/research_freeze.yaml`, `data/holdout/`, `docs/HOLDOUT_POLICY.md`, `docs/DEPRECATIONS.md`, `docs/LLM_POLICY.md` (since 2026-10-04 `p0/foundation` adds `.github/CODEOWNERS`, `.claude/settings.json`, `tests/integrity/`, `config/research_freeze.yaml`, `docs/HOLDOUT_POLICY.md`, `docs/GUARDRAIL_REDTEAM.md` and `src/firm/research/`; still absent: `research/`, `.claude/hooks/` (hooks are templates under `deploy/claude-research-hooks/`), `data/holdout/`, `docs/DEPRECATIONS.md`, `docs/LLM_POLICY.md`) |

## 2. Package map: `src/firm/`

| Subpackage / module | Contents (verified) | Live-imported? |
|---|---|---|
| `config.py`, `logging_setup.py`, `time_utils.py`, `runtime.py` | Settings; logging setup (rotating logs, wired via `firm-api` entrypoint); `runtime.py` = build helpers and loaders (below) | Yes (`runtime` imported at top of `live/engine.py`) |
| `contracts/models.py` | Shared pydantic/dataclass models | Yes |
| `strategies/` | `base.py` (`BaseStrategy`, `Strategy.generate(pit_view)`), `registry.py` (42 lines: `register`, `get`, `list_strategies`; no status metadata), 13 core strategy modules plus `danelfin_*` (4), `insider_cluster.py`, `investing_analyst_ratings.py`. `__init__.py` imports every module except `insider_cluster` (registered only when imported directly, e.g. by `tests/test_insider_cluster.py`); `investing_analyst_ratings` is imported and always registered, so it appears in every `list_strategies()` fallback (P0-03 must warn-and-skip it) | Yes |
| `agents/` | Pipeline: `analysts/` (HRP signal combination `hrp_signal_weights`, `combine_signals_hrp` in `analysts/__init__.py`), `research/` (`_combine.py`, `_circuit_breaker.py`), `risk.py` (`RiskAgent`, `_drawdown_breaker`), `trader.py`, `execution.py`, `_liquidity.py` (`market_impact_pct`), `_factor_risk.py`, `orchestrator.py`, `blackboard.py`, `memory.py`, `llm/` (`*_llm.py`, `base_llm_agent.py`, `news_anonymizer.py`) | Yes |
| `backtest/` | backtrader-based: `engine.py` (`BacktestEngine`), `run.py` (`execute_backtest`, `build_equity_data`), `firm_strategy.py`, `datafeeds.py`, `commissions.py`, `sizers.py`, `analyzers.py` | Partly (API job path). P1-12 adds `_capture_state.py` (stdlib only), a deliberately live-imported module that must be on the isolation test's ALLOW list |
| `data/` | `pit_store.py` (`PointInTimeDataStore`), `cache.py` (`ParquetCache`), `universe.py` (`UniverseResolver`), `schemas.py`, `synthetic.py`, `fundamentals_cache.py`, `sentiment_cache.py`, `insider_transactions.py`, `danelfin_market_percentile.py`, `investing/`, `providers/` | Yes |
| `data/providers/` | `base.py` (`DataProvider`), `fallback.py` (`FallbackProvider`), `alpaca.py`, `alphavantage.py`, `danelfin.py`, `edgar.py`, `finnhub.py`, `fmp.py`, `fred.py`, `ibkr.py`, `massive.py`, `tiingo.py`, `twelvedata.py`, `_rest.py`, `constants.py`, `sentiment_lexicon.py` | Yes |
| `brokers/` | `base.py` (`OrderRequest`, `BrokerPosition`), `ibkr.py` (stock-only contracts), `alpaca.py` | Yes |
| `live/` | `engine.py` (`LiveTradingEngine`, about 4250 lines), `scheduler.py`, `provider_utils.py` (`resolve_live_startup`), `execution_safety.py`, `capital_gate.py`, `approval.py`, `notifications.py`, `portfolio_sync.py`, `order_reconciliation.py`, `sleeve_reconciliation.py`, `planning_cycle.py`, `news_guard.py`, `pipeline_warmup.py`, `state_store.py`, `trade_history.py`, `data_feed.py`, `pattern_*`, `best_stocks_*`, `capital_reallocation*.py`, others | Yes (the services themselves) |
| `allocation/` | `allocator.py` (`Allocator.plan`, `AllocationPlan`, `build_allocator`), `sleeves.py` (`Sleeve` ABC, `StaticSleeve`, `SLEEVE_REGISTRY`, `_resolve_sleeve_class`, `build_sleeves`), `btc_trend.py` (`BtcTrendSleeve`), `calendar.py` (NYSE calendar) | Yes |
| `portfolio/` | `optimizer.py`, `state.py`, `attribution.py` only | Yes |
| `eval/` | `overfitting.py`, `robustness.py`, `metrics.py`, `tca.py`, `reports.py`, `tearsheet.py`, `plots.py`, `classification.py`, `rag_eval.py` | Yes (`robustness`, `reports`) |
| `experiments/` | `registry.py` (`RunRegistry` writing `runs/`), `runner.py` (`ExperimentRunner.run`, `run_walk_forward`, `aggregate_walk_forward`) | API-lazy |
| `patterns/` | Chart-pattern detectors, `significance.py` (iid null, BH), `scanner.py`, `ml/` (`purged_cv.py`, `sample_weights.py`, `xgb_*`, `cnn_validator.py`, `ppo_sizer.py`, others) | Yes (IBKR `pattern_recognition` strategy) |
| `regime/` | `detector.py`, `ensemble.py`, `features.py`, `model.py` | Yes |
| `llm/`, `rag/` | Provider layer, cache, compression; RAG store/retriever/ingestors | Yes |
| `api/` | `app.py` (`run`, FastAPI), `jobs.py`, `schemas.py`, `serializers.py`, `routers/{agents,decisions,live,llm,logs,meta,patterns,runs,system}.py` | Yes |
| `scripts/` + `scripts_entry.py` | Console entry points `fetch-data`, `run-backtest` | No |
| `signals/` | `vol.py` (P3-01: `ewma_vol` blended EWMA vol, `TRADING_DAYS_PER_YEAR=256`); `ewmac.py` (P3-02: `ewmac_raw`, `ewmac_forecast`, `estimate_pooled_scalar`, `select_speeds`; ETF path keeps per-rule forecasts signed, scalars and speed filter on real data only in P3-11); `breakout.py` (P3-03: `breakout_raw`, `breakout_forecast`; NOT related to `strategies/volatility_breakout.py`, the single-stock ATR breakout); research-only, not imported by live modules | No |
| Not present | `validation/`, `research/`, `costs/`, `risk/`, `monitoring/`, `lifecycle/`, `reporting/` | n/a |

Console scripts (`pyproject.toml`): `fetch-data` = `firm.scripts.fetch_data:main`,
`run-backtest` = `firm.scripts.run_backtest:main`, `firm-api` = `firm.api.app:run`.

## 3. Backtest engine and every launch path

Engine: backtrader (`src/firm/backtest/engine.py`), whole-share, equity-centric, long/short for equities
only; costs = flat commission 5 bp + spread 2 bp + slippage 5 bp (`config/settings.yaml`), plus sqrt-law
market impact (`market_impact_coefficient` 0.005 x sqrt(notional/ADV), via `agents/_liquidity.market_impact_pct`,
applied in `backtest/firm_strategy.py`) and a 0.3%/yr short-borrow fee (`short_borrow_annual_pct`); no stress
multiplier, no per-share/tiered commission, no roll cost. This is the "legacy convention" that G-RESEARCH 4
reports for comparison.

| Launch path | Location | Notes |
|---|---|---|
| `execute_backtest(config)` | `src/firm/backtest/run.py:37` | Main programmatic entry; callers include `experiments/runner.py`, `api/jobs.py`, `scripts/run_standalone_strategy_evaluation.py`, `scripts/run_combination_evaluation.py`, `scripts/calibrate_*.py` |
| `run_backtest_from_config(...)` | `src/firm/runtime.py:396` | Direct `BacktestEngine` construction; mostly dead path |
| `BacktestEngine(` direct sites | `src/firm/backtest/run.py`, `src/firm/scripts/run_backtest.py`, `runtime.py` | |
| `run-backtest` CLI | `src/firm/scripts/run_backtest.py` (also `scripts/run_backtest.py`) | |
| `ExperimentRunner.run` / `run_walk_forward` / `aggregate_walk_forward` | `src/firm/experiments/runner.py:48/196/430` | Records failed status on exception and re-raises (the pattern to copy for a ledger) |
| API job path | `src/firm/api/jobs.py` (`run_walk_forward_sync`), `routers/runs.py` | UI/API backtests |
| Walk-forward audits | `scripts/run_walk_forward_pbo_audit.py`, `scripts/run_pbo_trial_audit.py`, `scripts/validate_pattern_cnn_walkforward.py`, `scripts/validate_pattern_ml_workstream_d.py` | Go through `ExperimentRunner` |
| Vectorised research simulators | `scripts/run_alt_premia_evaluation.py` (`simulate`, long-only), `scripts/run_eodhd_s1..s5_evaluation.py`, `scripts/allocation_replay.py` | Do not use backtrader. Flat bps costs from each frozen prereg |
| Portfolio-level replay | `scripts/allocation_replay.py:43 replay(...)`, `scripts/allocation_portfolio_backtest.py` | Uses the real `Allocator`/sleeve classes |
| Does not exist | A signed, leveraged, multiplier-aware vector engine | New: `src/firm/backtest/vector_engine.py` (P3-09) |

## 4. Strategy registry

`src/firm/strategies/registry.py`: `_REGISTRY` dict, `register(name)`, `get(name)` (KeyError if unknown),
`list_strategies()`. No lifecycle status. Consumers that fall back to "all registered" when the strategy
list is empty (relevant to P0-03): `runtime.py:58` (`_build_categorized_strategies`, whose per-strategy
`try/except` logs and skips), `live/engine.py:210` (`__init__`), `:767-769` (`_all_strategy_names` helper), `:836` (`update_strategies`), `live/pipeline_warmup.py:95`,
`api/routers/live.py:1452`, `api/routers/meta.py:117`, and `config/settings.yaml` (`strategies: []`).
`tests/test_strategies.py::TestRegistry` asserts the full name list (including gann, ml_prediction,
danelfin_*) stays registered.

Enabled on IBKR (`config/live.yaml`): momentum, trend, mean_reversion, stat_arb, multi_factor, sentiment,
event_driven, volatility_breakout, seasonality, regime_hmm, pattern_recognition (11). Disabled: gann,
ml_prediction, danelfin_* (account closed 2026-08-16), investing_analyst_ratings, insider_cluster (the last is not registered by default). Alpaca runs `strategy_mode: allocation`, so its
`strategies` block is dormant.

## 5. Data layer

| Component | Path | Notes |
|---|---|---|
| PIT store | `src/firm/data/pit_store.py` | `PointInTimeDataStore`: `load`, `get_prices(symbols, asof, lookback_days)`, `get_fundamentals`, `get_estimates`, `get_ai_scores`, `get_sentiment`, `load_macro/get_macro`, `get_universe(asof)`, `get_universe_union(start,end)`, plus `get_live_signals` (L197), `get_best_stocks` (L211), `get_market_percentile_pool` (L220). The P0-02 seal guard must cover all of these, or sit at `load()` or one internal chokepoint rather than per method. asof-only; no `get(symbol,start,end,asof)` |
| Loaders (chokepoint) | `src/firm/runtime.py` | `load_prices` L181, `load_fundamentals` L212, `load_macro` L229, `load_sentiment` L250, `load_analyst_ratings`, `load_ai_scores`, `load_market_percentile`, `load_universe_membership` |
| Cache | `src/firm/data/cache.py`, `data/cache/` | sha256-keyed parquet; `combined/` holds prices/fundamentals keys |
| Universe | `src/firm/data/universe.py`, `scripts/import_universe_membership.py` | Index-membership resolver (not ETF/futures aware) |
| Providers | `src/firm/data/providers/` | Live and backtest data; `FallbackProvider` chain |
| EODHD store | `data/research/eodhd/` (about 4.7 GB, gitignored) | `etfs/`, `etfs_full/` (130 ETFs from 1993), `prices/`, `us_universe_full/`, `symbols_active`, `symbols_delisted`, `corporate_actions/`, `crypto/`, `forex*/`, `manifest.json` (no hashes) |
| Cleaning v2 (frozen) | `scripts/eodhd_clean.py` | `clean_bars`, `cleaning_fingerprint`; misses the 999999.9999 sentinel |
| Fetchers | `scripts/fetch_eodhd_prices.py`, `fetch_eodhd_extras.py`, `fetch_data.py`, `backfill_tiingo_prices.py`, `backfill_macro.py`, `alt_premia_data.py`, `fetch_insider_data.py` | No futures fetcher |
| Other research data | `data/research/{fred,insider,s2_forward}` | `s2_forward` is written by the frozen S2 shadow job |
| Futures data | None | No data, no roll code |

## 6. Brokers, live engine, allocation mode

- **Brokers**: `src/firm/brokers/{base,ibkr,alpaca}.py`. `OrderRequest` has `order_type` and `client_order_id`
  (engine builds `c{cycle}-{idx}-{symbol}-{side}`). IBKR adapter builds stock contracts only; never call
  `IBKRBroker.connect()` from uvicorn's asyncio loop.
- **Live engine**: `src/firm/live/engine.py`. Drawdown kill switch `_check_drawdown` (about L1302; IBKR 8%,
  Alpaca 25%; persisted to `kill_switch_state.json`; `reset_kill_switch` rebases the peak). Allocation cycle
  `_run_allocation_cycle` (about L3036) with once-per-day idempotency (`_allocation_should_run`). Startup
  config via `resolve_live_startup()` in `src/firm/live/provider_utils.py:61` (do not duplicate YAML merge).
- **Execution safety**: `live/execution_safety.py` (`FIRM_ALLOW_TRADING`, `guard_order`,
  `guard_live_submission`, audit JSONL). `live/capital_gate.py` is the existing weak G-PAPER analogue
  (60 days, 100 orders, 15% DD; served by `GET /live/capital-gate`).
- **Reconciliation**: `live/portfolio_sync.py`, `live/order_reconciliation.py`, `live/sleeve_reconciliation.py`.
- **Alerts**: `live/notifications.py` (Discord embeds; unknown alert kinds auto-titled).
- **Allocation mode** (`strategy_mode: allocation`, Alpaca live since 2026-09-30): `src/firm/allocation/`
  as in section 2. Sleeves register through `SLEEVE_REGISTRY` in `sleeves.py:154`. Config:
  `config/live_alpaca.yaml` (core 0.92 = SPY 0.6/IEF 0.4 monthly, `btc_trend` 0.08 weekly, `band_abs` 0.02,
  `max_gross` 1.0, `kill_switch_drawdown` 0.25). Docs: `docs/allocation_deploy_runbook.md`,
  `docs/allocation_forward_test_plan.md`.
- **Frozen forward tests**: `scripts/allocation_forward_test_preregistered.py` (bars I1-I5,
  `BAR_SLEEVE_DRIFT`; nothing computes `simulated_nav` yet), `scripts/s2_forward_preregistered.py`,
  `scripts/s2_forward_shadow.py` (`deploy/s2-forward-shadow.{service,timer}`, timer not installed),
  `docs/s2_forward_test_plan.md`, `docs/s2_forward_snapshot.json` (tracked, post-seal).

## 7. Eval, statistics, research harnesses, ledgers

**Statistics (`src/firm/eval/`)**

| Function | Location | Notes |
|---|---|---|
| `_norm_cdf`, `_norm_ppf` | `overfitting.py:40,45` | Private, but effectively frozen API: imported by `scripts/run_eodhd_s1_evaluation.py:49`, `run_eodhd_s3_evaluation.py:83`, `run_alt_premia_evaluation.py:688` and four frozen preregs (`eodhd_s1_..._bars.py:383`, `eodhd_s3_..._bars.py:507`, `eodhd_s4_..._bars.py:438`, `insider_cluster_..._bars.py:286`); `eodhd_s5_..._bars.py:473` keeps its own duplicate. Names, signatures and outputs must not change |
| `cscv_pbo(matrix, n_partitions=8, embargo_pct=0.0)` | `overfitting.py:99` | Returns a bare float; about 10 frozen callers |
| `probabilistic_sharpe` | `overfitting.py:171` | Per-period Sharpe; 0.0 sentinel for n<8 |
| `deflated_sharpe` | `overfitting.py:193` | Has `prior_trials` and `prior_trial_sharpes`; falls back to PSR(0) if variance <= 0 |
| `verdict`, `walk_forward_overfitting` | `overfitting.py:247,304` | |
| `MonteCarloAnalyzer` | `robustness.py:24` | `bootstrap_returns` is iid; used by `capital_gate`, `eval/reports`, `patterns/significance.py` |
| `sharpe_ratio` etc. | `metrics.py` | Annualised (do not mix with per-period PSR/DSR inputs) |
| `purged_kfold_splits` | `patterns/ml/purged_cv.py:59` | Calendar-day embargo; no CPCV |
| p-value and BH helpers | `patterns/significance.py` | `(1+c)/(1+n)`; BH |
| Stationary bootstrap | 4 copies in `scripts/run_alt_premia_evaluation.py:503`, `run_eodhd_s5_evaluation.py:528`, `run_standalone_strategy_evaluation.py:130`, `run_combination_evaluation.py:132` | Frozen scripts; do not edit |
| Not present anywhere | `min_trl`, effective-N/ENB/MP, RC/SPA/Romano-Wolf, CPCV, PBOResult, GARCH-t harness, CUSUM, stress suite, `strategy_correlation.py` | New |

**Frozen pre-registrations and harnesses (`scripts/`)**: `alt_premia_preregistered_bars.py`,
`combination_preregistered_bars.py`, `insider_cluster_preregistered_bars.py`,
`pattern_ml_preregistered_bars.py`, `standalone_strategy_preregistered_bars.py`,
`eodhd_s1_industry_momentum_preregistered_bars.py`, `eodhd_s2_breadth_overlay_preregistered_bars.py`,
`eodhd_s3_bond_commodity_trend_preregistered_bars.py`, `eodhd_s4_52wk_high_preregistered_bars.py`,
`eodhd_s5_crypto_momentum_preregistered_bars.py`, `allocation_forward_test_preregistered.py`,
`s2_forward_preregistered.py`, `futures_trend_preregistered_bars.py` (DRAFT, never run). Runners:
`run_alt_premia_evaluation.py`, `run_combination_evaluation.py`, `run_eodhd_s1..s5_evaluation.py`,
`run_insider_cluster_evaluation.py`, `run_standalone_strategy_evaluation.py`, `run_walk_forward_pbo_audit.py`,
`run_pbo_trial_audit.py`. Mechanism: each prereg has `bars_fingerprint()` over frozen constants; the runner
appends to its ledger.

**Legacy numeric conventions (pin in P1-02/P1-04 equivalence tests; `overfitting.py`)**

- `_norm_ppf` clamps p to [1e-9, 1-1e-9] (L47). `probabilistic_sharpe` returns 0.0 for n<8 and for sd==0 (L180); skew and kurt are moments divided by the ddof=1 sd (kurt non-excess); SE denominator clamped at 1e-12 (L189).
- `deflated_sharpe` falls back to PSR(0) when var<=0 or n_trials<2; `prior_trials` raise N but leave var_sr unchanged.
- `cscv_pbo` returns 0.5 when N<2 or T<2S (L134, L167), drops the trailing T mod S rows (the most recent data), ranks by ordinal `argsort` (tie order arbitrary, L161), clamps w to [1e-6, 1-1e-6] (L164), and skips splits whose OOS data the embargo removes entirely.
- Equivalence fixtures must feed legacy-convention moments, avoid the sentinel and fallback branches (or assert the new API raises there), use tie-free matrices with T%S==0, and fix the tolerance in the ticket (e.g. 1e-12). Do not loosen tolerances to make a failing test pass.

**Ledgers (`docs/*_trial_history.json`)**: 11 files, at least 6 shapes. Counts: `S1` 4, `S2` 4, `S3` 5, `S4` 4,
`s5` 3, `alt_premia` 10 (includes 3 benchmarks BM1_SPY, BM2_60_40, BM3_SPY_VT), `insider_cluster` 8,
`standalone_strategy` 11, `combination` 57 (16 entries), `pattern_ml` 104 (folds x candidates; 13 configs),
`allocation_forward_test` (0 trials). Ledgered total 210. `docs/eodhd_shortlist_protocol_2026_10.md:74` uses 190
(before S1-S5).

| File | Shape / count field | Sharpe array and basis | Mutable top-level fields | Notes |
|---|---|---|---|---|
| S1, S3, S4, alt_premia, insider_cluster, standalone_strategy | `entries[]` with `n_trials`, `trials`, `trial_daily_sharpes` | daily Sharpe array | `cumulative_trials` (alt_premia also `note`) | alt_premia includes 3 benchmarks |
| S2 | `entries[]`, `n_trials`, `trials` | `trial_daily_sharpes_cash_excess_governing` (use for var_sr); `trial_daily_sharpes_bm2_excess_alt` is an active-return Sharpe vs BM2, never pool it into var_sr | `cumulative_trials` | no plain `trial_daily_sharpes` key |
| s5 | list/entries with `appended_at`, `variant_names`, `prior_trials_used` (207); no `n_trials`, no date | `trial_daily_sharpes` (3) | none | count = len(sharpes) |
| combination | `entries[]` (16): `date`, `source`, `n_trials`, `candidates`, `oos_sharpes`, `verdicts` | OOS Sharpes per candidate (P1-01 must check the basis before pooling) | `_about` | |
| pattern_ml | `entries[]` (4): `n_folds` x `n_candidates`, `n_trials`, `reported_pbo`, `reported_deflated_sharpe` | none | `cumulative_trials_through_last_entry`, `_comment` | contributes counts to N only |
| allocation_forward_test | single forward-test record | none | `family` | not a trial ledger |

Top-level counters change on append, so `test_trial_history_append_only` must compare only the `entries` prefix
(old entries byte-identical), never the whole file. P1-01 `legacy_adapters` and that test follow this table.
Ledger helper code: `scripts/validate_pattern_ml_workstream_d.py` (`load_trial_history`, `append_trial_history`).
No shared ledger module exists. `docs/gann_research_closeout.md` and branch `origin/research/gann-archive` hold
the unledgered Gann work.

## 8. Review package

`scripts/build_review_package.py` (tests: `tests/test_build_review_package.py`) builds `review/`
(`data/MANIFEST.csv.gz` with sha256 per file, `ARTIFACT_INDEX.json`, `DATA_BUNDLE.md`, `ENVIRONMENT.json`,
`TIMELINE.md`, `studies/{edge_search_step1_standalone, edge_search_step2_alt_premia, insider_clusters,
shortlist_S1..S5}`). The licensed-data bundle lives outside the repo
(`/local/store/review_bundles/review_bundle_20261001T004018Z.tar.zst`).

## 9. CI and deploy

- **CI**: `.github/workflows/ci.yml`, triggers on `pull_request` to main only (no push trigger). Backend job:
  `ruff check src tests`, then `pytest -q --cov=firm --cov-fail-under=75` (Python 3.14). Frontend job: vitest
  and build. `testpaths = ["tests"]`, so `tests/integrity/` would be collected automatically. No mypy, no
  pytest markers, no pre-commit, no CODEOWNERS.
- **Deploy units (`deploy/`)**: `ai-trading.service` (IBKR :8000), `ai-trading-alpaca.service` (Alpaca :8001,
  `FIRM_DATA_DIR=data_alpaca`, `FIRM_LIVE_CONFIG=config/live_alpaca.yaml`, `FIRM_ENABLE_PATTERN_SCAN=1`),
  `ibgateway.service`, `backup-live-state.{service,timer}`, `best-stocks-arm.{service,timer}`,
  `s2-forward-shadow.{service,timer}`, `nginx-ai-trading.conf`, `nginx-rate-limit.conf`. Both firm-api units set
  `FIRM_LLM_CONFIG=config/llm_ab_llm.yaml`; the Alpaca unit also sets `FIRM_API_PORT=8001`. The firm-api,
  best-stocks-arm and s2-forward-shadow units use the main `.venv` (editable install points at the live `src/`),
  so they import live code; `ibgateway` and `backup-live-state` run shell scripts.

## 10. Plan path reconciliation

### 10.1 Source-plan proposed paths

| Source path | Actual path | Status | Note |
|---|---|---|---|
| `src/validation/sharpe_stats.py` | `src/firm/validation/sharpe_stats.py` (new) | new | Wraps `firm.eval.overfitting` primitives; legacy file not edited. P1-02 |
| `src/validation/pbo.py` | `src/firm/validation/pbo.py` (new) | new | New `pbo(M, S=16)`; legacy `cscv_pbo` (S=8) stays. P1-04 |
| `src/validation/cv.py` | `src/firm/validation/cv.py` (new) | extend | Generalises `src/firm/patterns/ml/purged_cv.py`, which stays. P1-05 |
| `src/validation/bootstrap.py`, `nulls.py` | `src/firm/validation/{bootstrap,nulls}.py` (new) | new | Lifts logic from `patterns/significance.py` and the 4 script copies. P1-07 |
| `src/validation/multiple_testing.py` | `src/firm/validation/multiple_testing.py` (new) | new | RC/SPA/Romano-Wolf absent everywhere. P1-06 |
| `src/validation/effective_trials.py` | `src/firm/validation/effective_trials.py` (new) | new | P1-03 |
| `src/validation/synthetic.py` | `src/firm/validation/synthetic.py` (new) | new | GARCH-t in numpy (`arch` not installed). P1-08. Not related to `src/firm/data/synthetic.py` |
| `src/validation/stress_suite.py` | `src/firm/validation/stress_suite.py` (new) | new | P3-07 |
| `src/validation/diversification.py` | `src/firm/validation/diversification.py` (new) | new | P1-03 / P4-02 |
| `src/research/ledger.py` | `src/firm/research/ledger.py` (new) | new | Canonical store is host-level `/local/store/research-ledger/trials.jsonl`. P1-01 |
| `src/research/prereg.py` | `src/firm/research/prereg.py` (new) | new | Indexes the frozen `scripts/*_preregistered*.py`. P1-09 |
| `src/research/` (seal, data access) | `src/firm/research/{seal,data_access,capture}.py` (new) | new | P0-02, P1-12 |
| `src/data/futures_loader.py` | `src/firm/data/futures_loader.py` (new) | new | Deferred (P2-07). Package is `src/firm/data/` |
| `src/data/etf_loader.py` | `src/firm/data/etf_loader.py` (new) | new | P2-02 |
| `src/data/qa.py` | `src/firm/data/qa.py` (new) | extend | Extends cleaning rules and `tests/test_eodhd_clean.py` (P2-03) |
| `src/data/PointInTimeDataStore.get(symbol,start,end,asof)` | `src/firm/data/pit_store.py` (`get_prices`, etc.) | extend | The 4-arg `get` does not exist; seal hook is a default-None `_ACCESS_GUARD` |
| `src/costs/model.py` | `src/firm/costs/model.py` (new) | new | Reuses `src/firm/agents/_liquidity.py` without editing. P2-04 |
| `src/signals/{vol,ewmac,breakout,carry}.py` | `src/firm/signals/*.py` (new) | new | Not under `strategies/` (different ABC). P3-01..P3-04 |
| `src/portfolio/forecast_combine.py`, `sizing.py`, `weights.py` | `src/firm/portfolio/*.py` (new) | new | Live-imported package, so new modules are listed in the isolation test. Distinct from `agents/research/_combine.py` and signal-level HRP |
| `src/portfolio/` (existing per author) | `src/firm/portfolio/{optimizer,state,attribution}.py` | exists | Pipeline-only; not the home for handcrafting |
| `src/risk/limits.py`, `kill_switch.py`, `kelly.py` | `src/firm/risk/*.py` (new) | new | Existing risk is `agents/risk.py` (pipeline) and `live/execution_safety.py` (order level) |
| `src/monitoring/{decay,fidelity}.py` | `src/firm/monitoring/*.py` (new) | new | No CUSUM today. `agents/research/_circuit_breaker.py` is a rolling-Sharpe damper, not CUSUM |
| `src/lifecycle/*` | `src/firm/lifecycle/*` (new) | new | P5-01. Naming: repo "sleeve" = capital bucket |
| `src/execution/rebalance.py` | `src/firm/allocation/` + `LiveTradingEngine._execute_orders` | n/a | No `src/execution/`. Reuse `Allocator`; add `carver_sleeve.py` via one `SLEEVE_REGISTRY` entry (P6-01) |
| `src/backtest/` engine for signed/multiplier | `src/firm/backtest/vector_engine.py` (new) | new | backtrader untouched. P3-09 |
| `src/strategy_correlation.py` (author: "existing") | none | n/a | Does not exist on any branch; written new in P1-03 |
| `strategies/_deprecated/` | `src/firm/strategies/registry.py` status sidecar | extend | No file moves; statuses in registry. P0-03 |
| `config/research_freeze.yaml` | same | new | P0-02 |
| `config/gates.yaml` | same | new | P0-08 |
| `config/costs.yaml`, `risk.yaml`, `tax_il.yaml`, `stress_periods.yaml`, `universe_etf.yaml`, `universe_futures.yaml` | same | new | Flat `config/`. Never edit `live*.yaml` or `llm_ab_*.yaml` |
| `data/holdout/` | `.gitignore` entry only | n/a | Directory never existed. Forward data is sealed by ACL; real data sits in `data/research`, `data/cache`, `data/`, `data_alpaca/` |
| `data/manifests/` | `research/data_manifests/` (new) | new | Tracked; reuses `build_review_package` MANIFEST format |
| `runs/research/` | `research/reports/` (e.g. `research/reports/core_v1/`, P3-08) | new | Source CI rule "every artefact has a ledger trial_id" becomes "every `research/reports` artefact references verified trial_ids". Not the ledger directory |
| `runs/approvals/` | `research/approvals/` (new) | new | |
| (ledger) | `research/ledger/` (new) | new | Separate row; ledger is not a report tree |
| `runs/monitoring/` | `research/monitoring_sealed/<name>/` for every post-seal output (P5-04 weekly snapshots, P6-02 paper reports, live comparator series); `research/monitoring/` only for non-data artefacts (procedures, templates) | new | Post-seal data in a tracked tree is checked out into every worktree and defeats the ACL seal, so `research/monitoring_sealed/` is gitignored (P0-06) and ACL-restricted (OD-05), written only by owner-run exports. Otherwise add `research/monitoring/**` to the research Read-deny and data_access deny-list. `runs/` is gitignored and holds only 9/25-9/28 runs; daily monitor state goes to `data/forward_monitors/<name>/` |
| `src/reporting/after_tax.py`, `diversification_report.py` | `src/firm/reporting/*` (new) | new | P2-05 / P4-02 / P3-08 |
| `.github/workflows/integrity.yml` | same | new | P0-04 |
| `scripts/daily_reconcile.py` | same | new | P5-04; timer only under OD-15 |
| `runs/validation/synthetic_<date>.md` | `research/validation/` (new) | new | P1-08 DoD |
| `research/charters/TEMPLATE.md` | same | new | P5-02 |
| `HOLDOUT_UNSEAL_TOKEN` env var | n/a | n/a | Replaced by an owner-held token preimage with its sha256 in config (P3-10 / OD-05) |
| Source sandbox block (`allowUnsandboxedCommands=false`, `filesystem.denyRead`) | not adopted globally | n/a | Research profile only (OD-07b/c); a global sandbox breaks ops systemctl/curl |
| `research/ledger/trials.jsonl` or `trials.sqlite` | `research/ledger/trials.jsonl` (mirror of host ledger) | new | Canonical copy at `/local/store/research-ledger/` |
| `research/ledger/legacy_backfill.csv` | same | new | P0-05 |
| `research/preregistration/<date>_<family>.yaml` | `research/preregistration/INDEX.yaml` plus new-family YAML | new | Existing mechanism is the frozen scripts; YAML indexes them |
| `research/charters/*` | same | new | P5-02 |
| `research/lifecycle/registry.yaml` | same | new | P5-01 |
| `scripts/backfill_legacy_trials.py` | same | new | P1-01 |
| `scripts/validate_stats_pipeline.py` | same | new | P1-08 |
| `scripts/unseal_forward_holdout.py` | same | new | P3-10 (deferred) |
| `scripts/select_instruments_for_capital.py` | same | new | P7-01 |
| `tests/integrity/` | `tests/integrity/` (new, with `__init__.py`) | new | First test subdirectory; picked up by existing `testpaths` |
| `tests/data_qa/` | `tests/data_qa/` (new, synthetic fixtures) | new | |
| `tests/validation/` etc. | `tests/test_*.py` flat | n/a | Repo convention is flat |
| `.claude/settings.json`, `.claude/hooks/*` | `.claude/settings.json` (committed, ops-compatible); hooks are templates in `deploy/claude-research-hooks/`, installed root-owned at `/etc/claude-code/hooks/` and registered only in the research user's settings (P0-04) | new | Deny Edit/Write on `.claude/settings.json`, `.claude/settings.local.json`, `.claude/hooks/**`, `research/preregistration/**`, `research/charters/**`, `config/gates.yaml`, `tests/integrity/**`, `tests/test_live_import_isolation.py`, `docs/s2_forward_snapshot.json`, `research/monitoring_sealed/**`; deny `git push --force`; `.claude/worktrees/**` allowed. Ops paths (`config/live*.yaml`, `deploy/`) are not denied in committed settings (OD-08). Managed settings template in `deploy/claude-managed-settings.json`: **managed settings are host-global** (every Claude Code session on the host, including root ops sessions). The template may hold only rules safe for ops sessions (deny Edit/Write on settings and hooks, deny force-push). Do not put `disableBypassPermissionsMode` or the holdout read-deny hook there unless the hook first checks effective uid and enforces only for the non-root research user (OD-07c), exiting 0 for root; alternatively put research-only restrictions in the research user's own root-owned read-only `~/.claude/settings.json`. P0-04 red-team must confirm a root ops session can still read `data_alpaca/` and `data/logs` and run systemctl/curl (feedback_production_incident_priority) |
| `.github/CODEOWNERS` | same | new | Only meaningful with a second identity and branch protection (OD-06). Protection must be a ruleset or classic rule that lists the owner account as a bypass actor (or leaves "include administrators" off) and applies the PR/code-owner requirement to the research-agent identity only; all recent commits are direct pushes by one identity. Document an emergency hotfix path in AGENTS.md's ops-session section (OD-08); P0-04 red-team checks the owner can still push a hotfix to main |
| `AGENTS.md`, `CLAUDE.md` | same | extend | Merge, not verbatim. Fold CLAUDE.md "Rules of thumb" into AGENTS.md first (OD-08) |
| `docs/HOLDOUT_POLICY.md`, `DEPRECATIONS.md`, `LLM_POLICY.md`, `JURISDICTION_NOTES.md`, `GUARDRAIL_REDTEAM.md` | same | new | Top-level `docs/`, each indexed in `docs/README.md` |
| `docs/REPO_MAP.md` | this file | new | P0-01 |
| `PLAN.md`, `tickets/` | `PLAN.md` (repo root; tracked) and `plan/tickets/<ID>.md` (56 files), plus `plan/OWNER_DECISIONS.md` and `plan/drafts/<ID>/` | new | Design said `plan/PLAN.md`; DECIDED 2026-10-02: root `PLAN.md`, and every plan file now cites the root path |

### 10.2 Design `actual_paths` by ticket (existence verified on 2026-10-02)

Legend: E = exists today, N = new.

| Ticket | Path | E/N | Status | Note |
|---|---|---|---|---|
| P0-01 | `docs/REPO_MAP.md` | N | new | This file |
| P0-01 | `PLAN.md` (repo root, exists, tracked; design said `plan/PLAN.md`), `plan/OWNER_DECISIONS.md` (E, tracked), `plan/tickets/` (E, 56 files) | E | review | Root location decided; P0-01 reviews and reconciles |
| P0-01 | `docs/README.md` | E | extend | Add index rows only |
| P0-06 | `tests/test_live_import_isolation.py` | N | new | Subprocess, fresh interpreter, `FIRM_DATA_DIR=tmp` |
| P0-06 | `scripts/live_import_smoke.py` | N | new | Runs `build_orchestrator` via `resolve_live_startup` on both live configs |
| P0-06 | `scripts/new_research_worktree.sh` | N | new | Disk check (20 GB free), shared research venv |
| P0-06 | `.gitignore` | E | extend | Add `.claude/worktrees/`, `data/holdout/`, `data/forward_monitors/`, `research/monitoring_sealed/`. No re-include for `research/ledger/returns/**/*.parquet`: returns stay on the host ledger (P1-01), so the global `*.parquet` rule keeps ignoring them |
| P0-02 | `config/research_freeze.yaml` | N | new | |
| P0-02 | `docs/HOLDOUT_POLICY.md` | N | new | |
| P0-02 | `src/firm/research/{__init__,seal,data_access}.py` | N | new | |
| P0-02 | `src/firm/data/pit_store.py` | E | extend | LIVE-IMPORT-PATH. Default-None `_ACCESS_GUARD` |
| P0-02 | `src/firm/runtime.py` | E | extend | LIVE-IMPORT-PATH. Same hook on `load_*` |
| P0-02 | `tests/integrity/__init__.py`, `tests/integrity/test_holdout_guard.py` | N | new | |
| P0-02 | `tests/test_live_fundamentals_unaffected_by_seal.py` | N | new | |
| P0-03 | `src/firm/strategies/registry.py` | E | extend | LIVE-IMPORT-PATH. Add `status()`, `list_allocatable()`; `list_strategies()` unchanged |
| P0-03 | `src/firm/runtime.py` | E | extend | LIVE-IMPORT-PATH (touches_live; imported at top of `live/engine.py`). Warn-and-skip for archived names in the empty-list fallback (L58) |
| P0-03 | `src/firm/live/engine.py` | E | extend | LIVE-IMPORT-PATH (touches_live). Fallbacks at L210 (`__init__`), L768 (`_all_strategy_names`), L836 (`update_strategies`) |
| P0-03 | `src/firm/live/pipeline_warmup.py` | E | extend | LIVE-IMPORT-PATH (touches_live). Fallback at L95 |
| P0-03 | `src/firm/api/routers/live.py`, `src/firm/api/routers/meta.py` | E | extend | LIVE-IMPORT-PATH (touches_live). Fallbacks at L1452 and L117 |
| P0-03 | `docs/DEPRECATIONS.md` | N | new | |
| P0-03 | `pyproject.toml` | E | extend | Register a `deprecated` marker; no packages change |
| P0-03 | `tests/test_strategy_lifecycle_status.py` | N | new | |
| P0-04 | `AGENTS.md`, `CLAUDE.md` | E | extend | See OD-08 |
| P0-04 | `.claude/settings.json` | N | new | |
| P0-04 | `deploy/claude-research-hooks/{deny_holdout.py,stop_integrity.sh}` (design: `.claude/hooks/*`) | N | new | Templates; owner installs root-owned at `/etc/claude-code/hooks/` and commits the templates |
| P0-04 | `deploy/claude-managed-settings.json` | N | new | Template; owner installs root-owned in `/etc/claude-code/` |
| P0-04 | `.github/CODEOWNERS` | N | new | |
| P0-04 | `.github/workflows/ci.yml` | E | extend | Add push trigger |
| P0-04 | `.github/workflows/integrity.yml` | N | new | |
| P0-04 | `tests/integrity/test_guardrails_present.py`, `test_trial_history_append_only.py` | N | new | |
| P0-04 | `.cursor/rules/research-integrity.mdc` | N | new | |
| P0-04 | `docs/GUARDRAIL_REDTEAM.md` | N | new | |
| P0-04 | `docs/claude-memory/MEMORY.md` | E | extend | |
| P0-08 | `docs/gate_deviation_register_2026_10.md`, `config/gates.yaml`, `tests/integrity/test_gates_frozen.py` | N | new | |
| P0-07 | `config/live.yaml`, `config/llm_ab_llm.yaml`, `config/llm_ab_quant.yaml` | E | extend | Deferred (OD-03); live restart ticket |
| P0-07 | `deploy/ai-trading.service`, `deploy/ai-trading-alpaca.service` | E | extend | Deferred |
| P0-05 | `research/ledger/legacy_backfill.csv`, `docs/legacy_trial_census_2026_10.md` | N | new | Seeded from the 11 `docs/*_trial_history.json` (read only) |
| P1-01 | `src/firm/research/{ledger,legacy_adapters}.py` | N | new | |
| P1-01 | `/local/store/research-ledger/trials.jsonl` | N | new | Host path outside every worktree |
| P1-01 | `research/ledger/trials.jsonl`, `research/ledger/returns_manifest.jsonl` | N | new | Git mirror; returns parquet stay on the host (`/local/store/research-ledger/returns/`) and in the private bundle |
| P1-01 | `scripts/backfill_legacy_trials.py`, `scripts/sync_ledger_mirror.py` | N | new | |
| P1-01 | `tests/test_ledger.py`, `tests/test_ledger_concurrency.py`, `tests/integrity/test_ledger_chain.py` | N | new | |
| P1-12 | `src/firm/backtest/run.py` | E | extend | LIVE-IMPORT-PATH. Capture call in `execute_backtest` |
| P1-12 | `src/firm/runtime.py` | E | extend | LIVE-IMPORT-PATH (touches_live). Capture call in `run_backtest_from_config`; also edited by P0-02 and P0-03, serialise merges |
| P1-12 | `src/firm/api/app.py` | E | extend | LIVE-IMPORT-PATH (touches_live; owner acknowledgement before start): set the `IN_API_PROCESS` tag in `run()` and the `lifespan` hook, NOT in `create_app()` (which runs at import) |
| P1-12 | `src/firm/backtest/_capture_state.py` | N | new | Stdlib-only, intentionally live-imported; on the isolation test's ALLOW list, not the forbidden list |
| P1-12 | `src/firm/experiments/runner.py` | E | extend | LIVE-IMPORT-PATH (touches_live; lazily imported by `api/jobs.py`). Capture call in `ExperimentRunner.run` |
| P1-12 | `src/firm/research/capture.py` | N | new | |
| P1-12 | `tests/test_entry_point_capture.py`, `tests/integrity/test_entry_points_wrapped.py` | N | new | |
| P1-02 | `src/firm/validation/{__init__,sharpe_stats}.py`, `tests/test_sharpe_stats.py` | N | new | `eval/overfitting.py` untouched |
| P1-07 | `src/firm/validation/{bootstrap,nulls}.py`, `tests/test_bootstrap.py`, `tests/test_nulls.py` | N | new | |
| P1-10 | `src/firm/eval/robustness.py` | E | extend | LIVE-IMPORT-PATH. Opt-in `method='stationary'` |
| P1-10 | `tests/test_robustness.py` | E | extend | Default output must stay bit-identical |
| P1-04 | `src/firm/validation/pbo.py`, `tests/test_pbo.py` | N | new | |
| P1-05 | `src/firm/validation/cv.py`, `tests/test_cv.py` | N | new | |
| P1-05 | `pyproject.toml` | E | extend | Add `hypothesis` to dev extras (worktree venv only) |
| P1-06 | `src/firm/validation/multiple_testing.py`, `tests/test_multiple_testing.py` | N | new | |
| P1-03 | `src/firm/validation/{effective_trials,diversification}.py`, `tests/test_effective_trials.py` | N | new | |
| P1-09 | `research/preregistration/INDEX.yaml`, `src/firm/research/prereg.py`, `tests/test_prereg.py` | N | new | |
| P1-08 | `src/firm/validation/synthetic.py`, `scripts/validate_stats_pipeline.py`, `research/validation/` | N | new | |
| P1-11 | `scripts/build_review_package.py`, `tests/test_build_review_package.py` | E | extend | Add ledger/prereg/report folders |
| P2-08 | `src/firm/data/cleaning.py`, `tests/test_cleaning_v3.py`, `docs/eodhd_cleaning_v3.md` | N | new | `scripts/eodhd_clean.py` (E) stays frozen |
| P2-01 | `config/universe_etf.yaml`, `config/universe_futures.yaml`, `docs/universe_rationale_2026_10.md` | N | new | Futures list reconciles the 19 markets in `scripts/futures_trend_preregistered_bars.py` (E, frozen) |
| P2-02 | `src/firm/data/{etf_loader,manifest}.py`, `research/data_manifests/`, `tests/test_etf_loader.py` | N | new | |
| P2-07 | `src/firm/data/{futures_loader,futures_roll}.py`, `tests/test_futures_loader.py` | N | new | Deferred (OD-01, OD-02) |
| P2-03 | `src/firm/data/qa.py`, `scripts/run_data_qa.py`, `tests/data_qa/{__init__.py,...}`, `research/data_qa/` | N | new | Own special-closures table; `allocation/calendar.py` (E) not edited |
| P2-04 | `src/firm/costs/{__init__,model}.py`, `config/costs.yaml`, `tests/test_costs.py` | N | new | `agents/_liquidity.py` (E) reused, not edited |
| P2-06 | `docs/JURISDICTION_NOTES.md` | N | new | Source: `docs/research_findings_beyond_equities_2026_09_30.md` (E) |
| P2-05 | `src/firm/reporting/{__init__,after_tax}.py`, `config/tax_il.yaml`, `tests/test_after_tax.py` | N | new | |
| P5-02 | `research/charters/{TEMPLATE,etf_trend,etf_breakout,core_combined}.md` | N | new | |
| P3-01..P3-04 | `src/firm/signals/{__init__,vol,ewmac,breakout,carry}.py`, `tests/test_signals_*.py` | N | new | |
| P3-05 | `src/firm/portfolio/forecast_combine.py`, `tests/test_forecast_combine.py` | N | new | |
| P3-06 | `src/firm/portfolio/sizing.py`, `tests/test_sizing.py` | N | new | Follows `Allocator.fractional_symbols` |
| P3-09 | `src/firm/backtest/vector_engine.py`, `tests/test_vector_engine.py` | N | new | |
| P3-11 | `research/preregistration/`, `scripts/core_v1_preregistered.py`, `research/reports/core_v1/constants.md` | N | new | New-family harness follows `scripts/<family>_preregistered.py` convention |
| P3-07 | `src/firm/validation/stress_suite.py`, `config/stress_periods.yaml`, `tests/test_stress_suite.py` | N | new | |
| P4-01 | `src/firm/portfolio/weights.py`, `tests/test_weights.py` | N | new | `agents/analysts/__init__.py` (E) only cross-checked |
| P4-02 | `src/firm/reporting/diversification_report.py`, `src/firm/validation/diversification.py`, `tests/test_diversification_report.py` | N | new | |
| P4-03 | `src/firm/risk/{__init__,limits}.py`, `config/risk.yaml`, `tests/test_risk_limits.py` | N | new | |
| P4-04 | `src/firm/risk/kill_switch.py`, `config/kill_switch.yaml`, `tests/test_kill_switch_tiers.py` | N | new | Pure module; engine `_check_drawdown` (E) unchanged until P6-01 |
| P4-05 | `src/firm/risk/kelly.py`, `tests/test_kelly.py` | N | new | Unrelated to the pipeline's `kelly` allocation method |
| P3-08 | `scripts/core_v1_preregistered.py`, `scripts/run_core_v1_evaluation.py`, `docs/core_v1_trial_history.json` | N | new | Follows repo convention |
| P3-08 | `src/firm/reporting/gate_report.py`, `research/reports/core_v1/REPORT.md` | N | new | |
| P3-10 | `scripts/unseal_forward_holdout.py`, `research/reports/core_v1/FORWARD_HOLDOUT.md`, `research/approvals/` | N | new | Deferred, earliest 2027-10-01 |
| P5-05 | `docs/LLM_POLICY.md`, `src/firm/strategies/registry.py` | N / E | new / extend | Policy and status only |
| P5-06 | `src/firm/monitoring/{__init__,allocation_forward}.py`, `scripts/allocation_forward_monitor.py` | N | new | Calls `scripts/allocation_replay.py::replay` (E) |
| P5-06 | `deploy/allocation-forward-monitor.{service,timer}` | N | new | Install only under OD-15 |
| P5-06 | `data/forward_monitors/allocation/`, `research/monitoring_sealed/allocation_forward/` | N | new | |
| P5-06 | `tests/test_allocation_forward_monitor.py` | N | new | |
| P5-01 | `src/firm/lifecycle/{__init__,state_machine,gates,decommission,gated_sleeve}.py` | N | new | |
| P5-01 | `research/lifecycle/registry.yaml`, `research/approvals/`, `tests/test_lifecycle.py`, `tests/test_decommission.py` | N | new | `live/capital_gate.py` (E) unchanged |
| P5-03 | `src/firm/monitoring/decay.py`, `tests/test_decay.py` | N | new | |
| P5-04 | `src/firm/monitoring/{fidelity,shadow_loader}.py`, `scripts/daily_reconcile.py`, `research/monitoring_sealed/candidates/` (gitignored), `tests/test_fidelity.py` | N | new | Weekly candidate snapshots are post-seal, so not `research/monitoring/` |
| P6-03 | `config/live_candidate.yaml`, `deploy/ai-trading-candidate.service` | N | new | touches_live, OD-15 |
| P6-03 | `deploy/nginx-ai-trading.conf` | E | extend | touches_live, OD-15 |
| P6-01 | `src/firm/allocation/carver_sleeve.py`, `tests/test_carver_sleeve.py`, `tests/test_allocation_golden_replay.py`, `scripts/rebalance_dry_run.py` | N | new | |
| P6-01 | `src/firm/allocation/sleeves.py`, `src/firm/live/engine.py` | E | extend | LIVE-IMPORT-PATH (touches_live; the running Alpaca forward test imports both). One registry entry; opt-in engine hooks; `Allocator.plan` unchanged. Needs golden replay, allocation forward test unaffected, OD-11 |
| P6-01 | `tests/test_live_import_isolation.py` | N (P0-06) | extend | Allowlist change reviewed by owner |
| P6-02 | `research/monitoring/`, `research/approvals/`, `src/firm/lifecycle/gates.py` | N | new | |
| P7-01 | `scripts/select_instruments_for_capital.py`, `tests/test_select_instruments_for_capital.py` | N | new | |
| P7-02 | `research/approvals/`, `src/firm/live/execution_safety.py` | N / E | extend | LIVE-IMPORT-PATH (touches_live), OD-20. Real money behind `FIRM_ALLOW_TRADING` |
| P7-03 | `research/reports/annual/` | N | new | |

Note: this table was built from the design summary. Section 10.3 adds every path the tickets' text names that is not
in 10.1-10.2 (regenerated from `plan/tickets/*.md` on 2026-10-02 by a scratch script; P0-01 acceptance re-runs that check).

### 10.3 Ticket paths not listed above (generated from `plan/tickets/*.md`, existence verified on 2026-10-02)

| Ticket | Path | E/N | Status | Note |
|---|---|---|---|---|
| P0-02 | `tests/integrity/_fresh_import.py` | N | new | Shared fresh-interpreter helper (tmp `FIRM_DATA_DIR`, tmp cwd) for every subprocess import test |
| P0-04 | `deploy/claude-research-user-settings.json` | N | new | Template; owner installs root-owned, read-only to the research user, at `~research/.claude/settings.json` |
| P0-04 | `deploy/claude-research-hooks/{deny_holdout.py,stop_integrity.sh}` | N | new | Templates; owner installs root-owned at `/etc/claude-code/hooks/` (plus `/etc/claude-code/research_freeze.deny.json`). Replaces the design's `.claude/hooks/*` |
| P0-05 | `scripts/seed_legacy_census.py` | N | new | Reads the 11 `docs/*_trial_history.json` read-only |
| P0-06 | `/local/store/research/ai-trading-system` (research clone; worktrees at `<clone>/.claude/worktrees/<ID>`) | N | new | Outside the live checkout, owned by the research user (OD-07). Never `git worktree add` in the live checkout |
| P0-06 | `/local/store/research-venvs/<depset>/` | N | new | Shared non-editable research venvs; `pytest-xdist` installed here only |
| P0-08 | `research/approvals/gates_signoff.yaml` | N | new | Alternative home for the signed gates hash (else the register's signed line) |
| P1-01 | `/local/store/research-ledger/{returns,inbox}/` | N | new | Owner-provisioned (mode 2775, research group). Returns parquet never enter git |
| P1-01 | `research/ledger/returns_manifest.jsonl` | N | new | Tracked manifest (trial_id, sha256, n_obs) replacing a git copy of the parquet files |
| P1-03 | `tests/test_diversification.py` | N | new | |
| P1-08 | `tests/test_synthetic.py` | N | new | |
| P1-09 | `research/preregistration/TEMPLATE.yaml` | N | new | |
| P1-11 | `review/research/` (`README.md`, `VERIFICATION.json`) | N | new | Review-package section for ledger, preregs, reports |
| P1-12 | `tests/integrity/test_host_ledger_untouched.py`, `tests/conftest.py` (E, extend: autouse ledger-root fixture) | N / E | new / extend | |
| P2-01 | `tests/test_universe_config.py` | N | new | |
| P2-02 | `scripts/fetch_il_macro.py`, `data/research/il_macro/` | N | new | Data dir is gitignored via `data/research/` |
| P2-03 | `tests/data_qa/{test_spikes,test_stale,test_missing_days,test_negative_prices,test_fx,test_roll_gaps}.py` | N | new | Synthetic fixtures |
| P3-01..P3-04 | `tests/test_signals_{vol,ewmac,breakout,carry}.py` | N | new | |
| P3-08 | `tests/test_gate_report.py`, `tests/test_run_core_v1_evaluation_smoke.py`, `tests/integrity/test_report_provenance.py` | N | new | Integrity test owned by P3-08 (agent drafts, owner commits) |
| P3-08 | `research/reports/core_v1/{results.json,drafts/}` | N | new | |
| P3-11 | `scripts/run_core_v1_constants.py`, `tests/test_core_v1_preregistered.py`, `tests/test_core_v1_constants_driver.py`, `tests/integrity/test_core_v1_prereg_order.py` | N | new | |
| P3-11 | `research/preregistration/{core_v1,core_v1_constants_addendum}.yaml`, `research/reports/core_v1/{constants.json,constants_addendum.draft.yaml}` | N | new | Driver writes the draft; owner copies it into `research/preregistration/` |
| P4-02 | `research/reports/core_v1/diversification.md` | N | new | |
| P4-04 | `config/kill_switch.yaml` | N | new | Tier settings; survival_ref from the charter artefact |
| P5-01 | `$FIRM_DATA_DIR/lifecycle_overrides.json` | N | new | Runtime state (`zero_weight` flags), not tracked |
| P5-02 | `plan/drafts/P5-02/` (incl. `test_charters.py`), `tests/integrity/test_charters.py`, `research/charters/` | N | new | Agent drafts in `plan/drafts/`; the owner commits charters and the integrity test |
| P5-04 | `src/firm/monitoring/shadow_loader.py`, `research/monitoring_sealed/candidates/` | N | new | Monitor-only loader (never via `data_access`); snapshots gitignored, owner-run export only |
| P5-05 | `plan/drafts/P5-05/test_docs_indexed.py`, `tests/integrity/test_docs_indexed.py` | N | new | Owner-authored integrity test |
| P6-01 | `tests/test_engine_optin_defaults.py`, `tests/test_rebalance_dry_run.py`, `tests/test_engine_limit_at_mid.py`, `research/approvals/P6-01_golden.json` | N | new | |
| P6-03 | `scripts/backup_live_state.sh`, `frontend/src/api/{client,types}.ts` (optional), `.gitignore` (`data_candidate/`) | E | extend | touches_live |
| P6-03 | `tests/test_candidate_config.py`, `data_candidate/` | N | new | `data_candidate/` is runtime state, gitignored |

Existing paths the tickets cite read-only (no change): `config/llm.yaml`, `config/experiments/`, `config/strategies/`,
`data/{approvals.json,cycle_history.json,execution_audit.jsonl,live_state.db,memory,order_history.json}` (IBKR live state, sealed
from research), `docs/PROJECT_CONTEXT.md`, `docs/claude-memory/`, `docs/{edge_search_plan,edge_search_verdict,ensemble_redundancy_audit,
futures_data_vendor_comparison,insider_cluster_verdict,optimal_combination_fix,pattern_ml_final_verdict}_2026_09.md`,
`docs/eodhd_shortlist_{protocol,verdict}_2026_10.md`, `docs/{formal_pbo_audit,remediation_progress,portfolio_construction_diagnosis,
pattern_recognition_plan,llm_ab_experiment_log,futures_trend_integration_sketch}.md`, `docs/archive/strategy_regime_weights_calibration.md`,
`docs/prompts/research_brief_new_instruments.md`, `review/README.md`, `review/data/MANIFEST.csv.gz`,
`scripts/{alt_premia_data,calibrate_danelfin_ai_score,fetch_eodhd_extras,fetch_insider_data,train_pattern_ml}.py`,
`scripts/run_eodhd_s{1,4,5}_evaluation.py`, the frozen preregs of section 10.4, and the existing tests `tests/test_{backtest_run,
capital_gate,cost_model,execution_cost,execution_safety,hrp_combine,hrp_wiring,overfitting,pit_store,purged_cv,scripts_run_backtest,
significance,strategies,tca}.py`. Gann archive paths (`scripts/gann_followup_study.py`, `research/gann-archive/...`) exist only on
`origin/research/gann-archive` (P0-05 reads them with `git show`).

### 10.4 Frozen or load-bearing existing paths (reference only, do not touch)

All verified present: `scripts/*_preregistered*.py` (13 files, section 7), `scripts/eodhd_clean.py`,
`docs/{S1,S2,S3,S4,s5}_trial_history.json`, `docs/{alt_premia,insider_cluster,standalone_strategy,combination,pattern_ml,allocation_forward_test}_trial_history.json`,
`docs/*evaluation*.json`, `docs/allocation_replay_2026_09.json`, `docs/allocation_portfolio_backtest_2026_09.json`,
`docs/s2_forward_snapshot.json`.

## 11. Open points found while mapping

- `AGENTS.md` and `CLAUDE.md` both still describe Alpaca as sleeved and allocation mode as "not live yet"; Alpaca has run `strategy_mode: allocation` since 2026-09-30. Correct this during the P0-04 merge.
- `docs/alt_premia_trial_history.json` text says combination = 52; the file sums to 57 (stale, not editable).
- S-family prereg `prior_trials` placeholders are 205/206/207; the running ledgered total (210) is recorded nowhere.
- `AGENTS.md` says CLAUDE.md mirrors it; they have diverged. Code and docs cite CLAUDE.md by name
  (`DEPLOY.md`, `src/firm/live/scheduler.py`, `src/firm/data/insider_transactions.py`), so keep the file.
- `.gitignore` globally ignores `*.parquet`; that is now intended for ledger returns (they stay on the host, P1-01), and `research/monitoring_sealed/` needs an explicit ignore (P0-06).
- `data/holdout/` has never existed; an explicit ignore rule is needed so a CSV there is not committed.
- The Stop-hook and tests must use the worktree's own interpreter or the shared research venv `/local/store/research-venvs/<depset>/`. The main `.venv` editable install points at the live `src/`.

## 12. Worktree and venv policy (P0-06)

- **Never edit, test in, or `git worktree add` inside the live checkout** (`/local/store/git/ai-trading-system`): the
  two firm-api services run from it with `Restart=always`, and worktrees share the parent's refs, config and hooks.
  Research agents work in their own clone, `/local/store/research/ai-trading-system` (owned by the `research` user,
  OD-07), one worktree per ticket at `<clone>/.claude/worktrees/<ID>`. The owner merges and pulls into the live checkout.
- Create a worktree with `scripts/new_research_worktree.sh <TICKET_ID> [dependency-set] [--create-venv]`. It refuses to run
  when the toplevel is the live checkout, when `/` has less than `FIRM_MIN_FREE_GB` (default 20) GB free, or when the
  worktree exists; it writes `<worktree>/.research-depset` and prints the test command. `.claude/worktrees/` and
  `.research-depset` are in `.gitignore`.
- Worktrees do not inherit gitignored files (`.env`, `.venv`, `data/`, `data_alpaca/`, `.claude/settings.local.json`), so
  tests that need data must skip; never copy `.env` into a worktree.
- **Interpreter.** The live `.venv` has an editable install pointing at the live `src/`; it must never be used to test a
  worktree without `PYTHONPATH=<worktree>/src`, and nothing is ever installed into it. Use a shared, non-editable research
  venv per dependency set at `/local/store/research-venvs/<set>/` (only the owner or `--create-venv` creates it; `pytest-xdist` is
  installed there only, not in `pyproject.toml`), run with `PYTHONPATH=<worktree>/src`, and check that `firm.__file__`
  is under the worktree.
- **Resources (6 cores, about 11 GB RAM, shared with IB Gateway and two firm-api processes).** At most 3 research agents at
  once; pytest under `nice -n 10 ionice -c3`, xdist `-n 2` at most; heavy runs (P1-08, P3-08, P3-11) outside 09:15 ET and outside the
  allocation rebalance windows, optionally under `prlimit --as=<bytes>`; at least 20 GB free before creating a worktree.
- **Live import graph.** `tests/test_live_import_isolation.py` (CODEOWNERS-protected) proves no new research module is
  imported by `firm.runtime`, `firm.live.engine` or `firm.api.app`; `scripts/live_import_smoke.py` proves both live configs
  still build (fresh interpreter, temp `FIRM_DATA_DIR`, temp cwd, config copy with state paths in the temp dir, no network,
  no broker connect, no `.env`). Both are part of the LIVE-IMPORT-PATH protocol (PLAN.md section 8).
