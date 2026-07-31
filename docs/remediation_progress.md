# Trading-system audit remediation — progress tracker

Tracks execution of the remediation plan produced from the original
full-system audit (weaknesses/biases review). The plan itself lives at
[`/root/.cursor/plans/trading_system_audit_remediation_7a945538.plan.md`](/root/.cursor/plans/trading_system_audit_remediation_7a945538.plan.md)
(not in this repo — it's a Cursor plan file; do not edit it directly, its
per-todo `status:` fields are kept in sync with the TodoWrite tool
automatically). This doc is the durable, repo-committed record of what's been
done, in progress, and left, plus context/detail beyond what the plan file's
todo list captures, so work can resume in a fresh session/chat without the
prior conversation history.

Last updated: 2026-07-31.

## How to resume

1. Read this file top to bottom — "In progress" says what to pick up next;
   "Completed" item #45 has the most recent detail.
2. Re-read the plan file (path above) for the original rationale/evidence
   behind each item if needed — its `todos:` frontmatter status should match
   the "Full task list" section below, other than the two ad-hoc items noted
   there that aren't part of the original plan.
3. Sanity-check the suite is still green before continuing:
   ```bash
   .venv/bin/python -m pytest -q --ignore=tests/test_api.py   # ~2min
   .venv/bin/python -m pytest tests/test_api.py -q             # ~4min, run separately (flaky under full-suite parallelism, not from these changes)
   cd frontend && npx vitest run && npx tsc --noEmit
   ```
4. Recreate the TODO list (see "Full task list" below) with the TodoWrite tool
   at the start of the new session, marking the right item `in_progress`.
5. Continue with the next `pending` item, or finish the `in_progress` one.

## Completed

All of the following are implemented, tested, and documented in
`docs/PROJECT_CONTEXT.md` (see its section headers for details):

1. **Portfolio-construction diagnosis** — instrumented per-stage attribution;
   see `docs/portfolio_construction_diagnosis.md`.
2. **Alerting wired into live engine** (`alert_callback` construction path).
3. **Durable kill-switch halt state** (`data/kill_switch_state.json`) +
   reset path.
4. **Live cost config wiring** (`config/live.yaml` `costs:` block →
   `ExecutionAgent`).
5. **LLM-enhanced analyst score bounds clamping** + re-z-scoring.
6. **Health endpoint IBKR connectivity check**.
7. **CI Python version fix** (matches `pyproject.toml` `requires-python`).
8. **Sector-map missing → hard failure at live startup** (was a warning).
9. **RiskAgent decision logging** — every clip/scale/veto now explicit.
10. **ADV/participation-rate liquidity checks in RiskAgent**
    (`src/firm/agents/_liquidity.py::estimate_adv_dollars`).
11. **Portfolio-level pairwise correlation caps in RiskAgent**.
12. **UniverseResolver wired into backtest + live** (survivorship bias fix).
13. **Sentiment data loaded into backtest PIT store**.
14. **Bid-ask spread + market-impact + short-borrow cost modeling** (see #16
    for the market-impact-model follow-up that made impact
    size/volume-aware instead of flat).
15. **CI**: pytest-cov gate, scheduler tests, frontend Vitest added.
16. **LLM JSON parsing**: replaced float()-casts with Pydantic validation.
17. **guard_order/RiskProfile wired into live engine order path**.
18. **News-guard fails closed** on calendar load failure; alerts on stale
    CSV fallback.
19. **True walk-forward redesign**: genuine train→select→test optimization
    via `param_grid`, fixed PBO trial semantics (see
    `src/firm/experiments/runner.py`, `src/firm/eval/overfitting.py`).
    Frontend: `frontend/src/pages/NewBacktest.tsx` parameter-grid textarea +
    overfitting diagnostics panel.
20. **Point-in-time universe membership**: `UniverseResolver.symbols_between`,
    `PointInTimeDataStore.get_universe_union`, dynamic per-rebalance universe
    resolution in `FirmStrategy` (handles mid-backtest listings/delistings).
21. **Durable live state persistence**: `src/firm/live/state_store.py`
    (SQLite) for portfolio NAV history + per-strategy attribution;
    `PortfolioState.restore_history`, `PerformanceAttribution.export_state`/
    `restore_state`. Also fixed a latent attribution fragility
    (`LiveTradingEngine._orders_to_fills`).
22. **Size/volume-aware market-impact model** (replaces flat-pct-only
    costs): `src/firm/agents/_liquidity.py` (`sqrt_impact_pct`, shared with
    RiskAgent's ADV cap), wired into `FirmStrategy` (backtest,
    `_apply_market_impact`) and `ExecutionAgent` (live, pre-trade cost
    estimate). `market_impact_coefficient` config knob end-to-end (config
    files → `RunRequest`/`WalkForwardRequest` → routers →
    `frontend/src/pages/NewBacktest.tsx`). Tests: `tests/test_cost_model.py`
    `TestMarketImpactModel`.
23. **Fundamentals PIT: real filing dates instead of period-end+45d
    heuristic**: `firm.data.providers.base.resolve_filing_date()` shared
    helper; `EdgarProvider` uses SEC's real `filed` XBRL field (max across
    contributing concepts per period); `FMPProvider` uses `fillingDate` from
    the income-statement record. Verified Massive/TwelveData/AlphaVantage/
    Finnhub genuinely don't expose a real filing date (checked their actual
    API docs) — heuristic correctly remains for those. Tests:
    `tests/test_provider_base.py`, `tests/test_edgar_provider.py`,
    `tests/test_fmp_provider.py`.
24. **Broker & host failover** (documented + implemented):
    - `Broker.reconnect()` concrete default method (`src/firm/brokers/base.py`)
      — disconnect-then-connect, swallows a failing disconnect.
    - `LiveTradingEngine` now attempts one inline reconnect on the cycle
      worker thread when a cycle's broker call raises `BrokerError`
      (`_try_broker_reconnect`), tracks `_consecutive_broker_failures`,
      escalates to a `broker_disconnected_sustained` critical alert past
      `broker_disconnect_alert_threshold` (config, default 3), and emits
      `broker_reconnected` once a cycle's broker calls succeed again.
    - Fixed a real systemd unit-name mismatch: `DEPLOY.md`/`setup.sh` were
      creating `ib-gateway.service` (hyphenated) while
      `deploy/ai-trading.service`'s `After=`/`Wants=` (and actual
      production) use `ibgateway.service` (no hyphen) — the dependency
      ordering was silently a no-op. Fixed both docs/scripts to the
      canonical no-hyphen name.
    - Full runbook written: `docs/PROJECT_CONTEXT.md` → "Broker & host
      failover" (disconnect handling table, IB Gateway restart checklist,
      host-crash recovery steps, full-host-loss manual rebuild steps,
      monitoring recommendations).
    - Tests: `tests/test_live_engine.py::TestBrokerReconnect` (5 tests:
      inline reconnect success, sustained-disconnect escalation, recovery
      alert + counter reset, default `Broker.reconnect()` behavior +
      disconnect-failure resilience).
25. **`regime_hmm` negative-Sharpe investigation resolved** (id:
    `regime-hmm-negative-drag`) — root cause was Bull/Bear states labeled
    purely by mean-return rank with no check on whether the gap to the
    runner-up state was statistically meaningful (label-switching risk on a
    short, noisy per-symbol fit). Two mechanisms built and A/B-tested on the
    3 windows from `docs/portfolio_construction_diagnosis.md`:
    - **Signal-logic fix (shipped, on by default)**: `GaussianRegimeModel`
      (`src/firm/regime/model.py`) now computes a `separation` effect size
      per label (`_build_separation`); `HMMRegimeStrategy`
      (`src/firm/strategies/regime_hmm.py`) damps Bull/Bear signals below
      `min_state_separation` (default `0.5`, floor
      `separation_damping_floor=0.15`) toward neutral. Confirmed via
      controlled A/B to flip `regime_hmm`'s own attributed Sharpe from
      negative to positive in all 3 windows; portfolio-level effect was
      positive in 2/3, mixed in the third (attributed to `optimal`
      combination's inverse-covariance reweighting, a `regime-conditional-
      weighting`-adjacent open question, not a bug in this fix).
    - **Generic per-strategy rolling-Sharpe circuit breaker (shipped, off by
      default)**: `src/firm/agents/research/_circuit_breaker.py`, wired into
      `net_scores_for_blackboard` upstream of signal combination, config
      end-to-end (`Settings.strategy_circuit_breaker`,
      `RunRequest`/`WalkForwardRequest`, `POST/PUT /api/live/*`,
      `config/{settings,live}.yaml`, `frontend/src/pages/{NewBacktest,
      LiveConfig}.tsx` marked experimental). A/B'd with natural-seeming
      default thresholds and found to *hurt* portfolio Sharpe in all 3
      windows (over-triggers on short/noisy trailing-Sharpe windows) — left
      disabled by default pending recalibration, not deleted.
    - Tests: `tests/test_regime_hmm.py` (separation metric +
      damping logic), `tests/test_circuit_breaker.py` (damping calc +
      `net_scores_for_blackboard` integration), `tests/test_live_engine.py::
      TestUpdateStrategyCircuitBreaker`, `tests/test_live_provider_utils.py`.
    - Full writeup: `docs/PROJECT_CONTEXT.md` → "Portfolio-construction
      diagnosis follow-up"; `docs/portfolio_construction_diagnosis.md` →
      "Follow-up (2026-07-27)".
26. **Fixed pre-existing flaky/failing tests** (id: `fix-preexisting-test-
    failures`) — all 4 previously-known issues:
    - `tests/test_live_engine.py::test_catch_up_starts_cycle_when_market_open`
      — deleted (not fixed in place): it raced a background daemon thread
      against a fixed 5s poll-loop deadline, which is exactly the flaky
      pattern; a deterministic, `threading.Event`-based replacement already
      existed at `tests/test_scheduler.py::
      test_starts_catch_up_cycle_when_market_open`, so the flaky duplicate
      added no coverage worth keeping.
    - `tests/test_api.py::TestLiveConfigRoundTrip::
      test_put_config_schedule_restarts_scheduler` — this was a **real bug**,
      not just a bad test: `PUT /api/live/config` (and `POST /api/live/start`)
      restart the scheduler on a background thread that waits on the
      pipeline-warmup gate before assigning `app.state.live_scheduler`, so
      `GET /api/live/config` immediately after either call read the *old*
      (or a hardcoded `"market_open"` default) schedule value, not the one
      just requested. Fixed by setting `engine._schedule` synchronously in
      `_start_live_engine`/`update_live_config` (`src/firm/api/routers/
      live.py`) and having `get_live_config` prefer it over the scheduler's
      own `_schedule_spec`. Un-skipped the test.
    - `frontend/src/pages/LiveConfig.test.tsx` (`filters the provider
      dropdown...`) — the test itself was wrong: it asserted the rendered
      `<option>` text equals the provider's raw `name` (e.g. `"groq"`), but
      `LiveConfig.tsx` correctly renders each option's display `label`
      (e.g. `"Groq (free tier)"`). The underlying component filtering logic
      (`configured || name === "ollama"`) was already correct. Fixed the
      test's expected strings and un-skipped it.
    - `test_api.py`'s `TestRuns` flakiness under full-suite load — not a
      logic bug (each test gets its own tmp-dir-backed `RunRegistry`, no
      shared state); a synthetic 2-strategy/2-year backtest takes ~5s
      standalone but `JobManager` serialises all runs behind one lock and
      can get CPU-starved when the whole suite (HMM fits, walk-forward
      folds, etc.) runs concurrently, occasionally exceeding the previous
      60s poll budget. Bumped `_launch_and_wait`'s poll budget from 120×0.5s
      (60s) to 240×0.5s (120s) — returns immediately on completion either
      way, so this only adds headroom under contention, not slack in the
      common case.
    - Verified: full backend suite green (960 passed, was 960 passed + 1
      skipped), `test_api.py` in isolation green (45 passed, was 44 passed +
      1 skipped), frontend Vitest green (82 passed, was 81 passed + 1
      skipped) + `tsc --noEmit` clean.
27. **Research roadmap scaffolding** (partial — 2026-07-27) — see "In
    progress" section above for per-item status. Code shipped:
    - `docs/longer_dataset_options.md` (vendor scoping)
    - `scripts/run_pbo_trial_audit.py` (formal PBO audit CLI)
    - `strategy_regime_weights` (`_regime_weights.py`, orchestrator +
      `_combine`, `settings.yaml`, `RunRequest`); tests:
      `tests/test_regime_weights.py`
    - LLM A/B presets: `config/llm_ab_{quant,llm}.yaml`,
      `FIRM_LLM_CONFIG`/`FIRM_LIVE_CONFIG` env overrides
      (`tests/test_config_env_overrides.py`), `docs/llm_ab_test_runbook.md`
28. **LLM A/B arm A started** (2026-07-27) — `deploy/ai-trading.service` now sets
    `FIRM_LLM_CONFIG=config/llm_ab_quant.yaml`; service restarted on production
    host (IBKR paper connected). Experiment log: `docs/llm_ab_experiment_log.md`.
    Target arm end: ≥2026-09-21 (8 weeks) before switching to arm B.
29. **Strategy regime weights — live parity + calibration** (2026-07-27) —
    `strategy_regime_weights` wired through `config/live.yaml`, live engine/API,
    frontend (`LiveConfig` / `NewBacktest`), `scripts/calibrate_strategy_regime_weights.py`;
    A/B on 3 diagnostic windows documented in `docs/strategy_regime_weights_calibration.md`
    (example weights hurt 2/3 windows — remain disabled).
30. **Membership ETL + paper-track-record policy** (2026-07-27) —
    `scripts/import_universe_membership.py`, `data/cache/README.md`, and
    "Minimum paper track record" gate in `docs/PROJECT_CONTEXT.md`.
31. **Formal PBO pipeline** (2026-07-27) — `scripts/run_walk_forward_pbo_audit.py`
    (walk-forward + genuine `param_grid` + PBO/DSR aggregate); `scripts/snapshot_llm_ab_arm.py`
    for weekly LLM A/B monitoring.
32. **Regime weights v2 research** (2026-07-28) — `scripts/suggest_strategy_regime_weights.py`
    (data-driven draft from train-window strategy × regime Sharpe); hold-out +0.051 Sharpe
    on `run_18mo_2025_2026`; `tests/test_import_universe_membership.py`;
    `docs/longer_dataset_vendor_decision.md` (Sharadar recommended).
33. **Tiingo price backfill** (2026-07-28) — `scripts/backfill_tiingo_prices.py` ran for
    live 25-symbol universe; `combined/prices` now ~2010-01-04 → present. Survivorship
    gap remains (no membership/delistings).
34. **Vendor alternatives + purchase decision** (2026-07-29) — reviewed EODHD, Norgate,
    Valuein, Massive, FMP vs Sharadar; **Bundle 10Y (~$49/mo) still recommended** for
    this stack. Clarified: 10Y covers current diagnostic/PBO windows (earliest 2020-12);
    15Y+ only if 2010-start panels. ROI is research/risk-reduction, not direct P&L.
    Operator on Sharadar free tier; purchase pending. Details:
    `docs/longer_dataset_vendor_decision.md`.

Also, earlier in this remediation pass (before the above numbered list):
eliminated all pytest warnings (datetime.utcnow deprecations, hmmlearn/
seaborn/chromadb third-party noise via `pytest.ini` filters, a pandas
`RuntimeWarning` in `multi_factor.py`), and killed stale/hung pytest
processes the user flagged mid-session.

**Items 35-41 below are from a separate, later audit** ("P/L-improvement
research" then a fresh "system improvement plan" audit) — not part of the
original Cursor remediation plan referenced at the top of this file, so they
have no `plan-id` and don't appear in the "Full task list" block below.

35. **P/L-improvement research, phases 1-4** (2026-07-29) — CPCV purge/embargo
    for the PBO audit (`eval/overfitting.py`, `experiments/runner.py`); HRP
    signal combination as an alternative to `optimal`
    (`agents/analysts/__init__.py`, opt-in via
    `signal_combination: {method: hrp}`); market-impact linear/sqrt crossover
    (`agents/_liquidity.py::market_impact_pct`, opt-in via
    `market_impact_crossover_participation`); CVaR tail-risk sizing overlay
    (`agents/risk.py::_cvar_overlay`, opt-in via `risk.cvar_limit`). All
    shipped disabled/unchanged-by-default; **none yet validated with a real
    backtest A/B** — do that before ever setting any of them in
    `config/live.yaml` (see "Next actionable picks" below).
36. **LLM RAG look-ahead audit** (2026-07-29) — full point-in-time audit of
    every RAG ingestor's date source. Found/fixed a real crash bug
    (`VectorStore.query` raised `TypeError` on an explicit `metadata["date"] =
    None` instead of failing closed) and de-duplicated the dense/BM25
    fail-closed logic into one shared `_doc_available_by()` helper. No
    look-ahead leak found in the LLM A/B's Arm B path. Two small non-blocking
    follow-ups noted, not done: tighten `earnings_ingestor.py` to real
    transcript dates; add a reporting lag to `run_ingestor.py`. Details:
    `docs/llm_lookahead_audit.md`.
37. **Structured trading-decision reflection** (2026-07-29) —
    `TradingMemoryLog.reflect()` now returns a `DecisionReflection`
    (verdict/what_worked/what_failed/lesson, `firm.llm.schemas`) instead of
    one unstructured prose blob. New `summarize_lessons()` aggregation +
    `GET /api/memory/lessons` + a lessons-learned digest panel on the
    Decisions page.
38. **Ensemble-HMM regime detector, implemented + A/B'd** (2026-07-29) —
    `EnsembleRegimeModel` (`regime/ensemble.py`, 5-seed majority vote) behind
    `MarketRegimeDetector(ensemble=True)`, wired into both
    `RiskAgent.regime_overlay` and `strategy_regime_weights` (both opt-in,
    off by default). A/B'd against the same 3 diagnostic windows: calms
    `regime_hmm`'s own attributed-Sharpe noise in every window but makes
    portfolio Sharpe *worse* in 2/3 with `strategy_regime_weights` enabled —
    detector noise wasn't the bottleneck for that feature. Shipped disabled.
    Details: `docs/regime_ensemble_scoping.md`.
39. **Frontend Live Dashboard crash fixed** (2026-07-29) — `LiveDashboard.tsx`
    rendered `cycle_id` as a truncated UUID string (`.slice(0, 8)`), but the
    backend (`CycleResult.cycle_id: int`) has always sent a plain integer —
    the page rendered fine for ~1s then crashed once the cycles query
    resolved and `.slice()` threw. Fixed end-to-end (component, types,
    client, test mocks) and rebuilt `frontend/dist`, which was independently
    stale (predated the crash-causing edits, from before this whole session).
40. **Live-engine boot-race outage found + fixed** (2026-07-30) — a fresh
    operational audit found the live engine had been silently stopped for
    **~27 hours**: `ai-trading.service`'s `After=`/`Wants=ibgateway.service`
    only waits for the IB Gateway *process* to fork, not for IBC's headless
    login to actually open the API port, so the boot-time
    `IBKRBroker.connect()` lost that race and nothing retried or alerted —
    this also explains why the LLM A/B experiment log looked frozen (no
    cycles ran). Fixed with `scripts/wait_for_ibgateway.sh` (a systemd
    `ExecStartPre` readiness wait, always exits 0) plus an app-level
    safety net (`auto_start_live_with_retries`, 1/60/180/300s backoff,
    disengages permanently the instant a start ever succeeds so it can't
    fight an operator's later `POST /live/stop`; fires a critical alert if
    every retry is exhausted). `ALERT_WEBHOOK_URL` (Discord) is now
    configured in production. Found and fixed a latent bug in the same pass:
    Discord's webhook API requires a `content` field and was silently
    rejecting the module's prior `text`-only payload.
41. **Operational hardening pass** (2026-07-30) — `.env`/`.htpasswd`
    permissions tightened (were world-readable, 644); nginx rate limiting
    (`limit_req`, 10r/s) + security headers (HSTS, X-Frame-Options,
    X-Content-Type-Options, Referrer-Policy) added in front of the
    dashboard; journald `SystemMaxUse=200M` cap (host was at 84% disk,
    journald alone held 675MB — freed ~500MB immediately); 4 pre-existing
    ruff lint findings fixed; leftover `ib-gateway`→`ibgateway` naming
    mismatches fixed in `setup.sh` (would have broken `systemctl enable` on
    a fresh automated install); `DEPLOY.md`'s nginx section rewritten to
    actually document basic auth + the new hardening (it previously didn't
    mention auth at all, despite that being what's actually protecting
    production).
42. **Logging/alert severity audit** (2026-07-30) — a two-agent audit
    (alert severities specifically, since those now drive the Discord
    webhook; general Python logging hygiene separately) found and fixed real
    mislabels. Most consequential: `broker_unavailable` was hardcoded
    `"critical"` for every branch in `engine.py`'s `except BrokerError`
    handler, including a same-cycle self-healed reconnect (e.g. IB Gateway's
    routine daily restart) — would have paged the new webhook for routine,
    already-resolved blips; now only the genuinely sustained case
    (`broker_disconnected_sustained`, past `broker_disconnect_alert_threshold`)
    is critical. `daily_limit_breach` now distinguishes `full_auto` (orders
    proceed past the guardrail unchecked → critical) from safely-held manual
    approval (→ warning) instead of both being warning. Also fixed: silent
    `except Exception:` blocks with zero logging in
    `agents/llm/base_llm_agent.py` (RAG retrieval failures were completely
    invisible) and `agents/trader.py`; several `debug`-level logs for real
    failures that should be visible at the default `INFO` level
    (`engine.py` attribution tracking, `risk.py` correlation/CVaR checks,
    `brokers/alpaca.py::get_position`); `brokers/ibkr.py`'s fabricated-0.0-
    price fallback bumped from `warning` to `error` (same level as a benign
    stale-price fallback, despite the method's own docstring calling it out
    as the worst case); missing `exc_info=True` on several safety-path
    exception logs (news-guard calendar fetch, broker reconnect, memory log
    reads, fallback-provider construction). **Not done** (lower-value, high
    file count): the same missing-`exc_info` pattern across ~15 data-provider
    files' `ProviderError` catch sites — left as a documented, low-priority
    follow-up rather than done in this pass.
43. **Investing.com Pro integration, Phases 0-2a** (2026-07-30/31) — full
    detail in `docs/investing_pro_integration.md`. Built an authenticated
    browser-driven (Playwright, not `requests` — Investing.com 403s plain
    HTTP clients even on public pages) scraper session
    (`src/firm/data/investing/`), verified end-to-end against the live site
    with real login. Phase 1 (economic calendar) wired into `news_guard` as
    an opt-in richer alternative to Forex Factory, fallback ladder preserved.
    Phase 2a added a new `estimates()` PitView capability end-to-end
    (provider ABC, `FallbackProvider`, both PitViewAdapters, `provider_utils`,
    `fetch_data.py`, and three separately-duplicated backtest data-loading
    paths that all needed the same fix) plus a new
    `investing_analyst_ratings` strategy backed by FMP's `grades-historical`
    (the only FMP analyst-data endpoint with genuine point-in-time history
    under this plan tier — `price-target-consensus`/`grades-consensus` are
    current-snapshot-only, `analyst-estimates`'s date is a forward target
    with no real as-of stamp). **A/B result across the standard 3 diagnostic
    windows: mixed/inconsistent (worse in the longest window, better in two
    shorter folds, no stable sign in the strategy's own attributed Sharpe)
    — shipped registered but NOT enabled in `config/live.yaml`**, same
    "ship disabled if inconclusive" discipline as the regime-ensemble
    precedent. Phase 2b (wiring the live Investing.com feed for this
    strategy) and Phase 3 (ProPicks/technical-summary) are **blocked, not
    just paused**: the per-stock Pro dashboard pages that carry this data
    (`investing.com/pro/<TICKER>`) return an interactive Cloudflare
    Turnstile challenge to the authenticated automated session every time —
    confirmed as a genuine bot-detection signal, not an account-tier issue,
    by the user loading the identical URL in their own real browser (loads
    fine, real data, no challenge). Getting past that reliably needs
    anti-detection/stealth browser tooling, which this integration
    deliberately does not build (see the doc's final section). Separately,
    the homepage's own "Fair Value" table — which looked like a shortcut
    around this — turned out to be a blurred marketing teaser widget
    (anonymized stock names linking to a pricing upsell page), not real
    per-symbol data.
44. **Danelfin AI-score integration — enabled in live paper trading**
    (2026-07-31) — full detail in `docs/investing_pro_integration.md`
    ("Danelfin AI-score" section). The user subscribed to Danelfin (Expert
    API plan) as a genuine-API replacement for the unreachable Investing.com
    Pro per-stock data. Verified live: `GET /ranking?ticker=X` has real
    historical AI/Fundamental/Technical/Sentiment/Low-Risk scores back to
    ~2016 (an undocumented `page=N` param is the only way to paginate that
    far — not in Danelfin's own official docs). Built the same
    `ai_scores()` PitView capability + `danelfin_ai_score` strategy pattern
    as `investing_analyst_ratings`, this time wiring all 3 backtest
    data-loading paths from the start (the missing-path bug from item #43
    was caught and fixed there, not repeated here). **A/B across the
    standard 3 diagnostic windows: portfolio Sharpe improved in every single
    window** (+0.279, +0.589, +1.576) — the most consistent result of
    anything tried this session (contrast with `investing_analyst_ratings`'
    mixed record, worse in 1 of 3). Honest caveat: the strategy's own
    standalone Sharpe was itself inconsistent (positive in the recent
    window, negative in the two older folds) — likely a genuine
    diversification effect under `optimal` combination rather than
    unambiguous proof of Danelfin's "AI Score predicts returns" marketing
    claim, but the portfolio-level metric this project has always promoted
    on was unambiguously better every time. **Enabled in `config/live.yaml`**
    (`strategies.enabled`/`auto_approve`, experiment renamed
    `paper_11_strategy`), live service restarted and verified healthy
    (`GET /api/live/status` shows `danelfin_ai_score` active,
    `broker_connected: true`). Also exposed Danelfin's `/v3/*`
    latest-snapshot endpoints (best-stocks/trading-parameters/
    price-forecast/performance) as read-only `DanelfinProvider` methods —
    not backtestable (no historical dates per Danelfin's docs) and
    deliberately not wired into any strategy/risk/execution logic yet.
45. **Danelfin live-signals (`/v3/*`) wired into a new strategy — enabled,
    unvalidated** (2026-07-31) — full detail in
    `docs/investing_pro_integration.md` ("Danelfin live-signals" section).
    Follow-up to item #44, per the user's explicit push to wire the `/v3/*`
    endpoints ("can't you feed all that goodness into my analysts
    implementation") rather than leaving them as read-only fetchers. Made
    one real live call per endpoint against AAPL to verify the actual field
    names/shape first (they'd been guessed, unverified, when item #44 first
    exposed them) — caught a real unit mismatch (`trading-parameters`'
    `stop_loss_pct`/`take_profit_pct` are percentage points; `price-forecast`'s
    `median_3m`/`q05_3m`/`q95_3m` are 0-1 decimals) and a real bug
    (`get_live_signals` always queried `/v3/performance` with `signal="buy"`
    regardless of what `trading-parameters` actually recommended — fixed to
    query the signal actually called). Built a new `live_signals()` PitView
    capability (same wiring points as `ai_scores()`, but fetched every live
    cycle with no opt-in env-flag gate, since snapshot-only data has no
    meaningful cache-only mode) and `danelfin_live_signals` strategy
    (direction from buy/sell, magnitude from forecast return, confidence
    from win-rate). **Cannot be A/B tested** — `pit_view.live_signals()` is
    always empty in a backtest by construction (no historical dates exist,
    ever), so this project's usual walk-forward promotion gate is
    structurally unavailable. **Enabled in `config/live.yaml` anyway**, per
    explicit user request rather than evidence — documented plainly as
    unvalidated in both the config comment and the docs section; experiment
    renamed `paper_12_strategy`, live service restarted and verified healthy
    (`GET /api/live/status` shows `danelfin_live_signals` active,
    `broker_connected: true`).

## In progress

### Research roadmap — partial progress (2026-07-27)

The five remaining plan todos are still **not fully closable** without external
data/time, but the following scaffolding landed this session:

| Plan id | What shipped | Still blocked on |
|---------|--------------|------------------|
| `longer-dataset` | Options + vendor decision docs; ETL scripts + tests; **Tiingo prices backfilled to ~2010** for live 25 names | **Sharadar Bundle 10Y purchase** (~$49/mo) + bulk import (membership + delistings) |
| `formal-pbo-correction` | `scripts/run_pbo_trial_audit.py` + `scripts/run_walk_forward_pbo_audit.py`; first cache-panel audit (`docs/formal_pbo_audit.md`: PBO=0.69, verdict `fail`) | Re-audit on longer delisting-inclusive panel after `longer-dataset` |
| `regime-conditional-weighting` | Feature + calibration + `scripts/suggest_strategy_regime_weights.py` (data-driven draft weights) | Hold-out validation passes before enabling live |
| `llm-ab-test` | `config/llm_ab_{quant,llm}.yaml`, `FIRM_LLM_CONFIG` / `FIRM_LIVE_CONFIG` env overrides, `docs/llm_ab_test_runbook.md` | ≥8 weeks sequential paper trading per arm |
| `min-paper-track-record` | Policy gate documented in `docs/PROJECT_CONTEXT.md` → "Real-capital allocation gate" (revised 2026-07-29 from a flat 6-12mo calendar bar to a trade-count/bootstrap-Sharpe-CI bar + tranched initial capital); `scripts/import_universe_membership.py` + `data/cache/README.md` | ≥60 trading days AND ≥100 trades AND bootstrap Sharpe CI lower bound > 0 |

Next actionable picks (operator): **purchase Sharadar Bundle 10Y** (see
`docs/longer_dataset_vendor_decision.md` ROI section) → run ETL → re-audit PBO.
**Weekly:** LLM A/B snapshot (`scripts/snapshot_llm_ab_arm.py`).

## Full task list (recreate with TodoWrite when resuming)

```
COMPLETED: Add SQLite/Postgres persistence for portfolio/kill-switch/attribution state (durable-state-persistence)
COMPLETED: Replace flat-pct costs with size/volume-aware market impact model (market-impact-model)
COMPLETED: Move fundamentals PIT to actual filing dates instead of period-end+45d heuristic (filing-date-pit)
COMPLETED: Document/implement broker and host failover runbook (broker-failover-runbook)
COMPLETED: Investigate regime_hmm strategy (negative Sharpe in 6/6 diagnostic runs) - fix signal logic or add strategy-level rolling-Sharpe circuit breaker (regime-hmm-negative-drag)
COMPLETED: Fix pre-existing flaky/failing tests: test_catch_up_starts_cycle_when_market_open, test_put_config_schedule_restarts_scheduler, frontend LiveConfig.test.tsx filter test, and test_api.py TestRuns flakiness under full-suite parallel run (fix-preexisting-test-failures)
PENDING: Acquire longer, delisting-inclusive historical dataset (longer-dataset)
PENDING: Run formal CSCV/PBO across actual historical strategy-selection process (formal-pbo-correction)
PENDING: Research regime-conditional strategy weighting (regime-conditional-weighting)
PENDING: Run controlled A/B of quant-only vs llm_enhanced in paper trading (llm-ab-test)
PENDING: Meet real-capital allocation gate (≥60 trading days, ≥100 trades, bootstrap Sharpe CI > 0), then fund at 10-20% tranche before scaling to full size (min-paper-track-record)
```

Note: `regime-hmm-negative-drag` and `fix-preexisting-test-failures` are **not**
in the original plan file's `todos:` list — they were added ad hoc mid-session
(the former surfaced from the plan's "research roadmap" section, which
mentions `regime_hmm` ranging down to a -1.50 Sharpe as supporting evidence
for the `regime-conditional-weighting` item, but didn't list it as its own
remediation task; the latter was added at the user's explicit request to
queue test-suite failures found along the way). Both are now done (see
"Completed" above) — they won't appear if you only look at the plan file's
frontmatter, so this note is just historical context for why they existed
outside it.

### Notes on the remaining `pending` items

- **longer-dataset**, **formal-pbo-correction**, **llm-ab-test**,
  **min-paper-track-record**: these are only partially (or not at all)
  actionable via code changes alone — they need real data
  acquisition/budget, an actual historical trial log to run CSCV against, or
  weeks/months of live paper-trading time respectively. Worth scoping what
  *can* be done now (e.g. investigate/shortlist data vendors for
  longer-dataset; scaffold an A/B config split for llm-ab-test) vs. what
  genuinely has to wait, next time these are picked up.
- **regime-conditional-weighting**: a real research+implementation task —
  worth checking whether it should be designed together with the
  `regime_hmm` circuit-breaker idea above (both touch "how much should a
  regime signal influence strategy weighting"), since solving them jointly
  may be more coherent than two independent bolt-ons.
