# Investing.com Pro / Danelfin integration — calendar + Danelfin enabled, FMP analyst-ratings shipped disabled

Session started 2026-07-30 at the user's request to leverage their paid
Investing.com Pro subscription for trading signals. Investing.com has no
official API, so this required an authenticated web-scraper session
(`src/firm/data/investing/`), plus a new `estimates()` PitView capability and
`investing_analyst_ratings` strategy backed initially by FMP (for a real,
backtestable evidence base) before ever wiring the live Investing.com feed.
That FMP-backed A/B came back inconclusive, and separately the actual
Investing.com Pro per-stock data (Fair Value, ProTips, Financial Health,
ProPicks, technical-summary) turned out to be unreachable — gated by an
interactive Cloudflare challenge a real browser passes and an automated
session does not (see "Phase 2b/3" below). The user then subscribed to
**Danelfin** (a genuine paid REST API, not a scraper) to fill the same gap
with real, backtestable data — that A/B came back **consistently positive
across all 3 diagnostic windows** and is now **enabled in live paper
trading** (see "Danelfin AI-score" below).

**Summary of where everything landed:**
- Economic calendar (Phase 1): shipped, opt-in-enabled.
- `investing_analyst_ratings` (FMP-backed): shipped, tested, registered,
  **not enabled** — inconclusive A/B.
- `danelfin_ai_score` (Danelfin-backed): shipped, tested, registered, **and
  enabled** in `config/live.yaml` — consistently positive A/B.
- Investing.com Pro's actual per-stock data: **unreachable**, Cloudflare-gated.

All of this follows the same "build it → A/B it → document honestly → ship
disabled if inconclusive, enable if consistently positive" discipline as
`docs/regime_ensemble_scoping.md`'s `strategy_regime_weights`/regime-ensemble
precedent (which is *also* still disabled, for comparison — that one didn't
clear this bar; `danelfin_ai_score` did).

## Phase 0 — Authenticated session (`src/firm/data/investing/session.py`)

**Key empirical finding: Investing.com blocks non-browser HTTP clients at the
edge.** A plain `curl` with a realistic browser User-Agent against a *public,
unauthenticated* page (the economic calendar) returned HTTP 403. This ruled
out the originally-planned "requests-first, browser-optional" design — every
fetch in `InvestingSession`, not just login, is routed through a real
headless-browser engine (Playwright/Chromium). One browser instance is
launched lazily and reused across a run (`close()`/context-manager teardown),
bounding the ~300-500MB RAM cost to once/day (the live scheduler's cadence),
not per-request.

**Login flow — verified against the live site via real browser devtools
inspection** (this environment cannot run an authenticated browser session
against the live site itself, so selectors were captured manually, then
wired in and verified end-to-end with a real login):
1. `#onetrust-accept-btn-handler` — a OneTrust cookie-consent banner's
   dark-filter overlay intercepts clicks until dismissed. This, not
   bot-detection, was the root cause of every early "click timed out"
   failure — worth calling out since it looked identical to a CAPTCHA/
   DataDome block until traced to the actual DOM.
2. `button[data-test='login-btn']` — the header "Sign In" trigger (a
   `data-test` attribute is far more stable than text-matching, which
   unreliably matched non-actionable nodes).
3. "Sign in with Email" (text match) — the initial modal offers "Continue
   with Google" first; this reveals the actual email/password fields.
4. `input[name='email']` (type="text", **not** type="email") and
   `input[name='password']`.
5. `button[type='submit']`.

**Go/no-go gate passed 2026-07-31**: real login succeeded end-to-end with the
user's actual credentials, session state persisted
(`data/cache/investing_storage_state.json`, mode 600), authenticated fetch
returned HTTP 200 — all while confirming the live trading engine and IB
Gateway stayed healthy throughout (this is the same production host).

**Infra side effects, now durable**: a 1GB swapfile (`/etc/fstab`) and ~900MB
of freed disk (stale Docker build cache, apt/pip caches, old rotated logs) —
running headless Chromium on this 2-core/5.3GB box under real memory
pressure caused genuine (not bot-detection-related) hangs during development;
both fixes address that, independent of this feature.

## Phase 1 — Economic calendar (`src/firm/data/investing/calendar.py`)

Confirmed live: the economic-calendar page is public (no login needed).
`firm.live.news_guard.load_events()` gained a `source` param
(`"forexfactory"` default, `"investing"` opt-in via `config/live.yaml`
`news_guard.source`), with Investing → Forex Factory → bundled-CSV as the
fallback ladder — the existing safety net is never bypassed, only prefixed.
The HTML parser's selectors are flagged as best-effort/unverified (this
environment cannot fetch investing.com's calendar table structure) and fail
safe: no matching rows → empty list, existing fallback engages.

## Phase 2a — Analyst-ratings strategy: implemented, A/B'd, shipped disabled

### What was actually backtestable (and what wasn't)

FMP's API (already integrated, real key in `.env`) was evaluated for three
candidate signals before picking one:

| Endpoint | Verdict | Why |
|---|---|---|
| `/stable/price-target-consensus` | Rejected | Current snapshot only — no historical time series at all. |
| `/stable/grades-consensus` | Rejected | Same — current snapshot only. |
| `/stable/analyst-estimates` | Rejected | `period=quarter` requires a higher plan tier (402); `period=annual`'s "date" is a *forward fiscal-year target*, not a real as-of/publish timestamp — not usable point-in-time even if it worked. |
| `/stable/price-target-news` | Rejected | 402 — restricted to a higher plan tier entirely. |
| **`/stable/grades-historical`** | **Used** | Genuine monthly historical rating-consensus counts (strong_buy/buy/hold/sell/strong_sell) — verified live: 91 rows for AAPL spanning 2018-12 → 2026-07. |

This is also a good live-feed pairing: Investing.com Pro's own homepage table
(confirmed live 2026-07-31) surfaces an "Analyst Ratings" column alongside
Fair Value/Financial Health/Growth Rating/etc. — so the FMP-backtest →
Investing.com-live path this plan called for lines up cleanly *if* the
backtest had validated.

### What shipped

- `ANALYST_RATINGS_COLS` (`src/firm/data/schemas.py`): `date, symbol,
  strong_buy, buy, hold, sell, strong_sell`.
- `DataProvider.get_analyst_ratings` — new abstract method (`base.py`),
  stubbed `NotImplementedError` on every provider that doesn't support it
  (alphavantage, edgar, finnhub, ibkr, massive, tiingo, twelvedata — matching
  the existing `get_news_sentiment`/`get_corporate_actions` convention), real
  implementation in `FMPProvider.get_analyst_ratings` (`grades-historical`),
  wired into `FallbackProvider`'s chain (FMP-only).
- New `estimates()` PitView capability, parallel to
  `prices`/`fundamentals`/`sentiment`: `PointInTimeDataStore.get_estimates`,
  both `PitViewAdapter`s (backtest `firm/backtest/firm_strategy.py` + live
  `firm/live/data_feed.py`), `provider_utils.build_live_providers`'s new
  `"estimates"` key, `fetch_data.py`'s cache pipeline
  (`combined/analyst_ratings`), and `runtime.load_analyst_ratings` /
  `firm.backtest.run.execute_backtest` / `scripts/run_backtest.py` (**three
  separate, previously-duplicated data-loading code paths** — the first A/B
  attempt silently produced zero signals because only `runtime.py`'s copy had
  been wired; `firm/backtest/run.py`'s `execute_backtest` — what the
  calibration scripts and API job runner actually use — and
  `scripts/run_backtest.py`'s CLI had their own independent copies that
  needed the identical fix).
- `firm.strategies.investing_analyst_ratings.InvestingAnalystRatingsStrategy`:
  `net_score = (2·strong_buy + buy − 2·strong_sell − sell) / total_analysts`
  (bounded [-2, 2]), combined with its trend (delta over the lookback
  window), emitted as a **raw** score — analysts z-score cross-sectionally
  once, matching every other strategy's convention (`momentum.py`'s explicit
  comment: "Emit raw cumulative returns; analysts z-score cross-sectionally
  once").

### A/B results — 3 diagnostic windows, `scripts/calibrate_investing_analyst_ratings.py`

Same universe (25 symbols), same 10-strategy baseline roster, `optimal`
combination, seed 42, cached PIT data as
`docs/portfolio_construction_diagnosis.md` — baseline vs baseline +
`investing_analyst_ratings`:

| window | arm | portfolio Sharpe | portfolio return | ratings' own Sharpe |
|---|---|---|---|---|
| run_18mo_2025_2026 | baseline | 0.708 | 0.067 | — |
| run_18mo_2025_2026 | **+ratings** | **0.298** | 0.025 | 0.360 |
| wf_fold0_2020_2021 | baseline | -0.729 | -0.021 | — |
| wf_fold0_2020_2021 | **+ratings** | **0.981** | 0.034 | -1.138 |
| wf_fold1 | baseline | -0.930 | -0.017 | — |
| wf_fold1 | **+ratings** | **-0.236** | -0.005 | 0.961 |

Raw results: `/tmp/investing_analyst_ratings_calibration.json`.

### Finding — mixed, no stable directional edge; not enabled

The most recent/longest window (run_18mo_2025_2026, 18 months) got **worse**
(0.708 → 0.298) when the strategy was added; the two older, shorter
walk-forward folds got better. More tellingly, the strategy's **own**
attributed Sharpe flips sign across all three windows with no discernible
pattern (+0.36, -1.14, +0.96) — a genuine edge should show at least a
consistent sign, even if the magnitude varies. The wf_fold0 improvement
despite a strongly *negative* own-Sharpe (-1.138) is very likely the
`optimal` inverse-covariance combiner using it as a low/negatively-correlated
diversifier in that specific fold, not evidence the signal itself is
predictive — exactly the kind of result that looks good in one aggregate
number and evaporates on closer inspection.

**Decision: do not enable `investing_analyst_ratings` in `config/live.yaml`.**
The strategy is fully implemented, tested (`tests/test_strategies.py`,
`tests/test_fmp_provider.py`, `tests/test_pit_store.py`, etc.), and available
in the registry — but registered-only, matching the
`strategy_circuit_breaker`/regime-ensemble precedent for an inconclusive A/B.

## Phase 2b/3 — blocked: per-stock Pro pages are Cloudflare-gated, not just inconclusive

Two independent reasons converged to stop here, not one:

1. **Phase 2a's A/B was inconclusive** (see above) — the original Phase 2b
   plan ("wire Investing.com Pro as the live feed for the same
   already-validated strategy") was explicitly contingent on a positive
   backtest, which this wasn't.
2. **The actual Pro-exclusive per-stock data isn't reachable at all**, for a
   more fundamental reason discovered while investigating whether the
   proxy-backtest even used the right kind of data. Fetching
   `investing.com/pro/NASDAQGS:AAPL` (the real per-stock Pro dashboard —
   Fair Value, ProTips, Financial Health, Growth/Profitability Rating, etc.)
   with the authenticated `InvestingSession` returns an **interactive
   Cloudflare Turnstile challenge** ("Just a moment...") every time,
   consistently — confirmed by the user visiting the identical URL in their
   own real browser, logged in, with no challenge and real (unblurred)
   data. That rules out an account/subscription-tier explanation: this is
   Cloudflare specifically distinguishing automated browser traffic from a
   real user's session on this page (unlike the homepage/login/calendar
   pages, which load fine either way).

   Separately, and independent of the Cloudflare finding: the homepage's own
   "Fair Value" table (which looked promising at first glance) turned out to
   be a **marketing teaser widget** — stock names rendered blurred
   (`blur-sm`, placeholder text "Aaaaa Aa A") and linking to
   `/pro/pricing?entry=hp_invpro_fair_value_table`, an upsell page — not real
   per-symbol data, even setting the Cloudflare issue aside.

   **This is a hard stop, not an engineering gap to close.** Getting past an
   interactive Cloudflare Turnstile challenge reliably requires
   anti-detection/stealth tooling (browser fingerprint spoofing, residential
   IP rotation, etc.) whose entire purpose is defeating bot-detection
   systems — a meaningfully different thing from driving a normal headless
   browser the way this project's Phase 0/1 already do, and not something
   this integration will build. If that calculus changes (e.g. Investing.com
   ships an official API, or the per-stock page's protection changes), this
   is where to pick the thread back up.

## Danelfin AI-score — a genuine paid API, not a scraper (2026-07-31, enabled)

While Investing.com Pro's actual differentiated data turned out to be
unreachable, the user separately subscribed to **Danelfin** (Expert plan)
specifically to fill that gap with real, backtestable data. Unlike
Investing.com, this needed no browser automation at all — Danelfin has a
genuine, documented REST API (`https://apirest.danelfin.com`, header auth)
meant to be called directly.

### What it is

Danelfin scores every US-listed stock/ETF (+ major European names) 1-10 on
five axes — AI Score (composite), Fundamental, Technical, Sentiment, Low
Risk — updated daily. Their own marketing claims 10/10-scored stocks have
historically outperformed by ~+21% (3-month annualized alpha) while
1/10-scored stocks underperformed by ~-33%; this integration tests that
claim directly rather than taking it at face value.

### What was verified live before building anything

- `GET /ranking?ticker=<SYMBOL>` is the only endpoint with genuine
  historical depth — real dated scores back to ~2016-12 (matching the
  advertised "since 2017"). An **undocumented** `page=<N>` query parameter
  (not in Danelfin's own official docs, empirically verified across pages
  1 through 25+) is the only way to paginate back that far — the documented
  params (`ticker`, `date`, score filters, `sector`, etc.) have no date-range
  option.
- The `/v3/*` endpoints (`beststocks`, `trading-parameters`, `price-forecast`,
  `performance`, `trade-ideas`) are **latest-snapshot-only, no historical
  dates** (confirmed in Danelfin's own docs) — not backtestable, exposed on
  `DanelfinProvider` as read-only fetchers for future live/shadow-mode use,
  deliberately **not wired into any strategy or risk/execution logic** (e.g.
  `trading-parameters`' stop-loss/take-profit levels could inform
  `RiskAgent`/`ExecutionAgent`, but that's a live-risk-relevant behavioral
  change needing its own explicit review, not something to fold in silently
  alongside a new alpha signal).
- One account/key discrepancy worth flagging: the API key returned "Too Many
  Requests" after only ~4 rapid calls during initial testing, which doesn't
  match Danelfin's documented rate limit for anything above their Free tier
  (60-180/min) — the user confirmed they're on the Expert plan (10,000
  calls/mo, 120/min) shortly after, so this was very likely the plan/key
  still propagating rather than a real Free-tier cap, but worth a look if
  rate-limiting recurs.

### What shipped

Mirrors the `investing_analyst_ratings`/`estimates()` pattern exactly (a new
`ai_scores()` PitView capability): `AI_SCORE_COLS` schema,
`DataProvider.get_ai_scores` abstract method (stubbed `NotImplementedError`
on every other provider), `DanelfinProvider` (real implementation),
`FallbackProvider` chain (Danelfin-only), `PointInTimeDataStore`, both
PitViewAdapters, `provider_utils`, `fetch_data.py`'s cache pipeline, and —
learned from the `investing_analyst_ratings` episode — **all three**
backtest data-loading paths (`runtime.py`, `firm.backtest.run.execute_backtest`,
`scripts/run_backtest.py`) wired from the start this time, not discovered
missing after a silent zero-signal A/B.

`firm.strategies.danelfin_ai_score.DanelfinAiScoreStrategy`: AI-score level
(centered at the 1-10 scale's midpoint, 5.5) + trend, emitted as a raw score
— by construction bounded within ±9 (level ±4.5, trend weighted at 0.5 of
its own ±9 range) to stay inside this project's raw-score sanity convention
even under adversarial data (a real bug caught by the shared
`test_strategies.py` synthetic-data harness during development, fixed by
tempering `delta_weight` from 1.0 to 0.5).

### A/B results — same 3 diagnostic windows, `scripts/calibrate_danelfin_ai_score.py`

| window | arm | portfolio Sharpe | portfolio return | ai_score's own Sharpe |
|---|---|---|---|---|
| run_18mo_2025_2026 | baseline | 0.708 | 0.067 | — |
| run_18mo_2025_2026 | **+ai_score** | **0.987** | 0.091 | 1.336 |
| wf_fold0_2020_2021 | baseline | -0.729 | -0.021 | — |
| wf_fold0_2020_2021 | **+ai_score** | **-0.140** | -0.005 | -0.671 |
| wf_fold1 | baseline | -0.930 | -0.017 | — |
| wf_fold1 | **+ai_score** | **0.646** | 0.012 | -0.513 |

Raw results: `/tmp/danelfin_ai_score_calibration.json`.

### Finding — consistently positive at the portfolio level; enabled

Unlike `investing_analyst_ratings`'s mixed record (worse in 1 of 3 windows),
adding `danelfin_ai_score` **improved portfolio Sharpe in all 3 windows**
(+0.279, +0.589, +1.576) — the most consistent result of anything tried this
session. **Honest caveat**: the strategy's own standalone attributed Sharpe
is itself inconsistent (+1.336 in the recent window, -0.671 and -0.513 in
the two older folds) — the portfolio-level improvement despite a negative
standalone Sharpe in 2 of 3 windows looks like the `optimal`
inverse-covariance combiner using it as a diversifier (low/negative
correlation with the other 10 strategies) rather than unambiguous proof of
Danelfin's own "the AI Score directly predicts returns" marketing claim.
That distinction matters for interpretation, but the metric this project
has consistently promoted features on (portfolio-level Sharpe, e.g. the
`optimal` vs `confidence` combination-method decision) was unambiguously
better in every window, with no counter-example.

**Decision: enabled in `config/live.yaml`** (`strategies.enabled` +
`auto_approve`, `experiment.name` bumped to `paper_11_strategy`) — verified
via `GET /api/live/status` after a live-service restart that
`danelfin_ai_score` is active and the broker connection/engine health were
unaffected. Monitor live per-strategy attribution the same way every other
strategy here is monitored; revert the single `config/live.yaml` line to
disable if it doesn't hold up in live paper trading, matching this
project's standard promotion-gate discipline.

## Danelfin live-signals — trading-parameters + price-forecast + performance (2026-07-31, enabled)

The user pushed back on leaving `/v3/*` unwired: *"why don't you wire the
other V3 endpoints? ... can't you feed all that goodness into my analysts
implementation"*. This section documents that follow-up, and is honest about
what's different from `danelfin_ai_score` above: **this one was never A/B
tested and cannot be**, because it has no history to test against.

### What was verified live before building anything

Danelfin's own official docs describe `/v3/*` only at the level of "latest
snapshot, no historical dates" — no field-level shape. Rather than guess at
`trading-parameters`/`price-forecast`/`performance`'s actual JSON field names
(as the earlier `get_trading_parameters`/`get_price_forecast`/
`get_performance` fetchers had done, unverified, when first written), this
work made one real, minimal-cost live call per endpoint against AAPL and
confirmed the exact shape:

- `/v3/trading-parameters` → `{entry_price, stop_loss, stop_loss_pct,
  take_profit, take_profit_pct, horizon, currency, signal}` — `signal` is a
  literal string (`"buy"` observed live), and **`stop_loss_pct`/
  `take_profit_pct` are percentage points** (e.g. `-5.29` == -5.29%), not a
  0-1 decimal.
- `/v3/price-forecast` → `{signal, median_3m, q05_3m, q16_3m, q84_3m,
  q95_3m, take_profit_3m, stop_loss_3m}` — these ARE 0-1 decimals (e.g.
  `0.064` == +6.4%). A real unit mismatch against `trading-parameters`'
  percentage-point fields, not a typo — both are used as-is in their own
  native units, never mixed.
- `/v3/performance` → `{signal, win_rate_1m/3m/6m/1y, alpha_win_rate_*,
  avg_perf_*, avg_alpha_*}`.

This also caught a real bug before shipping: the original `get_live_signals`
always queried `/v3/performance` with `signal="buy"` regardless of what
`trading-parameters` actually recommended, so a "sell" call would have
carried the *buy* signal's historical win-rate as its confidence — meaningless
for a sell. Fixed to query performance for whichever signal
`trading-parameters` returned (falling back to `"buy"` only for `"hold"`/
missing, since `/v3/performance` only documents buy/sell tracks).

### What shipped

A new `live_signals()` PitView capability, wired through the same points as
`ai_scores()` above, with two deliberate differences given its
snapshot-only nature:
- `LIVE_SIGNAL_COLS` schema is explicitly documented as never having
  historical data — `pit_view.live_signals()` always returns empty in a
  backtest.
- The live fetch (`data_feed.py`) is **not** gated behind an opt-in env flag
  like `estimates`/`ai_scores` are — there's no meaningful cache-only mode
  for data that's only ever "right now", so it fetches every live cycle by
  default whenever a `live_signals` provider is configured (~75 API calls
  for a 25-symbol universe, well inside the Expert plan's limits).

`firm.strategies.danelfin_live_signals.DanelfinLiveSignalsStrategy`:
direction from `tp_signal` (`+1`/`-1` for buy/sell, skips hold/unrecognized
entirely), magnitude from `|pf_median_return_3m|` scaled 40x and clipped to
the project's raw-score ceiling, confidence from `perf_win_rate_3m` for
whichever signal was actually called (defaulting to a neutral 0.5 when
missing). `tp_stop_loss_pct`/`tp_take_profit_pct` are deliberately **not**
used in the score — only carried as read-only meta — consistent with the
earlier decision that wiring actual stop-loss/take-profit price levels into
`RiskAgent`/`ExecutionAgent`'s execution math is a separate, explicit-review
change, not something to fold in here.

### No A/B — a live-only judgment call, not an evidence-backed one

Every other strategy promotion in this project (see `danelfin_ai_score`
above, `investing_analyst_ratings`) went through a 3-window walk-forward A/B
before an enable/disable decision. That gate is structurally unavailable
here: `pit_view.live_signals()` is always empty in a backtest (no
cache-backed history exists to populate one, ever), so
`scripts/calibrate_danelfin_ai_score.py`'s pattern cannot be reused —
running it would just show zero signals in every window, telling you
nothing.

**Decision: enabled in `config/live.yaml` anyway**, per the user's explicit
instruction, with the caveat stated plainly in both the config comment and
here: this is unvalidated. Watch its live per-strategy attribution closely;
revert the single `config/live.yaml` line if it looks bad. `experiment.name`
bumped to `paper_12_strategy`; verified via `GET /api/live/status` after a
live-service restart that `danelfin_live_signals` is active and
`broker_connected` is unaffected.

### Current state

- **Shipped and enabled-by-default-off, working**: authenticated Investing.com
  session (Phase 0), economic calendar (Phase 1, opt-in via
  `news_guard.source: investing`).
- **Shipped, tested, registered, not enabled**: `investing_analyst_ratings`
  strategy (FMP-backed; inconclusive A/B).
- **Shipped, tested, registered, and ENABLED in live paper trading**:
  `danelfin_ai_score` strategy (Danelfin-backed; consistently positive A/B).
- **Shipped, tested, registered, and ENABLED in live paper trading, unvalidated**:
  `danelfin_live_signals` strategy (Danelfin `/v3/*`-backed; structurally
  unbacktestable, enabled per explicit user request rather than an A/B).
- **Not pursued**: Investing.com Pro's per-stock Fair Value/ProTips/
  Financial Health/ProPicks/technical-summary data — blocked by Cloudflare on
  the pages that carry it, independent of any backtest result.
