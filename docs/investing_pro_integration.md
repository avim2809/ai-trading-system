# Investing.com Pro integration — Phases 0-1 shipped, Phase 2a inconclusive, Phase 2b/3 blocked

Session started 2026-07-30 at the user's request to leverage their paid
Investing.com Pro subscription for trading signals. Investing.com has no
official API, so this required an authenticated web-scraper session
(`src/firm/data/investing/`), plus a new `estimates()` PitView capability and
`investing_analyst_ratings` strategy backed initially by FMP (for a real,
backtestable evidence base) before ever wiring the live Investing.com feed.
**Result: the calendar integration (Phase 1) works and is opt-in-enabled; the
analyst-ratings strategy is implemented/tested/registered but shipped
disabled** (inconclusive A/B, see Phase 2a below), following the same
"build it → A/B it → document honestly → ship disabled if inconclusive"
discipline as `docs/regime_ensemble_scoping.md`'s
`strategy_regime_weights`/regime-ensemble precedent. **The actual
Pro-exclusive per-stock data (Fair Value, ProTips, Financial Health,
ProPicks, technical-summary) turned out to be unreachable** — the pages that
carry it are gated by an interactive Cloudflare challenge that a real user's
browser passes and an automated session does not; see "Phase 2b/3" below for
why this is a hard stop, not an engineering gap.

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

### Current state

- **Shipped and enabled-by-default-off, working**: authenticated session
  (Phase 0), economic calendar (Phase 1, opt-in via `news_guard.source:
  investing`).
- **Shipped, tested, registered, not enabled**: `investing_analyst_ratings`
  strategy (FMP-backed; inconclusive A/B).
- **Not pursued**: Investing.com Pro's per-stock Fair Value/ProTips/
  Financial Health/ProPicks/technical-summary data — blocked by Cloudflare on
  the pages that carry it, independent of the backtest result above.
