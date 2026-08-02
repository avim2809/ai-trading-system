# Danelfin "Best Stocks Strategy" — a separate, synthetic paper-tracking arm

Status: **live, initialized 2026-07-31**, running on its own daily systemd
timer. **A full 2018-2026 walk-forward backtest (below) came back decisively
negative vs. SPY — real broker execution was deliberately NOT built out as a
result.** A follow-up check found this reconstruction only matches
Danelfin's own real live "Best Stocks" output ~25-30% (see "Important
caveat" below) — a structural gap (their "Buy Track Record" eligibility
filter has zero historical depth anywhere in the API), not a bug fixable
with more ranking-order tweaks. The backtest is therefore best read as
testing *a strategy inspired by* Danelfin's described rule, not a verified
reproduction of their actual algorithm — the negative direction and
magnitude are still informative, but shouldn't be read as "this is what
Danelfin's real product would have done." The forward-tracking synthetic
ledger keeps running as-is (cheap, already working, genuinely informative
to keep watching), but this is not being escalated to actual paper-account
order flow.

## Why this exists

The user shared Danelfin's own published methodology
(https://danelfin.com/best-stock-investment-strategy — Cloudflare-blocked,
never independently fetched; this is built from the user's own verbatim
summary) and asked, in their own words:

> "Given your AI trading system project, this could be a useful model to
> compare against your own ranking engine: rank stocks, apply liquidity/risk
> filters, enforce sector diversification, and rebalance periodically rather
> than relying on individual trade signals."

Presented with the choice of (a) folding this into the existing 12-strategy
universe as a blended signal, (b) building it as a genuinely separate
tracked arm, or (c) skipping it, the user picked **(b), explicitly**. This
doc covers that separate arm — it has nothing to do with
`danelfin_ai_score`/`danelfin_live_signals` (see
`docs/investing_pro_integration.md`), which are strategies *inside* the main
engine. This is a standalone, parallel comparison portfolio.

## Danelfin's stated rules (as summarized by the user)

1. Filter the full stock universe: **Proven Buy Signal** + **Low Risk score
   >= 5/10** + **average 3-month volume > 100,000 shares**.
2. **Rank sectors** by the average AI Score of *all* qualifying stocks in
   that sector.
3. Select the **top 5 sectors**.
4. Within each of those 5 sectors, select the **5 highest-AI-Score
   stocks** — 25 stocks total, **equal-weighted**.
5. Rebalance: every **3 months**, replace stocks that no longer meet the
   criteria; every **12 months**, rebalance back to equal weighting.

Danelfin's own backtest (Jan 2017-Jun 2025, 10 Monte Carlo simulations)
claims S&P 500 outperformance with smaller drawdowns, with an explicit
disclaimer that this is backtested, not live, performance. That's their
claim; this arm exists to form an independent, honest view by actually
running it.

## Design decision: a synthetic NAV ledger, not a second broker-connected engine

A research pass into this project's existing "LLM A/B" precedent (see
`docs/llm_ab_test_runbook.md`) found something important: it is **not** a
concurrent multi-arm setup. It's a *sequential* A/B — one
`LiveTradingEngine`, one IBKR paper account, one `data/live_state.db`, with
only `FIRM_LLM_CONFIG` swapped between restarts. There is no existing
"run two arms at once" infrastructure to reuse. Standing up a genuinely
concurrent second engine would require, at minimum:

- A second, distinct IBKR client ID (the main engine's `IBKRProvider` uses
  `client_id=2`; `scripts/run_live_trading.py` defaults to `1` via
  `IBKR_CLIENT_ID`).
- A second systemd unit / long-running process — `firm-api` only ever
  instantiates one `LiveTradingEngine` singleton on `app.state.live_engine`.
- A **separate** `LiveStateStore` db file — `LiveStateStore`'s schema has no
  experiment-name column (`INSERT OR REPLACE` on 3 fixed keys), so two
  engines sharing `data/live_state.db` would silently overwrite each other's
  NAV/attribution history.

That's real, separate production infrastructure — not something to stand up
silently as a side effect of a comparison exercise, especially since the
user's actual stated goal was **comparison**, not necessarily real order
execution. So this arm is implemented as a lightweight **synthetic
mark-to-market ledger** instead:

- Real market data drives everything (Danelfin's live `/v3/trade-ideas`
  screener for selection, this project's own `FallbackProvider` price chain
  for valuation).
- No broker order is ever placed. No IBKR connection, no client-ID
  collision risk, no new always-on trading process.
- Its own tiny JSON state file (`data/best_stocks_ledger.json`) — completely
  independent of `data/live_state.db`, so it cannot corrupt the main
  engine's NAV history.

If a live-executed version is ever wanted, this ledger's `holdings`/
`selection_meta` already contain everything needed to place real orders —
that would be a deliberate follow-up decision, not something this arm did
silently.

## API details verified live before building this (not guessed)

Danelfin's own official docs describe `/v3/trade-ideas` only at the level
of "a filterable screener" — no field-level shape, no confirmed filter
param names. Before writing the selection logic, this work made a handful
of real, minimal-cost live calls and found:

- **Response shape**: `{date_str: {symbol: {...fields...}}}` — the *same*
  shape as `/ranking` and `/v3/beststocks`. The original
  `DanelfinProvider.get_trade_ideas` implementation had assumed a
  `{"items": [...]}` shape (an unverified guess made when `/v3/*` was first
  exposed) and **silently always returned empty** — a real, previously
  undetected bug, fixed as part of this work.
- **A second real bug**, also caught live: the top-level response carries
  sibling `total`/`limit`/`offset` **int** keys alongside the date key
  (unlike `/ranking`, which only ever has date keys) — the first fix still
  crashed with `AttributeError: 'int' object has no attribute 'items'`
  until the parser was taught to skip non-dict sibling keys.
- **Filter params, confirmed live**: `aiscore=N` and `low_risk=N` are
  **minimum-threshold** filters (e.g. `low_risk=5` returns 5/6/7, not just
  exact 5s — despite the singular parameter name suggesting exact match).
  `average_volume_3m=N` is likewise a minimum (confirmed: raising it from
  none to `50,000,000` cut a 23-candidate information-technology sample down
  to 3, all above that floor). `sector=<kebab-case>` works (confirmed
  values: `communication-services`, `consumer-staples`, `energy`,
  `health-care`, `industrials`, `information-technology`, `materials` —
  observed directly; `financials`, `consumer-discretionary`, `real-estate`,
  `utilities` follow the same convention but weren't individually
  fetch-tested, though they are used live in the selector as of this run
  and are the sectors that were actually selected — see below).
- **Hard limits confirmed**: `limit` caps at 100 (HTTP 400 above that);
  `page` is rejected as an unknown parameter — **no pagination**, unlike
  `/ranking`. A full-market sweep therefore needs one call per `sector`
  filter (11 calls), not one call with a high limit.
- **No `signal` filter param** (confirmed rejected as unknown) — there is no
  direct way to filter `/v3/trade-ideas` on buy/hold/sell. Three sampled
  trade-idea symbols were each independently cross-checked against
  `/v3/trading-parameters` and all three returned `signal: "buy"` — small-N
  evidence, not an exhaustive guarantee, that this endpoint's own purpose
  ("trade ideas") already implies a buy call. This stands in for Danelfin's
  "Proven Buy Signal" filter criterion.

See `firm.data.providers.danelfin.DanelfinProvider.get_trade_ideas`'s
docstring for the same detail closer to the code, and
`tests/test_danelfin_provider.py::TestDanelfinTradeIdeas` for regression
coverage of both bugs.

## What shipped

- `src/firm/live/best_stocks_arm.py` — `select_best_stocks()`: one
  `/v3/trade-ideas` call per sector (pre-filtered server-side on
  `low_risk`/`average_volume_3m`), sectors need >= 5 qualifying candidates
  to be eligible at all (otherwise they can't fill their slots), ranks
  eligible sectors by mean `aiscore` across *all* their qualifying
  candidates (per Danelfin's stated rule — not just the top 5), keeps the
  top 5, and within each takes the 5 highest-`aiscore` names (tie-broken on
  `win_rate_3m`).
- `src/firm/live/best_stocks_ledger.py` — `BestStocksLedger`: a
  `dataclass` with `holdings`/`cash`/`nav_history`/rebalance-date fields,
  JSON-persisted. `full_rebalance()` (liquidate + equal-weight redistribute
  — used for the initial construction), `quarterly_replace()` (swap out
  holdings no longer in a freshly re-run selection, redistribute freed
  dollars across new additions — a defensible-but-not-uniquely-specified
  reading of Danelfin's natural-language "replace stocks that no longer
  meet the criteria" rule, documented as such in the code), and
  `annual_rebalance()` (reset current holdings back to equal dollar
  weighting **without** changing symbols — distinct from a quarterly
  replace). `due_for_quarterly_replace()`/`due_for_annual_rebalance()` gate
  on 91/365 elapsed days from the relevant last-event date.
- `scripts/run_best_stocks_arm.py` — the daily driver: initializes on first
  run (full rebalance from a fresh selection), otherwise just marks
  existing holdings to market (zero Danelfin API calls on a normal day),
  checking whether a quarterly replace or annual rebalance is due first.
- `deploy/best-stocks-arm.service` + `deploy/best-stocks-arm.timer` —
  systemd oneshot + calendar timer, `OnCalendar=*-*-* 22:00:00 UTC`
  (~6pm US/Eastern, safely after the 4pm ET close). Installed and enabled
  on this host (`systemctl enable --now best-stocks-arm.timer`) —
  **the explicit `UTC` suffix matters**: this server's local timezone is
  `Asia/Jerusalem` (confirmed via `timedatectl`), and systemd's
  `OnCalendar` defaults to local time without it — an earlier version of
  this timer (before the UTC suffix was added) would have fired ~3 hours
  too early, before the US market even closes.
- `tests/test_best_stocks_arm.py` (14 tests, pure logic, no network) +
  `tests/test_danelfin_provider.py`'s new `TestDanelfinTradeIdeas` class (3
  tests, covering both real bugs found above).

## Current state (2026-07-31 initialization)

Ran `scripts/run_best_stocks_arm.py --initial-capital 100000` live. Top 5
sectors selected today (by mean AI Score among qualifying candidates):
**materials, energy, financials, industrials, utilities** — notably no
technology or healthcare sector made the cut on this particular day, which
is a real, if perhaps counterintuitive, output of the mechanical ranking
rule (not a bug — worth remembering when eyeballing later runs).

25 holdings, $100,000 initial synthetic capital, equal-weighted
(~$4,000/position). State: `data/best_stocks_ledger.json`. First NAV
snapshot: $100,000 (by construction, at cost basis on day 1).

**Update 2026-07-31 (same day, later):** a full 2018-2026 walk-forward
backtest of the underlying methodology came back decisively negative vs.
SPY (Sharpe 0.276 vs 0.706, total return +27.5% vs +169.3% — see "Walk-forward
backtest" below). The synthetic ledger above keeps running daily regardless
(cheap, informative), but real broker execution was NOT built out as a
result — see "Decision: real broker execution NOT built out".

## How to monitor

No dedicated UI/API endpoint exists yet (consistent with this project's
existing precedent — the LLM A/B arm is also monitored via a snapshot
script + manual log, not a UI page). To check current state:

```bash
python3 -c "
import json
d = json.load(open('data/best_stocks_ledger.json'))
print('nav_history:', d['nav_history'][-5:])
print('holdings:', d['holdings'])
"
```

To force an off-schedule run (e.g. to sanity-check after a code change):

```bash
python scripts/run_best_stocks_arm.py
```

To check the timer's schedule/last-run status:

```bash
systemctl list-timers best-stocks-arm.timer
journalctl -u best-stocks-arm.service --since "1 day ago"
```

## Real IBKR execution (partially built 2026-07-31, paused — see backtest below)

The user's explicit follow-up goal was to "hook it into trade" — actually
place real IBKR paper orders for this arm, not just track a synthetic
NAV. Before writing any execution code, the actual collision risk was
checked: `IBKRBroker.get_positions()`/`submit_order()` (in
`src/firm/brokers/ibkr.py`) have no account/model-code tagging at all —
IBKR nets all fills into one account-level position regardless of which
client_id submitted them. Running this arm's real orders through the same
account as the main engine would mean the two could silently unwind each
other's positions on any overlapping symbol.

**Chosen approach (user's explicit call, via options presented): same
account, with a symbol-collision guard**, not a separate IBKR account.
Built:

- `src/firm/live/best_stocks_execution.py` — `main_engine_excluded_symbols()`,
  the union of the main engine's static `config/live.yaml` universe and its
  live in-memory universe (via `GET /api/live/config`, since a runtime
  `PUT /api/live/config` universe edit wouldn't be reflected in the YAML
  file alone). This arm must never trade a symbol in this set.
- `select_best_stocks`'s `excluded_symbols` param — colliding candidates are
  dropped from each sector's pool BEFORE ranking, so a collision never
  silently shrinks the final 25-name portfolio if a non-colliding
  alternative exists.
- `BestStocksLedger.rebalance_via_broker()` — real whole-share IBKR orders
  (`full`/`quarterly`/`annual` variants), re-checking the collision guard
  fresh at order time (defense in depth against the main engine's universe
  changing between selection and execution), holding untouched any
  already-held symbol that becomes a collision after the fact rather than
  force-liquidating it.
- `scripts/run_best_stocks_arm.py --live-trading` — connects
  `IBKRBroker` on a distinct `client_id` (default 3; main engine uses 1
  for its broker connection and 2 for its data feed — see
  `.cursor/rules/ibkr-integration.mdc`).

**This was paused, untested end-to-end, once the walk-forward backtest
below came back decisively negative** — see "Decision: real broker
execution NOT built out". The code above is real and importable but was
never exercised against a live IBKR connection, never added to any
systemd unit, and is not part of any deployment.

## Known limitations / honest caveats

- **No A/B, by nature.** This isn't a strategy inside the backtestable
  engine — it's a live-only mechanical process compared against the main
  book's live NAV over time, the same way the LLM A/B arms are compared:
  by eyeballing NAV history, not a formal statistical test.
- **Quarterly-replace mechanics are a specific interpretation**, not a
  literal implementation of an unpublished symbol-level Danelfin
  algorithm — see `quarterly_replace()`'s docstring.
- **Sector value coverage is only partially verified live** (7 of 11) — see
  the API-details section above.
- **The "Proven Buy Signal" filter is inferred, not directly filterable** —
  `/v3/trade-ideas` has no `signal` param; this relies on the endpoint's
  own implicit buy-only purpose, spot-checked on 3 symbols.
- **Single point-in-time selection per rebalance event** — like Danelfin's
  own screener, this reflects "today's" scores at each rebalance/replace
  event, not a walk-forward historical replay of the LIVE arm's exact
  `/v3/trade-ideas`-based rules (those have no historical depth, ever).
  **Update:** a *separate* code path (`select_best_stocks_historical`,
  `scripts/backtest_best_stocks_arm.py`) CAN walk-forward test the
  underlying methodology via a different Danelfin endpoint mode — see
  the "Walk-forward backtest" section below. The live arm itself is
  unchanged by this; it's a genuinely separate reconstruction with its
  own disclosed simplifications, not a backtest of this exact code path.

## Walk-forward backtest (2026-07-31) — decisively negative vs. SPY

While building real order execution for this arm (see "Real IBKR
execution" below — since paused as a direct result of this section), the
user pointed out Danelfin's `/ranking` endpoint (already used for the
genuinely-historical `danelfin_ai_score` strategy) also supports a **bulk
historical mode**: `date`+`sector` with no `ticker` returns every matching
symbol for that historical date, not one ticker's timeline. That means the
Best-Stocks sector-ranking methodology CAN be honestly walk-forward
tested — unlike the live arm's `/v3/trade-ideas`, which is genuinely
snapshot-only.

### What was verified live before building on it

- `/ranking?date=2024-06-03&sector=information-technology&low_risk=5` (no
  `ticker`) returns every matching symbol for that exact historical date —
  confirmed with real dated scores (AMD, APPN, ARM, ...).
- Unlike `/v3/trade-ideas`'s minimum-threshold filters, this bulk mode's
  `low_risk`/`aiscore` filters are **exact match** (confirmed: querying
  information-technology/2024-06-03 across low_risk 5..10 individually
  returned real non-empty results for 5/6/7/9 and a 404 for 8/10 — no
  stocks had exactly that score that day). "Low_risk >= 5" therefore needs
  one call per exact value 5-10, unioned locally.
- **A 404 here means "zero rows match this exact combination"** — NOT
  "invalid date". An earlier version of this work got that wrong (based
  on a single observation that happened to coincide with a market
  holiday) and built a whole date-revalidation step around the wrong
  assumption, which silently rejected perfectly valid trading dates
  whenever a narrow probe query legitimately had zero matches. Fixed by
  removing the Danelfin-based date probe entirely and resolving rebalance
  dates from a real local price series (SPY) instead.
- A single call still caps at 100 rows, but (unlike `/v3/trade-ideas`)
  `page=N` **does** paginate past that cap here — the opposite
  pagination-support split between the two endpoints.
- There is **no historical equivalent of "Proven Buy Signal"** anywhere in
  `/ranking` — only low_risk/aiscore/sector can be reconstructed
  historically.

### A second real bug, found the same way: benchmark price truncation

The first full run silently corrupted itself: 7 of 9 candidate rebalance
dates (2018-2024) all resolved to the *same* date, because the live SPY
price fetch was silently truncated to the last ~2 years (Massive's own
tier limit, with Tiingo/FMP too rate-limited during this session to fill
the gap) — and the date-resolution logic picked the earliest available
row in that truncated series for every earlier target, without erroring.
Fixed two ways: (1) `PriceCache` now checks this project's own on-disk
parquet cache first (real SPY history back to 2010 already sat there,
unused, from earlier backfill work) before ever hitting the live provider
chain; (2) the backtest script now hard-fails with a clear error if it
ever resolves duplicate rebalance dates again, instead of silently
running a corrupted window.

### Disclosed methodology simplifications (not silent deviations)

1. **No "Proven Buy Signal" filter** — no historical equivalent exists;
   uses low_risk>=5 + aiscore ranking only.
2. **Annual rebalancing, not quarterly replace + annual reweight** — a
   full quarterly walk-forward would need on the order of tens of
   thousands of Danelfin API calls; annual (~9 events over 8.5 years) is
   Danelfin's own stated "reweight" cadence and a reasonable first pass.
3. **No real historical liquidity (>100k volume) filter by default** — a
   single rebalance date's low_risk-qualifying candidate pool spans
   hundreds to 500+ symbols per sector; fetching real historical volume
   for all of them hit the same real rate-limit wall described above.
   Sector ranking uses the full low_risk-qualifying pool (unfiltered by
   volume); `--check-volume` exists as a slow opt-in that bounds the
   check to only the names actually being selected, not the whole pool.

### Results: 2018-01-16 to 2026-07-01, 9 annual rebalance events, 25 holdings each

| | Sharpe | CAGR | Max Drawdown | Total Return |
|---|---|---|---|---|
| **Best-Stocks (this reconstruction)** | 0.276 | 3.8% | -43.0% | +27.5% |
| **SPY (benchmark)** | 0.706 | 12.5% | -34.1% | **+169.3%** |

Every one of the 9 rebalance events filled all 25 slots across 5 real
sectors (no underfilled/ineligible sectors) — this is a complete,
clean-run result, not a partial one. Full per-date selections in
`data/best_stocks_backtest_full.json` (gitignored — a data artifact, not
source).

**This is the opposite of Danelfin's own claimed outperformance** (their
marketing: S&P 500 outperformance with smaller drawdowns, Jan
2017-Jun 2025, via Monte Carlo simulation — their claim, never
independently verified here). Even generously accounting for this
reconstruction's disclosed simplifications (no buy-signal filter, no
volume filter, annual not quarterly), the gap is not close: SPY returned
roughly **6x** more, with a *smaller* drawdown, over the same window.
Plausible contributors, not excuses: annual (not quarterly) rebalancing
lets underperforming picks ride a full year before replacement; the
missing volume filter may let in thinly-traded, higher-volatility names
that drag down risk-adjusted return; 2018-2026 was an unusually strong,
narrow-leadership period for US large caps (SPY) that a
sector-rotating/lower-market-cap strategy would generally struggle to
keep pace with regardless of stock-picking skill.

### Important caveat found AFTER the backtest: only ~25-30% match with Danelfin's real live selection

Prompted by a direct question about whether the historical reconstruction
actually represents the same thing as Danelfin's real "Best Stocks"
product: it doesn't, closely. Fetched Danelfin's own live
`GET /v3/beststocks` (their actual curated Top-25) for the same day and
compared it against this reconstruction's live output for that day:
**only 6-7 of 25 symbols overlapped (24-28%)**, and even the *sectors*
selected only overlapped 3 of 5.

Investigating why (via `/ranking`'s per-symbol mode, `/v3/trading-parameters`,
and Danelfin's own help-center article — https://danelfin.com/best-stock-investment-strategy
and https://danelfin.com/docs/api are Cloudflare-blocked, as established
earlier this session, but a support-center article was reachable via web
search):

- The confirmed rule text: sectors ARE ranked by "the average AI Score of
  their **eligible** stocks" (matching what was built) — but "eligible"
  means Buy Track Record + low_risk>=5 + volume>100k, and ties are broken
  by "highest AI Score, prioritizing the ones with a Buy Track Record, and
  a Low Risk score 6/10 or above."
- **`/v3/trade-ideas` (the live arm's data source) is not an exhaustive
  screener at all** — confirmed live: querying `sector=real-estate` with
  *zero* filters returned only 3 symbols (AHR, CTRE, VTR), permanently
  excluding SPG and SKT even though both have `signal: "buy"`,
  `low_risk: 6`, `aiscore: 7` per `/ranking`/`/v3/trading-parameters` and
  ARE in Danelfin's real Top-25. It's a curated Danelfin subset, not a
  complete filter-based scan.
- **"Buy Track Record" is a real *eligibility filter* on the sector
  average, not just a stock-level tie-break** — and it has **zero
  historical depth anywhere in Danelfin's API** (confirmed earlier:
  `/v3/trading-parameters`'s buy/hold/sell call is snapshot-only, no
  historical dates, ever). That means the sector-ranking mechanism
  itself — not just tie-breaking within a sector — cannot be faithfully
  reconstructed for any past date. This reconstruction's sector averages
  are computed over a plausibly broader pool than Danelfin's real
  eligible set (since a buy-track-record filter can't be applied
  historically), which is a real, structural, not effort-fixable gap
  given the public API's actual surface — not a bug to keep chasing with
  more ranking-order tweaks.
- Tried one additional tie-break variant (low_risk as a secondary sort
  key) as a cheap check — it barely moved the live-day overlap (4-7/25),
  confirming the mismatch is mostly upstream at sector selection, not
  stock-level tie-breaking.

**What this means for the results above**: this is not a verified backtest
of Danelfin's actual proprietary "Best Stocks" algorithm — it's a backtest
of *a sector-rotation strategy inspired by their publicly described rule*,
built from the most complete data this project could get to reconstruct
it, which happens to diverge substantially (~70-75%) from their real
live output on stock/sector selection specifics. The magnitude of the
underperformance above (SPY returning ~6x more) is large enough that the
directional finding — this class of annual sector-rotation strategy
lagged a concentrated, mega-cap-led market badly over 2018-2026 — is
probably still informative in a general sense (see the NVDA-rotation
example in the session transcript: the reconstruction held NVDA/AVGO/ADBE
for exactly one year in 2018 then rotated into utilities/staples for
multiple years, missing the subsequent tech run). But the specific
Sharpe/CAGR numbers should not be read as "this is what Danelfin's real
product would have returned" — treat them as evidence about the general
methodology class, not a validated reproduction of Danelfin's own
algorithm.

### Decision: real broker execution NOT built out

Mid-session, real IBKR paper-order execution for this arm was partially
built (`BestStocksLedger.rebalance_via_broker`, a `--live-trading` flag on
`scripts/run_best_stocks_arm.py`, a shared-account collision guard in
`firm.live.best_stocks_execution`) before this backtest existed to inform
whether it was worth finishing. Given the decisively negative result
above, that work was **deliberately left unfinished and not wired into
any deployment** — building out, testing, and running real order
execution for a methodology that just failed its own backtest would
contradict this project's own promotion discipline (build → evidence →
honest documentation → enable only on positive evidence). The partially-built
code remains in the codebase (`rebalance_via_broker`, the collision guard,
the `--live-trading` flag) in case a future revisit — a different rebalance
cadence, an added volume/buy-signal filter, or a different rule set
entirely — produces a more promising backtest; it is not deleted, just
not finished or enabled. The synthetic paper-tracking ledger
(`scripts/run_best_stocks_arm.py`'s default mode, no `--live-trading`)
continues running on its daily timer regardless — it's cheap, already
working, and remains a genuinely informative forward-tracking comparison
even though real execution isn't happening.

## A separate, distinct thing: `danelfin_best_stocks_signal` main-engine strategy (2026-08-02)

Do not confuse this with the synthetic arm above. At the user's explicit
request ("I still want to use best stocks, not a backtested strategy but as
a very strong signal to buy"), a new strategy —
`src/firm/strategies/danelfin_best_stocks_signal.py`, registered in the main
strategy registry — was added to the **main portfolio engine** (enabled in
`config/live.yaml`). It reads Danelfin's real `/v3/beststocks` Top-25 list
directly via a new `PitView.best_stocks()` capability (mirrors
`live_signals()`'s wiring exactly: `pit_store.py`, `PitViewAdapter` in
backtests — always empty there, by construction — `LivePitViewAdapter` in
live trading) and emits a strong bullish signal for any universe symbol
present in it, with no signal (not bearish) for absent symbols. This is
completely independent of the synthetic ledger/arm above:

- The **arm** (this doc, above) is a side-channel paper-tracking experiment
  with its own NAV, not connected to the main portfolio at all.
- **`danelfin_best_stocks_signal`** is a real strategy inside the main
  12-strategy pipeline, contributing an actual signal to the actual
  portfolio construction (bull/bear debate → PM → risk → execution), same
  as `danelfin_live_signals`.

Same "cannot be backtested" structural caveat applies (`/v3/beststocks` has
no historical `date` param, confirmed live) — this is a live-only,
unvalidated judgment call, not evidence-backed like `danelfin_ai_score`.
Expected to rarely fire against the current fixed ~25-name US-mega-cap
universe, since Danelfin's real Top-25 skews toward smaller/rotating names
— a general-purpose dynamic-universe-growth mechanism (not Danelfin-specific
in its own persistence API) to give it more surface was built next (2026-08-02,
see `firm.live.danelfin_universe_sync` and `config/live.yaml`'s
`danelfin_dynamic_universe:` block — disabled by default, opt-in only).
