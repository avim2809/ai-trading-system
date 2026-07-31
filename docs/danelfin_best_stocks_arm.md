# Danelfin "Best Stocks Strategy" — a separate, synthetic paper-tracking arm

Status: **live, initialized 2026-07-31**, running on its own daily systemd timer.

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
  event, not a walk-forward historical replay; there's no way to
  backtest this arm's specific trajectory before today, since
  `/v3/trade-ideas` (like all `/v3/*` endpoints) has no historical dates.
