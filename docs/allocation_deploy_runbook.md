# Allocation portfolio: deploy runbook (Alpaca paper instance)

**Status: ready, awaiting owner sign-off. Nothing here has been run.**

## What changes

**The Alpaca paper instance** (`ai-trading-alpaca`, :8001, $~98k) moves from
the 11-strategy pipeline to `strategy_mode: allocation`:
- **Core, 92%:** 60% SPY / 40% IEF, rebalanced monthly, with a 2% drift band.
- **Satellite, 8%:** the pre-registered BTC 4-week trend rule. It's a forward
  paper test of an unproven (Tier C) rule.
- **Cash:** 1% of NAV is always kept in cash.

**The IBKR instance** (`ai-trading`, :8000) keeps the old system unchanged, as
the control.

**References:**
- Why: `docs/edge_search_verdict_2026_09.md`.
- How it will be judged: `scripts/allocation_forward_test_preregistered.py`,
  frozen at deploy, and `docs/allocation_forward_test_plan.md`.
- What it would have done 2015–2026: `docs/allocation_replay_2026_09.json`,
  which replays the real allocator code. It gave 12.0% a year, 10.1% vol,
  Sharpe 0.96 above cash and a 19.2% maximum drawdown. Most of the gap over
  60/40 comes from Bitcoin's 2015–2026 run.

## The exact config change (the only file to edit)

Copy the `strategy_mode:` + `allocation:` block from
`config/live_alpaca_allocation.example.yaml` into `config/live_alpaca.yaml`,
after `approval_mode: "full_auto"`. Nothing else in the file changes. Key values:

| Key | Value | Why |
|---|---|---|
| `strategy_mode` | `allocation` | Bypasses the strategy→analyst→PM→risk pipeline |
| sleeves | core 0.92 (SPY 0.6 / IEF 0.4, monthly); btc_trend 0.08 | The approved portfolio |
| `band_abs` | 0.02 | Daily drift trigger for the core |
| `cash_buffer` | 0.01 | Crypto needs settled cash; headroom for whole-share rounding |
| `max_gross` | 1.0 | Never lever |
| `liquidate_unmanaged` | true | Closes the old 21 positions (13 long, 8 short, about 11% gross) |
| `max_order_notional` | 70000 | Day-1 SPY buy is about 54% of NAV |
| `kill_switch_drawdown` | **0.25** | The replay's worst drawdown was 19.2%, and 8% would have tripped 4 times |
| `max_daily_turnover` / `max_daily_trades` | 3.0 / 80 | Lets the cut-over finish in one day |
| `max_attempts_per_day` | 3 | Same-day retries after a failed or rejected order |

## Steps (with the market closed, e.g. after 23:00 or before 16:00 IDT)

1. **Stop both services** (`systemctl stop ai-trading-alpaca ai-trading`).
   Merging changes tracked `src/` that both services import, so neither may be
   running during the merge.
2. **Cancel every open Alpaca order**, including protective stops from the old
   book. The allocator refuses to plan while any non-stop order is open, and
   old stops could lock shares it needs to sell.
3. **Merge:** fast-forward `main` to `feat/allocation-portfolio`, which contains
   `research/edge-search` (the split fix and results) plus allocation mode. Then
   `cd frontend && npm run build`.
4. **Apply the config block** above to `config/live_alpaca.yaml`.
5. **Kill-switch peak, owner's choice.** `data_alpaca/kill_switch_state.json`
   carries the old book's peak ($103,061). Keep it, so today's ~5% drawdown
   counts against the 25% limit, or reset it so the count starts from the
   new book.
6. **Start both services.** Verify:
   - `GET :8001/api/live/status` shows `strategy_mode: allocation` and the sleeves;
   - `GET :8000/api/live/status` shows `strategy_mode: pipeline`;
   - both brokers are connected;
   - there are no errors in the logs.
7. **First allocation cycle: the next 09:30 ET open.** Expect about 24 orders:
   - sell or cover the 21 old positions first;
   - then buy about 54.7% SPY and 36.4% IEF, plus BTC if the trend is on.

   Check fills, positions against targets, and the Discord alerts.
8. **Freeze the forward test.** Record `START_DATE` = the first allocation cycle,
   and the forward-test fingerprint, in `docs/allocation_forward_test_trial_history.json`.

## Safety behaviour added after two adversarial code reviews

- **Plans only while the market is open.** This holds even on a forced trigger,
  and a clock error fails closed.
- **Never plans while any non-stop order is still working** at the broker.
  Failing to read open orders also fails closed.
- **Every price is checked against the last completed close.** Tolerance is 10%,
  or 25% for BTC; anything outside is replaced by the close, and holdings and
  NAV are valued at the same checked prices.
- **A leverage backstop re-values the plan** at last-close prices before routing.
  Risk-reducing plans always pass. A traded symbol with no reference price
  fails closed.
- **A "submitting" marker is saved before orders go out,** so a crash
  mid-submission can't double-buy on restart.
- **Allocation mode requires `approval_mode: full_auto`.** It's enforced at
  engine start, in `PUT /api/live/config`, and on every cycle.
- **1% of NAV is always cash** (`cash_buffer`), because Alpaca crypto needs
  settled cash.

Regression tests: `tests/test_allocation_review_fixes.py` (20 tests, each
reproducing a reviewed failure scenario).

## Rollback

Remove the `strategy_mode` / `allocation` block and restart
`ai-trading-alpaca`. The pipeline's persisted sleeve books are left untouched,
but the account then holds the 60/40 positions, which the pipeline would treat
as its own and trade down. So rolling back also means flattening or re-seeding
by hand.

## Known limits (accepted, documented)

- **Market orders at the open.** A delayed start time would be a follow-up.
- **Stale pipeline data.** `/live/attribution` and the memory/reflection views
  show pipeline-era data in allocation mode.
- **The daily-budget trim re-sorts orders by size** (buys can go before sells).
  It isn't hit at 3.0/80.
- **A stuck order blocks planning.** Any non-stop order still open at the
  broker (e.g. a GTC crypto order that never fills) stops planning until it
  fills or is cancelled. A Discord alert fires: a warning on the first blocked
  trading day, then critical from the second day on.
- **Pre-existing pipeline bug, not fixed there.** The Alpaca adapter's
  `_mid_from_quote` accepts one-sided quotes on the pipeline path. Allocation
  mode now guards against it with a check against the last close.
