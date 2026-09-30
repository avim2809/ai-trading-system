# Forward paper-test plan: 92/8 core+satellite allocation portfolio

**For the owner. Plain language, no code required to read this.**

This is the plan for running the new portfolio on the Alpaca paper account
($100k) and watching it honestly. It is not a proposal to change anything on
the IBKR account, which keeps running the current 11-strategy system as the
control (unchanged, no config edited by this session).

Scripts: `scripts/allocation_portfolio_backtest.py` (historical sanity check,
already run), `scripts/allocation_forward_test_preregistered.py` (frozen rules
for the live test, not yet run — nothing has been deployed). Full numbers:
`docs/allocation_portfolio_backtest_2026_09.json`.

## 1. What the portfolio is

- **Core, 92% of the account**: 60% SPY / 40% IEF (bonds), reset back to that
  60/40 split on the first trading day of every month, and also any day the
  split drifts more than 2 percentage points from target.
- **Satellite, 8% of the account**: the Bitcoin trend rule from the edge
  search (`docs/edge_search_verdict_2026_09.md`) — hold BTC when it's been
  rising over the last 4 weeks, otherwise sit in cash. Sized down when BTC is
  volatile, reviewed once a week. This is the one lead that search found; it is
  **not proven**, only "not ruled out."

## 2. Historical sanity check (2015-02 → 2026-09)

This is not a new edge claim. It replays **the real allocator code**, the same
classes the live engine runs (`scripts/allocation_replay.py`,
`docs/allocation_replay_2026_09.json`), including:
- whole-share ETF orders;
- fractional BTC;
- the 2% drift band;
- the BTC sleeve's weekly review and its own 0.10 band;
- per-side costs.

| | Annual return | Volatility | Sharpe (above cash) | Worst drawdown |
|---|---|---|---|---|
| **This portfolio (92% 60/40 + 8% BTC trend, 1% cash reserve)** | **+12.0%** | 10.1% | **0.96** | **19.2%** (Oct 2022) |
| 60/40 alone (100%) | +8.7% | 10.4% | 0.65 | 21.0% |
| SPY alone | +14.2% | 17.5% | 0.72 | 33.7% |

Other figures from the replay:
- Trading is about 1× NAV a year, costing about 0.2% of NAV a year.
- The worst month was −6.9%.
- Calendar-year returns ranged from −15.9% (2022) to +26.9% (2019).

**Read carefully:** most of the gap over 60/40 is Bitcoin's enormous
2015–2026 run, not something the timing rule invented. Bitcoin is also
crypto's survivor, picked with hindsight. Don't expect this gap to repeat.

**The satellite stays near 8%.** Every target is a fixed share of *current*
total NAV. The BTC sleeve is re-sized at its weekly review whenever it's
more than 0.10 of its own sleeve away from target. An earlier draft of this
plan modelled the two sleeves as never rebalanced against each other, under
which BTC grew to about 78% of the account. That is not what the allocator
does, and it was corrected before deployment (see `REVISION` in the
pre-registration).

## 3. Kill switch: the live 8% threshold is too tight for this portfolio

Both live accounts halt trading at an **8%** drawdown (`kill_switch_drawdown`).
That was sized for the old, near-cash stock-picking book. In the replay:

| Drawdown threshold | Times it would have tripped, 2015–2026 |
|---|---|
| 8% | 4 (2022, 2020, 2018, early 2025) |
| 12% / 15% | 2 (2022 at about 19%, 2020 at about 19%) |
| 20% | 0 |
| 25% | 0 |

**Recommendation: 25%.** It rides out every drawdown in the sample, including
2020 and 2022, and still halts in something clearly worse than any of them.
You pick the final value when you sign off on the config.

## 4. What the forward test can and cannot show you

**It cannot tell you whether the BTC sleeve has a real edge**, and it isn't
designed to try. The backtested gap (BTC-trend vs. just holding BTC at the
same average exposure) is +0.28 Sharpe. Working out how much live data it
would take to tell that apart from noise:

- **At the historical correlation between the two strategies (0.73): about
  43 years.**
- **Being conservative and assuming no such correlation: about 158 years.**

Either way, **no forward-test duration anyone would actually run — 6 months,
1 year, even several years — can prove or disprove this edge.** That's stated
plainly in the frozen pre-registration
(`scripts/allocation_forward_test_preregistered.py`) so nobody reads an early
good or bad stretch as vindication or failure of the idea.

**What the forward test IS good for, starting from month 1:**
- Is the allocator actually doing what it's supposed to? (live account vs. a
  parallel calculation of what it should be doing, both from the same market
  data — tracking error, fill quality, trade counts, and a hard check that
  every order traces back to a logged decision.)
- Does the BTC sleeve behave safely in real trading? (its own drawdown, as a
  share of just its own capital, is watched against a 60% tripwire — well
  above the backtested 49% worst case, so it should only fire on something
  genuinely unusual.)
- Is the account's actual mix drifting the way §2 described?
- A running side-by-side with the IBKR control account (same reporting, not a
  race — neither side "winning" over months means anything statistically).

**Minimum run before the implementation checks are trusted:** about 6 months
— enough to see several monthly rebalances and around 25 weekly BTC reviews.
The 2%-drift-band trade is rarer (historically about once every 6 months), so
expect to see roughly zero or one of those in the first window, not several;
if you specifically want to verify that mechanism, extend monitoring until
one actually fires.

## 5. What would make us stop the BTC sleeve

- Any implementation check fails and can't be fixed same-cycle (the live
  account stops matching what it should be doing).
- The satellite's own drawdown passes 60% of its own capital without you
  signing off on continuing.
- A live parameter quietly drifts from what's written down here (caught by
  comparing the running config to the frozen file).

## 6. What would make us scale the BTC sleeve up

**Nothing from this forward test, by design** — see §4. A decision to run
more than 8% in BTC would need a new pre-registration with a much longer
planned commitment (or a much bigger claimed effect than +0.28 Sharpe), not a
good run of a few months on this one.

## 7. Where to find the numbers

- Backtest script + full JSON: `scripts/allocation_portfolio_backtest.py`,
  `docs/allocation_portfolio_backtest_2026_09.json`.
- Frozen forward-test rules: `scripts/allocation_forward_test_preregistered.py`
  (fingerprint printed by running the file directly).
- The original edge search this all comes from:
  `docs/edge_search_verdict_2026_09.md`.

**Nothing here has been deployed.** No `config/live*.yaml` file was touched,
no service was restarted, and the IBKR/Alpaca live APIs were not called. This
is the plan to review before anyone flips that switch.
