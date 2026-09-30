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

## 2. Historical sanity check (2015-02 → 2026-09, as far back as we have
   good BTC data) — not a new edge claim, just "what would this exact
   portfolio have done"

| | CAGR | Volatility | Sharpe (above cash) | Worst drawdown |
|---|---|---|---|---|
| **This portfolio (92/8)** | **+23.2%** | 17.7% | **1.15** | **27.9%** |
| 60/40 alone (100%) | +8.9% | 10.3% | 0.67 | 21.2% |
| SPY alone | +14.2% | 17.5% | 0.72 | 33.7% |
| 92% core + 8% left in cash (no BTC) | +8.5% | 9.9% | 0.66 | 20.2% |
| BTC-trend sleeve alone, 100% notional | +49.9% | 32.2% | 1.35 | 49.0% |

Two things to read carefully in that table:

- The 92/8 blend looks much better than 60/40 alone or SPY alone. Most of that
  gap is BTC's decade of enormous returns, not something the rule invented —
  see the caveat in §4.
- **2020 and 2022 stress test:** the blend's worst drawdown in the COVID crash
  (2020) was 15.5%, and in the 2022 bond/stock selloff it was 19.1% — both
  *smaller* than 60/40 alone in 2022 (21.2%) and much smaller than SPY alone in
  2020 (33.7%). The worst single month was November 2022, −9.5%.

**A design choice that matters, found while building this:** if the 92%/8%
split between the two sleeves is never reset (each sleeve just compounds on
its own, which is how "sleeved" capital already works live on Alpaca), a
decade of BTC's returns lets the satellite balloon from 8% of the account to
**over three-quarters of it** by 2026 in the backtest. That is very likely not
what anyone intends. Two honest options, and this plan does not pick one for
you:

1. **As literally specified (no cross-sleeve rebalancing):** the +23.2%/1.15
   Sharpe row above. Realistic risk: the "8% satellite" can quietly become the
   dominant position over years.
2. **With the 92/8 split also reset every month (alongside the core's own
   rebalance):** CAGR +12.3%, Sharpe 1.00, worst drawdown 19.2% — still much
   better than 60/40 alone, and the satellite never dominates the account.

**Recommendation: pick option 2 (reset the 92/8 split monthly) unless you
want the satellite's share to float.** It's cheap (one more rebalance
decision, same monthly cadence as the core already has) and removes a
surprise. The forward test tracks the satellite's actual share either way and
flags it if it moves outside 4%–16% of NAV.

## 3. Kill-switch: the live 8% threshold is very likely too tight for this
   portfolio

Both live accounts halt trading if the account drawdown hits **8%**
(`kill_switch_drawdown` in `config/live.yaml` / `config/live_alpaca.yaml`). That
number was set with the old, much-lower-volatility equity strategy in mind.

Over the full 2015–2026 backtest, this portfolio crossed an 8% drawdown **11
separate times** (out of 45 distinct drawdown episodes of 2% or worse) — most
recently in 2024 (−27.9%, the worst on record, a 99-day round trip) and again
in a 2025–26 episode (−22.6%). Even under the more conservative "reset
monthly" version above, the worst drawdown (19.2%) still clears 8% several
times over.

**Recommendation: raise the kill-switch threshold for this instance before
going live**, or accept that the switch will trip on ordinary, expected
drawdowns rather than a real problem. Where you set it is a judgment call
about how much drawdown you're willing to sit through before wanting a human
to look — the backtest can only tell you what each choice would have caught:

- **~24–27%** would only have caught the single worst historical episode
  (2024, −27.9%, a 99-day round trip).
- **~20–22%** would catch the three deepest episodes (2024, 2017–18 at
  −23.5%, and 2025–26 at −22.6%), while still letting the account ride out
  the 2022 bond/equity selloff (−19.1%) and the 2020 COVID crash (−15.5%)
  without halting.
- **Below ~16%**, the switch would have tripped on comparatively ordinary
  drawdowns several times a decade (11 of 45 drawdown episodes ≥2% reached 8%
  or worse over 2015–2026).

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
