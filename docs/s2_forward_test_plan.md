# S2 breadth overlay: shadow forward test

**Status:** running since 2026-10-01. Paper ledger only: it places no orders and changes no live config.

| | |
|---|---|
| Frozen design | `scripts/s2_forward_preregistered.py`, fingerprint `cbd6ad32…`, commit `a5142f2` (08:04:25Z) |
| Rule | the frozen S2 V1 variant (`docs/eodhd_shortlist_verdict_2026_10.md`): breadth < 50% on the first trading day of the month → SPY 30 / IEF 70, else 60/40 |
| Tracker | `scripts/s2_forward_shadow.py` (`run`, `validate`, `report`, `export`); tests `tests/test_s2_forward_shadow.py` |
| Ledger (private) | `data/research/s2_forward/` (`decisions.jsonl`, `nav.csv`, `status.json`) |
| Snapshot in git | `docs/s2_forward_snapshot.json` (refresh with `export`) |
| Launch validation | `docs/s2_forward_launch_validation_2026_10_01.json`: live pipeline vs the frozen builder on 11 past dates, max difference 0.42 pp, all pass |
| Scheduling | `deploy/s2-forward-shadow.{service,timer}` (daily 07:00 UTC). **Not installed**: needs the owner's go-ahead; until then run `scripts/s2_forward_shadow.py run` by hand each day |

## First check (2026-10-01, signal as of the 2026-09-30 close, recorded 08:23Z before the open)

Breadth **47.8%** over 3,174 eligible stocks → **cut state** (shadow SPY 30% / IEF 70%). The stricter V3 variant (40%) stays at plain 60/40. The previous reading, 49.2% on 9/29, was also just under the line, so the rule sits close to its threshold.

## What it can and cannot show

V1 changes state about 0.8 times a year. If the future resembles the past, 24 months has roughly a 45% chance of holding the two completed episodes the decision rule needs; otherwise the answer is "continue" (up to 60 months). The test confirms that the live signal matches the historical one and that the overlay behaves as modelled. It cannot establish an edge on its own.

## Decision rule (frozen)

Nothing is decided before 24 months. A live proposal needs ≥ 2 completed episodes, maxDD and Calmar better than plain 60/40, and a Sharpe gap ≥ −0.05, with ≤ 1 missed check. Rejection: ≥ 2 episodes and (maxDD worse or gap < −0.05). Putting the overlay live also needs allocator support for a signal-driven weight change, and the owner's sign-off on the exact config diff.

## Data dependency

The monthly check pulls ~6,100 stocks from EODHD (a one-month subscription at freeze time). If it lapses, checks are logged as missed, never silently replaced. A replacement source needs a data-equivalence validation like the launch one before it is used.
