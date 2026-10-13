# Proposal: what, if anything, to research next (after core_v1 Tier D)

Status: PROPOSAL for owner review, 2026-10-13. Nothing here is built, approved or pre-registered. Written for the owner.

## Where we are

core_v1 (ETF trend + breakout, long/flat) ended Tier D (`research/reports/core_v1/FINISHER.md`). The plan's outcome is the passive book, and Alpaca already runs it live as the 92/8 allocation forward test; IBKR stays the blended control. Failure is a legitimate result under the mission statement.

Three audits since the verdict (w23a, w24a, w25a) explain *why* the candidate looked the way it did:

1. The candidate earned 2.25% a year in USD at 2.35% volatility (Sharpe 0.96 against zero) on 2005-2026. Not a bug: gate 7's fail is a real property.
2. The volatility target of 9% was delivered at 26% (2.35%). The decomposition closes exactly (computed, not estimated): the **gross cap of 1.0 against a bond-heavy risk-share weight scheme** removes 52% of the volatility (47% of the log shortfall); the **fixed 100,000 sizing capital while equity compounds** removes 38% (30%); the long/flat floor removes 22% (16%); the instrument cap another 10% (7%). Required gross at the target was 1.99x, against a 1.0 cap.
3. Gate 6 (42% of days at the gross cap) and the volatility shortfall are the same fact, which the owner had flagged in OD-17.

## What the burned data allows

Everything up to 2026-09-30 is in-sample and burned for this family of ideas. A successor design evaluated on the same history can only be a pre-registered re-test on data we have already looked at: it can reject, but a pass is weak evidence. The only clean evidence is forward (post-seal) data. Any proposal below has to be read against that.

## Options

**A. Stop active research here (recommended).** Keep the Alpaca 92/8 forward test and the IBKR control running, report anything that breaks, spend no further research budget. Rationale: the mission is a boring diversified core, not new alpha; the record is 0/11 legacy strategies with standalone edge plus this Tier D; and the live book already is the passive outcome with a forward test attached.

**B. A successor trend design, pre-registered, whose feasibility is proven before any run.** Only if the owner wants active trend exposure for a reason beyond performance (for example diversification versus the 60/40 core). Structural requirements, all fixed ex ante from published sources, none tuned on core_v1 results (see w25a section 3):
- a required-gross identity shown to be at most the gross cap before the first registered row (costs no trials);
- a weight scheme and cap structure that are mutually consistent (either allow gross above 1 through futures or margin, or drop sub-2%-volatility instruments, or define the target on the invested sleeve);
- sizing capital defined as lagged current equity;
- a volatility scale measured after the caps;
- an instrument cap that is feasible every day;
- IDM and FDM as functions of the active set;
- a delivery check (realised over target volatility on synthetic data) as a pre-run gate input.
It needs a new charter, a new gates file or an explicit amendment, and a new pre-registration with a fresh trial count. Gate 7 deserves care: it compares unhedged shekel NAVs, so a low-USD-volatility strategy is judged mostly on FX noise (the candidate's shekel Sharpe was about 0.01-0.09 against a USD Sharpe of 0.96). Changing a gate after seeing it fail is exactly what pre-registration forbids; it is legitimate only if fixed for a NEW family before that family runs, and stated openly as a deliberate difference.
Cost: roughly the same as core_v1 (charter, prereg, drivers, a multi-hour run). Expected value: low, for the reasons above.

**C. A forward-only test of the successor (paper, no backtest verdict).** Pre-register the successor, then judge it only on post-seal data, like the Alpaca allocation test. Cheapest in compute, slowest in time: a credible verdict needs one to two years.

**D. Turn to the live prerequisites instead of more research:** the tax adviser (OD-12, with the deep-research briefing), the US withholding assumption in the after-tax model (about 25% for the equity ETFs, not 0%), and whether Irish UCITS ETFs are an option. These gate any live step regardless of the research outcome.

## Recommendation

A, with D started in parallel. If B is ever wanted, do C and not another backtest, and write the feasibility identity into the charter first. I would not reuse the burned history to chase a variant of core_v1: the one-more-variant rule (AGENTS rule 4) applies, and the audits show the candidate's problem was structural feasibility, not a parameter.

## Decisions needed from the owner

1. Option A, B, C or D (they combine: A+D, or B+C).
2. If B or C: whether a successor may use futures or margin (the single choice that changes feasibility), and whether gate 7 stays an unhedged-shekel comparison for a new family.
