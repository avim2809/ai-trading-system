"""DRAFT pre-registration for the S2 candidate: market-breadth overlay on the
live 60/40 core (docs/eodhd_shortlist_protocol_2026_10.md; the candidate is
docs/research_brief_eodhd_findings.md Sec.1 #2).

STATUS: DRAFT. PREREGISTERED_AT = None. This file may still be edited: it has
NOT been frozen, because no candidate, benchmark or placebo RETURN has been
computed yet -- only data AVAILABILITY (tickers present, date coverage,
eligible-stock counts per day) and the breadth SIGNAL itself (built by
scripts/eodhd_breadth.py, which never touches a return series). Freezing
(setting DRAFT = False and PREREGISTERED_AT) happens only after the owner
reviews this file's design choices; nothing below may be reshaped once any
return exists, per the protocol's own process rule.

Important caveat carried over from the research brief (Sec.1 #2): the cited
evidence -- "Herding for profits: Market breadth and the cross-section of
global equity returns" (Int. Rev. Fin. Analysis, 64 countries, 1973-2018) --
tests market breadth as a DIRECT RETURN PREDICTOR (a cross-sectional
signal), not as a DRAWDOWN/REGIME OVERLAY on a separate, already-fixed core
portfolio. This pre-registration tests the overlay framing specifically;
that framing is the open empirical question, not an established result, and
this file is written on that understanding.

Question: does cutting the live core's SPY weight when market breadth is
weak (and raising the destination asset's weight by the same amount),
checked at the core's own monthly rebalance, beat the plain 60/40 core (BM2)
on a DEFENSIVE basis (max drawdown, Calmar) without giving up much Sharpe,
net of costs, after deflating for this search's trial count? Per the shared
protocol, S2 is judged on tier bar A8 (drawdown/Calmar), not on raw Sharpe
alone -- but A1 (the generic Sharpe-gap bar) still gates Tier A, so a
purely-defensive, flat-or-negative-return overlay cannot reach Tier A under
this protocol even if A8 passes cleanly; it could still land in Tier B. That
tension is inherent to the frozen protocol (this file does not relax it) and
is flagged here for the reviewer, not resolved.

Design inputs below (OBSERVED, WINDOW) are real, computed 2026-09-30 from
local EODHD data ONLY (data/research/eodhd/us_universe_full + etfs_full +
FRED DTB3), via scripts/eodhd_breadth.py:build_breadth. No forward return,
and no relationship between breadth and any future return, was looked at to
produce any number in this file -- see that module's own docstring for the
exact screen and measures.

AMENDMENT 1 (protocol 2026-09-30 22:40Z, adopted here before this file froze
or any return was computed): switches the universe/ETF source from the
2005-start `us_universe`/`etfs` folders (superseded) to the full-history
`us_universe_full`/`etfs_full` folders (EODHD's real inception dates, back to
1985 where available); moves the cleaning rule to v2 (adds a `nav` asset type
for no-volume mutual-fund series; CLEANING_FINGERPRINT updated below); adds a
VFITX NAV proxy for BM2's bond leg before IEF existed (2002-07-26) and a FRED
DTB3 cash proxy before BIL existed (2007-05-30). The equity exchange calendar
is now SPY's full etfs_full history (from 1993-01-29), so no window may start
before 1993-02-01 (protocol Sec.3) -- see WINDOW below for where the real
binding constraint (eligible-stock-count stability) actually falls.

WINDOW START, revisited after amendment 1: an exact per-file scan of all
22,732 us_universe_full tickers (2026-09-30, availability only) found that
raw data coverage is sparse and survivor-tilted before ~1998 (1,885 tickers
have any 1995 bar, matching an independent sampling estimate of ~1,900; a
mechanical ~3x jump to 5,828 between 1996 and 1997 is a vendor historical-
depth artifact, not a real change in the listed-stock population) and does
not settle into a level consistent with the real listed-stock universe until
2002 (raw coverage is 6,855-7,253 every year from 2002 through 2010, a tight
~3% band, versus a rising-then-falling 2,058 -> 8,303 -> 7,041 path from 1996
through 2002). WINDOW['start'] = 2002-01-02 is chosen on that basis alone --
see WINDOW['start_rationale'] and OBSERVED['raw_ticker_file_coverage_by_year']
for the full by-year table and reasoning.
"""

from __future__ import annotations

import hashlib
import json

DRAFT = False
PREREGISTERED_AT = "2026-09-30T19:18:54Z"  # set only at freeze, after owner sign-off; see module docstring
DATA_END = "2026-09-29"  # last EODHD session in the local pull, all series (manifest.json / per-file scan)
TRADING_DAYS = 252
SEED = 20260930

CLEANING_FINGERPRINT = "fc0690f087edaac8c11ddf77b381e58b59c57a893d460f12c7fa782229693054"  # eodhd_clean.cleaning_fingerprint(), v2

# ---------------------------------------------------------------------------
# Observed inputs (real, from scripts/eodhd_breadth.py:build_breadth run
# 2026-09-30 over the full data/research/eodhd/us_universe_full universe --
# active AND delisted). Availability only: eligible counts and coverage,
# never a return. See WINDOW below for how these numbers fix the evaluation
# start date.
# ---------------------------------------------------------------------------
OBSERVED = {
    "n_ticker_files_scanned": 22_732,
    "n_ticker_files_usable": 22_624,          # passed eodhd_clean.clean_bars with >=1 kept bar
    "eligible_count_first_nonzero_date": "1993-11-11",
    "eligible_count_min_nonzero": 352,        # full history (1993-2026), any day with eligible_count > 0
    "eligible_count_max": 3_726,              # full history
    "eligible_count_median_full_history": 2_880,
    "eligible_count_median_post_window_start": 3_070,  # on/after WINDOW['start'], see below
    "eligible_count_at_window_start": 2_155,  # 2002-01-02
    "eligible_count_at_data_end": 3_182,       # 2026-09-29
    "n_months_in_window": 297,                 # 2002-01 .. 2026-09 inclusive
    "n_state_changes_primary_variant": 38,     # V1_primary on/off flips at the monthly rebalance cadence
    "n_state_changes_by_variant": {            # see POWER_NOTES: episode_count() applied to each variant's
        "V1_primary": {"n_flips": 38, "n_on_episodes": 19, "pct_on": 0.256},           # real monthly state,
        "V2_cash_destination": {"n_flips": 38, "n_on_episodes": 19, "pct_on": 0.256},  # computed 2026-09-30
        "V3_stricter_threshold": {"n_flips": 30, "n_on_episodes": 15, "pct_on": 0.162},
        "V4_alt_measure": {"n_flips": 146, "n_on_episodes": 73, "pct_on": 0.380},
    },
    "pct_of_months_overlay_on_primary": 0.256,
    # Coverage-by-year tables (raw file presence, and this module's OWN screened
    # eligible_count), computed 2026-09-30 from us_universe_full -- these are what
    # WINDOW['start'] is derived from; see WINDOW['start_rationale'].
    "raw_ticker_file_coverage_by_year": {
        # any us_universe_full ticker file with >=1 bar dated in that calendar year
        # (no price/ADV/SMA screen applied -- pure data-presence count)
        1993: 1509, 1994: 1686, 1995: 1885, 1996: 2058, 1997: 5828, 1998: 6225,
        1999: 8303, 2000: 8119, 2001: 7569, 2002: 7041, 2003: 7041, 2004: 7117,
        2005: 7170, 2006: 7178, 2007: 7253, 2008: 7044, 2009: 6855, 2010: 6951,
    },
    "eligible_count_by_year_median": {
        # this module's screened eligible_count (price/ADV/SMA200 all pass),
        # median across that year's trading days
        1993: 0, 1994: 391, 1995: 451, 1996: 523, 1997: 631, 1998: 733, 1999: 1750,
        2000: 2168, 2001: 2048, 2002: 2053, 2003: 2170, 2004: 2585, 2005: 2763,
    },
}

# ---------------------------------------------------------------------------
# Universe / eligibility screen (pre-declared, availability-only -- see
# scripts/eodhd_breadth.py module docstring for the exact rationale).
# ---------------------------------------------------------------------------
UNIVERSE = {
    "source": "data/research/eodhd/us_universe_full/*.parquet (full-history pull, supersedes the "
             "2005-start us_universe/), ACTIVE AND DELISTED common stock, survivorship-free",
    "cleaning": "scripts/eodhd_clean.py:clean_bars (v2), equity calendar = SPY's own trading dates in "
                "etfs_full/ (from 1993-01-29); no rolling window below is computed across a `segment` boundary",
    "sma_window": 200,
    "price_min_usd": 5.0,
    "price_basis": "raw (unadjusted) close, not adjusted_close -- adjusted_close can read as a few "
                   "dollars purely from cumulative forward splits while the stock trades at $100+ "
                   "today; raw close is the right number for a junk-microcap screen",
    "adv20_min_usd": 1_000_000.0,
    "adv_basis": "trailing 20-session median of adjusted_close x volume (protocol's own dollar-volume "
                "rule: EODHD volume is split-adjusted, open/close are raw)",
    "rationale": "keeps junk microcaps (illiquid, painfully thin, single-print-driven) from dominating "
                "the eligible count or the above/below-200sma tally, without using any return-based "
                "filter; both thresholds are round, pre-declared numbers, not tuned against the "
                "resulting breadth series or anything downstream of it",
}

# ---------------------------------------------------------------------------
# Breadth measures (the SIGNAL only; scripts/eodhd_breadth.py:build_breadth).
# ---------------------------------------------------------------------------
BREADTH = {
    "primary": {
        "id": "pct_above_200sma",
        "definition": "fraction of that day's eligible stocks with adjusted_close > trailing 200-session "
                      "SMA of adjusted_close (same-segment only)",
        "source": "standard, pre-literature choice (the brief's own 'e.g., % of stocks above their "
                  "200-day moving average'); not tuned",
    },
    "alternative": {
        "id": "net_ad_21d",
        "definition": "21-session trailing average of the daily net advance/decline ratio, "
                      "(advances - declines) / eligible_count, where an eligible stock advances/declines "
                      "by same-segment adjusted_close_t vs adjusted_close_{t-1}",
        "source": "trailing net advance/decline, as in the cited 'Herding for profits' paper's construct "
                  "(adapted from a cross-country portfolio measure to a single-country point-in-time "
                  "screened universe); 21 sessions is a standard trading-month window, not tuned",
    },
}

# ---------------------------------------------------------------------------
# Overlay rule and the (<=4) declared variant grid. Checked at the core's own
# monthly rebalance (first trading day of the calendar month, per
# src/firm/allocation/sleeves.py:is_calendar_rebalance_due -- read for
# reference only, not modified). Breadth is measured through the close of the
# session immediately before the rebalance day (no look-ahead into the
# rebalance day itself); the SPY-weight cut (or its absence) executes at the
# rebalance day's adjusted open, matching the protocol's own
# signal-through-close-t / fill-at-next-open convention with signal_lag_days=1.
# ---------------------------------------------------------------------------
OVERLAY = {
    "checked": "at the core's monthly rebalance (first trading day of the month)",
    "signal_measured_as_of": "close of the last trading day of the prior month",
    "executes_at": "the rebalance day's adjusted open",
    "rule_template": "if breadth(measure) < threshold: SPY weight = spy_weight_cut, destination weight "
                     "+= (0.60 - spy_weight_cut); else SPY 0.60 / IEF 0.40 (plain core)",
    "spy_weight_cut": 0.30,
}
VARIANTS = {
    "V1_primary": {
        "measure": "pct_above_200sma", "threshold": 0.50, "direction": "below", "destination": "IEF",
        "threshold_rationale": "50% is the standard, textbook 'more than half the market is below its "
                               "200dma' breadth-deterioration convention (widely cited in market-breadth "
                               "commentary as the natural equal-split point of the measure itself, which "
                               "is a proportion bounded in [0, 1]); not fit to this system's data",
    },
    "V2_cash_destination": {
        "measure": "pct_above_200sma", "threshold": 0.50, "direction": "below", "destination": "BIL",
        "threshold_rationale": "same signal/threshold as V1; tests whether the destination asset "
                               "(duration vs. cash-like) matters, holding the trigger fixed",
        "destination_proxy_before_bil": "DTB3 accrual (see PROXIES) -- BIL's own history starts "
                                        "2007-05-30; before that V2's destination leg uses the same "
                                        "cash proxy the protocol already gives for the portfolio's own "
                                        "cash balance, since BIL is itself a T-bill fund",
    },
    "V3_stricter_threshold": {
        "measure": "pct_above_200sma", "threshold": 0.40, "direction": "below", "destination": "IEF",
        "threshold_rationale": "a stricter, 'confirmed deterioration' bar (40%) -- also a round number, "
                               "chosen before any return was computed, to test threshold sensitivity "
                               "without opening a wide grid",
    },
    "V4_alt_measure": {
        "measure": "net_ad_21d", "threshold": 0.0, "direction": "below", "destination": "IEF",
        "threshold_rationale": "net_ad_21d is signed and centered near zero by construction (advances "
                               "minus declines); zero is the natural, undata-mined threshold for a "
                               "signed measure, exactly analogous to V1's 50% for a proportion measure",
    },
}
N_VARIANTS = len(VARIANTS)  # 4; Bonferroni/grid-size bookkeeping only -- alpha itself is fixed by protocol

# ---------------------------------------------------------------------------
# Execution / costs (protocol defaults; no deviation for S2).
# ---------------------------------------------------------------------------
EXECUTION = {
    "signal_lag_days": 1,  # signal as of close t-1 (prior month's last session); trade at day t's adjusted open
    "weights_drift_between_rebalances": True,
    "cash_rate": "BIL total return from 2007-05-30; FRED DTB3 / 100 / 252 per trading day, forward-filled, "
                "before that (protocol Sec.2, amendment 1)",
    "max_gross": 1.0,
    "two_day_lag_robustness_check": "reported, not gated (protocol Sec.2)",
}
COSTS = {
    "etf_bps_per_side_ge_50m_adv20": 3.0,
    "etf_bps_per_side_lt_50m_adv20": 10.0,
    "adv20_basis": "20-day median dollar volume, adjusted_close x volume",
    "note": "SPY, IEF and BIL all clear the $50M ADV20 threshold throughout their (etfs_full) history "
           "(re-checked 2026-09-30 post-amendment: full-history median ADV20 -- SPY $13.9B (from "
           "1993-01-29), IEF $86.5M (from 2002-07-26), BIL $36.3M (from 2007-05-30); trailing-5y "
           "minimum -- SPY $20.9B, IEF $380M, BIL $55.7M -- so 3bps/side applies essentially always; "
           "this is a liquidity check on the instruments, not a return). VFITX (the pre-IEF NAV proxy) "
           "and the DTB3 cash proxy are never traded, so they carry no transaction cost by construction.",
    "stress_multiplier": 2.0,  # bar A5
}

# ---------------------------------------------------------------------------
# Proxies (protocol Sec.1-3, amendment 1) -- used only to extend BM2 and the
# overlay's destination legs into years before the real ETF existed; never
# traded, never charged a cost, and no return computed across the proxy/ETF
# handoff date crosses a `segment` boundary (each series is cleaned and
# spliced on its own, not concatenated as raw prices).
# ---------------------------------------------------------------------------
PROXIES = {
    "bond_leg_before_ief": "VFITX (Vanguard Intermediate-Term Treasury, NAV total return, "
                           "asset='nav' cleaning) from its own first bar (1991-12-05) through "
                           "2002-07-25; IEF from 2002-07-26 (protocol Sec.3). Applies both to BM2's "
                           "bond leg and to every variant whose destination is IEF (V1, V3, V4) for "
                           "the ~7 months of the window (2002-01-02 through 2002-07-25) before IEF's "
                           "own history starts -- same proxy, same handoff date, no separate rule",
    "cash_before_bil": "FRED DTB3, accrued rate/100/252 per trading day, through 2007-05-29; "
                       "BIL total return from 2007-05-30 (protocol Sec.2)",
    "v2_destination_before_bil": "V2's BIL destination weight uses the cash_before_bil proxy "
                                 "(DTB3 accrual) before 2007-05-30 -- see VARIANTS['V2_cash_destination']",
}

# ---------------------------------------------------------------------------
# Benchmarks (protocol Sec.3). Primary = BM2 (the claim is that the overlay
# improves the core itself).
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "BM1_SPY": {"rule": "SPY buy-and-hold, 100%", "available_from": "1993-01-29"},
    "BM2_60_40": {"rule": "60% SPY / 40% (IEF, or VFITX before 2002-07-26 -- see PROXIES) rebalanced "
                          "monthly to target at the rebalance day's adjusted open",
                  "available_from": "1993-01-29 (SPY; VFITX itself starts 1991-12-05, so the bond leg "
                                    "is never the binding constraint)"},
    "BM3_SPY_VT": {"rule": "SPY weight = min(1, 0.12 / annualised std of last 21 daily SPY returns at close t), "
                          "traded at close t+1 only when |target - held| > 0.10; remainder in cash",
                   "target_vol": 0.12, "vol_window": 21, "band_abs": 0.10, "available_from": "1993-01-29"},
}
PRIMARY_BENCHMARK = "BM2_60_40"

# ---------------------------------------------------------------------------
# Window. Fixed from data AVAILABILITY only (never from a return). Per
# protocol Sec.3 (amendment 1), no window may start before 1993-02-01 (SPY's
# own calendar starts 1993-01-29; IEF/VFITX/BIL/DTB3 are never the binding
# constraint -- VFITX alone covers 1991-12 onward). The real binding
# constraint is the point-in-time ELIGIBLE STOCK COUNT reaching a stable
# level -- filled from OBSERVED once scripts/eodhd_breadth.py has run against
# us_universe_full; the midpoint below is the calendar midpoint of
# [start, DATA_END], fixed at freeze, not chosen from any half's outcome.
# ---------------------------------------------------------------------------
WINDOW = {
    "start": "2002-01-02",  # first US trading day of 2002
    "start_rationale": (
        "Derived from RAW ticker-file coverage (OBSERVED['raw_ticker_file_coverage_by_year']), not from "
        "this module's own screened eligible_count, because the coordinator's sampling check found the "
        "raw pull itself is sparse and survivor-tilted before ~1998 (independently reproduced here by an "
        "exact per-file year-presence scan of all 22,732 us_universe_full files: 1,509 names with any "
        "1993 bar, 1,885 in 1995 -- matching the coordinator's ~1,900 estimate almost exactly -- rising "
        "only to 2,058 by 1996). Raw coverage then jumps discontinuously from 2,058 (1996) to 5,828 "
        "(1997): a one-year ~3x jump that is a vendor historical-depth artifact, not a real change in how "
        "many US common stocks existed, since no real delisting/listing event moves the population that "
        "fast. 1997-2000 keep climbing/oscillating (5,828 -> 6,225 -> 8,303 -> 8,119), consistent with a "
        "mix of genuine dot-com-era listings and continued vendor-depth ramp-up, and 2000-2002 then fall "
        "(8,119 -> 7,569 -> 7,041), consistent with real dot-com-bust delistings. From 2002 through 2010 "
        "raw coverage sits in a tight band (6,855-7,253, i.e. within ~3% of its 9-year mean of ~7,072) "
        "with no further one-directional trend -- the first multi-year stretch that looks like coverage "
        "of a real, slowly-evolving population rather than a pull still catching up to it. 2002 is "
        "therefore the first year whose raw coverage is 'a level consistent with the real listed "
        "universe' rather than merely non-zero; it is also the first year in which this module's own "
        "screened eligible_count (1,755-2,194 across the year) sits on the same smooth upward trend that "
        "continues uninterrupted through 2007 (2,402-2,774 in 2004, 3,071-3,369 in 2007) -- i.e. nothing "
        "about the screened series itself looks discontinuous at the 2002 cut, only the raw pull does "
        "before it. No return, placebo, or benchmark series was inspected to make this choice."
    ),
    "end": DATA_END,
    "midpoint": "2014-05-16",  # nearest actual trading day to the calendar midpoint of [start, end]; A4's halves split here
    "protocol_floor": "1993-02-01 (SPY calendar start; not binding -- the 2002 coverage threshold binds first)",
}

# ---------------------------------------------------------------------------
# Inference (protocol Sec.4, fixed -- not re-derived per candidate).
# ---------------------------------------------------------------------------
BOOTSTRAP = {
    "method": "paired_stationary_block",  # same resampled days for candidate and benchmark; reuses
                                           # scripts/run_alt_premia_evaluation.py:stationary_indices
    "mean_block_days": 63,
    "n_boot": 5000,
    "seed": SEED,
    "alpha_one_sided": 0.01,  # protocol Sec.4: 0.05 / 5 candidates, fixed across the shortlist
}
PLACEBO = {
    "n_draws": 500,
    "pass_percentile": 95.0,
    "seed": SEED + 1,
    "rule": "the overlay's on/off state (per variant) permuted in 63-trading-day blocks, at the daily "
           "frequency the state is held between rebalances; sizing/weights otherwise unchanged "
           "(protocol Sec.4's S2-specific placebo rule, verbatim)",
}
PBO = {
    "n_partitions": 8,
    "series": "the 4 declared overlay variants + BM1, BM2, BM3 -- daily excess returns vs BM2 (the "
             "primary benchmark), on the evaluation window",
}
DSR = {
    "trial_sharpes": "daily (non-annualised) Sharpe of each of the 4 declared overlay variants, on the "
                     "evaluation window",
    "trials": N_VARIANTS,
    "prior_trials": 206,  # placeholder per task instruction;
                                                                         # the other 4 candidates' (S1, S3, S4, S5)
                                                                         # variant counts are not all frozen yet
    "ledger": "docs/S2_trial_history.json (new file; created only on the first real --append-ledger run, "
             "per protocol Sec.6)",
}

# ---------------------------------------------------------------------------
# Tiers -- the shared protocol's precedence and bar text (protocol wins where
# this differs from any older per-candidate style; see
# alt_premia_preregistered_bars.classify for the shared precedence function).
# S2 gets the protocol's extra defensive bar, A8.
# ---------------------------------------------------------------------------
TIER_A_BARS = [
    {"id": "A1", "rule": "Sharpe gap vs BM2 > 0, and its bootstrap one-sided lower bound at alpha 0.01 > 0"},
    {"id": "A2", "rule": "DSR > 0.95"},
    {"id": "A3", "rule": "Sharpe above the 95th percentile of the placebo Sharpes"},
    {"id": "A4", "rule": "Sharpe gap > 0 in both halves, vs BM2 and BM1/BM3"},
    {"id": "A5", "rule": "Sharpe gap > 0 at 2x costs, vs BM2 and BM1/BM3"},
    {"id": "A6", "rule": "CSCV PBO < 0.50"},
    {"id": "A7", "rule": "an independent recompute reproduces every bar outcome"},
    {"id": "A8", "rule": "S2-only defensive bar: max drawdown AND Calmar ratio both better than plain "
                         "60/40, with the Sharpe gap vs 60/40 > -0.05; judged on max drawdown and "
                         "Calmar, not on raw Sharpe alone"},
]
TIER_D_RULE = "bootstrap one-sided upper bound at alpha 0.01 of the Sharpe gap < 0 vs BM2"
TIER_B_BARS = [
    {"id": "B_a", "rule": "point estimate > 0 vs BM2"},
    {"id": "B_b", "rule": "A3, A4 and A5 pass"},
]
TIER_ACTIONS = {
    "A": "live config diff for the owner's sign-off (the 2-day-lag and per-year audits come first)",
    "B": "the owner decides",
    "C": "not deployed",
    "D": "rejected",
}
HALVES = {
    "rule": "the window split at its calendar midpoint, fixed at freeze from data availability, not "
           "returns; A4 requires a positive Sharpe gap in both halves",
}

# ---------------------------------------------------------------------------
# Power analysis (honest -- see module docstring re: A1 vs A8 tension). The
# overlay is a MONTHLY-CHECKED on/off switch, not a daily cross-sectional
# rank: the number of independent "bets" is bounded by how many times the
# state actually changes over the window, not by the number of trading days
# or the number of eligible stocks feeding the signal. OBSERVED above (once
# filled) gives the REAL count of monthly checks and state flips for each
# variant, computed from the signal only -- no return is used to produce
# these counts.
# ---------------------------------------------------------------------------
POWER_NOTES = {
    "unit_of_bet": "one contiguous run of the overlay being 'on' (SPY cut) is one independent episode; "
                  "a monthly-checked binary switch that changes state only a handful of times over a "
                  "20-year window has, at most, that many independent episodes -- nowhere near the "
                  "number of monthly rebalances or trading days in the window",
    "minimum_episodes_rule_of_thumb": "a stable win-rate/Calmar-improvement estimate over discrete "
                                      "regime episodes conventionally wants on the order of 20-30 "
                                      "independent episodes for even a rough normal-approximation "
                                      "read; fewer than that should be treated as descriptive, not "
                                      "as a statistically powered test, regardless of what the "
                                      "bootstrap/DSR numbers say formally",
    "caveat": "this is a rule of thumb, not a formal power calculation against an effect size -- there "
             "is no pre-existing literature estimate of 'how much better the Calmar ratio should be' "
             "for this untested overlay framing to build a classical N-required formula from (unlike "
             "insider_cluster_preregistered_bars.py's CMP effect-size anchor); the honest statement "
             "here is the raw episode count against the rule-of-thumb minimum, not a required-N figure",
    "observed_2026_09_30": (
        "297 monthly rebalance checks over the window (2002-01 through 2026-09). V1/V2 (pct_above_200sma "
        "< 0.50) flip 38 times -- 19 independent 'on' episodes, on for 25.6% of months. V3 (the stricter "
        "0.40 threshold) flips 30 times -- 15 episodes, on 16.2% of months. V4 (the alternative net_ad_21d "
        "measure, threshold 0) flips far more often -- 146 times, 73 episodes, on 38.0% of months. "
        "Against the 20-30-episode rule of thumb above: V1/V2 (19) and V3 (15) are BELOW it -- this "
        "candidate, on its primary and stricter-threshold variants, is UNDERPOWERED for a normally-"
        "powered test of whether the defensive claim (A8) is real, and that must be stated plainly to "
        "the reviewer regardless of what the bootstrap/DSR/PBO numbers say. V4 clears the rule-of-thumb "
        "episode count (73), but that is very likely because net_ad_21d is a much noisier, faster-"
        "reverting series than a 200-day SMA crossing -- more 'episodes' here plausibly means more "
        "whipsaw around a threshold, not more genuine, economically distinct breadth-deterioration "
        "regimes. Higher episode count is not, by itself, evidence that V4 is the more reliable variant."
    ),
}


def episode_count(monthly_state: list[bool]) -> dict:
    """Count contiguous True-runs ('on' episodes) and state flips in a monthly on/off series.

    Pure function over the SIGNAL's own on/off series (booleans only) -- no
    return is involved. Used both by the real OBSERVED numbers (from the
    actual breadth series) and by tests.
    """
    flips = 0
    episodes = 0
    prev = False
    for i, s in enumerate(monthly_state):
        if i > 0 and s != prev:
            flips += 1
        if s and not prev:
            episodes += 1
        prev = s
    return {"n_months": len(monthly_state), "n_flips": flips, "n_on_episodes": episodes,
            "pct_on": (sum(monthly_state) / len(monthly_state)) if monthly_state else 0.0}


def classify(bars: dict[str, bool], tier_d: bool, tier_b: dict[str, bool]) -> str:
    """Tier from bar outcomes, in the frozen precedence A > D > B > C (protocol Sec.5)."""
    if bars and all(bars.values()):
        return "A"
    if tier_d:
        return "D"
    if tier_b and all(tier_b.values()):
        return "B"
    return "C"


def bars_fingerprint() -> str:
    payload = json.dumps(
        {
            "DRAFT": DRAFT, "PREREGISTERED_AT": PREREGISTERED_AT, "DATA_END": DATA_END, "SEED": SEED,
            "CLEANING_FINGERPRINT": CLEANING_FINGERPRINT, "OBSERVED": OBSERVED, "UNIVERSE": UNIVERSE,
            "BREADTH": BREADTH, "OVERLAY": OVERLAY, "VARIANTS": VARIANTS, "N_VARIANTS": N_VARIANTS,
            "EXECUTION": EXECUTION, "COSTS": COSTS, "PROXIES": PROXIES, "BENCHMARKS": BENCHMARKS,
            "PRIMARY_BENCHMARK": PRIMARY_BENCHMARK, "WINDOW": WINDOW, "BOOTSTRAP": BOOTSTRAP,
            "PLACEBO": PLACEBO, "PBO": PBO, "DSR": DSR, "TIER_A_BARS": TIER_A_BARS,
            "TIER_D_RULE": TIER_D_RULE, "TIER_B_BARS": TIER_B_BARS, "TIER_ACTIONS": TIER_ACTIONS,
            "HALVES": HALVES, "POWER_NOTES": POWER_NOTES,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    print(("DRAFT" if DRAFT else "FROZEN") + " fingerprint:", bars_fingerprint())
