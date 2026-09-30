"""DRAFT pre-registration for S3 (EODHD shortlist): Treasury duration-bucket
momentum (Sihvonen) + commodity ETF dual momentum, combined into a 60/40-core
satellite.

Protocol: docs/eodhd_shortlist_protocol_2026_10.md (frozen 2026-09-30, amended
once — Amendment 1, 2026-09-30T22:40Z, adopted here, see AMENDMENT below) —
this file follows it exactly; where anything here differs, the protocol wins.
Source candidate: docs/research_brief_eodhd_findings.md §1 candidate #3.

STATUS: DRAFT, NOT FROZEN. This is Phase 1 (design only). No candidate,
benchmark or placebo return has been computed on the evaluation window — only
data AVAILABILITY was inspected (tickers present, first/last clean date,
counts, ADV20, via scripts/eodhd_clean.py:clean_bars), per the protocol's own
process rule ("design work may look at data availability ... never at
candidate or placebo returns on the window"). See AVAILABILITY below and
$S/runs/S3/availability.json for the raw per-ticker snapshot this design was
built against. Do not set DRAFT = False or PREREGISTERED_AT until a second,
separate pass reviews this file with no return series computed, per the
task's own phase split.

Question: do (a) a Treasury duration-bucket time-series momentum rule
(Sihvonen, "Yield Curve Momentum") and (b) a commodity-ETF cross-sectional
dual-momentum rule, run as sub-sleeves of a 60/40-core satellite, beat their
own asset-class buy-and-hold benchmarks out of sample, net of costs, after
deflating for this shortlist's trial count?

-----------------------------------------------------------------------------
Why this is NOT a re-run of the already-Tier-C'd alt-premia T2_cross_asset_trend
-----------------------------------------------------------------------------
T2 (scripts/alt_premia_preregistered_bars.py, result in
docs/alt_premia_evaluation_2026_09.json: Tier C, Sharpe gap negative point
estimate vs all three benchmarks, e.g. gap = -0.011 vs BM1, -0.092 vs BM2,
-0.131 vs BM3 over 2007-03 to 2026-09) was ONE portfolio that time-series-trend-
followed 8 single instruments across every asset class at once (SPY, EFA, EEM,
IEF, TLT, GLD, DBC, VNQ), each with a single 252-1-month formation window, each
independently on/off vs cash, inverse-vol-weighted into one book, vol-targeted
at the portfolio level. Fixed income was represented by exactly one instrument
(TLT, a single ~20y+ duration point) and commodities by exactly one instrument
(DBC, an already-diversified broad blend) — the momentum bet, in both cases,
was "trend or no trend" on one representative number per asset class.

S3 differs on most of those axes, but NOT all of them after this amendment,
and this section says so honestly rather than overstating the difference:

1. Bond mechanism is different in kind, not degree. T2 trend-follows a single
   duration point (TLT) over 252-1 months. S3 follows Sihvonen's own
   specification: 3 separate duration BUCKETS (SHY/IEF/TLT spanning ~1-3y,
   ~7-10y, 20y+), each judged over a 1-MONTH lookback, independently on/off.
   The bet is "has the yield curve just moved in this bucket's favour,"
   not "has this one bond ETF trended over the last year." Per the research
   brief's own instruction, the 1-month lookback is not extended — Sihvonen's
   own finding is that the effect is short-lived and "insignificant after the
   next month," so a 252-1 formation window (T2's window) would test a
   different, already-published-null claim, not this one. This axis of
   differentiation is untouched by the amendment.
2. Commodity mechanism is different in kind, but the PRIMARY lookback now
   overlaps T2's horizon, and that overlap is stated plainly rather than
   hidden. Per the coordinator's review, commodity_v1 (the primary variant)
   now uses a 12-1 month formation window — the same order of horizon as
   T2's 252-1-month window — because that is the peer-reviewed evidence
   (Moskowitz, Ooi & Pedersen 2012, time-series momentum; Asness, Moskowitz &
   Pedersen 2013, "Value and Momentum Everywhere," cross-sectional momentum),
   materially stronger than the practitioner-grade 3-month backtests that
   motivated the original brief (now the commodity_v2 sensitivity variant).
   What still makes S3 a genuinely different test is NOT the horizon: T2
   held ONE already-diversified broad commodity index (DBC) as a single
   trend-or-not bet mixed in with equities/bonds/REITs in the same
   inverse-vol blend. S3 never mixes commodities with equities or bonds in
   its ranking step: it builds a commodity-ONLY universe of 6 distinct
   exposures (energy, agriculture, base metals, gold, silver — see UNIVERSE
   below) and applies a CROSS-SECTIONAL relative-strength rank (hold the top
   3 of 6) plus an absolute-return filter (dual momentum) — a selection
   problem among commodities, evaluated against a commodity-only buy-and-hold
   benchmark, not a single time-series trend number blended with everything
   else and judged only against SPY/60-40/vol-target-SPY.
3. Benchmarks isolate the timing bet by asset class. T2's only benchmarks
   were SPY / 60-40 / vol-targeted SPY — none of which isolates whether
   bond-duration-timing or commodity-timing alone added anything over a
   static bond or commodity position. S3's primary benchmarks (protocol §3)
   are asset-class-matched buy-and-hold baskets of the SAME tickers, so the
   bond and commodity timing decisions are each judged against the thing they
   claim to beat, not against equities.
4. Granularity of evaluation. T2 was judged, and rejected, as ONE portfolio.
   S3 is pre-registered and will be tiered as THREE separate sub-candidates
   (bond, commodity, combined — protocol table, row S3), each on its own
   window and benchmark, so a bond-only or commodity-only edge (if real)
   is not hidden inside a single blended verdict the way it would be reusing
   T2's construction.

In short: the differentiator is the cross-sectional, commodity-only
construction and the own-asset-class benchmark, not the lookback horizon —
and after this amendment the horizon point is conceded rather than claimed.

-----------------------------------------------------------------------------
Protocol Amendment 1 (adopted here; see the protocol file's own changelog)
-----------------------------------------------------------------------------
Made before any candidate's pre-registration froze or computed a return.
Adopted in full:
  - data source switched from data/research/eodhd/etfs/ (2005-start, now
    superseded) to data/research/eodhd/etfs_full/ (full history from
    inception: SHY/IEF/TLT from 2002-07-26, GLD from 2004-11-18, USO/SLV from
    2006-04, DBA/DBB/DBE from 2007-01-05, BIL from 2007-05-30);
  - cleaning moved to v2 (fingerprint f62cb2e4…, see CLEANING below);
  - cash proxy is BIL from its first bar, and FRED DTB3 (accrued rate/100/252
    per trading day) before that — this REPLACES this file's earlier
    SHV-before-BIL rule;
  - BM2's bond leg before IEF's 2002-07-26 inception is VFITX (Vanguard
    Intermediate-Term Treasury, NAV total return) — a protocol-level,
    candidate-generic change (affects every candidate's BM2, not S3-specific
    logic), noted here for completeness.
"""

from __future__ import annotations

import hashlib
import json

# ---------------------------------------------------------------------------
# Freeze status (Phase 1: draft only)
# ---------------------------------------------------------------------------
DRAFT = True
PREREGISTERED_AT = None  # set only once this draft is reviewed with no return computed
DATA_END = "2026-09-29"  # last common clean close across every ticker below (AVAILABILITY)
TRADING_DAYS = 252
SEED = 20260930

AMENDMENT = {
    "id": "protocol_amendment_1",
    "adopted_at": "2026-09-30T22:40:00Z",  # protocol's own amendment timestamp
    "merged_commit": "aa22da3 (research/eodhd-shortlist, merged into research/eodhd-S3)",
    "changes_adopted": [
        "data source: etfs/ (superseded) -> etfs_full/ (full history from inception)",
        "cleaning: v1 (72a13e1e...) -> v2 (f62cb2e4...), adds asset='nav' for no-volume mutual-fund series",
        "cash proxy: BIL from inception (2007-05-30); FRED DTB3 (data/research/fred/DTB3.parquet, "
            "rate/100/252 per trading day) before that — replaces the earlier SHV-before-BIL rule",
        "BM2 bond leg before IEF's 2002-07-26 inception: VFITX NAV total return (protocol-generic, "
            "not S3-specific, noted for completeness)",
    ],
    "s3_specific_changes_in_this_pass": [
        "bond set: SHY/IEI/IEF/TLH/TLT (5 buckets) -> SHY/IEF/TLT (3 buckets); IEI/TLH dropped strictly "
            "for their later start date (2007-01-11 even in etfs_full), not for any return — verified by "
            "re-checking their etfs_full first-clean-date, which is unchanged from etfs/",
        "commodity set: 8 names -> 6 (DBA/DBB/DBE/USO/GLD/SLV); PALL/PPLT dropped, again strictly for "
            "start date (2010-01-08 even in etfs_full, unchanged from etfs/), not return",
        "commodity K: top-4-of-8 -> top-3-of-6",
        "commodity primary lookback: 3-month -> 12-1 month (now commodity_v1_primary); 3-month is now "
            "the sensitivity variant (commodity_v2_sensitivity)",
    ],
}

# ---------------------------------------------------------------------------
# Cleaning (shared protocol rule, v2 after Amendment 1 — scripts/eodhd_clean.py)
# ---------------------------------------------------------------------------
CLEANING = {
    "fingerprint": "f62cb2e4a139ea1d3cf240ce938f1d6f573d6208a9e2c004cedee66a47d07ccd",
    "version": 2,
    "source": "scripts/eodhd_clean.py:clean_bars / cleaning_fingerprint()",
    "rule": "every ETF series below is run through clean_bars() (equity calendar = etfs_full/SPY.parquet dates, "
            "from 1993-01-29) before any return is computed; no return is computed across a `segment` boundary",
    "nav_note": "VFITX (BM2's pre-IEF bond leg, protocol-generic) is cleaned with asset='nav' (skips the "
               "volume rule); it is never a traded holding in S3 itself, only a benchmark input",
}

# ---------------------------------------------------------------------------
# Data availability this design was built against (allowed under the process
# rule: coverage, start/end dates, counts, liquidity — never a return).
# Full per-ticker snapshot: $S/runs/S3/availability.json (this session).
# ---------------------------------------------------------------------------
AVAILABILITY = {
    "bond_tickers_first_clean_date": {"SHY": "2002-07-26", "IEF": "2002-07-26", "TLT": "2002-07-26"},
    "bond_common_start": "2002-07-26",  # all three share the same inception date
    "commodity_tickers_first_clean_date": {
        "DBA": "2007-01-05", "DBB": "2007-01-05", "DBE": "2007-01-05",
        "USO": "2006-04-10", "GLD": "2004-11-18", "SLV": "2006-04-28",
    },
    "commodity_common_start": "2007-01-05",  # DBA/DBB/DBE bind it
    "cash_proxy": {"BIL_first_clean_date": "2007-05-30", "fred_dtb3_from": "1954-01-04 (used pre-BIL only)"},
    "core_tickers_first_clean_date": {"SPY": "1993-01-29", "IEF": "2002-07-26"},
    "bm2_pre_ief_bond_leg_nav": {"VFITX_first_clean_date": "1993-01-29 (calendar-truncated to SPY's start)"},
    "last_clean_date_all_tickers": "2026-09-29",
    "adv20_usd_at_freeze": {  # from clean_bars output, last 20 clean bars, 2026-09 snapshot
        "SHY": 419_870_218, "IEF": 829_515_407, "TLT": 3_230_380_081,
        "DBA": 42_583_050, "DBB": 5_957_708, "DBE": 2_206_726,
        "USO": 986_606_133, "GLD": 3_816_395_731, "SLV": 894_764_733,
    },
    "adv20_note": "all 3 bond buckets and 3 of 6 commodity ETFs (USO/GLD/SLV) clear the protocol's $50M "
                 "ETF-cost-tier threshold (3 bps/side); DBA/DBB/DBE sit in the 10 bps tier (ADV20 evaluated "
                 "per rebalance per asset, not fixed at freeze)",
    "no_gaps_found": "all listed tickers show n_segments == 1 in clean_bars (no unadjusted-split-style break) "
                     "over their full clean history as of this snapshot",
    "considered_and_excluded_bond": {
        "IEI": "2007-01-11 even in etfs_full — later than SHY/IEF/TLT's common 2002-07-26 start; dropped "
              "for start date only, per the coordinator's review",
        "TLH": "2007-01-11 even in etfs_full — same reason as IEI",
        "VGIT": "2009-11-23 — near-duplicate of IEI's band, later start, dropped in the original draft too",
        "VGLT": "2009-11-24 — near-duplicate of TLT/TLH's band, later start",
        "GOVT": "2012-02-24 — blended duration, latest start of all bond candidates considered",
    },
    "considered_and_excluded_commodity": {
        "PALL": "2010-01-08 even in etfs_full — later than DBA/DBB/DBE's common 2007-01-05 start; dropped "
               "for start date only, per the coordinator's review",
        "PPLT": "2010-01-08 even in etfs_full — same reason as PALL",
    },
}

# ---------------------------------------------------------------------------
# Universe — fixed, justified sets. Treasuries only for bonds (per the task).
# ---------------------------------------------------------------------------
UNIVERSE = {
    "bond_buckets": {
        "tickers": ["SHY", "IEF", "TLT"],
        "why_these_three": "one liquid ETF per standard maturity rung spanning the curve (~1-3y, ~7-10y, "
                          "20y+), matching Sihvonen's own duration-bucket design (the brief specifies 3-4 "
                          "buckets) and sharing a single common inception date (2002-07-26), giving the "
                          "longest possible common bond window. IEI (~3-7y) and TLH (~10-20y) were "
                          "considered — they would give 5 rungs instead of 3 — but both still start "
                          "2007-01-11 even in the full-history etfs_full/ files, 4.5 years later than "
                          "SHY/IEF/TLT; they are excluded on that start-date basis alone, not on any "
                          "return, which was never computed. VGIT/VGLT/GOVT remain excluded as "
                          "near-duplicates that also start later.",
        "excluded_near_duplicates_or_later_start": ["IEI (2007-01-11)", "TLH (2007-01-11)",
                                                    "VGIT (2009-11-23)", "VGLT (2009-11-24)", "GOVT (2012-02-24)"],
        "treasuries_only": True,
    },
    "commodity_basket": {
        "tickers": ["DBA", "DBB", "DBE", "USO", "GLD", "SLV"],
        "n": 6,
        "coverage": {
            "agriculture": ["DBA"],
            "energy": ["DBE", "USO"],
            "base_metals": ["DBB"],
            "precious_metals": ["GLD", "SLV"],
        },
        "why_these_six": "one sector BASKET per broad category (agriculture, base metals, energy) plus "
                         "gold and silver as the two most liquid, genuinely distinct precious-metal single "
                         "names (gold: monetary/safe-haven; silver: industrial + monetary) — a dispersed "
                         "cross-section for the ranking step. PALL/PPLT (platinum, palladium) were "
                         "considered for even more within-metals dispersion, but both start 2010-01-08 "
                         "even in etfs_full/, 3 years later than DBA/DBB/DBE's common 2007-01-05 start; "
                         "dropping them (start-date basis only, no return computed) is what lets this "
                         "sub-candidate's window include 2008, which the coordinator flagged as important "
                         "for a commodity-timing test.",
        "excluded_near_duplicates_or_later_start": {
            "PALL (2010-01-08)": "later start than the binding DBA/DBB/DBE date; would cost the window 2008-2009",
            "PPLT (2010-01-08)": "same reason as PALL",
            "DBC/GSG/DJP (broad, all-sector blends)": "excluded entirely — including a broad index "
                "alongside its own sector components would just re-litigate T2's single-blended-index "
                "construction inside this ranking, and a broad index sits near the middle of any "
                "cross-sectional rank by construction, diluting the K-of-N selection",
            "DBP (precious-metals basket, ~80% gold/20% silver)": "excluded in favour of GLD+SLV held "
                "separately — same underlying exposure at far lower liquidity, no diversification lost",
            "CORN/SOYB/WEAT (single grains)": "excluded — near-duplicates of DBA's own agriculture basket",
            "CPER (single copper)": "excluded — copper is already a material weight inside DBB",
            "UNG (single natural gas)": "excluded — DBE already carries meaningful natural-gas weight "
                "alongside crude/products; kept USO instead as the second energy name for its far higher "
                "liquidity (ADV20 ~$987M vs UNG's ~$274M)",
        },
        "k": 3,
    },
    "core": {"tickers": ["SPY", "IEF"], "rule": "60% SPY / 40% IEF, rebalanced monthly (docs/allocation_deploy_runbook.md)"},
    "cash": {"primary": "BIL", "pre_inception_fallback": "FRED DTB3 (accrued rate/100/252 per trading day), "
                                                        "used before BIL's 2007-05-30 first bar"},
}

# ---------------------------------------------------------------------------
# Satellite weight (fixed, not tuned) — mirrors the live BTC-trend satellite's
# pro-rata-from-core construction (docs/allocation_deploy_runbook.md: core
# 92% / satellite 8%; here 90%/10%, split evenly across the two new sleeves).
# Confirmed as proposed by the coordinator's review (no change this pass).
# ---------------------------------------------------------------------------
SATELLITE = {
    "total_weight": 0.10,
    "split": {"bond_sleeve": 0.05, "commodity_sleeve": 0.05},
    "funding": "taken pro rata from the 60/40 core (core scaled by 0.90: SPY 54% / IEF 36% / satellite 10%)",
    "cash_buffer": "sleeve allocation not deployed this month (all buckets/assets off) sits in the cash proxy, "
                  "not redistributed back to the core intra-month",
}

# ---------------------------------------------------------------------------
# Windows. Fixed at freeze from AVAILABILITY only (never from a return).
# Midpoint = the trading-day exactly splitting the SPY-calendar (etfs_full)
# session count in the window (computed from the calendar only, not from any
# return). Recomputed this pass for the new, longer bond window and the
# commodity window that now reaches back to include 2008.
# ---------------------------------------------------------------------------
WINDOWS = {
    "bond": {"start": "2002-07-26", "end": DATA_END, "midpoint": "2014-08-25", "n_sessions": 6083},
    "commodity": {"start": "2007-01-05", "end": DATA_END, "midpoint": "2016-11-11", "n_sessions": 4964},
    "combined": {"start": "2007-01-05", "end": DATA_END, "midpoint": "2016-11-11", "n_sessions": 4964,
                 "note": "bound by the commodity sleeve's later start (DBA/DBB/DBE, 2007-01-05); the bond "
                         "sleeve and the 60/40 core both have data back to 2002-2007 (SPY to 1993), but the "
                         "combined portfolio can't run before every sleeve it holds exists"},
}

# ---------------------------------------------------------------------------
# Execution / costs (protocol §2 — not redeclared where it just restates the
# protocol; only the ADV-bucket assignment logic is candidate-specific).
# ---------------------------------------------------------------------------
EXECUTION = {
    "signal_timing": "closes through day t; trades fill at the next bar's adjusted open (protocol §2)",
    "robustness_2day_lag": "reported for every variant below, not gated (protocol §2)",
    "bond_rebalance": "monthly, on the first trading day of the month, using the prior calendar month's "
                      "bucket excess return (close-to-close) over the cash proxy",
    "commodity_rebalance": "monthly, same schedule, rank computed from each variant's declared lookback",
    "cash_accrual": "BIL total return from 2007-05-30; FRED DTB3 (data/research/fred/DTB3.parquet), "
                    "rate/100/252 per trading day, before that (protocol §2, Amendment 1) — REPLACES this "
                    "file's original SHV-before-BIL rule",
}
COSTS = {
    "etf_tier_bps": {"adv20_ge_50m": 3.0, "adv20_lt_50m": 10.0},
    "assignment": "evaluated per asset, per rebalance, from that asset's own trailing 20-day dollar volume "
                  "(adjusted_close x volume) — not fixed at freeze (protocol §1 dollar-volume rule, §2 cost rule)",
    "stress_multiplier": 2.0,  # A5
}

# ---------------------------------------------------------------------------
# Benchmarks (protocol §3)
# ---------------------------------------------------------------------------
BENCHMARKS_GENERIC = {
    "BM1_SPY": "SPY buy-and-hold, from 1993-01-29",
    "BM2_60_40": "60% SPY / 40% IEF, rebalanced monthly; bond leg is VFITX (NAV total return) before IEF's "
                "2002-07-26 inception (protocol §3, Amendment 1)",
    "BM3_SPY_VT": "SPY weight = min(1, 0.12 / 21-day realised vol), traded when target moves > 0.10",
}
PRIMARY_BENCHMARKS = {
    "bond": "equal-weight buy-and-hold of SHY/IEF/TLT, rebalanced monthly",
    "commodity": "equal-weight buy-and-hold of DBA/DBB/DBE/USO/GLD/SLV, rebalanced monthly",
    "combined": "90% (60/40 SPY/IEF) + 5% equal-weight bond basket (buy-and-hold) + 5% equal-weight "
               "commodity basket (buy-and-hold), rebalanced monthly — the same weights as SATELLITE, "
               "held static instead of timed",
}

# ---------------------------------------------------------------------------
# Variants (the declared grid — total 5, well under the task's cap of 6).
# Every trial Sharpe that feeds DSR/PBO below comes from exactly these five.
# commodity_v1 is now the 12-1 month (peer-reviewed) lookback, per the
# coordinator's review; the 3-month practitioner backtest is now the
# sensitivity variant.
# ---------------------------------------------------------------------------
VARIANTS = {
    "bond_v1": {
        "sleeve": "bond",
        "signal": "prior-calendar-month excess return (close-to-close) over the cash proxy, per bucket",
        "lookback_months": 1,  # Sihvonen's own spec — brief §1 candidate #3 explicitly says do not extend this
        "rule": "hold a bucket (equal weight across the buckets currently 'on') iff its signal > 0, else that "
                "bucket's share sits in the cash proxy",
        "why_only_one_bond_variant": "the source paper's own finding is that the effect is short-lived and "
                                     "insignificant beyond the next month; there is no literature-supported "
                                     "second lookback to grid over without extending past what's been "
                                     "evidenced, so this sub-candidate is not gridded",
    },
    "commodity_v1_primary": {
        "sleeve": "commodity",
        "signal": "trailing 12-1 month total return per commodity ETF (skip the most recent month, the "
                 "standard Jegadeesh-Titman convention)",
        "k": 3,
        "rule": "rank all 6 by the signal; hold the top 3 equal-weight, but only those of the top 3 whose own "
                "trailing signal (excess over the cash proxy) is > 0 — dual momentum (relative rank + "
                "absolute filter); unfilled slots and a fully-off month sit in the cash proxy",
        "source": "Moskowitz, Ooi & Pedersen (2012), 'Time Series Momentum'; Asness, Moskowitz & Pedersen "
                 "(2013), 'Value and Momentum Everywhere' — peer-reviewed, 12-1 month is the standard "
                 "formation window in both. Made primary over the 3-month practitioner backtest per the "
                 "coordinator's review: this is materially stronger evidence than Quantpedia-style "
                 "practitioner-grade 3-month backtests. NOTE (see module docstring): this horizon overlaps "
                 "T2's 252-1-month window — the differentiation from T2 is the commodity-only cross-"
                 "sectional construction and own-asset-class benchmark, not the horizon.",
    },
    "commodity_v2_sensitivity": {
        "sleeve": "commodity",
        "signal": "trailing 3-month total return per commodity ETF (formation, no skip month)",
        "k": 3,
        "rule": "same rank + absolute-filter rule as commodity_v1_primary, different (shorter) formation window",
        "source": "Quantpedia-style multi-commodity dual momentum, practitioner-grade evidence only",
        "reported_not_primary": True,
    },
    "combined_v1_primary": {
        "sleeve": "combined", "uses": ["bond_v1", "commodity_v1_primary"],
        "rule": "90% (60/40 SPY/IEF, monthly) + 5% bond_v1 + 5% commodity_v1_primary, monthly",
    },
    "combined_v2_sensitivity": {
        "sleeve": "combined", "uses": ["bond_v1", "commodity_v2_sensitivity"],
        "rule": "90% (60/40 SPY/IEF, monthly) + 5% bond_v1 + 5% commodity_v2_sensitivity, monthly",
        "reported_not_primary": True,
    },
}
N_VARIANTS = len(VARIANTS)  # 5

# ---------------------------------------------------------------------------
# Inference (protocol §4 — alpha and bootstrap/placebo mechanics are fixed by
# the shared protocol across all 5 shortlist candidates, not derived here).
# ---------------------------------------------------------------------------
BOOTSTRAP = {
    "method": "paired_stationary_block (protocol §4); reuse stationary_indices from "
             "scripts/run_alt_premia_evaluation.py",
    "mean_block_days": 63,
    "n_boot": 5000,
    "seed": SEED,
    "alpha_one_sided": 0.01,  # protocol §4: 0.05 / 5 shortlist candidates, fixed — not derived from N_VARIANTS
}
PLACEBO = {
    "n_draws": 500,
    "pass_percentile": 95.0,
    "seed": SEED + 1,
    "rule": "each asset's monthly on/off series permuted in 12-month blocks (protocol §4, the S3-specific rule)",
}
PBO = {"n_partitions": 8, "series": "all 5 variants above plus BM1-BM3, each sub-candidate's own daily excess returns"}
DSR = {
    "trial_sharpes": "daily Sharpe of every one of the 5 VARIANTS above, on the PBO window",
    "prior_trials": "190 + other shortlist variants (fixed at freeze)",  # placeholder — see protocol §4;
    # 190 = combination 57 + pattern_ml 104 + standalone 11 + alt_premia 10 + insider 8, all as of 2026-09-30;
    # the S1/S2/S4/S5 variant counts are added once every shortlist candidate is frozen, not before.
    # Confirmed as proposed (no change) by the coordinator's review.
    "ledger": "docs/S3_trial_history.json (new family; created on the first real --append-ledger run)",
}

# ---------------------------------------------------------------------------
# Tiers (protocol §5, precedence A > D > B > C — same as alt_premia's classify).
# Applied independently to each of bond / commodity / combined, each against
# ITS OWN primary benchmark (PRIMARY_BENCHMARKS) plus BM1-BM3 where the bar
# says "every benchmark".
# ---------------------------------------------------------------------------
TIER_A_BARS = [
    {"id": "A1", "rule": "Sharpe gap vs the primary benchmark > 0 AND bootstrap one-sided lower bound "
                         "at alpha_one_sided > 0"},
    {"id": "A2", "rule": "DSR > 0.95"},
    {"id": "A3", "rule": "Sharpe above the 95th percentile of the placebo Sharpes"},
    {"id": "A4", "rule": "Sharpe gap > 0 in both halves (WINDOWS midpoint), vs the primary benchmark and BM1-BM3"},
    {"id": "A5", "rule": "Sharpe gap > 0 at 2x costs, vs the primary benchmark and BM1-BM3"},
    {"id": "A6", "rule": "CSCV PBO < 0.50"},
    {"id": "A7", "rule": "an independent recompute reproduces every bar outcome"},
]
TIER_D_RULE = "bootstrap one-sided upper bound at alpha_one_sided of the Sharpe gap < 0 vs the primary benchmark"
TIER_B_BARS = [
    {"id": "B_a", "rule": "point estimate > 0 vs the primary benchmark, and A3, A4 and A5 pass"},
]
TIER_ACTIONS = {
    "A": "live config diff for the owner's sign-off (2-day-lag + per-year audits first)",
    "B": "owner decides (flagged 'not proven alpha')",
    "C": "not deployed",
    "D": "rejected",
}


def classify(bars: dict[str, bool], tier_d: bool, tier_b: dict[str, bool]) -> str:
    """Tier from bar outcomes, in the frozen precedence A > D > B > C (matches
    alt_premia_preregistered_bars.classify / insider_cluster_preregistered_bars.classify)."""
    if bars and all(bars.values()):
        return "A"
    if tier_d:
        return "D"
    if tier_b and all(tier_b.values()):
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Power analysis — honest, from ASSUMED general priors about these instruments'
# return distributions (NOT computed from this window's actual return series;
# no candidate, benchmark or placebo return exists yet). Same mean-detection
# method as insider_cluster_preregistered_bars.required_n, because neither
# sub-candidate has a realised return series yet to run the Jobson-Korkie/
# Sharpe-gap machinery in run_alt_premia_evaluation.py.
#
#     N = ((z_alpha + z_beta) * sigma / mu) ** 2
#
# BOND: the 3 duration buckets are NOT 3 independent monthly bets — they share
# a large common factor (the level/slope of the Treasury curve), so a bucket-
# count x month-count bet tally would overstate power. "effective_bets_per_year"
# below treats the whole curve as 1-2 quasi-independent bets per month, not 3
# (unchanged reasoning from the 5-bucket draft: fewer buckets doesn't change
# how correlated they are with each other). The longer window this amendment
# gives (2002-07-26 vs 2007-01-11, +24 years vs +19.7) pushes the single most
# optimistic pair (optimistic effect x central bet rate) just over the line —
# every other pair remains underpowered. Say so plainly in the final report.
#
# COMMODITY: 6 distinct commodities (was 8) with more genuinely separate macro
# drivers than the bond buckets, but still share a general "commodity risk
# premium" common factor; the conservative scenario halves the naive 6 x 12
# bet count, same halving convention as the original 8-name draft.
# ---------------------------------------------------------------------------
POWER = {
    "bond": {
        "sigma_bps_per_month": 150.0,  # ASSUMED blended duration-bucket monthly vol, general prior, "
                                       # not derived from this window's data
        "effect_bps_per_month": {"conservative_assumed": 5.0, "optimistic_assumed": 20.0},
        "effective_bets_per_year": {"conservative": 12, "central": 24},
        "window_years": round(WINDOWS["bond"]["n_sessions"] / TRADING_DAYS, 2),  # 2002-07-26 -> 2026-09-29
    },
    "commodity": {
        "sigma_bps_per_month": 500.0,  # ASSUMED blended commodity monthly vol, general prior
        "effect_bps_per_month": {"conservative_assumed": 20.0, "optimistic_assumed": 50.0},
        "effective_bets_per_year": {"conservative": 36, "central": 72},  # i.e. 3-6 of 6 names effectively independent, x 12
        "window_years": round(WINDOWS["commodity"]["n_sessions"] / TRADING_DAYS, 2),
    },
    "alpha_one_sided": BOOTSTRAP["alpha_one_sided"],
    "target_power": 0.80,
    "note": "UNVERIFIED-exact-figures throughout (sigma and effect assumptions are general, order-of-magnitude "
           "priors about Treasury-duration-ETF and commodity-ETF monthly return distributions, not re-derived "
           "from this window's own return series, and not sourced to a specific citation for an exact bp figure "
           "— the same caveat insider_cluster_preregistered_bars.POWER applies to its small/micro-cap vol "
           "prior). Confirmed fine as labelled by the coordinator's review.",
}


def required_n(mu_bps: float, sigma_bps: float, alpha_one_sided: float, power: float) -> float:
    """N = ((z_alpha + z_beta) * sigma / mu) ** 2 for a one-sample mean-detection test."""
    from firm.eval.overfitting import _norm_ppf

    z_a = _norm_ppf(1 - alpha_one_sided)
    z_b = _norm_ppf(power)
    return ((z_a + z_b) * sigma_bps / mu_bps) ** 2


def power_report() -> dict:
    """Required N vs. available effective bets, per sub-candidate x effect x bet-rate scenario."""
    rows = []
    for sub in ("bond", "commodity"):
        p = POWER[sub]
        for effect_key, mu in p["effect_bps_per_month"].items():
            n_req = required_n(mu, p["sigma_bps_per_month"], POWER["alpha_one_sided"], POWER["target_power"])
            for rate_key, bets_per_year in p["effective_bets_per_year"].items():
                n_avail = bets_per_year * p["window_years"]
                rows.append({
                    "sub_candidate": sub, "effect": effect_key, "mu_bps": mu,
                    "sigma_bps": p["sigma_bps_per_month"], "bet_rate_scenario": rate_key,
                    "effective_bets_per_year": bets_per_year,
                    "n_required_80pct_power": round(n_req, 1),
                    "n_available": round(n_avail, 1),
                    "adequately_powered": bool(n_avail >= n_req),
                })
    return {"rows": rows}


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------
def bars_fingerprint() -> str:
    payload = json.dumps(
        {
            "DRAFT": DRAFT, "PREREGISTERED_AT": PREREGISTERED_AT, "DATA_END": DATA_END, "SEED": SEED,
            "AMENDMENT": AMENDMENT, "CLEANING": CLEANING, "AVAILABILITY": AVAILABILITY,
            "UNIVERSE": UNIVERSE, "SATELLITE": SATELLITE,
            "WINDOWS": WINDOWS, "EXECUTION": EXECUTION, "COSTS": COSTS,
            "BENCHMARKS_GENERIC": BENCHMARKS_GENERIC, "PRIMARY_BENCHMARKS": PRIMARY_BENCHMARKS,
            "VARIANTS": VARIANTS, "N_VARIANTS": N_VARIANTS,
            "BOOTSTRAP": BOOTSTRAP, "PLACEBO": PLACEBO, "PBO": PBO, "DSR": DSR,
            "TIER_A_BARS": TIER_A_BARS, "TIER_D_RULE": TIER_D_RULE, "TIER_B_BARS": TIER_B_BARS,
            "TIER_ACTIONS": TIER_ACTIONS, "POWER": POWER,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    print(("DRAFT" if DRAFT else "FROZEN") + " fingerprint:", bars_fingerprint())
    import pprint
    pprint.pprint(power_report())
