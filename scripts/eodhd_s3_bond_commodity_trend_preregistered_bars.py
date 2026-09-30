"""DRAFT pre-registration for S3 (EODHD shortlist): Treasury duration-bucket
momentum (Sihvonen) + commodity ETF dual momentum, combined into a 60/40-core
satellite.

Protocol: docs/eodhd_shortlist_protocol_2026_10.md (frozen 2026-09-30) — this
file follows it exactly; where anything here differs, the protocol wins.
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

S3 differs on every one of those axes, not just in parameter values:

1. Bond mechanism is different in kind, not degree. T2 trend-follows a single
   duration point (TLT) over 252-1 months. S3 follows Sihvonen's own
   specification: 5 separate duration BUCKETS (SHY/IEI/IEF/TLH/TLT spanning
   ~1-3y to 20y+), each judged over a 1-MONTH lookback, independently on/off.
   The bet is "has the yield curve just moved in this bucket's favour,"
   not "has this one bond ETF trended over the last year." Per the research
   brief's own instruction, the 1-month lookback is not extended — Sihvonen's
   own finding is that the effect is short-lived and "insignificant after the
   next month," so a 252-1 formation window (T2's window) would test a
   different, already-published-null claim, not this one.
2. Commodity mechanism is different in kind. T2 held ONE already-diversified
   broad commodity index (DBC) as a single trend-or-not bet mixed in with
   equities/bonds/REITs in the same inverse-vol blend. S3 never mixes
   commodities with equities or bonds in its ranking step: it builds a
   commodity-ONLY universe of 8 distinct exposures (energy, agriculture,
   base metals, and four distinct precious/industrial single-metal names —
   see UNIVERSE below) and applies a CROSS-SECTIONAL relative-strength rank
   (hold the top K) plus an absolute-return filter (dual momentum) — a
   selection problem among commodities, not a single time-series trend
   number blended with everything else.
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

Reusing T2's exact single-TLT / single-DBC / 252-1-month construction here
would not be a new test; this file is checked to confirm it does not do that.
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

# ---------------------------------------------------------------------------
# Cleaning (shared, frozen 2026-09-30 — scripts/eodhd_clean.py)
# ---------------------------------------------------------------------------
CLEANING = {
    "fingerprint": "72a13e1edfb03c9ad62ac06b93fd1381353bd39292851f6d6c2d8b1b06bb42b5",
    "source": "scripts/eodhd_clean.py:clean_bars / cleaning_fingerprint()",
    "rule": "every ETF series below is run through clean_bars() (equity calendar = SPY dates) before any "
            "return is computed; no return is computed across a `segment` boundary",
}

# ---------------------------------------------------------------------------
# Data availability this design was built against (allowed under the process
# rule: coverage, start/end dates, counts, liquidity — never a return).
# Full per-ticker snapshot: $S/runs/S3/availability.json (this session).
# ---------------------------------------------------------------------------
AVAILABILITY = {
    "bond_tickers_first_clean_date": {
        "SHY": "2005-01-03", "IEI": "2007-01-11", "IEF": "2005-01-03",
        "TLH": "2007-01-11", "TLT": "2005-01-03",
    },
    "bond_common_start": "2007-01-11",  # max of the 5 bucket start dates (IEI/TLH bind)
    "commodity_tickers_first_clean_date": {
        "DBA": "2007-01-05", "DBB": "2007-01-05", "DBE": "2007-01-05", "USO": "2006-04-10",
        "GLD": "2005-01-03", "SLV": "2006-04-28", "PALL": "2010-01-08", "PPLT": "2010-01-08",
    },
    "commodity_common_start": "2010-01-08",  # max of the 8 tickers (PALL/PPLT bind)
    "cash_proxy_tickers": {"SHV": "2007-01-11", "BIL": "2007-05-30"},
    "core_tickers_first_clean_date": {"SPY": "2005-01-03", "IEF": "2005-01-03"},
    "last_clean_date_all_tickers": "2026-09-29",
    "adv20_usd_at_freeze": {  # from clean_bars output, last 20 clean bars, 2026-09 snapshot
        "SHY": 419_870_218, "IEI": 280_903_566, "IEF": 829_515_407, "TLH": 216_023_404, "TLT": 3_230_380_081,
        "DBA": 42_583_050, "DBB": 5_957_708, "DBE": 2_206_726, "USO": 986_606_133,
        "GLD": 3_816_395_731, "SLV": 894_764_733, "PALL": 20_640_830, "PPLT": 36_371_332,
    },
    "adv20_note": "all 5 bond buckets clear the protocol's $50M ETF-cost-tier threshold (3 bps/side); "
                 "of the 8 commodity ETFs, only USO/GLD/SLV clear it today (3 bps) — DBA/DBB/DBE/PALL/PPLT sit "
                 "in the 10 bps tier (ADV20 evaluated per rebalance per asset, not fixed at freeze)",
    "no_gaps_found": "all listed tickers show n_segments == 1 in clean_bars (no unadjusted-split-style break) "
                     "over their full clean history as of this snapshot",
}

# ---------------------------------------------------------------------------
# Universe — fixed, justified sets. Treasuries only for bonds (per the task).
# ---------------------------------------------------------------------------
UNIVERSE = {
    "bond_buckets": {
        "tickers": ["SHY", "IEI", "IEF", "TLH", "TLT"],
        "why_these_five": "one liquid ETF per standard maturity rung spanning the curve "
                          "(~1-3y, ~3-7y, ~7-10y, ~10-20y, 20y+), matching Sihvonen's own "
                          "duration-bucket design directly. VGIT/VGLT/GOVT were deliberately "
                          "left out: they are near-duplicates of IEI/TLT/IEF respectively "
                          "(same duration band, different issuer) and start later (2009-2012) "
                          "with no diversification benefit, so including them would only shorten "
                          "the window for no new information.",
        "excluded_near_duplicates": ["VGIT (~IEI band)", "VGLT (~TLT/TLH band)", "GOVT (blended, ~IEF-ish)"],
        "treasuries_only": True,
    },
    "commodity_basket": {
        "tickers": ["DBA", "DBB", "DBE", "USO", "GLD", "SLV", "PALL", "PPLT"],
        "n": 8,
        "coverage": {
            "agriculture": ["DBA"],
            "energy": ["DBE", "USO"],
            "base_metals": ["DBB"],
            "precious_industrial_metals": ["GLD", "SLV", "PALL", "PPLT"],
        },
        "why_these_eight": "one sector BASKET per broad category (agriculture, base metals, "
                          "energy) plus four genuinely distinct single-name precious/industrial "
                          "metals, each with a different dominant demand driver (gold: monetary/"
                          "safe-haven; silver: industrial + monetary; platinum: diesel autocatalyst "
                          "+ jewellery; palladium: gasoline autocatalyst, Russia/South-Africa supply "
                          "concentrated) — a genuinely dispersed cross-section for the ranking step, "
                          "not four bets on the same metal cycle.",
        "excluded_near_duplicates": {
            "DBC/GSG/DJP (broad, all-sector blends)": "excluded entirely — including a broad index "
                "alongside its own sector components would just re-litigate T2's single-blended-"
                "index construction inside this ranking, and a broad index sits near the middle of "
                "any cross-sectional rank by construction, diluting the K-of-N selection",
            "DBP (precious-metals basket, ~80% gold/20% silver)": "excluded in favour of GLD+SLV held "
                "separately — same underlying exposure at ~28x GLD's and ~800x SLV's ADV20, "
                "no diversification lost by dropping the basket",
            "CORN/SOYB/WEAT (single grains)": "excluded — near-duplicates of DBA's own agriculture "
                "basket, and the latest of the three (SOYB/WEAT) starts 2011-09, 20 months later "
                "than the binding PALL/PPLT start, for a sub-sector already covered by DBA",
            "CPER (single copper)": "excluded — copper is already a material (~30%+) weight inside "
                "DBB; adding it as a ninth, highly correlated name would inflate the apparent bet "
                "count without adding real cross-sectional information, and its 2011-11-15 start is "
                "later than the binding PALL/PPLT start anyway",
            "UNG (single natural gas)": "excluded — DBE already carries meaningful natural-gas "
                "weight alongside crude/products; kept USO instead of UNG as the second energy name "
                "because USO is far more liquid (ADV20 ~$987M vs UNG's ~$274M) for the same "
                "diversification purpose",
        },
    },
    "core": {"tickers": ["SPY", "IEF"], "rule": "60% SPY / 40% IEF, rebalanced monthly (docs/allocation_deploy_runbook.md)"},
    "cash": {"primary": "BIL", "pre_inception_fallback": "SHV (used 2007-01-11 to 2007-05-29, before BIL existed)"},
}

# ---------------------------------------------------------------------------
# Satellite weight (fixed, not tuned) — mirrors the live BTC-trend satellite's
# pro-rata-from-core construction (docs/allocation_deploy_runbook.md: core
# 92% / satellite 8%; here 90%/10%, split evenly across the two new sleeves).
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
# Midpoint = the trading-day exactly splitting the SPY-calendar session count
# in the window (computed from the calendar only, not from any return).
# ---------------------------------------------------------------------------
WINDOWS = {
    "bond": {"start": "2007-01-11", "end": DATA_END, "midpoint": "2016-11-15", "n_sessions": 4960},
    "commodity": {"start": "2010-01-08", "end": DATA_END, "midpoint": "2018-05-17", "n_sessions": 4206},
    "combined": {"start": "2010-01-08", "end": DATA_END, "midpoint": "2018-05-17", "n_sessions": 4206,
                 "note": "bound by the commodity sleeve's later start (PALL/PPLT, 2010-01-08); the bond "
                         "sleeve and the 60/40 core both have data back to 2005-2007, but the combined "
                         "portfolio can't run before every sleeve it holds exists"},
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
    "BM1_SPY": "SPY buy-and-hold",
    "BM2_60_40": "60% SPY / 40% IEF, rebalanced monthly",
    "BM3_SPY_VT": "SPY weight = min(1, 0.12 / 21-day realised vol), traded when target moves > 0.10",
}
PRIMARY_BENCHMARKS = {
    "bond": "equal-weight buy-and-hold of SHY/IEI/IEF/TLH/TLT, rebalanced monthly",
    "commodity": "equal-weight buy-and-hold of DBA/DBB/DBE/USO/GLD/SLV/PALL/PPLT, rebalanced monthly",
    "combined": "90% (60/40 SPY/IEF) + 5% equal-weight bond basket (buy-and-hold) + 5% equal-weight "
               "commodity basket (buy-and-hold), rebalanced monthly — the same weights as SATELLITE, "
               "held static instead of timed",
}

# ---------------------------------------------------------------------------
# Variants (the declared grid — total 5, well under the task's cap of 6).
# Every trial Sharpe that feeds DSR/PBO below comes from exactly these five.
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
        "signal": "trailing 3-month total return per commodity ETF (formation, no skip month)",
        "k": 4,
        "rule": "rank all 8 by the signal; hold the top 4 equal-weight, but only those of the top 4 whose own "
                "trailing signal (excess over the cash proxy) is > 0 — dual momentum (relative rank + "
                "absolute filter); unfilled slots and a fully-off month sit in the cash proxy",
        "source": "Quantpedia-style multi-commodity dual momentum, practitioner-grade evidence only "
                 "(brief §1 candidate #3); 3-month is the most commonly cited formation window in that "
                 "literature",
    },
    "commodity_v2_sensitivity": {
        "sleeve": "commodity",
        "signal": "trailing 12-1 month total return per commodity ETF (skip the most recent month, "
                 "the standard Jegadeesh-Titman convention already used for S1)",
        "k": 4,
        "rule": "same rank + absolute-filter rule as commodity_v1, different formation window",
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
# BOND: the 5 duration buckets are NOT 5 independent monthly bets — they share
# a large common factor (the level/slope of the Treasury curve), so a bucket-
# count x month-count bet tally would overstate power. "effective_bets_per_year"
# below treats the whole curve as 1-2 quasi-independent bets per month, not 5.
# This, plus the source paper's own "short-lived, insignificant after the next
# month" caveat (no magnitude given), is why this sub-candidate is flagged
# UNDERPOWERED below under every assumption pair but the most optimistic one —
# and even that one is borderline. Say so plainly in the final report.
#
# COMMODITY: 8 distinct commodities with more genuinely separate macro drivers
# than the bond buckets, but still share a general "commodity risk premium"
# common factor; the conservative scenario halves the naive 8 x 12 bet count.
# ---------------------------------------------------------------------------
POWER = {
    "bond": {
        "sigma_bps_per_month": 150.0,  # ASSUMED blended duration-bucket monthly vol, general prior, "
                                       # not derived from this window's data
        "effect_bps_per_month": {"conservative_assumed": 5.0, "optimistic_assumed": 20.0},
        "effective_bets_per_year": {"conservative": 12, "central": 24},
        "window_years": (2026 - 2007) + (9 - 1) / 12,  # 2007-01-11 -> 2026-09-29, approx
    },
    "commodity": {
        "sigma_bps_per_month": 500.0,  # ASSUMED blended commodity monthly vol, general prior
        "effect_bps_per_month": {"conservative_assumed": 20.0, "optimistic_assumed": 50.0},
        "effective_bets_per_year": {"conservative": 48, "central": 96},  # i.e. 4-8 of 8 names effectively independent, x 12
        "window_years": (2026 - 2010) + (9 - 1) / 12,
    },
    "alpha_one_sided": BOOTSTRAP["alpha_one_sided"],
    "target_power": 0.80,
    "note": "UNVERIFIED-exact-figures throughout (sigma and effect assumptions are general, order-of-magnitude "
           "priors about Treasury-duration-ETF and commodity-ETF monthly return distributions, not re-derived "
           "from this window's own return series, and not sourced to a specific citation for an exact bp figure "
           "— the same caveat insider_cluster_preregistered_bars.POWER applies to its small/micro-cap vol prior)",
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
            "CLEANING": CLEANING, "AVAILABILITY": AVAILABILITY, "UNIVERSE": UNIVERSE, "SATELLITE": SATELLITE,
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
