"""DRAFT pre-registration for a multi-asset futures trend-following satellite,
evaluated as a PORTFOLIO addition, not as a standalone alpha source.

STATUS: DRAFT. Pattern: scripts/alt_premia_preregistered_bars.py. This file is
NOT frozen and has NOT been fingerprinted for a real run -- no futures price
data has been purchased (see docs/futures_data_vendor_comparison_2026_09.md).
It becomes eligible to freeze only after real data is in hand; freezing means:
(1) filling in every UNVERIFIED/PLACEHOLDER value below from the purchased
vendor's actual coverage and a real cost re-check, (2) recording
PREREGISTERED_AT as the true freeze timestamp, (3) committing the frozen file
and its fingerprint (this module's bars_fingerprint()) BEFORE any candidate
return series is computed, and (4) a new docs/futures_trend_trial_history.json
ledger entry, per this repo's evaluation standard.

Plan/context: docs/edge_search_verdict_2026_09.md (S4A/S4B),
docs/research_findings_beyond_equities_2026_09_30.md (Part 1 item 1),
docs/futures_data_vendor_comparison_2026_09.md,
docs/futures_trend_integration_sketch.md.

--------------------------------------------------------------------------
THE QUESTION THIS PREREG ANSWERS (deliberately reframed from earlier work)
--------------------------------------------------------------------------
Earlier alt-premia work (T2: cross-asset ETF trend, long/flat) tested "does a
trend sleeve beat SPY/60-40/vol-targeted-SPY" and landed Tier C (inconclusive,
point-estimate gap -0.01 to -0.13). Futures trend-following's own literature
frames its benefit as crisis convexity and diversification, not raw Sharpe
superiority over equities -- so re-running the same "beats SPY" test would be
testing the wrong hypothesis a second time with a fancier instrument.

The corrected finding in research_findings_beyond_equities_2026_09_30.md S0
is also directly relevant and is NOT re-litigated here: a quick empirical
check found the *current live book* (beta ~0, already low-vol) does NOT
reliably benefit from a diversifying sleeve, because it barely draws down on
its own. That finding is about the LIVE BOOK. This pre-registration is
deliberately about a DIFFERENT, more classical host portfolio -- 60/40
SPY/IEF -- which is equity-like and does draw down in the windows that
matter (2008, 2013, 2020, 2022). If a trend sleeve cannot show a benefit
against 60/40 in exactly the windows it is famous for helping, it should not
be expected to help this system's actual (very different) live book either.

Question: does adding a fixed-weight, literature-parameterised multi-asset
futures trend sleeve to a 60/40 SPY/IEF core improve that portfolio
out-of-sample, net of costs -- primarily judged on crisis-window drawdown and
secondarily on full-period Sharpe -- after deflating for this search's trial
count? This is NOT "does trend beat SPY."
"""

from __future__ import annotations

import hashlib
import json

STATUS = "DRAFT"  # becomes "FROZEN" only once data is purchased and every
                   # PLACEHOLDER/UNVERIFIED value below is replaced for real.
PREREGISTERED_AT = "2026-09-30T00:00:00Z"  # DRAFT authoring date, NOT a freeze
                                            # timestamp -- do not treat this as
                                            # "committed before any data ran."

# ---------------------------------------------------------------------------
# Universe -- the ~19-market list from the research brief. Substitutes are
# named because vendor coverage of the exact CME/Eurex/ICE ticker may differ;
# whichever substitute is actually used must be recorded here at freeze time.
# ---------------------------------------------------------------------------
UNIVERSE = {
    "equity_index": ["ES", "NQ", "RTY", "FDAX"],
    "rates": ["ZF", "ZN", "ZB", "FGBL", "FGBS"],
    "fx": ["6E", "6J", "6B", "6A", "6C"],
    "commodities": ["CL", "GC", "HG", "ZC", "ZS"],
}
N_MARKETS = sum(len(v) for v in UNIVERSE.values())  # 19
DATA_SOURCE = "PLACEHOLDER -- not purchased; see docs/futures_data_vendor_comparison_2026_09.md"
DATA_END = "PLACEHOLDER -- set to the vendor data's true as-of date at freeze"

# ---------------------------------------------------------------------------
# Trend rule -- fixed parameters taken directly from the literature, never
# tuned on this system's own data (matching this repo's pre-registration
# convention: T2/C1/M1/M2 in alt_premia_preregistered_bars.py all do the same).
# ---------------------------------------------------------------------------
TREND_RULE = {
    "source": "Moskowitz, Ooi & Pedersen, 'Time Series Momentum' (JFE 2012); "
               "parameter choice matches Hurst, Ooi & Pedersen, 'A Century of "
               "Evidence' (AQR/JOIM 2017) for the vol-targeting convention",
    "signal": "sign(12-month excess return over the matched-tenor risk-free rate) per market, "
              "at each month-end close t",
    "sizing": "per-market position = signal * (per_market_target_vol / sigma_i), "
              "sigma_i = annualised std of that market's last 63 daily returns "
              "(ex-ante, no look-ahead)",
    "per_market_target_vol": 0.10,
    "portfolio_scaling": "sleeve scaled so realised annualised sleeve vol targets "
                          "sleeve_target_vol (below), matching this system's own "
                          "vol_targeting_enabled convention in src/firm/agents/risk.py "
                          "(RiskAgent.vol_targeting_enabled), applied here at the sleeve "
                          "level rather than per-instrument",
    "sleeve_target_vol": 0.10,
    "rebalance": "monthly, at the first trading day's close after each month-end signal date "
                 "(1-day lag, matching this repo's signal_lag_days convention)",
    "vol_window": 63,
    "lookback_days": 252,
    "cross_market_weighting": "equal risk contribution across the 19 markets (inverse-vol, "
                               "not equal-notional), consistent with the brief's own "
                               "pre-registrable-rule text",
}

# ---------------------------------------------------------------------------
# Portfolio candidates -- the sleeve is a PORTFOLIO ADDITION to 60/40, not a
# standalone book. Unfunded overlay: total = (1 - alloc) * 60/40 + alloc * sleeve.
# ---------------------------------------------------------------------------
BENCHMARK = {
    "BM_60_40": {
        "rule": "60% SPY / 40% IEF, rebalanced to target at close t+1 after each month-end t",
        "note": "identical definition to BM2_60_40 in alt_premia_preregistered_bars.py -- "
                "reused deliberately so this test and the earlier T2 test share a benchmark",
    }
}
CANDIDATES = {
    "F10_trend_10pct": {
        "rule": "0.90 * BM_60_40 + 0.10 * futures_trend_sleeve, both legs rebalanced monthly",
        "trend_alloc": 0.10,
    },
    "F20_trend_20pct": {
        "rule": "0.80 * BM_60_40 + 0.20 * futures_trend_sleeve, both legs rebalanced monthly",
        "trend_alloc": 0.20,
    },
}
N_CANDIDATES = len(CANDIDATES)  # Bonferroni divisor = 2

# ---------------------------------------------------------------------------
# Costs -- commissions sourced from IBKR's own futures pricing page where
# fetched; everything else is an explicit, cited literature/internal estimate
# and is marked UNVERIFIED where a live re-check is still needed.
# ---------------------------------------------------------------------------
COSTS = {
    "commission_per_contract_round_turn": {
        "ES": {"value_usd": 2.24 * 2, "source": "interactivebrokers.com/en/pricing/commissions-futures.php, "
               "IBKR Cost-Plus: execution $0.85 + exchange $1.38 + reg $0.00 per side, fetched 2026-09-30; "
               "doubled here for a round turn"},
        "OTHER_MARKETS": {"value_usd": "UNVERIFIED -- placeholder $2.50/contract/side round-turn "
                           "(same order of magnitude as ES/E7 quotes seen), needs a direct re-fetch "
                           "of commissions-futures.php per symbol at freeze time (that page returned "
                           "403 to automated fetch during this research pass; a logged-in or manual "
                           "check is needed)"},
    },
    "roll_slippage_drag_pct_per_year": {
        "value": "0.3% - 0.6%/year",
        "source": "research_findings_beyond_equities_2026_09_30.md Part 1 item 1: \"a conservative "
                  "net-of-cost estimate for this system's specific 19-market universe, after modeling "
                  "roll/slippage drag of 0.3-0.6%/year\" -- an internal estimate from that brief, "
                  "itself not independently re-derived here. UNVERIFIED against real fill data; "
                  "must be replaced with an empirical estimate once real bid/ask and roll-date data "
                  "are available from the purchased vendor.",
    },
    "stress_multiplier": 2.0,  # bar A5-equivalent, matches alt_premia convention
}

# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
WINDOWS = {
    "full_period": {
        "start": "PLACEHOLDER -- earliest date with continuous back-adjusted data across "
                 "all 19 markets from the purchased vendor (Norgate: plausibly ~1990s-2000s "
                 "depending on the newest instrument's inception; set for real at freeze)",
        "end": "DATA_END",
        "purpose": "full-history point estimate and DSR/PBO trial pool, matching this "
                   "repo's convention of reporting full-period alongside OOS",
    },
    "primary_oos": {
        "start": "2010-01-01",
        "end": "DATA_END",
        "rationale": "the trend rule's parameters are taken from papers published 2012 "
                     "(Moskowitz/Ooi/Pedersen) and 2017 (Hurst/Ooi/Pedersen, using data "
                     "through ~2016); treating everything before the first paper's "
                     "publication as in-sample-by-construction and 2010-onward as the "
                     "honest OOS window matches this repo's post_publication_start "
                     "convention used throughout alt_premia_preregistered_bars.py",
    },
    "crisis_windows": {
        # Pre-specified, not chosen after seeing results -- this is the direct fix for
        # the "single scenario, not pre-registered" caveat flagged in
        # research_findings_beyond_equities_2026_09_30.md S0 item 2, which found the
        # LIVE BOOK didn't benefit from a diversifying sleeve in an ad hoc 2020-2022 check.
        # These are the same/superset windows plus the classic 2008 and 2013 episodes
        # that the live-book check did not include (and where 60/40 drew down harder).
        "gfc_2008": {"start": "2008-01-01", "end": "2009-06-30"},
        "taper_tantrum_2013": {"start": "2013-05-01", "end": "2013-09-30"},
        "covid_2020": {"start": "2020-02-15", "end": "2020-04-30"},
        "selloff_2022": {"start": "2022-01-01", "end": "2022-12-31"},
    },
    "bar": "for each crisis window, report BOTH candidate and benchmark max drawdown and "
           "window total return, per-window -- never a single blended number across "
           "windows, matching the reporting fix implied by the S0 correction",
}

# ---------------------------------------------------------------------------
# Power calculation
# ---------------------------------------------------------------------------
POWER = {
    "reuse": "same corrected methodology as docs/alt_premia_power_analysis_2026_09.json "
             "-- daily (not annualised) Sharpes in the analytic formula, and the "
             "simulation must NOT fix the realised gap (both were real bugs found and "
             "fixed in that file per docs/edge_search_verdict_2026_09.md S6 disclosures; "
             "this prereg inherits the fix, not the bug)",
    "known_constraint": "~15-20 years of daily data (this system's stated constraint) "
                        "can only detect a Sharpe gap of roughly 0.4-0.9 between "
                        "candidate and benchmark",
    "priors_for_power_calc_only": {
        # (correlation of daily excess returns with BM_60_40, literature-claimed annual
        # Sharpe gap from adding a trend sleeve) -- never used to pass/fail a bar.
        "F10_trend_10pct": {"corr_with_benchmark": 0.15, "claimed_sharpe_gap": 0.15},
        "F20_trend_20pct": {"corr_with_benchmark": 0.10, "claimed_sharpe_gap": 0.25},
    },
    "implication": "given the 0.15-0.5 realistic net Sharpe range for the sleeve itself "
                  "(per the research brief) and its low correlation to 60/40, the "
                  "expected full-period Sharpe GAP versus BM_60_40 is plausibly inside "
                  "or below the 0.4-0.9 detection floor -- exactly like every other "
                  "alt-premia candidate tested so far. The crisis-window bars below are "
                  "designed to be informative even when the full-period Sharpe test is "
                  "underpowered, since drawdown-in-a-specific-window is a much lower-"
                  "variance statistic than an annualised Sharpe gap over the same span.",
}

# ---------------------------------------------------------------------------
# Inference (paired block bootstrap, placebo, PBO, DSR) -- same machinery as
# alt_premia_preregistered_bars.py, applied to this family.
# ---------------------------------------------------------------------------
SEED = 20260930
BOOTSTRAP = {
    "method": "paired_stationary_block",
    "mean_block_days": 63,
    "n_boot": 5000,
    "seed": SEED,
    "alpha_one_sided": 0.05 / N_CANDIDATES,
}
PLACEBO = {
    "n_draws": 500,
    "pass_percentile": 95.0,
    "seed": SEED + 1,
    "rule": "each market's monthly on/off (long/flat via sign) state permuted in 12-month "
            "blocks, independently per market; inverse-vol sizing recomputed on the "
            "permuted state with the same realised vol path (i.e. permute direction, "
            "not magnitude) -- same family of test as T2's placebo in "
            "alt_premia_preregistered_bars.py",
}
PBO = {
    "n_partitions": 8,
    "window": "primary_oos",
    "series": "F10, F20, BM_60_40 daily excess returns",
}
DSR = {
    "trials": N_CANDIDATES + 1,  # 2 candidates + 1 benchmark
    "prior_trials": "cumulative across this search's own ledger families: combination=57, "
                    "standalone_strategy=11, alt_premia=10 (see docs/edge_search_verdict_2026_09.md); "
                    "the DSR at freeze time must use the running total from "
                    "docs/*_trial_history.json, not a reset count -- this repo's stated "
                    "standard is deflation with cumulative trials",
    "trial_sharpes": "daily (non-annualised) Sharpe of every candidate and benchmark on the primary_oos window",
}

# ---------------------------------------------------------------------------
# Tiers -- reframed around "does it help when it's needed most", per the
# owner's explicit framing. Tier A/B now gate primarily on crisis-window
# behavior, with full-period Sharpe as a secondary, likely-underpowered bar.
# ---------------------------------------------------------------------------
TIER_A_BARS = [
    {"id": "A1_crisis_dd", "rule": "in every crisis_window: maxDD(candidate) <= maxDD(BM_60_40)"},
    {"id": "A2_crisis_return", "rule": "in every crisis_window: window total return of candidate "
     ">= window total return of BM_60_40"},
    {"id": "A3_full_period", "rule": "full_period Sharpe(candidate) - Sharpe(BM_60_40) > 0 AND "
     "bootstrap one-sided lower bound at alpha_one_sided > 0"},
    {"id": "A4_placebo", "rule": "Sharpe(candidate) on primary_oos > 95th percentile of its placebo Sharpes"},
    {"id": "A5_dsr", "rule": "DSR(candidate on primary_oos) > 0.95 with cumulative DSR trials"},
    {"id": "A6_cost_stress", "rule": "A1 and A3 still hold with every cost x stress_multiplier"},
    {"id": "A7_pbo", "rule": "CSCV PBO < 0.50 (search-wide, this family)"},
]
TIER_D_RULE = "any crisis_window: maxDD(candidate) > maxDD(BM_60_40) AND window return(candidate) " \
              "< window return(BM_60_40) in 3 of 4 crisis windows -- i.e. the sleeve makes the " \
              "portfolio worse exactly when trend-following's own literature claims it helps"
TIER_B_BARS = [
    {"id": "B_a_crisis_dd", "rule": "in at least 3 of 4 crisis_windows: maxDD(candidate) <= maxDD(BM_60_40)"},
    {"id": "B_b_sharpe_floor", "rule": "full_period bootstrap one-sided lower bound of the Sharpe gap "
     "at alpha_one_sided > -0.10 (i.e. not measurably worse, even if not measurably better)"},
    {"id": "B_c_placebo", "rule": "A4 passes"},
]
TIER_ACTIONS = {
    "A": "live proposal (config diff for owner sign-off) -- crisis-tested diversification benefit, "
         "not just a Sharpe claim",
    "B": "live proposal labelled 'crisis-drawdown improvement in most pre-registered windows, not a "
         "proven full-period edge'; owner decides",
    "C": "not deployed; recommendation stays the pre-registered 60/40 fallback alone (no trend sleeve)",
    "D": "rejected -- the sleeve underperforms 60/40 in the exact scenarios trend-following is "
         "supposed to help with",
}
POST_PASS_AUDIT = [
    "re-run with signal_lag_days=2",
    "per-year Sharpe-gap and per-crisis-window table",
    "independent recompute by a second agent (this repo's standing practice per "
    "docs/edge_search_verdict_2026_09.md S6)",
    "re-run excluding each single crisis window, one at a time, to check no single "
    "episode drives the whole result (same audit already applied to C1 in alt_premia)",
]


def classify(bars: dict[str, bool], tier_d: bool, tier_b: dict[str, bool]) -> str:
    """Tier from bar outcomes, in the frozen precedence A > D > B > C."""
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
            "STATUS": STATUS,
            "PREREGISTERED_AT": PREREGISTERED_AT,
            "UNIVERSE": UNIVERSE,
            "DATA_SOURCE": DATA_SOURCE,
            "DATA_END": DATA_END,
            "TREND_RULE": TREND_RULE,
            "BENCHMARK": BENCHMARK,
            "CANDIDATES": CANDIDATES,
            "COSTS": COSTS,
            "WINDOWS": WINDOWS,
            "POWER": POWER,
            "BOOTSTRAP": BOOTSTRAP,
            "PLACEBO": PLACEBO,
            "PBO": PBO,
            "DSR": DSR,
            "TIER_A_BARS": TIER_A_BARS,
            "TIER_D_RULE": TIER_D_RULE,
            "TIER_B_BARS": TIER_B_BARS,
            "TIER_ACTIONS": TIER_ACTIONS,
            "POST_PASS_AUDIT": POST_PASS_AUDIT,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    print(f"status={STATUS} fingerprint={bars_fingerprint()}")
