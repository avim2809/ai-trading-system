"""DRAFT pre-registration for the S4 shortlist candidate: 52-week-high proximity,
liquid US stocks (George & Hwang 2004).

STATUS: PHASE 1 DRAFT (see DRAFT / PREREGISTERED_AT below). Written after the universe
panel (scripts/eodhd_s4_build_panel.py -> panel.parquet) was built from real OHLCV
history and BEFORE any candidate, benchmark or placebo RETURN was computed on the
evaluation window -- this file freezes the design against real eligibility/coverage
counts only, per docs/eodhd_shortlist_protocol_2026_10.md §6 ("design work may look at
data availability ... never at candidate or placebo returns on the window"). Do not add
a return computation to this module; that belongs in a separate phase-2 evaluation
script, mirroring run_alt_premia_evaluation.py's relationship to
alt_premia_preregistered_bars.py.

Protocol: docs/eodhd_shortlist_protocol_2026_10.md (frozen 2026-09-30, amended once
2026-09-30 22:40Z -- full-history *_full/ folders, cash/60-40 pre-inception proxies,
cleaning v2). Where this file differs from the protocol, the protocol wins.
Research basis: docs/research_brief_eodhd_findings.md §1 candidate #4 -- George & Hwang
(2004), *J. Finance* 59:2145-2176 (~0.45%/month gross decile spread, does not mean-revert
long-run, unlike classic JT momentum); Bettman, Sault & von Reibnitz (2010) find the
strategy fails net of realistic costs/liquidity broadly, BUT liquid names only still
show significant positive raw returns while illiquid names show negative returns --
this is exactly why the universe below is liquidity-screened, not a compromise made for
convenience. The HXZ (2020) stricter-replication verdict on 52-week-high variants is
flagged UNVERIFIED-conflicting in the brief (two secondary summaries disagree) and is
not relied on here either way.

Data: data/research/eodhd/us_universe_full/*.parquet (full 1985-or-inception history,
superseding the 2005-start us_universe/ folder per protocol amendment 1), cleaned with
scripts/eodhd_clean.py:clean_bars (v2, asset="equity", SPY-from-1993 calendar). No
point-in-time index-membership list exists or is used; the universe at each month-end
is built purely from trailing, point-in-time-known bar history (see UNIVERSE below and
the builder's own module docstring for the exact per-ticker screens and the
segment-resets-the-lookback-window design decision).

Survivorship check requested at the protocol-amendment hand-off, and independently
confirmed against a raw file-level date-range scan of all 22,732 us_universe_full
series (see OBSERVED["delisted_coverage_check"] for the exact counts): coverage of
DELISTED names is genuinely thin before ~1998 -- only 585 of 16,628 delisted tickers
with a us_universe_full file have any bar overlapping 1995 (vs 1,313 of the ~6,105
still-active tickers), rising to 6,439 delisted / 8,172 total by 2000. This is exactly
the survivorship-tilt risk the coordinator flagged: a raw full-universe count in the
mid-1990s is NOT reliable, because it under-represents names that would go on to
delist. The liquidity (top-N-by-dollar-volume) cut is less exposed -- large, liquid
names of that era are the ones EODHD is most likely to have backfilled regardless of
listing status -- but this is checked directly, not assumed: OBSERVED also reports,
for each candidate month-end 1994-2005, what fraction of that month's top-N selection
eventually shows listing=="delisted" in us_universe_symbols.parquet (a same-day data
fact about which names existed and traded then, not a forward return). The window
start (WINDOW below) is fixed from this table alone: the first year the eligible
(bars>=252, price>=$5) population is both >= N and shows a delisted-fraction in its
top-N broadly consistent with later, better-covered years (i.e. not a thin, survivor-
only slice) is used as the start, never a return-based or performance-based criterion.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eodhd_clean as ec  # noqa: E402

DRAFT = False
PREREGISTERED_AT = "2026-09-30T19:18:54Z"
SEED = 20260930

# ---------------------------------------------------------------------------
# Observed inputs (real, from data availability only -- see module docstring).
# Filled in from scripts/eodhd_s4_build_panel.py's panel.parquet /
# eligibility_summary.parquet once the *_full download completed. This block
# is descriptive; nothing here gates a tier outcome.
# ---------------------------------------------------------------------------
OBSERVED = {
    "cleaning_fingerprint_v2": "fc0690f087edaac8c11ddf77b381e58b59c57a893d460f12c7fa782229693054",
    "universe_dir": "data/research/eodhd/us_universe_full",
    "n_ticker_files": 22732,           # "extras final" us:{ok:22732,empty:9}, 2026-09-30
    "calendar_start": "1993-01-29",    # SPY's first bar in etfs_full/ (protocol amendment 1)
    "month_ends_first_reaches_N500_252bars_price5": "1994-01-31",   # NUMERIC only -- see
    "month_ends_first_reaches_N1000_252bars_price5": "1994-03-31",  # WINDOW: NOT used as
        # the start (thin, survivor-tilted universe of ~1,100-1,650 names -- see
        # topN_delisted_fraction_by_year below), superseded by the credible-coverage date
    "credible_coverage_transition": {
        "finding": "a sharp one-month discontinuity in the eligibility panel itself: "
                   "n_pass_both (bars>=252 & price>=$5) jumps from 1,638 (1998-11-30) to "
                   "3,292 (1998-12-31), and the fraction of that month's top-500-by-adv63 "
                   "selection that shows listing=='delisted' TODAY jumps from 0.244 to "
                   "0.648 in the same single month -- a step change, not a gradual trend "
                   "(monthly series 1996-01 to 2001-12 checked; every other month-to-month "
                   "change in that series is <= 0.014). This is the vendor coverage cliff "
                   "the coordinator flagged, now pinned to an exact month from the "
                   "eligibility panel itself (not just the raw file date-range scan).",
        "first_credible_month_end": "1998-12-31",
        "used_as_window_start": "1999-01-31 (one month's buffer past the cliff, in case "
                                "1998-12-31 itself is a partial-month artifact of whatever "
                                "caused the jump)",
    },
    "delisted_coverage_check": {
        "note": "requested at the protocol-amendment-1 hand-off: are 1990s delisted "
                "names really in us_universe_full, or does coverage silently start "
                "later for anything no longer listed? Confirmed thin: a raw file-level "
                "date-range scan (not the cleaned/screened panel) of all 22,732 series.",
        "n_delisted_symbols_total": 16636,
        "n_delisted_with_file": 16628,
        "n_delisted_with_a_bar_in_1995": 585,
        "n_delisted_with_a_bar_in_2000": 6439,
        "n_active_with_a_bar_in_1995": 1313,   # for comparison: active names are NOT thin that far back
        "n_active_with_a_bar_in_2000": 1733,
        "full_universe_series_covering_year": {
            # total series (active+delisted) whose file's [first,last] date range spans
            # that calendar year -- a raw availability count, NOT the screened/eligible
            # population in WINDOW below. 1993 is the calendar floor (SPY inception).
            "1993": 1541, "1995": 1898, "1998": 6230, "2000": 8172, "2003": 7200,
            "2005": 7344, "2008": 7259, "2010": 7186, "2015": 7721, "2020": 8234, "2025": 6946,
        },
    },
    "topN_delisted_fraction_by_year": {
        # Per the coordinator's explicit request: for the December month-end of each
        # year 1994-2005, of that month's top-500-by-adv63 selection (the actual
        # liquidity-cut candidate population, not the raw universe), what fraction shows
        # listing=='delisted' in us_universe_symbols.parquet today. A LOW fraction this
        # far back is the survivorship-tilt signature (the selection is dominated by
        # names that happened to still be listed decades later); a fraction in the same
        # range as the long-run sample (~0.55-0.70, checked through 2025) is the signal
        # that coverage is credible. n_eligible is the per-ticker-screen-passing
        # population that month (before the top-N cut), for scale.
        "1994": {"n_eligible": 1144, "frac_delisted_top500": 0.236},
        "1995": {"n_eligible": 1303, "frac_delisted_top500": 0.242},
        "1996": {"n_eligible": 1431, "frac_delisted_top500": 0.244},
        "1997": {"n_eligible": 1601, "frac_delisted_top500": 0.256},
        "1998": {"n_eligible": 3292, "frac_delisted_top500": 0.648},  # the cliff (Dec)
        "1999": {"n_eligible": 4281, "frac_delisted_top500": 0.692},
        "2000": {"n_eligible": 4102, "frac_delisted_top500": 0.656},
        "2001": {"n_eligible": 4188, "frac_delisted_top500": 0.610},
        "2002": {"n_eligible": 3742, "frac_delisted_top500": 0.548},
        "2003": {"n_eligible": 4385, "frac_delisted_top500": 0.578},
        "2004": {"n_eligible": 4540, "frac_delisted_top500": 0.586},
        "2005": {"n_eligible": 4666, "frac_delisted_top500": 0.584},
        "reference_2025": {"n_eligible": 3763, "frac_delisted_top500": None},  # not
            # recomputed (informational range only; full long-run series checked via the
            # December eligibility_summary table, frac range ~0.55-0.70 every year 1998-2025)
    },
}

# ---------------------------------------------------------------------------
# Window -- fixed from OBSERVED (data availability only, never returns), per the
# coordinator's explicit instruction after the credible_coverage_transition finding.
# ---------------------------------------------------------------------------
WINDOW = {
    "start": "1999-01-31",   # one month's buffer past the 1998-12-31 coverage cliff
    "end": "2026-09-29",     # last available month-end in the panel (data end, not a choice)
    "midpoint": "2012-11-30",  # nearest actual month-end to the exact calendar midpoint
                               # (2012-11-29) of [start, end] -- fixed at freeze, protocol §4
    "n_month_ends": 332,
    "protocol_floor_note": "protocol §3's absolute floor is 1993-02-01 (SPY inception); "
                           "this window starts far later, purely because the eligible "
                           "population before 1999 is thin and survivor-tilted (see "
                           "credible_coverage_transition and topN_delisted_fraction_by_year "
                           "above), not because of the protocol floor itself",
    "n_eligible_min_in_window": 3300,   # smallest n_pass_both anywhere in [start, end]
    "n_eligible_max_in_window": 4862,
    "N_primary_and_N_variant_supported_throughout": True,  # 3300 > 1000 > 500 at every
                                                           # month-end in the window
}

# ---------------------------------------------------------------------------
# Universe (per-ticker screens + cross-sectional liquidity cut).
# ---------------------------------------------------------------------------
UNIVERSE = {
    "source": "data/research/eodhd/us_universe_full/*.parquet, Type=='Common Stock' "
              "(us_universe_symbols.parquet), active AND delisted -- no point-in-time "
              "index membership used or needed",
    "per_ticker_screens_at_month_end": {
        "clean_bars_in_current_segment": ">= 252 (a segment break, e.g. an unadjusted "
                                         "reverse split, resets this to 0 -- treated like "
                                         "a fresh IPO for lookback purposes, see the "
                                         "builder's module docstring)",
        "price_min_usd": 5.0,
        "price_field": "raw (unadjusted) close, NOT adjusted_close -- adjusted_close is "
                       "back-adjusted for splits/dividends *after* the month-end in "
                       "question, so it is not the price that was actually quoted/"
                       "tradable at that date (design decision for reviewer sign-off)",
    },
    "liquidity_cut": "top N by trailing 63-session median dollar volume "
                     "(adjusted_close x volume, within-segment only, per protocol §1's "
                     "adjusted_close x volume rule)",
    "N_primary": 500,
    "N_variant": 1000,
    "N_decision_basis": "data availability only (eligible-count summary), not returns -- "
                        "both N=500 and N=1000 are comfortably supported at every "
                        "month-end within WINDOW (min eligible population 3,300, always "
                        ">> 1000); N is not itself what determined WINDOW's start -- "
                        "the coverage-cliff/survivorship finding did (see WINDOW and "
                        "OBSERVED['credible_coverage_transition'])",
    "panel_builder": "scripts/eodhd_s4_build_panel.py -> panel.parquet (ticker, "
                     "month_end, price, adv63, n_bars_in_segment, ratio_52wk, entry_date)",
}

# ---------------------------------------------------------------------------
# Signal (George & Hwang 2004).
# ---------------------------------------------------------------------------
SIGNAL = {
    "definition": "ratio_52wk_t = adjusted_close_t / max(adjusted_close over the "
                  "trailing 252 clean sessions in the SAME segment)",
    "rank": "cross-sectional decile within that month-end's eligible (post top-N "
           "liquidity cut) universe",
    "long_bucket": "top decile only (highest ratio_52wk = nearest to its own 52-week "
                   "high), long-only, no short leg",
    "weighting": "equal-weight within the long bucket at formation",
}

# ---------------------------------------------------------------------------
# Hold structure. Primary = George & Hwang's own JT-style overlapping monthly
# cohorts; at most one simpler variant (1-month hold), per the task's own cap.
# ---------------------------------------------------------------------------
HOLD_STRUCTURES = {
    "6_month_overlapping": {
        "rule": "a new top-decile cohort is FORMED every month-end; each cohort is HELD "
               "6 months. The live monthly return is the equal-weighted average across "
               "every cohort still open that month (up to 6 concurrently open cohorts, "
               "fewer during the first 5 months of the window) -- standard "
               "Jegadeesh-Titman overlapping-portfolio construction, matching George & "
               "Hwang's own paper.",
        "is_primary": True,
    },
    "1_month": {
        "rule": "reform every month-end, hold exactly 1 month, no overlap -- the "
               "declared 'at most one simpler variant'.",
        "is_primary": False,
    },
}

# ---------------------------------------------------------------------------
# Candidates: N (primary/variant) x hold (primary/simpler) = 4, the declared cap.
# ---------------------------------------------------------------------------
CANDIDATES = {
    "S4_p1_6mo_N500": {"hold": "6_month_overlapping", "N": 500, "is_primary": True},
    "S4_p2_6mo_N1000": {"hold": "6_month_overlapping", "N": 1000, "is_primary": False},
    "S4_p3_1mo_N500": {"hold": "1_month", "N": 500, "is_primary": False},
    "S4_p4_1mo_N1000": {"hold": "1_month", "N": 1000, "is_primary": False},
}
N_VARIANTS = len(CANDIDATES)
assert N_VARIANTS <= 4, "task cap: total variants <= 4"

# ---------------------------------------------------------------------------
# Execution / costs (protocol §2, verbatim where it applies to stocks).
# ---------------------------------------------------------------------------
EXECUTION = {
    "signal_lag_days": 1,   # closes through day t -> trade at next bar's adjusted open
    "adjusted_open_formula": "open * adjusted_close / close",
    "robustness_2day_lag": "reported, not gated, per protocol §2",
    "entry_date_field": "panel.parquet's entry_date column (first trading session "
                        "strictly after month_end)",
}
COSTS = {
    "stock_round_trip_bps_by_adv20_bucket_PER_SIDE": {
        "adv_lt_1m": 60.0, "adv_1m_5m": 30.0, "adv_5m_20m": 15.0, "adv_gt_20m": 5.0,
    },
    "note": "per protocol §2 -- half the insider pre-registration's round-trip table, "
           "charged per side here, on ADV20 (20-session median dollar volume), which "
           "the phase-2 evaluation must compute separately from panel.parquet's adv63 "
           "(63-session, used only for the liquidity cut, not for costing)",
    "cash_rate": "BIL total return from BIL's first bar (2007-05-30); FRED DTB3 "
                "accrued rate/100/252 per trading day before that -- not expected to "
                "bind for S4 (top decile is always populated when N reaches even a "
                "modest liquid universe), included for protocol completeness only",
    "stress_multiplier": 2.0,   # A5
}

# ---------------------------------------------------------------------------
# Delisting handling (mirrors insider_cluster_preregistered_bars.py's pattern).
# ---------------------------------------------------------------------------
DELISTING = {
    "handling": "a holding whose clean series ends, or hits a segment break, before "
               "its scheduled exit closes out at the last available clean close on "
               "that date -- no assumption of recovery or of the position surviving "
               "to the scheduled exit",
    "stress_variant_pct": -0.30,
    "stress_rationale": "Shumway (1997) average performance-delisting return, reused "
                        "from the insider-cluster pre-registration's own delisting-"
                        "stress bar; applied to every early exit as an EXTRA return on "
                        "top of the last clean close",
    "gating": "reported, not a bar unless phase 2's actual early-exit rate/impact "
             "argues it should be promoted to one (task's own instruction)",
}

# ---------------------------------------------------------------------------
# Benchmarks (protocol §3).
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "BM1_SPY": {"rule": "SPY buy-and-hold, 100%", "start": "1993-01-29"},
    "BM2_60_40": {
        "rule": "60% SPY / 40% bond leg, rebalanced monthly. Bond leg = IEF from "
               "IEF's first bar (2002-07-26); VFITX (NAV total return, cleaned with "
               "asset='nav') before that, from VFITX's own first bar.",
    },
    "BM3_SPY_VT": {
        "rule": "SPY weight = min(1, 0.12 / 21-day realised vol of SPY), traded when "
               "the target weight moves by more than 0.10; remainder in cash",
        "target_vol": 0.12, "vol_window": 21, "band_abs": 0.10,
    },
    "PRIMARY_liquid_universe_ewbh": {
        "rule": "equal-weight buy-and-hold of the SAME point-in-time eligible "
               "(post top-N liquidity cut) universe as the matching CANDIDATES entry, "
               "rebalanced monthly -- isolates the 52-week-high ranking from pure "
               "liquid-universe beta, per protocol §3",
    },
    "window_floor": "no candidate window may start before 1993-02-01 (protocol §3); S4's "
                   "OWN window (WINDOW above) starts materially later, at 1999-01-31, per "
                   "the credible-coverage finding -- BM1/BM2/BM3 are computed on S4's own "
                   "window, not extended back to 1993 just because the protocol permits it",
}

# ---------------------------------------------------------------------------
# Placebo (protocol §4: S4 is a rank-based candidate).
# ---------------------------------------------------------------------------
PLACEBO = {
    "method": "random rankings of the same eligible universe, same number of "
             "holdings (the decile size that month) and the same rebalance dates as "
             "the matching real CANDIDATES entry -- no ratio_52wk information used",
    "n_draws": 500,
    "pass_percentile": 95.0,
    "seed": SEED + 1,
}

# ---------------------------------------------------------------------------
# Inference (protocol §4).
# ---------------------------------------------------------------------------
BOOTSTRAP = {
    "method": "paired stationary block bootstrap of the daily Sharpe gap (candidate "
             "minus its primary benchmark), resampling the same days for both",
    "mean_block_days": 63,
    "n_boot": 5000,
    "seed": SEED,
    "alpha_one_sided": 0.05 / 5,   # 5 shortlist candidates tested in parallel
}
DSR = {
    "trials": N_VARIANTS,  # the daily Sharpes of every variant S4 itself declares
    "prior_trials": 206,
    "trial_sharpes": "daily (non-annualised) Sharpe of each of the 4 CANDIDATES entries",
    "ledger": "docs/S4_trial_history.json (new family; untouched by every other ledger)",
}
PBO = {"n_partitions": 8, "series": "the 4 CANDIDATES variants + BM1-BM3, daily excess returns"}

# ---------------------------------------------------------------------------
# Tiers -- protocol §5, verbatim scope (A1 vs primary only; A4/A5/D vs primary+BM1-3).
# ---------------------------------------------------------------------------
TIER_A_BARS = [
    {"id": "A1", "rule": "Sharpe gap vs PRIMARY_liquid_universe_ewbh > 0, and its "
                         "bootstrap one-sided lower bound at alpha_one_sided > 0"},
    {"id": "A2", "rule": "DSR > 0.95 (DSR block, this candidate's 4 trial Sharpes)"},
    {"id": "A3", "rule": "Sharpe above the 95th percentile of the 500 placebo Sharpes"},
    {"id": "A4", "rule": "Sharpe gap > 0 in both halves (split at the window midpoint, "
                         "fixed at freeze from data availability), vs the primary "
                         "benchmark AND BM1-BM3"},
    {"id": "A5", "rule": "Sharpe gap > 0 at 2x every cost, vs the primary benchmark "
                         "AND BM1-BM3"},
    {"id": "A6", "rule": "CSCV PBO < 0.50 (search-wide, the 4 variants + BM1-3)"},
    {"id": "A7", "rule": "an independent recompute reproduces every bar outcome"},
]
TIER_D_RULE = ("bootstrap one-sided upper bound at alpha_one_sided of the Sharpe gap "
               "< 0 vs the primary benchmark OR vs BM1-BM3")
TIER_B_BARS = [
    {"id": "B_a", "rule": "point estimate > 0 vs the primary benchmark"},
    {"id": "B_b", "rule": "A3, A4 and A5 all pass"},
]
TIER_ACTIONS = {
    "A": "live config diff for owner sign-off (2-day-lag + per-year audits first)",
    "B": "owner decides; labelled 'not proven alpha'",
    "C": "not deployed",
    "D": "rejected",
}


def classify(bars: dict[str, bool], tier_d: bool, tier_b: dict[str, bool]) -> str:
    """Tier from bar outcomes, frozen precedence A > D > B > C (protocol §5)."""
    if bars and all(bars.values()):
        return "A"
    if tier_d:
        return "D"
    if tier_b and all(tier_b.values()):
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Power analysis (task's own instruction: honest, with realistic decay --
# nothing here passes or fails anything). Method mirrors
# insider_cluster_preregistered_bars.required_n: a one-sample mean-detection
# calculation, T (months) required vs. months available in the window, NOT the
# bootstrap/DSR machinery above (that needs a real return series, which does
# not exist in phase 1).
# ---------------------------------------------------------------------------
POWER = {
    "effect_size_bps_per_month_gross_decile_spread": {
        # George & Hwang's own headline (docs/research_brief_eodhd_findings.md §1.4):
        "original_gh_gross": 45.0,
        # Bettman, Sault & von Reibnitz (2010): the strategy still shows SIGNIFICANT
        # positive raw returns in liquid names once illiquid names are excluded, but
        # give no single bps/month figure directly comparable to G&H's -- this scenario
        # applies a documented, UNVERIFIED-exact haircut (2/3 of gross) as a stand-in
        # for "liquid-only, still real but smaller than the full-universe headline":
        "liquid_only_haircut": 30.0,
        # McLean & Pontiff (2016)-style post-publication decay is well-documented across
        # the anomaly literature generally (~50% average); applied here as a third,
        # more conservative scenario specific to this candidate, not re-derived from a
        # 52-week-high-specific post-2004 replication (none found in the brief's search):
        "decayed_post_publication": 15.0,
    },
    # Monthly return-gap volatility for a liquid decile portfolio (long-only, top
    # decile vs the same liquid universe's equal-weight mean) is NOT re-derived from
    # panel.parquet here (that would require a return series -- out of scope for
    # phase 1). Assumption, UNVERIFIED-exact: a diversified ~50-100 name decile's
    # monthly tracking gap vs its own universe is well below single-stock idiosyncratic
    # vol; two scenarios bracket typical published long-short factor portfolio monthly
    # vol (e.g. HML/MOM, which run high-single-digit to low-double-digit %/month for the
    # long-short spread; a long-only decile-vs-universe gap is smaller):
    "sigma_monthly_bps_scenarios": {"central": 400.0, "conservative_wider": 600.0},
    "alpha_one_sided": BOOTSTRAP["alpha_one_sided"],
    "target_power": 0.80,
    # Effective independent bets per year differ sharply by hold structure: the
    # 6-month overlapping design's monthly observations are highly autocorrelated
    # (each pair of adjacent months shares 5 of 6 open cohorts), while the 1-month
    # design's monthly observations are close to independent. Treated here as a
    # documented haircut, exactly as the insider pre-registration haircut event
    # counts for time-clustering -- not derived from any return series:
    "effective_independent_months_per_calendar_month": {"6_month_overlapping": 1 / 6, "1_month": 1.0},
    "available_window_months": WINDOW["n_month_ends"],  # 332, WINDOW (1999-01 -> 2026-09)
}


def required_n(mu_bps: float, sigma_bps: float, alpha_one_sided: float, power: float) -> float:
    """N (effective independent observations) = ((z_alpha + z_beta) * sigma / mu) ** 2."""
    from firm.eval.overfitting import _norm_ppf
    z_a = _norm_ppf(1 - alpha_one_sided)
    z_b = _norm_ppf(power)
    return ((z_a + z_b) * sigma_bps / mu_bps) ** 2


def power_report() -> dict:
    """Required effective-N vs. available effective-N, for every (hold x effect x sigma)."""
    rows = []
    for hold, haircut in POWER["effective_independent_months_per_calendar_month"].items():
        avail_calendar_months = POWER["available_window_months"]
        avail_effective = (avail_calendar_months * haircut) if avail_calendar_months else None
        for effect_key, mu in POWER["effect_size_bps_per_month_gross_decile_spread"].items():
            for sigma_key, sigma in POWER["sigma_monthly_bps_scenarios"].items():
                n_req = required_n(mu, sigma, POWER["alpha_one_sided"], POWER["target_power"])
                rows.append({
                    "hold": hold, "effect": effect_key, "sigma_scenario": sigma_key,
                    "mu_bps": mu, "sigma_bps": sigma,
                    "n_required_80pct_power_effective_months": round(n_req, 1),
                    "avail_calendar_months": avail_calendar_months,
                    "avail_effective_months": (round(avail_effective, 1) if avail_effective else None),
                    "adequately_powered": (bool(avail_effective >= n_req) if avail_effective else None),
                })
    return {"rows": rows, "observed": OBSERVED}


def bars_fingerprint() -> str:
    payload = json.dumps(
        {
            "PREREGISTERED_AT": PREREGISTERED_AT, "SEED": SEED,
            "OBSERVED": OBSERVED, "WINDOW": WINDOW, "UNIVERSE": UNIVERSE, "SIGNAL": SIGNAL,
            "HOLD_STRUCTURES": HOLD_STRUCTURES, "CANDIDATES": CANDIDATES,
            "EXECUTION": EXECUTION, "COSTS": COSTS, "DELISTING": DELISTING,
            "BENCHMARKS": BENCHMARKS, "PLACEBO": PLACEBO, "BOOTSTRAP": BOOTSTRAP,
            "DSR": DSR, "PBO": PBO, "TIER_A_BARS": TIER_A_BARS, "TIER_D_RULE": TIER_D_RULE,
            "TIER_B_BARS": TIER_B_BARS, "TIER_ACTIONS": TIER_ACTIONS, "POWER": POWER,
        },
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    print(("DRAFT" if DRAFT else "FROZEN") + " fingerprint:", bars_fingerprint())
    import pprint
    pprint.pprint(power_report())

# Frozen cleaning rule: fail loudly if scripts/eodhd_clean.py changed after the freeze.
assert ec.cleaning_fingerprint() == OBSERVED["cleaning_fingerprint_v2"], "cleaning rule changed since freeze"
