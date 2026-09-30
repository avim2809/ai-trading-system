"""DRAFT pre-registration for S1 (industry/sector ETF momentum), EODHD shortlist.

Protocol: docs/eodhd_shortlist_protocol_2026_10.md, AS AMENDED (Amendment 1,
2026-09-30 22:40Z) — this file follows it exactly for cleaning, execution,
costs, benchmarks, inference and tiers; where anything here differs, the
protocol wins. Source candidate: docs/research_brief_eodhd_findings.md §1
candidate #1. House style: scripts/alt_premia_preregistered_bars.py
(structure) and scripts/insider_cluster_preregistered_bars.py (a
DRAFT-then-freeze design built against real availability counts before any
return existed).

STATUS: DRAFT. PREREGISTERED_AT is None. Design work below used ONLY data
*availability* (tickers present, first/last clean dates, ADV levels, count of
eligible ETFs over time — see $S/runs/S1/availability.json, produced by a
one-off scan, not committed here, that calls scripts/eodhd_clean.py:clean_bars
on every candidate ETF and scripts/eodhd_clean.py:equity_calendar for the
exchange calendar). NO candidate, benchmark or placebo RETURN has been computed
on the evaluation window. This file is promoted to frozen (DRAFT = False,
PREREGISTERED_AT set) only once the owner signs off on the design below; no
further edits to the numbers after that, per the protocol's "Freeze first" rule.

Question: does a monthly-rebalanced, top-tercile-of-trailing-6-month-return
(skip-1-month) long-only rotation among liquid US sector/industry ETFs beat an
equal-weight buy-and-hold of the same eligible universe (plus SPY / 60-40 /
vol-targeted-SPY, reported), net of realistic ETF costs, after deflating for
this shortlist's trial count? Source: Moskowitz & Grinblatt (1999), "Do
Industries Explain Momentum?", J. Finance 54:1249-1290 — see the research
brief's citation and caveats (Grundy & Martin 2001 lag critique UNVERIFIED at
first-hand level; HXZ 2020 survival UNVERIFIED at exact t-stat level).

Data availability actually found this session (see AVAILABILITY below), UPDATED
after Amendment 1 switched every candidate from `etfs/` (2005-01-01 download
start, an artifact of a fixed setting in the download script, not true fund
inception) to `etfs_full/` (full history from inception):
 - `etfs_full/SPY` starts 1993-01-29 (the exchange calendar); the 9 original
   SPDR sector funds (XLB/XLE/XLF/XLI/XLK/XLP/XLU/XLV/XLY) start 1998-12-22;
   XLRE starts 2015-10, XLC starts 2018-06. All 44 universe tickers have clean
   data through 2026-09-29, one clean segment each at cleaning fingerprint v2
   (0 spike/segment-break events across the whole universe).
 - The window start is now re-derived from `etfs_full` and tied to this file's
   OWN cash-fallback threshold (ELIGIBILITY['n_min_for_trading'] = 9): the
   first month-end with >= 9 eligible ETFs THAT NEVER SUBSEQUENTLY DIPS BELOW 9
   through data end. That turned out to be 2002-01-31, not "mid-1999" as a
   naive reading of "9 sectors x 210 days" would suggest (2002-01-31 IS still
   ~2 years earlier than the old 2005-based window did allow). The real
   constraint in 1999-2001 is NOT history length (the 9 original sectors
   already have 210+ days by mid-1999) but the ADV20 >= $1M eligibility floor:
   the eligible count is genuinely noisy and repeatedly drops as low as 3
   through 2000-2001 (e.g. only XLE/XLF/XLK clear $1M ADV for most of 2000) --
   real, thin early trading volume in the sector SPDRs before they became
   popular, not a bug. 2002-01-31 is the first point after which the count is
   stable at >= 9 for the rest of the 297-month window (see AVAILABILITY).
 - This window start (2002-01-31) is BEFORE IEF's first bar (2002-07-26), so
   S1's own BM2 benchmark uses the VFITX NAV proxy (protocol §3) for its first
   ~6 months, exactly as the amended protocol anticipates.
 - PBJ (food & beverage; part of the frozen UNIVERSE list) is currently BELOW
   the $1M ADV20 eligibility floor (~$0.37-0.42M) and has been for a while —
   it stays in the declared universe (so it isn't a) hand-picked out after
   looking at anything performance-related, and b) the point-in-time filter
   below is exactly what keeps an ETF that's gone illiquid out of the traded
   book, mirroring the protocol's S4 point-in-time-eligibility spirit).
 - The universe is NOT static: it grows from 9 eligible names at window start
   to 43 at data end (PBJ self-excludes throughout on ADV) as later sector/
   industry ETFs launch and season. The primary benchmark (equal-weight
   buy-and-hold of the SAME point-in-time eligible set) absorbs this growth by
   construction -- it is never compared against a fixed later-date universe.
"""

from __future__ import annotations

import hashlib
import json
import math

DRAFT = True
PREREGISTERED_AT = None

DATA_END = "2026-09-29"
TRADING_DAYS = 252
SEED = 20260930

# ---------------------------------------------------------------------------
# Data availability (frozen from availability only; see module docstring).
# ---------------------------------------------------------------------------
SECTOR_11 = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "XLRE", "XLC"]
INDUSTRY_33 = [
    "SMH", "SOXX", "XSD", "XBI", "IBB", "XPH", "IHI", "KBE", "KRE", "KIE", "IAI", "XHB",
    "ITB", "XRT", "XOP", "OIH", "XES", "XME", "GDX", "GDXJ", "IYT", "XTN", "ITA", "XAR",
    "IGV", "FDN", "VNQ", "IYR", "PBJ", "XHS", "IYZ", "TAN", "ICLN",
]
UNIVERSE = SECTOR_11 + INDUSTRY_33  # frozen list, 44 tickers; excludes broad/size/bond/
                                    # commodity/currency/country ETFs per the task's own rule

AVAILABILITY = {
    "source_scan": "one-off script over data/research/eodhd/etfs_full/*.parquet (Amendment 1: "
                   "etfs/ superseded), cleaned with scripts/eodhd_clean.py:clean_bars (equity "
                   "calendar), availability only (no returns); output frozen at "
                   "$S/runs/S1/availability.json",
    "cleaning_fingerprint": "f62cb2e4a139ea1d3cf240ce938f1d6f573d6208a9e2c004cedee66a47d07ccd",  # v2
    "n_universe": len(UNIVERSE),
    "data_start_note": "etfs_full pulls from inception (SPY 1993-01-29; the 9 original SPDR "
                       "sectors 1998-12-22; XLRE 2015-10; XLC 2018-06); the 2005-01-01 download "
                       "start in the superseded etfs/ folder is gone (Amendment 1)",
    "data_end_common_to_all": DATA_END,
    "n_segments_all_tickers": 1,  # 0 spike-reversal / segment-break events across the universe
    "adv20_eligibility_floor_usd": 1_000_000,
    "adv20_definition": "20-session median of adjusted_close x volume, computed WITHIN the "
                        "bar's own segment (none of these 44 series has a segment break at this "
                        "fingerprint, so this is the plain trailing 20-session median dollar volume)",
    "min_history_calendar_days_for_signal": 210,  # ~7 months: 6-month formation + 1-month skip + buffer
    "n_eligible_at_window_start_2002_01_31": 9,
    "min_n_eligible_over_window": 9,  # never dips below 9 from window start onward (checked through
                                      # all 297 remaining months); REPEATEDLY dips as low as 3
                                      # between 1999 and 2001 on the ADV20 floor -- see module
                                      # docstring, this is real early-sector-SPDR illiquidity, not a bug
    "n_eligible_at_data_end": 43,  # PBJ excluded by its own ADV20 (~$0.37-0.42M, below the $1M floor)
    "pbj_adv20_usd_at_data_end": 422_980,
    "n_months_in_window": 297,
    "universe_growth_note": "the eligible set is NOT static: 9 names at window start, 43 at data "
                            "end, as later sector/industry ETFs season past the ADV floor -- the "
                            "primary benchmark (equal-weight of the SAME point-in-time eligible "
                            "set) absorbs this by construction",
}

# ---------------------------------------------------------------------------
# Universe / eligibility / fallback rule
# ---------------------------------------------------------------------------
ELIGIBILITY = {
    "rule": "on each month-end rebalance date t, ticker i is ELIGIBLE iff (a) i has >= "
           "min_history_calendar_days_for_signal of clean history within its CURRENT "
           "(as-of-t) segment ending at or before t, covering the 6-1 (or 12-1) formation "
           "window plus a buffer, and (b) its ADV20 (see AVAILABILITY) at t is >= "
           "adv20_eligibility_floor_usd. Point-in-time: only information dated <= t is used; "
           "an ETF that later goes illiquid (e.g. PBJ) drops out of future rebalances on its "
           "own via (b), it is never hand-excluded.",
    "min_history_calendar_days": AVAILABILITY["min_history_calendar_days_for_signal"],
    "adv20_floor_usd": AVAILABILITY["adv20_eligibility_floor_usd"],
    "n_min_for_trading": 9,  # below this, HOLD CASH for that month. This is ALSO the window-start
                             # threshold (WINDOW): the eligible count never dips below 9 from
                             # window start onward, so this bar never actually triggers cash
                             # in-window either -- it is still a documented fallback for a data
                             # edge case. Chosen so a "top tercile" of >=9 eligible names still
                             # holds >=3 ETFs.
    "cash_instrument": "BIL total return from BIL's first bar (2007-05-30); FRED DTB3 "
                       "(data/research/fred/DTB3.parquet, rate/100/252 per trading day) before "
                       "that (protocol §2, Amendment 1) -- needed for S1, whose window starts "
                       "2002-01-31, well before BIL existed",
}

# ---------------------------------------------------------------------------
# Evaluation window, fixed from availability only (never from returns).
# ---------------------------------------------------------------------------
WINDOW = {
    "start": "2002-01-31",  # first month-end with >= n_min_for_trading (9) eligible ETFs that
                            # NEVER SUBSEQUENTLY DIPS BELOW 9 through data end
    "end": DATA_END,
    "rationale": "start = first month-end where the point-in-time eligible count reaches "
                "ELIGIBILITY['n_min_for_trading'] (9) AND stays >= 9 every month through data end "
                "(checked over the full remaining 297 months, not just at the crossing date). "
                "NOT simply '9 original sectors x 210 days' (~mid-1999): the binding constraint in "
                "1999-2001 is the $1M ADV20 floor, not history length -- the eligible count "
                "repeatedly falls back to as low as 3 through 2000-2001 as early sector-SPDR "
                "trading volume was genuinely thin, so 2002-01-31 is the first date the >= 9 bar "
                "holds for good. This adds the 2000-02 US equity bear market to the window, a key "
                "regime for momentum crash risk. The universe itself keeps growing after window "
                "start (9 -> 43 eligible names by data end); the primary benchmark is built from "
                "the SAME point-in-time eligible set every month, so it absorbs that growth.",
    "midpoint_date": "2014-05-31",  # halves split (protocol §4); calendar midpoint of start/end,
                                    # fixed at freeze from availability only
}

# ---------------------------------------------------------------------------
# Variant grid (<= 4, per the task). Primary = the paper's own parameters.
# ---------------------------------------------------------------------------
VARIANTS = {
    "primary_6_1_tercile": {
        "formation_months": 6, "skip_months": 1, "selection": "top_tercile",
        "note": "the paper's parameters (Moskowitz & Grinblatt 1999; Jegadeesh-Titman skip "
               "convention) -- the PRIMARY variant",
    },
    "12_1_tercile": {
        "formation_months": 12, "skip_months": 1, "selection": "top_tercile",
        "note": "classic Jegadeesh-Titman alternate formation horizon",
    },
    "6_1_top3": {
        "formation_months": 6, "skip_months": 1, "selection": "top_n", "top_n": 3,
        "note": "the brief's own alternate concentration ('top-3 (or top-tercile)')",
    },
    "12_1_top3": {
        "formation_months": 12, "skip_months": 1, "selection": "top_n", "top_n": 3,
        "note": "combines both alternates",
    },
}
N_VARIANTS = len(VARIANTS)  # 4
PRIMARY_VARIANT = "primary_6_1_tercile"

SELECTION_RULE = {
    "top_tercile": "n_hold = max(1, round(n_eligible / 3)); ties in the signal broken by lower "
                  "ticker symbol (alphabetical) -- a fixed, arbitrary, pre-declared tie-break, "
                  "never by anything data-dependent",
    "top_n": "n_hold = min(top_n, n_eligible); if n_eligible < n_min_for_trading, cash (see "
            "ELIGIBILITY)",
    "weighting": "equal weight across the n_hold held names, rest 0 (long-only, fully invested "
                "in the selection; cash only via ELIGIBILITY['n_min_for_trading'])",
    "signal": "trailing total return over formation_months calendar months of adjusted_close, "
             "ending skip_months calendar months before the rebalance date t (6-1: the 6-month "
             "return ending 1 month before t)",
}

# ---------------------------------------------------------------------------
# Execution / costs (protocol §2, verbatim)
# ---------------------------------------------------------------------------
EXECUTION = {
    "signal_lag_days": 1,  # closes through day t, trade at the NEXT bar's adjusted open
    "rebalance": "monthly, decision at the last close of each calendar month, trade at the "
                "adjusted open of the first trading day of the following month",
    "adjusted_open_formula": "open * adjusted_close / close",
    "weights_drift_between_rebalances": True,
    "robustness_check_not_a_bar": "each variant is also reported with signal_lag_days = 2 (protocol §2)",
}
COSTS = {
    "etf_bps_per_side_adv_ge_50m": 3.0,
    "etf_bps_per_side_adv_lt_50m": 10.0,
    "adv_threshold_usd": 50_000_000,
    "adv_definition": "same 20-session median dollar volume as AVAILABILITY/ELIGIBILITY, evaluated "
                      "per traded ETF at the trade date",
    "stress_multiplier": 2.0,  # bar A5
}

# ---------------------------------------------------------------------------
# Benchmarks (protocol §3, verbatim)
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "BM1_SPY": {"rule": "SPY buy-and-hold, 100%"},
    "BM2_60_40": {"rule": "60% SPY / 40% IEF, rebalanced to target at close t+1 after each month-end t. "
                         "Before IEF's first bar (2002-07-26), the bond leg is VFITX (Vanguard "
                         "Intermediate-Term Treasury, NAV total return, asset='nav' cleaning, from "
                         "1991-12). S1's own WINDOW starts 2002-01-31, i.e. ~6 months before IEF "
                         "exists, so BM2 uses VFITX for that opening stretch of S1's window."},
    "BM3_SPY_VT": {"rule": "SPY weight = min(1, 0.12 / annualised std of last 21 daily SPY returns at "
                          "close t), traded at close t+1 only when |target - held| > 0.10; remainder "
                          "in cash",
                   "target_vol": 0.12, "vol_window": 21, "band_abs": 0.10},
}
PRIMARY_BENCHMARK = {
    "id": "EW_universe_bh",
    "rule": "equal-weight buy-and-hold of the SAME point-in-time ELIGIBLE universe (not the fixed "
           "44-ticker list), rebalanced to equal weight at the same monthly cadence as the "
           "candidate -- isolates the momentum RANKING from sector beta, per protocol §3",
}

# ---------------------------------------------------------------------------
# Inference (protocol §4)
# ---------------------------------------------------------------------------
N_SHORTLIST_CANDIDATES = 5  # S1..S5, protocol §4
BOOTSTRAP = {
    "method": "paired_stationary_block",  # same resampled days for candidate and benchmark
    "mean_block_days": 63,
    "n_boot": 5000,
    "seed": SEED,
    "alpha_one_sided": 0.05 / N_SHORTLIST_CANDIDATES,  # 0.01
}
PLACEBO = {
    "rule": "random rankings with the SAME number of holdings (n_hold, per variant/month) on the "
           "SAME rebalance dates, drawn from that month's eligible universe",
    "n_draws": 500,
    "pass_percentile": 95.0,
    "seed": SEED + 1,
}
PBO = {
    "n_partitions": 8,
    "series": "4 variants + BM1-BM3, daily excess returns over the primary benchmark, WINDOW",
}
DSR = {
    "trials": N_VARIANTS,
    "prior_trials": "190 + other shortlist variants (fixed at freeze)",
    "trial_sharpes": "daily (non-annualised) Sharpe of every declared variant on WINDOW",
    "ledger": "docs/S1_trial_history.json (new family; protocol §6)",
}

# ---------------------------------------------------------------------------
# Tiers (protocol §5, transcribed verbatim; precedence A > D > B > C)
# ---------------------------------------------------------------------------
TIER_A_BARS = [
    {"id": "A1", "rule": "Sharpe gap vs the primary benchmark > 0, and its bootstrap lower bound "
                        "at alpha_one_sided > 0"},
    {"id": "A2", "rule": "DSR > 0.95"},
    {"id": "A3", "rule": "Sharpe above the 95th percentile of the placebo Sharpes"},
    {"id": "A4", "rule": "Sharpe gap > 0 in both halves (split at WINDOW['midpoint_date']), vs the "
                        "primary benchmark and BM1-BM3"},
    {"id": "A5", "rule": "Sharpe gap > 0 at 2x costs, vs the primary benchmark and BM1-BM3"},
    {"id": "A6", "rule": "PBO < 0.50"},
    {"id": "A7", "rule": "an independent recompute reproduces every bar outcome"},
]
TIER_D_RULE = "the bootstrap upper bound at alpha_one_sided of the Sharpe gap is < 0 vs the primary benchmark"
TIER_B_BARS = [
    {"id": "B", "rule": "point estimate > 0 vs the primary benchmark, and A3, A4 and A5 pass"},
]
TIER_ACTIONS = {
    "A": "live proposal (config diff for owner sign-off); the 2-day-lag and per-year audits come first",
    "B": "the owner decides ('not proven')",
    "C": "not deployed",
    "D": "rejected",
}
POST_PASS_AUDIT = ["re-run with signal_lag_days=2", "per-year Sharpe gap table", "independent recompute"]


def classify(bars: dict[str, bool], tier_d: bool, tier_b: dict[str, bool]) -> str:
    """Tier from bar outcomes, in the frozen precedence A > D > B > C."""
    if bars and all(bars.values()):
        return "A"
    if tier_d:
        return "D"
    if tier_b and all(tier_b.values()):
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Power analysis. PHASE 1 CONSTRAINT: no candidate, benchmark or placebo return
# has been computed on WINDOW, so (unlike run_alt_premia_evaluation.py's
# cmd_power, which is allowed to use REAL benchmark returns) every input below
# is a literature/engineering PRIOR, not a measurement. Method generalises the
# alt-premia analytic Jobson-Korkie/Memmel Sharpe-gap MDE to a bet frequency
# below daily (since consecutive monthly rebalances here share 5/6 of their
# 6-month formation window and are NOT close to independent), the same spirit
# as insider_cluster_preregistered_bars.py's "effective bets" haircut on raw
# event counts.
# ---------------------------------------------------------------------------
POWER_PRIORS = {
    # rho: prior correlation of the candidate's daily excess return with the primary
    #   benchmark. High: the candidate longs a subset of the SAME universe the
    #   equal-weight benchmark holds in full, so most of both series' variance is
    #   shared sector/market beta.
    # benchmark_annual_sharpe: prior for the equal-weight sector/industry ETF basket's
    #   own long-run Sharpe -- NOT measured on WINDOW; set near a generic diversified
    #   US-equity-beta prior (consistent with the owner's own 0.4-0.9 detection floor,
    #   research_brief_eodhd_findings.md, bottom of that range since sector baskets add
    #   idiosyncratic sector risk without added return).
    # sharpe_gap_original: M&G (1999)'s own in-sample industry-momentum spread was large
    #   and gross-of-cost (pre-1995 sample); carried over from memory as an UNVERIFIED
    #   rough magnitude this session (no network access in this phase), flagged honestly.
    # sharpe_gap_decayed: post-publication/crowding decay is well documented for momentum
    #   generally (e.g. McLean & Pontiff 2016-style "predictability decay after
    #   publication" finding, cited from memory, not re-verified live this session) --
    #   applied here as a blunt ~55-60% haircut, not a per-paper-specific number.
    "primary_6_1_tercile": {"rho": 0.75, "benchmark_annual_sharpe": 0.40,
                            "sharpe_gap_original": 0.55, "sharpe_gap_decayed": 0.25},
    "12_1_tercile": {"rho": 0.70, "benchmark_annual_sharpe": 0.40,
                     "sharpe_gap_original": 0.45, "sharpe_gap_decayed": 0.20},
    "6_1_top3": {"rho": 0.55, "benchmark_annual_sharpe": 0.40,
                "sharpe_gap_original": 0.60, "sharpe_gap_decayed": 0.28},
    "12_1_top3": {"rho": 0.50, "benchmark_annual_sharpe": 0.40,
                 "sharpe_gap_original": 0.50, "sharpe_gap_decayed": 0.22},
}
BET_FREQUENCY_PRIORS = {
    # Consecutive month-end 6-1 signals share 5 of 6 formation months (83% overlap), so
    # "one independent piece of information" arrives roughly every formation-window
    # length, not every rebalance. Three honest scenarios, none used to pass/fail
    # anything -- this is reported, not gated.
    "conservative": 2,   # one independent bet every ~6 months (= the formation window length)
    "central": 4,        # partial decorrelation: one independent bet every ~3 months
    "naive_monthly": 12,  # upper bound if every monthly rebalance were fully independent
                         # (not believed; included only to bound the range)
}
POWER = {
    "years_available": 24.66,  # (date(2026,9,29) - date(2002,1,31)).days / 365.25 = 9007/365.25
    "alpha_one_sided": BOOTSTRAP["alpha_one_sided"],
    "target_power": 0.80,
}


def _required_mde_sharpe_gap(benchmark_annual_sharpe: float, rho: float, periods_per_year: float,
                              n_periods: float, alpha_one_sided: float, power: float) -> float:
    """Analytic Jobson-Korkie/Memmel minimum-detectable annualised Sharpe gap.

    Same fixed-point iteration as run_alt_premia_evaluation.py:cmd_power's
    ``g = (z_a + z_b) * SE(g)``, generalised from daily periods (ANN = sqrt(252))
    to an arbitrary ``periods_per_year`` (here, a prior "effective independent
    bet" rate well below 252 or even 12) -- this file never has real returns to
    feed it, so every quantity is a declared prior, not a measurement.
    """
    from firm.eval.overfitting import _norm_ppf

    z_a = _norm_ppf(1 - alpha_one_sided)
    z_b = _norm_ppf(power)
    ann = math.sqrt(periods_per_year)
    sb_per_period = benchmark_annual_sharpe / ann
    g = 0.5
    for _ in range(50):
        sc_per_period = (benchmark_annual_sharpe + g) / ann
        var_gap = (2 * (1 - rho) + 0.5 * (sb_per_period**2 + sc_per_period**2
                                          - 2 * sb_per_period * sc_per_period * rho**2)) / n_periods
        g = (z_a + z_b) * math.sqrt(var_gap) * ann
    return g


def power_report() -> dict:
    """MDE at 80% power, alpha 0.01, vs literature Sharpe-gap priors, for every
    (variant x bet-frequency scenario), using PRIORS only (see module note)."""
    rows = []
    for variant, priors in POWER_PRIORS.items():
        for scenario, periods_per_year in BET_FREQUENCY_PRIORS.items():
            n_periods = periods_per_year * POWER["years_available"]
            mde = _required_mde_sharpe_gap(
                priors["benchmark_annual_sharpe"], priors["rho"], periods_per_year,
                n_periods, POWER["alpha_one_sided"], POWER["target_power"],
            )
            for effect_key in ("sharpe_gap_original", "sharpe_gap_decayed"):
                claimed = priors[effect_key]
                rows.append({
                    "variant": variant, "bet_frequency_scenario": scenario,
                    "periods_per_year": periods_per_year, "n_periods": round(n_periods, 1),
                    "rho_prior": priors["rho"], "benchmark_sharpe_prior": priors["benchmark_annual_sharpe"],
                    "literature_effect": effect_key, "literature_sharpe_gap": claimed,
                    "mde_sharpe_gap_80pct_power": round(mde, 3),
                    "adequately_powered": bool(mde <= claimed),
                })
    n_adequate = sum(1 for r in rows if r["adequately_powered"])
    return {
        "rows": rows,
        "n_rows": len(rows),
        "n_adequately_powered": n_adequate,
        "headline": (
            "NOT adequately powered against the decayed (post-publication-realistic) effect "
            "size under the conservative/central bet-frequency priors -- only the naive_monthly "
            "(not believed) scenario and/or the original (pre-decay) effect size clear the bar "
            "for most variants. Consistent with the research brief's own verdict: '~20-30 assets, "
            "12 rebalances/year -> modest bet count ... underpowered for a clean statistical "
            "verdict at 20 years of monthly data.' Treat any single-history WINDOW result as "
            "indicative only; the placebo/PBO/DSR machinery (protocol §4) is the load-bearing "
            "check here, not the point-estimate Sharpe gap alone."
        ),
        "availability": AVAILABILITY,
    }


def bars_fingerprint() -> str:
    payload = json.dumps(
        {
            "PREREGISTERED_AT": PREREGISTERED_AT, "DATA_END": DATA_END, "SEED": SEED,
            "UNIVERSE": UNIVERSE, "AVAILABILITY": AVAILABILITY, "ELIGIBILITY": ELIGIBILITY,
            "WINDOW": WINDOW, "VARIANTS": VARIANTS, "SELECTION_RULE": SELECTION_RULE,
            "EXECUTION": EXECUTION, "COSTS": COSTS, "BENCHMARKS": BENCHMARKS,
            "PRIMARY_BENCHMARK": PRIMARY_BENCHMARK, "BOOTSTRAP": BOOTSTRAP, "PLACEBO": PLACEBO,
            "PBO": PBO, "DSR": DSR, "TIER_A_BARS": TIER_A_BARS, "TIER_D_RULE": TIER_D_RULE,
            "TIER_B_BARS": TIER_B_BARS, "TIER_ACTIONS": TIER_ACTIONS,
            "POST_PASS_AUDIT": POST_PASS_AUDIT, "POWER_PRIORS": POWER_PRIORS,
            "BET_FREQUENCY_PRIORS": BET_FREQUENCY_PRIORS, "POWER": POWER,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    print(("DRAFT" if DRAFT else "FROZEN") + " fingerprint:", bars_fingerprint())
    import pprint
    pprint.pprint(power_report())
