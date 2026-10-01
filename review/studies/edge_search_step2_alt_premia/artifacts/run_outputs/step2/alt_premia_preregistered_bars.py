"""FROZEN pre-registration for the 2026-09-29 alternative-premia edge search (Step 2).

Plan: docs/edge_search_plan_2026_09.md (v2, approved with defaults 2026-09-29).
Written and fingerprinted BEFORE any candidate return series was computed. Only
raw data had been downloaded and quality-checked (alt_premia_data.py). Do not
edit after results exist. A new hypothesis needs a new file and a new ledger
entry in docs/alt_premia_trial_history.json.

Question: does any of these pre-specified, literature-parameterised return
sources (or a fixed-rule combination of them) beat buy-and-hold SPY, 60/40 and
vol-targeted SPY out of sample, net of costs, after deflating for this search's
trial count?

Changes from the v2 plan text, made after the data QC and before any candidate
ran (none depends on a candidate's performance):
- V2: CBOE's PUT history is daily only from 2007 (sparse before), so V2 starts
  2007-02 rather than 1991.
- V1: uses SVXY's own traded returns (exposure factor 1 before 2018-02-28, 2 after
  the -1x to -0.5x change), not a synthetic -1 x VXX series. QC showed VXX
  close-to-close shows +33% on 2018-02-05 while SVXY lost ~85% on 2018-02-06, so
  the synthetic series would hide the real tail. V1 therefore starts 2011-12, not 2009-10.
- V1 size cap: 0.30 -> 0.25 of NAV, so that a total wipeout of the short-vol
  product loses at most 25% of NAV (the plan's own tail bar). The bar now holds
  by construction; it is still checked.
- C1 cost: 25 bps/side (Alpaca crypto base-tier taker fee), not the ~10 bps in the
  plan. C1 is compared with BTC buy-and-hold at matched exposure (plan §5.1). The
  SPY/60-40/VT comparisons are reported, not gated.
- Fund expense ratios (SVXY, ETFs) are already embedded in adjClose, so they are
  not charged again.
- K1 vs BM2 is evaluated from 2002-09 (IEF inception 2002-07). K1 vs BM1/BM3 uses 1994-02 onward.
"""

from __future__ import annotations

import hashlib
import json

PREREGISTERED_AT = "2026-09-29T11:40:00Z"

DATA_END = "2026-09-28"
TRADING_DAYS = 252
SEED = 20260929

# ---------------------------------------------------------------------------
# Shared execution / cost model
# ---------------------------------------------------------------------------
EXECUTION = {
    # Data signals: computed from closes <= t, the position is set at the close of
    # t+1, so it first earns the return close(t+1) -> close(t+2).
    "signal_lag_days": 1,
    # Calendar signals (K1) come from a schedule that is public in advance. The
    # position is entered at the close before the first scheduled return day.
    "calendar_known_in_advance": True,
    "weights_drift_between_rebalances": True,
    "cash_rate": "FRED DTB3 / 100 / 252 per trading day, forward-filled",
    "max_gross": 1.0,
}
COSTS = {
    "etf_bps_per_side": 5.0,
    "btc_bps_per_side": 25.0,
    "put_write_drag_bps_per_year": 60.0,   # V2 implementation drag (option spreads/rolls)
    "stress_multiplier": 2.0,               # bar A5
}

# ---------------------------------------------------------------------------
# Benchmarks (not selected, reported on each candidate's own window)
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "BM1_SPY": {"rule": "SPY buy-and-hold, 100%"},
    "BM2_60_40": {"rule": "60% SPY / 40% IEF, rebalanced to target at close t+1 after each month-end t"},
    "BM3_SPY_VT": {"rule": "SPY weight = min(1, 0.12 / annualised std of last 21 daily SPY returns at close t), "
                           "traded at close t+1 only when |target - held| > 0.10; remainder in cash",
                   "target_vol": 0.12, "vol_window": 21, "band_abs": 0.10},
}

# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------
CANDIDATES = {
    "V1_short_vol_timed": {
        "instrument": "SVXY (exposure factor L=1 before 2018-02-28, L=0.5 from 2018-02-28)",
        "signal": "on iff VIX close_t < VIX3M close_t (contango)",
        "sizing": "short-vol index exposure e_t = min(0.25, 0.10 / sigma_t), sigma_t = annualised std of last 21 "
                  "daily index returns (r_SVXY / L); SVXY weight = e_t / L",
        "rebalance": "daily check; trade if signal flips, or |target_e - held_e| > 0.25 * held_e, or held_e (after drift) > exposure_cap",
        "exposure_cap": 0.25, "target_vol": 0.10, "vol_window": 21, "band_rel": 0.25,
        "eval_start": "2011-12-01", "post_publication_start": "2015-01-01",
        "source": "Simon & Campasano (2014), VIX futures basis; term-structure filter",
        "benchmarks": ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"],
    },
    "V2_put_write": {
        "instrument": "CBOE PUT index (collateralised 1-month ATM SPX put-write, includes T-bill interest)",
        "signal": "always held, 100%",
        "costs": "put_write_drag_bps_per_year, charged daily",
        "eval_start": "2007-02-01", "post_publication_start": "2010-01-01",
        "source": "CBOE PUT index; Ungar & Moran (2009)",
        "benchmarks": ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"],
    },
    "K1_calendar": {
        "instrument": "SPY",
        "signal": "hold SPY for the returns of: the last trading day of each month and the first 3 trading days "
                  "of the next (turn of month), and each scheduled FOMC announcement day (return close(t-1) -> close(t)); "
                  "cash otherwise",
        "trading_days_source": "SPY trading dates in the data",
        "fomc_source": "federalreserve.gov scheduled meetings, last day of meeting (fomc_scheduled.parquet)",
        "eval_start": "1994-02-01", "post_publication_start": "2009-01-01",
        "bm2_eval_start": "2002-09-01",
        "source": "Lakonishok & Smidt (1988); McConnell & Xu (2008); Lucca & Moench (2015)",
        "benchmarks": ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"],
    },
    "C1_btc_trend": {
        "instrument": "BTC/USD spot (Tiingo crypto daily, UTC bars)",
        "signal": "reviewed each Sunday (UTC) on day-d close: on iff close_d / close_{d-28} - 1 > 0",
        "sizing": "weight when on = min(1, 0.40 / sigma_d), sigma_d = std of last 63 daily BTC returns x sqrt(365)",
        "rebalance": "at review, trade if on/off flips or |target - held| > 0.10; position effective from day d+1's close (1-day lag)",
        "target_vol": 0.40, "vol_window": 63, "lookback_days": 28, "band_abs": 0.10,
        "calendar": "BTC UTC-day bar d is compounded onto the first SPY trading date strictly after d (a bar closes 3-4h after the US close, so same-date mapping would leak)",
        "eval_start": "2015-02-01", "post_publication_start": "2021-01-01",
        "source": "Liu & Tsyvinski (2021), time-series momentum at 1-4 week horizons",
        "benchmarks": ["BTC_BH_MATCHED"],
        "benchmark_rule": "BTC_BH_MATCHED = constant BTC weight equal to C1's average weight over the window, "
                          "rebalanced monthly at the same cost",
        "reported_only_vs": ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"],
    },
    "T2_cross_asset_trend": {
        "instrument": ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "VNQ"],
        "signal": "at month-end close t, asset on iff its 252-day adjClose return > compounded cash return over the same 252 days",
        "sizing": "on-assets weighted by 1/sigma63_i, normalised to 1; scale s = min(1, 0.10 / sigma_p), sigma_p = "
                  "annualised std of the last 63 daily returns of that weighted basket; final w = s * w",
        "rebalance": "monthly, at close t+1",
        "target_vol": 0.10, "vol_window": 63, "lookback_days": 252,
        "eval_start": "2007-03-01", "post_publication_start": "2013-01-01",
        "source": "Moskowitz, Ooi & Pedersen (2012), long/flat, no leverage",
        "benchmarks": ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"],
    },
    "M1_combo": {
        "streams": ["BM1_SPY", "T2_cross_asset_trend", "V1_short_vol_timed", "K1_calendar"],
        "rule": "at each month-end close t, stream weights proportional to 1 / annualised std of each stream's last 63 daily "
                "net returns, normalised to 1; set at close t+1, drift in between; rebalance cost = sum |dw| x etf_bps "
                "(stream-level, no netting)",
        "vol_window": 63,
        "eval_start": "2012-04-01", "post_publication_start": "2015-01-01",
        "benchmarks": ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"],
    },
    "M2_combo_crypto": {
        "streams": ["BM1_SPY", "T2_cross_asset_trend", "V1_short_vol_timed", "K1_calendar", "C1_btc_trend"],
        "rule": "as M1 with C1 as a 5th stream",
        "vol_window": 63,
        "eval_start": "2015-05-01", "post_publication_start": "2021-01-01",
        "benchmarks": ["BM1_SPY", "BM2_60_40", "BM3_SPY_VT"],
    },
}
N_CANDIDATES = len(CANDIDATES)  # Bonferroni divisor

# Priors for the power analysis only (from each family's literature, rough). They
# are never used to pass or fail anything.
POWER_PRIORS = {
    # candidate: (correlation of daily excess returns with the benchmark, literature-claimed annual Sharpe gap)
    "V1_short_vol_timed": (0.6, 0.4),
    "V2_put_write": (0.85, 0.1),
    "K1_calendar": (0.5, 0.3),
    "C1_btc_trend": (0.7, 0.5),
    "T2_cross_asset_trend": (0.3, 0.1),
    "M1_combo": (0.6, 0.4),
    "M2_combo_crypto": (0.55, 0.4),
}

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
BOOTSTRAP = {
    "method": "paired_stationary_block",  # same resampled days for candidate and benchmark
    "mean_block_days": 63,
    "n_boot": 5000,
    "seed": SEED,
    "alpha_one_sided": 0.05 / N_CANDIDATES,
}
PLACEBO = {
    "n_draws": 500,
    "pass_percentile": 95.0,
    "seed": SEED + 1,
    "rules": {
        "V1_short_vol_timed": "daily on/off signal permuted in 63-day blocks; sizing path unchanged",
        "K1_calendar": "each month: a random 4-consecutive-trading-day window inside that month replaces the TOM days, "
                       "and each FOMC day is replaced by a random non-held trading day of the same month",
        "C1_btc_trend": "weekly on/off states permuted in 13-week blocks; sizing path unchanged",
        "T2_cross_asset_trend": "each asset's monthly on/off series permuted in 12-month blocks, independently per asset; "
                                "weights recomputed with the same vol logic",
        "M1_combo": "same allocation rule applied to the placebo streams of the same draw (SPY stream unchanged)",
        "M2_combo_crypto": "same allocation rule applied to the placebo streams of the same draw (SPY stream unchanged)",
        "V2_put_write": None,  # untimed; A3 not applicable
    },
}
PBO = {"n_partitions": 8, "window_start": "2015-05-01",
       "series": "all 7 candidates + 3 benchmarks, daily excess returns (C1 on the SPY calendar)"}
DSR = {"trials": N_CANDIDATES + len(BENCHMARKS), "prior_trials": 0,
       "trial_sharpes": "daily (non-annualised) Sharpe of every candidate and benchmark on the PBO window",
       "ledger": "docs/alt_premia_trial_history.json (new family; combination=52 and pattern_ml=104 ledgers untouched)"}
VOL_MATCH = ("for drawdown comparisons the benchmark is rescaled to the candidate's realised vol over the window: "
             "r = rf + lambda * (r_b - rf), lambda = sigma_c / sigma_b, a reporting construct with no costs")
TAIL = {"applies_to": ["V1_short_vol_timed", "M1_combo", "M2_combo_crypto"],
        "rule": "max over days of short-vol index exposure held (V1 e_t, or stream weight x e_t inside M1/M2) <= 0.25, "
                "i.e. a total wipeout of the short-vol product loses at most 25% of NAV"}

# ---------------------------------------------------------------------------
# Tiers. Each candidate lands in exactly one, checked in the order A, D, B, C.
# ---------------------------------------------------------------------------
TIER_A_BARS = [
    {"id": "A1", "rule": "vs every benchmark: Sharpe(c) - Sharpe(b) > 0 AND bootstrap one-sided lower bound at alpha_one_sided > 0"},
    {"id": "A2", "rule": "DSR(c on its window) > 0.95 with DSR trials"},
    {"id": "A3", "rule": "Sharpe(c) > 95th percentile of its placebo Sharpes (n/a for V2)"},
    {"id": "A4", "rule": "vs every benchmark: Sharpe gap point estimate > 0 on the post-publication window"},
    {"id": "A5", "rule": "vs every benchmark: Sharpe gap point estimate > 0 with every cost x stress_multiplier"},
    {"id": "A6", "rule": "vs every benchmark: maxDD(c) <= maxDD(vol-matched b)"},
    {"id": "A7", "rule": "CSCV PBO < 0.50 (search-wide)"},
    {"id": "TAIL", "rule": "TAIL rule, where it applies"},
]
TIER_D_RULE = "vs any benchmark: bootstrap one-sided upper bound at alpha_one_sided of the Sharpe gap < 0"
TIER_B_BARS = [
    {"id": "B_a", "rule": "vs every benchmark: bootstrap one-sided lower bound at alpha_one_sided of the Sharpe gap > -0.10"},
    {"id": "B_b", "rule": "vs every benchmark: maxDD(c) <= 0.75 x maxDD(vol-matched b)"},
    {"id": "B_c", "rule": "vs every benchmark: mean candidate return over the 3 worst non-overlapping peak-to-trough "
                          "episodes of the vol-matched benchmark >= 0.5 x the benchmark's mean episode return"},
    {"id": "B_d", "rule": "A3, A4 and A5 pass"},
    {"id": "TAIL", "rule": "TAIL rule, where it applies"},
]
TIER_ACTIONS = {
    "A": "live proposal (config diff for owner sign-off)",
    "B": "live proposal labelled 'risk improvement, not proven alpha'; owner decides",
    "C": "not deployed; that slot's recommendation is the fallback benchmark",
    "D": "rejected",
}
FALLBACK_BENCHMARK = "BM2_60_40"  # owner decision D1 = default (no drawdown tolerance given)
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


def bars_fingerprint() -> str:
    payload = json.dumps(
        {
            "PREREGISTERED_AT": PREREGISTERED_AT, "DATA_END": DATA_END, "SEED": SEED,
            "EXECUTION": EXECUTION, "COSTS": COSTS, "BENCHMARKS": BENCHMARKS,
            "CANDIDATES": CANDIDATES, "POWER_PRIORS": POWER_PRIORS, "BOOTSTRAP": BOOTSTRAP,
            "PLACEBO": PLACEBO, "PBO": PBO, "DSR": DSR, "VOL_MATCH": VOL_MATCH, "TAIL": TAIL,
            "TIER_A_BARS": TIER_A_BARS, "TIER_D_RULE": TIER_D_RULE, "TIER_B_BARS": TIER_B_BARS,
            "TIER_ACTIONS": TIER_ACTIONS, "FALLBACK_BENCHMARK": FALLBACK_BENCHMARK,
            "POST_PASS_AUDIT": POST_PASS_AUDIT,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    print(bars_fingerprint())
