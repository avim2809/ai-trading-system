"""DRAFT pre-registration for the SEC insider-purchase-cluster candidate.

STATUS: DRAFT, not frozen for evaluation. Unlike
``scripts/alt_premia_preregistered_bars.py`` (frozen before any return series
existed), this candidate has NO PRICE DATA YET for its event universe — see
docs/research_findings_beyond_equities_2026_09_30.md §"SEC-filings-driven
insider-purchase clustering" and the vendor recommendation in this session's
final report. The event list itself (``compute_cluster_events`` in
``src/firm/strategies/insider_cluster.py``) IS real, computed from the full
SEC DERA Insider Transactions Data Sets, 2006q1-2026q2 (see
``scripts/fetch_insider_data.py``). This file freezes the evaluation DESIGN
now, against the real event counts, so the design can't be quietly reshaped
once return data starts arriving — but it must stay DRAFT and MUST NOT be
promoted to "frozen" until:
    1. a price vendor is chosen and paid for (owner sign-off — see this
       session's report for the recommendation and cost), and
    2. daily prices (including delisted names) are actually loaded for the
       ~8,800 distinct tickers appearing in the 2006-2026 event list.

Observed inputs this design is built against (real, computed 2026-09-30):
    - 1,208,663 point-in-time open-market-purchase rows, 2006q1-2026q2
      (after amendment dedup), 13,982 distinct issuers, 17,050 distinct
      tickers across ALL purchases (some issuers span multiple tickers
      over 20 years via renames).
    - 33,258 cluster events (>=3 distinct insiders, >=1 opportunistic,
      30-calendar-day window) across 8,131 distinct issuers, 8,815 distinct
      tickers. ~1,400 events/year in the 2013-2025 window (2006-2008 ran
      much higher, ~2,000-3,000/year — plausibly a crisis-era effect, not
      necessarily the "clustering itself decayed" story; this design does
      not assume which).
    - Only 22 of 33,258 events (0.07%) fall in this system's current fixed
      25-name mega-cap universe — confirms the research brief's own
      prediction almost exactly ("expect ~none; that's the point"): this
      candidate is structurally incompatible with today's book and requires
      the small/mid-cap universe expansion described below.

Question: does trading pre-defined insider-purchase-cluster events, entered
at the next tradable session after the cluster is fully public, beat a
size-matched buy-and-hold benchmark and a placebo (random entry dates on the
same tickers), net of realistic small/mid-cap costs, after deflating for
this search's trial count?
"""

from __future__ import annotations

import hashlib
import json

DRAFT = True  # must be flipped to False (and PREREGISTERED_AT set) only once price data exists
PREREGISTERED_AT = None  # set at the moment this file is frozen, not before
DATA_END_INSIDER_EVENTS = "2026-06-30"  # last complete quarter in the SEC DERA data set as of 2026-09-30
TRADING_DAYS = 252
SEED = 20260930

# ---------------------------------------------------------------------------
# Observed inputs (real, from the computed event list — see module docstring)
# ---------------------------------------------------------------------------
OBSERVED = {
    "n_purchase_rows_deduped": 1_208_663,
    "n_distinct_issuers_all_purchases": 13_982,
    "n_distinct_tickers_all_purchases": 17_050,
    "n_cluster_events_total": 33_258,
    "n_distinct_issuers_with_events": 8_131,
    "n_distinct_tickers_in_events": 8_815,
    "events_per_year_2013_2025_avg": 1_398,
    "events_in_current_25_name_universe": 22,
    "unresolved_ticker_rate": 0.004,  # 4,563 / 1,208,663 — see insider_transactions.resolve_tickers
}

# ---------------------------------------------------------------------------
# Universe filters — keeps costs sane on a small/mid-cap book (per the
# research brief's own pre-registrable rule, docs/research_findings_..., and
# the task's own instruction to floor on market cap / ADV).
# ---------------------------------------------------------------------------
UNIVERSE = {
    "market_cap_min_usd": 50_000_000,
    "market_cap_max_usd": 2_000_000_000,
    "adv_min_usd_20d": 500_000,
    "note": "Market cap is NOT available from this dataset (SEC Form 4/5 carry no "
            "market-cap field) — see this session's report. Applying this filter "
            "for real requires a shares-outstanding x price join against the chosen "
            "price vendor once selected; until then this is a DESIGN placeholder, "
            "not something already checked against the 33,258 events above.",
}

# ---------------------------------------------------------------------------
# Execution / cost model
# ---------------------------------------------------------------------------
EXECUTION = {
    # known_date = max(SUBMISSION.FILING_DATE) across the qualifying window's
    # purchases (see compute_cluster_events) — a calendar DATE, no intraday
    # timestamp in the source data (see insider_transactions.py docstring).
    # Entry is therefore the NEXT tradable session's OPEN after known_date,
    # not close-of-known_date (unlike this system's usual t -> t+1 close
    # convention) — filings commonly land after the 5:30pm ET EDGAR cutoff,
    # so acting at the same day's close would already be look-ahead for a
    # same-day-after-hours filing, and acting at next-day open is the more
    # conservative, defensible choice for an event with no intraday
    # timestamp at all.
    "entry_rule": "next tradable session's OPEN strictly after known_date",
    "holds_trading_days": {"3_month": 63, "6_month": 126},
    "signal_lag_days": 1,
}
COSTS = {
    # Cited to the SEC's own market-quality research (verified 2026-09-30):
    #   https://www.sec.gov/marketstructure/research/small_cap_liquidity.pdf
    #   ("A Characterization of Market Quality for Small Capitalization US
    #   Equities" — median effective spreads by cap bucket)
    #   https://www.sec.gov/data-research/sec-markets-data/marketstructuredata-decile-quartile
    #   (SEC's own decile/quartile summary metrics, ongoing series)
    # The exact 0.92-1.235% smallest-decile figure is carried over from the
    # research brief, NOT independently re-derived from the raw decile files
    # in this pass — flagged UNVERIFIED-exact-figure, sourced-to-real-data.
    "round_trip_bps_by_adv_bucket": {
        "adv_lt_1m": 120.0,     # smallest-decile proxy, UNVERIFIED-exact-figure
        "adv_1m_5m": 60.0,
        "adv_5m_20m": 30.0,
        "adv_gt_20m": 10.0,
    },
    "stress_multiplier": 2.0,
    "source": "SEC small-cap liquidity study + decile/quartile summary metrics (both verified live 2026-09-30); "
              "exact bps figures carried over from the research brief, not re-derived here",
}

# ---------------------------------------------------------------------------
# Benchmark and placebo
# ---------------------------------------------------------------------------
BENCHMARK = {
    "primary": "size-decile-matched buy-and-hold return over the same hold window "
               "(true CRSP/Sharadar size deciles if the chosen price vendor provides "
               "market cap; else a small/mid-cap ETF blend proxy — IJR (small) / IJH "
               "(mid) weighted by which band the event ticker's ADV bucket falls in "
               "— a documented approximation, not a true size-matched control)",
    "secondary": "IWM (Russell 2000) buy-and-hold, same window, unmatched — reported not gated",
}
PLACEBO = {
    "method": "for each event ticker, draw as many random known_dates (uniform over that ticker's "
              "tradable history, excluding +/- 63 trading days around any real event for that ticker) "
              "as it has real events; same entry/hold/cost rules",
    "n_draws": 500,
    "pass_percentile": 95.0,
    "seed": SEED + 1,
}

# ---------------------------------------------------------------------------
# Windows. See module docstring: CMP's (2012) own SAMPLE ends 2007, so the
# strongest un-contaminated out-of-sample test starts right after that
# (2008), not after the paper's 2012 PUBLICATION date — waiting for 2012
# conflates "did the underlying pattern already decay" with "did publication
# cause crowding," and the task's own correction prefers the former as
# primary.
# ---------------------------------------------------------------------------
WINDOWS = {
    "primary_post_sample": {"start": "2008-01-01", "end": DATA_END_INSIDER_EVENTS,
                            "rationale": "immediately after CMP's own 1986-2007 sample — the strongest "
                                        "test of whether the pattern had already decayed, independent of "
                                        "publication-driven crowding"},
    "secondary_post_publication": {"start": "2013-01-01", "end": DATA_END_INSIDER_EVENTS,
                                   "rationale": "after CMP's 2012 JF publication — tests publication-driven crowding specifically"},
    "full_available": {"start": "2006-04-01", "end": DATA_END_INSIDER_EVENTS},
}

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
BOOTSTRAP = {
    "method": "paired_stationary_block over event-level (not daily) returns, block = 8 events",
    "n_boot": 5000,
    "seed": SEED,
    # 2 holds x 2 primary windows x 2 benchmarks (primary + secondary) = 8 nominal comparisons.
    "n_comparisons_bonferroni": 8,
    "alpha_one_sided": 0.05 / 8,
}
DSR = {
    "trials": 8,
    "prior_trials": 0,
    "ledger": "docs/insider_cluster_trial_history.json (NEW file — does not exist yet; "
              "created only on the first real --append-ledger run, same convention as "
              "docs/alt_premia_trial_history.json)",
}
PBO = {"n_partitions": 8, "series": "event-level excess returns, both holds, both primary windows"}

# ---------------------------------------------------------------------------
# Power calculation from the OBSERVED event counts (real numbers above), not
# from a return series that doesn't exist yet — this is the task's own
# explicit ask ("a power calculation using the observed event counts").
#
# Method: a simple two-sample-mean detection-power calculation per hold
# period, NOT the Jobson-Korkie/Memmel Sharpe-gap machinery in
# run_alt_premia_evaluation.py (that requires a realised daily return
# series for both the candidate and the benchmark, which doesn't exist for
# this candidate yet). Treats each cluster event as one approximately
# independent bet after a documented decorrelation haircut (events cluster
# in time — e.g. 2008 alone contributed ~3,000 of the 33,258 total, plainly
# not independent of each other), required N solved from:
#     N = ((z_alpha + z_beta) * sigma / mu) ** 2
# ---------------------------------------------------------------------------
POWER = {
    "effective_bets_per_year": {
        "conservative": 300,   # low end of the research brief's own 300-1500/year estimate
        "central": 700,        # brief's rough midpoint
    },
    "effect_size_bps_per_hold": {
        # CMP (2012)'s own 82bp/month figure, simple (not compounded) sum over the hold:
        "original_cmp_3mo": 82 * 3, "original_cmp_6mo": 82 * 6,
        # midpoint of the UNVERIFIED "~1/3 of original" 2024 replication (0.2-0.3%/month):
        "decayed_replication_3mo": 25 * 3, "decayed_replication_6mo": 25 * 6,
    },
    "per_bet_return_stdev_bps": {
        # Small/micro-cap idiosyncratic volatility proxy, NOT re-derived from
        # vendor data yet (no price data exists — see module docstring).
        # ~45%/year assumed idiosyncratic vol for a small/micro-cap single
        # name (well above the Russell 2000 INDEX's ~18-25%/year — a basket,
        # not a single stock); scaled to the hold length by sqrt(time).
        "3mo": 4500 * (3 / 12) ** 0.5,
        "6mo": 4500 * (6 / 12) ** 0.5,
    },
    "alpha_one_sided": BOOTSTRAP["alpha_one_sided"],
    "target_power": 0.80,
    "available_years_primary_window": 2026 - 2008,  # through 2026-06, treated as a full year here
}


def required_n(mu_bps: float, sigma_bps: float, alpha_one_sided: float, power: float) -> float:
    """N = ((z_alpha + z_beta) * sigma / mu) ** 2 for a one-sample mean-detection test."""
    from firm.eval.overfitting import _norm_ppf
    z_a = _norm_ppf(1 - alpha_one_sided)
    z_b = _norm_ppf(power)
    return ((z_a + z_b) * sigma_bps / mu_bps) ** 2


def power_report() -> dict:
    """Required N vs. available effective bets, for every (effect size x hold) pair."""
    rows = []
    for hold in ("3mo", "6mo"):
        sigma = POWER["per_bet_return_stdev_bps"][hold]
        for effect_key in (f"original_cmp_{hold}", f"decayed_replication_{hold}"):
            mu = POWER["effect_size_bps_per_hold"][effect_key]
            n_req = required_n(mu, sigma, POWER["alpha_one_sided"], POWER["target_power"])
            for bet_rate_key, bets_per_year in POWER["effective_bets_per_year"].items():
                n_available = bets_per_year * POWER["available_years_primary_window"]
                rows.append({
                    "hold": hold, "effect": effect_key, "mu_bps": mu, "sigma_bps": round(sigma, 1),
                    "bet_rate_scenario": bet_rate_key, "effective_bets_per_year": bets_per_year,
                    "n_required_80pct_power": round(n_req, 1),
                    "n_available_2008_2026": n_available,
                    "adequately_powered": bool(n_available >= n_req),
                })
    return {"rows": rows, "observed": OBSERVED}


# ---------------------------------------------------------------------------
# Tiers — same frozen-precedence pattern as alt_premia_preregistered_bars.py.
# ---------------------------------------------------------------------------
TIER_A_BARS = [
    {"id": "A1", "rule": "vs primary benchmark, BOTH holds: mean excess return > 0 AND bootstrap "
                         "one-sided lower bound at alpha_one_sided > 0"},
    {"id": "A2", "rule": "DSR > 0.95 on the primary post-sample window (2008-2026)"},
    {"id": "A3", "rule": "event excess return > 95th percentile of its placebo distribution"},
    {"id": "A4", "rule": "primary post-sample window (2008+) AND secondary post-publication window "
                         "(2013+) both show a positive point-estimate excess return"},
    {"id": "A5", "rule": "excess return remains positive with costs at stress_multiplier"},
    {"id": "A6", "rule": "CSCV PBO < 0.50"},
]
TIER_D_RULE = "vs primary benchmark, either hold: bootstrap one-sided upper bound at alpha_one_sided < 0"
TIER_B_BARS = [
    {"id": "B_a", "rule": "bootstrap one-sided lower bound at alpha_one_sided > -0.10 x sigma"},
    {"id": "B_b", "rule": "A3 and A5 pass"},
]
TIER_ACTIONS = {
    "A": "small/mid-cap live proposal (config diff + universe-expansion diff for owner sign-off)",
    "B": "flagged 'not proven alpha' — owner decides",
    "C": "not deployed",
    "D": "rejected",
}


def classify(bars: dict[str, bool], tier_d: bool, tier_b: dict[str, bool]) -> str:
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
            "DATA_END_INSIDER_EVENTS": DATA_END_INSIDER_EVENTS, "SEED": SEED, "OBSERVED": OBSERVED,
            "UNIVERSE": UNIVERSE, "EXECUTION": EXECUTION, "COSTS": COSTS, "BENCHMARK": BENCHMARK,
            "PLACEBO": PLACEBO, "WINDOWS": WINDOWS, "BOOTSTRAP": BOOTSTRAP, "DSR": DSR, "PBO": PBO,
            "POWER": POWER, "TIER_A_BARS": TIER_A_BARS, "TIER_D_RULE": TIER_D_RULE,
            "TIER_B_BARS": TIER_B_BARS, "TIER_ACTIONS": TIER_ACTIONS,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    print("DRAFT fingerprint (will change until frozen — see module docstring):", bars_fingerprint())
    import pprint
    pprint.pprint(power_report())
