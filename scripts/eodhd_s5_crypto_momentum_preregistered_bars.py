"""DRAFT pre-registration for the S5 crypto cross-sectional momentum candidate.

Protocol: docs/eodhd_shortlist_protocol_2026_10.md (amended once, 2026-09-30
22:40Z -- this file already reflects amendment 1: cleaning v2 fingerprint,
BM1-3 sourced from ``etfs_full/``, cash = BIL/DTB3). Candidate brief:
docs/research_brief_eodhd_findings.md Sec 1 candidate #5. THE GATE (must pass
before this file exists at all): docs/eodhd_s5_crypto_gate_2026_10.md --
**PASSED** 2026-09-30 (commit fdca0e7): delisted coins that were once large
are present in data/research/eodhd/crypto/ with EOD history running through
their collapse to a plausible death date, not truncated continuations.

STATUS: DRAFT. This is PHASE 1(b): design only. No strategy, benchmark or
placebo return has been computed on the evaluation window, and no
relationship between past and future returns has been examined -- only data
*availability* (coverage, start dates, counts, liquidity; see
``first_eligible_sunday`` and the window/midpoint derivation below, both
computed from counts, not returns). This file must not be promoted to frozen
(``DRAFT = False``, ``PREREGISTERED_AT`` set) until the owner has reviewed
and signed off on the design decisions flagged throughout, per the protocol's
own "Freeze first" process rule.

Coordinator review round (2026-10-01), incorporated here: (1) REQUIRED a
liquidity floor, ``UNIVERSE["liquidity_floor_usd"]`` = $1,000,000/day
trailing-30d median dollar volume (implementability-based, given this book's
position sizes) -- this moved ``WINDOWS["start"]`` from 2014-01-12 (no floor)
to 2017-05-14, and the midpoint from 2020-05-24 to 2022-01-23; the power
analysis below was recomputed against the new (smaller) available-block
count. (2) Added the ``BIASES`` dict declaring four known biases and their
direction: residual survivorship (upward, on the long book), CC-volume wash
trading (inflates measured liquidity), the ANC-style bare-ticker-collision
gap, and a new segment-break-adjacent ineligibility rule (28 days after any
cleaning segment break). (3) Confirmed as proposed: the absolute-filter
primary variant, the C1 crypto-to-SPY calendar mapping for BM1-3, the Alpaca
subset as report-only, and the 3-variant grid.

Question: does a weekly cross-sectional momentum basket of the largest
liquid coins (ex. stablecoins/wrapped/pegged/leveraged tokens) beat BTC
buy-and-hold and the live ``BtcTrendSleeve`` (C1) rule out of sample, net of
realistic crypto costs, after deflating for the shortlist's trial count?

Evidence anchors (docs/research_brief_eodhd_findings.md Sec 1 #5): Liu &
Tsyvinski (2021), *Risks and Returns of Cryptocurrency*, RFS 34(6):2689-2727
-- time-series crypto momentum, network/adoption-driven. Liu, Tsyvinski & Wu
(2022), *Common Risk Factors in Cryptocurrency*, JF 77(2):1133-1177 -- a
3-factor model (incl. momentum) spans 10 characteristic long-short crypto
strategies. Both are time-series/individual-asset or long-short results, NOT
a direct test of "rank the top-N coins, go long the top tercile" -- that
basket framing is this design's own construction, not lifted from either
paper. The brief also flags credible, UNVERIFIED-exact-citation decay
evidence post-July-2020 (momentum reportedly negative/insignificant); POWER
below treats that as an illustrative haircut scenario, not a verified figure.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

DRAFT = False
PREREGISTERED_AT = "2026-09-30T19:18:54Z"  # set only at freeze, after owner sign-off; see module docstring

DATA_END = "2026-09-28"     # last Sunday-review-then-Monday-execute pair before this session's 2026-09-30
TRADING_DAYS = 252
SEED = 20261001

# ---------------------------------------------------------------------------
# Data / cleaning (protocol Sec 1, amendment 1)
# ---------------------------------------------------------------------------
DATA = {
    "source": "data/research/eodhd/crypto/*.parquet (local only, no network); "
              "crypto history is unaffected by amendment 1's etfs_full/us_universe_full switch "
              "(crypto has always been full-history, per the coordinator's amendment note)",
    "cleaning_fingerprint": "fc0690f087edaac8c11ddf77b381e58b59c57a893d460f12c7fa782229693054",  # v2, scripts/eodhd_clean.py:cleaning_fingerprint()
    "cleaning_asset_type": "crypto (no calendar filter, no NAV volume exemption -- v1/v2 identical in behaviour for crypto)",
    "segment_rule": "a coin is eligible at a review date only if its most recent cleaned `segment` "
                    "(scripts/eodhd_clean.py:clean_bars) covers at least the trailing 30 calendar days "
                    "needed for the dollar-volume ranking and the 28-day lookback return; a segment "
                    "boundary inside that window makes the coin ineligible at that date. Additionally, "
                    "PER COORDINATOR REVIEW: a coin is ineligible for the SIGNAL['lookback_days'] "
                    "(28) calendar days immediately after any segment break in its own series, even "
                    "once 30 raw calendar days have again accumulated -- a return spanning the break is "
                    "undefined by the protocol's own rule, and 28 days is the shortest window in which "
                    "an undefined cross-break return could otherwise silently re-enter the lookback. "
                    "See BIASES['segment_break_adjacent'].",
    "dollar_volume": "adjusted_close x volume, per protocol Sec 1 -- numerically identical to "
                     "close x volume for crypto in every series checked at the gate (no splits), "
                     "confirmed for BTC-USD/ETH-USD 2026-09-30 (docs/eodhd_s5_crypto_gate_2026_10.md Sec 5)",
    "quote_pairs": "USD-quoted files only (code ending '-USD'); a coin's EUR/GBP/BTC-quoted pairs "
                  "(e.g. BTC-EUR, ETH-GBP) are separate files for the same underlying asset and are "
                  "excluded so they don't double-count dollar volume (gate Sec 4 finding)",
}

# ---------------------------------------------------------------------------
# Universe construction and exclusions (design decision, from data
# availability only -- see docs/eodhd_s5_crypto_gate_2026_10.md Sec 3/4).
# ---------------------------------------------------------------------------
# Exact base-symbol (the part of the code before "-USD", upper-cased, with any
# trailing >=3-digit ticker-collision suffix stripped -- e.g. "CBBTC32994" and
# "TAO22974" both normalise to "CBBTC"/"TAO") exclusions: stablecoins,
# wrapped/staked/liquid-restaking derivatives of another coin's price,
# tokenized non-crypto assets (metals, equity indices), and leveraged/inverse
# tokens -- all four categories were observed polluting the raw
# top-30-by-dollar-volume scan at the gate (docs/eodhd_s5_crypto_gate_2026_10.md
# Sec 4). This is a declared, reproducible rule, not an exhaustive hand-curated
# list -- any coin missed by it is a known limitation, not silently corrected
# later (protocol: "Freeze first").
EXCLUDE_EXACT = {
    # stablecoins
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "GUSD", "USDD", "FRAX",
    "LUSD", "MIM", "SUSD", "USDE", "PYUSD", "FDUSD", "UST", "USTC", "EURT",
    "EURS", "USDX", "USDK", "OUSD", "USDN", "VUSD", "MUSD", "DUSD",
    # wrapped / staked / liquid-restaking derivatives of another coin
    "WBTC", "WETH", "WBNB", "STETH", "WSTETH", "RETH", "CBETH", "CBBTC",
    "BTCB", "RENBTC", "IBETH", "WBETH", "RSETH", "WEETH", "METH", "SFRXETH",
    "FRXETH",
    # tokenized non-crypto assets (metals, equity indices)
    "XAU", "PAXG", "XAUT", "DJ30", "SPX",
    # leveraged/inverse tokens seen in the gate's top-30 scan
    "BULL1", "XRPBULL", "XRPBEAR", "XRPUP", "XRPDOWN", "3X-LONG-BITCOIN-TOKEN",
}
_TICKER_COLLISION_SUFFIX = re.compile(r"^([A-Za-z-]+?)([0-9]{3,})$")

UNIVERSE = {
    "quote_pairs": "code ending '-USD' only",
    "exclude_exact": sorted(EXCLUDE_EXACT),
    "n_primary": 20,
    "n_variant": 30,
    "rank_metric": "trailing 30-calendar-day median dollar volume (adjusted_close x volume), "
                   "computed over the 30 days strictly before the review date",
    "liquidity_floor_usd": 1_000_000.0,
    "liquidity_floor_rationale": "REQUIRED per coordinator review: this book's position sizes are "
                                 "~$2-8k, so a $1,000,000/day trailing-30d median dollar volume floor "
                                 "is a modest, implementability-based bar, not a tuned threshold -- "
                                 "below it, 35bps/side (COSTS) is not a credible cost assumption. See "
                                 "BIASES['wash_trading_volume'] for why the floor is necessary but not "
                                 "sufficient for genuine tradable liquidity.",
    "eligibility": "non-excluded, USD-quoted, >=29 calendar days of same-segment history before the "
                  "review date (28-day lookback + 1) with no segment break in the preceding "
                  "SIGNAL['lookback_days'] (28) days, a bar within the trailing 3 calendar days "
                  "(not stale -- mirrors BtcTrendSleeve.max_stale_days), and trailing-30d median "
                  "dollar volume >= liquidity_floor_usd",
    "ticker_collision_rule": "if two eligible codes share the same normalised base symbol (after "
                            "stripping a trailing >=3-digit suffix, e.g. TAO-USD vs TAO22974-USD; "
                            "gate Sec 4), keep only the one with the higher trailing dollar volume at "
                            "that review date and log the drop -- observed rarely in the gate's scan, "
                            "but declared up front since it changes which name is 'the' coin",
    "known_gap": "see BIASES['ticker_collision_bare'].",
}

# ---------------------------------------------------------------------------
# Known biases and their DIRECTION (per coordinator review 2026-10-01). Data-
# availability/design facts only -- none of these was derived from a return
# series, and none is "fixed" by this draft; they are carried forward as
# declared limitations for the owner/reviewer to weigh, per the protocol's
# "freeze first" rule.
# ---------------------------------------------------------------------------
BIASES = {
    "residual_survivorship": (
        "18.5% of delisted crypto symbols (1,187/6,404 -- "
        "docs/eodhd_s5_crypto_gate_2026_10.md Sec 1) have NO downloaded EOD history at all and are "
        "therefore excluded from the eligible universe by construction, not because they were checked "
        "and found uninteresting. Missing dead coins bias a LONG-ONLY book UPWARD: some unknown "
        "fraction of that 18.5% failed exactly like the 81.5% that do have history, and their absence "
        "here is not evidence they behaved differently, only that this vendor never captured them. THE "
        "GATE's pass (delisted coins that were once large ARE present with history through their "
        "decline) does not erase this -- it shows the data contains real losers, not that it contains "
        "every loser. Treat any candidate result as an upper bound on the true historical return, not "
        "a bias-free estimate."
    ),
    "wash_trading_volume": (
        "EODHD's CC (crypto) `volume` is aggregated across multiple venues and is not itself a "
        "wash-trade-filtered figure. Reported wash-trading rates on unregulated/low-cap venues can be "
        "large (a well-known industry concern; the exact rate is UNVERIFIED this session, no network "
        "access). This inflates measured dollar volume for SOME coins, which can pull a wash-traded "
        "name over UNIVERSE['liquidity_floor_usd'] when its genuine tradable liquidity would not clear "
        "it. The floor is therefore NECESSARY but NOT SUFFICIENT for real tradable liquidity at this "
        "book's size -- it screens out the thinnest names, not every over-stated one."
    ),
    "ticker_collision_bare": (
        "ticker reuse across UNRELATED projects sharing a bare ticker with no numeric-suffix "
        "disambiguation (e.g. ANC-USD is NOT Terra's Anchor Protocol token -- gate Sec 3) is NOT caught "
        "by ticker_collision_rule, since both files pass eligibility independently and there is no "
        "shared base symbol to collide on. Known gap, not patched by hand-curation in this draft -- "
        "the protocol's 'freeze first' rule means this is a design call for owner sign-off."
    ),
    "segment_break_adjacent": (
        "a coin is ineligible for the 28 calendar days immediately following any cleaning `segment` "
        "break in its own series (see DATA['segment_rule']) -- a return spanning the break is undefined "
        "by the protocol's own rule, and 28 days matches SIGNAL['lookback_days'] exactly, so a break "
        "cannot silently re-enter the lookback the moment raw history re-accumulates."
    ),
}


def base_symbol(code: str) -> str:
    """Upper-cased base symbol: code without the '-USD' suffix, and without a
    trailing >=3-digit ticker-collision suffix (see UNIVERSE['ticker_collision_rule'])."""
    stem = code[:-4] if code.endswith("-USD") else code
    stem = stem.upper()
    m = _TICKER_COLLISION_SUFFIX.match(stem)
    return m.group(1) if m else stem


def is_excluded(code: str) -> bool:
    return base_symbol(code) in EXCLUDE_EXACT


def tercile_count(n_universe: int) -> int:
    """Top-tercile holding count: round(n/3), half-up (n=20 -> 7, n=30 -> 10)."""
    return int(math.floor(n_universe / 3.0 + 0.5))


# ---------------------------------------------------------------------------
# Rebalance/signal rule (protocol: matches the live C1 BtcTrendSleeve exactly
# on lookback/calendar; src/firm/allocation/btc_trend.py:49).
# ---------------------------------------------------------------------------
SIGNAL = {
    "lookback_days": 28,
    "rule": "close_review / close_{review-28d} - 1, on the coin's own daily-reindexed/ffilled series, "
           "mirroring BtcTrendSleeve.target_weights's reindex/ffill pattern for crypto's UTC daily calendar",
    "review": "Sunday 00:00 UTC bar close (last completed Sunday <= asof), matching BtcTrendSleeve",
    "rank": "descending by lookback return, among the eligible top-N-by-dollar-volume universe",
}

# ---------------------------------------------------------------------------
# Variants (<=4, small grid, declared up front, never extended after a run --
# protocol Sec 4 "Inference"/DSR note).
# ---------------------------------------------------------------------------
VARIANTS = {
    "S5_top20_abs": {
        "primary": True,
        "n_universe": 20,
        "absolute_filter": True,
        "rule": "rank the top-20-by-dollar-volume universe by 28d return; equal-weight 1/tercile_count "
               "on each top-tercile name whose OWN 28d return > 0 (else that slot is cash) -- mirrors "
               "C1's own on/off absolute-return gate, applied per-holding instead of to a single asset. "
               "Declared PRIMARY: consistency with the live C1 rule this candidate is meant to extend "
               "is judged more important than the small extra turnover/underexposure the filter adds.",
    },
    "S5_top20_rel": {
        "primary": False,
        "n_universe": 20,
        "absolute_filter": False,
        "rule": "as S5_top20_abs, but equal-weight the full top tercile regardless of sign (pure relative "
               "momentum, no absolute filter) -- isolates whether the absolute filter itself matters",
    },
    "S5_top30_abs": {
        "primary": False,
        "n_universe": 30,
        "absolute_filter": True,
        "rule": "as S5_top20_abs, universe widened to the top 30 by dollar volume (protocol's declared "
               "'one variant 30') -- isolates sensitivity to universe breadth/liquidity floor",
    },
}
N_VARIANTS = len(VARIANTS)
assert N_VARIANTS <= 4

# ---------------------------------------------------------------------------
# Execution and costs (protocol Sec 2; crypto row unaffected by amendment 1)
# ---------------------------------------------------------------------------
EXECUTION = {
    "timing": "signal at Sunday 00:00 UTC close; trade at the next daily bar's (Monday) adjusted open "
             "(open x adjusted_close / close) -- degenerates to `open` itself since crypto has no splits "
             "(adjusted_close == close in every series checked at the gate)",
    "robustness_not_a_bar": "also report results with a 2-day signal lag (protocol Sec 2)",
    "weights_drift_between_rebalances": True,
    "rebalance_cadence": "weekly, Sunday-UTC review / Monday-UTC execution -- matches the live C1 rule",
}
COSTS = {
    "crypto_bps_per_side": 35.0,  # protocol Sec 2: Alpaca taker 25bps + 10bps spread
    "cash_rate": "BIL total return from 2007-05-30; FRED DTB3 (data/research/fred/DTB3.parquet, rate/100/252) before that",
    "stress_multiplier": 2.0,  # A5
}

# ---------------------------------------------------------------------------
# Benchmarks (protocol Sec 3)
# ---------------------------------------------------------------------------
BENCHMARKS = {
    "primary": [
        "BTC_BH: BTC/USD buy-and-hold, 100%, same window",
        "C1_live: the live BtcTrendSleeve rule exactly as coded (src/firm/allocation/btc_trend.py), "
        "i.e. C1_btc_trend from scripts/alt_premia_preregistered_bars.py, evaluated on this candidate's "
        "own window -- the honest comparison, since S5's claim is 'a basket beats what already runs live'",
    ],
    "BM1_SPY": "SPY buy-and-hold (data/research/eodhd/etfs_full/SPY.parquet, from 1993-01-29)",
    "BM2_60_40": "60% SPY / 40% IEF, rebalanced monthly; VFITX NAV total return proxies the bond leg "
                "before IEF's first bar (2002-07-26)",
    "BM3_SPY_VT": "SPY weight = min(1, 0.12 / 21d realised vol), traded when |target-held| > 0.10",
    "calendar_alignment": "S5 rebalances weekly on a 7-day crypto calendar; BM1-3 live on the SPY "
                          "trading-day calendar. Comparisons vs BM1-3 (for A4/A5/D, which the protocol "
                          "gates on BM1-3 too, not just the primary pair) use the SAME convention already "
                          "frozen for C1 in scripts/alt_premia_preregistered_bars.py: each UTC calendar "
                          "day's candidate return is compounded onto the first SPY trading date strictly "
                          "after that day (a Sunday-UTC bar closes ~3-4h after the US equity close, so "
                          "same-date mapping would leak; weekend/holiday crypto returns compound onto "
                          "the next SPY session). BTC_BH and C1_live (the primary pair) need no such "
                          "mapping -- both trade on the same native crypto calendar as S5.",
}

# ---------------------------------------------------------------------------
# Placebo (protocol Sec 4: rank-based candidates get random rankings, same
# count, same rebalance dates)
# ---------------------------------------------------------------------------
PLACEBO = {
    "n_draws": 500,
    "pass_percentile": 95.0,
    "seed": SEED + 1,
    "rule": "at each real rebalance date, replace the ranked top-tercile selection with a uniform random "
           "draw (without replacement) of tercile_count names from that date's SAME eligible universe "
           "(same n_universe as the variant); everything else -- absolute filter (if the variant has "
           "one, applied to the randomly drawn names' REAL 28d returns), equal weighting, costs -- is "
           "unchanged, so only the ranking mechanism is randomised",
}

# ---------------------------------------------------------------------------
# Window and midpoint -- fixed from data AVAILABILITY only (counts of
# eligible coins under UNIVERSE's rule, INCLUDING the liquidity_floor_usd
# floor added per coordinator review 2026-10-01), never from a return series.
# Superseded values below (reproducible from data/research/eodhd/crypto via a
# scratch scan, not committed -- same exclusion/eligibility rule as UNIVERSE,
# plus the >=$1,000,000/day trailing-30d median-$vol floor):
#   - WITHOUT the floor, first Sunday-UTC with >=20 eligible coins was
#     2014-01-12 (>=30 reached the same week) -- this was this draft's
#     original window before the floor was required.
#   - WITH the floor: first Sunday-UTC with >=20 eligible coins is
#     2017-05-14 (the count jumps from thin single digits in early 2017 to
#     14 by 2017-04-02 and clears 20 by 2017-05-14 -- consistent with 2017
#     being crypto's first broad-altcoin bull run, not a data artefact);
#     >=30 eligible is reached 2 weeks later, 2017-05-28. The floor therefore
#     moves the window start ~3.3 years later than the unfiloored version,
#     roughly matching the coordinator's own "likely ~2017" expectation.
# ---------------------------------------------------------------------------
WINDOWS = {
    "start": "2017-05-14",  # first Sunday-UTC review with >=20 eligible coins AFTER the liquidity floor
    "end": DATA_END,
    "midpoint": "2022-01-23",  # nearest Sunday-UTC review to the calendar midpoint of [start, end] (2022-01-20)
    "post_2020_decay_window": {"start": "2020-07-01", "end": DATA_END,
                                "rationale": "the brief's own cited (UNVERIFIED-exact-citation) "
                                            "post-July-2020 crypto-momentum decay date; used only for "
                                            "the illustrative decay scenario in POWER below, and for an "
                                            "A4-style post-decay half-window check at evaluation time"},
    "note_variant_windows": "S5_top20_abs and S5_top20_rel use this window's start (2017-05-14, "
                            ">=20 eligible). S5_top30_abs's OWN eligibility (>=30 coins passing the "
                            "floor) is only reached 2 weeks later, 2017-05-28 -- it trades from "
                            "2017-05-28 within this SAME overall window/midpoint (not a separately "
                            "recomputed midpoint), a declared simplification given the gap is small "
                            "relative to the window's ~9.4-year length.",
}

# ---------------------------------------------------------------------------
# Inference (protocol Sec 4-5)
# ---------------------------------------------------------------------------
BOOTSTRAP = {
    "method": "paired stationary block bootstrap of the daily Sharpe gap (candidate minus benchmark), "
             "resampling the same days for both",
    "mean_block_calendar_days": 91,  # protocol: "for S5, 91 calendar days"
    "n_boot": 5000,
    "seed": SEED,
    "n_shortlist_candidates": 5,  # S1-S5, protocol Sec 4
    "alpha_one_sided": 0.05 / 5,
}
DSR = {
    "trials": N_VARIANTS,
    "trial_sharpes": "the daily (non-annualised) Sharpe of every S5 variant (3) on the PBO window",
    "prior_trials": 207,  # placeholder: resolved to an
    # int only once every shortlist pre-registration (S1-S4) has frozen its own declared variant grid,
    # per protocol Sec 4 ("plus the variant counts of the other four shortlist candidates, fixed at
    # freeze from each pre-registration's declared grid") -- NOT computable from S5 alone
    "ledger": "docs/s5_trial_history.json (new family, never reset, per protocol Sec 6)",
}
PBO = {"n_partitions": 8, "series": "S5's 3 variants + BM1_SPY + BM2_60_40 + BM3_SPY_VT, daily excess returns"}

# ---------------------------------------------------------------------------
# Implementability diagnostic (report-only, NOT a bar -- per task instructions).
# Any coin list written from memory rather than fetched this session is
# marked UNVERIFIED and is survivorship-biased (a broker's CURRENT supported
# list can only ever contain coins that are still around), so it cannot be
# used to judge the historical strategy, only to scope a forward paper test.
# ---------------------------------------------------------------------------
IMPLEMENTABILITY = {
    "status": "UNVERIFIED (from memory, not fetched this session -- no network calls are permitted in "
             "this phase)",
    "alpaca_crypto_subset_illustrative": [
        "BTC", "ETH", "LTC", "BCH", "LINK", "UNI", "AAVE", "SOL", "DOGE", "AVAX",
        "DOT", "GRT", "MKR", "SUSHI", "YFI", "BAT", "CRV", "XTZ", "SHIB",
    ],
    "caveat": "report-only diagnostic, not a bar or a universe filter -- it is inherently "
             "survivorship-biased (only currently-supported, i.e. currently-alive, coins can appear on "
             "a broker's live list today) and would silently reintroduce the exact bias THE GATE was "
             "run to rule out if used to restrict the historical backtest universe. Its only legitimate "
             "use is scoping what a live forward paper test could actually hold.",
}

# ---------------------------------------------------------------------------
# Tiers (protocol Sec 5, precedence A > D > B > C; no S2-only A8 here)
# ---------------------------------------------------------------------------
TIER_A_BARS = [
    {"id": "A1", "rule": "Sharpe gap vs the primary benchmarks (BTC_BH AND C1_live) > 0, and the "
                        "bootstrap one-sided lower bound at alpha_one_sided > 0, for both"},
    {"id": "A2", "rule": "DSR > 0.95"},
    {"id": "A3", "rule": "Sharpe above the 95th percentile of the placebo Sharpes"},
    {"id": "A4", "rule": "Sharpe gap > 0 in both halves (split at WINDOWS['midpoint']), vs BTC_BH, "
                        "C1_live and BM1-BM3"},
    {"id": "A5", "rule": "Sharpe gap > 0 at 2x costs, vs BTC_BH, C1_live and BM1-BM3"},
    {"id": "A6", "rule": "CSCV PBO < 0.50"},
    {"id": "A7", "rule": "an independent recompute reproduces every bar outcome"},
]
TIER_D_RULE = "bootstrap one-sided upper bound at alpha_one_sided of the Sharpe gap < 0 vs BTC_BH or C1_live"
TIER_B_BARS = [
    {"id": "B_a", "rule": "point estimate > 0 vs BTC_BH and C1_live"},
    {"id": "B_b", "rule": "A3, A4 and A5 pass"},
]
TIER_ACTIONS = {
    "A": "live config diff for owner sign-off (2-day-lag and per-year audits first)",
    "B": "owner decides (labelled 'risk improvement/not proven alpha')",
    "C": "not deployed",
    "D": "rejected",
}


def classify(bars: dict[str, bool], tier_d: bool, tier_b: dict[str, bool]) -> str:
    """Tier from bar outcomes, frozen precedence A > D > B > C (identical shape to
    alt_premia_preregistered_bars.classify and insider_cluster_preregistered_bars.classify)."""
    if bars and all(bars.values()):
        return "A"
    if tier_d:
        return "D"
    if tier_b and all(tier_b.values()):
        return "B"
    return "C"


# ---------------------------------------------------------------------------
# Honest power analysis -- analytic, from DECLARED assumptions and DATA-
# AVAILABILITY counts only (number of independent 91-day blocks in the
# window), never from a computed return series (no network, no backtest was
# run for this). Two Sharpe-gap scenarios:
#   - "literature_anchor": reuses the SAME (rho, Sharpe-gap) prior as the live
#     C1 candidate's own entry in alt_premia_preregistered_bars.POWER_PRIORS
#     ("C1_btc_trend": (0.7, 0.5)) -- an upper-bound anchor, since that prior
#     is for a single-asset BTC trend rule, not cross-sectional basket
#     momentum, and no citation found this session gives a basket-specific
#     number.
#   - "post_2020_decay_illustrative": the brief's own flagged, UNVERIFIED-
#     exact-citation decay evidence (post-July-2020 momentum reportedly
#     negative/insignificant) is not a number this design can use directly --
#     this scenario is a documented, arbitrary haircut (0.15, i.e. 30% of the
#     literature anchor) meant only to show whether the available post-2020
#     block count could even detect a much smaller surviving effect, not a
#     literature-sourced figure.
# ---------------------------------------------------------------------------
POWER = {
    "block_calendar_days": BOOTSTRAP["mean_block_calendar_days"],
    "alpha_one_sided": BOOTSTRAP["alpha_one_sided"],
    "target_power": 0.80,
    "scenarios": {
        "literature_anchor": {"sharpe_gap_annual": 0.5,
                              "source": "same prior as C1_btc_trend in alt_premia_preregistered_bars.POWER_PRIORS "
                                       "(Liu & Tsyvinski 2021 anchor) -- single-asset trend, not this design's own basket claim"},
        "post_2020_decay_illustrative": {"sharpe_gap_annual": 0.15,
                                         "source": "arbitrary 30%-of-anchor haircut, NOT a verified post-2020 figure "
                                                  "(the brief flags the underlying decay claim itself as UNVERIFIED)"},
    },
    "windows_available_blocks": {
        "full_window": WINDOWS["start"] + " to " + DATA_END,
        "post_2020_decay_window": WINDOWS["post_2020_decay_window"]["start"] + " to " + DATA_END,
    },
}


def _norm_ppf(p: float) -> float:
    """Acklam's rational approximation to the standard normal inverse CDF
    (duplicated from firm.eval.overfitting._norm_ppf to keep this module
    import-light; values agree to >=8 significant figures)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def required_blocks(sharpe_gap_annual: float, alpha_one_sided: float, power: float,
                     block_calendar_days: int) -> float:
    """Number of independent (block_calendar_days-long) blocks needed to detect
    an annualised Sharpe gap at the given one-sided alpha and power, i.e. the
    smallest N such that sharpe_gap_annual * sqrt(N * block_calendar_days / 365)
    > z_alpha + z_beta. The per-block volatility cancels out of a Sharpe-ratio
    detection test, so (unlike insider_cluster_preregistered_bars.required_n,
    which needs an assumed per-bet bps stdev) this needs no volatility
    assumption at all."""
    z_a = _norm_ppf(1 - alpha_one_sided)
    z_b = _norm_ppf(power)
    return ((z_a + z_b) / sharpe_gap_annual) ** 2 * (365.0 / block_calendar_days)


def power_report() -> dict:
    """Required vs. available 91-day blocks, full window and the post-2020
    decay sub-window, for both POWER scenarios."""
    from datetime import date
    start = date.fromisoformat(WINDOWS["start"])
    end = date.fromisoformat(DATA_END)
    decay_start = date.fromisoformat(WINDOWS["post_2020_decay_window"]["start"])
    windows = {
        "full_window": (end - start).days // POWER["block_calendar_days"],
        "post_2020_decay_window": (end - decay_start).days // POWER["block_calendar_days"],
    }
    rows = []
    for scen_name, scen in POWER["scenarios"].items():
        n_req = required_blocks(scen["sharpe_gap_annual"], POWER["alpha_one_sided"],
                                 POWER["target_power"], POWER["block_calendar_days"])
        for win_name, n_avail in windows.items():
            rows.append({
                "scenario": scen_name, "sharpe_gap_annual": scen["sharpe_gap_annual"],
                "window": win_name, "n_blocks_available": n_avail,
                "n_blocks_required_80pct_power": round(n_req, 1),
                "adequately_powered": bool(n_avail >= n_req),
            })
    return {"rows": rows, "windows_days": {
        "full_window": (end - start).days, "post_2020_decay_window": (end - decay_start).days,
    }}


def bars_fingerprint() -> str:
    payload = json.dumps(
        {
            "DATA_END": DATA_END, "SEED": SEED, "DATA": DATA, "UNIVERSE": UNIVERSE, "BIASES": BIASES,
            "SIGNAL": SIGNAL, "VARIANTS": VARIANTS, "N_VARIANTS": N_VARIANTS,
            "EXECUTION": EXECUTION, "COSTS": COSTS, "BENCHMARKS": BENCHMARKS,
            "PLACEBO": PLACEBO, "WINDOWS": WINDOWS, "BOOTSTRAP": BOOTSTRAP, "DSR": DSR, "PBO": PBO,
            "IMPLEMENTABILITY": IMPLEMENTABILITY, "TIER_A_BARS": TIER_A_BARS,
            "TIER_D_RULE": TIER_D_RULE, "TIER_B_BARS": TIER_B_BARS, "TIER_ACTIONS": TIER_ACTIONS,
            "POWER": POWER,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    print(("DRAFT" if DRAFT else "FROZEN") + " fingerprint:", bars_fingerprint())
    import pprint
    pprint.pprint(power_report())
