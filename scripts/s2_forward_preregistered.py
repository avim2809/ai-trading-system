"""FROZEN pre-registration: SHADOW forward test of the S2 breadth overlay (variant V1).

What this is: a paper ledger that tracks, going forward, what the frozen S2 V1
overlay on the 60/40 core WOULD have done, next to plain 60/40 and SPY. It places
NO orders and changes NO live config. Putting the overlay on the Alpaca instance
would need new allocator machinery and the owner's sign-off on the exact config
diff; that is a separate, later decision (PROMOTION below).

Why it exists: S2 was Tier C (near miss) in the historical evaluation
(docs/eodhd_shortlist_verdict_2026_10.md). The owner asked for a forward test of
the primary variant. The rule, universe, screens and costs are NOT redefined here:
they are the frozen S2 pre-registration, referenced by fingerprint.

Written and frozen BEFORE the first check (the check for 2026-10-01 is computed
after PREREGISTERED_AT and before the 2026-10-01 US open). Do not edit once the
ledger has started: a changed fingerprint means the run no longer tests this
design. A different design needs a new file and a new ledger.

What it can and cannot show (see POWER): V1 switches state about 0.8 times a year,
so a realistic forward horizon holds very few switches. The test can verify that
the live signal pipeline matches the historical one and that the overlay behaves
as modelled; it cannot, on its own, establish an edge. That is stated up front.
"""

from __future__ import annotations

import hashlib
import json
import math

DRAFT = False
PREREGISTERED_AT = "2026-10-01T08:04:20Z"  # set at freeze (UTC; verify against the freeze commit time)
LEDGER_ID = "S2F_V1"

# ---------------------------------------------------------------------------
# The rule: the frozen S2 pre-registration, by reference. Not restated as new.
# ---------------------------------------------------------------------------
FROZEN_REFERENCE = {
    "preregistration": "scripts/eodhd_s2_breadth_overlay_preregistered_bars.py (freeze commit ff6f44a)",
    "bars_fingerprint": "4fd97659e9affce3c2fb6391daba522ad83d8c36be4698f2370e78c39ff7b6ab",
    "variant": "V1_primary",
    "cleaning_fingerprint": "fc0690f087edaac8c11ddf77b381e58b59c57a893d460f12c7fa782229693054",
    "breadth_builder": "scripts/eodhd_breadth.py:_ticker_signal (per-ticker eligibility and above-200sma flag)",
}
RULE = {
    "breadth": "fraction of the day's eligible US common stocks with adjusted_close > their 200-session SMA",
    "eligible": "same segment >= 200 bars, raw close >= $5, ADV20 (median of adjusted_close x volume) >= $1,000,000",
    "check_day": "first NYSE trading day of each calendar month (src/firm/allocation/calendar.py), measured at the "
                 "close of the last session before it, executed at the check day's adjusted open",
    "cut_if": "breadth < 0.50",
    "cut_weights": {"SPY": 0.30, "IEF": 0.70},
    "plain_weights": {"SPY": 0.60, "IEF": 0.40},
    "no_look_ahead": "the signal for check day F uses bars through the session before F only",
}
DESCRIPTIVE_ONLY = {
    "V3_stricter": "threshold 0.40, same cut weights; tracked and reported but NEVER used for a decision "
                   "(it was not the declared primary; choosing it after the fact would be multiple testing)",
}

# ---------------------------------------------------------------------------
# Shadow portfolios (all start 2026-10-01 at the adjusted open, NAV 100)
# ---------------------------------------------------------------------------
START = {
    "first_check_day": "2026-10-01",
    "signal_asof": "2026-09-30",
    "nav_start": 100.0,
}
PORTFOLIOS = {
    "overlay_v1": "weights per RULE at each monthly check; drift between checks",
    "plain_60_40": "60/40 SPY/IEF, rebalanced to target at every monthly check day's adjusted open",
    "spy": "SPY bought at the first open and held",
    "overlay_v3_descriptive": "as overlay_v1 with threshold 0.40",
}
COSTS = {"bps_per_side": 3.0, "applies_to": "notional traded at each rebalance, including the initial purchase",
         "basis": "protocol default for SPY/IEF (>= $50M ADV20); the forward ledger charges it at every rebalance"}
CASH = {"series": "BIL adjusted close (total return), used only as the risk-free rate for Sharpe/Calmar reporting"}
PRICES = {
    "source": "EODHD end-of-day (adjusted open derived as open x adjusted_close / close), SPY, IEF, BIL",
    "recompute_rule": "the NAV ledger is recomputed from scratch from the decision log on every run, from the "
                      "current adjusted series, so dividends and vendor revisions flow through consistently",
}

# ---------------------------------------------------------------------------
# Data, schedule, integrity
# ---------------------------------------------------------------------------
DATA = {
    "breadth_source": "EODHD: exchange-symbol-list/US (active common stock on NASDAQ/NYSE/AMEX/NYSE MKT/NYSE ARCA/BATS) "
                      "and per-ticker end-of-day history, pulled fresh at each check (history window ~760 days)",
    "cleaning": "scripts/eodhd_clean.py clean_bars, v2 (fingerprint above), calendar = SPY's dates",
    "pit_note": "the forward universe is the set of stocks with a bar on the signal date; delisted stocks have no "
                "future bars, so this equals the survivorship-free historical construction",
    "source_continuity": "EODHD Historian is a one-month subscription at the time of freezing. If it lapses, the "
                         "checks are MISSED, not silently replaced: a replacement source needs a new data-equivalence "
                         "validation and an amendment recorded in the ledger before it is used",
}
SCHEDULE = {"runner": "scripts/s2_forward_shadow.py run", "when": "daily 07:00 UTC (systemd timer)",
            "missed_check_rule": "a check not computed on its day may be back-filled later from PIT-valid history "
                                 "(signal still as of the prior session's close), logged status 'late_backfill'; "
                                 "if it cannot be computed it is 'missed' and the previous weights are held"}
INTEGRITY = {
    "eligible_count_band": [1500, 5500],
    "fresh_data_ratio_min": 0.90,
    "anomaly_rule": "outside the band, or fewer than 90% of the stocks that had a bar on the previous session have "
                    "one on the signal date (data not yet published): the check is recorded 'anomaly', the previous "
                    "weights are held (initially plain 60/40) and a Discord alert is sent; it is re-run, not "
                    "overridden by hand",
    "determinism": "re-running the ledger build from the decision log reproduces nav.csv exactly (tested)",
    "launch_validation": {
        "rule": "before the first check counts, the live pipeline must reproduce the frozen builder's breadth "
                "(docs: review artifacts S2/breadth_full.parquet) on >= 5 overlapping historical session dates "
                "to within 0.5 percentage points and eligible count within 3%",
        "if_failed": "the start is VOID; fix the tracker (not this file) and restart with a new START entry",
    },
}

# ---------------------------------------------------------------------------
# Decision rules (evaluated by scripts/s2_forward_shadow.py report)
# ---------------------------------------------------------------------------
MIN_MONTHS = 24
MAX_MONTHS = 60
CHECKPOINTS_DESCRIPTIVE = [6, 12, 18]
DEFINITIONS = {
    "completed_episode": "a maximal run of >= 1 consecutive monthly checks in the cut state that is followed by a "
                         "check in the plain state",
    "stats_basis": "daily returns of the shadow NAV over the whole forward window; Sharpe and Calmar are in "
                   "excess of BIL; the Sharpe gap is overlay minus plain 60/40",
}
PROMOTE = {
    "outcome": "PROPOSE_LIVE_TEST",
    "all_of": [
        "months >= 24",
        "missed + unresolved anomaly checks <= 1",
        "completed_episodes >= 2",
        "maxDD(overlay) <= maxDD(plain_60_40)",
        "Calmar(overlay) > Calmar(plain_60_40)",
        "Sharpe_gap(overlay - plain) >= -0.05",
    ],
    "then": "a live proposal for the owner's sign-off on the exact config diff; it also needs allocator support "
            "for a signal-driven weight change, which does not exist yet. Nothing goes live automatically.",
}
REJECT = {
    "outcome": "REJECT",
    "when": "months >= 24 and completed_episodes >= 2 and (maxDD(overlay) > maxDD(plain_60_40) or "
            "Sharpe_gap < -0.05)",
}
OTHERWISE = {"outcome": "CONTINUE", "until": "MAX_MONTHS, after which the result is closed as INCONCLUSIVE"}

# ---------------------------------------------------------------------------
# Power: stated up front
# ---------------------------------------------------------------------------
HISTORICAL_EPISODES = {"on_episodes": 19, "years": 24.66, "source": "S2 pre-registration OBSERVED, V1"}


def power_report(months: int = MIN_MONTHS) -> dict:
    """Poisson chance of seeing >= k completed episodes in ``months``, at the historical rate."""
    rate_per_year = HISTORICAL_EPISODES["on_episodes"] / HISTORICAL_EPISODES["years"]
    lam = rate_per_year * months / 12.0
    p0 = math.exp(-lam)
    p_ge1 = 1.0 - p0
    p_ge2 = 1.0 - p0 * (1.0 + lam)
    return {"months": months, "episodes_per_year": rate_per_year, "expected_episodes": lam,
            "p_at_least_1": p_ge1, "p_at_least_2": p_ge2,
            "reading": "if the future resembles the past, the chance that 24 months contain the two completed "
                       "episodes the decision needs is roughly even; otherwise the rule answers CONTINUE. "
                       "A forward test of a rare-event overlay is slow by construction."}


def bars_fingerprint() -> str:
    payload = json.dumps(
        {"PREREGISTERED_AT": PREREGISTERED_AT, "LEDGER_ID": LEDGER_ID, "FROZEN_REFERENCE": FROZEN_REFERENCE,
         "RULE": RULE, "DESCRIPTIVE_ONLY": DESCRIPTIVE_ONLY, "START": START, "PORTFOLIOS": PORTFOLIOS,
         "COSTS": COSTS, "CASH": CASH, "PRICES": PRICES, "DATA": DATA, "SCHEDULE": SCHEDULE,
         "INTEGRITY": INTEGRITY, "MIN_MONTHS": MIN_MONTHS, "MAX_MONTHS": MAX_MONTHS, "DEFINITIONS": DEFINITIONS,
         "PROMOTE": PROMOTE, "REJECT": REJECT, "OTHERWISE": OTHERWISE},
        sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


if __name__ == "__main__":
    print(("DRAFT" if DRAFT else "FROZEN") + " fingerprint:", bars_fingerprint())
    print(json.dumps(power_report(), indent=1))
