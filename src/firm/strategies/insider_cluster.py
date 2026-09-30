"""Insider-purchase cluster strategy — satellite candidate, NOT wired into the
live pipeline.

Financial intuition:
    Cohen, Malloy & Pomorski ("Decoding Inside Information", Journal of
    Finance 2012) find that clusters of open-market purchases by multiple
    distinct company insiders within a short window predict abnormal
    returns over the following 3-6 months, and that the effect is
    concentrated in "opportunistic" insiders — those who do not trade on a
    fixed, routine schedule (routine := has traded in the same calendar
    month in each of the prior 3 years) and are therefore more plausibly
    acting on genuine private information rather than a pre-scheduled plan
    (a 10b5-1 program, a vesting-linked sale, etc.). A UNVERIFIED 2024
    replication reportedly finds only ~1/3 of the original 82bp/month
    effect post-2008 (see docs/research_findings_beyond_equities_2026_09_30.md
    §"SEC-filings-driven insider-purchase clustering" and the pre-registration
    in scripts/insider_cluster_preregistered_bars.py for the exact source
    caveat) — this module makes no claim the effect survives costs; it only
    detects the pre-defined event.

Data inputs:
    NOT ``PitView`` — the live/backtest ``PointInTimeDataStore`` and its
    ``PitView`` protocol (``src/firm/strategies/base.py``) have no insider-
    transaction accessor today (the research brief's own estimate: a new
    ``PitView.insider_transactions()`` accessor plus small/micro-cap
    universe breadth is a ~24-36 person-day integration, not part of this
    scoping pass). This module's real entry points are the pure functions
    below (``classify_owner``, ``compute_cluster_events``), which operate
    directly on the parsed tables from
    ``src/firm/data/insider_transactions.py``
    (``point_in_time_purchases.parquet`` / ``owner_activity_calendar.parquet``).
    ``InsiderClusterStrategy.generate()`` is kept only for structural
    parity with the other strategy modules and to make future PitView
    wiring a small diff; it is registered under "insider_cluster" but is
    NEVER added to any config (`config/live.yaml`, `config/live_alpaca.yaml`,
    `config/settings.yaml`), so it cannot be accidentally scheduled — it
    degrades to a no-op with a logged warning if ever invoked without the
    (currently nonexistent) data feed.

Signal logic:
    1. Classify every purchase as "opportunistic" or "routine" per CMP:
       an owner is routine in month M/year Y if they show ANY Form 4/5
       transaction (not only purchases) in month M in each of Y-1, Y-2, Y-3;
       opportunistic otherwise.
    2. Group purchases by issuer. Walk each issuer's purchases in
       ``trans_date`` order, tracking the count of distinct owners with a
       purchase in the trailing 30-calendar-day window ending on the
       current purchase's ``trans_date``.
    3. A cluster event fires on the *rising edge* — the first purchase at
       which that trailing distinct-owner count reaches >= 3 — provided at
       least one purchase in the qualifying window is opportunistic. This
       avoids re-firing on every subsequent purchase inside an
       already-recognized, still-active cluster.
    4. The event's ``known_date`` (the earliest date a signal consumer
       could legally act on it) is the MAX ``known_date`` (SUBMISSION
       filing date, not transaction date — see insider_transactions.py's
       module docstring) across every purchase in the qualifying window:
       you cannot trade a 3-insider cluster before the 3rd insider's
       purchase has actually been filed and made public.

Portfolio construction approach (as pre-registered, not implemented here):
    Long the issuer's stock from the next tradable session after
    ``known_date``, held 3-6 months; see
    ``scripts/insider_cluster_preregistered_bars.py`` (DRAFT — blocked on
    survivorship-free small/mid-cap daily price data, see the research
    report).

Risk notes:
    - Small-cap liquidity and spread costs plausibly exceed the (already
      decayed, per the UNVERIFIED-flagged 2024 replication) net edge for
      the most micro-cap, most cluster-dense names — this is exactly the
      pre-registration's job to check, not assumed here.
    - Trans code "P" also covers *private* (not only open-market)
      purchases — a known imprecision inherited from the source data and
      from CMP's own methodology.
    - The "routine" classifier uses month-of-``trans_date`` (not
      ``known_date``) for the prior-three-years pattern; this is a
      description of the owner's historical behavior, not a live PIT
      signal, so using the economic transaction date rather than the
      filing date here is intentional (see classify_owner's docstring).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

log = logging.getLogger(__name__)

MIN_DISTINCT_INSIDERS = 3
CLUSTER_WINDOW_DAYS = 30


def _prior_year_months(year_month: str) -> list[str]:
    """["2020-01"] -> the same calendar month in each of the prior 3 years."""
    year, month = (int(x) for x in year_month.split("-"))
    return [f"{year - k}-{month:02d}" for k in (1, 2, 3)]


def build_owner_activity_index(activity: pd.DataFrame) -> dict[str, set[str]]:
    """owner_cik -> set of "YYYY-MM" strings the owner filed ANY transaction in."""
    idx: dict[str, set[str]] = defaultdict(set)
    for owner_cik, year_month in zip(activity["owner_cik"], activity["year_month"]):
        idx[owner_cik].add(year_month)
    return idx


def classify_owner(owner_cik: str, trans_date: pd.Timestamp, activity_index: dict[str, set[str]]) -> str:
    """"routine" if *owner_cik* traded in the same calendar month as
    *trans_date* in each of the prior 3 years (Cohen, Malloy & Pomorski
    2012's definition), else "opportunistic".

    Uses the transaction's own calendar month, not its filing/known date:
    this classifies the owner's real-world trading *pattern*, which is a
    retrospective description of behavior over the prior three years, not
    a live signal that itself needs point-in-time gating (see module
    docstring's Risk notes).
    """
    ref_month = f"{trans_date.year}-{trans_date.month:02d}"
    months_known = activity_index.get(owner_cik, set())
    is_routine = all(m in months_known for m in _prior_year_months(ref_month))
    return "routine" if is_routine else "opportunistic"


def compute_cluster_events(
    purchases: pd.DataFrame,
    activity: pd.DataFrame,
    min_insiders: int = MIN_DISTINCT_INSIDERS,
    window_days: int = CLUSTER_WINDOW_DAYS,
) -> pd.DataFrame:
    """Detect insider-purchase cluster events per Cohen, Malloy & Pomorski (2012).

    *purchases* must have (at minimum): issuer_cik, issuer_name, ticker,
    owner_cik, trans_date, known_date — i.e. the output of
    ``insider_transactions.resolve_tickers(insider_transactions.dedupe_amendments(...))``.
    *activity* is ``owner_activity_calendar`` from the same pipeline.

    Returns one row per detected cluster (the "rising edge" only — see
    module docstring point 3), sorted by known_date, with columns:
    issuer_cik, ticker, cluster_trans_date (the triggering purchase's own
    transaction date), known_date, n_distinct_insiders,
    has_opportunistic, owner_ciks (tuple), window_start_trans_date.
    """
    required = {"issuer_cik", "owner_cik", "trans_date", "known_date"}
    missing = required - set(purchases.columns)
    if missing:
        raise ValueError(f"purchases is missing required columns: {missing}")
    if purchases.empty:
        return _empty_events_frame()

    activity_index = build_owner_activity_index(activity)
    df = purchases.copy()
    df["trans_date"] = pd.to_datetime(df["trans_date"])
    df["known_date"] = pd.to_datetime(df["known_date"])
    df["_opportunistic"] = [
        classify_owner(o, t, activity_index) == "opportunistic"
        for o, t in zip(df["owner_cik"], df["trans_date"])
    ]

    window = pd.Timedelta(days=window_days - 1)  # inclusive trailing window
    events = []
    for issuer_cik, grp in df.sort_values("trans_date").groupby("issuer_cik", sort=False):
        grp = grp.reset_index(drop=True)
        n = len(grp)
        if n < min_insiders:
            continue  # can't possibly reach min_insiders distinct owners
        trans_dates = grp["trans_date"].to_numpy()
        owner_ciks = grp["owner_cik"].to_numpy()

        # Two-pointer trailing window: as i advances, drop from the left
        # (lo) every row that has fallen outside [t_i - window, t_i]. This
        # is O(n) per issuer in total (each row enters/leaves the window
        # exactly once), instead of O(n^2) from re-scanning the whole
        # group at every row — the difference matters once a handful of
        # mega-cap issuers each have thousands of purchase filings across
        # 20 years.
        owner_counts: dict = defaultdict(int)
        distinct = 0
        lo = 0
        in_cluster = False
        for i in range(n):
            t_i = trans_dates[i]
            window_start = t_i - window
            oc_i = owner_ciks[i]
            if owner_counts[oc_i] == 0:
                distinct += 1
            owner_counts[oc_i] += 1

            while trans_dates[lo] < window_start:
                oc_lo = owner_ciks[lo]
                owner_counts[oc_lo] -= 1
                if owner_counts[oc_lo] == 0:
                    distinct -= 1
                lo += 1

            if distinct < min_insiders:
                in_cluster = False
                continue
            if in_cluster:
                continue  # already fired for this run of the cluster — rising edge only

            window_mask = slice(lo, i + 1)
            has_opportunistic = bool(grp["_opportunistic"].iloc[window_mask].any())
            if not has_opportunistic:
                # CMP's own headline result is specific to clusters with at
                # least one opportunistic buyer; a purely-routine cluster
                # (e.g. a scheduled buyback-adjacent pattern) is not the
                # tested hypothesis and is intentionally excluded here.
                in_cluster = False
                continue

            events.append({
                "issuer_cik": issuer_cik,
                "issuer_name": grp["issuer_name"].iloc[i] if "issuer_name" in grp.columns else None,
                "ticker": grp["ticker"].iloc[i] if "ticker" in grp.columns else None,
                "cluster_trans_date": t_i,
                "window_start_trans_date": window_start,
                "known_date": grp["known_date"].iloc[window_mask].max(),
                "n_distinct_insiders": int(distinct),
                "has_opportunistic": has_opportunistic,
                "owner_ciks": tuple(sorted(grp["owner_cik"].iloc[window_mask].unique())),
            })
            in_cluster = True

    if not events:
        return _empty_events_frame()
    out = pd.DataFrame(events).sort_values("known_date").reset_index(drop=True)
    log.info("compute_cluster_events: %d cluster events across %d issuers (min_insiders=%d, window_days=%d)",
              len(out), out["issuer_cik"].nunique(), min_insiders, window_days)
    return out


def _empty_events_frame() -> pd.DataFrame:
    cols = [
        "issuer_cik", "issuer_name", "ticker", "cluster_trans_date", "window_start_trans_date",
        "known_date", "n_distinct_insiders", "has_opportunistic", "owner_ciks",
    ]
    return pd.DataFrame(columns=cols)


@register("insider_cluster")
class InsiderClusterStrategy(BaseStrategy):
    """Structural placeholder only — see module docstring. Not wired into
    any live config; ``generate()`` no-ops (with a logged warning) unless a
    future PitView exposes insider-cluster events directly, since the
    current ``PitView`` protocol (src/firm/strategies/base.py) has no such
    accessor.
    """

    def __init__(self, params: dict | None = None):
        super().__init__("insider_cluster", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        events_accessor = getattr(pit_view, "insider_cluster_events", None)
        if events_accessor is None:
            log.warning(
                "insider_cluster: PitView has no insider_cluster_events() accessor — "
                "this strategy is a research placeholder (see module docstring); "
                "returning no signals. Not an error if this strategy isn't wired into "
                "any live config, which it currently is not."
            )
            return []

        events = events_accessor()
        if events is None or (hasattr(events, "empty") and events.empty):
            return []

        hold_days: int = self.params.get("hold_days", 90)
        universe = set(pit_view.universe or [])
        asof = pit_view.asof
        signals: list[Signal] = []
        for _, ev in events.iterrows():
            ticker = ev.get("ticker")
            if not ticker or (universe and ticker not in universe):
                continue
            known_date = pd.Timestamp(ev["known_date"])
            days_since = (pd.Timestamp(asof) - known_date).days
            if days_since < 0 or days_since > hold_days:
                continue
            signals.append(
                Signal(
                    symbol=str(ticker),
                    strategy="insider_cluster",
                    score=1.0,  # long-only event signal, no magnitude in CMP's own design
                    confidence=1.0 if ev.get("has_opportunistic") else 0.5,
                    horizon=f"{hold_days}d",
                    asof=asof,
                    meta={
                        "n_distinct_insiders": int(ev["n_distinct_insiders"]),
                        "has_opportunistic": bool(ev["has_opportunistic"]),
                        "known_date": str(known_date.date()),
                        "days_since_event": days_since,
                    },
                )
            )
        return signals
