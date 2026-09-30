"""BTC trend-following satellite sleeve (Tier C, forward paper test only).

Implements candidate ``C1_btc_trend`` from the pre-registered alt-premia edge
search *exactly as pre-registered* -- no tuning:
``scripts/alt_premia_preregistered_bars.py``'s ``CANDIDATES["C1_btc_trend"]``,
evaluated by ``scripts/run_alt_premia_evaluation.py``'s ``c1_signal``/``c1_btc``,
verdict in ``docs/edge_search_verdict_2026_09.md`` §3 (Tier C: a small,
positive, *not statistically proven* point estimate -- Tier C is exactly why
this only ever runs as a small satellite sleeve, not a validated strategy).

THE RULE (verbatim from the pre-registration):
    BTC/USD spot; reviewed each Sunday (UTC) on day-d close: on iff
    close_d / close_{d-28} - 1 > 0; weight when on = min(1, 0.40 / sigma_d)
    where sigma_d = std of last 63 daily BTC returns x sqrt(365); at review,
    trade if on/off flips or |target - held| > 0.10; position effective from
    the next day (1-day lag).

Design note -- why ``target_weights`` doesn't track "held" itself: the
``Sleeve.target_weights`` contract takes only ``(asof, history)`` -- no
currently-held weight -- and must be a pure function of that history (no
network, no hidden state). This implementation therefore returns the
*signal-derived target* for the most recently completed Sunday review found
in ``history`` (0 when the trend is off, or when there isn't yet enough
history for a lookback/vol window), exactly mirroring ``c1_signal``'s
on/sigma computation. The rule's own trade-or-hold threshold
(``band_abs`` -- "trade if ... |target - held| > 0.10") and the actual
held/drifted position are therefore the allocator's concern (it knows the
sleeve's real current allocation; this sleeve does not) -- ``band_abs``,
``target_vol``, ``lookback_days`` and ``vol_window`` are exposed as public
attributes for exactly that purpose, matching this codebase's existing
``rebalance_band_pct`` convention (``src/firm/agents/execution.py``) applied
per-sleeve instead of globally.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

import numpy as np
import pandas as pd

from firm.allocation.sleeves import Sleeve

log = logging.getLogger(__name__)


class BtcTrendSleeve(Sleeve):
    """Weekly BTC/USD trend-following satellite sleeve (Tier C, forward
    paper test only -- see this module's docstring)."""

    # Allocator contract (src/firm/allocation/sleeves.py): Alpaca crypto
    # needs fractional quantities and GTC orders; the pre-registered rule
    # trades only at its weekly review, with its own within-sleeve band.
    fractional = True
    time_in_force = "gtc"
    drift_check = False
    # Staleness guard: a last bar older than this (UTC calendar days before
    # asof's date) means the data feed is broken, not that BTC is flat.
    max_stale_days = 3

    def __init__(
        self,
        weight: float,
        *,
        name: str = "btc_trend",
        symbol: str = "BTC/USD",
        lookback_days: int = 28,
        vol_window: int = 63,
        target_vol: float = 0.40,
        band_abs: float = 0.10,
    ) -> None:
        self.name = name
        self.weight = weight
        self.symbol = symbol
        self.lookback_days = lookback_days
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.band_abs = band_abs
        self.band_within = band_abs

    def symbols(self) -> list[str]:
        return [self.symbol]

    def target_weights(self, asof: datetime, history: dict[str, pd.Series]) -> dict[str, float]:
        closes = history.get(self.symbol)
        if closes is None or len(closes) == 0:
            raise ValueError(f"no {self.symbol} history as of {asof}")

        closes = closes.sort_index()
        closes = closes[~closes.index.duplicated(keep="last")]
        asof_ts = pd.Timestamp(asof)
        closes = closes[closes.index <= asof_ts]
        if closes.empty:
            raise ValueError(f"{self.symbol} history has no bars <= asof={asof}")

        # THE RULE's calendar is UTC calendar days, not a trading-day
        # calendar (BTC trades every day) -- reindex to the full daily range
        # and forward-fill, mirroring the evaluation harness's own
        # load_inputs() (any data-vendor gap in an otherwise-liquid crypto
        # series is filled forward, never left to silently shrink the
        # lookback/vol windows).
        full_idx = pd.date_range(closes.index.min(), closes.index.max(), freq="D")
        px = closes.reindex(full_idx).ffill()

        stale = (asof_ts.normalize() - px.index.max()).days
        if stale > self.max_stale_days:
            raise ValueError(
                f"{self.symbol} history is stale: last bar {px.index.max().date()} is {stale} days before {asof_ts.date()}"
            )
        sundays = px.index[px.index.dayofweek == 6]
        if len(sundays) == 0:
            raise ValueError(f"no completed Sunday-UTC review in {self.symbol} history up to {asof}")
        review_date = sundays[-1]
        needed = max(self.lookback_days, self.vol_window) + 1
        if px.index.get_loc(review_date) < needed:
            raise ValueError(
                f"{self.symbol} history too short: need {needed} daily bars before review {review_date.date()}"
            )

        # Same math as scripts/run_alt_premia_evaluation.py's c1_signal(),
        # evaluated at the latest available Sunday close.
        on_series = (px / px.shift(self.lookback_days) - 1.0) > 0
        sigma_series = px.pct_change().rolling(self.vol_window).std() * math.sqrt(365)

        on = bool(on_series.loc[review_date])
        sigma = float(sigma_series.loc[review_date])

        if not np.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"{self.symbol}: unusable volatility {sigma} at review {review_date.date()}")
        if not on:
            log.info("BtcTrendSleeve: flat (trend off) as of review %s", review_date.date())
            return {self.symbol: 0.0}

        target = min(1.0, self.target_vol / sigma)
        log.info(
            "BtcTrendSleeve: target=%.4f as of review %s (sigma=%.4f annualised)",
            target, review_date.date(), sigma,
        )
        return {self.symbol: target}

    def is_rebalance_due(self, asof: datetime, last_rebalance: datetime | None) -> bool:
        """True on the first check after each Sunday-UTC close not yet
        acted on.

        "Acted on" is tracked purely via *last_rebalance* (the timestamp the
        caller last treated as a review), consistent with this sleeve having
        no internal state of its own -- if the caller updates
        ``last_rebalance`` on every review (whether or not it actually
        traded, since the rule may leave on/off and weight unchanged), this
        returns True exactly once per week.
        """
        last_sunday = self._last_completed_sunday(asof)
        if last_rebalance is None:
            return True
        last_rebalance_date = pd.Timestamp(last_rebalance).normalize()
        return last_rebalance_date < last_sunday

    @staticmethod
    def _last_completed_sunday(asof: datetime) -> pd.Timestamp:
        """Most recent Sunday (UTC) whose close is already reflected in
        "completed bars only" history as of *asof* -- i.e. the most recent
        Sunday strictly before ``asof``'s own calendar date (that date's own
        bar, if it's a Sunday, is still forming until its own close)."""
        d = pd.Timestamp(asof).normalize() - pd.Timedelta(days=1)
        offset = (d.weekday() - 6) % 7
        return d - pd.Timedelta(days=int(offset))
