"""Calendar / seasonality effects strategy.

Financial intuition:
    Well-documented calendar anomalies create exploitable return patterns:
    - **Turn-of-month (TOM)** effect: equity returns are significantly
      higher during the last 1-2 and first 2-3 **trading days** of each
      calendar month (Lakonishok & Smidt, 1988).
    - **Day-of-week** effect: Mondays historically show lower returns
      (weekend uncertainty), while Fridays show slightly higher returns
      (position squaring).

Data inputs:
    Prices from PitView.prices() for historical day-of-week return
    computation, plus PitView.asof for calendar positioning.

Signal logic:
    1. **Day-of-week score**: compute the historical average return for
       each weekday across the universe.  Score the current day's expected
       return relative to the weekly mean.
    2. **Turn-of-month score**: +1 if asof falls within the TOM
       window measured in **trading days** (last *tom_days_before* or
       first *tom_days_after* sessions of the calendar month).
    3. Final score = (dow_score + tom_score) / 2, applied uniformly across
       the universe (calendar effects are market-wide).

Portfolio construction approach:
    This is primarily a timing signal: scale overall equity exposure up
    during favorable calendar windows and down otherwise.  Applied as a
    uniform tilt to all symbols.

Risk notes:
    Calendar effects are well-known and may be arbitraged away.  Effect
    sizes are small (a few basis points per day) so transaction costs
    matter.  Use as a supplementary overlay, not a primary alpha source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register


@register("seasonality")
class SeasonalityStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("seasonality", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        tom_days_before: int = self.params.get("tom_days_before", 1)
        tom_days_after: int = self.params.get("tom_days_after", 3)

        universe = pit_view.universe
        if not universe:
            return []

        asof = pd.Timestamp(pit_view.asof).normalize()

        prices_df = pit_view.prices(symbols=universe, lookback_days=252)
        dow_score = 0.0
        trading_dates: list[pd.Timestamp] = []
        if not prices_df.empty:
            dow_score = self._day_of_week_score(prices_df, asof)
            trading_dates = sorted(
                pd.to_datetime(prices_df["date"]).dt.normalize().unique()
            )

        tom_score = self._turn_of_month_score(
            asof, tom_days_before, tom_days_after, trading_dates
        )

        raw_score = (dow_score + tom_score) / 2.0
        score = float(np.clip(raw_score, -1, 1))
        confidence = min(abs(score) + 0.1, 1.0)

        signals: list[Signal] = []
        for symbol in universe:
            signals.append(
                Signal(
                    symbol=symbol,
                    strategy="seasonality",
                    score=score,
                    confidence=confidence,
                    horizon="1d",
                    asof=pit_view.asof,
                    meta={
                        "dow_score": dow_score,
                        "tom_score": tom_score,
                        "day_of_week": asof.day_name(),
                        "tom_uses_trading_days": bool(trading_dates),
                    },
                )
            )
        return signals

    @staticmethod
    def _day_of_week_score(prices_df: pd.DataFrame, asof: pd.Timestamp) -> float:
        """Score the current day-of-week based on historical return patterns."""
        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        pivot = (
            prices_df.pivot_table(index="date", columns="symbol", values="adj_close")
            .sort_index()
        )
        if len(pivot) < 20:
            return 0.0

        market_ret = pivot.pct_change().mean(axis=1).dropna()
        if market_ret.empty:
            return 0.0

        market_ret.index = pd.to_datetime(market_ret.index)
        dow_returns = market_ret.groupby(market_ret.index.dayofweek).mean()

        if dow_returns.empty:
            return 0.0

        overall_mean = dow_returns.mean()
        overall_std = dow_returns.std()
        if overall_std == 0:
            return 0.0

        current_dow = asof.dayofweek
        if current_dow not in dow_returns.index:
            return 0.0

        return float((dow_returns[current_dow] - overall_mean) / overall_std)

    @staticmethod
    def _turn_of_month_score(
        asof: pd.Timestamp,
        tom_days_before: int,
        tom_days_after: int,
        trading_dates: list[pd.Timestamp],
    ) -> float:
        """Return +1 if *asof* is in the turn-of-month trading-day window."""
        if trading_dates:
            eligible = [d for d in trading_dates if d <= asof]
            if not eligible:
                return 0.0
            ref = eligible[-1]
            month_start = ref.replace(day=1)
            month_end = month_start + pd.offsets.MonthEnd(0)
            month_sessions = list(pd.bdate_range(month_start, month_end))
            if ref not in month_sessions:
                return 0.0
            idx = month_sessions.index(ref)
            near_end = idx >= len(month_sessions) - tom_days_before
            near_start = idx < tom_days_after
            if near_end or near_start:
                return 1.0
            return 0.0

        # Fallback when no price calendar is available (should be rare).
        import calendar

        day = asof.day
        _, last_day = calendar.monthrange(asof.year, asof.month)
        near_end = day >= (last_day - tom_days_before)
        near_start = day <= tom_days_after
        return 1.0 if (near_end or near_start) else 0.0
