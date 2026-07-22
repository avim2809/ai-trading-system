"""Event-driven strategy — post-earnings announcement drift (PEAD).

Financial intuition:
    Stocks that report positive earnings surprises tend to drift upward for
    weeks after the announcement, and vice versa for negative surprises.
    This well-documented anomaly (Ball & Brown, 1968; Bernard & Thomas, 1989)
    persists because the market under-reacts to the information content of
    earnings.

Data inputs:
    - Fundamentals from PitView.fundamentals(): EPS data to detect earnings
      events and compute surprises.
    - Prices from PitView.prices(): to detect large post-earnings moves as
      a proxy when direct surprise data is unavailable.

Signal logic:
    1. Detect earnings events by looking for changes in reported EPS across
       consecutive fundamental snapshots.
    2. Compute the earnings surprise as the percentage change in EPS.
    3. If surprise > +threshold → bullish drift signal that decays over
       drift_days.
    4. If surprise < -threshold → bearish drift signal that decays.
    5. Signal score = sign(surprise) * magnitude * decay_factor^(days_since).

Portfolio construction approach:
    Go long on positive surprises and short on negative surprises for
    drift_days after the event.  Size proportionally to surprise magnitude.

Risk notes:
    PEAD is strongest for small-caps and around Q4 earnings.  Crowding has
    reduced the alpha over time.  Event detection from fundamental snapshots
    can be noisy — pair with sentiment data for confirmation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register


@register("event_driven")
class EventDrivenStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("event_driven", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        surprise_threshold: float = self.params.get("surprise_threshold", 0.05)
        drift_days: int = self.params.get("drift_days", 21)
        decay_factor: float = self.params.get("decay_factor", 0.95)

        universe = pit_view.universe
        if not universe:
            return []

        fund_df = pit_view.fundamentals(symbols=universe)
        prices_df = pit_view.prices(symbols=universe, lookback_days=drift_days + 10)

        signals: list[Signal] = []
        asof = pit_view.asof

        has_fundamentals = (
            not fund_df.empty
            and "eps" in fund_df.columns
            and "date" in fund_df.columns
        )

        if has_fundamentals:
            signals.extend(
                self._signals_from_fundamentals(
                    fund_df, asof, surprise_threshold, drift_days, decay_factor
                )
            )

        # Price-move proxy per the docstring is "a proxy when direct
        # surprise data is unavailable" — per SYMBOL, not per universe. Now
        # that get_fundamentals() can actually return the 2+ snapshots this
        # needs (see PointInTimeDataStore.get_fundamentals), some symbols
        # will have a real fundamentals-based signal while others won't
        # (no recent earnings event) — only fall back to the proxy for the
        # latter, instead of skipping it for the WHOLE universe the moment
        # any single symbol got a real signal.
        covered = {s.symbol for s in signals}
        remaining_universe = [s for s in universe if s not in covered]
        if remaining_universe and not prices_df.empty:
            remaining_prices = prices_df[prices_df["symbol"].isin(remaining_universe)]
            signals.extend(
                self._signals_from_price_moves(
                    remaining_prices, asof, surprise_threshold, drift_days, decay_factor
                )
            )

        return signals

    def _signals_from_fundamentals(
        self,
        fund_df: pd.DataFrame,
        asof,
        surprise_threshold: float,
        drift_days: int,
        decay_factor: float,
    ) -> list[Signal]:
        fund_df = fund_df.copy()
        fund_df["date"] = pd.to_datetime(fund_df["date"])
        fund_df = fund_df.sort_values("date")

        signals: list[Signal] = []
        for symbol, grp in fund_df.groupby("symbol"):
            eps_vals = grp[["date", "eps"]].dropna(subset=["eps"])
            if len(eps_vals) < 2:
                continue

            prev_eps = eps_vals["eps"].iloc[-2]
            curr_eps = eps_vals["eps"].iloc[-1]
            event_date = eps_vals["date"].iloc[-1]

            if prev_eps == 0:
                if curr_eps == 0:
                    continue
                surprise = float(np.sign(curr_eps))
            else:
                surprise = float((curr_eps - prev_eps) / abs(prev_eps))

            if abs(surprise) < surprise_threshold:
                continue

            days_since = (pd.Timestamp(asof) - pd.Timestamp(event_date)).days
            # drift_days is a TRADING-day count elsewhere in this file
            # (prices(lookback_days=drift_days + 10) is a row count, per
            # PointInTimeDataStore's contract), but days_since is measured
            # in raw CALENDAR days — comparing them directly truncated the
            # effective drift window to ~70% of what drift_days actually
            # specifies (a 21-trading-day window is ~29 calendar days, not
            # 21). Convert once for this cutoff check.
            drift_days_calendar = drift_days * 7 / 5
            if days_since < 0 or days_since > drift_days_calendar:
                continue

            decay = decay_factor ** days_since
            score = float(np.sign(surprise) * min(abs(surprise), 1.0) * decay)
            score = float(np.clip(score, -1, 1))

            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="event_driven",
                    score=score,
                    confidence=min(abs(surprise), 1.0) * decay,
                    horizon=f"{drift_days}d",
                    asof=asof,
                    meta={
                        "earnings_surprise": surprise,
                        "days_since_event": days_since,
                        "decay": decay,
                        "event_date": str(event_date.date()),
                    },
                )
            )
        return signals

    def _signals_from_price_moves(
        self,
        prices_df: pd.DataFrame,
        asof,
        surprise_threshold: float,
        drift_days: int,
        decay_factor: float,
    ) -> list[Signal]:
        """Fallback: use large single-day price moves as event proxies."""
        pivot = (
            prices_df.pivot_table(index="date", columns="symbol", values="adj_close")
            .sort_index()
        )
        if len(pivot) < 5:
            return []

        daily_ret = pivot.pct_change()
        vol = daily_ret.rolling(20, min_periods=10).std()

        signals: list[Signal] = []
        for col in daily_ret.columns:
            rets = daily_ret[col].dropna()
            # Align volatility to the returns index; dropping NaNs on each
            # series separately yields mismatched labels and a pandas
            # "identically-labeled" comparison error.
            vols = vol[col].reindex(rets.index)
            if rets.empty or vols.dropna().empty:
                continue

            big_moves = rets[(rets.abs() > vols * 3.0).fillna(False)]
            if big_moves.empty:
                continue

            last_event_date = big_moves.index[-1]
            days_since = (pd.Timestamp(pivot.index[-1]) - pd.Timestamp(last_event_date)).days
            # Same calendar-vs-trading-day unit mismatch as
            # _signals_from_fundamentals — see the comment there.
            drift_days_calendar = drift_days * 7 / 5
            if days_since > drift_days_calendar:
                continue

            surprise = float(big_moves.iloc[-1])
            if abs(surprise) < surprise_threshold:
                continue

            decay = decay_factor ** days_since
            score = float(np.sign(surprise) * min(abs(surprise) * 5, 1.0) * decay)

            signals.append(
                Signal(
                    symbol=str(col),
                    strategy="event_driven",
                    score=float(np.clip(score, -1, 1)),
                    confidence=min(abs(surprise) * 3, 1.0) * decay,
                    horizon=f"{drift_days}d",
                    asof=asof,
                    meta={
                        "proxy_event": True,
                        "big_move_return": surprise,
                        "days_since_event": days_since,
                    },
                )
            )
        return signals
