"""Volatility breakout strategy.

Financial intuition:
    Markets alternate between periods of low volatility (compression) and
    high volatility (expansion).  A breakout from a low-volatility regime
    often signals the start of a sustained directional move.  The strategy
    waits for vol compression and then enters when price exceeds the
    recent range, riding the expansion (Bollinger, 2001).

Data inputs:
    OHLC prices from PitView.prices() with lookback for range and ATR
    computation (~30-60 trading days).

Signal logic:
    1. Compute N-day price range as (max high - min low) over range_days.
    2. Compute Average True Range (ATR) over atr_period.
    3. Identify a breakout: current close > previous range high +
       breakout_multiplier * ATR → bullish; close < range low -
       breakout_multiplier * ATR → bearish.
    4. Vol filter: only emit signals when current realized vol is below
       vol_threshold (low vol precedes breakout — the "squeeze").
    5. Signal score = breakout_magnitude / ATR, normalized to [-1, 1].

Portfolio construction approach:
    Enter in the direction of the breakout.  Size inversely to ATR so
    that each position targets a similar dollar risk.

Risk notes:
    False breakouts are frequent — many setups will reverse quickly.
    A trailing stop at 1-2× ATR is recommended.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register


@register("volatility_breakout")
class VolatilityBreakoutStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("volatility_breakout", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        range_days: int = self.params.get("range_days", 20)
        atr_period: int = self.params.get("atr_period", 14)
        vol_threshold: float = self.params.get("vol_threshold", 0.25)
        breakout_multiplier: float = self.params.get("breakout_multiplier", 0.5)

        universe = pit_view.universe
        if not universe:
            return []

        needed = max(range_days, atr_period) + 30
        prices_df = pit_view.prices(symbols=universe, lookback_days=needed)
        if prices_df.empty:
            return []

        prices_df = prices_df.copy()
        prices_df["date"] = pd.to_datetime(prices_df["date"])

        signals: list[Signal] = []
        for symbol in universe:
            sym_df = (
                prices_df[prices_df["symbol"] == symbol]
                .sort_values("date")
                .reset_index(drop=True)
            )
            if len(sym_df) < range_days + 2:
                continue

            high = sym_df["high"].values if "high" in sym_df.columns else sym_df["adj_close"].values
            low = sym_df["low"].values if "low" in sym_df.columns else sym_df["adj_close"].values
            close = sym_df["adj_close"].values if "adj_close" in sym_df.columns else sym_df["close"].values

            range_high = np.max(high[-range_days - 1:-1])
            range_low = np.min(low[-range_days - 1:-1])

            true_ranges = []
            for i in range(1, min(atr_period + 1, len(close))):
                tr = max(
                    high[-i] - low[-i],
                    abs(high[-i] - close[-i - 1]) if i < len(close) else 0,
                    abs(low[-i] - close[-i - 1]) if i < len(close) else 0,
                )
                true_ranges.append(tr)

            if not true_ranges:
                continue
            atr = float(np.mean(true_ranges))
            if atr == 0:
                continue

            returns = np.diff(close[-21:]) / close[-21:-1]
            realized_vol = float(np.std(returns) * np.sqrt(252)) if len(returns) > 5 else 999.0

            current_close = close[-1]
            upper_break = range_high + breakout_multiplier * atr
            lower_break = range_low - breakout_multiplier * atr

            if realized_vol > vol_threshold and vol_threshold > 0:
                continue

            if current_close > upper_break:
                magnitude = (current_close - range_high) / atr
                score = float(np.clip(magnitude / 3.0, 0, 1))
            elif current_close < lower_break:
                magnitude = (range_low - current_close) / atr
                score = float(np.clip(-magnitude / 3.0, -1, 0))
            else:
                continue

            signals.append(
                Signal(
                    symbol=symbol,
                    strategy="volatility_breakout",
                    score=score,
                    confidence=min(abs(score), 1.0),
                    horizon="5d",
                    asof=pit_view.asof,
                    meta={
                        "atr": atr,
                        "realized_vol": realized_vol,
                        "range_high": float(range_high),
                        "range_low": float(range_low),
                        "breakout_direction": "long" if score > 0 else "short",
                    },
                )
            )

        return signals
