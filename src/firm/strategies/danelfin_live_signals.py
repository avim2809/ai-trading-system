"""Danelfin live-signals strategy — trading-parameters + price-forecast + performance.

Financial intuition:
    Danelfin's ``/v3/trading-parameters`` publishes a buy/hold/sell call per
    ticker; ``/v3/price-forecast`` gives a probabilistic 3-month return
    distribution; ``/v3/performance`` gives that signal's own historical
    win-rate track record. Combining them: trade in the direction of the
    call, size by the forecast's expected magnitude, weight confidence by
    how often that signal has actually worked historically.

Data inputs:
    PitView.live_signals(): LIVE_SIGNAL_COLS, backed by
    firm.data.providers.danelfin.DanelfinProvider.get_live_signals.

Cannot be backtested — read this before touching promotion discipline:
    Per Danelfin's own docs, ``/v3/*`` always reflects "right now" with no
    historical dates. This project's normal promotion gate (walk-forward
    A/B across 3 diagnostic windows, see danelfin_ai_score.py) is
    structurally impossible here: pit_view.live_signals() always returns an
    empty frame in backtests (no cache-backed history exists to populate
    one, ever), so this strategy always emits zero signals in a backtest —
    that's expected, not a bug. Enabling this live is a live-only,
    unbacktested-by-construction judgment call, not an A/B-validated one;
    see docs/investing_pro_integration.md for how that's documented.

Signal logic:
    1. direction: +1.0 for tp_signal == "buy", -1.0 for "sell", skip
       everything else (including "hold" and any unrecognized value) — no
       basis to take a position on a name Danelfin isn't calling either way.
    2. magnitude: |pf_median_return_3m| (a 0-1 decimal, e.g. 0.064 == 6.4%)
       scaled by return_scale (default 40x, so a ~25% forecast return caps
       out near this project's raw-score sanity ceiling) and clipped to
       [0, 10] — bounded by construction regardless of how large a forecast
       return Danelfin ever returns.
    3. confidence: perf_win_rate_3m (already a 0-1 win-rate) for whichever
       signal trading-parameters actually called (see
       DanelfinProvider.get_live_signals' buy/sell fix), defaulting to a
       neutral 0.5 when missing rather than treating an absent track record
       as either strong or zero confidence.

Portfolio construction approach:
    Long Danelfin's live "buy" calls, short its live "sell" calls, sized by
    forecast magnitude and win-rate confidence.

Risk notes:
    Same black-box-vendor caveat as danelfin_ai_score, compounded by having
    no backtest evidence at all for *this* specific signal combination —
    treat as a much higher-risk, lower-evidence input than danelfin_ai_score.
    tp_stop_loss_pct/tp_take_profit_pct are deliberately NOT used here (see
    DanelfinProvider's module docstring) — this strategy only reads the
    directional call, not the suggested execution levels.
"""

from __future__ import annotations

import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

_SIGNAL_DIRECTION: dict[str, float] = {"buy": 1.0, "sell": -1.0}
_MAX_RAW = 10.0


@register("danelfin_live_signals")
class DanelfinLiveSignalsStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("danelfin_live_signals", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        universe = pit_view.universe
        if not universe:
            return []

        # 40x: a 25% pf_median_return_3m forecast (large, but observed
        # magnitudes are typically single-digit percent — see AAPL's 6.4%
        # verified live) reaches the 10.0 clip; smaller, more typical
        # forecasts land well inside the sanity range without it.
        return_scale: float = self.params.get("return_scale", 40.0)

        df = pit_view.live_signals(symbols=universe)
        if df.empty or "tp_signal" not in df.columns:
            return []

        signals: list[Signal] = []
        for _, row in df.iterrows():
            tp_signal = row.get("tp_signal")
            direction = _SIGNAL_DIRECTION.get(str(tp_signal).lower() if pd.notna(tp_signal) else "")
            if direction is None:
                continue

            median_return = row.get("pf_median_return_3m")
            if pd.isna(median_return):
                continue
            magnitude = min(abs(float(median_return)) * return_scale, _MAX_RAW)
            if magnitude == 0.0:
                continue
            raw = direction * magnitude

            win_rate = row.get("perf_win_rate_3m")
            confidence = float(win_rate) if pd.notna(win_rate) else 0.5
            confidence = max(0.0, min(confidence, 1.0))

            signals.append(
                Signal(
                    symbol=str(row["symbol"]),
                    strategy="danelfin_live_signals",
                    score=raw,
                    confidence=confidence,
                    horizon="60d",
                    asof=pit_view.asof,
                    meta={
                        "tp_signal": str(tp_signal),
                        "pf_median_return_3m": float(median_return),
                        "perf_win_rate_3m": float(win_rate) if pd.notna(win_rate) else float("nan"),
                        "perf_alpha_win_rate_3m": float(row["perf_alpha_win_rate_3m"])
                        if pd.notna(row.get("perf_alpha_win_rate_3m")) else float("nan"),
                    },
                )
            )
        return signals
