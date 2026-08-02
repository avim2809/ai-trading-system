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
    3. confidence: a weighted blend of perf_win_rate_{1m,3m,6m,1y} (each a
       0-1 win-rate) for whichever signal trading-parameters actually called
       (see DanelfinProvider.get_live_signals' buy/sell fix) — weighted
       toward 3m (matches this strategy's own return horizon) but informed
       by shorter/longer track records too, since a signal that only works
       on one specific horizon is less trustworthy than one that's
       consistently right across horizons. Missing horizons are simply
       excluded from the weighted average (renormalized over whatever is
       present), and the whole thing defaults to a neutral 0.5 if every
       horizon is missing. Then nudged by perf_avg_alpha_3m — a genuine
       alpha-vs-benchmark figure, not just a raw win-rate, which
       distinguishes "beats a falling market" from "beats a rising one" —
       clamped to a modest +/-10% adjustment so one noisy alpha figure
       can't swing confidence on its own. Final confidence is clamped to
       [0, 1] regardless.

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

# Weighted toward 3m (matches this strategy's own pf_median_return_3m
# horizon) but blended with shorter/longer track records — a signal that's
# only right on one specific horizon is less trustworthy than one that's
# consistently right across horizons.
_WIN_RATE_WEIGHTS: dict[str, float] = {
    "perf_win_rate_1m": 0.15,
    "perf_win_rate_3m": 0.40,
    "perf_win_rate_6m": 0.30,
    "perf_win_rate_1y": 0.15,
}
# avg_alpha_3m of +/-0.5 (a large alpha figure) nudges confidence by at
# most +/-10% — a secondary adjustment, not the primary confidence driver.
_ALPHA_ADJUSTMENT_SCALE = 0.2
_ALPHA_CLIP = 0.5


def _blend_confidence(row: pd.Series) -> float:
    """Weighted win-rate blend across horizons, nudged by avg_alpha_3m."""
    weighted_sum = 0.0
    weight_total = 0.0
    for col, weight in _WIN_RATE_WEIGHTS.items():
        val = row.get(col)
        if pd.notna(val):
            weighted_sum += float(val) * weight
            weight_total += weight

    confidence = (weighted_sum / weight_total) if weight_total > 0 else 0.5

    alpha = row.get("perf_avg_alpha_3m")
    if pd.notna(alpha):
        clipped_alpha = max(-_ALPHA_CLIP, min(float(alpha), _ALPHA_CLIP))
        confidence *= 1.0 + clipped_alpha * _ALPHA_ADJUSTMENT_SCALE

    return max(0.0, min(confidence, 1.0))


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

            confidence = _blend_confidence(row)

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
                        "perf_win_rate_1m": float(row["perf_win_rate_1m"])
                        if pd.notna(row.get("perf_win_rate_1m")) else float("nan"),
                        "perf_win_rate_3m": float(row["perf_win_rate_3m"])
                        if pd.notna(row.get("perf_win_rate_3m")) else float("nan"),
                        "perf_win_rate_6m": float(row["perf_win_rate_6m"])
                        if pd.notna(row.get("perf_win_rate_6m")) else float("nan"),
                        "perf_win_rate_1y": float(row["perf_win_rate_1y"])
                        if pd.notna(row.get("perf_win_rate_1y")) else float("nan"),
                        "perf_alpha_win_rate_3m": float(row["perf_alpha_win_rate_3m"])
                        if pd.notna(row.get("perf_alpha_win_rate_3m")) else float("nan"),
                        "perf_avg_alpha_3m": float(row["perf_avg_alpha_3m"])
                        if pd.notna(row.get("perf_avg_alpha_3m")) else float("nan"),
                    },
                )
            )
        return signals
