"""Multi-bar chart pattern recognition strategy (Strategy #13).

Financial intuition:
    Classical technical analysis treats certain multi-week price shapes
    (Head & Shoulders, triangles, flags, cup & handle, ...) as evidence of
    accumulated order-flow information not yet fully reflected in price —
    e.g. a neckline in a Head & Shoulders top marks a support level that,
    once broken, releases trapped long positions. TA-Lib only covers 1-3 bar
    candlestick formations; this fills that gap with genuine multi-bar
    structural pattern detection (see docs/pattern_recognition_plan.md).
    Raw/unscored pattern detection is close to a coin flip, so every
    detection is quality-scored (geometry tightness, trendline/neckline fit,
    breakout volume, formation duration, follow-through) and only the
    highest-scoring confirmed pattern per symbol becomes a signal.

Data inputs:
    OHLCV from PitView.prices(), adjusted for splits/dividends by scaling
    raw high/low by the adj_close/close ratio (adj_close alone isn't enough —
    using it for "close" while leaving high/low raw would break the
    high >= close >= low invariant across a split boundary and corrupt the
    zigzag/geometry calculations).

Signal logic:
    1. Extract alternating peak/trough pivots via a percentage-reversal
       zigzag (firm.patterns.extrema.zigzag_pivots).
    2. Run all 9 rule detectors spanning reversal (H&S/IHS/Double & Triple
       Top-Bottom), triangle/wedge/rectangle, flag/pennant, and cup & handle
       families (firm.patterns.scanner.scan_symbol).
    3. Keep only patterns that have actually confirmed (price closed through
       the neckline/trendline/handle-high within a recent lookback window)
       and cleared min_score and min_risk_reward.
    4. Take the single highest-quality match per symbol (emitting more than
       one would double-count that symbol in the downstream cross-sectional
       z-score — see docs/pattern_recognition_plan.md deviation #4) and
       signal it as +quality/100 (long) or -quality/100 (short); consumers
       z-score across the universe in the analyst layer like every other
       strategy.
    5. The "quality" fed into that score is, when available, the CNN/GAF
       image validator's learned score (``firm.patterns.ml.inference``) —
       not the rule-based scanner's own geometric ``quality_score``. Named
       rule-based chart patterns have no consistently replicated edge once
       corrected for data-snooping (Marshall & Cahan; Sullivan/Timmermann/
       White), whereas a CNN trained on raw price-chart images does show
       real out-of-sample edge (Jiang/Kelly/Xiu, J. Finance 2023) — so the
       rule-based scanner is used here only as a candidate generator (where
       a pattern-like shape/breakout exists at all), with the CNN deciding
       how good it actually is. If the ONNX model or ``onnxruntime`` isn't
       available at runtime, this falls back cleanly to the original
       rule-based ``quality_score`` — never a hard failure. See
       ``firm.patterns.ml.inference`` for the fail-soft contract.

Portfolio construction approach:
    Entry/stop/target/risk-reward for the winning pattern are carried in
    ``meta`` for the trader/risk layer and UI to surface — this strategy
    itself only emits a directional conviction score, not position sizing.

Risk notes:
    Pattern detection has no notion of market regime by itself (no strategy
    reaches into another's output in this codebase — regime blending already
    happens downstream, in agents/research/_regime_weights.py, exactly as it
    does for every other strategy including regime_hmm). A confirmed pattern
    in a strongly adverse regime will still fire here at full score; regime
    context should be layered on afterward, not inside this strategy.
    Registered but NOT added to config/live.yaml's strategies.enabled list —
    see docs/pattern_recognition_plan.md 5.1 for why that's deliberate.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from firm.contracts.models import Signal
from firm.patterns.ml import inference as cnn_inference
from firm.patterns.scanner import scan_symbol
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

log = logging.getLogger(__name__)


def _adjusted_ohlc(sym_df: pd.DataFrame) -> pd.DataFrame:
    """Scale raw high/low by adj_close/close so the adjusted series stays
    internally consistent (high >= close >= low) across split/dividend
    boundaries, instead of mixing raw high/low with an adjusted close.
    """
    close = sym_df["close"].to_numpy(dtype=float)
    adj_close = sym_df["adj_close"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(close != 0, adj_close / close, 1.0)
    factor = np.nan_to_num(factor, nan=1.0, posinf=1.0, neginf=1.0)
    return pd.DataFrame(
        {
            "high": sym_df["high"].to_numpy(dtype=float) * factor,
            "low": sym_df["low"].to_numpy(dtype=float) * factor,
            "close": adj_close,
            "volume": sym_df["volume"].to_numpy(dtype=float),
        }
    )


@register("pattern_recognition")
class PatternRecognitionStrategy(BaseStrategy):
    #: Surfaced by the /api/strategies endpoint so the UI renders editable
    #: parameter fields. Also the single source of truth for runtime defaults.
    default_params: dict = {
        "lookback_days": 252,
        "zigzag_pct": 0.03,
        "min_score": 60.0,
        "min_risk_reward": 1.5,
        "confirm_lookback_bars": 3,
        "stop_atr_floor": 1.5,
        "horizon": "10d",
        # None = all patterns enabled; else a list of pattern names from
        # firm.patterns.match.PatternMatch.pattern (e.g. ["cup_handle",
        # "head_shoulders_top"]) to restrict the scan to.
        "enabled_patterns": None,
        # Off by default (2026-09-20) -- deliberately NOT implied by ONNX
        # model file presence alone. A real walk-forward comparison on
        # cached data (2010-2023) found the CNN-reweighted signal WORSE
        # across every metric than the rule-based-only scanner (Sharpe
        # 0.415->0.314, max DD 3.85%->4.63%, whole-book Sharpe also worse)
        # against the model artifact that happened to be on disk --
        # consistent with that artifact having been trained on the CLI's
        # synthetic-data default, not real cached history (unconfirmed --
        # see docs/pattern_recognition_plan.md, no training-run entry for
        # the CNN unlike XGBoost/PPO). Flip true only after a real training
        # run (scripts/train_cnn_validator.py --data-source cache) is
        # re-validated the same way and clearly wins.
        "cnn_scoring_enabled": False,
    }

    def __init__(self, params: dict | None = None):
        super().__init__("pattern_recognition", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        p = {**self.default_params, **(self.params or {})}
        lookback_days = int(p["lookback_days"])
        zigzag_pct = float(p["zigzag_pct"])
        min_score = float(p["min_score"])
        min_risk_reward = float(p["min_risk_reward"])
        confirm_lookback_bars = int(p["confirm_lookback_bars"])
        stop_atr_floor = float(p["stop_atr_floor"])
        horizon = str(p["horizon"])
        enabled_patterns = p["enabled_patterns"]
        enabled_set = set(enabled_patterns) if enabled_patterns else None

        universe = pit_view.universe
        if not universe:
            return []

        prices_df = pit_view.prices(symbols=universe, lookback_days=lookback_days)
        if prices_df.empty:
            return []

        # One availability check per generate() call (the underlying session
        # load is itself a cached lazy singleton -- see
        # firm.patterns.ml.inference._load_session) so the active scoring
        # mode is logged clearly without spamming per-symbol.
        cnn_available = bool(p["cnn_scoring_enabled"]) and cnn_inference.is_available()
        log.info(
            "pattern_recognition: quality scoring mode=%s",
            "cnn" if cnn_available else "rule_based",
        )

        signals: list[Signal] = []
        cnn_scored = 0
        for symbol, sym_df in prices_df.groupby("symbol"):
            sym_df = sym_df.sort_values("date")
            try:
                ohlcv = _adjusted_ohlc(sym_df)
                matches = scan_symbol(
                    ohlcv,
                    enabled_patterns=enabled_set,
                    zigzag_pct=zigzag_pct,
                    min_score=min_score,
                    confirm_lookback_bars=confirm_lookback_bars,
                    stop_atr_floor=stop_atr_floor,
                )
            except Exception:
                log.debug("pattern scan failed for %s", symbol, exc_info=True)
                continue
            if not matches:
                continue

            best = matches[0]
            if best.risk_reward < min_risk_reward:
                continue

            direction_sign = 1.0 if best.direction == "long" else -1.0
            rule_based_fraction = min(best.quality_score / 100.0, 1.0)
            cnn_quality = None
            if cnn_available:
                cnn_quality = cnn_inference.score_pattern_quality(
                    ohlcv["close"].to_numpy(dtype=float), best.confirm_index,
                )
            if cnn_quality is not None:
                quality_fraction = cnn_quality
                scoring_mode = "cnn"
                cnn_scored += 1
            else:
                quality_fraction = rule_based_fraction
                scoring_mode = "rule_based"
            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="pattern_recognition",
                    score=direction_sign * quality_fraction,
                    confidence=quality_fraction,
                    horizon=horizon,
                    asof=pit_view.asof,
                    meta={
                        "pattern": best.pattern,
                        "direction": best.direction,
                        "entry": best.entry,
                        "stop": best.stop,
                        "target": best.target,
                        "risk_reward": best.risk_reward,
                        "quality_score": best.quality_score,
                        "rule_based_quality_fraction": rule_based_fraction,
                        "cnn_quality_fraction": cnn_quality,
                        "scoring_mode": scoring_mode,
                        "score_breakdown": best.score_breakdown,
                        "volume_ratio": best.volume_ratio,
                        "duration_bars": best.duration_bars,
                        "bars_since_confirm": len(ohlcv) - 1 - best.confirm_index,
                        "other_patterns": [m.pattern for m in matches[1:4]],
                    },
                )
            )

        log.info(
            "pattern_recognition: %d symbols scanned, %d signals (%d CNN-scored, %d rule-based)",
            len(universe), len(signals), cnn_scored, len(signals) - cnn_scored,
        )
        return signals
