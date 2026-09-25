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
from firm.patterns.ml import calibration as ml_calibration
from firm.patterns.ml import inference as cnn_inference
from firm.patterns.ml import xgb_inference
from firm.patterns.ml.feature_engineering import build_features
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
        # 2026-09 pattern-recognition improvement round (see
        # docs/pattern_recognition_plan.md's newest phase). Every knob below
        # is OFF/no-op by default -- same convention as cnn_scoring_enabled
        # above: none of these has yet had its own dedicated walk-forward
        # re-validation (scripts/validate_pattern_cnn_walkforward.py is the
        # harness that validates cnn_scoring_enabled specifically; the same
        # pattern should be repeated per-knob before flipping any of these
        # true in config/live.yaml).
        #
        # ATR-scaled ZigZag threshold (quick-win): None = unchanged fixed
        # `zigzag_pct`; a positive float replaces it with a per-bar
        # ATR-scaled threshold (see firm.patterns.extrema.zigzag_pivots's
        # `threshold_fn` docstring) that adapts to each symbol's/regime's
        # own volatility instead of one fixed percentage universe-wide.
        "zigzag_atr_mult": None,
        # Retest-hold-vs-fail and weekly-timeframe-confluence quality
        # modifiers (detection-quality): each is a small, bounded
        # (+-*_modifier_scale points, added to the 0-100 quality_score and
        # re-clipped) adjustment -- never a hard gate, never an independent
        # signal. See firm.patterns.confirmation.retest_score_modifier /
        # firm.patterns.confluence.confluence_modifier.
        "retest_modifier_enabled": False,
        "retest_lookback_bars": 10,
        "retest_modifier_scale": 3.0,
        "confluence_modifier_enabled": False,
        "confluence_lookback_weeks": 8,
        "confluence_modifier_scale": 3.0,
        # XGBoost pattern-confirmation ensemble (model-ensemble-sizing):
        # when enabled (and the ONNX model/onnxruntime are available --
        # firm.patterns.ml.xgb_inference.is_available()), blends the
        # XGBoost classifier's calibrated P(target hit) for this specific
        # match with the CNN-or-rule-based quality fraction using a fixed
        # weight, then *agreement-gates* the blend: when the two scores
        # disagree by more than `xgb_agreement_gate_threshold`, the blended
        # quality is dampened by `xgb_agreement_gate_dampen` rather than
        # trusted outright -- two independently-trained models agreeing is
        # much stronger evidence than either alone, so a sharp disagreement
        # should reduce confidence, not just average it away. See
        # generate()'s inline comments for the exact blend/gate formula.
        "xgb_confirmation_enabled": False,
        "xgb_blend_weight": 0.5,
        "xgb_agreement_gate": True,
        "xgb_agreement_gate_threshold": 0.35,
        "xgb_agreement_gate_dampen": 0.7,
        # Optional calibration sidecars (paths previously written by
        # firm.patterns.ml.calibration.save_calibration -- a JSON dict with
        # a "type" discriminator: "temperature" for the CNN's pre-softmax
        # logits, "sigmoid" for XGBoost's raw target-hit probability). None
        # (default) means "proceed uncalibrated" for that model -- see
        # firm.patterns.ml.calibration's module docstring for why each model
        # family needs a different calibration technique.
        "cnn_calibration_path": None,
        "xgb_calibration_path": None,
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

        zigzag_atr_mult = p["zigzag_atr_mult"]
        zigzag_atr_mult = float(zigzag_atr_mult) if zigzag_atr_mult is not None else None
        retest_enabled = bool(p["retest_modifier_enabled"])
        retest_lookback_bars = int(p["retest_lookback_bars"])
        retest_modifier_scale = float(p["retest_modifier_scale"])
        confluence_enabled = bool(p["confluence_modifier_enabled"])
        confluence_lookback_weeks = int(p["confluence_lookback_weeks"])
        confluence_modifier_scale = float(p["confluence_modifier_scale"])

        xgb_enabled = bool(p["xgb_confirmation_enabled"])
        xgb_blend_weight = float(p["xgb_blend_weight"])
        xgb_agreement_gate = bool(p["xgb_agreement_gate"])
        xgb_gate_threshold = float(p["xgb_agreement_gate_threshold"])
        xgb_gate_dampen = float(p["xgb_agreement_gate_dampen"])

        universe = pit_view.universe
        if not universe:
            return []

        prices_df = pit_view.prices(symbols=universe, lookback_days=lookback_days)
        if prices_df.empty:
            return []

        # One availability check per generate() call (the underlying session
        # load is itself a cached lazy singleton -- see
        # firm.patterns.ml.inference._load_session /
        # firm.patterns.ml.xgb_inference._load_session) so the active
        # scoring mode is logged clearly without spamming per-symbol.
        cnn_available = bool(p["cnn_scoring_enabled"]) and cnn_inference.is_available()
        xgb_available = xgb_enabled and xgb_inference.is_available()

        cnn_calibration = None
        cnn_temperature = 1.0
        if p["cnn_calibration_path"]:
            cnn_calibration = ml_calibration.load_calibration(p["cnn_calibration_path"])
            if cnn_calibration and cnn_calibration.get("type") == "temperature":
                cnn_temperature = float(cnn_calibration.get("temperature", 1.0))
            elif cnn_calibration:
                log.debug(
                    "pattern_recognition: cnn_calibration_path=%s has type=%r, not "
                    "'temperature' -- ignoring", p["cnn_calibration_path"], cnn_calibration.get("type"),
                )

        xgb_calibration = None
        if p["xgb_calibration_path"]:
            xgb_calibration = ml_calibration.load_calibration(p["xgb_calibration_path"])

        log.info(
            "pattern_recognition: quality scoring mode=%s, xgb_confirmation=%s, "
            "zigzag=%s, retest_modifier=%s, confluence_modifier=%s",
            "cnn" if cnn_available else "rule_based",
            "on" if xgb_available else "off",
            f"atr(mult={zigzag_atr_mult})" if zigzag_atr_mult else f"fixed(pct={zigzag_pct})",
            "on" if retest_enabled else "off",
            "on" if confluence_enabled else "off",
        )

        signals: list[Signal] = []
        cnn_scored = 0
        xgb_scored = 0
        rule_based_only = 0
        for symbol, sym_df in prices_df.groupby("symbol"):
            sym_df = sym_df.sort_values("date")
            try:
                ohlcv = _adjusted_ohlc(sym_df)
                matches = scan_symbol(
                    ohlcv,
                    enabled_patterns=enabled_set,
                    zigzag_pct=zigzag_pct,
                    zigzag_atr_mult=zigzag_atr_mult,
                    min_score=min_score,
                    confirm_lookback_bars=confirm_lookback_bars,
                    stop_atr_floor=stop_atr_floor,
                    retest_modifier_enabled=retest_enabled,
                    retest_lookback_bars=retest_lookback_bars,
                    retest_modifier_scale=retest_modifier_scale,
                    confluence_modifier_enabled=confluence_enabled,
                    confluence_lookback_weeks=confluence_lookback_weeks,
                    confluence_modifier_scale=confluence_modifier_scale,
                    dates=sym_df["date"] if confluence_enabled else None,
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
                    temperature=cnn_temperature,
                )
            if cnn_quality is not None:
                base_quality_fraction = cnn_quality
                scoring_mode = "cnn"
                cnn_scored += 1
            else:
                base_quality_fraction = rule_based_fraction
                scoring_mode = "rule_based"

            # XGBoost pattern-confirmation ensemble (model-ensemble-sizing,
            # 2026-09): a fixed-weight blend of the XGBoost classifier's
            # calibrated P(target hit) for *this* match with the
            # CNN-or-rule-based quality fraction above, agreement-gated --
            # see default_params' docstring comments for the rationale.
            calibrated_probability = None
            xgb_p_target = None
            if xgb_available:
                features = build_features(best, ohlcv)
                xgb_result = xgb_inference.score_pattern_confirmation(
                    np.fromiter(features.values(), dtype=np.float32, count=len(features)),
                    apply_calibration=xgb_calibration,
                )
                if xgb_result is not None:
                    _, _, xgb_p_target = xgb_result
                    calibrated_probability = xgb_p_target

            if xgb_p_target is not None:
                disagreement = abs(xgb_p_target - base_quality_fraction)
                blended = (
                    xgb_blend_weight * xgb_p_target
                    + (1.0 - xgb_blend_weight) * base_quality_fraction
                )
                if xgb_agreement_gate and disagreement > xgb_gate_threshold:
                    log.debug(
                        "pattern_recognition: %s %s XGBoost/%s disagreement=%.3f > "
                        "threshold=%.3f -- dampening blended quality %.3f by %.2fx",
                        symbol, best.pattern, scoring_mode, disagreement,
                        xgb_gate_threshold, blended, xgb_gate_dampen,
                    )
                    blended *= xgb_gate_dampen
                quality_fraction = float(np.clip(blended, 0.0, 1.0))
                scoring_mode = f"{scoring_mode}+xgb"
                xgb_scored += 1
            else:
                quality_fraction = base_quality_fraction
                if cnn_quality is None:
                    rule_based_only += 1

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
                        "xgb_p_target": xgb_p_target,
                        # Meta-labeling convention consumed by
                        # TraderAgent._signal_calibrated_edge -- only
                        # populated when the XGBoost ensemble actually
                        # scored this match (never a placeholder value).
                        "calibrated_probability": calibrated_probability,
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
            "pattern_recognition: %d symbols scanned, %d signals (%d CNN-scored, "
            "%d XGB-ensembled, %d rule-based-only)",
            len(universe), len(signals), cnn_scored, xgb_scored, rule_based_only,
        )
        return signals
