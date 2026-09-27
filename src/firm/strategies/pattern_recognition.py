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
    Pattern detection originally had no notion of market regime by itself
    (no strategy reaches into another's output in this codebase — regime
    blending already happens downstream, in agents/research/_regime_weights.py,
    exactly as it does for every other strategy including regime_hmm, and
    still does). As of Part A item 6 (2026-09-27) this strategy additionally
    builds its own MarketRegimeDetector instance the same way RiskAgent's
    regime_overlay does (strategies never receive ctx.market_regime, so this
    is a second, independent detector call, not a shared one) and applies a
    direction-aware confidence discount: a pattern whose direction conflicts
    with the labelled regime (e.g. a long pattern confirming in a Bear
    regime), or any pattern confirming in Chop, is discounted rather than
    fired at full score regardless of regime — gated by
    regime_discount_enabled (off by default, pending its own walk-forward
    re-validation, same convention as every other knob here). The downstream
    per-strategy score multiplier in _regime_weights.py is unaffected and
    still applies afterward on top of this — the two are complementary
    (this one touches quality_fraction/confidence at the point of
    detection; that one touches score post-hoc across all strategies).
    Enabled in both config/live.yaml and config/live_alpaca.yaml's
    strategies.enabled list since 2026-09-09 (see docs/pattern_recognition_plan.md
    §8 for the CNN/XGBoost-ensemble rollout and rollback history — the
    rule-based scanner runs live on both instances; the ML layer is
    config-gated off pending its own walk-forward re-validation).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from firm.contracts.models import Signal
from firm.patterns import sample_size, significance
from firm.patterns.ml import calibration as ml_calibration
from firm.patterns.ml import inference as cnn_inference
from firm.patterns.ml import xgb_inference
from firm.patterns.ml.feature_engineering import build_features
from firm.patterns.scanner import scan_symbol
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register

log = logging.getLogger(__name__)

#: Shared code-level asset (like the CNN/XGBoost model files themselves --
#: see firm.patterns.ml.inference.DEFAULT_MODEL_PATH's identical rationale):
#: both live instances share one checkout and should read the same
#: per-pattern sample counts from the same place, not a FIRM_DATA_DIR-
#: relative per-instance path.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SAMPLE_COUNTS_PATH = _PROJECT_ROOT / "data" / "models" / sample_size.DEFAULT_SAMPLE_COUNTS_FILENAME


def _regime_confidence_discount(
    regime_state,
    direction: str,
    *,
    chop_discount: float,
    misaligned_discount: float,
    min_separation: float,
    separation_damping_floor: float,
) -> float:
    """Direction-aware regime confidence discount (Part A item 6).

    A confirmed long pattern in an HMM-labelled Bear regime (or a short
    pattern in a Bull regime) is directionally misaligned with the
    prevailing market state and gets discounted; Chop gets a universal
    discount regardless of direction (breakout-style patterns are prone to
    whipsaw in range-bound markets). Bull/Bear discounts are themselves
    damped toward 1.0 (no-op) when ``RegimeState.separation`` indicates the
    label was assigned by a thin, noise-level margin -- same idiom as
    ``firm.strategies.regime_hmm``'s own separation-based damping. Chop's
    separation is always ``inf`` by convention (see ``RegimeState``'s own
    docstring), so its discount is never damped.
    """
    from firm.regime.model import BEAR, BULL, CHOP

    if regime_state.label == CHOP:
        return max(0.0, min(1.0, chop_discount))

    aligned = (regime_state.label == BULL and direction == "long") or (
        regime_state.label == BEAR and direction == "short"
    )
    if aligned:
        return 1.0

    misaligned_discount = max(0.0, min(1.0, misaligned_discount))
    if min_separation <= 0:
        return misaligned_discount
    damping = max(
        separation_damping_floor, min(1.0, regime_state.separation / min_separation)
    )
    return 1.0 + (misaligned_discount - 1.0) * damping


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
        # Meta-labeling secondary model (Part B item 2, 2026-09-27 -- see
        # firm.patterns.ml.labeling.label_meta_binary /
        # firm.patterns.ml.xgb_inference.score_pattern_meta_confirmation's
        # docstrings): a SEPARATE binary act/no-act model, distinct from
        # the 3-class model blended into quality_fraction above. This is
        # the correct source for meta["calibrated_probability"] (de
        # Prado's recipe: the secondary/meta model IS the calibrated
        # confidence gate; a 3-class model's own p_target answering "which
        # of three things happens" is a different question). Independent
        # on/off switch from xgb_confirmation_enabled -- one strategy can
        # use either, both, or neither model. Off by default, same
        # convention as every other knob here; a real
        # data/models/pattern_xgb_meta.onnx artifact now exists (retrained
        # 2026-09-27) but the isolated walk-forward evaluation of this
        # strategy's own edge still failed its pre-registered PBO/DSR bar
        # (see docs/pattern_ml_isolated_evaluation_2026_09.md) -- flipping
        # this true needs its own re-validation, not just artifact
        # existence.
        "xgb_meta_confirmation_enabled": False,
        "xgb_meta_calibration_path": None,
        # Meta-confidence HARD gate (Workstream D harness, 2026-09-27 --
        # see docs/pattern_recognition_plan.md's Workstream D and
        # scripts/validate_pattern_ml_workstream_d.py). None (default) is a
        # full no-op. Unlike calibrated_probability's existing role (an
        # input to TraderAgent._signal_calibrated_edge's Kelly sizing
        # only -- a down-weight, never a drop), a float here is a genuine
        # inclusion/exclusion threshold on the SEPARATE meta model's own raw
        # P(act) (xgb_meta_p_act, not calibrated_probability -- independent
        # of whether a calibration sidecar happens to be loaded): matches
        # scored below it are dropped from this cycle's emitted signals
        # entirely. This is the lever the plan's "finding a profitable
        # subset of a zero-mean population is exactly a classifier's job"
        # question needs -- down-weighting a losing subset can still leave
        # it net-negative; dropping it is the only way to test whether
        # trading ONLY the high-confidence subset is itself profitable.
        # Only ever applies when xgb_meta_confirmation_enabled scored this
        # match (a signal with no xgb_meta_p_act at all -- meta model off,
        # or unavailable this cycle -- is never gated by this knob).
        "xgb_meta_min_confidence": None,
        # Optional calibration sidecars (paths previously written by
        # firm.patterns.ml.calibration.save_calibration -- a JSON dict with
        # a "type" discriminator: "temperature" for the CNN's pre-softmax
        # logits, "sigmoid" for XGBoost's raw target-hit probability). None
        # (default) means "proceed uncalibrated" for that model -- see
        # firm.patterns.ml.calibration's module docstring for why each model
        # family needs a different calibration technique.
        "cnn_calibration_path": None,
        "xgb_calibration_path": None,
        # Statistical-significance test against a matched-volatility null
        # (Part A of the 2026-09-27 false-positive-rate fix -- see
        # firm.patterns.significance's module docstring). Off by default,
        # same convention as every other new knob here: needs its own
        # walk-forward re-validation before being trusted live, and is
        # computationally heavier per-symbol than the rest of the scanner
        # (though memoized per distinct scanned window -- see
        # significance.cached_null_score_distribution).
        "significance_test_enabled": False,
        "significance_n_draws": 200,
        "significance_max_p_value": 0.05,
        "significance_seed": 42,
        # Per-pattern minimum-sample confidence discount (Part A -- see
        # firm.patterns.sample_size's module docstring). Off by default,
        # same convention. None (default path) resolves to the shared
        # model-artifact directory, same repo-level-asset convention as
        # DEFAULT_MODEL_PATH in firm.patterns.ml.inference/xgb_inference
        # (both live instances read the same trained-model directory).
        "sample_size_discount_enabled": False,
        "min_reliable_samples": sample_size.DEFAULT_MIN_RELIABLE_SAMPLES,
        "sample_counts_path": None,
        # Market-regime-aware confidence discount (Part A item 6 -- see
        # firm.regime.detector.MarketRegimeDetector / firm.regime.model.
        # RegimeState). Already used by RiskAgent's regime_overlay and by
        # agents/research/_regime_weights.py's per-strategy score
        # multiplier, but neither reaches into a strategy's own
        # confidence -- strategies never receive ctx.market_regime (see
        # every analyst call site: technical.py/sentiment.py/fundamental.py
        # all call strat.generate(pit_view), never passing ctx), so this
        # strategy builds its own detector instance the same way
        # RiskAgent._detect_regime does. Off by default, same convention
        # as every other knob here; needs its own walk-forward
        # re-validation before flipping true in config/live.yaml.
        "regime_discount_enabled": False,
        "regime_n_states": 3,
        "regime_lookback_days": 504,
        "regime_retrain_frequency": 21,
        "regime_benchmark_symbol": None,
        # Chop: a universal (direction-agnostic) discount -- breakout-style
        # patterns (most of the 9 detector families here) are prone to
        # whipsaw in range-bound/mean-reverting markets.
        "regime_chop_discount": 0.85,
        # Bull/Bear: applied only when the pattern's own direction
        # conflicts with the labelled regime (e.g. a long pattern
        # confirming in a Bear regime) -- an aligned pattern (long-in-Bull,
        # short-in-Bear) is left at full confidence, matching the intuition
        # that a pattern fighting the prevailing trend is weaker evidence
        # than one riding it.
        "regime_misaligned_discount": 0.7,
        # Bull/Bear discounts are damped toward 1.0 (no-op) when
        # RegimeState.separation shows the label was assigned by a thin,
        # noise-level margin -- same idiom as firm.strategies.regime_hmm's
        # own separation-based damping (min_state_separation /
        # separation_damping_floor there). Chop's separation is always inf
        # by convention (see RegimeState's own docstring), so its discount
        # is never damped.
        "regime_min_separation": 0.5,
        "regime_separation_damping_floor": 0.15,
    }

    def __init__(self, params: dict | None = None):
        super().__init__("pattern_recognition", params)
        # Lazily constructed on first use with a regime-discount-enabled
        # generate() call -- same pattern as RiskAgent._regime_detector
        # (retrain cadence lives on the detector instance itself, so it
        # must persist across generate() calls, not be rebuilt each cycle).
        self._regime_detector = None

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
        xgb_meta_enabled = bool(p["xgb_meta_confirmation_enabled"])
        xgb_meta_min_confidence = p["xgb_meta_min_confidence"]
        xgb_meta_min_confidence = (
            float(xgb_meta_min_confidence) if xgb_meta_min_confidence is not None else None
        )

        significance_enabled = bool(p["significance_test_enabled"])
        significance_n_draws = int(p["significance_n_draws"])
        significance_max_p_value = float(p["significance_max_p_value"])
        significance_seed = int(p["significance_seed"])

        sample_discount_enabled = bool(p["sample_size_discount_enabled"])
        min_reliable_samples = int(p["min_reliable_samples"])
        sample_counts_path = p["sample_counts_path"] or DEFAULT_SAMPLE_COUNTS_PATH
        # Loaded once per generate() call (cheap: a single small JSON file),
        # not per-symbol -- fail-soft to None (no discount) if missing/corrupt,
        # same convention as firm.patterns.ml.calibration.load_calibration.
        sample_counts = sample_size.load_sample_counts(sample_counts_path) if sample_discount_enabled else None

        regime_discount_enabled = bool(p["regime_discount_enabled"])
        regime_chop_discount = float(p["regime_chop_discount"])
        regime_misaligned_discount = float(p["regime_misaligned_discount"])
        regime_min_separation = float(p["regime_min_separation"])
        regime_separation_damping_floor = float(p["regime_separation_damping_floor"])
        # Detected once per generate() call (a single market-wide read, not
        # per-symbol) -- same cadence as RiskAgent._detect_regime. Fail-soft
        # to None (no discount applied below) on missing history, missing
        # hmmlearn, or any fit/decode failure -- never a hard failure.
        regime_state = None
        if regime_discount_enabled:
            if self._regime_detector is None:
                from firm.regime.detector import MarketRegimeDetector

                self._regime_detector = MarketRegimeDetector(
                    n_states=int(p["regime_n_states"]),
                    lookback_days=int(p["regime_lookback_days"]),
                    retrain_frequency=int(p["regime_retrain_frequency"]),
                    benchmark_symbol=p["regime_benchmark_symbol"],
                )
            try:
                regime_state = self._regime_detector.detect(pit_view)
            except Exception:
                log.debug("pattern_recognition: regime detection failed", exc_info=True)
                regime_state = None

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
        xgb_meta_available = xgb_meta_enabled and xgb_inference.is_meta_available()
        # Either model consumes the SAME build_features output (only the
        # model artifact/training target differs -- see
        # score_pattern_meta_confirmation's own docstring), so features are
        # built whenever either is on.
        xgb_features_needed = xgb_available or xgb_meta_available

        # Market-proxy close series, date-keyed, built ONCE per generate()
        # call (Part B item 6, 2026-09-27) -- only when xgb_features_needed,
        # since this feeds build_features' market_ohlcv argument, which
        # only matters when features actually get built below; skipping it
        # otherwise avoids the pivot/mean cost on the (default) path where
        # both XGBoost knobs are off. Equal-weight average of adj_close
        # across the whole scanned universe, same convention as
        # firm.regime.detector.MarketRegimeDetector._market_proxy's own
        # universe-average fallback (no single benchmark_symbol config
        # here -- this strategy has no equivalent knob, and an equal-weight
        # proxy needs no extra config to already be point-in-time correct).
        market_proxy_by_date = None
        if xgb_features_needed:
            try:
                market_proxy_by_date = (
                    prices_df.pivot_table(index="date", columns="symbol", values="adj_close")
                    .mean(axis=1)
                    .sort_index()
                )
            except Exception:
                log.debug("pattern_recognition: market-proxy construction failed", exc_info=True)
                market_proxy_by_date = None

        cnn_calibration = None
        cnn_temperature = 1.0
        if p["cnn_calibration_path"]:
            cnn_calibration = ml_calibration.load_calibration(p["cnn_calibration_path"])
            # "model" discriminator (2026-09-27 fix): "type" alone doesn't
            # prove this file was actually fit against the CNN's own
            # score distribution -- an XGBoost-fit sigmoid calibration
            # could coincidentally never collide on "type" today (CNN
            # uses "temperature", XGBoost uses "sigmoid"), but checking
            # both is the honest guard against a future mismatch, and
            # documents the intent explicitly rather than relying on the
            # two types never colliding by chance. See
            # scripts/fit_pattern_calibration.py's module docstring for
            # the real mismatch this closes (it used to fit on
            # quality_score and label the file for XGBoost's use).
            if cnn_calibration and cnn_calibration.get("type") == "temperature" and cnn_calibration.get("model") == "cnn":
                cnn_temperature = float(cnn_calibration.get("temperature", 1.0))
            elif cnn_calibration:
                log.debug(
                    "pattern_recognition: cnn_calibration_path=%s has type=%r model=%r, "
                    "expected type='temperature' model='cnn' -- ignoring",
                    p["cnn_calibration_path"], cnn_calibration.get("type"), cnn_calibration.get("model"),
                )
                cnn_calibration = None

        xgb_calibration = None
        if p["xgb_calibration_path"]:
            loaded = ml_calibration.load_calibration(p["xgb_calibration_path"])
            if loaded and loaded.get("type") == "sigmoid" and loaded.get("model") == "xgboost":
                xgb_calibration = loaded
            elif loaded:
                log.debug(
                    "pattern_recognition: xgb_calibration_path=%s has type=%r model=%r, "
                    "expected type='sigmoid' model='xgboost' -- ignoring",
                    p["xgb_calibration_path"], loaded.get("type"), loaded.get("model"),
                )

        # "xgboost_meta" (2026-09-27, Part B item 2) -- deliberately its own
        # discriminator value, distinct from "xgboost" above: a calibration
        # curve fit against the 3-class model's p_target distribution is
        # NOT valid for the meta model's own p_act distribution even though
        # both models are "xgboost" -- exactly the class of mismatch the
        # "model" discriminator itself exists to catch (see
        # scripts/fit_pattern_calibration.py's module docstring for the
        # original CNN/XGBoost version of this same bug).
        xgb_meta_calibration = None
        if p["xgb_meta_calibration_path"]:
            loaded = ml_calibration.load_calibration(p["xgb_meta_calibration_path"])
            if loaded and loaded.get("type") == "sigmoid" and loaded.get("model") == "xgboost_meta":
                xgb_meta_calibration = loaded
            elif loaded:
                log.debug(
                    "pattern_recognition: xgb_meta_calibration_path=%s has type=%r model=%r, "
                    "expected type='sigmoid' model='xgboost_meta' -- ignoring",
                    p["xgb_meta_calibration_path"], loaded.get("type"), loaded.get("model"),
                )

        log.info(
            "pattern_recognition: quality scoring mode=%s, xgb_confirmation=%s, "
            "xgb_meta_confirmation=%s, zigzag=%s, retest_modifier=%s, confluence_modifier=%s, "
            "significance_test=%s, regime_discount=%s",
            "cnn" if cnn_available else "rule_based",
            "on" if xgb_available else "off",
            "on" if xgb_meta_available else "off",
            f"atr(mult={zigzag_atr_mult})" if zigzag_atr_mult else f"fixed(pct={zigzag_pct})",
            "on" if retest_enabled else "off",
            "on" if confluence_enabled else "off",
            "on" if significance_enabled else "off",
            f"on(regime={regime_state.label})" if regime_discount_enabled and regime_state else (
                "on(unavailable)" if regime_discount_enabled else "off"
            ),
        )

        signals: list[Signal] = []
        # Parallel to `signals` (same index) -- the significance p-value
        # each signal was computed with, or None (not computed / not
        # applicable). Used for the post-loop FDR pass below.
        signal_p_values: list[float | None] = []
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

            # Significance test (2026-09-27 Part A): compute the p-value
            # here, but do NOT reject inline against a raw per-symbol
            # threshold -- that would be exactly the uncorrected,
            # across-symbol multiple-comparisons problem
            # benjamini_hochberg_accept exists to fix. The actual
            # accept/reject decision happens once, after this loop, across
            # every candidate this cycle produced (see below). A p_value
            # of None (couldn't be computed -- e.g. too little history)
            # is a pass-through, same fail-soft convention as the CNN/
            # XGBoost layers: never held against the candidate.
            p_value = None
            if significance_enabled:
                try:
                    null_scores = significance.cached_null_score_distribution(
                        ohlcv["high"].to_numpy(dtype=float),
                        ohlcv["low"].to_numpy(dtype=float),
                        ohlcv["close"].to_numpy(dtype=float),
                        ohlcv["volume"].to_numpy(dtype=float),
                        n_draws=significance_n_draws,
                        zigzag_pct=zigzag_pct,
                        seed=significance_seed,
                    )
                    p_value = significance.pattern_p_value(best.quality_score, null_scores)
                except Exception:
                    log.debug("significance test failed for %s", symbol, exc_info=True)

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
            xgb_meta_p_act = None
            if xgb_features_needed:
                # Market-proxy window aligned to THIS symbol's own dates
                # (Part B item 6, 2026-09-27) -- reindexing the shared,
                # once-per-cycle date-keyed proxy onto sym_df["date"]
                # rather than assuming every symbol shares one common
                # length/date range (a newly-listed symbol, or one with a
                # gap, would otherwise silently misalign). None (fail-soft)
                # if any date in this symbol's own window isn't present in
                # the proxy -- build_features' own market_ohlcv contract
                # requires exact positional/date alignment, never a
                # best-effort partial one.
                market_ohlcv = None
                if market_proxy_by_date is not None:
                    try:
                        aligned = market_proxy_by_date.reindex(sym_df["date"].to_numpy())
                        if not aligned.isna().any():
                            market_ohlcv = pd.DataFrame({"close": aligned.to_numpy()})
                    except Exception:
                        log.debug(
                            "pattern_recognition: market-proxy alignment failed for %s",
                            symbol, exc_info=True,
                        )
                        market_ohlcv = None

                # Same feature vector feeds both models below -- only the
                # model artifact/training target differs (see
                # score_pattern_meta_confirmation's own docstring).
                features = build_features(best, ohlcv, market_ohlcv=market_ohlcv)
                feature_vec = np.fromiter(features.values(), dtype=np.float32, count=len(features))

                if xgb_available:
                    xgb_result = xgb_inference.score_pattern_confirmation(
                        feature_vec, apply_calibration=xgb_calibration,
                    )
                    if xgb_result is not None:
                        _, _, xgb_p_target = xgb_result

                if xgb_meta_available:
                    # The meta model IS the calibrated confidence gate by
                    # construction (de Prado's recipe -- see
                    # default_params' own docstring comment for this knob),
                    # so it takes priority over the 3-class model's p_target
                    # for meta["calibrated_probability"] whenever it's on
                    # (Part B item 2, 2026-09-27 -- closes the conflation
                    # this strategy previously had: sourcing that field from
                    # a model trained to answer a different question).
                    xgb_meta_result = xgb_inference.score_pattern_meta_confirmation(
                        feature_vec, apply_calibration=xgb_meta_calibration,
                    )
                    if xgb_meta_result is not None:
                        xgb_meta_p_act = xgb_meta_result
                        # Same "only trust as calibrated once a real
                        # calibration file was fit for THIS model" caution
                        # as the 3-class path above -- XGBoost's raw
                        # probability output is not calibrated by
                        # construction just because the target it was
                        # trained on happens to match calibrated_probability's
                        # semantics.
                        if xgb_meta_calibration and xgb_meta_calibration.get("type") == "sigmoid":
                            calibrated_probability = xgb_meta_p_act

                if calibrated_probability is None and xgb_p_target is not None:
                    # Only populate meta["calibrated_probability"] -- the
                    # generic TraderAgent._signal_calibrated_edge Kelly
                    # convention -- when xgb_calibration was actually loaded
                    # and is of the type score_pattern_confirmation applies
                    # ("sigmoid"); otherwise xgb_p_target is the RAW,
                    # uncalibrated model output and mislabeling it as
                    # calibrated would silently feed an uncalibrated number
                    # into Kelly sizing if allocation_method is ever set to
                    # "kelly". The raw value is still exposed below under
                    # its own honest key (xgb_p_target), never hidden. Only
                    # reached when the meta model above didn't already
                    # populate this field -- see the priority comment there.
                    if xgb_calibration and xgb_calibration.get("type") == "sigmoid":
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

            sample_size_discount = 1.0
            if sample_discount_enabled:
                sample_size_discount = sample_size.confidence_discount(
                    best.pattern, sample_counts, min_reliable_samples=min_reliable_samples,
                )
                quality_fraction = float(np.clip(quality_fraction * sample_size_discount, 0.0, 1.0))

            regime_discount = 1.0
            regime_label = None
            if regime_discount_enabled and regime_state is not None:
                regime_label = regime_state.label
                regime_discount = _regime_confidence_discount(
                    regime_state, best.direction,
                    chop_discount=regime_chop_discount,
                    misaligned_discount=regime_misaligned_discount,
                    min_separation=regime_min_separation,
                    separation_damping_floor=regime_separation_damping_floor,
                )
                quality_fraction = float(np.clip(quality_fraction * regime_discount, 0.0, 1.0))

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
                        # Raw P(act) from the SEPARATE meta-labeling model
                        # (Part B item 2, 2026-09-27) -- None whenever
                        # xgb_meta_confirmation_enabled is off or the match
                        # wasn't scored, never a placeholder. Distinct from
                        # xgb_p_target above (a different model, a
                        # different question -- see default_params' own
                        # docstring comment for this knob).
                        "xgb_meta_p_act": xgb_meta_p_act,
                        # Meta-labeling convention consumed by
                        # TraderAgent._signal_calibrated_edge -- only
                        # populated when a real calibration file was fit
                        # for whichever model actually scored this match
                        # (never a placeholder value). Prefers the meta
                        # model over the 3-class model's own p_target when
                        # both are available -- see the priority comment
                        # above where this is computed.
                        "calibrated_probability": calibrated_probability,
                        "scoring_mode": scoring_mode,
                        "score_breakdown": best.score_breakdown,
                        "volume_ratio": best.volume_ratio,
                        "duration_bars": best.duration_bars,
                        "bars_since_confirm": len(ohlcv) - 1 - best.confirm_index,
                        "other_patterns": [m.pattern for m in matches[1:4]],
                        # None whenever significance_test_enabled is False
                        # (the default) -- never a placeholder value, same
                        # convention as calibrated_probability above.
                        "significance_p_value": p_value,
                        # 1.0 (no-op) whenever sample_size_discount_enabled
                        # is False (the default) -- never a placeholder.
                        "sample_size_discount": sample_size_discount,
                        # 1.0/None (no-op) whenever regime_discount_enabled
                        # is False, or regime detection was unavailable
                        # this cycle -- never a placeholder value.
                        "regime_discount": regime_discount,
                        "regime_label": regime_label,
                    },
                )
            )
            signal_p_values.append(p_value)

        # Post-loop False Discovery Rate control (2026-09-27 Part A): the
        # cross-sectional multiple-comparisons correction across every
        # candidate THIS cycle produced, not a per-symbol threshold (see
        # the in-loop comment above and firm.patterns.significance.
        # benjamini_hochberg_accept's own docstring for why the two are
        # different problems). A signal with no computed p_value (None)
        # passes through untouched.
        if significance_enabled and signals:
            has_p = np.array([pv is not None for pv in signal_p_values])
            if has_p.any():
                p_arr = np.array([pv if pv is not None else 1.0 for pv in signal_p_values])
                accept = np.ones(len(signals), dtype=bool)
                accept[has_p] = significance.benjamini_hochberg_accept(
                    p_arr[has_p], q=significance_max_p_value,
                )
                n_rejected = int(np.sum(~accept))
                if n_rejected:
                    log.debug(
                        "pattern_recognition: FDR control (q=%.3f) rejected %d/%d "
                        "significance-tested signal(s) this cycle",
                        significance_max_p_value, n_rejected, int(has_p.sum()),
                    )
                signals = [s for s, keep in zip(signals, accept) if keep]

        # Meta-confidence hard gate (Workstream D harness, 2026-09-27 -- see
        # xgb_meta_min_confidence's own docstring above for why this is a
        # DROP, not a down-weight, and gates on xgb_meta_p_act specifically).
        # None (default) is a full no-op -- every existing caller/test is
        # unaffected.
        n_meta_gated = 0
        if xgb_meta_min_confidence is not None and signals:
            kept = []
            for s in signals:
                conf = s.meta.get("xgb_meta_p_act")
                if conf is not None and conf < xgb_meta_min_confidence:
                    n_meta_gated += 1
                    continue
                kept.append(s)
            signals = kept
            if n_meta_gated:
                log.debug(
                    "pattern_recognition: meta-confidence gate (>=%.3f) dropped "
                    "%d signal(s) this cycle",
                    xgb_meta_min_confidence, n_meta_gated,
                )

        log.info(
            "pattern_recognition: %d symbols scanned, %d signals (%d CNN-scored, "
            "%d XGB-ensembled, %d rule-based-only)",
            len(universe), len(signals), cnn_scored, xgb_scored, rule_based_only,
        )
        return signals
