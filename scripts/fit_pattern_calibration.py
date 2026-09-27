#!/usr/bin/env python
"""Fit and persist a real Platt/sigmoid calibration for ONE of
pattern_recognition's models (rule-based quality_score, or the XGBoost
ensemble's raw p_target), from resolved pattern-scan outcomes.

Closes the gap this operationalizes: ``firm.patterns.ml.calibration.fit_sigmoid_calibration``/
``fit_temperature`` have always been correctly implemented but had zero
callers anywhere in the repo -- ``xgb_calibration_path``/``cnn_calibration_path``
were never set in live config because nothing ever produced a calibration
file to point them at. ``firm.strategies.pattern_recognition`` now refuses to
label its XGBoost score ``meta["calibrated_probability"]`` (the field
``TraderAgent._signal_calibrated_edge``'s Kelly path trusts at face value)
unless a real calibration file of the right type is configured -- this
script is how one gets produced.

**Bug fixed 2026-09-27**: this script previously fit its sigmoid curve on
``quality_score`` (the rule-based scanner's own score) unconditionally, but
the earlier version's own examples pointed ``--output`` at
``pattern_xgb.calibration.json`` -- i.e. the resulting file was meant to be
loaded as *XGBoost's* calibration (``xgb_calibration_path``), even though
it was never actually fit against XGBoost's ``p_target``. Both scores are
nominally in ``[0, 1]``, but a Platt fit's ``(a, b)`` parameters are
specific to the score distribution they were fit on -- applying
quality_score's fit to p_target's differently-shaped distribution is not a
valid calibration, just two numbers that happen to type-check. Fixed by
requiring an explicit ``--model {rule_based, xgboost}`` and only ever
calibrating that model's own real output; the saved file now carries a
``"model"`` discriminator (alongside the existing ``"type"``) that
``pattern_recognition.py`` checks at load time and refuses if mismatched.

Only "sigmoid" (Platt) calibration is supported here. "temperature"
calibration (for the CNN's pre-softmax 3-class logits) needs a full
per-class logits matrix that nothing here logs -- use
``scripts/train_cnn_validator.py``'s own held-out split instead.

Two outcome sources, since ``data/pattern_scan_history.db`` has zero
resolved rows as of this writing (every row is still ``outcome IS NULL`` --
see ``scripts/analyze_pattern_scan_outcomes.py``'s own docstring):

- ``--source history`` (default): reads real resolved rows from
  ``PatternScanHistoryStore``. Only supports ``--model rule_based`` --
  history rows store summary stats (``quality_score``, etc.), not the raw
  OHLCV window a match was detected from, so XGBoost's 47 features cannot
  be recomputed from a history row; ``--model xgboost --source history``
  refuses with a clear error rather than guessing. Refuses to write a
  calibration file below ``_MIN_RESOLVED_FOR_STATS`` resolved rows (same
  bar as ``analyze_pattern_scan_outcomes.py``), printing a report instead
  so this degrades gracefully on today's near-empty DB rather than
  pretending to calibrate off noise.
- ``--source synthetic``: builds a labeled sample from synthetic OHLCV
  (``firm.data.synthetic.make_synthetic_prices``) via the real
  scan-then-triple-barrier-label pipeline
  (``firm.patterns.scanner.scan_symbol`` +
  ``firm.patterns.ml.labeling.label_triple_barrier``). Supports both
  models: ``--model rule_based`` uses ``quality_score`` directly;
  ``--model xgboost`` additionally runs the real trained XGBoost ONNX
  model (``firm.patterns.ml.xgb_inference.score_pattern_confirmation``)
  over each match's real engineered features
  (``firm.patterns.ml.feature_engineering.build_features``) to get a
  genuine (uncalibrated) ``p_target`` -- this is what actually gets
  calibrated, not a proxy. Not a substitute for calibrating against real
  outcomes -- treat any resulting calibration file as a smoke-test
  artifact, not a production one.

Examples:
    # Against the real on-disk history DB (will likely report
    # "not enough resolved rows" today).
    python scripts/fit_pattern_calibration.py --model rule_based

    # Exercise the full XGBoost calibration pipeline against synthetic data.
    python scripts/fit_pattern_calibration.py --model xgboost --source synthetic \\
        --output data/models/pattern_xgb.calibration.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.live.pattern_scan_history import PatternScanHistoryStore  # noqa: E402
from firm.patterns.ml.calibration import fit_sigmoid_calibration, save_calibration  # noqa: E402

log = logging.getLogger(__name__)

# Same bar as analyze_pattern_scan_outcomes.py's _MIN_RESOLVED_FOR_STATS --
# below this many decisive (target_hit/stop_hit) rows, a Platt fit is fit to
# noise, not signal.
_MIN_RESOLVED_FOR_CALIBRATION = 20

_OUTCOME_TO_LABEL = {"target_hit": 1.0, "stop_hit": 0.0}


_VALID_MODELS = ("rule_based", "xgboost")


def build_calibration(
    raw_scores: np.ndarray, labels_binary: np.ndarray, *, model: str,
) -> dict[str, Any] | None:
    """Pure: fit a sigmoid calibration for *model*'s own raw score, or
    ``None`` if there isn't enough decisive data to trust the fit.

    ``model`` (2026-09-27 fix) is saved as an explicit discriminator
    alongside ``"type"`` -- two different models' raw scores can both be
    sigmoid-calibrated (same ``"type"``) while being completely
    incompatible fits (different underlying score distributions), so
    ``"type"`` alone was never a sufficient guard against loading a
    calibration file fit on the wrong model's output.
    """
    if model not in _VALID_MODELS:
        raise ValueError(f"build_calibration: model must be one of {_VALID_MODELS}, got {model!r}")
    if len(raw_scores) < _MIN_RESOLVED_FOR_CALIBRATION:
        log.warning(
            "build_calibration: only %d decisive (target_hit/stop_hit) sample(s), "
            "need >= %d -- refusing to fit (would calibrate to noise)",
            len(raw_scores), _MIN_RESOLVED_FOR_CALIBRATION,
        )
        return None
    a, b = fit_sigmoid_calibration(raw_scores, labels_binary)
    return {
        "type": "sigmoid",
        "model": model,
        "a": a,
        "b": b,
        "n_samples": int(len(raw_scores)),
        "source": "fit_pattern_calibration.py",
    }


def _load_all_rows(store: PatternScanHistoryStore, *, page_size: int = 500) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = store.list_history(limit=page_size, offset=offset)
        if not page:
            break
        rows.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return rows


def _resolved_sample_from_history(store: PatternScanHistoryStore, *, score_field: str) -> tuple[np.ndarray, np.ndarray]:
    rows = _load_all_rows(store)
    scores: list[float] = []
    labels: list[float] = []
    for row in rows:
        outcome = row.get("outcome")
        score = row.get(score_field)
        if outcome not in _OUTCOME_TO_LABEL or score is None:
            continue
        scores.append(float(score) / 100.0 if score_field == "quality_score" else float(score))
        labels.append(_OUTCOME_TO_LABEL[outcome])
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def _iter_synthetic_confirmed_matches(*, n_symbols: int, seed: int):
    """Yield (symbol, match, ohlcv) for every confirmed match found while
    rolling a growing as-of cutoff through synthetic OHLCV -- shared by
    both the rule_based and xgboost synthetic samplers below so the two
    models are calibrated against the *same* underlying match set (a fair
    comparison), not two independently-seeded scans.

    Same rolling-cutoff technique as scripts/train_pattern_ml.py's
    build_dataset, for the same reason (see that function's docstring): a
    single whole-series scan_symbol call confirms almost every match at
    the last row, leaving no genuine subsequent bars to label against.
    """
    from firm.data.synthetic import make_synthetic_prices
    from firm.patterns.scanner import scan_symbol

    symbols = [f"SYN{i:02d}" for i in range(n_symbols)]
    panel = make_synthetic_prices(symbols=symbols, n_days=750, seed=seed)

    for symbol, sym_df in panel.groupby("symbol"):
        sym_df = sym_df.sort_values("date").reset_index(drop=True)
        # Synthetic data has no splits, so close == adj_close -- no need for
        # firm.strategies.pattern_recognition._adjusted_ohlc's scaling here.
        ohlcv = sym_df[["high", "low", "close", "volume"]]
        n = len(ohlcv)
        seen_confirm_indices: set[int] = set()
        for cutoff in range(min(60, n), n + 1, 10):
            window = ohlcv.iloc[:cutoff]
            try:
                matches = scan_symbol(window, min_score=0.0)
            except Exception:
                log.debug("scan_symbol failed for synthetic %s at cutoff=%d", symbol, cutoff, exc_info=True)
                continue
            for match in matches:
                if not match.confirmed or match.confirm_index in seen_confirm_indices:
                    continue
                seen_confirm_indices.add(match.confirm_index)
                yield symbol, match, ohlcv


def _resolved_sample_from_synthetic_rule_based(*, n_symbols: int, seed: int, timeout_bars: int) -> tuple[np.ndarray, np.ndarray]:
    from firm.patterns.ml.labeling import label_triple_barrier

    scores: list[float] = []
    labels: list[float] = []
    for _symbol, match, ohlcv in _iter_synthetic_confirmed_matches(n_symbols=n_symbols, seed=seed):
        try:
            # Label against the FULL series (real subsequent bars),
            # matching build_dataset's own feature/label split intent.
            label = label_triple_barrier(match, ohlcv, timeout_bars=timeout_bars)
        except ValueError:
            continue
        if label == 0:
            continue  # timeout -- not decisive, same exclusion as history path
        scores.append(match.quality_score / 100.0)
        labels.append(1.0 if label == 1 else 0.0)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def _resolved_sample_from_synthetic_xgboost(*, n_symbols: int, seed: int, timeout_bars: int) -> tuple[np.ndarray, np.ndarray]:
    """Same match set as the rule_based sampler, but scores each match with
    the REAL trained XGBoost ONNX model's raw (uncalibrated) p_target --
    computed from that match's real engineered features, not a proxy --
    since that is exactly the quantity xgb_calibration_path calibrates in
    production (firm.strategies.pattern_recognition.py).
    """
    from firm.patterns.ml import xgb_inference
    from firm.patterns.ml.feature_engineering import build_features
    from firm.patterns.ml.labeling import label_triple_barrier

    if not xgb_inference.is_available():
        raise RuntimeError(
            "fit_pattern_calibration: --model xgboost requires the trained XGBoost "
            "ONNX model + onnxruntime to be available (xgb_inference.is_available() "
            "returned False) -- train one first (scripts/train_pattern_ml.py + "
            "xgb_classifier.export_onnx)."
        )

    scores: list[float] = []
    labels: list[float] = []
    for _symbol, match, ohlcv in _iter_synthetic_confirmed_matches(n_symbols=n_symbols, seed=seed):
        try:
            label = label_triple_barrier(match, ohlcv, timeout_bars=timeout_bars)
        except ValueError:
            continue
        if label == 0:
            continue
        features = build_features(match, ohlcv)
        result = xgb_inference.score_pattern_confirmation(
            np.fromiter(features.values(), dtype=np.float32, count=len(features)),
        )
        if result is None:
            continue
        _, _, p_target = result
        scores.append(p_target)
        labels.append(1.0 if label == 1 else 0.0)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", choices=list(_VALID_MODELS), required=True,
        help="Which model's raw score to calibrate -- rule_based (quality_score) or "
        "xgboost (the trained ONNX model's real p_target). Required: this script used "
        "to silently assume rule_based's score would also serve as a stand-in for "
        "xgboost's, which is not a valid calibration (see module docstring).",
    )
    parser.add_argument("--source", choices=["history", "synthetic"], default="history")
    parser.add_argument(
        "--db-path", default="data/pattern_scan_history.db",
        help="Path to the pattern_scan_history SQLite DB (--source history only)",
    )
    parser.add_argument(
        "--score-field", default="quality_score",
        help="Raw-score column to calibrate (--source history --model rule_based only)",
    )
    parser.add_argument("--n-symbols", type=int, default=20, help="--source synthetic only")
    parser.add_argument("--seed", type=int, default=42, help="--source synthetic only")
    parser.add_argument(
        "--timeout-bars", type=int, default=40,
        help="--source synthetic only -- triple-barrier timeout window "
        "(default 40, wider than firm.patterns.ml.labeling's own 20-bar "
        "default: GBM-noise matches' target/stop distances rarely resolve "
        "within 20 bars, so the default 20 tends to label everything a "
        "timeout and starve this smoke test of decisive samples)",
    )
    parser.add_argument(
        "--calibration-type", choices=["sigmoid"], default="sigmoid",
        help="Only sigmoid/Platt is supported today -- see module docstring for why "
        "temperature scaling isn't (it needs per-class logits nothing currently logs).",
    )
    parser.add_argument("--output", default="data/models/pattern_xgb.calibration.json")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    if args.source == "history":
        if args.model == "xgboost":
            log.error(
                "--model xgboost --source history is not supported: history rows "
                "store summary stats (quality_score, entry/stop/target, ...), not "
                "the raw OHLCV window a match was detected from, so XGBoost's 47 "
                "features cannot be recomputed from a history row. Use "
                "--source synthetic, or extend pattern_scan_history's schema to log "
                "ML features/scores directly (tracked as a known gap, not solved here)."
            )
            return 1
        store = PatternScanHistoryStore(db_path=args.db_path)
        scores, labels = _resolved_sample_from_history(store, score_field=args.score_field)
        log.info("Loaded %d resolved (target_hit/stop_hit) row(s) from %s", len(scores), store.db_path)
    else:
        sampler = (
            _resolved_sample_from_synthetic_xgboost if args.model == "xgboost"
            else _resolved_sample_from_synthetic_rule_based
        )
        scores, labels = sampler(n_symbols=args.n_symbols, seed=args.seed, timeout_bars=args.timeout_bars)
        log.info(
            "Built %d decisive sample(s) from synthetic data (model=%s, seed=%d)",
            len(scores), args.model, args.seed,
        )

    calibration = build_calibration(scores, labels, model=args.model)
    if calibration is None:
        log.error(
            "Not enough decisive data to fit a calibration (model=%s, source=%s) -- no file written. "
            "%s", args.model, args.source,
            "Wait for more real resolved outcomes." if args.source == "history"
            else "Try a larger --n-symbols.",
        )
        return 1

    out_path = Path(args.output)
    save_calibration(calibration, out_path)
    log.info(
        "Fitted sigmoid calibration (model=%s) a=%.4f b=%.4f over %d samples -> %s",
        calibration["model"], calibration["a"], calibration["b"], calibration["n_samples"], out_path,
    )
    print(json.dumps(calibration, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
