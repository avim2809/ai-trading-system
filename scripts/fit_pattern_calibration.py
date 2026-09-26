#!/usr/bin/env python
"""Fit and persist a real Platt/sigmoid calibration for pattern_recognition's
XGBoost-ensemble raw score, from resolved pattern-scan outcomes.

Closes the gap this operationalizes: ``firm.patterns.ml.calibration.fit_sigmoid_calibration``/
``fit_temperature`` have always been correctly implemented but had zero
callers anywhere in the repo -- ``xgb_calibration_path``/``cnn_calibration_path``
were never set in live config because nothing ever produced a calibration
file to point them at. ``firm.strategies.pattern_recognition`` now refuses to
label its XGBoost score ``meta["calibrated_probability"]`` (the field
``TraderAgent._signal_calibrated_edge``'s Kelly path trusts at face value)
unless a real calibration file of the right type is configured -- this
script is how one gets produced.

Only "sigmoid" (Platt) calibration is supported here, over a single raw
score column (default ``quality_score``, the rule-based scanner's 0-100
score fed through as a fraction -- the same shape XGBoost's ``p_target``
output has). "temperature" calibration (for the CNN's pre-softmax 3-class
logits) needs a full per-class logits matrix, which
``pattern_scan_history``'s current schema does not log anywhere (the
scheduled scan job never runs the CNN/XGBoost ML layer at all -- see
``firm.live.pattern_scan_job``'s own docstring) -- fitting that properly
requires a schema change to log those, out of scope here; use
``scripts/train_cnn_validator.py``'s own held-out split instead until then.

Two outcome sources, since ``data/pattern_scan_history.db`` has zero
resolved rows as of this writing (every row is still ``outcome IS NULL`` --
see ``scripts/analyze_pattern_scan_outcomes.py``'s own docstring):

- ``--source history`` (default): reads real resolved rows from
  ``PatternScanHistoryStore``. Refuses to write a calibration file below
  ``_MIN_RESOLVED_FOR_STATS`` resolved rows (same bar as
  ``analyze_pattern_scan_outcomes.py``), printing a report instead so this
  degrades gracefully on today's near-empty DB rather than pretending to
  calibrate off noise.
- ``--source synthetic``: builds a labeled sample from synthetic OHLCV
  (``firm.data.synthetic.make_synthetic_prices``) via the real
  scan-then-triple-barrier-label pipeline
  (``firm.patterns.scanner.scan_symbol`` +
  ``firm.patterns.ml.labeling.label_triple_barrier``), so the wiring here
  can be exercised end to end before any real resolved outcomes exist. Not
  a substitute for calibrating against real outcomes -- treat any resulting
  calibration file as a smoke-test artifact, not a production one.

Examples:
    # Against the real on-disk history DB (will likely report
    # "not enough resolved rows" today).
    python scripts/fit_pattern_calibration.py

    # Exercise the full pipeline against synthetic data.
    python scripts/fit_pattern_calibration.py --source synthetic \\
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


def build_calibration(raw_scores: np.ndarray, labels_binary: np.ndarray) -> dict[str, Any] | None:
    """Pure: fit a sigmoid calibration, or ``None`` if there isn't enough
    decisive data to trust the fit.
    """
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


def _resolved_sample_from_synthetic(*, n_symbols: int, seed: int, timeout_bars: int) -> tuple[np.ndarray, np.ndarray]:
    from firm.data.synthetic import make_synthetic_prices
    from firm.patterns.ml.labeling import label_triple_barrier
    from firm.patterns.scanner import scan_symbol

    symbols = [f"SYN{i:02d}" for i in range(n_symbols)]
    panel = make_synthetic_prices(symbols=symbols, n_days=750, seed=seed)

    scores: list[float] = []
    labels: list[float] = []
    for symbol, sym_df in panel.groupby("symbol"):
        sym_df = sym_df.sort_values("date").reset_index(drop=True)
        # Synthetic data has no splits, so close == adj_close -- no need for
        # firm.strategies.pattern_recognition._adjusted_ohlc's scaling here.
        ohlcv = sym_df[["high", "low", "close", "volume"]]
        n = len(ohlcv)
        # A single whole-series scan_symbol call confirms almost every
        # match at (or within a few bars of) the *last* row -- see
        # confirmation.find_confirmation's newest-first search -- leaving
        # no genuine subsequent bars to label against (every label comes
        # back a trivial timeout). Roll the cutoff backward through
        # history instead, exactly like scripts/train_pattern_ml.py's
        # build_dataset does for the same reason, so each match has real
        # forward price action to resolve against.
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["history", "synthetic"], default="history")
    parser.add_argument(
        "--db-path", default="data/pattern_scan_history.db",
        help="Path to the pattern_scan_history SQLite DB (--source history only)",
    )
    parser.add_argument(
        "--score-field", default="quality_score",
        help="Raw-score column to calibrate (--source history only; default: quality_score)",
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
        store = PatternScanHistoryStore(db_path=args.db_path)
        scores, labels = _resolved_sample_from_history(store, score_field=args.score_field)
        log.info("Loaded %d resolved (target_hit/stop_hit) row(s) from %s", len(scores), store.db_path)
    else:
        scores, labels = _resolved_sample_from_synthetic(
            n_symbols=args.n_symbols, seed=args.seed, timeout_bars=args.timeout_bars,
        )
        log.info("Built %d decisive sample(s) from synthetic data (seed=%d)", len(scores), args.seed)

    calibration = build_calibration(scores, labels)
    if calibration is None:
        log.error(
            "Not enough decisive data to fit a calibration (source=%s) -- no file written. "
            "%s", args.source,
            "Wait for more real resolved outcomes." if args.source == "history"
            else "Try a larger --n-symbols.",
        )
        return 1

    out_path = Path(args.output)
    save_calibration(calibration, out_path)
    log.info(
        "Fitted sigmoid calibration a=%.4f b=%.4f over %d samples -> %s",
        calibration["a"], calibration["b"], calibration["n_samples"], out_path,
    )
    print(json.dumps(calibration, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
