#!/usr/bin/env python
"""Train the CNN/GAF chart-pattern image validator end to end (Phase 4b,
docs/pattern_recognition_plan.md §4b).

**2026-09-26 update:** ``torch`` now ships a Python 3.14 wheel (confirmed
present in the main ``.venv``, version 2.14.0) -- the "no 3.14 wheel"
claim below is stale (this repo's original ``.venv-ml`` also did not
survive a later host migration). ``pyts`` still has no 3.14 wheel, but is
no longer required for the default configuration: ``encode_gaf`` now uses
a pure-numpy GASF replica (``firm.patterns.ml.cnn_validator._encode_gasf_numpy``,
verified bit-for-bit identical to real ``pyts`` output) whenever
``--window-bars == --image-size`` (both default to 32, the only
combination this script or the live inference path actually uses) --
``pyts`` is only still required if you deliberately set them to different
values. In practice: this script now runs directly under the **main
venv** for the default configuration; no isolated environment needed.

Pipeline: load an OHLCV panel (synthetic or cached) -> scan every symbol for
confirmed chart patterns, rolling the as-of cutoff backward through history
exactly like ``scripts/train_pattern_ml.py``'s ``build_dataset`` (same
rationale: a single whole-series scan only ever finds patterns confirmed at
the very end, with no subsequent price history left to label) -> for each
confirmed match, extract the OHLCV window leading up to its breakout and
GAF-encode it -> triple-barrier label it against its own subsequent price
path (the same ``firm.patterns.ml.labeling`` used by the XGBoost pipeline,
so the two models are directly comparable) -> train a small CNN -> report
train/test accuracy -> save the fitted model (+ optional ONNX export).

**Scope note:** this is a standalone research/training tool; it is not
itself wired into any live signal-generation path. Its *output* (an ONNX
export) is, though, via the light, torch-free
``firm.patterns.ml.inference`` module (``onnxruntime`` also now has a
working Python 3.14 wheel, confirmed installed) -- gated off in both live
configs (``cnn_scoring_enabled: false``) pending its own walk-forward
re-validation, not because the runtime can't load it.

Usage:
    # Fast, dependency-free smoke run (synthetic prices, seconds):
    python scripts/train_cnn_validator.py --data-source synthetic

    # Real run against this repo's actual cached historical data:
    python scripts/train_cnn_validator.py --data-source cache \\
        --symbols AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,JPM,V,JNJ \\
        --start 2018-01-01 --end 2025-01-01 \\
        --output data/models/pattern_cnn.pt --onnx data/models/pattern_cnn.onnx
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sklearn.metrics import accuracy_score  # noqa: E402

from firm.data.synthetic import DEFAULT_SYMBOLS, make_synthetic_prices  # noqa: E402
from firm.patterns.ml import cnn_validator  # noqa: E402
from firm.patterns.ml.cnn_validator import DEFAULT_IMAGE_SIZE, DEFAULT_WINDOW_BARS, encode_gaf, extract_window  # noqa: E402
from firm.patterns.ml.labeling import label_triple_barrier  # noqa: E402
from firm.patterns.scanner import scan_symbol  # noqa: E402

# Reuses the strategy's own split/dividend adjustment -- see
# scripts/train_pattern_ml.py's identical import for the full rationale.
# firm/strategies/pattern_recognition.py is read-only for this initiative;
# importing its helper is not modifying it.
from firm.strategies.pattern_recognition import _adjusted_ohlc  # noqa: E402

log = logging.getLogger(__name__)

# Mirrors scripts/train_pattern_ml.py's own hardcoded default universe.
_DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "AMD",
    "CRM", "NFLX", "ADBE", "JPM", "GS", "BAC", "V", "MA", "JNJ", "UNH",
    "LLY", "XOM", "CVX", "SPY", "QQQ", "IWM",
]


def _load_ohlcv_panel(args: argparse.Namespace) -> pd.DataFrame:
    """Same shape/contract as train_pattern_ml.py's helper of the same name
    -- duplicated rather than cross-imported since these are independent
    CLI entry points, not a shared library.
    """
    if args.data_source == "synthetic":
        symbols = args.symbols or list(DEFAULT_SYMBOLS)
        log.info("loading synthetic prices: symbols=%s n_days=%d end=%s", symbols, args.n_days, args.end or "2023-12-31")
        return make_synthetic_prices(symbols, n_days=args.n_days, end_date=args.end or "2023-12-31")

    from firm.config import get_settings
    from firm.runtime import load_prices

    symbols = args.symbols or list(_DEFAULT_UNIVERSE)
    settings = get_settings()
    log.info("loading cached prices from %s", settings.data.cache_dir)
    panel = load_prices(settings)

    if "symbol" in panel.columns:
        panel = panel[panel["symbol"].isin(symbols)]
    if "date" in panel.columns:
        panel = panel.copy()
        panel["date"] = pd.to_datetime(panel["date"])
        if args.start:
            panel = panel[panel["date"] >= pd.Timestamp(args.start)]
        if args.end:
            panel = panel[panel["date"] <= pd.Timestamp(args.end)]
    if "adj_close" not in panel.columns:
        log.warning("cached panel has no adj_close column -- treating close as already-adjusted")
        panel = panel.copy()
        panel["adj_close"] = panel["close"]
    return panel


def build_image_dataset(
    panel: pd.DataFrame,
    *,
    zigzag_pct: float,
    min_score: float,
    confirm_lookback_bars: int,
    stop_atr_floor: float,
    timeout_bars: int,
    window_bars: int,
    image_size: int,
    min_window_bars: int = 60,
    step_bars: int = 10,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Same rolling as-of scan as train_pattern_ml.py's ``build_dataset``
    (see that function's docstring for why rolling, not a single
    whole-series scan) but extracting a GAF-encoded price-window image per
    confirmed match instead of a flat feature vector.

    Returns ``(X_images, y, meta)`` -- ``X_images`` is ``(n, image_size,
    image_size)``.
    """
    images: list[np.ndarray] = []
    labels: list[int] = []
    meta_rows: list[dict] = []

    if panel.empty or "symbol" not in panel.columns:
        return np.empty((0, image_size, image_size)), np.array([], dtype=int), pd.DataFrame()

    for symbol, sym_df in panel.groupby("symbol"):
        sym_df = sym_df.sort_values("date").reset_index(drop=True)
        try:
            ohlcv = _adjusted_ohlc(sym_df)
        except Exception:
            log.exception("build_image_dataset: failed to build adjusted OHLC for %s", symbol)
            continue

        n = len(ohlcv)
        if n < min_window_bars:
            continue
        close_full = ohlcv["close"].to_numpy(dtype=float)
        seen_confirm_indices: set[int] = set()
        for cutoff in range(min_window_bars, n + 1, step_bars):
            window = ohlcv.iloc[:cutoff]
            try:
                matches = scan_symbol(
                    window, zigzag_pct=zigzag_pct, min_score=min_score,
                    confirm_lookback_bars=confirm_lookback_bars, stop_atr_floor=stop_atr_floor,
                )
            except Exception:
                log.exception("build_image_dataset: scan_symbol failed for %s at cutoff=%d", symbol, cutoff)
                continue

            for match in matches:
                if not match.confirmed or match.confirm_index in seen_confirm_indices:
                    continue
                seen_confirm_indices.add(match.confirm_index)
                price_window = extract_window(close_full, match.confirm_index, window_bars=window_bars)
                if price_window is None:
                    continue
                try:
                    label = label_triple_barrier(match, ohlcv, timeout_bars=timeout_bars)
                except ValueError:
                    log.exception("build_image_dataset: labeling failed for %s/%s", symbol, match.pattern)
                    continue
                images.append(encode_gaf(price_window, image_size=image_size))
                labels.append(label)
                meta_rows.append({
                    "symbol": symbol, "pattern": match.pattern, "direction": match.direction,
                    "confirm_index": match.confirm_index, "quality_score": match.quality_score,
                    # See train_pattern_ml.py's identical field/comment --
                    # used for a time-ordered (not random) train/test split.
                    "confirm_date": sym_df["date"].iloc[match.confirm_index],
                })

    X = np.array(images) if images else np.empty((0, image_size, image_size))
    y = np.array(labels, dtype=int)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def time_ordered_split(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    test_size: float,
    embargo_bars: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Trailing-time train/test split -- see
    ``scripts/train_pattern_ml.py``'s identical-in-spirit function for the
    full rationale (this dataset is built the same rolling-as-of way and is
    equally autocorrelated). Duplicated rather than imported: scripts/ in
    this repo don't share code via cross-imports (see e.g.
    tests/test_pattern_ml.py's own note on independent copies), and X here
    is an image ndarray, not a DataFrame, so the indexing differs slightly.
    """
    n = len(meta)
    if n == 0:
        empty_y = np.array([], dtype=y.dtype if len(y) else int)
        empty_x = X[0:0]
        return empty_x, empty_x, empty_y, empty_y

    order = np.argsort(pd.to_datetime(meta["confirm_date"]).to_numpy(), kind="stable")
    X_sorted = X[order]
    y_sorted = y[order]
    dates_sorted = pd.to_datetime(meta["confirm_date"]).to_numpy()[order]

    n_test = min(n - 1, max(1, int(round(n * test_size)))) if n > 1 else 0
    split_at = n - n_test

    train_end = split_at
    if embargo_bars > 0 and split_at > 0:
        cutoff = pd.Timestamp(dates_sorted[split_at])
        embargo_start = cutoff - pd.Timedelta(days=embargo_bars)
        train_dates = pd.to_datetime(dates_sorted[:split_at])
        train_end = int(np.sum(train_dates < embargo_start))

    return X_sorted[:train_end], X_sorted[split_at:], y_sorted[:train_end], y_sorted[split_at:]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-source", choices=["synthetic", "cache"], default="synthetic")
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--n-days", type=int, default=400)
    parser.add_argument("--zigzag-pct", type=float, default=0.03)
    parser.add_argument("--min-score", type=float, default=60.0)
    parser.add_argument("--confirm-lookback-bars", type=int, default=3)
    parser.add_argument("--stop-atr-floor", type=float, default=1.5)
    parser.add_argument("--timeout-bars", type=int, default=20)
    parser.add_argument("--window-bars", type=int, default=DEFAULT_WINDOW_BARS, help="Price-window length fed to GAF encoding")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--min-window-bars", type=int, default=60)
    parser.add_argument("--step-bars", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument(
        "--test-size", type=float, default=0.25,
        help="Fraction of rows (by count, after sorting by confirm_date) held out as a "
             "trailing time-ordered test set -- see time_ordered_split.",
    )
    parser.add_argument(
        "--embargo-bars", type=int, default=None,
        help="Calendar-day embargo dropped from the trailing edge of the train period, "
             "immediately before the test cutoff. Defaults to --timeout-bars.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="No longer used for the (now time-ordered, not random) train/test split -- "
             "still used to seed cnn_validator.train's own initialization.",
    )
    parser.add_argument("--output", default="data/models/pattern_cnn.pt")
    parser.add_argument("--onnx", default=None, help="Optional path to also export an ONNX version")
    args = parser.parse_args(argv)
    if args.symbols:
        args.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.embargo_bars is None:
        args.embargo_bars = args.timeout_bars
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)

    panel = _load_ohlcv_panel(args)
    if panel.empty:
        print("No OHLCV data loaded -- nothing to scan. Exiting.", file=sys.stderr)
        return 1
    print(f"Loaded {len(panel)} rows across {panel['symbol'].nunique()} symbols (source={args.data_source})")

    X, y, meta = build_image_dataset(
        panel, zigzag_pct=args.zigzag_pct, min_score=args.min_score,
        confirm_lookback_bars=args.confirm_lookback_bars, stop_atr_floor=args.stop_atr_floor,
        timeout_bars=args.timeout_bars, window_bars=args.window_bars, image_size=args.image_size,
        min_window_bars=args.min_window_bars, step_bars=args.step_bars,
    )
    n_distinct = len(set(y.tolist()))
    if len(X) == 0 or n_distinct < 2:
        print(
            f"Only {len(X)} confirmed-pattern image(s) with {n_distinct} distinct label(s) -- "
            "not enough to train/evaluate. Try a wider universe, longer date range, or a "
            "lower --min-score.", file=sys.stderr,
        )
        return 1

    label_counts = pd.Series(y).value_counts().sort_index()
    print(f"Dataset: {len(X)} confirmed-pattern images ({args.image_size}x{args.image_size}).")
    print(f"Label counts (target=+1 / timeout=0 / stop=-1): {label_counts.to_dict()}")
    print("Matches by pattern:")
    print(meta["pattern"].value_counts().to_string())

    X_train, X_test, y_train, y_test = time_ordered_split(
        X, y, meta, test_size=args.test_size, embargo_bars=args.embargo_bars,
    )
    if len(set(y_train.tolist())) < 2 or len(X_train) == 0:
        print(
            f"Time-ordered split left only {len(X_train)} train row(s) with "
            f"{len(set(y_train.tolist()))} distinct label(s) -- try a wider universe, "
            "longer date range, smaller --test-size, or smaller --embargo-bars.",
            file=sys.stderr,
        )
        return 1

    try:
        model = cnn_validator.train(X_train, y_train, image_size=args.image_size, epochs=args.epochs, seed=args.seed)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    train_pred = cnn_validator.predict_label(model, X_train)
    test_pred = cnn_validator.predict_label(model, X_test)
    print(f"Train accuracy: {accuracy_score(y_train, train_pred):.3f} (n={len(y_train)})")
    print(f"Test accuracy:  {accuracy_score(y_test, test_pred):.3f} (n={len(y_test)})")

    cnn_validator.save(model, args.output)
    print(f"Model saved to {args.output}")
    if args.onnx:
        cnn_validator.export_onnx(model, args.onnx)
        print(f"ONNX model exported to {args.onnx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
