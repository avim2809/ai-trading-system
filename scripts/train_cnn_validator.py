#!/usr/bin/env python
"""Train the CNN/GAF chart-pattern image validator end to end (Phase 4b,
docs/pattern_recognition_plan.md §4b).

**Run this under the isolated ML environment, not the main venv**::

    .venv-ml/bin/python scripts/train_cnn_validator.py --data-source synthetic

``torch``/``pyts`` have no Python 3.14 wheels yet (verified against PyPI
when ``.venv-ml`` was set up, see §4a) -- this script cannot even import
``firm.patterns.ml.cnn_validator`` under the main venv.

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

**Scope note:** this is a standalone research tool. It is not wired into
any live signal-generation path -- see docs/pattern_recognition_plan.md §4
for why (no Python 3.14 wheel for torch means the main firm-api process
can never import this module at all; consuming a trained model there would
need an ONNX export loaded via a runtime that also doesn't exist for 3.14
yet, tracked as a known follow-on constraint, not solved here).

Usage:
    # Fast, dependency-free smoke run (synthetic prices, seconds):
    .venv-ml/bin/python scripts/train_cnn_validator.py --data-source synthetic

    # Real run against this repo's actual cached historical data:
    .venv-ml/bin/python scripts/train_cnn_validator.py --data-source cache \\
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
from sklearn.model_selection import train_test_split  # noqa: E402

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
                })

    X = np.array(images) if images else np.empty((0, image_size, image_size))
    y = np.array(labels, dtype=int)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


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
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/models/pattern_cnn.pt")
    parser.add_argument("--onnx", default=None, help="Optional path to also export an ONNX version")
    args = parser.parse_args(argv)
    if args.symbols:
        args.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
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

    _, counts = np.unique(y, return_counts=True)
    stratify = y if counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=stratify,
    )

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
