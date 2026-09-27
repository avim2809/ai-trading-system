#!/usr/bin/env python
"""Train the XGBoost pattern-confirmation classifier end to end.

Phase 4 of docs/pattern_recognition_plan.md, deliberately scoped to just the
XGBoost confirmation classifier -- no CNN/GAF image validator, no PPO RL
position sizer (those need torch/stable-baselines3/gymnasium/pyts, out of
scope for this pass; see the task report for this initiative).

Pipeline: load an OHLCV panel (synthetic or cached) -> scan every symbol for
confirmed chart patterns (``firm.patterns.scanner.scan_symbol``) -> build one
feature row per confirmed match (``firm.patterns.ml.feature_engineering``) ->
triple-barrier label it against its own subsequent price path
(``firm.patterns.ml.labeling``) -> train an XGBoost classifier
(``firm.patterns.ml.xgb_classifier``) on a train/test split -> report
train/test accuracy and (where the split has enough class diversity) AUC ->
save the fitted model.

Requires the optional ``xgboost`` extra: ``pip install '.[patterns_ml]'``,
then ``pip uninstall -y nvidia-nccl-cu13`` -- xgboost's PyPI wheel pulls that
transitively on Linux even for pure-CPU use (see pyproject.toml's comment on
the ``patterns_ml`` extra); it's a ~290MB unused GPU library, safe to remove.

Usage:
    # Fast, dependency-free smoke run (synthetic GBM prices, seconds) -- a
    # handy manual sanity check of this exact path (tests/test_pattern_ml.py
    # exercises the same pipeline directly, not via this script):
    python scripts/train_pattern_ml.py --data-source synthetic

    # Real run -- a human deliberately kicking this off later, not part of
    # building this initiative. Reads whatever `scripts/fetch_data.py` has
    # already cached under data/cache; no network access of its own:
    python scripts/train_pattern_ml.py --data-source cache \\
        --symbols AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,JPM,V,JNJ \\
        --start 2018-01-01 --end 2025-01-01 \\
        --output data/models/pattern_xgb.pkl
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

from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402

from firm.data.synthetic import DEFAULT_SYMBOLS, make_synthetic_prices  # noqa: E402
from firm.patterns.ml import xgb_classifier  # noqa: E402
from firm.patterns.ml.feature_engineering import build_features  # noqa: E402
from firm.patterns.ml.labeling import label_triple_barrier  # noqa: E402
from firm.patterns.sample_size import DEFAULT_SAMPLE_COUNTS_FILENAME, save_sample_counts  # noqa: E402
from firm.patterns.scanner import scan_symbol  # noqa: E402

# Reuses the strategy's own split/dividend adjustment (scales raw high/low by
# adj_close/close so high>=close>=low holds across split boundaries) rather
# than duplicating that logic -- see firm/strategies/pattern_recognition.py's
# docstring. That module is read-only for this change (per
# docs/pattern_recognition_plan.md); importing its helper is not modifying
# it. If that ever changes, inline a local copy here instead.
from firm.strategies.pattern_recognition import _adjusted_ohlc  # noqa: E402

log = logging.getLogger(__name__)

# Mirrors scripts/calibrate_regime_ensemble.py's own hardcoded universe --
# config/settings.yaml's `universe:` block is an index/screen (SP500 +
# market-cap/volume filters), not a literal symbol list a script can import.
_DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "AMD",
    "CRM", "NFLX", "ADBE", "JPM", "GS", "BAC", "V", "MA", "JNJ", "UNH",
    "LLY", "XOM", "CVX", "SPY", "QQQ", "IWM",
]


def _load_ohlcv_panel(args: argparse.Namespace) -> pd.DataFrame:
    """Tidy ``(date, symbol, open, high, low, close, adj_close, volume)``
    panel per ``--data-source``. ``"cache"`` only ever reads local disk
    (``firm.runtime.load_prices``) -- never a network fetch of its own.
    """
    if args.data_source == "synthetic":
        symbols = args.symbols or list(DEFAULT_SYMBOLS)
        log.info(
            "loading synthetic prices: symbols=%s n_days=%d end=%s",
            symbols, args.n_days, args.end or "2023-12-31",
        )
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


def build_dataset(
    panel: pd.DataFrame,
    *,
    zigzag_pct: float,
    min_score: float,
    confirm_lookback_bars: int,
    stop_atr_floor: float,
    timeout_bars: int,
    min_window_bars: int = 60,
    step_bars: int = 10,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Scan every symbol in ``panel`` for confirmed patterns, then build one
    feature row + triple-barrier label per confirmed match.

    Walks each symbol's OHLCV forward in ``step_bars`` increments (a rolling
    "as of" scan, like a lightweight point-in-time backtest) instead of
    scanning the whole series once. This matters:
    ``firm.patterns.confirmation.find_confirmation``'s search is "most
    recent breakout within ``confirm_lookback_bars`` of the window's *last*
    bar" by construction (exactly what a live strategy wants -- "did
    something just confirm as of today") -- so a single whole-series scan
    only ever turns up patterns confirmed at (or within a couple of bars of)
    the very last row, which leaves virtually no genuine subsequent price
    history to triple-barrier-label against (verified empirically while
    building this script: every match came back with ``confirm_index`` in
    the last 1-2 bars of the input, and every label was therefore a
    trivial/degenerate ``0`` -- "no bars after confirm_index to look at").
    Rolling the cutoff backward through history instead surfaces patterns
    confirmed well before "today", each with real forward bars already
    present in the same series to label from. ``min_window_bars``/
    ``step_bars`` trade off dataset size against scan time -- a smaller
    ``step_bars`` finds more (highly autocorrelated, since the same pattern
    is typically still "the most recent one" across several consecutive
    cutoffs) rows per symbol at proportionally higher cost.

    Returns ``(X, y, meta)``; ``meta`` (symbol/pattern/confirm_index/
    quality_score per row) is diagnostic only, for the printed report -- not
    fed to the model.
    """
    feature_rows: list[dict] = []
    labels: list[int] = []
    meta_rows: list[dict] = []

    if panel.empty or "symbol" not in panel.columns:
        return pd.DataFrame(), np.array([], dtype=int), pd.DataFrame()

    for symbol, sym_df in panel.groupby("symbol"):
        sym_df = sym_df.sort_values("date").reset_index(drop=True)
        try:
            ohlcv = _adjusted_ohlc(sym_df)
        except Exception:
            log.exception("build_dataset: failed to build adjusted OHLC for %s", symbol)
            continue

        n = len(ohlcv)
        if n < min_window_bars:
            continue
        seen_confirm_indices: set[int] = set()
        for cutoff in range(min_window_bars, n + 1, step_bars):
            window = ohlcv.iloc[:cutoff]
            try:
                matches = scan_symbol(
                    window,
                    zigzag_pct=zigzag_pct,
                    min_score=min_score,
                    confirm_lookback_bars=confirm_lookback_bars,
                    stop_atr_floor=stop_atr_floor,
                )
            except Exception:
                log.exception("build_dataset: scan_symbol failed for %s at cutoff=%d", symbol, cutoff)
                continue

            for match in matches:
                if not match.confirmed or match.confirm_index in seen_confirm_indices:
                    continue
                seen_confirm_indices.add(match.confirm_index)
                try:
                    # Label against the *full* series (real subsequent bars),
                    # but build features only from `window` (what a live scan
                    # as-of that date would actually have seen) to avoid
                    # leaking future data into the features themselves.
                    label = label_triple_barrier(match, ohlcv, timeout_bars=timeout_bars)
                except ValueError:
                    log.exception("build_dataset: labeling failed for %s/%s", symbol, match.pattern)
                    continue
                feature_rows.append(build_features(match, window))
                labels.append(label)
                meta_rows.append({
                    "symbol": symbol,
                    "pattern": match.pattern,
                    "direction": match.direction,
                    "confirm_index": match.confirm_index,
                    "quality_score": match.quality_score,
                    # sym_df["date"] survives _adjusted_ohlc dropping it from
                    # `ohlcv`/`window` -- confirm_index is still positionally
                    # valid against it since ohlcv preserves sym_df's row
                    # order. Used for a time-ordered (not random) train/test
                    # split -- see time_ordered_split -- since this dataset
                    # is explicitly autocorrelated/overlapping-window (see
                    # this function's own docstring).
                    "confirm_date": sym_df["date"].iloc[match.confirm_index],
                })

    X = pd.DataFrame(feature_rows)
    y = np.array(labels, dtype=int)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def time_ordered_split(
    X: pd.DataFrame,
    y: np.ndarray,
    meta: pd.DataFrame,
    *,
    test_size: float,
    embargo_bars: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Trailing-time train/test split, replacing a random ``train_test_split``
    on data this module's own docstring already calls "highly autocorrelated"
    (overlapping rolling-cutoff windows) -- a random shuffle split lets
    near-duplicate rows from adjacent cutoffs of the *same* underlying
    pattern leak between train and test, inflating reported accuracy/AUC.

    Sorts every row by ``meta["confirm_date"]`` and takes the last
    ``test_size`` fraction (by row count) as the held-out test set,
    everything strictly before as train. ``embargo_bars`` additionally
    drops the trailing ``embargo_bars`` *calendar* days of the train period
    immediately before the test cutoff (a calendar-day approximation, not a
    per-symbol trading-day count, since rows aren't aligned to one shared
    bar index across symbols) -- mirrors
    ``scripts/validate_pattern_cnn_walkforward.py``'s own ``--embargo-days``
    convention, guarding against a train row whose own triple-barrier label
    window could still overlap into the test period.
    """
    n = len(meta)
    if n == 0:
        empty_y = np.array([], dtype=y.dtype if len(y) else int)
        return X.iloc[0:0], X.iloc[0:0], empty_y, empty_y

    order = np.argsort(pd.to_datetime(meta["confirm_date"]).to_numpy(), kind="stable")
    X_sorted = X.iloc[order].reset_index(drop=True)
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

    X_train = X_sorted.iloc[:train_end]
    y_train = y_sorted[:train_end]
    X_test = X_sorted.iloc[split_at:]
    y_test = y_sorted[split_at:]
    return X_train, X_test, y_train, y_test


def _safe_multiclass_auc(model, X_test, y_test) -> float | None:
    if len(set(y_test.tolist())) < 2:
        return None
    proba = xgb_classifier.predict_proba(model, X_test)
    try:
        return float(roc_auc_score(y_test, proba, multi_class="ovr", labels=list(xgb_classifier.LABELS)))
    except ValueError as exc:
        log.warning("AUC computation failed (%s) -- reporting n/a", exc)
        return None


# xgb_classifier.py is a deliberately thin, label-agnostic wrapper (see its
# own module docstring) whose predict_proba/predict_label nonetheless remap
# every fit's present_labels back onto the FIXED, hardcoded 3-slot layout
# `xgb_classifier.LABELS = (-1, 0, 1)` -- not a bug for a binary {0, 1} fit
# specifically (2026-09-27, Part B item 2's meta-label model): both 0 and 1
# are themselves members of that fixed tuple, at indices 1 and 2
# respectively, so predict_proba's column 2 reliably holds P(label==1) for
# ANY model fit on a subset of {-1, 0, 1} that includes 1 -- column 0
# ("-1"/stop) simply comes back all-zero for a fit that never saw -1, which
# argmax-based predict_label already handles correctly (verified: it can
# only ever emit 0 or 1 back out, never a spurious -1). This helper makes
# that reliance explicit and documented rather than a silent assumption
# baked into inline indexing at each call site below.
def _meta_p_act(model, X) -> np.ndarray:
    return xgb_classifier.predict_proba(model, X)[:, xgb_classifier._LABEL_TO_INDEX[1]]


def _safe_binary_auc(p_act: np.ndarray, y_test: np.ndarray) -> float | None:
    if len(set(y_test.tolist())) < 2:
        return None
    try:
        return float(roc_auc_score(y_test, p_act))
    except ValueError as exc:
        log.warning("Meta-label AUC computation failed (%s) -- reporting n/a", exc)
        return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-source", choices=["synthetic", "cache"], default="synthetic",
        help="'synthetic' (default -- safe/fast, no real data) or 'cache' (reads "
             "data/cache; a deliberate real run, see module docstring)",
    )
    parser.add_argument("--symbols", default=None, help="Comma-separated tickers; default depends on --data-source")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (--data-source cache only)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (also synthetic's end_date)")
    parser.add_argument("--n-days", type=int, default=400, help="--data-source synthetic only: bars per symbol")
    parser.add_argument("--zigzag-pct", type=float, default=0.03)
    parser.add_argument("--min-score", type=float, default=60.0)
    parser.add_argument("--confirm-lookback-bars", type=int, default=3)
    parser.add_argument("--stop-atr-floor", type=float, default=1.5)
    parser.add_argument("--timeout-bars", type=int, default=20, help="Triple-barrier vertical-barrier horizon (bars)")
    parser.add_argument(
        "--min-window-bars", type=int, default=60,
        help="Rolling as-of scan: smallest prefix window to scan (see build_dataset docstring)",
    )
    parser.add_argument(
        "--step-bars", type=int, default=10,
        help="Rolling as-of scan: bars between successive scan cutoffs -- smaller finds more "
             "(highly autocorrelated) rows per symbol at proportionally higher cost",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.25,
        help="Fraction of rows (by count, after sorting by confirm_date) held out as a "
             "trailing time-ordered test set -- see time_ordered_split.",
    )
    parser.add_argument(
        "--embargo-bars", type=int, default=None,
        help="Calendar-day embargo dropped from the trailing edge of the train period, "
             "immediately before the test cutoff (see time_ordered_split). Defaults to "
             "--timeout-bars, since that's the label window that could otherwise overlap "
             "into the test period.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="No longer used for the (now time-ordered, not random) train/test split -- "
             "kept in the CLI contract for any future use of XGBoost's own internal "
             "randomness.",
    )
    parser.add_argument("--output", default="data/models/pattern_xgb.pkl")
    parser.add_argument(
        "--meta-output", default=None,
        help="Path for the SEPARATE binary meta-labeling model (Part B item 2 -- see "
             "firm.patterns.ml.labeling.label_meta_binary / "
             "firm.patterns.ml.xgb_inference.score_pattern_meta_confirmation). Defaults "
             "to --output with a '_meta' suffix inserted before the extension (e.g. "
             "pattern_xgb.pkl -> pattern_xgb_meta.pkl).",
    )
    args = parser.parse_args(argv)
    if args.symbols:
        args.symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.embargo_bars is None:
        args.embargo_bars = args.timeout_bars
    if args.meta_output is None:
        out = Path(args.output)
        args.meta_output = str(out.with_name(out.stem + "_meta" + out.suffix))
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)

    panel = _load_ohlcv_panel(args)
    if panel.empty:
        print("No OHLCV data loaded -- nothing to scan. Exiting.", file=sys.stderr)
        return 1
    print(f"Loaded {len(panel)} rows across {panel['symbol'].nunique()} symbols (source={args.data_source})")

    X, y, meta = build_dataset(
        panel,
        zigzag_pct=args.zigzag_pct,
        min_score=args.min_score,
        confirm_lookback_bars=args.confirm_lookback_bars,
        stop_atr_floor=args.stop_atr_floor,
        timeout_bars=args.timeout_bars,
        min_window_bars=args.min_window_bars,
        step_bars=args.step_bars,
    )
    n_distinct = len(set(y.tolist()))
    if X.empty or n_distinct < 2:
        print(
            f"Only {len(X)} confirmed-pattern rows with {n_distinct} distinct label(s) -- "
            "not enough to train/evaluate a classifier. Try a wider universe, longer date "
            "range, or a lower --min-score.",
            file=sys.stderr,
        )
        return 1

    label_counts = pd.Series(y).value_counts().sort_index()
    print(f"Dataset: {len(X)} confirmed patterns, {X.shape[1]} features.")
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
        model = xgb_classifier.train(X_train, y_train)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    train_pred = xgb_classifier.predict_label(model, X_train)
    test_pred = xgb_classifier.predict_label(model, X_test)
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"Train accuracy: {train_acc:.3f} (n={len(y_train)})")
    print(f"Test accuracy:  {test_acc:.3f} (n={len(y_test)})")

    auc = _safe_multiclass_auc(model, X_test, y_test)
    print(f"Test AUC (OVR): {auc:.3f}" if auc is not None else "Test AUC: n/a (test split lacks class diversity)")

    xgb_classifier.save(model, args.output)
    print(f"Model saved to {args.output}")

    # Part B item 2 (2026-09-27): a SEPARATE binary act/no-act meta-labeling
    # model, trained on its own target (target-hit vs. stop-hit-or-timeout
    # collapsed together -- see label_meta_binary's docstring for why this
    # must be a distinct model, not the 3-class model's own p_target reused).
    # Reuses the SAME time-ordered train/test rows and features as the
    # direction model above -- only the label differs -- so there is no
    # second scan/build_dataset pass.
    y_meta_train = (y_train == 1).astype(int)
    y_meta_test = (y_test == 1).astype(int)
    print(
        f"\nMeta-label counts (act=1 / dont-act=0): "
        f"train={pd.Series(y_meta_train).value_counts().to_dict()} "
        f"test={pd.Series(y_meta_test).value_counts().to_dict()}"
    )
    if len(set(y_meta_train.tolist())) < 2:
        print(
            "WARNING: meta-label train split has only one distinct class -- the fitted "
            "meta model will be degenerate (always predicts that class; see "
            "xgb_classifier.train's own warning). Proceeding anyway (a real, if "
            "uninformative, model is still saved) rather than aborting, since the "
            "3-class direction model above already trained successfully.",
            file=sys.stderr,
        )

    meta_model = xgb_classifier.train(X_train, y_meta_train)
    meta_train_p_act = _meta_p_act(meta_model, X_train)
    meta_test_p_act = _meta_p_act(meta_model, X_test)
    meta_train_acc = accuracy_score(y_meta_train, (meta_train_p_act >= 0.5).astype(int))
    meta_test_acc = accuracy_score(y_meta_test, (meta_test_p_act >= 0.5).astype(int))
    print(f"Meta-label train accuracy: {meta_train_acc:.3f} (n={len(y_meta_train)})")
    print(f"Meta-label test accuracy:  {meta_test_acc:.3f} (n={len(y_meta_test)})")

    meta_auc = _safe_binary_auc(meta_test_p_act, y_meta_test)
    print(f"Meta-label test AUC: {meta_auc:.3f}" if meta_auc is not None else "Meta-label test AUC: n/a (test split lacks class diversity)")

    xgb_classifier.save(meta_model, args.meta_output)
    print(f"Meta-label model saved to {args.meta_output}")

    # Part A (2026-09-27 false-positive-rate fix): persist per-pattern
    # historical sample counts alongside the model, so
    # firm.patterns.sample_size.confidence_discount can discount rare
    # patterns (e.g. this run's own cup_handle=1, bull_flag=8) instead of
    # trusting every pattern name at face value regardless of how much
    # historical evidence backs it.
    counts_path = Path(args.output).with_name(DEFAULT_SAMPLE_COUNTS_FILENAME)
    save_sample_counts(meta["pattern"].value_counts().to_dict(), counts_path)
    print(f"Per-pattern sample counts saved to {counts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
