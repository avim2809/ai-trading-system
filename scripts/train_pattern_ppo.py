#!/usr/bin/env python
"""Train the PPO position-sizing agent end to end (Phase 4c, docs/
pattern_recognition_plan.md §4c).

**Run this under the isolated ML environment, not the main venv**::

    .venv-ml/bin/python scripts/train_pattern_ppo.py --data-source synthetic

``stable-baselines3``/``gymnasium`` have no Python 3.14 wheels yet (see
§4a) -- this script cannot import ``firm.patterns.ml.ppo_sizer`` under the
main venv.

Pipeline: load an OHLCV panel (synthetic or cached) -> scan every symbol for
confirmed chart patterns, rolling the as-of cutoff backward through history
exactly like ``scripts/train_pattern_ml.py``'s ``build_dataset`` (same
rolling-scan rationale) -> for each confirmed match, record its context
(quality_score/risk_reward/direction/volume_ratio/duration_bars) and
triple-barrier label -> build a single-step "contextual bandit" environment
(:class:`firm.patterns.ml.ppo_sizer.PatternSizingEnv`) over that dataset ->
train PPO -> report mean realized reward on a held-out split -> save the
fitted policy.

**Scope note:** standalone research tool, not wired into any live
position-sizing path -- see docs/pattern_recognition_plan.md §4c: unlike
the XGBoost/CNN models, a saved SB3 policy doesn't export cleanly to ONNX,
so consuming it even for inference still needs stable-baselines3 itself,
which the main firm-api process (Python 3.14) cannot install. A future
"live wiring" step would need to solve that first.

Usage:
    # Fast, dependency-free smoke run (synthetic prices, seconds):
    .venv-ml/bin/python scripts/train_pattern_ppo.py --data-source synthetic

    # Real run against this repo's actual cached historical data:
    .venv-ml/bin/python scripts/train_pattern_ppo.py --data-source cache \\
        --symbols AAPL,MSFT,NVDA,GOOG,AMZN,META,TSLA,JPM,V,JNJ \\
        --start 2018-01-01 --end 2025-01-01 --total-timesteps 50000 \\
        --output data/models/pattern_ppo.zip
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

from firm.data.synthetic import DEFAULT_SYMBOLS, make_synthetic_prices  # noqa: E402
from firm.patterns.ml import ppo_sizer  # noqa: E402
from firm.patterns.ml.labeling import label_triple_barrier  # noqa: E402
from firm.patterns.scanner import scan_symbol  # noqa: E402

# Reuses the strategy's own split/dividend adjustment -- see
# scripts/train_pattern_ml.py's identical import for the full rationale.
from firm.strategies.pattern_recognition import _adjusted_ohlc  # noqa: E402

log = logging.getLogger(__name__)

# Mirrors scripts/train_pattern_ml.py's own hardcoded default universe.
_DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "AMD",
    "CRM", "NFLX", "ADBE", "JPM", "GS", "BAC", "V", "MA", "JNJ", "UNH",
    "LLY", "XOM", "CVX", "SPY", "QQQ", "IWM",
]


def _load_ohlcv_panel(args: argparse.Namespace) -> pd.DataFrame:
    """Same shape/contract as train_pattern_ml.py's helper of the same name."""
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


def build_context_dataset(
    panel: pd.DataFrame,
    *,
    zigzag_pct: float,
    min_score: float,
    confirm_lookback_bars: int,
    stop_atr_floor: float,
    timeout_bars: int,
    min_window_bars: int = 60,
    step_bars: int = 10,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Same rolling as-of scan as train_pattern_ml.py's ``build_dataset``,
    collecting the richer per-match context ``firm.patterns.ml.ppo_sizer
    .build_observations`` needs (risk_reward/volume_ratio/duration_bars, in
    addition to quality_score/direction) instead of a flat feature vector.

    Returns ``(meta, labels)`` -- pass straight to
    ``ppo_sizer.build_observations``.
    """
    meta_rows: list[dict] = []
    labels: list[int] = []

    if panel.empty or "symbol" not in panel.columns:
        return pd.DataFrame(), np.array([], dtype=int)

    for symbol, sym_df in panel.groupby("symbol"):
        sym_df = sym_df.sort_values("date").reset_index(drop=True)
        try:
            ohlcv = _adjusted_ohlc(sym_df)
        except Exception:
            log.exception("build_context_dataset: failed to build adjusted OHLC for %s", symbol)
            continue

        n = len(ohlcv)
        if n < min_window_bars:
            continue
        seen_confirm_indices: set[int] = set()
        for cutoff in range(min_window_bars, n + 1, step_bars):
            window = ohlcv.iloc[:cutoff]
            try:
                matches = scan_symbol(
                    window, zigzag_pct=zigzag_pct, min_score=min_score,
                    confirm_lookback_bars=confirm_lookback_bars, stop_atr_floor=stop_atr_floor,
                )
            except Exception:
                log.exception("build_context_dataset: scan_symbol failed for %s at cutoff=%d", symbol, cutoff)
                continue

            for match in matches:
                if not match.confirmed or match.confirm_index in seen_confirm_indices:
                    continue
                seen_confirm_indices.add(match.confirm_index)
                try:
                    label = label_triple_barrier(match, ohlcv, timeout_bars=timeout_bars)
                except ValueError:
                    log.exception("build_context_dataset: labeling failed for %s/%s", symbol, match.pattern)
                    continue
                labels.append(label)
                meta_rows.append({
                    "symbol": symbol, "pattern": match.pattern, "direction": match.direction,
                    "confirm_index": match.confirm_index, "quality_score": match.quality_score,
                    "risk_reward": match.risk_reward, "volume_ratio": match.volume_ratio,
                    "duration_bars": match.duration_bars,
                })

    meta = pd.DataFrame(meta_rows)
    return meta, np.array(labels, dtype=int)


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
    parser.add_argument("--min-window-bars", type=int, default=60)
    parser.add_argument("--step-bars", type=int, default=10)
    parser.add_argument("--total-timesteps", type=int, default=20_000)
    parser.add_argument(
        "--risk-aversion", type=float, default=1.0,
        help="Quadratic penalty on position size (see ppo_sizer.PatternSizingEnv) -- 0 makes "
             "'always bet max size' reward-optimal whenever the dataset's average outcome is "
             "positive, which isn't a useful sizing lesson. The default is calibrated against "
             "a real run over this repo's cached data (mean outcome*scale ~1.14): 0.5 was "
             "still too weak there (99.4%% of predictions saturated at max size regardless of "
             "context) -- 1.0 is the smallest value verified to break that saturation.",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="data/models/pattern_ppo.zip")
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

    meta, labels = build_context_dataset(
        panel, zigzag_pct=args.zigzag_pct, min_score=args.min_score,
        confirm_lookback_bars=args.confirm_lookback_bars, stop_atr_floor=args.stop_atr_floor,
        timeout_bars=args.timeout_bars, min_window_bars=args.min_window_bars, step_bars=args.step_bars,
    )
    if len(meta) < 20:
        print(
            f"Only {len(meta)} confirmed-pattern context row(s) -- not enough to train/evaluate. "
            "Try a wider universe, longer date range, or a lower --min-score.", file=sys.stderr,
        )
        return 1

    print(f"Dataset: {len(meta)} confirmed patterns.")
    print(f"Label counts (target=+1 / timeout=0 / stop=-1): {pd.Series(labels).value_counts().sort_index().to_dict()}")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(meta))
    n_test = int(len(meta) * args.test_size)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    obs, outcome, scale = ppo_sizer.build_observations(meta, labels)
    obs_train, outcome_train, scale_train = obs[train_idx], outcome[train_idx], scale[train_idx]
    obs_test, outcome_test, scale_test = obs[test_idx], outcome[test_idx], scale[test_idx]

    try:
        model = ppo_sizer.train_ppo(
            obs_train, outcome_train, scale_train, total_timesteps=args.total_timesteps,
            seed=args.seed, risk_aversion=args.risk_aversion,
        )
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    def _reward(size: float, outcome: float, scale: float) -> float:
        return size * outcome * scale - 0.001 * abs(size) - args.risk_aversion * size * size

    def _mean_reward(observations, outcomes, scales) -> float:
        rewards = [
            _reward(ppo_sizer.predict_position_size(model, observations[i]), outcomes[i], scales[i])
            for i in range(len(observations))
        ]
        return float(np.mean(rewards)) if rewards else 0.0

    # Baseline: always bet the max size in the pattern's own signaled
    # direction (size=1.0) -- what a naive "trust every confirmed pattern
    # fully" strategy would do, for the trained policy's mean reward to beat.
    baseline_train = float(np.mean([_reward(1.0, o, s) for o, s in zip(outcome_train, scale_train)]))
    baseline_test = float(np.mean([_reward(1.0, o, s) for o, s in zip(outcome_test, scale_test)]))

    print(f"Mean reward (train): policy={_mean_reward(obs_train, outcome_train, scale_train):.4f}  always-max-size={baseline_train:.4f}  (n={len(obs_train)})")
    print(f"Mean reward (test):  policy={_mean_reward(obs_test, outcome_test, scale_test):.4f}  always-max-size={baseline_test:.4f}  (n={len(obs_test)})")

    ppo_sizer.save(model, args.output)
    print(f"Model saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
