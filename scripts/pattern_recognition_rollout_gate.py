#!/usr/bin/env python
"""Staged keep/hold/rollback gate for pattern_recognition's live CNN toggle.

Research finding this operationalizes: with only ~2 weeks of real live
trading history for pattern_recognition, there is essentially zero
statistical power to judge live performance from raw P&L alone -- acting on
early noisy live results (e.g. rolling back because of a couple of bad days)
is a classic false-positive-prone mistake. The correct discipline is a
pre-committed, staged gate:

  1. HOLD any judgment until BOTH (a) the backtest-side walk-forward/PBO/DSR
     audit (``scripts/validate_pattern_cnn_walkforward.py``, Task A) passes,
     AND (b) enough live days/signals have accrued (a pre-committed
     sample-size gate: ``--min-live-days`` / ``--min-live-signals``).
  2. Only once the sample-size gate is met does this apply live-performance
     rollback thresholds (rolling live Sharpe < a fraction of the backtest
     OOS Sharpe, or live drawdown > a multiple of the backtest's 95th
     percentile OOS drawdown) to decide KEEP vs ROLLBACK.

The pure decision logic lives in :func:`evaluate_rollout_gate` (no I/O, no
network) so it is unit-tested directly in
``tests/test_pattern_recognition_rollout_gate.py`` -- everything else in this
module is I/O plumbing (reading Task A's JSON, hitting the live API, shaping
its response into the numbers that function needs) that is not meaningfully
unit-testable and is instead exercised by actually running this script
against whatever live instances are reachable (best-effort; see the
``--live-data-json`` override below for feeding it a fixture when they are
not).

Usage:
    # Best-effort against both live instances (localhost:8000 IBKR/blended,
    # localhost:8001 Alpaca/sleeved), using Task A's default output path.
    python scripts/pattern_recognition_rollout_gate.py

    # Custom thresholds / endpoints / backtest audit path.
    python scripts/pattern_recognition_rollout_gate.py \\
        --backtest-audit-json /tmp/pattern_cnn_walkforward_audit.json \\
        --endpoints http://localhost:8000 http://localhost:8001 \\
        --min-live-days 20 --min-live-signals 30 \\
        --sharpe-ratio-threshold-fraction 0.5 --drawdown-multiple-threshold 2.0

    # Fully offline / testable: skip the live HTTP calls, feed a fixture.
    python scripts/pattern_recognition_rollout_gate.py --live-data-json fixture.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import requests

log = logging.getLogger(__name__)

DEFAULT_BACKTEST_AUDIT_JSON = "/tmp/pattern_cnn_walkforward_audit.json"
DEFAULT_ENDPOINTS = ["http://localhost:8000", "http://localhost:8001"]
DEFAULT_STRATEGY = "pattern_recognition"

# Pre-committed sample-size gate (research finding above) -- deliberately
# generous defaults, not tuned after the fact to whatever the current live
# history happens to look like.
DEFAULT_MIN_LIVE_DAYS = 20
DEFAULT_MIN_LIVE_SIGNALS = 30

# Rollback thresholds, only ever applied once the sample-size gate passes.
DEFAULT_SHARPE_RATIO_THRESHOLD_FRACTION = 0.5
DEFAULT_DRAWDOWN_MULTIPLE_THRESHOLD = 2.0

DEFAULT_HTTP_TIMEOUT_SECONDS = 5.0
DEFAULT_OUTPUT = "/tmp/pattern_recognition_rollout_gate_result.json"
TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Pure decision logic -- see tests/test_pattern_recognition_rollout_gate.py
# ---------------------------------------------------------------------------

def evaluate_rollout_gate(
    *,
    backtest_verdict: str | None,
    backtest_oos_sharpe_mean: float | None,
    backtest_oos_drawdown_p95: float | None,
    live_sample_days: int,
    live_sample_signals: int | None,
    live_sharpe_estimate: float | None,
    live_max_drawdown: float | None,
    min_live_days: int = DEFAULT_MIN_LIVE_DAYS,
    min_live_signals: int = DEFAULT_MIN_LIVE_SIGNALS,
    sharpe_ratio_threshold_fraction: float = DEFAULT_SHARPE_RATIO_THRESHOLD_FRACTION,
    drawdown_multiple_threshold: float = DEFAULT_DRAWDOWN_MULTIPLE_THRESHOLD,
) -> dict[str, Any]:
    """Given already-fetched backtest + live numbers, decide KEEP/HOLD/ROLLBACK.

    Explicit, deterministic decision rule:

    1. **Insufficient live sample** (``live_sample_days < min_live_days`` AND,
       when a signal count is available, ``live_sample_signals <
       min_live_signals``) -> ``HOLD (insufficient live sample, backtest
       verdict=...)`` *regardless* of how good/bad the live numbers look --
       this is the whole point of the staged gate: a live Sharpe/drawdown
       read from a couple of weeks of data has ~zero statistical power, so
       acting on it (either direction) is a false-positive-prone mistake.
    2. **Sufficient sample** -> compute the two rollback trips:
       - Sharpe trip: ``live_sharpe_estimate < sharpe_ratio_threshold_fraction
         * backtest_oos_sharpe_mean`` (skipped if either side is unavailable
         or the backtest Sharpe isn't positive -- a fraction of a
         non-positive benchmark is not a meaningful floor).
       - Drawdown trip: ``live_max_drawdown > drawdown_multiple_threshold *
         backtest_oos_drawdown_p95`` (skipped if either side is unavailable
         or the backtest drawdown is not positive).
       - Either trip -> ``ROLLBACK``, independent of the backtest verdict:
         once there is enough live data to say something, bad live
         performance is real signal and overrides a passing backtest.
       - No trip AND backtest verdict == ``"pass"`` -> ``KEEP``: both halves
         of the "BOTH (a) and (b)" requirement are satisfied.
       - No trip AND backtest verdict != ``"pass"`` -> ``HOLD``: this is the
         documented judgment call for the ambiguous case (sufficient live
         sample, neutral/fine live performance, but the backtest side never
         cleared PBO/DSR). Live performance alone being fine is not enough
         to KEEP without the backtest supporting it, but it's also not bad
         enough to justify an active ROLLBACK -- so this stays at HOLD
         rather than acting in either direction on a mixed read.

    Returns a dict with ``recommendation`` (``"KEEP"``/``"HOLD (...)"``/
    ``"ROLLBACK"``), ``sample_ok``, ``rollback_triggered``, ``sharpe_trip``,
    ``drawdown_trip``, and a ``reasoning`` list of human-readable strings
    documenting exactly what was checked.
    """
    sample_ok = live_sample_days >= min_live_days or (
        live_sample_signals is not None and live_sample_signals >= min_live_signals
    )
    reasoning: list[str] = []

    if not sample_ok:
        signal_clause = (
            f"and {live_sample_signals} signal(s) < {min_live_signals} required"
            if live_sample_signals is not None
            else "(no live signal count available -- only the day-count gate applies)"
        )
        reasoning.append(
            f"live sample insufficient: {live_sample_days} day(s) < "
            f"{min_live_days} required {signal_clause}"
        )
        reasoning.append(
            f"HOLD regardless of live performance -- acting on fewer than "
            f"{min_live_days} days of live history has essentially zero "
            "statistical power; rolling back on a couple of noisy days is a "
            "classic false-positive-prone mistake"
        )
        return {
            "recommendation": f"HOLD (insufficient live sample, backtest verdict={backtest_verdict})",
            "sample_ok": False,
            "rollback_triggered": False,
            "sharpe_trip": False,
            "drawdown_trip": False,
            "reasoning": reasoning,
        }

    signal_clause = (
        f"or {live_sample_signals} signal(s) >= {min_live_signals}"
        if live_sample_signals is not None
        else "(no live signal count available -- day-count alone satisfied the gate)"
    )
    reasoning.append(
        f"live sample sufficient: {live_sample_days} day(s) >= {min_live_days} "
        f"{signal_clause}"
    )

    sharpe_trip = False
    if (
        live_sharpe_estimate is not None
        and backtest_oos_sharpe_mean is not None
        and backtest_oos_sharpe_mean > 0
    ):
        threshold = sharpe_ratio_threshold_fraction * backtest_oos_sharpe_mean
        sharpe_trip = live_sharpe_estimate < threshold
        reasoning.append(
            f"Sharpe check: live={live_sharpe_estimate:.3f} vs "
            f"{sharpe_ratio_threshold_fraction * 100:.0f}% of backtest OOS Sharpe "
            f"({backtest_oos_sharpe_mean:.3f}) = {threshold:.3f} -> "
            + ("TRIP (below threshold)" if sharpe_trip else "ok")
        )
    else:
        reasoning.append(
            "Sharpe rollback check skipped (missing live Sharpe estimate, or "
            "backtest OOS Sharpe missing/non-positive)"
        )

    drawdown_trip = False
    if (
        live_max_drawdown is not None
        and backtest_oos_drawdown_p95 is not None
        and backtest_oos_drawdown_p95 > 0
    ):
        threshold = drawdown_multiple_threshold * backtest_oos_drawdown_p95
        drawdown_trip = live_max_drawdown > threshold
        reasoning.append(
            f"Drawdown check: live={live_max_drawdown:.3f} vs "
            f"{drawdown_multiple_threshold:.1f}x backtest 95th-pct OOS drawdown "
            f"({backtest_oos_drawdown_p95:.3f}) = {threshold:.3f} -> "
            + ("TRIP (exceeds threshold)" if drawdown_trip else "ok")
        )
    else:
        reasoning.append(
            "Drawdown rollback check skipped (missing live max drawdown, or "
            "backtest 95th-pct OOS drawdown missing/non-positive)"
        )

    rollback_triggered = sharpe_trip or drawdown_trip

    if rollback_triggered:
        recommendation = "ROLLBACK"
        reasoning.append(
            "ROLLBACK: live performance tripped a rollback threshold with a "
            "sufficient sample -- this overrides a passing backtest, since "
            "enough live data now exists for the live read to be real signal"
        )
    elif backtest_verdict == "pass":
        recommendation = "KEEP"
        reasoning.append(
            "KEEP: backtest verdict='pass' AND sufficient live sample shows "
            "no rollback trip -- both halves of the staged gate are satisfied"
        )
    else:
        recommendation = "HOLD"
        reasoning.append(
            f"HOLD: backtest verdict={backtest_verdict!r} (not 'pass'), so the "
            "two-gate KEEP requirement is not met even though live performance "
            "alone is neutral/fine (no rollback trip). Documented judgment "
            "call: a non-passing backtest blocks KEEP, but fine live "
            "performance alone does not justify an active ROLLBACK either -- "
            "stay at HOLD rather than act in either direction on this mixed read"
        )

    return {
        "recommendation": recommendation,
        "sample_ok": True,
        "rollback_triggered": rollback_triggered,
        "sharpe_trip": sharpe_trip,
        "drawdown_trip": drawdown_trip,
        "reasoning": reasoning,
    }


# ---------------------------------------------------------------------------
# I/O plumbing (not unit-tested; exercised by actually running the script)
# ---------------------------------------------------------------------------

def _load_backtest_audit(path: Path) -> dict[str, Any]:
    """Read Task A's (``validate_pattern_cnn_walkforward.py``) output JSON.

    Degrades to ``{}`` (never raises) on a missing/unparseable file, logged
    as a warning -- every downstream backtest-side check then reads as
    "unavailable" rather than crashing this operational tool.
    """
    if not path.exists():
        log.warning(
            "backtest audit JSON not found at %s -- backtest-side checks "
            "(verdict/Sharpe/drawdown) will be treated as unavailable",
            path,
        )
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning(
            "could not read/parse backtest audit JSON %s (%s) -- treating "
            "as unavailable",
            path, exc,
        )
        return {}


def _backtest_oos_stats(audit: dict[str, Any]) -> dict[str, Any]:
    """Extract verdict + OOS Sharpe mean + 95th-pct OOS drawdown from Task
    A's aggregate walk-forward output (``ExperimentRunner.aggregate_walk_
    forward``'s ``overfitting``/``metrics`` blocks)."""
    overfit = audit.get("overfitting") or {}
    metrics = audit.get("metrics") or {}
    sharpe_block = metrics.get("sharpe_ratio") or {}
    dd_values = (metrics.get("max_drawdown") or {}).get("values") or []
    drawdown_p95 = None
    if dd_values:
        drawdown_p95 = float(np.percentile(np.asarray(dd_values, dtype=float), 95))
    return {
        "verdict": overfit.get("verdict"),
        "pbo": overfit.get("pbo"),
        "deflated_sharpe": overfit.get("deflated_sharpe"),
        "oos_sharpe_mean": sharpe_block.get("mean"),
        "oos_drawdown_p95": drawdown_p95,
        "n_folds": audit.get("n_folds"),
    }


def _fetch_json(url: str, timeout: float) -> dict[str, Any] | None:
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:  # noqa: BLE001 - network/HTTP: report, never crash
        log.warning("could not reach %s (%s) -- treating as unreachable", url, exc)
        return None


def _fetch_live_series(
    base_url: str, strategy: str, timeout: float
) -> tuple[list[str], list[float]] | None:
    """GET {base_url}/api/live/attribution/history and pull out *strategy*'s
    ``dates``/``returns``. Returns None if the endpoint is unreachable or the
    strategy has no history there yet (e.g. blended attribution hasn't
    recorded a pattern_recognition-attributed trade)."""
    data = _fetch_json(f"{base_url.rstrip('/')}/api/live/attribution/history", timeout)
    if data is None:
        return None
    entry = data.get(strategy)
    if not entry:
        log.info(
            "%s: no attribution history for strategy=%s yet (0 days)",
            base_url, strategy,
        )
        return [], []
    return entry.get("dates", []), entry.get("returns", [])


def _fetch_live_signal_count(base_url: str, strategy: str, timeout: float) -> int | None:
    """Best-effort live signal count from GET /api/live/attribution's
    aggregate scalar metrics, if that response happens to expose one under
    any of a few plausible key names. Returns None (falls back to the
    day-count-only gate) if the endpoint is unreachable or exposes nothing
    resembling a signal/trade count for this strategy.
    """
    data = _fetch_json(f"{base_url.rstrip('/')}/api/live/attribution", timeout)
    if not data:
        return None
    entry = data.get(strategy) or {}
    for key in ("signal_count", "n_signals", "trade_count", "n_trades", "num_trades"):
        if key in entry:
            try:
                return int(entry[key])
            except (TypeError, ValueError):
                continue
    return None


def _live_stats_from_series(returns: list[float]) -> dict[str, Any]:
    """Rough, low-power live stats from a raw daily-return series.

    Deliberately simple mean/std annualized Sharpe-like statistic and a
    plain running-peak drawdown -- NOT a rigorous statistical test (no
    autocorrelation/regime adjustment, no confidence interval, no
    correction for the number of observations). With a small N this number
    is close to meaningless on its own; that is exactly why
    evaluate_rollout_gate refuses to act on it until the sample-size gate
    (min_live_days/min_live_signals) is met.
    """
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {
            "n_days": 0, "cumulative_return": 0.0,
            "sharpe_estimate": None, "max_drawdown": None,
        }
    cumulative_return = float(np.prod(1.0 + arr) - 1.0)
    sharpe_estimate = None
    if n >= 2:
        std = float(arr.std(ddof=1))
        if std > 1e-12:
            sharpe_estimate = float(
                arr.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)
            )
    nav = np.cumprod(1.0 + arr)
    running_max = np.maximum.accumulate(nav)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(running_max > 0, (running_max - nav) / running_max, 0.0)
    max_dd = float(drawdown.max()) if drawdown.size else None
    return {
        "n_days": n,
        "cumulative_return": cumulative_return,
        "sharpe_estimate": sharpe_estimate,
        "max_drawdown": max_dd,
    }


def _evaluate_instance(
    label: str,
    dates: list[str],
    returns: list[float],
    signal_count: int | None,
    backtest_stats: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    live_stats = _live_stats_from_series(returns)
    decision = evaluate_rollout_gate(
        backtest_verdict=backtest_stats.get("verdict"),
        backtest_oos_sharpe_mean=backtest_stats.get("oos_sharpe_mean"),
        backtest_oos_drawdown_p95=backtest_stats.get("oos_drawdown_p95"),
        live_sample_days=live_stats["n_days"],
        live_sample_signals=signal_count,
        live_sharpe_estimate=live_stats["sharpe_estimate"],
        live_max_drawdown=live_stats["max_drawdown"],
        min_live_days=args.min_live_days,
        min_live_signals=args.min_live_signals,
        sharpe_ratio_threshold_fraction=args.sharpe_ratio_threshold_fraction,
        drawdown_multiple_threshold=args.drawdown_multiple_threshold,
    )
    log.info(
        "%s: n_days=%d signal_count=%s cumulative_return=%s sharpe_estimate=%s "
        "max_drawdown=%s -> %s",
        label, live_stats["n_days"], signal_count,
        live_stats["cumulative_return"], live_stats["sharpe_estimate"],
        live_stats["max_drawdown"], decision["recommendation"],
    )
    return {
        "label": label,
        "reachable": True,
        "date_range": {"first": dates[0], "last": dates[-1]} if dates else None,
        "live_stats": live_stats,
        "signal_count": signal_count,
        "decision": decision,
    }


_PRIORITY = {"ROLLBACK": 2, "HOLD": 1, "KEEP": 0}


def _overall_recommendation(per_instance: list[dict[str, Any]], backtest_verdict: str | None) -> str:
    """Most conservative recommendation across every reachable instance
    (ROLLBACK > HOLD > KEEP) -- a rollback signal on either the blended
    (IBKR, :8000) or sleeved (Alpaca, :8001) book should not be masked by
    the other one looking fine, since they run independent capital.
    """
    reachable = [p for p in per_instance if p["reachable"]]
    if not reachable:
        return f"HOLD (no live instance reachable, backtest verdict={backtest_verdict})"

    def _base(rec: str) -> str:
        return rec.split(" ", 1)[0]

    worst = max(reachable, key=lambda p: _PRIORITY.get(_base(p["decision"]["recommendation"]), 1))
    return worst["decision"]["recommendation"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-audit-json", default=DEFAULT_BACKTEST_AUDIT_JSON)
    parser.add_argument("--endpoints", nargs="*", default=list(DEFAULT_ENDPOINTS))
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--min-live-days", type=int, default=DEFAULT_MIN_LIVE_DAYS)
    parser.add_argument("--min-live-signals", type=int, default=DEFAULT_MIN_LIVE_SIGNALS)
    parser.add_argument(
        "--sharpe-ratio-threshold-fraction", type=float,
        default=DEFAULT_SHARPE_RATIO_THRESHOLD_FRACTION,
        help="Rollback if rolling live Sharpe < this fraction of the "
        "backtest OOS Sharpe (default 0.5).",
    )
    parser.add_argument(
        "--drawdown-multiple-threshold", type=float,
        default=DEFAULT_DRAWDOWN_MULTIPLE_THRESHOLD,
        help="Rollback if live drawdown > this multiple of the backtest's "
        "95th-percentile OOS drawdown (default 2.0).",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_HTTP_TIMEOUT_SECONDS)
    parser.add_argument(
        "--live-data-json", default=None,
        help="Optional fixture JSON overriding the live HTTP calls entirely "
        "-- {label: {\"dates\": [...], \"returns\": [...], "
        "\"signal_count\": <int, optional>}}. Lets this script (and its "
        "logic) be exercised end-to-end without a reachable live instance.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    audit = _load_backtest_audit(Path(args.backtest_audit_json))
    backtest_stats = _backtest_oos_stats(audit)
    log.info(
        "backtest-side (Task A, %s): verdict=%s pbo=%s dsr=%s oos_sharpe_mean=%s "
        "oos_drawdown_p95=%s (n_folds=%s)",
        args.backtest_audit_json, backtest_stats["verdict"], backtest_stats["pbo"],
        backtest_stats["deflated_sharpe"], backtest_stats["oos_sharpe_mean"],
        backtest_stats["oos_drawdown_p95"], backtest_stats["n_folds"],
    )

    per_instance: list[dict[str, Any]] = []

    if args.live_data_json:
        log.info("using --live-data-json fixture %s (real HTTP calls skipped)", args.live_data_json)
        fixture = json.loads(Path(args.live_data_json).read_text(encoding="utf-8"))
        for label, entry in fixture.items():
            dates = entry.get("dates", [])
            returns = entry.get("returns", [])
            signal_count = entry.get("signal_count")
            per_instance.append(
                _evaluate_instance(label, dates, returns, signal_count, backtest_stats, args)
            )
    else:
        for base_url in args.endpoints:
            series = _fetch_live_series(base_url, args.strategy, args.timeout)
            if series is None:
                log.warning("%s: unreachable -- excluded from this gate run", base_url)
                per_instance.append({
                    "label": base_url, "reachable": False,
                    "date_range": None, "live_stats": None,
                    "signal_count": None, "decision": None,
                })
                continue
            dates, returns = series
            signal_count = _fetch_live_signal_count(base_url, args.strategy, args.timeout)
            per_instance.append(
                _evaluate_instance(base_url, dates, returns, signal_count, backtest_stats, args)
            )

    unreachable = [p["label"] for p in per_instance if not p["reachable"]]
    if unreachable:
        log.warning(
            "could not fetch live attribution history from: %s -- gate "
            "result for those instance(s) is simply omitted, not treated "
            "as a pass or a fail",
            unreachable,
        )

    overall = _overall_recommendation(per_instance, backtest_stats.get("verdict"))

    result = {
        "strategy": args.strategy,
        "backtest_audit_json": args.backtest_audit_json,
        "backtest": backtest_stats,
        "thresholds": {
            "min_live_days": args.min_live_days,
            "min_live_signals": args.min_live_signals,
            "sharpe_ratio_threshold_fraction": args.sharpe_ratio_threshold_fraction,
            "drawdown_multiple_threshold": args.drawdown_multiple_threshold,
        },
        "instances": per_instance,
        "overall_recommendation": overall,
    }

    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(json.dumps(result, indent=2, default=str))
    print(f"\nOVERALL RECOMMENDATION: {overall}")
    for inst in per_instance:
        if not inst["reachable"]:
            print(f"  [{inst['label']}] unreachable")
            continue
        print(f"  [{inst['label']}] {inst['decision']['recommendation']}")
        for line in inst["decision"]["reasoning"]:
            print(f"      - {line}")
    log.info("Full results: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
