#!/usr/bin/env python
"""Shadow/validation report for the rule-based chart-pattern quality_score.

Reads every persisted match from ``PatternScanHistoryStore``
(``src/firm/live/pattern_scan_history.py`` — populated by the independently
scheduled ``src/firm/live/pattern_scan_job.py``, docs/pattern_recognition_plan.md
§2a/§2b) and reports hit-rate / correlation statistics that answer one
question: is the scorer's ``quality_score`` actually predictive of whether a
pattern goes on to hit its target vs. its stop?

Right now (2026) this DB has only a handful of rows from a one-off
historical test run, all still ``outcome IS NULL`` — this script is written
to degrade gracefully on that near-empty case (see ``_MIN_RESOLVED_FOR_STATS``
below) and to get more useful as the scheduled job accumulates real,
resolved rows over time.

Examples:
    # Human-readable report against the default on-disk DB
    python scripts/analyze_pattern_scan_outcomes.py

    # Only rows with quality_score >= 70, also dump the same data as JSON
    python scripts/analyze_pattern_scan_outcomes.py --min-quality 70 --output /tmp/report.json

    # Scope to one symbol/pattern
    python scripts/analyze_pattern_scan_outcomes.py --symbol AAPL --pattern rising_wedge
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.live.pattern_scan_history import PatternScanHistoryStore  # noqa: E402

log = logging.getLogger(__name__)

# Quality-score bands reported individually, in this fixed display order.
_QUALITY_BANDS: list[tuple[float, float, str]] = [
    (float("-inf"), 60.0, "<60"),
    (60.0, 70.0, "60-70"),
    (70.0, 80.0, "70-80"),
    (80.0, 90.0, "80-90"),
    (90.0, float("inf"), "90-100"),
]

# Below this many *decisive* (target_hit/stop_hit) rows, hit-rate and
# correlation numbers are too noisy to draw conclusions from -- still
# computed and printed (never suppressed), just flagged with a caution.
_MIN_RESOLVED_FOR_STATS = 20

# Below this many (quality_score, outcome) pairs, a Pearson correlation is
# nearly meaningless (e.g. n=2 is trivially +-1) -- reported as None instead
# of a misleadingly precise-looking number.
_MIN_PAIRS_FOR_CORRELATION = 5

# Outcome -> numeric label fed into the quality_score/outcome correlation.
# `timeout` rows are deliberately EXCLUDED (not coded as 0): a timeout means
# neither barrier was touched in the tracked window, i.e. the scorer's
# target/stop pair was simply never resolved one way or the other, which is
# a statement about the *risk/reward geometry*, not about whether
# quality_score correctly ranked the setup. Mixing that in as a "0" would
# bias the correlation toward zero regardless of how good the scorer is.
_OUTCOME_TO_NUMERIC = {"target_hit": 1.0, "stop_hit": -1.0}


def _quality_band(quality_score: float | None) -> str | None:
    if quality_score is None:
        return None
    for lo, hi, label in _QUALITY_BANDS:
        if lo <= quality_score < hi:
            return label
    return _QUALITY_BANDS[-1][2]  # exactly 100 (or above) falls in the top band


def compute_hit_rate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Outcome counts + decisive hit rate for one group of rows.

    ``hit_rate`` is ``target_hit / (target_hit + stop_hit)``, excluding
    timeouts and still-pending rows from the denominator -- both would
    otherwise silently dilute the rate without telling you anything about
    scorer accuracy.
    """
    target_hit = stop_hit = timeout = pending = 0
    for row in rows:
        outcome = row.get("outcome")
        if outcome == "target_hit":
            target_hit += 1
        elif outcome == "stop_hit":
            stop_hit += 1
        elif outcome == "timeout":
            timeout += 1
        else:
            pending += 1

    decisive = target_hit + stop_hit
    hit_rate = (target_hit / decisive) if decisive > 0 else None
    return {
        "n_total": len(rows),
        "target_hit": target_hit,
        "stop_hit": stop_hit,
        "timeout": timeout,
        "pending": pending,
        "n_decisive": decisive,
        "hit_rate": hit_rate,
    }


def group_hit_rates(
    rows: list[dict[str, Any]], key_fn: Callable[[dict[str, Any]], str | None]
) -> dict[str, dict[str, Any]]:
    """``compute_hit_rate`` broken out by ``key_fn(row)``; rows for which
    ``key_fn`` returns ``None`` (e.g. no quality_score recorded) are
    dropped from the grouping entirely rather than forming a bogus "None"
    bucket.
    """
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = key_fn(row)
        if key is None:
            continue
        buckets.setdefault(key, []).append(row)
    return {key: compute_hit_rate(group_rows) for key, group_rows in buckets.items()}


def quality_score_outcome_correlation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pearson correlation between ``quality_score`` and a numeric outcome
    (target_hit=1, stop_hit=-1; timeouts and pending rows excluded -- see
    ``_OUTCOME_TO_NUMERIC``'s module-level docstring for why). This is the
    shadow/validation check for whether the scorer's quality_score is
    actually doing its job: a strong positive correlation means higher
    scores really do predict more target hits.

    Uses plain ``numpy.corrcoef`` (no scipy import needed for a single
    Pearson coefficient).
    """
    pairs = [
        (row["quality_score"], _OUTCOME_TO_NUMERIC[row["outcome"]])
        for row in rows
        if row.get("quality_score") is not None and row.get("outcome") in _OUTCOME_TO_NUMERIC
    ]
    n = len(pairs)
    if n < _MIN_PAIRS_FOR_CORRELATION:
        log.debug(
            "quality_score/outcome correlation: only %d decisive, scored row(s) -- "
            "need >= %d for a meaningful Pearson r",
            n, _MIN_PAIRS_FOR_CORRELATION,
        )
        return {"n": n, "pearson_r": None}

    scores = np.array([p[0] for p in pairs], dtype=float)
    outcomes = np.array([p[1] for p in pairs], dtype=float)
    if np.std(scores) == 0 or np.std(outcomes) == 0:
        # Degenerate (constant) input -- corrcoef would return NaN.
        log.debug("quality_score/outcome correlation: degenerate (zero-variance) input")
        return {"n": n, "pearson_r": None}

    pearson_r = float(np.corrcoef(scores, outcomes)[0, 1])
    return {"n": n, "pearson_r": pearson_r}


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the full stats report for ``rows`` (already filtered by any
    CLI symbol/pattern/min-quality options)."""
    overall = compute_hit_rate(rows)
    report = {
        "n_total": len(rows),
        "overall": overall,
        "by_quality_band": group_hit_rates(rows, lambda r: _quality_band(r.get("quality_score"))),
        "by_pattern": group_hit_rates(rows, lambda r: r.get("pattern")),
        "by_direction": group_hit_rates(rows, lambda r: r.get("direction")),
        "quality_score_outcome_correlation": quality_score_outcome_correlation(rows),
        "min_resolved_for_stats": _MIN_RESOLVED_FOR_STATS,
    }
    if overall["n_decisive"] < _MIN_RESOLVED_FOR_STATS:
        log.info(
            "Only %d decisive (target/stop) row(s) out of %d total -- below the %d-row "
            "threshold for reliable stats; report below is computed anyway but treat it "
            "as provisional",
            overall["n_decisive"], overall["n_total"], _MIN_RESOLVED_FOR_STATS,
        )
    return report


def _fmt_pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def _fmt_hit_rate_row(label: str, stats: dict[str, Any]) -> str:
    return (
        f"  {label:<20} n={stats['n_total']:<5} target_hit={stats['target_hit']:<4} "
        f"stop_hit={stats['stop_hit']:<4} timeout={stats['timeout']:<4} "
        f"pending={stats['pending']:<4} hit_rate={_fmt_pct(stats['hit_rate'])}"
    )


def format_report_text(report: dict[str, Any]) -> str:
    """Render ``build_report``'s output as a clean plain-text report."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("Pattern-scan quality_score validation report")
    lines.append("=" * 78)

    n_total = report["n_total"]
    overall = report["overall"]
    if n_total == 0:
        lines.append("No pattern-scan history rows match the given filters.")
        return "\n".join(lines)

    if overall["n_decisive"] < report["min_resolved_for_stats"]:
        lines.append(
            f"NOTE: {n_total} row(s) total, {overall['n_decisive']} with a decisive "
            f"(target_hit/stop_hit) outcome -- too few for meaningful statistics yet. "
            f"Numbers below are printed anyway (from whatever data IS available), but "
            f"treat them as provisional until more scheduled-job history accumulates."
        )
        lines.append("")

    lines.append("Overall:")
    lines.append(_fmt_hit_rate_row("all rows", overall))
    lines.append("")

    corr = report["quality_score_outcome_correlation"]
    if corr["pearson_r"] is not None:
        lines.append(
            f"quality_score vs. outcome (target_hit=+1, stop_hit=-1, timeouts excluded) "
            f"Pearson r = {corr['pearson_r']:.3f} (n={corr['n']})"
        )
    else:
        lines.append(
            f"quality_score vs. outcome correlation: n/a (only {corr['n']} decisive, "
            f"scored row(s) -- need >= {_MIN_PAIRS_FOR_CORRELATION})"
        )
    lines.append("")

    def _section(title: str, groups: dict[str, dict[str, Any]], order: list[str] | None = None) -> None:
        lines.append(f"By {title}:")
        if not groups:
            lines.append("  (no rows with a usable value)")
            lines.append("")
            return
        keys = order if order else sorted(groups.keys())
        for key in keys:
            if key in groups:
                lines.append(_fmt_hit_rate_row(key, groups[key]))
        lines.append("")

    _section(
        "quality_score band", report["by_quality_band"],
        order=[label for _, _, label in _QUALITY_BANDS],
    )
    _section("pattern", report["by_pattern"])
    _section("direction", report["by_direction"])

    lines.append("=" * 78)
    return "\n".join(lines)


def _load_all_rows(
    store: PatternScanHistoryStore,
    *,
    symbol: str | None = None,
    pattern: str | None = None,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    """Page through ``list_history`` until exhausted -- its own ``limit``
    defaults to 100 and is meant for UI pagination, not a "give me
    everything" call.
    """
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = store.list_history(symbol=symbol, pattern=pattern, limit=page_size, offset=offset)
        if not page:
            break
        rows.extend(page)
        log.debug("Loaded page of %d row(s) at offset %d", len(page), offset)
        if len(page) < page_size:
            break
        offset += page_size
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path", default="data/pattern_scan_history.db",
        help="Path to the pattern_scan_history SQLite DB (default: data/pattern_scan_history.db)",
    )
    parser.add_argument("--symbol", default=None, help="Filter to one symbol (passed through to list_history)")
    parser.add_argument("--pattern", default=None, help="Filter to one pattern type (passed through to list_history)")
    parser.add_argument(
        "--min-quality", type=float, default=None,
        help="Only include rows with quality_score >= this value (applied client-side)",
    )
    parser.add_argument("--output", default=None, help="Optional path to also write the report as JSON")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    store = PatternScanHistoryStore(db_path=args.db_path)
    rows = _load_all_rows(store, symbol=args.symbol, pattern=args.pattern)
    log.info("Loaded %d pattern-scan history row(s) from %s", len(rows), store.db_path)

    if args.min_quality is not None:
        before = len(rows)
        rows = [
            row for row in rows
            if row.get("quality_score") is not None and row["quality_score"] >= args.min_quality
        ]
        log.info("Applied --min-quality=%s: %d -> %d row(s)", args.min_quality, before, len(rows))

    report = build_report(rows)
    print(format_report_text(report))

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log.info("Wrote JSON report to %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
