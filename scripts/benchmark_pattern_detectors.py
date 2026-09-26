#!/usr/bin/env python
"""Golden benchmark for the rule-based chart-pattern detectors
(firm.patterns.scanner.scan_symbol), answering the question no other test
or script in this repo measures: is pattern detection actually any good,
in precision/recall/PR-AUC/Brier-score terms, not just "does the wiring
run without raising"?

Runs every fixture in tests/pattern_fixtures.py (17 pattern names across
POSITIVE/AMBIGUOUS/NEGATIVE/BOUNDARY categories -- see that module's
docstring for the corpus's honestly-scoped coverage) plus a larger
synthetic-GBM-noise negative sample through the real scan_symbol pipeline,
and reports:

- Per-fixture pass/fail (did the best-quality-score match's pattern name
  land in that fixture's expected set, or correctly find nothing).
- Per-pattern-family breakdown (reversal/triangle/continuation/cup_handle).
- A corpus-wide binary classification report (firm.eval.classification):
  "should this fixture confirm *some* pattern at all" (y_true) vs. "did it"
  (y_pred) -- precision/recall/F1/confusion matrix.
- PR-AUC and Brier score treating quality_score/100 as the predicted
  probability of "a real pattern is here", against that same binary
  ground truth -- the first calibration-style measurement anywhere in
  this repo for detection quality itself (as opposed to portfolio
  returns, which is all firm.eval's other modules measure).

Examples:
    python scripts/benchmark_pattern_detectors.py
    python scripts/benchmark_pattern_detectors.py --output /tmp/bench.json
    python scripts/benchmark_pattern_detectors.py --n-negative-symbols 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
_TESTS = Path(__file__).resolve().parents[1] / "tests"
if _TESTS.exists() and str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from firm.eval.classification import (  # noqa: E402
    binary_classification_report,
    brier_score,
    pr_auc,
    reliability_diagram_bins,
)
from firm.patterns.ml.feature_engineering import PATTERN_FAMILY_MAP  # noqa: E402
from firm.patterns.scanner import scan_symbol  # noqa: E402

from pattern_fixtures import (  # noqa: E402
    AMBIGUOUS_FIXTURES,
    BOUNDARY_FIXTURES,
    CUP_HANDLE_AMBIGUOUS_EXPECTED,
    NEGATIVE_FIXTURES,
    POSITIVE_FIXTURES,
    PatternFixture,
    _cup_handle_frame,
    build_frame,
    synthetic_negative_frames,
)

log = logging.getLogger(__name__)


def _family_of(expected_patterns: frozenset[str]) -> str | None:
    families = {PATTERN_FAMILY_MAP.get(p) for p in expected_patterns}
    families.discard(None)
    return sorted(families)[0] if families else None


def run_fixture(fixture: PatternFixture, *, frame=None, min_score: float = 60.0) -> dict[str, Any]:
    """Scan one fixture and grade it against its expected_patterns.

    ``min_score`` defaults to 60.0 -- ``PatternRecognitionStrategy``'s own
    live default -- deliberately, not 0.0: this benchmark measures whether
    a fixture would actually produce a live signal, not just whether the
    underlying geometry is recognizable to an unfiltered detector call (a
    real, if perhaps surprising, finding from building this corpus: the
    bull_flag/bear_flag ambiguous fixtures score ~52, correctly geometry-
    matched but below the live quality bar -- reported as misses here on
    purpose, not tuned away, since that's a genuine live-threshold
    calibration question this benchmark exists to surface).

    A NEGATIVE/BOUNDARY fixture (``expected_patterns`` empty) is correct
    iff nothing confirms. A POSITIVE/AMBIGUOUS fixture is correct iff the
    best (top quality_score) match's pattern is in ``expected_patterns``.
    """
    df = frame if frame is not None else build_frame(fixture)
    matches = scan_symbol(df, min_score=min_score, zigzag_pct=0.03)
    best = matches[0] if matches else None
    should_detect = bool(fixture.expected_patterns)
    did_detect = best is not None
    if should_detect:
        correct = did_detect and best.pattern in fixture.expected_patterns
    else:
        correct = not did_detect
    return {
        "name": fixture.name,
        "category": fixture.category,
        "family": _family_of(fixture.expected_patterns),
        "expected_patterns": sorted(fixture.expected_patterns),
        "detected_pattern": best.pattern if best else None,
        "detected_quality_fraction": min(best.quality_score / 100.0, 1.0) if best else 0.0,
        "should_detect_something": should_detect,
        "did_detect_something": did_detect,
        "correct": correct,
    }


def run_benchmark(*, n_negative_symbols: int = 10, seed: int = 42, min_score: float = 60.0) -> list[dict[str, Any]]:
    results = [run_fixture(fx, min_score=min_score) for fx in POSITIVE_FIXTURES]
    results += [run_fixture(fx, min_score=min_score) for fx in AMBIGUOUS_FIXTURES]
    cup_handle_fixture = PatternFixture(
        "cup_handle_or_rounding_bottom", "ambiguous", [], 0, None, CUP_HANDLE_AMBIGUOUS_EXPECTED,
    )
    results.append(run_fixture(cup_handle_fixture, frame=_cup_handle_frame(), min_score=min_score))
    results += [run_fixture(fx, min_score=min_score) for fx in NEGATIVE_FIXTURES]
    results += [run_fixture(fx, min_score=min_score) for fx in BOUNDARY_FIXTURES]

    negative_fixture = PatternFixture("synthetic_noise", "negative", [], 0, None, frozenset())
    for i, frame in enumerate(synthetic_negative_frames(n_symbols=n_negative_symbols, seed=seed)):
        result = run_fixture(negative_fixture, frame=frame, min_score=min_score)
        result["name"] = f"synthetic_noise_{i:02d}"
        results.append(result)
    return results


def build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    n_correct = sum(1 for r in results if r["correct"])

    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({r["category"] for r in results}):
        rows = [r for r in results if r["category"] == category]
        by_category[category] = {
            "n": len(rows),
            "n_correct": sum(1 for r in rows if r["correct"]),
            "accuracy": sum(1 for r in rows if r["correct"]) / len(rows) if rows else None,
        }

    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({r["family"] for r in results if r["family"]}):
        rows = [r for r in results if r["family"] == family]
        by_family[family] = {
            "n": len(rows),
            "n_correct": sum(1 for r in rows if r["correct"]),
            "accuracy": sum(1 for r in rows if r["correct"]) / len(rows) if rows else None,
        }

    y_true = [1 if r["should_detect_something"] else 0 for r in results]
    y_pred = [1 if r["did_detect_something"] else 0 for r in results]
    y_score = [r["detected_quality_fraction"] for r in results]

    return {
        "overall": {"n": n, "n_correct": n_correct, "accuracy": n_correct / n if n else None},
        "by_category": by_category,
        "by_family": by_family,
        "classification": binary_classification_report(y_true, y_pred),
        "pr_auc": pr_auc(y_true, y_score),
        "brier_score": brier_score(y_true, y_score),
        "reliability": reliability_diagram_bins(y_true, y_score),
        "fixtures": results,
    }


def format_report_text(report: dict[str, Any]) -> str:
    lines = ["=" * 78, "PATTERN DETECTOR BENCHMARK", "=" * 78, ""]
    overall = report["overall"]
    lines.append(f"Overall: {overall['n_correct']}/{overall['n']} correct (accuracy={overall['accuracy']:.3f})")
    lines.append("")

    lines.append("By category:")
    for category, stats in report["by_category"].items():
        lines.append(f"  {category:12s} n={stats['n']:3d}  accuracy={stats['accuracy']:.3f}")
    lines.append("")

    lines.append("By pattern family:")
    for family, stats in report["by_family"].items():
        lines.append(f"  {family:14s} n={stats['n']:3d}  accuracy={stats['accuracy']:.3f}")
    lines.append("")

    c = report["classification"]
    lines.append(
        f"Classification (should detect SOME pattern vs. did detect SOME pattern): "
        f"precision={c['precision']:.3f} recall={c['recall']:.3f} f1={c['f1']:.3f}"
    )
    pr = report["pr_auc"]
    lines.append(f"PR-AUC: {pr:.3f}" if pr is not None else "PR-AUC: n/a (single class)")
    lines.append(f"Brier score: {report['brier_score']:.3f} (0.0=perfect, 0.25=uninformative on a balanced sample)")
    lines.append("")

    lines.append("Reliability (quality_score/100 vs. observed should-detect rate):")
    for band in report["reliability"]:
        mp = f"{band['mean_predicted']:.3f}" if band["mean_predicted"] is not None else "n/a"
        obs = f"{band['observed_rate']:.3f}" if band["observed_rate"] is not None else "n/a"
        lines.append(f"  {band['band']:>14s}  n={band['n']:3d}  mean_predicted={mp}  observed_rate={obs}")
    lines.append("")

    failures = [f for f in report["fixtures"] if not f["correct"]]
    if failures:
        lines.append(f"FAILURES ({len(failures)}):")
        for f in failures:
            lines.append(
                f"  {f['name']}: expected={f['expected_patterns']} got={f['detected_pattern']}"
            )
    else:
        lines.append("No failures.")

    lines.append("=" * 78)
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-negative-symbols", type=int, default=10, help="Synthetic-GBM-noise negative sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-score", type=float, default=60.0,
        help="quality_score threshold applied (matches PatternRecognitionStrategy's own "
             "live default of 60.0 -- see run_fixture's docstring for why this isn't 0.0)",
    )
    parser.add_argument("--output", default=None, help="Optional path to also write the report as JSON")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    results = run_benchmark(n_negative_symbols=args.n_negative_symbols, seed=args.seed, min_score=args.min_score)
    report = build_report(results)
    print(format_report_text(report))

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log.info("Wrote JSON report to %s", out_path)

    return 0 if report["overall"]["n_correct"] == report["overall"]["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
