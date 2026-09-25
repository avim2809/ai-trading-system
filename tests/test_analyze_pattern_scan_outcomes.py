"""Tests for scripts/analyze_pattern_scan_outcomes.py.

Loaded via importlib.util.spec_from_file_location (it's a standalone script
under scripts/, not part of the firm package) -- same convention as
tests/test_backfill_tiingo_prices.py.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

from firm.live.pattern_scan_history import PatternScanHistoryStore

_REPO = Path(__file__).resolve().parents[1]


def _load_analyzer():
    path = _REPO / "scripts" / "analyze_pattern_scan_outcomes.py"
    spec = importlib.util.spec_from_file_location("analyze_pattern_scan_outcomes", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def analyzer():
    return _load_analyzer()


# ---------------------------------------------------------------------------
# Fixture helpers: build rows the same shape build_report()/list_history()
# produce, without going through a full pattern scan.
# ---------------------------------------------------------------------------


def _row(
    *,
    symbol: str = "AAPL",
    pattern: str = "cup_handle",
    direction: str = "long",
    quality_score: float = 75.0,
    outcome: str | None = None,
) -> dict:
    return {
        "symbol": symbol, "asof": "2024-01-01T00:00:00", "pattern": pattern,
        "direction": direction, "confirmed": True, "confirm_index": 10,
        "entry": 100.0, "stop": 90.0, "target": 120.0, "fit_quality": 1.0,
        "geometry_tolerance_used": 1.0, "volume_ratio": 2.0, "duration_bars": 20,
        "follow_through_atr": 2.0, "risk_reward": 2.0, "quality_score": quality_score,
        "score_breakdown": {"total": quality_score}, "pivots": [], "meta": {},
        "outcome": outcome,
    }


def _seed_store(tmp_path, rows: list[dict]) -> PatternScanHistoryStore:
    """Insert ``rows`` (each from ``_row``) and apply their ``outcome`` via
    ``update_outcome`` afterwards, mirroring insert_matches + update_outcome
    -- the store's actual write lifecycle (insert first with no outcome,
    resolve later in place)."""
    store = PatternScanHistoryStore(db_path=str(tmp_path / "history.db"))
    store.insert_matches(rows, source="test")
    all_rows = store.list_history(limit=1000)
    # list_history returns newest-first; match back up by (symbol, pattern,
    # quality_score) since that's unique enough for these small fixtures.
    for row, inserted in zip(rows, reversed(all_rows)):
        if row["outcome"] is not None:
            store.update_outcome(inserted["id"], row["outcome"])
    return store


# ---------------------------------------------------------------------------
# compute_hit_rate
# ---------------------------------------------------------------------------


class TestComputeHitRate:
    def test_mixed_outcomes(self, analyzer):
        rows = [
            _row(outcome="target_hit"), _row(outcome="target_hit"),
            _row(outcome="stop_hit"), _row(outcome="timeout"), _row(outcome=None),
        ]
        stats = analyzer.compute_hit_rate(rows)
        assert stats == {
            "n_total": 5, "target_hit": 2, "stop_hit": 1, "timeout": 1,
            "pending": 1, "n_decisive": 3, "hit_rate": pytest.approx(2 / 3),
        }

    def test_all_pending_does_not_crash_and_reports_no_hit_rate(self, analyzer):
        rows = [_row(outcome=None), _row(outcome=None)]
        stats = analyzer.compute_hit_rate(rows)
        assert stats["hit_rate"] is None
        assert stats["pending"] == 2
        assert stats["n_decisive"] == 0

    def test_empty_rows_does_not_crash(self, analyzer):
        stats = analyzer.compute_hit_rate([])
        assert stats["n_total"] == 0
        assert stats["hit_rate"] is None


# ---------------------------------------------------------------------------
# group_hit_rates
# ---------------------------------------------------------------------------


class TestGroupHitRates:
    def test_groups_by_pattern(self, analyzer):
        rows = [
            _row(pattern="cup_handle", outcome="target_hit"),
            _row(pattern="cup_handle", outcome="stop_hit"),
            _row(pattern="double_top", outcome="target_hit"),
        ]
        grouped = analyzer.group_hit_rates(rows, lambda r: r["pattern"])
        assert set(grouped.keys()) == {"cup_handle", "double_top"}
        assert grouped["cup_handle"]["n_total"] == 2
        assert grouped["double_top"]["n_total"] == 1

    def test_single_group_when_all_rows_share_a_key(self, analyzer):
        rows = [_row(pattern="cup_handle", outcome="target_hit") for _ in range(3)]
        grouped = analyzer.group_hit_rates(rows, lambda r: r["pattern"])
        assert list(grouped.keys()) == ["cup_handle"]
        assert grouped["cup_handle"]["n_total"] == 3

    def test_rows_with_no_key_value_are_dropped_not_bucketed_as_none(self, analyzer):
        rows = [_row(quality_score=75.0), _row(quality_score=None)]
        grouped = analyzer.group_hit_rates(rows, lambda r: r.get("quality_score"))
        assert None not in grouped
        assert len(grouped) == 1

    def test_empty_rows_returns_empty_dict(self, analyzer):
        assert analyzer.group_hit_rates([], lambda r: r["pattern"]) == {}


# ---------------------------------------------------------------------------
# quality_score_outcome_correlation
# ---------------------------------------------------------------------------


class TestQualityScoreOutcomeCorrelation:
    def test_perfect_positive_correlation(self, analyzer):
        rows = [
            _row(quality_score=60.0, outcome="stop_hit"),
            _row(quality_score=70.0, outcome="stop_hit"),
            _row(quality_score=80.0, outcome="target_hit"),
            _row(quality_score=90.0, outcome="target_hit"),
            _row(quality_score=95.0, outcome="target_hit"),
        ]
        result = analyzer.quality_score_outcome_correlation(rows)
        assert result["n"] == 5
        # Monotonically increasing quality_score with stop_hit at the low
        # end and target_hit at the high end -- strongly positive, though
        # not exactly 1.0 since the outcome is binary (+-1) not continuous.
        assert result["pearson_r"] > 0.8

    def test_timeouts_and_pending_are_excluded_from_the_pair_count(self, analyzer):
        decisive = [
            _row(quality_score=60.0, outcome="stop_hit"),
            _row(quality_score=70.0, outcome="stop_hit"),
            _row(quality_score=80.0, outcome="target_hit"),
            _row(quality_score=90.0, outcome="target_hit"),
            _row(quality_score=95.0, outcome="target_hit"),
        ]
        noise = [_row(quality_score=99.0, outcome="timeout"), _row(quality_score=1.0, outcome=None)]
        result_with_noise = analyzer.quality_score_outcome_correlation(decisive + noise)
        result_without_noise = analyzer.quality_score_outcome_correlation(decisive)
        assert result_with_noise["n"] == result_without_noise["n"] == 5
        assert result_with_noise["pearson_r"] == pytest.approx(result_without_noise["pearson_r"])

    def test_below_minimum_pairs_returns_none_not_a_misleading_number(self, analyzer):
        rows = [_row(quality_score=80.0, outcome="target_hit"), _row(quality_score=60.0, outcome="stop_hit")]
        result = analyzer.quality_score_outcome_correlation(rows)
        assert result["pearson_r"] is None
        assert result["n"] == 2

    def test_no_decisive_rows_returns_none_without_crashing(self, analyzer):
        rows = [_row(outcome=None), _row(outcome="timeout")]
        result = analyzer.quality_score_outcome_correlation(rows)
        assert result == {"n": 0, "pearson_r": None}

    def test_zero_variance_quality_scores_returns_none(self, analyzer):
        rows = [
            _row(quality_score=75.0, outcome="target_hit"),
            _row(quality_score=75.0, outcome="stop_hit"),
            _row(quality_score=75.0, outcome="target_hit"),
            _row(quality_score=75.0, outcome="stop_hit"),
            _row(quality_score=75.0, outcome="target_hit"),
        ]
        result = analyzer.quality_score_outcome_correlation(rows)
        assert result["pearson_r"] is None


# ---------------------------------------------------------------------------
# build_report / format_report_text: end-to-end, including the near-empty
# and all-pending cases the real on-disk DB is in right now.
# ---------------------------------------------------------------------------


class TestBuildReportEndToEnd:
    def test_all_pending_report_has_no_nan_and_prints_a_clear_note(self, analyzer):
        rows = [_row(pattern="cup_handle", quality_score=65.0, outcome=None) for _ in range(3)]
        report = analyzer.build_report(rows)
        assert report["n_total"] == 3
        assert report["overall"]["hit_rate"] is None
        text = analyzer.format_report_text(report)
        assert "too few for meaningful statistics" in text
        assert "nan" not in text.lower()

    def test_empty_report_does_not_crash(self, analyzer):
        report = analyzer.build_report([])
        text = analyzer.format_report_text(report)
        assert "No pattern-scan history rows" in text

    def test_mixed_outcomes_report_includes_bands_pattern_and_direction(self, analyzer):
        rows = [
            _row(pattern="cup_handle", direction="long", quality_score=65.0, outcome="target_hit"),
            _row(pattern="cup_handle", direction="long", quality_score=62.0, outcome="stop_hit"),
            _row(pattern="double_top", direction="short", quality_score=85.0, outcome="target_hit"),
        ]
        report = analyzer.build_report(rows)
        assert set(report["by_pattern"].keys()) == {"cup_handle", "double_top"}
        assert set(report["by_direction"].keys()) == {"long", "short"}
        assert set(report["by_quality_band"].keys()) == {"60-70", "80-90"}
        text = analyzer.format_report_text(report)
        assert "cup_handle" in text
        assert "double_top" in text

    def test_single_pattern_and_single_quality_band_group(self, analyzer):
        rows = [_row(pattern="cup_handle", quality_score=61.0, outcome="target_hit") for _ in range(4)]
        report = analyzer.build_report(rows)
        assert list(report["by_pattern"].keys()) == ["cup_handle"]
        assert list(report["by_quality_band"].keys()) == ["60-70"]
        # Must not raise formatting a single-group report.
        analyzer.format_report_text(report)


# ---------------------------------------------------------------------------
# _quality_band edges
# ---------------------------------------------------------------------------


class TestQualityBand:
    @pytest.mark.parametrize(
        "score,expected",
        [(0.0, "<60"), (59.9, "<60"), (60.0, "60-70"), (69.99, "60-70"),
         (70.0, "70-80"), (85.0, "80-90"), (90.0, "90-100"), (100.0, "90-100"), (150.0, "90-100")],
    )
    def test_band_boundaries(self, analyzer, score, expected):
        assert analyzer._quality_band(score) == expected

    def test_none_quality_score_has_no_band(self, analyzer):
        assert analyzer._quality_band(None) is None


# ---------------------------------------------------------------------------
# _load_all_rows: real PatternScanHistoryStore, pagination + filters.
# ---------------------------------------------------------------------------


class TestLoadAllRows:
    def test_loads_more_rows_than_a_single_default_page(self, analyzer, tmp_path):
        rows = [_row(symbol="AAPL", pattern="cup_handle") for _ in range(5)]
        store = _seed_store(tmp_path, rows)
        loaded = analyzer._load_all_rows(store, page_size=2)
        assert len(loaded) == 5

    def test_symbol_and_pattern_filters_pass_through(self, analyzer, tmp_path):
        rows = [
            _row(symbol="AAPL", pattern="cup_handle"),
            _row(symbol="MSFT", pattern="double_top"),
        ]
        store = _seed_store(tmp_path, rows)
        loaded = analyzer._load_all_rows(store, symbol="AAPL")
        assert len(loaded) == 1
        assert loaded[0]["symbol"] == "AAPL"

    def test_empty_store_returns_empty_list(self, analyzer, tmp_path):
        store = PatternScanHistoryStore(db_path=str(tmp_path / "empty.db"))
        assert analyzer._load_all_rows(store) == []


# ---------------------------------------------------------------------------
# Real store, mixed outcomes end-to-end via insert_matches + update_outcome
# (per the task's suggested seeding path).
# ---------------------------------------------------------------------------


class TestRealStoreIntegration:
    def test_seeded_store_flows_through_build_report(self, analyzer, tmp_path):
        rows = [
            _row(symbol="AAPL", pattern="cup_handle", direction="long", quality_score=65.0, outcome="target_hit"),
            _row(symbol="MSFT", pattern="cup_handle", direction="long", quality_score=61.0, outcome="stop_hit"),
            _row(symbol="NVDA", pattern="double_top", direction="short", quality_score=88.0, outcome=None),
        ]
        store = _seed_store(tmp_path, rows)
        loaded = analyzer._load_all_rows(store)
        assert len(loaded) == 3
        report = analyzer.build_report(loaded)
        assert report["overall"]["n_decisive"] == 2
        assert report["overall"]["pending"] == 1
        assert not any(
            isinstance(v, float) and math.isnan(v)
            for group in report["by_pattern"].values()
            for v in group.values()
            if isinstance(v, float)
        )
