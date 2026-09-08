"""Tests for the pattern-recognition ML confirmation layer (firm.patterns.ml).

Phase 4 of docs/pattern_recognition_plan.md, scoped to just the XGBoost
confirmation classifier (no CNN/GAF validator, no PPO sizer -- see the task
report for this initiative). Mirrors tests/test_patterns.py's style: hand-
built OHLCV fixtures via linear interpolation between chosen anchor points,
plain pytest functions/classes, no mocking framework.

``xgboost`` is an optional dependency (the ``patterns_ml`` extra in
pyproject.toml) -- ``feature_engineering``/``labeling`` tests always run;
anything touching ``xgb_classifier`` training/prediction is gated behind
``requires_xgboost`` so this file degrades gracefully (skips, doesn't fail)
on an environment without the extra installed.

The end-to-end smoke test (``TestEndToEndSmoke``) reuses eight of
tests/test_patterns.py's exact anchor fixtures (already proven to each
confirm exactly one pattern) and extends each one with a small, hand-
designed forward continuation engineered to deterministically realize one
of the three triple-barrier outcomes (favorable/adverse/timeout) -- see
``_build_labeled_dataset``. This gives a real ``scan_symbol`` -> real
``build_features`` -> real ``label_triple_barrier`` -> real
``xgb_classifier.train``/``predict_proba`` pipeline run with guaranteed,
deterministic class diversity, rather than hoping organic random-walk data
happens to produce enough labeled diversity by chance. A second smoke test
separately throws ``firm.data.synthetic.make_synthetic_prices`` (organic
GBM noise) at the same scan/feature/label functions purely as a
never-raises robustness check -- it deliberately does not assert a minimum
match count, since raw pattern detection on noise is rare by design (see
docs/pattern_recognition_plan.md's discussion of quality-scoring).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from firm.data.synthetic import make_synthetic_prices
from firm.patterns.extrema import Pivot
from firm.patterns.match import PatternMatch
from firm.patterns.ml import xgb_classifier
from firm.patterns.ml.feature_engineering import (
    PATTERN_FAMILIES,
    PATTERN_NAMES,
    build_feature_frame,
    build_features,
)
from firm.patterns.ml.labeling import DEFAULT_TIMEOUT_BARS, label_matches, label_triple_barrier
from firm.patterns.scanner import scan_symbol

try:
    import xgboost  # noqa: F401
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False

requires_xgboost = pytest.mark.skipif(
    not _XGBOOST_AVAILABLE, reason="xgboost not installed (optional `patterns_ml` extra)"
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------


def _make_match(**overrides) -> PatternMatch:
    """A fully-populated (post-scanner) PatternMatch with sane defaults,
    overridable per test. Mirrors what scan_symbol._score_and_finalize
    actually fills in, so build_features/label_triple_barrier see realistic
    shapes rather than a bare detector-only match.
    """
    defaults = dict(
        pattern="bull_flag",
        direction="long",
        pivots=(Pivot(4, 90.0, "trough"), Pivot(10, 120.0, "peak")),
        confirm_index=30,
        entry=118.0,
        stop=114.0,
        target=148.0,
        fit_quality=0.9,
        geometry_tolerance_used=0.8,
        volume_ratio=2.0,
        duration_bars=26,
        follow_through_atr=1.5,
        risk_reward=4.0,
        quality_score=73.0,
        score_breakdown={
            "geometry": 28.0, "trendline_fit": 18.0, "volume_confirmation": 20.0,
            "duration": 9.0, "follow_through": 7.0, "total": 82.0,
        },
    )
    defaults.update(overrides)
    return PatternMatch(**defaults)


def _ohlcv_frame(closes: list[float], *, wick: float = 0.5) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "high": closes + wick,
        "low": closes - wick,
        "close": closes,
        "volume": np.full(len(closes), 1_000_000.0),
    })


def _ohlcv(anchors: list[tuple[int, float]], total_bars: int, *, wick: float = 0.002):
    """Same construction as tests/test_patterns.py's private helper of the
    same name -- kept as an independent copy since test files in this repo
    don't share fixtures via import (see that file).
    """
    idxs = [a[0] for a in anchors]
    prices = [a[1] for a in anchors]
    x = np.arange(total_bars)
    close = np.interp(x, idxs, prices)
    high = close * (1 + wick)
    low = close * (1 - wick)
    volume = np.full(total_bars, 1_000_000.0)
    return high, low, close, volume


def _spike(volume: np.ndarray, at: int, multiple: float = 2.5) -> np.ndarray:
    volume = volume.copy()
    volume[at] = volume[at] * multiple
    return volume


def _frame(anchors: list[tuple[int, float]], total_bars: int, *, spike_at: int | None = None) -> pd.DataFrame:
    high, low, close, volume = _ohlcv(anchors, total_bars)
    if spike_at is not None:
        volume = _spike(volume, spike_at)
    return pd.DataFrame({"high": high, "low": low, "close": close, "volume": volume})


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------


class TestBuildFeatures:
    def test_returns_all_finite_floats(self):
        feats = build_features(_make_match())
        assert feats
        assert all(isinstance(v, float) for v in feats.values())
        assert np.isfinite(np.array(list(feats.values()))).all()

    def test_direction_sign(self):
        assert build_features(_make_match(direction="long"))["direction_sign"] == 1.0
        assert build_features(_make_match(direction="short"))["direction_sign"] == -1.0

    def test_pattern_and_family_one_hot_are_exclusive(self):
        feats = build_features(_make_match(pattern="cup_handle"))
        assert sum(feats[f"pattern_{p}"] for p in PATTERN_NAMES) == 1.0
        assert sum(feats[f"family_{f}"] for f in PATTERN_FAMILIES) == 1.0
        assert feats["pattern_cup_handle"] == 1.0
        assert feats["family_cup_handle"] == 1.0
        assert feats["pattern_bull_flag"] == 0.0
        assert feats["family_reversal"] == 0.0

    def test_every_pattern_name_maps_to_exactly_one_family(self):
        for pattern in PATTERN_NAMES:
            feats = build_features(_make_match(pattern=pattern))
            assert sum(feats[f"family_{f}"] for f in PATTERN_FAMILIES) == 1.0
            assert feats[f"pattern_{pattern}"] == 1.0

    def test_unrecognized_pattern_degrades_gracefully(self, caplog):
        with caplog.at_level(logging.WARNING):
            feats = build_features(_make_match(pattern="totally_new_pattern_xyz"))
        assert sum(feats[f"pattern_{p}"] for p in PATTERN_NAMES) == 0.0
        assert sum(feats[f"family_{f}"] for f in PATTERN_FAMILIES) == 0.0
        assert "unrecognized pattern" in caplog.text

    def test_missing_volume_ratio_is_filled_not_nan(self):
        feats = build_features(_make_match(volume_ratio=float("nan")))
        assert feats["volume_ratio_missing"] == 1.0
        assert feats["volume_ratio"] == 1.0  # neutral fill, not NaN
        assert np.isfinite(feats["volume_ratio"])

    def test_present_volume_ratio_passes_through(self):
        feats = build_features(_make_match(volume_ratio=2.5))
        assert feats["volume_ratio_missing"] == 0.0
        assert feats["volume_ratio"] == 2.5

    def test_score_breakdown_is_unpacked(self):
        feats = build_features(_make_match())
        assert feats["score_geometry"] == 28.0
        assert feats["score_total"] == 82.0

    def test_stop_target_distance_pct_are_scale_invariant(self):
        near = build_features(_make_match(entry=100.0, stop=95.0, target=110.0))
        far = build_features(_make_match(entry=1000.0, stop=950.0, target=1100.0))
        assert near["stop_distance_pct"] == pytest.approx(far["stop_distance_pct"])
        assert near["target_distance_pct"] == pytest.approx(far["target_distance_pct"])

    def test_no_ohlcv_context_is_zeroed_but_present(self):
        feats = build_features(_make_match(), None)
        assert feats["ohlcv_context_available"] == 0.0
        assert feats["pre_pattern_return"] == 0.0
        assert feats["pre_pattern_volatility"] == 0.0
        assert feats["atr_pct"] == 0.0

    def test_ohlcv_context_computed_when_available(self):
        match = _make_match(confirm_index=20)
        closes = 100.0 + 2.0 * np.arange(25)
        ohlcv = pd.DataFrame({
            "high": closes + 0.5, "low": closes - 0.5, "close": closes,
            "volume": np.full(25, 1_000_000.0),
        })
        feats = build_features(match, ohlcv)
        assert feats["ohlcv_context_available"] == 1.0
        expected_return = closes[20] / closes[0] - 1.0
        assert feats["pre_pattern_return"] == pytest.approx(expected_return)
        assert feats["pre_pattern_volatility"] >= 0.0
        assert feats["atr_pct"] >= 0.0
        assert np.isfinite(feats["atr_pct"])

    def test_ohlcv_context_out_of_range_confirm_index_is_safe(self):
        match = _make_match(confirm_index=999)
        ohlcv = pd.DataFrame({"high": [1.0, 2.0], "low": [1.0, 2.0], "close": [1.0, 2.0], "volume": [1.0, 1.0]})
        feats = build_features(match, ohlcv)
        assert feats["ohlcv_context_available"] == 0.0

    def test_build_feature_frame_batch(self):
        matches = [_make_match(pattern="bull_flag"), _make_match(pattern="cup_handle", direction="short")]
        frame = build_feature_frame(matches)
        assert len(frame) == 2
        assert frame.loc[0, "pattern_bull_flag"] == 1.0
        assert frame.loc[1, "pattern_cup_handle"] == 1.0
        assert not frame.isna().any().any()

    def test_build_feature_frame_with_per_match_ohlcv_pairs(self):
        ohlcv = _ohlcv_frame([100, 101, 102, 103, 104])
        pairs = [(_make_match(confirm_index=3), ohlcv)]
        frame = build_feature_frame(pairs)
        assert frame.loc[0, "ohlcv_context_available"] == 1.0

    def test_build_feature_frame_empty_input(self):
        assert build_feature_frame([]).empty


# ---------------------------------------------------------------------------
# label_triple_barrier
# ---------------------------------------------------------------------------


class TestLabelTripleBarrier:
    def test_long_target_hit_first(self):
        match = _make_match(direction="long", confirm_index=2, entry=100.0, stop=95.0, target=110.0)
        ohlcv = _ohlcv_frame([100, 101, 100, 103, 108, 111, 120])
        assert label_triple_barrier(match, ohlcv, timeout_bars=20) == 1

    def test_long_stop_hit_first(self):
        match = _make_match(direction="long", confirm_index=2, entry=100.0, stop=95.0, target=110.0)
        ohlcv = _ohlcv_frame([100, 101, 100, 98, 94, 108])
        assert label_triple_barrier(match, ohlcv, timeout_bars=20) == -1

    def test_neither_hit_within_timeout_is_zero(self):
        match = _make_match(direction="long", confirm_index=2, entry=100.0, stop=90.0, target=120.0)
        ohlcv = _ohlcv_frame([100, 101, 100, 102, 103, 101, 104, 102, 103, 101, 104])
        assert label_triple_barrier(match, ohlcv, timeout_bars=5) == 0

    def test_short_direction_target_hit_first(self):
        match = _make_match(direction="short", confirm_index=2, entry=100.0, stop=105.0, target=90.0)
        ohlcv = _ohlcv_frame([100, 99, 100, 97, 92, 85])
        assert label_triple_barrier(match, ohlcv, timeout_bars=20) == 1

    def test_short_direction_stop_hit_first(self):
        match = _make_match(direction="short", confirm_index=2, entry=100.0, stop=105.0, target=90.0)
        ohlcv = _ohlcv_frame([100, 99, 100, 103, 107, 80])
        assert label_triple_barrier(match, ohlcv, timeout_bars=20) == -1

    def test_same_bar_collision_prefers_stop(self):
        match = _make_match(direction="long", confirm_index=0, entry=100.0, stop=95.0, target=110.0)
        ohlcv = pd.DataFrame({
            "high": [100.5, 115.0], "low": [99.5, 90.0], "close": [100.0, 102.0], "volume": [1e6, 1e6],
        })
        assert label_triple_barrier(match, ohlcv, timeout_bars=20) == -1

    def test_unconfirmed_match_raises(self):
        match = _make_match(confirm_index=-1)
        with pytest.raises(ValueError):
            label_triple_barrier(match, _ohlcv_frame([100, 101, 102]))

    def test_no_bars_after_confirm_index_returns_zero(self):
        match = _make_match(confirm_index=2, direction="long", stop=90.0, target=120.0)
        ohlcv = _ohlcv_frame([100, 101, 102])  # confirm_index is the last bar
        assert label_triple_barrier(match, ohlcv, timeout_bars=20) == 0

    def test_default_timeout_is_20_bars(self):
        assert DEFAULT_TIMEOUT_BARS == 20

    def test_label_matches_batch_skips_unconfirmed(self):
        confirmed = _make_match(confirm_index=2, direction="long", stop=95.0, target=110.0)
        unconfirmed = _make_match(confirm_index=-1)
        ohlcv = _ohlcv_frame([100, 101, 100, 103, 108, 111])
        labels = label_matches([confirmed, unconfirmed], ohlcv, timeout_bars=20)
        assert labels.tolist() == [1]


# ---------------------------------------------------------------------------
# xgb_classifier wrapper (requires the optional xgboost extra)
# ---------------------------------------------------------------------------


def _toy_xy(n: int = 30, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.rand(n, 5), columns=[f"f{i}" for i in range(5)])
    y = rng.randint(-1, 2, size=n)  # {-1, 0, 1}
    return X, y


@requires_xgboost
class TestXGBClassifierWrapper:
    def test_train_predict_proba_bounds_and_shape(self):
        X, y = _toy_xy()
        model = xgb_classifier.train(X, y)
        proba = xgb_classifier.predict_proba(model, X)
        assert proba.shape == (len(X), 3)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_label_matches_argmax_of_proba(self):
        X, y = _toy_xy()
        model = xgb_classifier.train(X, y)
        proba = xgb_classifier.predict_proba(model, X)
        expected = np.array([xgb_classifier.LABELS[i] for i in np.argmax(proba, axis=1)])
        np.testing.assert_array_equal(xgb_classifier.predict_label(model, X), expected)

    def test_predict_proba_handles_class_missing_from_training_data(self):
        # y only ever contains -1/1 -- the model never sees a 0 (timeout)
        # label. predict_proba must still return a full 3-column result.
        X = pd.DataFrame(np.random.RandomState(1).rand(10, 4))
        y = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1, 1])
        model = xgb_classifier.train(X, y)
        proba = xgb_classifier.predict_proba(model, X)
        assert proba.shape == (10, 3)
        assert np.allclose(proba[:, 1], 0.0)  # label 0's column, untrained

    def test_train_rejects_labels_outside_domain(self):
        X = pd.DataFrame(np.random.RandomState(2).rand(5, 3))
        y = np.array([0, 1, 2, 0, 1])  # 2 is not a valid triple-barrier label
        with pytest.raises(ValueError):
            xgb_classifier.train(X, y)

    def test_train_rejects_empty_input(self):
        X = pd.DataFrame(np.zeros((0, 3)))
        with pytest.raises(ValueError):
            xgb_classifier.train(X, np.array([], dtype=int))

    def test_save_load_roundtrip(self, tmp_path):
        X, y = _toy_xy()
        model = xgb_classifier.train(X, y)
        path = tmp_path / "nested" / "model.pkl"
        xgb_classifier.save(model, path)
        assert path.exists()
        loaded = xgb_classifier.load(path)
        np.testing.assert_allclose(
            xgb_classifier.predict_proba(model, X), xgb_classifier.predict_proba(loaded, X)
        )


# ---------------------------------------------------------------------------
# End-to-end smoke: scan -> features -> triple-barrier labels -> train -> predict
# ---------------------------------------------------------------------------


def _extend_forward(df: pd.DataFrame, end_price: float, *, timeout_bars: int, ramp_bars: int = 6,
                     buffer_bars: int = 5, wick: float = 0.002) -> pd.DataFrame:
    """Append forward bars that ramp from ``df``'s last close to
    ``end_price`` within ``ramp_bars``, then hold flat -- guarantees the
    barrier crossing lands well inside the timeout window rather than at
    its edge (a purely linear ramp timed to *just* reach the barrier at the
    edge of the window is a common off-by-one trap here).
    """
    last_close = float(df["close"].iloc[-1])
    ramp = np.linspace(last_close, end_price, ramp_bars + 1)[1:]
    hold = np.full(max(timeout_bars + buffer_bars - ramp_bars, 0), end_price)
    fwd_close = np.concatenate([ramp, hold])
    fwd = pd.DataFrame({
        "high": fwd_close * (1 + wick), "low": fwd_close * (1 - wick),
        "close": fwd_close, "volume": np.full(len(fwd_close), 1_000_000.0),
    })
    return pd.concat([df, fwd], ignore_index=True)


def _flat_forward(df: pd.DataFrame, price: float, *, timeout_bars: int, buffer_bars: int = 5,
                   wick: float = 0.002) -> pd.DataFrame:
    """Append forward bars holding at a constant ``price`` -- used for the
    deliberate "timeout" (neither barrier touched) fixtures.
    """
    n = timeout_bars + buffer_bars
    fwd_close = np.full(n, price)
    fwd = pd.DataFrame({
        "high": fwd_close * (1 + wick), "low": fwd_close * (1 - wick),
        "close": fwd_close, "volume": np.full(n, 1_000_000.0),
    })
    return pd.concat([df, fwd], ignore_index=True)


# Reuses eight of tests/test_patterns.py's exact anchor fixtures (each
# independently proven there to yield one confirmed match) -- see that file
# for the geometric rationale behind each anchor list. Each is paired with
# the triple-barrier outcome its forward continuation is engineered to
# realize, giving 3 favorable (+1) / 3 adverse (-1) / 2 timeout (0) rows.
_PATTERN_FIXTURES: dict[str, tuple[list[tuple[int, float]], int, int, str]] = {
    "bull_flag": (
        [(0, 100.0), (4, 90.0), (10, 120.0), (14, 117.0), (18, 119.0),
         (22, 116.0), (26, 118.0), (30, 130.0)],
        31, 30, "favorable",
    ),
    "bear_flag": (
        [(0, 100.0), (4, 110.0), (10, 80.0), (14, 83.0), (18, 81.0),
         (22, 84.0), (26, 82.0), (30, 70.0)],
        31, 30, "favorable",
    ),
    "triple_bottom": (
        [(0, 130.0), (8, 100.0), (16, 118.0), (24, 99.0), (32, 117.0), (40, 100.0), (55, 125.0)],
        56, 55, "favorable",
    ),
    "head_shoulders_top": (
        [(0, 100.0), (10, 130.0), (20, 110.0), (30, 140.0), (40, 112.0), (50, 131.0), (70, 100.0)],
        71, 70, "adverse",
    ),
    "inverse_head_shoulders": (
        [(0, 100.0), (10, 70.0), (20, 90.0), (30, 60.0), (40, 88.0), (50, 70.0), (70, 100.0)],
        71, 70, "adverse",
    ),
    "triple_top": (
        [(0, 90.0), (8, 120.0), (16, 102.0), (24, 121.0), (32, 103.0), (40, 120.0), (55, 90.0)],
        56, 55, "adverse",
    ),
    "double_top": (
        [(0, 90.0), (10, 120.0), (20, 100.0), (30, 121.0), (45, 85.0)],
        46, 45, "timeout",
    ),
    "double_bottom": (
        [(0, 120.0), (10, 90.0), (20, 110.0), (30, 89.0), (45, 130.0)],
        46, 45, "timeout",
    ),
}


def _build_labeled_dataset(timeout_bars: int = DEFAULT_TIMEOUT_BARS) -> tuple[pd.DataFrame, np.ndarray]:
    """Scan each fixture for its confirmed match, build its feature row, and
    label it against a hand-designed forward continuation that
    deterministically realizes the fixture's intended outcome. Returns
    ``(X, y)`` spanning all three triple-barrier labels.
    """
    rows: list[dict] = []
    labels: list[int] = []
    for name, (anchors, total_bars, spike_at, outcome) in _PATTERN_FIXTURES.items():
        df = _frame(anchors, total_bars, spike_at=spike_at)
        matches = scan_symbol(df, min_score=0.0, zigzag_pct=0.03)
        assert matches, f"fixture {name!r} produced no confirmed match"
        match = matches[0]
        is_long = match.direction == "long"

        if outcome == "favorable":
            end_price = match.target * (1.05 if is_long else 0.95)
            extended = _extend_forward(df, end_price, timeout_bars=timeout_bars)
        elif outcome == "adverse":
            end_price = match.stop * (0.95 if is_long else 1.05)
            extended = _extend_forward(df, end_price, timeout_bars=timeout_bars)
        else:
            lo, hi = sorted([match.stop, match.target])
            extended = _flat_forward(df, (lo + hi) / 2.0, timeout_bars=timeout_bars)

        label = label_triple_barrier(match, extended, timeout_bars=timeout_bars)
        rows.append(build_features(match, df))
        labels.append(label)

    return pd.DataFrame(rows), np.array(labels, dtype=int)


class TestBuildLabeledDataset:
    """Sanity-checks the test helper itself: the fixture/outcome table above
    must actually realize what it claims before it's trusted as ground
    truth for the training smoke test below.
    """

    def test_dataset_has_all_three_labels(self):
        X, y = _build_labeled_dataset()
        assert len(X) == 8
        counts = pd.Series(y).value_counts()
        assert set(counts.index) == {-1, 0, 1}
        assert counts[1] == 3
        assert counts[-1] == 3
        assert counts[0] == 2

    def test_no_nan_features(self):
        X, _ = _build_labeled_dataset()
        assert not X.isna().any().any()


@requires_xgboost
class TestEndToEndSmoke:
    def test_scan_label_train_predict_pipeline(self):
        X, y = _build_labeled_dataset()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        model = xgb_classifier.train(X_train, y_train)

        proba = xgb_classifier.predict_proba(model, X_test)
        assert proba.shape == (len(X_test), 3)
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

        pred = xgb_classifier.predict_label(model, X_test)
        assert set(pred.tolist()) <= {-1, 0, 1}
        acc = accuracy_score(y_test, pred)
        assert 0.0 <= acc <= 1.0  # must be computable -- this is a mechanics
        # smoke test on an 8-row toy dataset, not a claim about real skill.

    def test_pipeline_runs_on_organic_synthetic_data_without_raising(self):
        """Robustness check with firm.data.synthetic.make_synthetic_prices.

        Organic GBM noise isn't guaranteed to contain any confirmed pattern
        (raw pattern detection on noise is rare by design -- see
        docs/pattern_recognition_plan.md), so this deliberately does not
        assert a minimum match count -- only that scanning, feature-
        building, and labeling never raise on realistic, messy data.
        """
        panel = make_synthetic_prices(["AAPL", "MSFT", "GOOG"], n_days=300, seed=7)
        for _symbol, sym_df in panel.groupby("symbol"):
            sym_df = sym_df.sort_values("date")
            ohlcv = sym_df[["high", "low", "close", "volume"]].reset_index(drop=True)
            matches = scan_symbol(ohlcv, min_score=0.0)
            for match in matches:
                if not match.confirmed:
                    continue
                feats = build_features(match, ohlcv)
                assert np.isfinite(np.array(list(feats.values()))).all()
                label = label_triple_barrier(match, ohlcv, timeout_bars=20)
                assert label in (-1, 0, 1)
