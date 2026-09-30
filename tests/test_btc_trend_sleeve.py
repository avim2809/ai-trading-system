"""Tests for BtcTrendSleeve (src/firm/allocation/btc_trend.py) -- the
weekly BTC/USD trend-following satellite sleeve implementing pre-registered
candidate C1_btc_trend exactly (see scripts/alt_premia_preregistered_bars.py
and docs/edge_search_verdict_2026_09.md §3/§4B).

The parity test below cross-checks this sleeve's weekly on/off + target
weight against the evaluation harness's own C1 signal computation
(scripts/run_alt_premia_evaluation.py's c1_signal/c1_btc) on every Sunday
2015-2026, using the cached Tiingo BTC/USD daily series. It is skipped with a
clear reason if that cached parquet isn't available in the environment.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.allocation.btc_trend import BtcTrendSleeve
from firm.allocation.sleeves import Sleeve

_ROOT = Path(__file__).resolve().parents[1]
_PARQUET = Path(
    "/tmp/claude-0/-local-store-git-ai-trading-system/"
    "c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/data/tiingo_BTCUSD.parquet"
)


def _load_harness():
    for p in (str(_ROOT / "src"), str(_ROOT / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import alt_premia_preregistered_bars as prereg  # noqa: PLC0415
    import run_alt_premia_evaluation as harness  # noqa: PLC0415

    return prereg, harness


def _synthetic_btc_series(n_days: int = 400, start: str = "2025-01-06") -> pd.Series:
    """A deterministic, gap-free daily BTC-like series starting on a Monday,
    for tests that don't need the real cached data (fast, hermetic, and
    still exercises the same weekly-review/lookback/vol-window machinery)."""
    idx = pd.date_range(start, periods=n_days, freq="D")
    rng = np.random.default_rng(20260930)
    rets = rng.normal(0.0005, 0.03, size=n_days)
    prices = 30000.0 * np.cumprod(1 + rets)
    return pd.Series(prices, index=idx)


class TestBtcTrendSleeveBasics:
    def test_symbols_returns_single_configured_symbol(self):
        sleeve = BtcTrendSleeve(weight=0.05, symbol="BTC/USD")
        assert sleeve.symbols() == ["BTC/USD"]

    def test_is_instance_of_sleeve_abc(self):
        assert isinstance(BtcTrendSleeve(weight=0.05), Sleeve)

    def test_constructor_defaults_match_the_preregistered_rule(self):
        sleeve = BtcTrendSleeve(weight=0.05)
        assert sleeve.name == "btc_trend"
        assert sleeve.symbol == "BTC/USD"
        assert sleeve.lookback_days == 28
        assert sleeve.vol_window == 63
        assert sleeve.target_vol == 0.40
        assert sleeve.band_abs == 0.10

    def test_no_history_for_symbol_returns_flat(self):
        sleeve = BtcTrendSleeve(weight=0.05)
        assert sleeve.target_weights(asof=datetime(2026, 1, 5), history={}) == {}

    def test_empty_series_returns_flat(self):
        sleeve = BtcTrendSleeve(weight=0.05)
        result = sleeve.target_weights(
            asof=datetime(2026, 1, 5), history={"BTC/USD": pd.Series(dtype=float)}
        )
        assert result == {}

    def test_insufficient_history_before_first_sunday_returns_flat(self):
        """Fewer days than the lookback/vol windows require -- no crash, no
        false signal, just flat (cash)."""
        sleeve = BtcTrendSleeve(weight=0.05)
        series = _synthetic_btc_series(n_days=10)
        asof = series.index[-1]
        result = sleeve.target_weights(asof=asof, history={"BTC/USD": series})
        assert result == {}

    def test_history_with_no_sunday_yet_returns_flat(self):
        sleeve = BtcTrendSleeve(weight=0.05)
        # Monday through Saturday only -- six days, no Sunday close yet.
        idx = pd.date_range("2025-01-06", periods=6, freq="D")
        series = pd.Series(np.linspace(30000, 30500, 6), index=idx)
        result = sleeve.target_weights(asof=idx[-1], history={"BTC/USD": series})
        assert result == {}

    def test_target_weight_never_exceeds_one(self):
        sleeve = BtcTrendSleeve(weight=0.05)
        series = _synthetic_btc_series(n_days=400)
        for asof in series.index[::30]:
            result = sleeve.target_weights(asof=asof, history={"BTC/USD": series[series.index <= asof]})
            for w in result.values():
                assert 0.0 <= w <= 1.0

    def test_target_stays_flat_between_sundays_until_next_review(self):
        """The rule only re-evaluates at Sunday close; a mid-week asof must
        return exactly the same target as the most recent Sunday review,
        not silently drift with a fresh (wrong) lookback computed against a
        non-review day."""
        sleeve = BtcTrendSleeve(weight=0.05)
        series = _synthetic_btc_series(n_days=400)
        sundays = series.index[series.index.dayofweek == 6]
        # Pick a Sunday comfortably past the warm-up window.
        review = [d for d in sundays if series.index.get_loc(d) > 100][2]
        review_result = sleeve.target_weights(asof=review, history={"BTC/USD": series[series.index <= review]})

        for offset in range(1, 6):
            mid_week = review + pd.Timedelta(days=offset)
            mid_result = sleeve.target_weights(
                asof=mid_week, history={"BTC/USD": series[series.index <= mid_week]}
            )
            assert mid_result == review_result, f"offset={offset} diverged from the Sunday review"

    def test_duplicate_index_entries_are_deduplicated(self):
        sleeve = BtcTrendSleeve(weight=0.05)
        series = _synthetic_btc_series(n_days=200)
        dup = pd.concat([series, series.iloc[[-1]]])  # duplicate the last entry
        asof = series.index[-1]
        result_dup = sleeve.target_weights(asof=asof, history={"BTC/USD": dup})
        result_clean = sleeve.target_weights(asof=asof, history={"BTC/USD": series})
        assert result_dup == result_clean


class TestIsRebalanceDue:
    def test_never_rebalanced_is_always_due(self):
        sleeve = BtcTrendSleeve(weight=0.05)
        assert sleeve.is_rebalance_due(asof=datetime(2026, 1, 5), last_rebalance=None) is True

    def test_not_due_again_same_week(self):
        sleeve = BtcTrendSleeve(weight=0.05)
        # Sunday 2026-01-04 close reviewed; Monday 2026-01-05 already acted on it.
        assert sleeve.is_rebalance_due(
            asof=datetime(2026, 1, 5), last_rebalance=datetime(2026, 1, 5)
        ) is False
        assert sleeve.is_rebalance_due(
            asof=datetime(2026, 1, 8), last_rebalance=datetime(2026, 1, 5)
        ) is False

    def test_due_again_the_following_monday(self):
        sleeve = BtcTrendSleeve(weight=0.05)
        # last_rebalance acted on the 2026-01-04 Sunday; the next completed
        # Sunday is 2026-01-11, first visible on Monday 2026-01-12.
        assert sleeve.is_rebalance_due(
            asof=datetime(2026, 1, 12), last_rebalance=datetime(2026, 1, 5)
        ) is True

    def test_checking_on_a_sunday_itself_is_not_yet_due(self):
        """The Sunday's own bar isn't complete until its close -- a check
        made on the Sunday calendar date itself must use the *previous*
        week's review, not a not-yet-complete one."""
        sleeve = BtcTrendSleeve(weight=0.05)
        # Acted on through 2026-01-04 (a Sunday); checking again exactly one
        # week later, on Sunday 2026-01-11 itself, must not be due yet
        # (2026-01-11's own bar isn't complete).
        assert sleeve.is_rebalance_due(
            asof=datetime(2026, 1, 11), last_rebalance=datetime(2026, 1, 4)
        ) is False
        # But the very next day (Monday), it is.
        assert sleeve.is_rebalance_due(
            asof=datetime(2026, 1, 12), last_rebalance=datetime(2026, 1, 4)
        ) is True


@pytest.mark.skipif(
    not _PARQUET.exists(),
    reason=f"cached Tiingo BTC/USD parquet not available at {_PARQUET} in this environment",
)
class TestParityWithEvaluationHarness:
    """Cross-check against the pre-registered evaluation harness's own
    C1 signal computation (c1_signal) on every Sunday 2015-2026, using the
    real cached Tiingo BTC/USD daily series -- not a synthetic one."""

    def test_weekly_on_off_and_target_match_c1_signal_every_sunday(self):
        prereg, harness = _load_harness()
        raw = pd.read_parquet(_PARQUET).set_index("date")["close"]
        raw = raw[~raw.index.duplicated(keep="last")].sort_index()

        btc = raw.reindex(pd.date_range(raw.index.min(), raw.index.max(), freq="D")).ffill()
        rf_btc = pd.Series(0.0, index=btc.index)
        inp = harness.Inputs(
            dates=pd.DatetimeIndex([]),
            px=pd.DataFrame(),
            vix=pd.Series(dtype=float),
            vix3m=pd.Series(dtype=float),
            put=pd.Series(dtype=float),
            rf=pd.Series(dtype=float),
            btc=btc,
            rf_btc=rf_btc,
            fomc=pd.DatetimeIndex([]),
        )
        on, sig = harness.c1_signal(inp)

        sleeve = BtcTrendSleeve(weight=0.075)
        sunday_positions = np.flatnonzero(btc.index.dayofweek == 6)
        # Sanity: this really is a multi-year, weekly-cadence comparison.
        assert len(sunday_positions) > 500

        mismatches = []
        for i in sunday_positions:
            d = btc.index[i]
            history = {"BTC/USD": raw[raw.index <= d]}
            result = sleeve.target_weights(asof=d, history=history)

            expected_on = bool(on[i])
            expected_sigma = sig[i]
            if not expected_on or not np.isfinite(expected_sigma) or expected_sigma <= 0:
                expected_target = 0.0
            else:
                expected_target = min(1.0, sleeve.target_vol / expected_sigma)

            got = result.get("BTC/USD", 0.0)
            if abs(got - expected_target) > 1e-9:
                mismatches.append((str(d.date()), got, expected_target, expected_on, expected_sigma))

        assert not mismatches, (
            f"{len(mismatches)}/{len(sunday_positions)} Sunday mismatches "
            f"(first 5): {mismatches[:5]}"
        )

    # Note: this deliberately does NOT also try to match the harness's
    # simulate()-produced `held` array bar-for-bar. `held` embeds one extra
    # day of organic price drift within each non-trading week (simulate()
    # drifts the weight by that day's own return *before* checking
    # decide()), which is portfolio/execution state the harness tracks for
    # itself -- not part of what target_weights() (stateless, asof+history
    # only, per the Sleeve contract) is asked to reproduce. Tracking actual
    # held/drifted weight and applying band_abs against it is the
    # allocator's job (see this module's docstring "Design note").
