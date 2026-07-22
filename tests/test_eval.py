"""Tests for eval metrics, reports, plots, and portfolio attribution."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from firm.contracts.models import PortfolioSnapshot
from firm.eval import metrics
from firm.eval.plots import (
    plot_drawdown,
    plot_equity_curve,
    plot_exposure,
    plot_monthly_returns,
    plot_rolling_sharpe,
    plot_strategy_attribution,
    save_all_plots,
)
from firm.eval.reports import BacktestReport
from firm.portfolio.attribution import PerformanceAttribution


# ======================================================================
# Helpers
# ======================================================================

def _daily_returns(n: int = 252, mean: float = 0.0004, std: float = 0.01, seed: int = 42):
    """Deterministic daily return series."""
    rng = np.random.default_rng(seed)
    vals = rng.normal(mean, std, n)
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(vals, index=dates, name="returns")


def _constant_returns(value: float = 0.001, n: int = 252):
    """Every day the same return — useful for analytical checks."""
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.Series(value, index=dates, name="returns")


# ======================================================================
# 1. metrics.py
# ======================================================================

class TestTotalReturn:
    def test_positive(self):
        r = _constant_returns(0.01, 10)
        expected = (1.01**10) - 1
        assert math.isclose(metrics.total_return(r), expected, rel_tol=1e-9)

    def test_zero(self):
        r = _constant_returns(0.0, 10)
        assert metrics.total_return(r) == 0.0

    def test_empty(self):
        assert metrics.total_return(pd.Series(dtype=float)) == 0.0


class TestCAGR:
    def test_positive(self):
        r = _constant_returns(0.001, 252)
        tr = metrics.total_return(r)
        expected = (1 + tr) ** (252 / 252) - 1
        assert math.isclose(metrics.cagr(r), expected, rel_tol=1e-9)

    def test_empty(self):
        assert metrics.cagr(pd.Series(dtype=float)) == 0.0


class TestAnnualizedVolatility:
    def test_constant(self):
        r = _constant_returns(0.001, 252)
        assert metrics.annualized_volatility(r) == 0.0

    def test_known(self):
        r = _daily_returns(252)
        ann_vol = metrics.annualized_volatility(r)
        assert ann_vol > 0

    def test_too_few(self):
        r = pd.Series([0.01])
        assert metrics.annualized_volatility(r) == 0.0


class TestSharpeRatio:
    def test_constant_positive(self):
        r = _constant_returns(0.001, 252)
        assert metrics.sharpe_ratio(r) == 0.0  # std == 0

    def test_positive_drift(self):
        r = _daily_returns(500, mean=0.001, std=0.01)
        sr = metrics.sharpe_ratio(r)
        assert sr > 0

    def test_empty(self):
        assert metrics.sharpe_ratio(pd.Series(dtype=float)) == 0.0


class TestSortinoRatio:
    def test_all_positive(self):
        r = _constant_returns(0.001, 252)
        assert metrics.sortino_ratio(r) == 0.0  # no downside -> 0

    def test_mixed(self):
        r = _daily_returns(500)
        s = metrics.sortino_ratio(r)
        assert isinstance(s, float)

    def test_empty(self):
        assert metrics.sortino_ratio(pd.Series(dtype=float)) == 0.0


class TestMaxDrawdown:
    def test_known_drawdown(self):
        r = pd.Series([0.10, -0.20, 0.05])
        mdd = metrics.max_drawdown(r)
        cumulative = (1 + r).cumprod()  # [1.10, 0.88, 0.924]
        peak = cumulative.cummax()  # [1.10, 1.10, 1.10]
        expected = float(abs(((cumulative - peak) / peak).min()))
        assert math.isclose(mdd, expected, rel_tol=1e-9)

    def test_always_up(self):
        r = _constant_returns(0.01, 10)
        assert metrics.max_drawdown(r) == 0.0

    def test_empty(self):
        assert metrics.max_drawdown(pd.Series(dtype=float)) == 0.0


class TestCalmarRatio:
    def test_no_drawdown(self):
        r = _constant_returns(0.01, 100)
        assert metrics.calmar_ratio(r) == 0.0

    def test_positive(self):
        r = _daily_returns(500)
        cr = metrics.calmar_ratio(r)
        assert isinstance(cr, float)


class TestTurnover:
    def test_no_change(self):
        w = [{"AAPL": 0.5, "GOOG": 0.5}] * 5
        assert metrics.turnover(w) == 0.0

    def test_full_flip(self):
        w = [{"AAPL": 1.0}, {"GOOG": 1.0}]
        assert metrics.turnover(w) == 2.0  # AAPL |-1| + GOOG |+1| = 2

    def test_single(self):
        assert metrics.turnover([{"AAPL": 0.5}]) == 0.0


class TestHitRate:
    def test_all_positive(self):
        r = _constant_returns(0.01, 10)
        assert metrics.hit_rate(r) == 1.0

    def test_half(self):
        r = pd.Series([0.01, -0.01, 0.01, -0.01])
        assert metrics.hit_rate(r) == 0.5

    def test_empty(self):
        assert metrics.hit_rate(pd.Series(dtype=float)) == 0.0


class TestComputeAllMetrics:
    def test_keys(self):
        r = _daily_returns(100)
        m = metrics.compute_all_metrics(r)
        expected_keys = {
            "total_return", "cagr", "annualized_volatility",
            "sharpe_ratio", "sortino_ratio", "max_drawdown",
            "calmar_ratio", "hit_rate",
        }
        assert set(m.keys()) == expected_keys

    def test_all_float(self):
        m = metrics.compute_all_metrics(_daily_returns(100))
        for v in m.values():
            assert isinstance(v, float)


class TestBenchmarkMetrics:
    def test_beta_of_self_is_one(self):
        r = _daily_returns(120, seed=1)
        assert metrics.beta(r, r) == pytest.approx(1.0, abs=1e-9)

    def test_alpha_of_self_is_zero(self):
        r = _daily_returns(120, seed=1)
        assert metrics.alpha(r, r) == pytest.approx(0.0, abs=1e-9)

    def test_information_ratio_zero_when_identical(self):
        r = _daily_returns(120, seed=1)
        # Zero active return -> zero tracking error -> IR defined as 0.
        assert metrics.information_ratio(r, r) == 0.0

    def test_excess_return_positive_when_outperforming(self):
        bench = _daily_returns(120, mean=0.0, seed=2)
        strat = bench + 0.001  # uniformly outperform each day
        assert metrics.excess_return(strat, bench) > 0

    def test_beta_scales_with_leverage(self):
        bench = _daily_returns(120, seed=3)
        assert metrics.beta(2 * bench, bench) == pytest.approx(2.0, abs=1e-9)

    def test_misaligned_or_empty_returns_zero(self):
        r = _daily_returns(50, seed=4)
        assert metrics.beta(r, pd.Series(dtype=float)) == 0.0
        assert metrics.alpha(r, pd.Series(dtype=float)) == 0.0

    def test_rollup_keys_and_floats(self):
        r = _daily_returns(120, seed=5)
        b = _daily_returns(120, seed=6)
        m = metrics.compute_benchmark_metrics(r, b)
        assert set(m) == {
            "benchmark_total_return", "alpha", "beta",
            "information_ratio", "excess_return",
        }
        assert all(isinstance(v, float) for v in m.values())


# ======================================================================
# 2. portfolio/attribution.py
# ======================================================================

class TestPerformanceAttribution:
    def _make_attribution(self):
        attr = PerformanceAttribution()
        base = datetime(2023, 6, 1)
        prices_0 = {"AAPL": 100.0, "GOOG": 200.0}
        attr.record_trades(
            [
                {"symbol": "AAPL", "shares": 10, "price": 100.0, "strategy": "momentum"},
                {"symbol": "GOOG", "shares": 5, "price": 200.0, "strategy": "value"},
            ],
            prices_0,
        )
        attr._prev_prices = prices_0
        for i in range(1, 6):
            d = base + timedelta(days=i)
            p = {"AAPL": 100.0 + i, "GOOG": 200.0 - i * 0.5}
            attr.update_daily(d, p, nav=100_000.0)
        return attr

    def test_update_daily_normalizes_by_nav_not_raw_dollar_pnl(self):
        """Regression: update_daily used to feed raw dollar P&L straight into
        compute_all_metrics(), which assumes period *percentage* returns
        (total_return does (1+r).prod()) — nonsense metrics resulted the
        moment daily P&L exceeded a few dollars."""
        attr = PerformanceAttribution()
        attr.record_trades(
            [{"symbol": "AAPL", "shares": 10, "price": 100.0, "strategy": "momentum"}],
            {"AAPL": 100.0},
        )
        attr._prev_prices = {"AAPL": 100.0}
        attr.update_daily(datetime(2023, 6, 2), {"AAPL": 105.0}, nav=100_000.0)

        s = attr.get_strategy_returns("momentum")
        # $10 * 5 = $50 P&L on a $100,000 NAV -> 0.05% return, not "50".
        assert s.iloc[0] == pytest.approx(0.0005)

    def test_strategies(self):
        attr = self._make_attribution()
        assert set(attr.strategies) == {"momentum", "value"}

    def test_strategy_returns_series(self):
        attr = self._make_attribution()
        s = attr.get_strategy_returns("momentum")
        assert isinstance(s, pd.Series)
        assert len(s) == 5

    def test_strategy_metrics(self):
        attr = self._make_attribution()
        m = attr.get_strategy_metrics()
        assert "momentum" in m
        assert "total_return" in m["momentum"]

    def test_summary_dataframe(self):
        attr = self._make_attribution()
        df = attr.summary()
        assert isinstance(df, pd.DataFrame)
        assert "momentum" in df.index

    def test_trade_log(self):
        attr = self._make_attribution()
        assert len(attr.trade_log) == 2

    def test_factor_attribution_empty(self):
        attr = PerformanceAttribution()
        df = attr.get_factor_attribution()
        assert df.empty

    def test_factor_attribution_with_data(self):
        attr = self._make_attribution()
        attr.set_factor_exposures("momentum", {"market": 0.8, "size": 0.3})
        df = attr.get_factor_attribution()
        assert "market" in df.columns


# ======================================================================
# 3. eval/reports.py
# ======================================================================

class TestBacktestReport:
    @pytest.fixture()
    def report(self):
        r = _daily_returns(100)
        attr = PerformanceAttribution()
        snaps = [
            PortfolioSnapshot(
                asof=r.index[0].to_pydatetime(),
                nav=1_000_000,
            ),
            PortfolioSnapshot(
                asof=r.index[-1].to_pydatetime(),
                nav=1_050_000,
            ),
        ]
        return BacktestReport(r, attr, snaps)

    def test_portfolio_summary(self, report):
        s = report.portfolio_summary()
        assert isinstance(s, dict)
        assert "sharpe_ratio" in s

    def test_to_text(self, report):
        txt = report.to_text()
        assert "BACKTEST REPORT" in txt
        assert "sharpe_ratio" in txt

    def test_to_dict(self, report):
        d = report.to_dict()
        assert "portfolio" in d
        assert "data_points" in d

    def test_save(self, report, tmp_path):
        out = tmp_path / "report.json"
        report.save(str(out))
        assert out.exists()
        data = json.loads(out.read_text())
        assert "portfolio" in data

    def test_benchmark_summary_empty_without_benchmark(self, report):
        # No benchmark supplied -> benchmark section is omitted, not invented.
        assert report.benchmark_summary() == {}
        assert "benchmark" not in report.to_dict()

    def test_benchmark_summary_present_with_benchmark(self):
        r = _daily_returns(100, seed=11)
        bench = _daily_returns(100, seed=12)
        rep = BacktestReport(r, PerformanceAttribution(), [], benchmark_returns=bench)
        bm = rep.benchmark_summary()
        assert "alpha" in bm and "beta" in bm and "information_ratio" in bm
        assert "benchmark" in rep.to_dict()
        assert "Benchmark-Relative" in rep.to_text()


# ======================================================================
# 4. eval/plots.py
# ======================================================================

class TestPlots:
    def test_equity_curve(self):
        fig = plot_equity_curve(_daily_returns(100))
        assert isinstance(fig, Figure)

    def test_drawdown(self):
        fig = plot_drawdown(_daily_returns(100))
        assert isinstance(fig, Figure)

    def test_monthly_returns(self):
        fig = plot_monthly_returns(_daily_returns(300))
        assert isinstance(fig, Figure)

    def test_monthly_returns_empty(self):
        fig = plot_monthly_returns(pd.Series(dtype=float))
        assert isinstance(fig, Figure)

    def test_strategy_attribution_empty(self):
        fig = plot_strategy_attribution(PerformanceAttribution())
        assert isinstance(fig, Figure)

    def test_exposure(self):
        history = [
            (datetime(2023, 1, i), {"AAPL": 0.5, "GOOG": 0.5})
            for i in range(1, 11)
        ]
        fig = plot_exposure(history)
        assert isinstance(fig, Figure)

    def test_exposure_empty(self):
        fig = plot_exposure([])
        assert isinstance(fig, Figure)

    def test_rolling_sharpe(self):
        fig = plot_rolling_sharpe(_daily_returns(200), window=63)
        assert isinstance(fig, Figure)

    def test_rolling_sharpe_short(self):
        fig = plot_rolling_sharpe(_daily_returns(10), window=63)
        assert isinstance(fig, Figure)

    def test_save_all_plots(self, tmp_path):
        r = _daily_returns(300)
        attr = PerformanceAttribution()
        save_all_plots(r, attr, str(tmp_path / "plots"))
        assert (tmp_path / "plots" / "equity_curve.png").exists()
        assert (tmp_path / "plots" / "drawdown.png").exists()
