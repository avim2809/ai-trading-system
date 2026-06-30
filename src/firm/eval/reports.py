"""Backtest report generation: text, dict, and JSON outputs.

Combines portfolio-level metrics with per-strategy attribution into a
single structured report that can be serialised, printed, or fed into
downstream dashboards.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from firm.contracts.models import PortfolioSnapshot
from firm.eval.metrics import compute_all_metrics, compute_benchmark_metrics
from firm.portfolio.attribution import PerformanceAttribution

# Stable column order for the per-trade log written to ``trades.parquet``.
# Kept explicit so an empty run still produces a schema-valid parquet file
# (and so the DuckDB ``trades`` view always has the same columns).
TRADE_COLUMNS = [
    "entry_dt", "exit_dt", "symbol", "strategy", "size",
    "entry_price", "exit_price", "pnl", "pnl_net", "commission",
    "return_pct", "bars_held",
]


class BacktestReport:
    """Generate comprehensive backtest reports."""

    def __init__(
        self,
        returns: pd.Series,
        attribution: PerformanceAttribution,
        snapshots: list[PortfolioSnapshot],
        benchmark_returns: pd.Series | None = None,
        trades: list[dict] | None = None,
    ) -> None:
        self.returns = returns
        self.attribution = attribution
        self.snapshots = snapshots
        self.benchmark_returns = (
            benchmark_returns if benchmark_returns is not None else pd.Series(dtype=float)
        )
        self.trades = trades or []

    # ------------------------------------------------------------------
    # Benchmark-relative
    # ------------------------------------------------------------------

    def benchmark_summary(self) -> dict[str, float]:
        """Alpha/beta/information-ratio/excess vs an equal-weight buy-and-hold
        of the traded universe. Empty dict when no benchmark is available."""
        if self.returns.empty or self.benchmark_returns.empty:
            return {}
        return compute_benchmark_metrics(self.returns, self.benchmark_returns)

    # ------------------------------------------------------------------
    # Portfolio-level
    # ------------------------------------------------------------------

    def portfolio_summary(self) -> dict[str, float]:
        """Overall portfolio metrics."""
        return compute_all_metrics(self.returns)

    # ------------------------------------------------------------------
    # Strategy-level
    # ------------------------------------------------------------------

    def strategy_summary(self) -> pd.DataFrame:
        """Per-strategy metrics table (strategy × metric)."""
        return self.attribution.summary()

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_text(self) -> str:
        """Format as a human-readable text report."""
        lines: list[str] = []

        lines.append("=" * 60)
        lines.append("BACKTEST REPORT")
        lines.append("=" * 60)

        if self.snapshots:
            lines.append(
                f"Period : {self.snapshots[0].asof:%Y-%m-%d}"
                f" → {self.snapshots[-1].asof:%Y-%m-%d}"
            )
            lines.append(f"Final NAV: {self.snapshots[-1].nav:,.2f}")
        lines.append(f"Data points: {len(self.returns)}")
        lines.append("")

        lines.append("--- Portfolio Metrics ---")
        for k, v in self.portfolio_summary().items():
            lines.append(f"  {k:30s}: {v:>12.6f}")
        lines.append("")

        bench = self.benchmark_summary()
        if bench:
            lines.append("--- Benchmark-Relative (vs equal-weight buy & hold) ---")
            for k, v in bench.items():
                lines.append(f"  {k:30s}: {v:>12.6f}")
            lines.append("")

        strat = self.strategy_summary()
        if not strat.empty:
            lines.append("--- Strategy Attribution ---")
            lines.append(strat.to_string())
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Full report as nested dict (JSON-serialisable)."""
        result: dict = {
            "portfolio": self.portfolio_summary(),
            "data_points": len(self.returns),
        }

        bench = self.benchmark_summary()
        if bench:
            result["benchmark"] = bench

        if self.snapshots:
            result["period"] = {
                "start": self.snapshots[0].asof.isoformat(),
                "end": self.snapshots[-1].asof.isoformat(),
            }
            result["final_nav"] = self.snapshots[-1].nav

        strat = self.strategy_summary()
        if not strat.empty:
            result["strategies"] = strat.to_dict(orient="index")

        return result

    def save(self, path: str) -> None:
        """Save report to JSON file."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def save_trades(self, path: str) -> int:
        """Write the per-trade log to a Parquet file. Returns the row count.

        An empty trade list still produces a schema-valid (zero-row) Parquet
        file with :data:`TRADE_COLUMNS`, so the structured-query layer can
        always register a consistent ``trades`` view.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.trades, columns=TRADE_COLUMNS)
        df.to_parquet(out, index=False)
        return len(df)
