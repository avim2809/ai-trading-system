"""Structured (SQL) query layer over backtest run artifacts.

Research finding (TableRAG, GTR): numeric/tabular data should be preserved as
structured data and queried with SQL, *not* flattened into text chunks for
vector retrieval — flat chunking causes structural information loss and breaks
multi-hop/aggregation queries. And because even frontier LLMs hallucinate on
multi-step financial arithmetic, numeric questions must be answered by
deterministic SQL, with the LLM reserved for narrating the result.

:class:`RunStore` exposes two read-only DuckDB views over ``runs/``:

* ``runs``   — one row per run: portfolio metrics from ``report.json`` plus
               registry metadata (status, notes, strategies, dates).
* ``trades`` — every row from each run's ``trades.parquet``, tagged ``run_id``.

DuckDB reads Parquet natively and needs no server. The store is read-only:
:meth:`query` rejects anything that is not a single ``SELECT``/``WITH``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from firm.eval.reports import TRADE_COLUMNS

log = logging.getLogger(__name__)

# Columns guaranteed on the ``runs`` view even when no runs exist yet, so the
# assistant's generated SQL and the schema description stay stable.
_RUN_BASE_COLUMNS = [
    "run_id", "status", "notes", "seed", "strategies",
    "period_start", "period_end", "final_nav", "data_points",
]

_FORBIDDEN_SQL = (
    "insert", "update", "delete", "drop", "create", "alter",
    "attach", "copy", "pragma", "install", "load", "export",
)


class ReadOnlyQueryError(ValueError):
    """Raised when a non-SELECT statement is passed to :meth:`RunStore.query`."""


def _assert_select(sql: str) -> None:
    """Reject anything that isn't a single read-only SELECT/WITH statement."""
    stripped = sql.strip().rstrip(";").lstrip("(")
    lowered = stripped.lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        raise ReadOnlyQueryError("Only SELECT/WITH queries are allowed.")
    # Block stacked statements and any mutating keyword as a whole word.
    if ";" in stripped:
        raise ReadOnlyQueryError("Multiple statements are not allowed.")
    import re
    for kw in _FORBIDDEN_SQL:
        if re.search(rf"\b{kw}\b", lowered):
            raise ReadOnlyQueryError(f"Disallowed keyword in query: {kw}")


class RunStore:
    """Read-only SQL view over ``runs/*/{report.json, trades.parquet}``."""

    def __init__(self, runs_dir: str = "runs") -> None:
        self._runs_dir = Path(runs_dir)
        self._con: Any = None
        self._runs_df: pd.DataFrame | None = None
        self._trades_df: pd.DataFrame | None = None

    # ── building ────────────────────────────────────────────────────

    def _registry(self) -> dict[str, dict[str, Any]]:
        """Map run_id → registry entry from ``runs/registry.json`` (if present)."""
        reg_path = self._runs_dir / "registry.json"
        if not reg_path.exists():
            return {}
        try:
            raw = json.loads(reg_path.read_text(encoding="utf-8"))
            return {entry["run_id"]: entry for entry in raw if "run_id" in entry}
        except Exception:
            log.warning("Could not parse %s", reg_path, exc_info=True)
            return {}

    @staticmethod
    def _strategies_str(config: dict[str, Any]) -> str:
        """Render the run's strategy list as a comma-joined string."""
        strategies = config.get("strategies")
        if isinstance(strategies, dict):
            strategies = strategies.get("enabled")
        if isinstance(strategies, (list, tuple)):
            return ",".join(str(s) for s in strategies)
        return str(strategies) if strategies else ""

    def _build(self) -> None:
        registry = self._registry()
        run_rows: list[dict[str, Any]] = []
        trade_frames: list[pd.DataFrame] = []

        if self._runs_dir.exists():
            for run_dir in sorted(p for p in self._runs_dir.iterdir() if p.is_dir()):
                report_path = run_dir / "report.json"
                if not report_path.exists():
                    continue
                run_id = run_dir.name
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except Exception:
                    log.warning("Skipping unreadable report %s", report_path, exc_info=True)
                    continue

                row: dict[str, Any] = {"run_id": run_id}
                row.update(report.get("portfolio", {}))
                for k, v in (report.get("benchmark", {}) or {}).items():
                    row[f"bench_{k}"] = v
                period = report.get("period", {}) or {}
                row["period_start"] = period.get("start")
                row["period_end"] = period.get("end")
                row["final_nav"] = report.get("final_nav")
                row["data_points"] = report.get("data_points")

                entry = registry.get(run_id, {})
                row["status"] = entry.get("status")
                row["notes"] = entry.get("notes")
                row["seed"] = entry.get("seed")
                row["strategies"] = self._strategies_str(entry.get("config", {}) or {})
                run_rows.append(row)

                trades_path = run_dir / "trades.parquet"
                if trades_path.exists():
                    try:
                        tdf = pd.read_parquet(trades_path)
                        if not tdf.empty:
                            # Tag with run_id (dropping any stale column first).
                            if "run_id" in tdf.columns:
                                tdf = tdf.drop(columns=["run_id"])
                            tdf.insert(0, "run_id", run_id)
                            trade_frames.append(tdf)
                    except Exception:
                        log.warning("Could not read %s", trades_path, exc_info=True)

        if run_rows:
            self._runs_df = pd.DataFrame(run_rows)
        else:
            self._runs_df = pd.DataFrame(columns=_RUN_BASE_COLUMNS)

        if trade_frames:
            self._trades_df = pd.concat(trade_frames, ignore_index=True)
        else:
            self._trades_df = pd.DataFrame(columns=["run_id", *TRADE_COLUMNS])

    def _connect(self) -> Any:
        if self._con is not None:
            return self._con
        try:
            import duckdb
        except ImportError as exc:
            raise ImportError(
                "duckdb is required for RunStore. Install with: pip install 'firm[llm]'"
            ) from exc
        if self._runs_df is None or self._trades_df is None:
            self._build()
        con = duckdb.connect(database=":memory:")
        con.register("runs", self._runs_df)
        con.register("trades", self._trades_df)
        self._con = con
        return con

    def refresh(self) -> None:
        """Re-scan ``runs/`` and rebuild the views (e.g. after a new backtest)."""
        if self._con is not None:
            try:
                self._con.close()
            except Exception:
                pass
        self._con = None
        self._runs_df = None
        self._trades_df = None

    # ── querying ────────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        """Run a read-only SQL query against the ``runs``/``trades`` views."""
        _assert_select(sql)
        con = self._connect()
        return con.execute(sql).fetchdf()

    def runs(self) -> pd.DataFrame:
        """Return the full ``runs`` table as a DataFrame."""
        if self._runs_df is None:
            self._build()
        return self._runs_df.copy()

    def trades(self) -> pd.DataFrame:
        """Return the full ``trades`` table as a DataFrame."""
        if self._trades_df is None:
            self._build()
        return self._trades_df.copy()

    def schema(self) -> str:
        """Human/LLM-readable description of the two tables and their columns.

        Fed to the assistant so it can write correct SQL. Column lists are
        derived from the actual data so newly-added metrics are reflected.
        """
        if self._runs_df is None or self._trades_df is None:
            self._build()
        runs_cols = ", ".join(self._runs_df.columns)
        trades_cols = ", ".join(self._trades_df.columns)
        return (
            "Two read-only tables are available (DuckDB SQL):\n\n"
            f"runs({runs_cols})\n"
            "  - one row per backtest run. Portfolio metrics (sharpe_ratio, "
            "max_drawdown, cagr, total_return, hit_rate, ...) come straight "
            "from the run report; bench_* are benchmark-relative metrics.\n\n"
            f"trades({trades_cols})\n"
            "  - one row per closed trade. size is signed (negative = short); "
            "pnl is gross, pnl_net is after commission; return_pct is net P&L "
            "over entry notional; join to runs via run_id.\n\n"
            "Rules: SELECT/WITH only. Compute all numbers in SQL — never "
            "estimate them. Example: "
            "SELECT run_id, sharpe_ratio FROM runs ORDER BY sharpe_ratio DESC LIMIT 5;"
        )
