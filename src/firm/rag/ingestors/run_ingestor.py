"""Ingestor for the system's own backtest-run artifacts.

Indexes a short *narrative* summary of each completed run into the
``run_notes`` collection so the assistant can retrieve "which run / what
happened" context by semantic search. Only descriptive text is embedded —
the authoritative numbers live in the structured DuckDB views
(:class:`firm.rag.structured.RunStore`), which the assistant queries with SQL.
Embedding numbers in prose here is purely to make a run *findable*; it is not
the source of truth for any figure.

Availability dating uses each run's period-end (via
:func:`firm.rag.dates.normalize_date`) so point-in-time retrieval stays
correct, consistent with the other ingestors.
"""

from __future__ import annotations

import math
from typing import Any

from firm.rag.chunker import DocumentChunker
from firm.rag.dates import normalize_date
from firm.rag.ingestors.base_ingestor import BaseIngestor
from firm.rag.models import Document
from firm.rag.store import VectorStore
from firm.rag.structured import RunStore

COLLECTION = "run_notes"


def _pct(value: Any) -> str:
    """Format a fraction as a percentage, tolerating None/NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def _num(value: Any, fmt: str = ".2f") -> str:
    """Format a number, tolerating None/NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return "n/a"


class RunArtifactIngestor(BaseIngestor):
    """Indexes narrative summaries of backtest runs into ``run_notes``."""

    def __init__(
        self,
        store: VectorStore,
        chunker: DocumentChunker,
        runs_dir: str = "runs",
    ) -> None:
        super().__init__(store, chunker)
        self._run_store = RunStore(runs_dir)

    def _narrative(self, row: dict[str, Any]) -> str:
        """Build a descriptive paragraph for one run row."""
        strategies = row.get("strategies") or "(default set)"
        status = row.get("status") or "completed"
        start = row.get("period_start") or "?"
        end = row.get("period_end") or "?"
        notes = (row.get("notes") or "").strip()

        text = (
            f"Backtest run {row.get('run_id')} ({status}) used strategies "
            f"[{strategies}] over the period {start} to {end}. "
            f"Performance: total return {_pct(row.get('total_return'))}, "
            f"CAGR {_pct(row.get('cagr'))}, Sharpe ratio {_num(row.get('sharpe_ratio'))}, "
            f"Sortino {_num(row.get('sortino_ratio'))}, "
            f"max drawdown {_pct(row.get('max_drawdown'))}, "
            f"Calmar {_num(row.get('calmar_ratio'))}, "
            f"hit rate {_pct(row.get('hit_rate'))}. "
        )
        if "bench_alpha" in row:
            text += (
                f"Benchmark-relative: alpha {_pct(row.get('bench_alpha'))}, "
                f"beta {_num(row.get('bench_beta'))}, "
                f"information ratio {_num(row.get('bench_information_ratio'))}. "
            )
        if notes:
            text += f"Notes: {notes}"
        return text.strip()

    def ingest(self, **kwargs: Any) -> int:
        """Index all runs that have a report. Returns docs added."""
        self._run_store.refresh()
        runs = self._run_store.runs()
        if runs.empty:
            return 0

        docs: list[Document] = []
        for row in runs.to_dict(orient="records"):
            run_id = row.get("run_id")
            if not run_id:
                continue
            date = normalize_date(row.get("period_end") or row.get("period_start"))
            metadata = {
                "source": f"run:{run_id}",
                "run_id": run_id,
                "doc_type": "run_note",
                "date": date,
            }
            docs.extend(self.chunker.chunk(self._narrative(row), metadata))

        if docs:
            return self.store.add_documents(COLLECTION, docs)
        return 0
