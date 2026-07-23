"""QuantStats HTML tear-sheet rendering.

A thin wrapper over `QuantStats <https://github.com/ranaroussi/quantstats>`_ that
turns a firm daily-returns Series (e.g. ``BacktestReport.returns``) into a rich
HTML performance tear-sheet. QuantStats is an *optional* dependency: install the
``report`` extra (``pip install -e '.[report]'``) to enable it. The import is
lazy and fails fast with a clear message when the extra is missing.

Ported from the external trading-suite ``tear-sheet`` skill (Apache-2.0).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

_INSTALL_HINT = (
    "quantstats is required for tear-sheet rendering. Install the optional "
    "'report' extra: pip install -e '.[report]'  (or: pip install quantstats)."
)


def _require_quantstats():
    try:
        import quantstats as qs  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only when missing
        raise ImportError(_INSTALL_HINT) from exc
    return qs


def render_tearsheet(
    returns: pd.Series,
    benchmark: pd.Series | None = None,
    out_html: str | Path = "tearsheet.html",
    title: str = "Firm Strategy Tear-Sheet",
) -> Path:
    """Render an HTML tear-sheet for a daily-returns series.

    Parameters
    ----------
    returns:
        Daily returns as a Series indexed by date (not an equity curve).
    benchmark:
        Optional benchmark daily returns for relative stats.
    out_html:
        Output path for the generated HTML file.
    title:
        Report title.

    Returns
    -------
    Path to the written HTML file.

    Raises
    ------
    ImportError
        If quantstats is not installed.
    ValueError
        If *returns* is empty.
    """
    qs = _require_quantstats()

    series = pd.Series(returns).dropna()
    if series.empty:
        raise ValueError("returns series is empty; nothing to render")
    series.index = pd.to_datetime(series.index)

    bench = None
    if benchmark is not None:
        bench = pd.Series(benchmark).dropna()
        bench.index = pd.to_datetime(bench.index)
        if bench.empty:
            bench = None

    out = Path(out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "Rendering tear-sheet %r (%d return points, benchmark=%s) -> %s",
        title, len(series), bench is not None, out,
    )
    qs.reports.html(series, benchmark=bench, output=str(out), title=title)
    log.info("Tear-sheet written: %s", out)
    return out
