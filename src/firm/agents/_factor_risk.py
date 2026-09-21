"""Factor-exposure helpers for :class:`~firm.agents.risk.RiskAgent`'s
factor-risk overlay.

Per-name and per-sector weight caps miss a book that is diversified by
*name* but not by *factor* — ten different momentum-tilted tech names can
each clear ``max_position_pct``/``max_sector_pct`` while the whole portfolio
is effectively one leveraged bet on a single systematic driver. This module
computes two such drivers directly from price history already available via
``PitView``:

- Market beta: covariance with a benchmark ETF, the standard single-factor
  exposure measure.
- Momentum: a cross-sectional z-score of trailing return, capturing a book's
  tilt toward (or against) recent relative winners.

Sector is deliberately not recomputed here — ``RiskAgent.sector_map`` and
``_cap_sector_concentration`` already own that, so the overlay reuses them
rather than maintaining a second sector model. Two statistical factors,
no fundamentals, no PCA/eigen-risk-model estimation: a small universe
(~25-35 names) doesn't have the cross-sectional breadth to support a full
multi-factor risk model, and an over-fit one would be worse than none.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _returns_pivot(
    pit_view: Any,
    symbols: list[str],
    lookback_days: int,
) -> pd.DataFrame | None:
    """Wide (date x symbol) daily-return matrix, or ``None`` on any data gap.

    Shared by beta and momentum below so both read price history the same
    way — the prices -> pivot -> pct_change idiom RiskAgent's other
    pit_view-backed overlays (``_cap_correlated_exposure``, ``_cvar_overlay``)
    already use.
    """
    try:
        price_df = pit_view.prices(symbols, lookback_days=lookback_days)
    except Exception as exc:
        log.debug("Factor risk: price lookup failed (%s)", exc, exc_info=True)
        return None
    if price_df is None or price_df.empty or "close" not in price_df.columns:
        return None
    try:
        pivot = price_df.pivot_table(index="date", columns="symbol", values="close")
        return pivot.pct_change().dropna(how="all")
    except Exception as exc:
        log.debug("Factor risk: malformed price history (%s)", exc, exc_info=True)
        return None


def compute_beta_exposures(
    pit_view: Any,
    symbols: list[str],
    benchmark_symbol: str = "SPY",
    lookback_days: int = 60,
) -> dict[str, float]:
    """Per-symbol market beta: ``cov(symbol, benchmark) / var(benchmark)``.

    The two-moment beta identity rather than a fitted regression — equivalent
    for a single-factor OLS but needs no separate library. A symbol without
    enough return history overlapping the benchmark's is simply omitted, so
    callers can distinguish "no exposure reading" from a real zero beta.
    """
    all_symbols = sorted(set(symbols) | {benchmark_symbol})
    rets = _returns_pivot(pit_view, all_symbols, lookback_days)
    if rets is None or benchmark_symbol not in rets.columns:
        return {}
    bench = rets[benchmark_symbol]
    bench_var = bench.var()
    if not bench_var or bench_var != bench_var:  # zero or NaN
        return {}

    betas: dict[str, float] = {}
    for sym in symbols:
        if sym == benchmark_symbol:
            betas[sym] = 1.0
            continue
        if sym not in rets.columns:
            continue
        paired = pd.concat([rets[sym], bench], axis=1).dropna()
        if len(paired) < 5:
            continue
        cov = paired.iloc[:, 0].cov(paired.iloc[:, 1])
        beta = cov / bench_var
        if beta == beta:  # not NaN
            betas[sym] = float(beta)
    return betas


def compute_momentum_zscores(
    pit_view: Any,
    symbols: list[str],
    lookback_days: int = 60,
) -> dict[str, float]:
    """Cross-sectional z-score of each symbol's trailing cumulative return.

    Cross-sectional, not an absolute return threshold: "momentum exposure"
    is inherently relative — a name up 5% only carries a momentum tilt if
    its peers in the same book are flat or down. Mirrors the z-score
    convention ``firm.strategies.multi_factor`` already uses for its own
    momentum leg. Clipped to [-5, 5] so one outlier return can't dominate
    the aggregate below.
    """
    rets = _returns_pivot(pit_view, symbols, lookback_days)
    if rets is None:
        return {}
    cum_return = (1.0 + rets).prod() - 1.0
    cum_return = cum_return.replace([np.inf, -np.inf], np.nan).dropna()
    if cum_return.empty:
        return {}
    std = cum_return.std()
    if not std or std != std:
        return {}
    z = ((cum_return - cum_return.mean()) / std).clip(-5, 5)
    return z.to_dict()


def portfolio_factor_exposures(
    pit_view: Any,
    targets: dict[str, float],
    benchmark_symbol: str = "SPY",
    beta_lookback_days: int = 60,
    momentum_lookback_days: int = 60,
    momentum_universe: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate the book's net weighted exposure to each factor.

    ``beta``/``momentum`` are the target-weighted (signed, not absolute)
    sums of each held symbol's own reading — the same sense in which a
    book's "net beta" is normally quoted: long high-beta names against
    short low-beta names nets to a large positive beta exposure even though
    no single position breached a per-name cap. A symbol with no exposure
    reading (missing/short history) contributes zero rather than being
    dropped from the weight sum, so a single bad symbol only understates
    the aggregate, never raises.

    Momentum is z-scored against ``momentum_universe`` (default:
    ``pit_view.universe``, falling back to the held symbols only if that
    isn't available) rather than against the held symbols alone — a book
    that holds several names *all* extreme relative to the broader universe
    would otherwise partially cancel out and look like an ordinary in-sample
    spread. Beta has no equivalent issue (it's an absolute reading against
    the benchmark, not relative to whichever names happen to be held), so
    it's computed for the held symbols directly.
    """
    held = [s for s, w in targets.items() if w != 0.0]
    if not held:
        return {"beta": 0.0, "momentum": 0.0, "per_symbol": {}}

    universe = momentum_universe or list(getattr(pit_view, "universe", None) or held)
    universe = sorted(set(universe) | set(held))

    betas = compute_beta_exposures(pit_view, held, benchmark_symbol, beta_lookback_days)
    momentum = compute_momentum_zscores(pit_view, universe, momentum_lookback_days)

    per_symbol: dict[str, dict[str, float]] = {}
    port_beta = 0.0
    port_momentum = 0.0
    for sym in held:
        w = targets[sym]
        entry: dict[str, float] = {}
        if sym in betas:
            entry["beta"] = betas[sym]
            port_beta += w * betas[sym]
        if sym in momentum:
            entry["momentum"] = momentum[sym]
            port_momentum += w * momentum[sym]
        if entry:
            per_symbol[sym] = entry

    return {"beta": port_beta, "momentum": port_momentum, "per_symbol": per_symbol}
