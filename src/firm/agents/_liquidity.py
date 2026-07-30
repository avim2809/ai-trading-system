"""Shared average-daily-volume (ADV) and market-impact helpers.

Used by both :class:`~firm.agents.risk.RiskAgent`'s participation-rate
liquidity cap and the size/volume-aware market-impact cost model
(:class:`~firm.agents.execution.ExecutionAgent` live, and
:mod:`firm.backtest.firm_strategy` in backtest) so the risk cap and the cost
estimate agree on what "ADV" means — computing them from two different ADV
definitions would let a trade the liquidity cap clears still be priced as if
it were a very different size, or vice versa.
"""

from __future__ import annotations

import logging
import math
from typing import Any

log = logging.getLogger(__name__)


def estimate_adv_dollars(
    pit_view: Any,
    symbol: str,
    lookback_days: int = 20,
) -> float | None:
    """Trailing average daily dollar volume for *symbol*, or ``None``.

    Reads *lookback_days* of OHLCV history from *pit_view* — any object
    exposing ``.prices(symbols, lookback_days)`` (``PitViewAdapter`` in
    backtest, ``LivePitViewAdapter`` in live). Returns ``None`` — never
    raises — when volume data isn't available, so callers degrade
    gracefully (skip the liquidity cap / fall back to flat-pct cost) instead
    of failing a cycle over a missing/thin data provider.
    """
    try:
        price_df = pit_view.prices([symbol], lookback_days=lookback_days)
    except Exception as exc:
        log.debug("ADV lookup failed for %s: %s", symbol, exc, exc_info=True)
        return None
    if price_df is None or price_df.empty or "volume" not in price_df.columns:
        return None
    sym_df = (
        price_df[price_df["symbol"] == symbol]
        if "symbol" in price_df.columns
        else price_df
    )
    if sym_df.empty:
        return None
    adv_dollars = (sym_df["volume"] * sym_df["close"]).mean()
    if adv_dollars is None or adv_dollars != adv_dollars or adv_dollars <= 0:  # NaN-safe
        return None
    return float(adv_dollars)


def market_impact_pct(
    participation: float,
    coefficient: float,
    crossover: float | None = None,
) -> float:
    """Market-impact cost, as a fraction of trade notional.

    Implements the "square-root law" of market impact widely used in the
    execution literature (e.g. Almgren, Thum, Hauptmann & Li 2005; the BARRA
    market-impact model): temporary price impact scales roughly with the
    square root of participation rate (trade notional / ADV dollars) rather
    than linearly — a size-*aware* cost, unlike a flat percentage of
    notional that charges the same rate whether a trade is 0.1% or 50% of a
    name's daily volume.

    ``coefficient`` is the fraction-of-notional impact at 100% participation
    (trading a full day's ADV in one order); it should be calibrated from
    realised fill data where available. ``0.0`` (the default everywhere in
    this codebase's Python-level config) disables the model entirely,
    preserving flat-pct-only cost behaviour; ``config/live.yaml`` and
    ``config/settings.yaml`` opt in with a conservative default.

    ``crossover`` (optional) accounts for empirical work showing impact is
    closer to *linear* in participation for small orders, crossing over to
    the square-root regime only as participation grows (e.g. Kyle &
    Obizhaeva) — a pure sqrt law calibrated at higher participation can
    overstate the cost of small trades, which is this system's usual
    regime (a small account relative to the mega-cap universe's ADV).
    ``None`` (default) preserves the original pure square-root law at every
    participation level, unchanged. When set, participation below
    ``crossover`` scales *linearly* from zero, continuous (C0) with the
    square-root branch at the crossover point itself:
    ``coefficient * sqrt(crossover) * (participation / crossover)`` below
    ``crossover``, ``coefficient * sqrt(participation)`` at or above it.
    """
    if coefficient <= 0 or participation <= 0:
        return 0.0
    if crossover is None or crossover <= 0 or participation >= crossover:
        return coefficient * math.sqrt(participation)
    return coefficient * math.sqrt(crossover) * (participation / crossover)


def sqrt_impact_pct(participation: float, coefficient: float) -> float:
    """Pure square-root market-impact cost — see :func:`market_impact_pct`.

    Kept as a thin, unchanged-behaviour alias (``crossover=None``) so every
    pre-existing call site is unaffected unless it explicitly opts into the
    linear/sqrt crossover.
    """
    return market_impact_pct(participation, coefficient, crossover=None)
