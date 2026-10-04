"""EWMAC trend forecasts (credibility plan P3-02).

Source: plan P3-02 and A3; Carver's EWMAC continuous forecast (*Systematic Trading*,
*Advanced Futures Trading Strategies*); time-series momentum in Moskowitz, Ooi and Pedersen
(2012) and Hurst, Ooi and Pedersen (2017).

    raw_t = (EWMA_fast(price) - EWMA_slow(price)) / sigma_price_t      slow = 4 * fast
    f_t   = clip(raw_t * scalar, floor, cap)                           cap = 20

``sigma_price`` is the price-unit daily vol from ``firm.signals.vol.price_unit_vol`` and must be
the vol known at t: the forecast at close t is traded at t+1 (P3-09). EWMAs use
``ewm(span=n, adjust=False)``; NaN prices are skipped, never filled.

Scalars and the speed filter on REAL data are produced only by P3-11. This module ships the
estimators and is tested on synthetic data. The pooled scalar is closed form on the uncapped,
unfloored signed raw forecast: ``target_abs / mean|raw|``, one number across instruments, never
per instrument, never tuned for performance.

The ETF path keeps per-rule forecasts SIGNED (``floor=-FORECAST_CAP``); only the COMBINED
forecast is floored at 0, in P3-05 ``combine_forecasts(floor=0.0)``. Flooring each rule first
would put a long bias into the book (rules at -10 and +10 would give +5 instead of 0) and bias
the FDM. ``floor=0.0`` here is for single-rule diagnostics only.

Speed selection is by cost only: a speed is kept for an instrument iff its annual cost in Sharpe
units is at most ``max_cost_fraction`` (1/3) of ``expected_rule_sharpe``. That Sharpe must be a
PRE-REGISTERED constant (the charter's haircut expected Sharpe, P5-02), never computed from
backtest P&L. ``cost_per_trade`` is a return-units cost per unit of turnover from the P2-04 model
(``firm.costs.model``); this module does not import it.
"""

from __future__ import annotations

import logging

import pandas as pd

from firm.signals.vol import TRADING_DAYS_PER_YEAR

logger = logging.getLogger(__name__)

# Source plan P3-02 / Carver: slow = 4 * fast. Do not extend or select on performance.
SPEEDS: tuple[tuple[int, int], ...] = ((2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256))
FORECAST_CAP = 20.0  # Carver: average |f| = 10, cap 20
DEFAULT_MAX_COST_FRACTION = 1.0 / 3.0  # source plan: annual cost above 1/3 of expected rule Sharpe drops the speed


def ewmac_raw(price: pd.Series, sigma_price: pd.Series, fast: int, slow: int | None = None) -> pd.Series:
    """(EWMA_fast(price) - EWMA_slow(price)) / sigma_price; slow defaults to 4*fast."""
    slow = 4 * fast if slow is None else slow
    if fast < 1 or slow <= fast:
        raise ValueError("need 1 <= fast < slow")
    price = price.astype(float)
    diff = price.ewm(span=fast, adjust=False).mean() - price.ewm(span=slow, adjust=False).mean()
    return (diff / sigma_price.where(sigma_price > 0)).where(price.notna())


def ewmac_forecast(
    price: pd.Series,
    sigma_price: pd.Series,
    fast: int,
    scalar: float,
    slow: int | None = None,
    cap: float = FORECAST_CAP,
    floor: float | None = -FORECAST_CAP,
) -> pd.Series:
    """raw * scalar, clipped to [floor, cap]. ``floor=0.0`` is for single-rule diagnostics only."""
    return (ewmac_raw(price, sigma_price, fast, slow) * scalar).clip(lower=floor, upper=cap)


def estimate_pooled_scalar(
    raw_by_instrument: dict[str, pd.Series],
    target_abs: float = 10.0,
    start=None,
    end=None,
    max_research_date=None,
) -> float:
    """Single scalar with pooled mean |raw*scalar| == target_abs over [start, end].

    ``raw`` is the uncapped, unfloored signed raw forecast, so the scalar is the closed form
    ``target_abs / mean|raw|``. ``start`` and ``end`` are required and must lie in the research
    window: ``end`` later than the last observation of any series raises, as does ``end`` later
    than ``max_research_date`` when the caller passes it (from ``config/research_freeze.yaml``).
    """
    if start is None or end is None:
        raise ValueError("start and end are required (research window)")
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end < start:
        raise ValueError("end before start")
    if max_research_date is not None and end > pd.Timestamp(max_research_date):
        raise ValueError(f"end {end.date()} is after max_research_date {pd.Timestamp(max_research_date).date()}")
    if not raw_by_instrument:
        raise ValueError("no instruments")
    parts = []
    for name, raw in raw_by_instrument.items():
        if len(raw) == 0 or end > raw.index.max():
            raise ValueError(f"end {end.date()} is beyond the data of {name}")
        parts.append(raw.loc[start:end].dropna().abs())
    pooled = pd.concat(parts)
    mean_abs = float(pooled.mean()) if len(pooled) else float("nan")
    if not mean_abs > 0:
        raise ValueError("pooled mean |raw| is not positive")
    return target_abs / mean_abs


def forecast_turnover(forecast: pd.Series) -> float:
    """Annualised turnover of the implied position: 256 * mean(|df|) / mean(|f|)."""
    f = forecast.dropna()
    mean_abs = float(f.abs().mean()) if len(f) else float("nan")
    if not mean_abs > 0:
        raise ValueError("mean |forecast| is not positive")
    return TRADING_DAYS_PER_YEAR * float(f.diff().abs().mean()) / mean_abs


def speed_cost_sharpe(turnover: float, cost_per_trade: float, sigma_pct: float) -> float:
    """Annual cost in Sharpe units: turnover * cost_per_trade / sigma_pct (author's reading of the source rule)."""
    if sigma_pct <= 0:
        raise ValueError("sigma_pct must be positive")
    return turnover * cost_per_trade / sigma_pct


def select_speeds(
    candidates: dict[str, dict[tuple[int, int], dict]],
    expected_rule_sharpe: float,
    max_cost_fraction: float = DEFAULT_MAX_COST_FRACTION,
) -> dict[str, list[tuple[int, int]]]:
    """Per instrument, keep a speed iff its annual cost in Sharpe units <= max_cost_fraction * expected_rule_sharpe.

    ``candidates[instrument][speed] = {turnover, cost_per_trade, sigma_pct}``. No return/P&L input.
    ``expected_rule_sharpe`` is a pre-registered constant, never measured on the backtest.
    """
    if not expected_rule_sharpe > 0 or not max_cost_fraction > 0:
        raise ValueError("expected_rule_sharpe and max_cost_fraction must be positive")
    limit = max_cost_fraction * expected_rule_sharpe
    out: dict[str, list[tuple[int, int]]] = {}
    for inst, speeds in candidates.items():
        out[inst] = [
            sp
            for sp, d in speeds.items()
            if speed_cost_sharpe(d["turnover"], d["cost_per_trade"], d["sigma_pct"]) <= limit + 1e-12
        ]
        if len(out[inst]) < len(speeds):
            logger.debug("select_speeds %s: dropped %d of %d speeds", inst, len(speeds) - len(out[inst]), len(speeds))
    return out
