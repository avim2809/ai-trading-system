"""Multi-factor composite equity strategy.

Financial intuition:
    Academic and practitioner research identifies persistent return premia
    associated with value (cheap stocks outperform), quality (profitable/
    low-leverage firms outperform), momentum (winners keep winning), and
    low-volatility (less volatile stocks deliver higher risk-adjusted
    returns).  Combining these factors diversifies across alpha sources
    and smooths returns (Asness, Moskowitz & Pedersen, 2013).

Data inputs:
    - Fundamentals from PitView.fundamentals(): PE ratio, PB ratio, ROE,
      debt-to-equity.
    - Prices from PitView.prices(): for momentum (12-1 month return) and
      realized volatility computation.

Signal logic:
    1. Value score  = cross-sectional z-score of 1/PE + 1/PB (higher = cheaper).
    2. Quality score = z-score of ROE + z-score of inverse debt-to-equity.
    3. Momentum score = z-score of 12-1 month cumulative return.
    4. Low-vol score   = z-score of inverse realized vol (252-day).
    5. Composite = weighted sum of the four factor z-scores.
    6. Final signal score = composite z-score (re-standardized).

Portfolio construction approach:
    Long the highest composite scores, short the lowest.  Factor tilts
    can be adjusted via the factor_weights parameter.

Risk notes:
    Individual factors can underperform for extended periods (value drawdown
    2018-2020).  The composite mitigates this but does not eliminate it.
    Sector concentration is possible; pair with sector-neutral constraints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from firm.contracts.models import Signal
from firm.strategies.base import BaseStrategy, PitView
from firm.strategies.registry import register


def _zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score, dropping NaNs and infs.

    A single bad input (e.g. a zero price feeding a division elsewhere)
    producing +/-inf must not poison every symbol's score: dropna() alone
    doesn't remove inf, and an inf in the series makes std() == inf, which
    slips past the "std == 0 or NaN" guard — every entry then evaluates to
    NaN via inf-involving arithmetic (finite - inf, then dividing by inf).
    """
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    std = s.std()
    if std == 0 or np.isnan(std):
        return s * 0.0
    return ((s - s.mean()) / std).clip(-5, 5)


def _mean_available(parts: list[pd.Series]) -> pd.Series:
    """Per-symbol mean across sub-metric series, skipping missing values.

    A symbol present in only one part gets that value unchanged (not divided
    by the number of potential sub-metrics), so missing data doesn't silently
    dampen a factor.
    """
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def _weighted_composite(
    scores: dict[str, pd.Series], factor_weights: dict[str, float]
) -> pd.Series:
    """Combine per-factor z-scores into one composite, weighted per symbol.

    The weight denominator is tracked PER SYMBOL (only the weights of
    factors that symbol actually has a value for), not as one universe-wide
    total — mirroring _mean_available's own anti-dilution logic one level
    up. Dividing every symbol by the same total_weight regardless of which
    factors it actually had would silently shrink a symbol's lone real
    signal toward zero by weight it never earned (e.g. a symbol with only
    a value-factor score, divided by all 4 factors' combined weight instead
    of just value's).
    """
    all_symbols: set[str] = set()
    for s in scores.values():
        all_symbols.update(s.index)
    if not all_symbols:
        return pd.Series(dtype=float)

    composite = pd.Series(0.0, index=sorted(all_symbols))
    weight_sum = pd.Series(0.0, index=sorted(all_symbols))
    for factor_name, w in factor_weights.items():
        if factor_name in scores:
            factor_scores = scores[factor_name]
            composite = composite.add(factor_scores * w, fill_value=0)
            weight_sum = weight_sum.add(
                pd.Series(w, index=factor_scores.index), fill_value=0
            )

    if (weight_sum == 0).all():
        return pd.Series(dtype=float)
    return (composite / weight_sum.replace(0, np.nan)).dropna()


@register("multi_factor")
class MultiFactorStrategy(BaseStrategy):
    def __init__(self, params: dict | None = None):
        super().__init__("multi_factor", params)

    def generate(self, pit_view: PitView) -> list[Signal]:
        default_weights = {
            "value": 0.25,
            "quality": 0.25,
            "momentum": 0.25,
            "low_vol": 0.25,
        }
        factor_weights: dict[str, float] = self.params.get(
            "factor_weights", default_weights
        )
        mom_lookback: int = self.params.get("momentum_lookback_months", 12)
        mom_skip: int = self.params.get("momentum_skip_months", 1)
        vol_lookback: int = self.params.get("vol_lookback_days", 252)

        universe = pit_view.universe
        if not universe:
            return []

        fund_df = pit_view.fundamentals(symbols=universe)
        prices_df = pit_view.prices(
            symbols=universe, lookback_days=max(vol_lookback, mom_lookback * 21) + 10
        )

        scores: dict[str, pd.Series] = {}

        # --- Value factor ---
        if not fund_df.empty and "pe_ratio" in fund_df.columns:
            latest = fund_df.sort_values("date").groupby("symbol").last()
            inv_pe = (1.0 / latest["pe_ratio"].replace(0, np.nan)).dropna()
            inv_pb = pd.Series(dtype=float)
            if "pb_ratio" in latest.columns:
                inv_pb = (1.0 / latest["pb_ratio"].replace(0, np.nan)).dropna()
            # Average only the sub-metrics each symbol actually has: a symbol
            # with PE but no PB gets its pure PE z-score, not a halved one.
            value_parts = [_zscore(inv_pe)]
            if not inv_pb.empty:
                value_parts.append(_zscore(inv_pb))
            scores["value"] = _mean_available(value_parts)

        # --- Quality factor ---
        if not fund_df.empty and "roe" in fund_df.columns:
            latest = fund_df.sort_values("date").groupby("symbol").last()
            roe_z = _zscore(latest["roe"].dropna())
            inv_de = pd.Series(dtype=float)
            if "debt_to_equity" in latest.columns:
                inv_de = (
                    1.0 / latest["debt_to_equity"].replace(0, np.nan)
                ).dropna()
                inv_de = _zscore(inv_de)
            quality_parts = [roe_z]
            if not inv_de.empty:
                quality_parts.append(inv_de)
            scores["quality"] = _mean_available(quality_parts)

        # --- Momentum factor ---
        if not prices_df.empty:
            pivot = (
                prices_df.pivot_table(
                    index="date", columns="symbol", values="adj_close"
                )
                .sort_index()
            )
            mom_days = mom_lookback * 21
            skip_days = mom_skip * 21
            # Degrade to whatever's actually available (mirroring
            # momentum.py's own strategy) instead of silently computing a
            # much-shorter, unflagged window against the skip_days+21
            # check alone, which only required ~22 days of data regardless
            # of how much shorter than the intended 12-1 month window that is.
            effective_mom_days = min(mom_days, len(pivot))
            if effective_mom_days > skip_days + 21:
                end = len(pivot) - skip_days if skip_days > 0 else len(pivot)
                start = max(0, end - (effective_mom_days - skip_days))
                # A zero/bad price at the start of the window must not
                # propagate +/-inf into the whole factor's z-score (see
                # _zscore's own hardening) — guard the division directly,
                # matching every other factor's .replace(0, np.nan) pattern.
                start_prices = pivot.iloc[start].replace(0, np.nan)
                cum_ret = (pivot.iloc[end - 1] / start_prices) - 1.0
                scores["momentum"] = _zscore(cum_ret.dropna())

        # --- Low-vol factor ---
        if not prices_df.empty:
            pivot = (
                prices_df.pivot_table(
                    index="date", columns="symbol", values="adj_close"
                )
                .sort_index()
            )
            daily_ret = pivot.pct_change()
            vol = daily_ret.iloc[-vol_lookback:].std() * np.sqrt(252)
            inv_vol = (1.0 / vol.replace(0, np.nan)).dropna()
            scores["low_vol"] = _zscore(inv_vol)

        if not scores:
            return []

        composite = _weighted_composite(scores, factor_weights)
        if composite.empty:
            return []

        signals: list[Signal] = []
        for symbol in composite.index:
            raw = float(composite[symbol])
            if np.isnan(raw):
                continue
            factor_detail = {
                fname: float(s.get(symbol, np.nan))
                for fname, s in scores.items()
            }
            signals.append(
                Signal(
                    symbol=str(symbol),
                    strategy="multi_factor",
                    score=raw,
                    confidence=min(abs(raw) / 2.0, 1.0),
                    horizon="21d",
                    asof=pit_view.asof,
                    meta={"factor_scores": factor_detail},
                )
            )
        return signals
