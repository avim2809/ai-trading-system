"""Generic per-strategy rolling-Sharpe circuit breaker.

Complementary to, not a replacement for, the ``optimal`` (inverse-covariance)
signal combination: ``optimal`` down-weights *correlated/high-variance*
strategies (:func:`firm.agents.analysts.optimal_signal_weights` weights by
``Σ⁻¹·1``) but has no notion of the *sign* of a strategy's edge — a strategy
with a steady, low-variance NEGATIVE mean return can still receive material
minimum-variance weight. This is not hypothetical: the portfolio-construction
diagnosis (``docs/portfolio_construction_diagnosis.md``) found ``regime_hmm``
had a negative Sharpe in 6/6 independently-tested historical windows, and a
signal-logic fix targeting the suspected root cause (thin-margin/label-
switching state labelling — see ``firm.regime.model.GaussianRegimeModel``)
did not reliably fix it, implying the drag is more structural than a labelling
artifact.

This module tracks each strategy's own trailing realized Sharpe (from
:class:`firm.portfolio.attribution.PerformanceAttribution`, threaded through
as ``ctx.strategy_returns``) and progressively damps — never hard-zeroes — its
raw score contribution into signal combination once that trailing Sharpe is
persistently and materially negative. This is analogous to the position-level
kill switch (``firm.live.engine``), but scoped to a single strategy's signal
weight rather than halting the whole book.

Fails safe: disabled by default (opt-in via ``config['strategy_circuit_breaker']``,
mirroring the ``regime_overlay`` convention), and any strategy without enough
track record is left undamped rather than guessed at.
"""

from __future__ import annotations

import logging
import math
from dataclasses import replace
from typing import Any

import pandas as pd

from firm.contracts.models import Signal

log = logging.getLogger(__name__)

#: Default knobs, overridable per-key via config['strategy_circuit_breaker'].
DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    # Trailing window (trading days) the rolling Sharpe is computed over.
    "lookback_days": 60,
    # Minimum observations required before a strategy is judged at all.
    "min_track_record_days": 20,
    # Trailing annualized Sharpe at/below which damping begins.
    "trigger_sharpe": -0.5,
    # Trailing annualized Sharpe at/below which damping is fully floored.
    "full_cutoff_sharpe": -1.5,
    # Minimum multiplier ever applied — signal is damped, never fully zeroed,
    # so a strategy can recover organically rather than being locked out.
    "damping_floor": 0.25,
}


def _annualized_sharpe(returns: pd.Series) -> float | None:
    r = returns.dropna()
    if len(r) < 2:
        return None
    std = float(r.std(ddof=1))
    if not math.isfinite(std) or std <= 0:
        return None
    mean = float(r.mean())
    return mean / std * math.sqrt(252)


def compute_strategy_damping(
    strategy_returns: dict[str, pd.Series] | None,
    config: dict[str, Any] | None,
) -> dict[str, float]:
    """Return ``{strategy: damping_factor}`` for degraded strategies only.

    Strategies not present in the returned dict should be treated as
    undamped (factor ``1.0``) by callers — omission, not an explicit 1.0
    entry, is used so the common case (no strategy currently gated) is a
    cheap empty-dict check.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if not cfg["enabled"] or not strategy_returns:
        return {}

    lookback = int(cfg["lookback_days"])
    min_track = int(cfg["min_track_record_days"])
    trigger = float(cfg["trigger_sharpe"])
    cutoff = float(cfg["full_cutoff_sharpe"])
    floor = float(cfg["damping_floor"])
    if cutoff >= trigger:
        log.warning(
            "strategy_circuit_breaker: full_cutoff_sharpe (%.2f) must be < "
            "trigger_sharpe (%.2f); disabling this cycle to avoid a bad gate",
            cutoff, trigger,
        )
        return {}

    damping: dict[str, float] = {}
    for strategy, series in strategy_returns.items():
        r = series.dropna() if hasattr(series, "dropna") else pd.Series(series).dropna()
        if len(r) < min_track:
            continue
        window = r.iloc[-lookback:] if len(r) > lookback else r
        sharpe = _annualized_sharpe(window)
        if sharpe is None or sharpe >= trigger:
            continue
        if sharpe <= cutoff:
            factor = floor
        else:
            frac = (trigger - sharpe) / (trigger - cutoff)
            factor = 1.0 - frac * (1.0 - floor)
        damping[strategy] = factor
        log.warning(
            "Strategy circuit breaker: %s trailing %d-day Sharpe=%.2f is "
            "below trigger=%.2f — damping its signal contribution to %.2fx "
            "(floor=%.2fx at cutoff=%.2f)",
            strategy, len(window), sharpe, trigger, factor, floor, cutoff,
        )
    return damping


def apply_circuit_breaker(
    signals: list[Signal],
    strategy_returns: dict[str, pd.Series] | None,
    config: dict[str, Any] | None,
) -> list[Signal]:
    """Return *signals* with any degraded strategies' scores damped.

    Pure/side-effect-free: returns a new list, never mutates *signals*.
    Applied upstream of both the ``confidence`` and ``optimal`` combination
    paths so it works uniformly regardless of which is active.
    """
    damping = compute_strategy_damping(strategy_returns, config)
    if not damping:
        return signals
    return [
        replace(s, score=s.score * damping[s.strategy]) if s.strategy in damping else s
        for s in signals
    ]
