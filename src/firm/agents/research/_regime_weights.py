"""Per-strategy score multipliers conditioned on the prevailing market regime.

Complements :class:`firm.agents.risk.RiskAgent`'s ``regime_overlay``, which
scales *gross exposure* after sizing. This module instead damps or boosts each
strategy's raw signal *before* bull/bear researchers combine them — the
``regime-conditional-weighting`` research item from the audit remediation plan.

Disabled by default (``strategy_regime_weights.enabled: false``). When enabled,
the orchestrator detects the market regime once per cycle (same
:class:`~firm.regime.detector.MarketRegimeDetector` as the risk overlay) and
research combination applies confidence-blended per-strategy multipliers::

    effective = 1 + (target - 1) * confidence

so an uncertain regime read barely moves weights while a confident one applies
the full playbook factor.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from firm.contracts.models import Signal

log = logging.getLogger(__name__)

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "benchmark_symbol": "SPY",
    "n_states": 3,
    "lookback_days": 252,
    "retrain_frequency": 21,
    # ``weights[regime_label][strategy_name]`` → multiplier on raw score.
    # Strategies omitted for a regime keep multiplier 1.0.
    "weights": {
        "Bull": {},
        "Bear": {},
        "Chop": {},
    },
    "min_multiplier": 0.0,
    "max_multiplier": 2.0,
}


def _effective_multiplier(
    target: float,
    confidence: float,
    *,
    min_multiplier: float,
    max_multiplier: float,
) -> float:
    raw = 1.0 + (target - 1.0) * confidence
    return max(min_multiplier, min(max_multiplier, raw))


def apply_strategy_regime_weights(
    signals: list[Signal],
    regime_state: Any | None,
    config: dict[str, Any] | None,
) -> list[Signal]:
    """Scale each signal's score by a regime-conditional strategy multiplier."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if not cfg.get("enabled"):
        return signals
    if regime_state is None:
        log.debug("strategy_regime_weights: no regime state — no-op")
        return signals

    label = str(getattr(regime_state, "label", "") or "")
    confidence = float(getattr(regime_state, "confidence", 0.0) or 0.0)
    regime_weights = (cfg.get("weights") or {}).get(label) or {}
    if not regime_weights:
        log.debug(
            "strategy_regime_weights: no weights for regime=%s — no-op",
            label,
        )
        return signals

    min_mult = float(cfg.get("min_multiplier", 0.0))
    max_mult = float(cfg.get("max_multiplier", 2.0))
    out: list[Signal] = []
    for sig in signals:
        target = float(regime_weights.get(sig.strategy, 1.0))
        if abs(target - 1.0) < 1e-9:
            out.append(sig)
            continue
        mult = _effective_multiplier(
            target, confidence, min_multiplier=min_mult, max_multiplier=max_mult,
        )
        if abs(mult - 1.0) < 1e-9:
            out.append(sig)
            continue
        log.debug(
            "strategy_regime_weights: %s/%s regime=%s conf=%.2f mult=%.3f",
            sig.strategy, sig.symbol, label, confidence, mult,
        )
        out.append(replace(sig, score=sig.score * mult))
    return out
