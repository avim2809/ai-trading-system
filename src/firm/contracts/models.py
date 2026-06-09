"""Immutable data contracts shared across the entire firm.

Every inter-agent message type is defined here as a frozen dataclass so that
contracts are lightweight, hashable, and impossible to mutate after creation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class Signal:
    """One strategy's view on one symbol."""

    symbol: str
    strategy: str
    score: float  # normalized z-score or [-1, 1]
    confidence: float  # 0.0 to 1.0
    horizon: str  # e.g. "1d", "5d", "21d"
    asof: datetime
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SignalSet:
    """Analyst output: collection of signals for the universe."""

    domain: str  # "technical", "fundamental", "sentiment"
    asof: datetime
    signals: list[Signal] = field(default_factory=list)


@dataclass(frozen=True)
class Thesis:
    """Bull or bear researcher output."""

    side: str  # "bull" or "bear"
    symbol: str
    conviction: float  # 0.0 to 1.0
    rationale: str
    supporting: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DebateResult:
    """Synthesized output from bull/bear debate."""

    symbol: str
    net_conviction: float  # positive = bullish, negative = bearish
    bull_thesis: Thesis | None = None
    bear_thesis: Thesis | None = None


@dataclass(frozen=True)
class TradeProposal:
    """PM output: target portfolio weights."""

    asof: datetime
    targets: dict[str, float] = field(default_factory=dict)  # symbol -> target weight
    per_strategy: dict[str, dict[str, float]] = field(
        default_factory=dict
    )  # strategy -> {symbol: weight}
    notes: str = ""


@dataclass(frozen=True)
class RiskDecision:
    """Risk manager output."""

    approved: bool
    adjusted_targets: dict[str, float] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExecutionReport:
    """Post-execution summary."""

    fills: list[dict[str, Any]] = field(default_factory=list)
    turnover: float = 0.0
    costs: float = 0.0


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Point-in-time portfolio state."""

    asof: datetime
    holdings: dict[str, float] = field(default_factory=dict)  # symbol -> shares
    weights: dict[str, float] = field(default_factory=dict)  # symbol -> weight
    cash: float = 0.0
    nav: float = 0.0
    per_strategy_pnl: dict[str, float] = field(default_factory=dict)
