"""Per-timestep shared data store for inter-agent communication.

The Blackboard holds all artifacts produced during a single orchestration step
(one rebalance bar): signal sets, theses, debate results, trade proposals,
risk decisions, and the final execution report.  Agents read from and write to
it, keeping a full audit trail without direct coupling.
"""

from __future__ import annotations

from datetime import datetime

from firm.contracts.models import (
    DebateResult,
    ExecutionReport,
    RiskDecision,
    Signal,
    SignalSet,
    Thesis,
    TradeProposal,
)


class Blackboard:
    """Per-timestep shared data store for agent communication."""

    def __init__(self, asof: datetime) -> None:
        self.asof = asof
        self.signal_sets: list[SignalSet] = []
        self.theses: list[Thesis] = []
        self.debate_results: list[DebateResult] = []
        self.proposal: TradeProposal | None = None
        self.risk_decision: RiskDecision | None = None
        self.execution_report: ExecutionReport | None = None

    def get_signals_by_symbol(self, symbol: str) -> list[Signal]:
        """Return all signals across every domain for *symbol*."""
        return [
            sig
            for ss in self.signal_sets
            for sig in ss.signals
            if sig.symbol == symbol
        ]

    def get_signals_by_domain(self, domain: str) -> list[Signal]:
        """Return all signals from signal-sets matching *domain*."""
        return [
            sig
            for ss in self.signal_sets
            if ss.domain == domain
            for sig in ss.signals
        ]

    def get_all_symbols(self) -> set[str]:
        """Return the union of symbols referenced across all signal sets."""
        return {sig.symbol for ss in self.signal_sets for sig in ss.signals}

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Blackboard(asof={self.asof!r}, "
            f"signal_sets={len(self.signal_sets)}, "
            f"theses={len(self.theses)}, "
            f"debate_results={len(self.debate_results)}, "
            f"proposal={'set' if self.proposal else 'None'}, "
            f"risk_decision={'set' if self.risk_decision else 'None'}, "
            f"execution_report={'set' if self.execution_report else 'None'})"
        )
