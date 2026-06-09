"""Minimal agent interface for the governance pipeline.

Concrete agents (analysts, bull/bear researchers, PM, risk manager, execution)
implemented in a later phase subclass :class:`Agent`. They are deliberately thin:
each ``run`` is a pure-ish transform from the current :class:`AgentContext` (and
the shared blackboard, passed in later phases) to some contract artifact. Keeping
the surface tiny lets downstream workers build concrete agents independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid import cycles; these are type-only references
    from firm.contracts.models import PortfolioState
    from firm.strategies.base import PitView


@dataclass
class AgentContext:
    """Per-timestep context shared with agents during an orchestration step.

    Attributes:
        now: The decision timestamp for this step.
        pit_view: Read-only point-in-time data accessor bound to ``now``.
        portfolio: Immutable snapshot of the current book.
        config: Arbitrary run configuration (settings/strategy params).
    """

    now: datetime
    pit_view: "PitView | None" = None
    portfolio: "PortfolioState | None" = None
    config: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Base class for all decision agents.

    Subclasses set :attr:`role` and implement :meth:`run`. The return type is
    intentionally ``Any`` at this layer because each agent role produces a
    different contract (``SignalSet``, ``Thesis``, ``TradeProposal``,
    ``RiskDecision``, ``ExecutionReport``).
    """

    #: Short role identifier, e.g. ``"fundamental_analyst"`` or ``"risk"``.
    role: str = "agent"

    def __init__(self, name: str | None = None, config: dict[str, Any] | None = None) -> None:
        self.name = name or self.role
        self.config: dict[str, Any] = dict(config or {})

    @abstractmethod
    def run(self, ctx: AgentContext, **inputs: Any) -> Any:
        """Execute the agent for one timestep.

        Args:
            ctx: The current :class:`AgentContext`.
            **inputs: Upstream artifacts this agent depends on (e.g. an analyst's
                ``signal_sets`` or the PM's ``proposal``).

        Returns:
            A contract artifact appropriate to the agent's role.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"{type(self).__name__}(name={self.name!r}, role={self.role!r})"
