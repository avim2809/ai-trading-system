"""Orchestrator – drives the per-bar agent pipeline.

Calls analysts (in parallel) -> bull/bear researchers -> debate ->
trader/PM -> risk approval loop -> execution, managing the blackboard
between steps.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from firm.agents.base import Agent, AgentContext
from firm.agents.blackboard import Blackboard
from firm.contracts.models import TradeProposal

log = logging.getLogger(__name__)


class Orchestrator(Agent):
    """Coordinates the full agent pipeline for one timestep."""

    role = "orchestrator"

    def __init__(
        self,
        analysts: list[Agent],
        bull: Agent,
        bear: Agent,
        debate: Agent,
        trader: Agent,
        risk: Agent,
        execution: Agent,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name="orchestrator", config=config)
        self.analysts = analysts
        self.bull = bull
        self.bear = bear
        self.debate = debate
        self.trader = trader
        self.risk = risk
        self.execution = execution
        self._max_risk_retries: int = (config or {}).get("max_risk_retries", 3)
        # If set, abort the bar when any analyst/strategy failed rather than
        # trading on a silently-truncated signal set.
        self._abort_on_degraded: bool = (config or {}).get("abort_on_degraded", False)

    def run(self, ctx: AgentContext, **inputs: Any) -> tuple[list[dict], Blackboard]:
        """ABC-compliant entry point – delegates to :meth:`step`."""
        context: dict[str, Any] = {
            "pit_view": ctx.pit_view,
            "portfolio": ctx.portfolio,
            "prices": inputs.get("prices", ctx.config.get("prices", {})),
        }
        return self.step(context)

    def step(self, context: dict[str, Any]) -> tuple[list[dict], Blackboard]:
        """Run the full agent pipeline for one timestep.

        Args:
            context: Must contain ``pit_view`` (:class:`PitView`),
                ``portfolio`` (:class:`PortfolioState`), and ``prices``
                (``dict[str, float]``).

        Returns:
            ``(orders_list, blackboard)`` where *orders_list* feeds the
            execution engine.
        """
        pit_view = context["pit_view"]
        portfolio = context.get("portfolio")
        prices: dict[str, float] = context.get("prices", {})

        bb = Blackboard(asof=pit_view.asof)
        ctx = AgentContext(now=pit_view.asof, pit_view=pit_view, portfolio=portfolio)

        # 1. Run analysts in parallel, deterministic merge by domain
        with ThreadPoolExecutor() as pool:
            futures = {pool.submit(analyst.run, ctx): analyst for analyst in self.analysts}
            for future in futures:
                analyst = futures[future]
                try:
                    signal_set = future.result()
                    bb.signal_sets.append(signal_set)
                    # Surface per-strategy failures the analyst swallowed.
                    for err in getattr(analyst, "_last_errors", []) or []:
                        bb.errors.append({"agent": analyst.name, **err})
                except Exception as exc:
                    log.error("Analyst %s failed", analyst.name, exc_info=True)
                    bb.errors.append({"agent": analyst.name, "error": str(exc)})

        bb.signal_sets.sort(key=lambda ss: ss.domain)

        if bb.errors:
            bb.degraded = True
            log.warning("Pipeline degraded: %d signal-source failure(s)", len(bb.errors))
            if self._abort_on_degraded:
                log.warning("Aborting bar (abort_on_degraded=True) – returning empty orders")
                return [], bb

        if not any(ss.signals for ss in bb.signal_sets):
            log.warning("No signals produced – returning empty orders")
            return [], bb

        # 2. Bull + bear researchers
        bull_theses = self.bull.run(ctx, blackboard=bb)
        bear_theses = self.bear.run(ctx, blackboard=bb)
        bb.theses.extend(bull_theses)
        bb.theses.extend(bear_theses)

        # 3. Debate synthesis
        debate_results = self.debate.run(ctx, bull_theses=bull_theses, bear_theses=bear_theses)
        bb.debate_results = debate_results

        if not debate_results:
            log.warning("No debate results – returning empty orders")
            return [], bb

        # 4. Trade proposal
        proposal = self.trader.run(ctx, debate_results=debate_results, blackboard=bb)
        bb.proposal = proposal

        # 5. Risk approval loop
        decision = None
        for attempt in range(self._max_risk_retries):
            decision = self.risk.run(ctx, proposal=proposal, portfolio=portfolio)
            bb.risk_decision = decision
            if decision.approved:
                break
            log.info(
                "Risk veto (attempt %d/%d): %s",
                attempt + 1,
                self._max_risk_retries,
                decision.violations,
            )
            scale = 0.5 ** (attempt + 1)
            new_targets = {s: w * scale for s, w in proposal.targets.items()}
            proposal = TradeProposal(
                asof=proposal.asof,
                targets=new_targets,
                per_strategy=proposal.per_strategy,
                notes=f"Retry {attempt + 1}: scaled by {scale:.4f}",
            )
            bb.proposal = proposal

        if decision is None or not decision.approved:
            log.warning("Proposal rejected after %d attempts", self._max_risk_retries)
            return [], bb

        # 6. Execution
        report = self.execution.run(
            ctx,
            decision=decision,
            portfolio=portfolio,
            prices=prices,
            per_strategy=proposal.per_strategy,
        )
        bb.execution_report = report

        self._collect_llm_usage(bb)

        return report.fills, bb

    def _collect_llm_usage(self, bb: Blackboard) -> None:
        """Aggregate LLM call logs from all agents onto the blackboard."""
        all_agents = list(self.analysts) + [
            self.bull, self.bear, self.debate, self.trader, self.risk,
        ]
        total_tokens = 0
        calls: list[dict] = []
        for agent in all_agents:
            agent_log = getattr(agent, "_llm_log", None)
            if agent_log:
                for entry in agent_log:
                    total_tokens += entry.get("tokens", 0)
                    calls.append({**entry, "agent": getattr(agent, "name", "?")})
        if calls:
            cost_per_1k = 0.003  # rough estimate
            bb.llm_usage = {  # type: ignore[attr-defined]
                "total_tokens": total_tokens,
                "calls": calls,
                "estimated_cost": total_tokens / 1000.0 * cost_per_1k,
            }
