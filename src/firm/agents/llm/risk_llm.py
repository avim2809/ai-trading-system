"""LLM-enhanced risk manager agent."""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.base import AgentContext
from firm.agents.llm.base_llm_agent import LLMAgentMixin
from firm.agents.risk import RiskAgent
from firm.contracts.models import RiskDecision

log = logging.getLogger(__name__)


class LLMRiskAgent(RiskAgent, LLMAgentMixin):
    """Runs quant risk checks then asks LLM to identify non-quantitative risks."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> None:
        RiskAgent.__init__(self, config=config)
        LLMAgentMixin.__init__(self, llm_config=llm_config)

    def run(self, ctx: AgentContext, **inputs: Any) -> RiskDecision:
        quant_decision = RiskAgent.run(self, ctx, **inputs)

        proposal = inputs.get("proposal")
        symbols = list((proposal.targets if proposal else {}).keys())
        if not symbols:
            return quant_decision

        all_context_parts: list[str] = []
        for sym in symbols[:5]:
            part = self._retrieve_context(
                sym,
                f"regulatory litigation macro event risk for {sym}",
                collections=["news", "sec_filings"],
                asof=ctx.now,
            )
            if part:
                all_context_parts.append(part)

        # Inject past decision outcomes so the agent considers historical risk events.
        memory = inputs.get("memory")
        past_context = memory.get_context() if memory is not None else ""

        if not all_context_parts and not past_context:
            return quant_decision

        context = self._compress("\n\n".join(all_context_parts)) if all_context_parts else ""
        violations_str = "; ".join(quant_decision.violations) if quant_decision.violations else "None"
        prompt = (
            f"Symbols: {', '.join(symbols)}\n"
            f"Quant violations: {violations_str}\n"
            f"Quant approved: {quant_decision.approved}\n"
        )
        if past_context:
            prompt += f"\n{past_context}\n"
        if context:
            prompt += f"Context:\n{context}\n\n"
        prompt += (
            "Identify non-quantitative risks (regulatory, litigation, macro events). "
            "Return JSON: "
            '{"additional_violations": ["..."], "additional_actions": ["..."], "override_approval": null or bool}'
        )
        try:
            result = self._call_llm(
                "You are a risk manager assessing non-quantitative risks.",
                prompt,
                json_mode=True,
            )
            extra_violations = result.get("additional_violations", [])
            extra_actions = result.get("additional_actions", [])
            override = result.get("override_approval")

            merged_violations = list(quant_decision.violations) + [
                f"[LLM] {v}" for v in extra_violations if isinstance(v, str)
            ]
            merged_actions = list(quant_decision.actions) + [
                f"[LLM] {a}" for a in extra_actions if isinstance(a, str)
            ]
            approved = override if isinstance(override, bool) else quant_decision.approved

            return RiskDecision(
                approved=approved,
                adjusted_targets=quant_decision.adjusted_targets,
                violations=merged_violations,
                actions=merged_actions,
            )
        except Exception:
            log.warning("LLM risk review failed, using quant decision", exc_info=True)

        return quant_decision
