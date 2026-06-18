"""LLM-enhanced trader / portfolio manager agent."""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.base import AgentContext
from firm.agents.llm.base_llm_agent import LLMAgentMixin
from firm.agents.trader import TraderAgent
from firm.contracts.models import TradeProposal

log = logging.getLogger(__name__)


class LLMTraderAgent(TraderAgent, LLMAgentMixin):
    """Runs quant allocation then asks LLM to review and adjust weights."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        llm_config: dict[str, Any] | None = None,
    ) -> None:
        TraderAgent.__init__(self, config=config)
        LLMAgentMixin.__init__(self, llm_config=llm_config)

    def run(self, ctx: AgentContext, **inputs: Any) -> TradeProposal:
        quant_proposal = TraderAgent.run(self, ctx, **inputs)

        targets_str = ", ".join(
            f"{sym}: {w:.4f}" for sym, w in sorted(quant_proposal.targets.items())
        )
        rag_context = self._retrieve_context(
            "PORTFOLIO",
            "portfolio allocation concentration correlation risk",
            collections=["research", "system_docs"],
            asof=ctx.now,
        )
        context = self._compress(rag_context) if rag_context else ""
        prompt = (
            f"Proposed portfolio weights: {targets_str}\n"
            f"Method: {self.allocation_method}\n"
        )
        if context:
            prompt += f"Research context:\n{context}\n\n"
        prompt += (
            "Review allocation, flag concentration/correlation risks, suggest adjustments. "
            "Return JSON: "
            '{"adjusted_targets": {"SYMBOL": weight, ...}, "notes": "..."}'
        )
        try:
            result = self._call_llm(
                "You are a portfolio manager reviewing a trade proposal.",
                prompt,
                json_mode=True,
            )
            adjusted = result.get("adjusted_targets")
            if isinstance(adjusted, dict) and adjusted:
                return TradeProposal(
                    asof=quant_proposal.asof,
                    targets={k: float(v) for k, v in adjusted.items()},
                    per_strategy=quant_proposal.per_strategy,
                    notes=result.get("notes", quant_proposal.notes),
                )
        except Exception:
            log.debug("LLM trader review failed, using quant proposal", exc_info=True)

        return quant_proposal
