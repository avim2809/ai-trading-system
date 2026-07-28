"""LLM-enhanced trader / portfolio manager agent."""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.base import AgentContext
from firm.agents.llm.base_llm_agent import LLMAgentMixin
from firm.agents.trader import TraderAgent
from firm.contracts.models import TradeProposal
from firm.llm.schemas import PortfolioReviewResponse, parse_llm_response

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

        if not self._allow_portfolio_llm():
            return quant_proposal

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

        # Inject past decision outcomes so the agent learns from history.
        memory = inputs.get("memory")
        past_context = memory.get_context() if memory is not None else ""

        prompt = (
            f"Proposed portfolio weights: {targets_str}\n"
            f"Method: {self.allocation_method}\n"
        )
        if past_context:
            prompt += f"\n{past_context}\n"
        if context:
            prompt += f"Research context:\n{context}\n\n"
        prompt += (
            "Review allocation, flag concentration/correlation risks, suggest adjustments. "
            "Return JSON: "
            '{"adjusted_targets": {"SYMBOL": weight, ...}, "notes": "..."}'
        )
        try:
            raw = self._call_llm(
                "You are a portfolio manager reviewing a trade proposal.",
                prompt,
                json_mode=True,
            )
            parsed = parse_llm_response(PortfolioReviewResponse, raw, context="portfolio review")
            if parsed is not None and parsed.adjusted_targets:
                return TradeProposal(
                    asof=quant_proposal.asof,
                    targets=parsed.adjusted_targets,
                    per_strategy=quant_proposal.per_strategy,
                    notes=parsed.notes or quant_proposal.notes,
                )
        except Exception:
            log.warning("LLM trader review failed, using quant proposal", exc_info=True)

        return quant_proposal
