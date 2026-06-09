"""Risk manager agent – has veto power over trade proposals.

Applies a constraint pipeline to the proposed target weights:

1. Per-name position cap (e.g. 5 % max weight per symbol)
2. Gross exposure cap (sum of |weights|)
3. Net exposure cap (sum of weights)
4. Sector concentration limit (if sector data available)
5. Volatility targeting (scale weights to achieve target vol)
6. Drawdown circuit breaker (reduce exposure when drawdown > threshold)

If the cumulative adjustments are too severe the proposal is **vetoed**
(``RiskDecision.approved = False``).
"""

from __future__ import annotations

import logging
from typing import Any

from firm.agents.base import Agent, AgentContext
from firm.contracts.models import RiskDecision, TradeProposal

log = logging.getLogger(__name__)


class RiskAgent(Agent):
    """Constraint-based risk manager with veto authority."""

    role = "risk"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(name="risk_manager", config=config)
        cfg = config or {}
        self.max_position_pct: float = cfg.get("max_position_pct", 0.05)
        self.max_gross_exposure: float = cfg.get("max_gross_exposure", 2.0)
        self.max_net_exposure: float = cfg.get("max_net_exposure", 0.5)
        self.max_sector_pct: float = cfg.get("max_sector_pct", 0.25)
        self.vol_target: float = cfg.get("vol_target", 0.15)
        self.max_drawdown_pct: float = cfg.get("max_drawdown_pct", 0.20)
        self.veto_threshold: float = cfg.get("veto_threshold", 0.5)

    def run(self, ctx: AgentContext, **inputs: Any) -> RiskDecision:
        proposal: TradeProposal = inputs["proposal"]
        portfolio = inputs.get("portfolio")

        targets = dict(proposal.targets)
        original_gross = sum(abs(w) for w in targets.values())
        violations: list[str] = []
        actions: list[str] = []

        targets, v, a = self._clip_position_sizes(targets)
        violations.extend(v)
        actions.extend(a)

        targets, v, a = self._cap_gross_exposure(targets)
        violations.extend(v)
        actions.extend(a)

        targets, v, a = self._cap_net_exposure(targets)
        violations.extend(v)
        actions.extend(a)

        sector_map = inputs.get("sector_map")
        if sector_map:
            targets, v, a = self._cap_sector_concentration(targets, sector_map)
            violations.extend(v)
            actions.extend(a)

        vol_estimates = inputs.get("vol_estimates")
        if vol_estimates:
            targets, v, a = self._vol_targeting(targets, vol_estimates)
            violations.extend(v)
            actions.extend(a)

        targets, v, a = self._drawdown_breaker(targets, portfolio)
        violations.extend(v)
        actions.extend(a)

        adjusted_gross = sum(abs(w) for w in targets.values())
        if original_gross > 1e-10:
            clipping_severity = abs(original_gross - adjusted_gross) / original_gross
        else:
            clipping_severity = 0.0

        if clipping_severity > self.veto_threshold:
            violations.append(
                f"VETO: clipping severity {clipping_severity:.1%} exceeds "
                f"threshold {self.veto_threshold:.1%}"
            )
            actions.append("Proposal vetoed due to excessive constraint violations")
            return RiskDecision(
                approved=False,
                adjusted_targets=targets,
                violations=violations,
                actions=actions,
            )

        return RiskDecision(
            approved=True,
            adjusted_targets=targets,
            violations=violations,
            actions=actions,
        )

    # ------------------------------------------------------------------
    # constraint helpers – each returns (targets, violations, actions)
    # ------------------------------------------------------------------

    def _clip_position_sizes(
        self, targets: dict[str, float]
    ) -> tuple[dict[str, float], list[str], list[str]]:
        violations: list[str] = []
        actions: list[str] = []
        clipped: dict[str, float] = {}
        for sym, w in targets.items():
            if abs(w) > self.max_position_pct:
                violations.append(
                    f"{sym} weight {w:.3f} exceeds per-name cap {self.max_position_pct}"
                )
                capped = self.max_position_pct if w > 0 else -self.max_position_pct
                actions.append(f"Clipped {sym} from {w:.4f} to {capped:.4f}")
                clipped[sym] = capped
            else:
                clipped[sym] = w
        return clipped, violations, actions

    def _cap_gross_exposure(
        self, targets: dict[str, float]
    ) -> tuple[dict[str, float], list[str], list[str]]:
        gross = sum(abs(w) for w in targets.values())
        if gross <= self.max_gross_exposure:
            return targets, [], []
        scale = self.max_gross_exposure / gross
        scaled = {s: w * scale for s, w in targets.items()}
        return (
            scaled,
            [f"Gross exposure {gross:.3f} exceeds cap {self.max_gross_exposure}"],
            [f"Scaled all weights by {scale:.4f}"],
        )

    def _cap_net_exposure(
        self, targets: dict[str, float]
    ) -> tuple[dict[str, float], list[str], list[str]]:
        net = sum(targets.values())
        if abs(net) <= self.max_net_exposure:
            return targets, [], []

        excess = abs(net) - self.max_net_exposure
        sign = 1.0 if net > 0 else -1.0
        same_side = {s: w for s, w in targets.items() if (w * sign) > 0}
        total_same = sum(abs(w) for w in same_side.values())

        adjusted = dict(targets)
        if total_same > 0:
            for sym, w in same_side.items():
                reduction = excess * (abs(w) / total_same) * sign
                adjusted[sym] = w - reduction

        return (
            adjusted,
            [f"Net exposure {net:.3f} exceeds cap {self.max_net_exposure}"],
            [f"Reduced same-side weights to bring net within bounds"],
        )

    def _cap_sector_concentration(
        self,
        targets: dict[str, float],
        sector_map: dict[str, str],
    ) -> tuple[dict[str, float], list[str], list[str]]:
        sector_weights: dict[str, float] = {}
        for sym, w in targets.items():
            sector = sector_map.get(sym, "unknown")
            sector_weights[sector] = sector_weights.get(sector, 0.0) + abs(w)

        violations: list[str] = []
        actions: list[str] = []
        adjusted = dict(targets)

        for sector, total_w in sector_weights.items():
            if total_w > self.max_sector_pct:
                scale = self.max_sector_pct / total_w
                for sym, w in list(adjusted.items()):
                    if sector_map.get(sym, "unknown") == sector:
                        adjusted[sym] = w * scale
                violations.append(
                    f"Sector '{sector}' weight {total_w:.3f} exceeds cap {self.max_sector_pct}"
                )
                actions.append(f"Scaled sector '{sector}' by {scale:.4f}")

        return adjusted, violations, actions

    def _vol_targeting(
        self,
        targets: dict[str, float],
        vol_estimates: dict[str, float],
    ) -> tuple[dict[str, float], list[str], list[str]]:
        """Scale weights so that estimated portfolio vol ~ vol_target.

        Uses a simplified diagonal-covariance approximation:
        port_vol = sqrt(sum(w_i^2 * vol_i^2)).
        """
        port_var = sum(
            (targets.get(s, 0.0) ** 2) * (vol_estimates.get(s, 0.20) ** 2)
            for s in targets
        )
        port_vol = port_var**0.5
        if port_vol < 1e-10:
            return targets, [], []
        scale = self.vol_target / port_vol
        if abs(scale - 1.0) < 0.01:
            return targets, [], []
        scaled = {s: w * scale for s, w in targets.items()}
        return (
            scaled,
            [f"Portfolio vol {port_vol:.3f} vs target {self.vol_target}"],
            [f"Scaled weights by {scale:.4f} for vol targeting"],
        )

    def _drawdown_breaker(
        self,
        targets: dict[str, float],
        portfolio: Any,
    ) -> tuple[dict[str, float], list[str], list[str]]:
        if portfolio is None or not getattr(portfolio, "history", None):
            return targets, [], []
        navs = [snap.nav for snap in portfolio.history]
        if not navs:
            return targets, [], []

        peak = max(navs)
        current = navs[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0.0

        if drawdown <= self.max_drawdown_pct:
            return targets, [], []

        scale = 0.5
        scaled = {s: w * scale for s, w in targets.items()}
        return (
            scaled,
            [f"Drawdown {drawdown:.1%} exceeds threshold {self.max_drawdown_pct:.1%}"],
            [f"Reduced exposure by {1 - scale:.0%} (drawdown circuit breaker)"],
        )
