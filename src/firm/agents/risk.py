"""Risk manager agent – has veto power over trade proposals.

Applies a constraint pipeline to the proposed target weights:

1. Per-name position cap (e.g. 5 % max weight per symbol)
2. Gross exposure cap (sum of |weights|)
3. Net exposure cap (sum of weights)
4. Sector concentration limit (if sector data available)
5. ADV/participation-rate liquidity cap (if point-in-time price data available)
6. Volatility targeting (scale weights to achieve target vol)
7. Drawdown circuit breaker (reduce exposure when drawdown > threshold)

If the cumulative adjustments are too severe the proposal is **vetoed**
(``RiskDecision.approved = False``).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from firm.agents._liquidity import estimate_adv_dollars
from firm.agents.base import Agent, AgentContext
from firm.contracts.models import RiskDecision, TradeProposal
from firm.eval.metrics import conditional_value_at_risk

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
        # vol_target above had a numeric default and was *always* set in
        # every deployed config, but _vol_targeting only ever ran when the
        # caller separately passed inputs["vol_estimates"] -- nothing in the
        # orchestrator ever did, so this was silently dead code: config/
        # live.yaml documented "12% annualised vol target - conservative
        # live baseline" as if it were an active control while it did
        # nothing. Fixed to self-compute vol estimates from ctx.pit_view
        # (matching _cap_correlated_exposure/_cvar_overlay's own pattern
        # below, both of which are already self-sufficient this way) instead
        # of depending on an input nothing supplies. Given an explicit
        # opt-in flag anyway, consistent with every other risk-bearing
        # toggle in this class defaulting to off (correlation_threshold,
        # cvar_limit, regime_overlay) -- this is a behavior change (it will
        # actually de-risk now) even though it looks like a bugfix, so it
        # gets the same "must be turned on deliberately" treatment.
        self.vol_targeting_enabled: bool = bool(cfg.get("vol_targeting_enabled", False))
        self.vol_lookback_days: int = int(cfg.get("vol_lookback_days", 60))
        self.max_drawdown_pct: float = cfg.get("max_drawdown_pct", 0.20)
        self.veto_threshold: float = cfg.get("veto_threshold", 0.5)
        # Optional static sector map used when none is supplied per-call.
        self.sector_map: dict[str, str] = cfg.get("sector_map", {})

        # Optional ADV/participation-rate liquidity cap (disabled unless
        # explicitly configured — see config/live.yaml risk.max_participation_pct).
        # Caps the implied trade size (not the resting position) at a fraction
        # of a symbol's trailing average daily dollar volume: a target weight
        # that's fully achievable in a flat-cost backtest may be unfillable
        # (or move the market) live if it represents e.g. 50% of a stock's
        # ADV. Requires ctx.pit_view for OHLCV history, so it's a no-op
        # wherever that isn't wired up (most unit tests, some backtests).
        self.max_participation_pct: float | None = cfg.get("max_participation_pct")
        self.adv_lookback_days: int = int(cfg.get("adv_lookback_days", 20))

        # Optional pairwise-correlation concentration cap (disabled unless
        # explicitly configured). Sector tags alone miss cross-sector
        # correlated risk (e.g. two rate-sensitive names in different GICS
        # sectors) and don't distinguish a diversifying long/short hedge
        # from two reinforcing same-direction bets — this catches the
        # latter directly from realised return correlation. Requires
        # ctx.pit_view, so it's a no-op wherever that isn't wired up.
        self.correlation_threshold: float | None = cfg.get("correlation_threshold")
        self.max_correlated_pair_pct: float = cfg.get("max_correlated_pair_pct", 0.25)
        self.correlation_lookback_days: int = int(cfg.get("correlation_lookback_days", 60))

        # Optional CVaR (Conditional Value-at-Risk / Expected Shortfall) tail-
        # risk overlay (disabled unless explicitly configured). Complements
        # ``_vol_targeting``'s diagonal-covariance vol estimate, which has no
        # notion of tail shape — two books with identical variance can have
        # very different downside severity. Requires ctx.pit_view, so it's a
        # no-op wherever that isn't wired up. See
        # firm.eval.metrics.conditional_value_at_risk.
        self.cvar_limit: float | None = cfg.get("cvar_limit")
        self.cvar_confidence: float = cfg.get("cvar_confidence", 0.95)
        self.cvar_lookback_days: int = int(cfg.get("cvar_lookback_days", 60))

        # Optional HMM market-regime overlay (off by default).  When enabled,
        # gross exposure is scaled by the prevailing market regime per the
        # playbooks in the regime design doc (Bull → lever up, Bear/Chop →
        # de-risk).  The detector reads the point-in-time view on ctx, so no
        # orchestrator/contract changes are required.
        overlay_cfg = cfg.get("regime_overlay", {}) or {}
        self.regime_overlay_enabled: bool = bool(overlay_cfg.get("enabled", False))
        self._regime_overlay_cfg: dict[str, Any] = overlay_cfg
        # exposure_map: regime label -> target gross-scale at full confidence.
        # The effective scale is blended by posterior confidence so a partial
        # regime update produces a partial sizing change (regime-lag, §6.1).
        self.regime_exposure_map: dict[str, float] = overlay_cfg.get(
            "exposure_map", {"Bull": 1.5, "Bear": 0.5, "Chop": 0.25}
        )
        # Separation-based damping, mirroring the per-symbol regime_hmm
        # strategy's guard against label-switching (see
        # HMMRegimeStrategy.generate's docstring) — confidence blending alone
        # doesn't protect against a *label* flip near a retrain boundary
        # where two states are barely distinguishable: the detector can be
        # highly "confident" in a label that's really noise-level apart from
        # its neighbour, applying the full playbook factor to a read that
        # will likely flip again next retrain. Same param names/defaults as
        # regime_hmm.py for consistency.
        self._regime_min_state_separation: float = float(
            overlay_cfg.get("min_state_separation", 0.5)
        )
        self._regime_separation_damping_floor: float = float(
            overlay_cfg.get("separation_damping_floor", 0.15)
        )
        self._regime_detector = None

        # Optional calendar/seasonality exposure overlay (off by default).
        # PART 3 of the remediation plan: `seasonality` used to run as an
        # ordinary strategy, but its own docstring says it's a market-wide
        # timing signal (one scalar per day, applied identically to every
        # symbol) -- fundamentally a different *shape* of signal than the
        # per-symbol stock-picker pipeline expects. zscore_signals'
        # zero-cross-sectional-std pass-through (see agents/analysts/
        # __init__.py) let that identical score survive unchanged into
        # combine_signals_by_symbol and TraderAgent's sizing as if it were
        # real per-symbol information -- not "silently disabled" as one
        # early research report guessed (confirmed by reading the actual
        # zscore_signals code: std<1e-10 passes the raw score through,
        # it does not zero it), but genuinely the wrong mechanism for a
        # signal that can't discriminate between symbols. Rerouted here as
        # a multiplicative gross-exposure scalar, the same pattern
        # `_regime_exposure_overlay` already uses for the HMM regime
        # signal -- a new sibling method, not folded into `regime_state`'s
        # discrete label/confidence contract, since this signal is
        # continuous in [-1, 1], not a 3-state label.
        seasonality_cfg = cfg.get("seasonality_overlay", {}) or {}
        self.seasonality_overlay_enabled: bool = bool(seasonality_cfg.get("enabled", False))
        # Scale applied to the raw [-1, 1] seasonality score to get an
        # exposure multiplier: effective = 1 + scale * score. 0.15 caps the
        # tilt at ±15% gross exposure at the signal's most extreme reading --
        # seasonality.py's own docstring calls this "a supplementary
        # overlay, not a primary alpha source," so this stays modest by
        # default, well inside the regime overlay's own factor range.
        self.seasonality_overlay_scale: float = float(seasonality_cfg.get("scale", 0.15))
        self._seasonality_params: dict[str, Any] = {
            "tom_days_before": seasonality_cfg.get("tom_days_before", 1),
            "tom_days_after": seasonality_cfg.get("tom_days_after", 3),
        }

        # Optional macro/rate exposure overlay (off by default). PART 3
        # Phase 4 of the remediation plan: none of the 10 live strategies
        # capture interest-rate direction/yield-curve shape -- the primary
        # driver of the 2022 bear market (fold 2 in every walk-forward audit
        # this session, consistently the worst OOS result regardless of
        # combination mechanism). T10Y2Y (the 10Y-2Y Treasury spread, a
        # heavily documented recession-warning signal) is fetched via
        # firm.data.providers.fred + cached in combined/macro
        # (scripts/backfill_macro.py) -- see ctx.pit_view.macro(). Same
        # multiplicative-gross-exposure seam as the regime/seasonality
        # overlays above; a *level* threshold, not a z-score, since an
        # inverted curve (T10Y2Y < 0) is an absolute, economically
        # meaningful signal, not a relative one.
        macro_cfg = cfg.get("macro_overlay", {}) or {}
        self.macro_overlay_enabled: bool = bool(macro_cfg.get("enabled", False))
        self.macro_overlay_series_id: str = macro_cfg.get("series_id", "T10Y2Y")
        # Neutral level: at or above this, no de-risking (effective=1.0).
        self.macro_overlay_neutral_level: float = float(macro_cfg.get("neutral_level", 0.0))
        # Risk-off level: at or below this (a meaningfully inverted curve),
        # the full de-risk factor applies. Linear interpolation between the
        # two levels, matching how the regime overlay blends by confidence.
        self.macro_overlay_risk_off_level: float = float(macro_cfg.get("risk_off_level", -0.5))
        self.macro_overlay_risk_off_scale: float = float(macro_cfg.get("risk_off_scale", 0.5))

    def run(self, ctx: AgentContext, **inputs: Any) -> RiskDecision:
        proposal: TradeProposal = inputs["proposal"]
        portfolio = inputs.get("portfolio")

        targets = dict(proposal.targets)
        original_gross = sum(abs(w) for w in targets.values())
        violations: list[str] = []
        actions: list[str] = []
        log.debug(
            "RiskAgent evaluating proposal asof=%s: %d names, gross=%.3f",
            ctx.now, len(targets), original_gross,
        )

        targets, v, a = self._clip_position_sizes(targets)
        violations.extend(v)
        actions.extend(a)

        targets, v, a = self._cap_gross_exposure(targets)
        violations.extend(v)
        actions.extend(a)

        targets, v, a = self._cap_net_exposure(targets)
        violations.extend(v)
        actions.extend(a)

        sector_map = inputs.get("sector_map") or self.sector_map
        if sector_map:
            targets, v, a = self._cap_sector_concentration(targets, sector_map)
            violations.extend(v)
            actions.extend(a)
        else:
            # Fail loud, not silent: a documented hard control is not being
            # enforced because no sector data was supplied.
            log.warning("Sector concentration cap NOT enforced (no sector_map provided)")
            actions.append("Sector concentration cap skipped (no sector_map)")

        if self.correlation_threshold:
            targets, v, a = self._cap_correlated_exposure(targets, ctx)
            violations.extend(v)
            actions.extend(a)

        if self.max_participation_pct:
            targets, v, a = self._cap_liquidity(targets, portfolio, ctx)
            violations.extend(v)
            actions.extend(a)

        if self.cvar_limit:
            targets, v, a = self._cvar_overlay(targets, ctx)
            violations.extend(v)
            actions.extend(a)

        # Explicit inputs["vol_estimates"] (if a caller ever supplies it)
        # takes precedence; otherwise self-compute from ctx.pit_view when
        # enabled -- see vol_targeting_enabled's docstring above for why
        # this used to be unreachable in the real orchestrator path.
        vol_estimates = inputs.get("vol_estimates")
        if vol_estimates is None and self.vol_targeting_enabled:
            vol_estimates = self._estimate_vol_estimates(targets, ctx)
        if vol_estimates:
            targets, v, a = self._vol_targeting(targets, vol_estimates)
            violations.extend(v)
            actions.extend(a)

        targets, v, a = self._drawdown_breaker(targets, portfolio)
        violations.extend(v)
        actions.extend(a)

        # Final enforcement pass: non-uniform stages (sector) and de-risking
        # can perturb exposures set earlier, so re-assert the hard caps once
        # more.  On an already-compliant book this is a no-op.
        targets, v, a = self._enforce_hard_caps(targets)
        violations.extend(v)
        actions.extend(a)

        # Veto severity is the L1 distance between the final and proposed
        # weights (normalized by original gross), so heavy restructuring or
        # sign flips are captured – not just a change in total gross.
        if original_gross > 1e-10:
            l1_distance = sum(
                abs(targets.get(s, 0.0) - proposal.targets.get(s, 0.0))
                for s in set(targets) | set(proposal.targets)
            )
            clipping_severity = l1_distance / original_gross
        else:
            clipping_severity = 0.0

        if clipping_severity > self.veto_threshold:
            violations.append(
                f"VETO: clipping severity {clipping_severity:.1%} exceeds "
                f"threshold {self.veto_threshold:.1%}"
            )
            actions.append("Proposal vetoed due to excessive constraint violations")
            log.warning(
                "Risk VETO asof=%s: clipping severity %.1f%% exceeds threshold %.1f%% "
                "(%d prior violations: %s)",
                ctx.now, clipping_severity * 100, self.veto_threshold * 100,
                len(violations) - 1, violations[:-1],
            )
            return RiskDecision(
                approved=False,
                adjusted_targets=targets,
                violations=violations,
                actions=actions,
            )

        # Regime overlay is an *intentional* portfolio-sizing policy, not a
        # constraint breach, so it is applied to the already-approved book
        # (after the veto decision) and re-capped by the hard caps — it must
        # never by itself trigger a veto/abort.
        if self.regime_overlay_enabled:
            regime_state = inputs.get("regime_state") or self._detect_regime(ctx)
            targets, _v, a = self._regime_exposure_overlay(targets, regime_state)
            log.info("Risk regime overlay asof=%s: %s", ctx.now, a)
            actions.extend(a)
            targets, _v2, _a2 = self._enforce_hard_caps(targets)

        # Same "intentional sizing policy, not a constraint breach" posture
        # as the regime overlay above -- applied post-veto, re-capped after.
        # Composing multiple multiplicative overlays has no lower floor
        # today (see PART 3 of the remediation plan); if a future overlay
        # is added alongside this one, add a composed-scaler floor before
        # enabling both together.
        if self.seasonality_overlay_enabled:
            targets, _v3, a = self._seasonality_exposure_overlay(targets, ctx)
            log.info("Risk seasonality overlay asof=%s: %s", ctx.now, a)
            actions.extend(a)
            targets, _v4, _a4 = self._enforce_hard_caps(targets)

        if self.macro_overlay_enabled:
            targets, _v5, a = self._macro_exposure_overlay(targets, ctx)
            log.info("Risk macro overlay asof=%s: %s", ctx.now, a)
            actions.extend(a)
            targets, _v6, _a6 = self._enforce_hard_caps(targets)

        if violations:
            log.info(
                "Risk decision APPROVED asof=%s with %d violation(s) remediated: %s",
                ctx.now, len(violations), violations,
            )
        else:
            log.debug("Risk decision APPROVED asof=%s: no violations", ctx.now)

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
                log.warning(
                    "Risk clip: %s weight %.4f exceeds per-name cap %.4f — clipped to %.4f",
                    sym, w, self.max_position_pct, capped,
                )
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
        log.warning(
            "Risk scale: gross exposure %.4f exceeds cap %.4f — scaled all weights by %.4f",
            gross, self.max_gross_exposure, scale,
        )
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

        log.warning(
            "Risk scale: net exposure %.4f exceeds cap %.4f — reduced %d same-side "
            "positions by %.4f total",
            net, self.max_net_exposure, len(same_side), excess,
        )
        return (
            adjusted,
            [f"Net exposure {net:.3f} exceeds cap {self.max_net_exposure}"],
            ["Reduced same-side weights to bring net within bounds"],
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
                log.warning(
                    "Risk clip: sector '%s' weight %.4f exceeds cap %.4f — scaled by %.4f",
                    sector, total_w, self.max_sector_pct, scale,
                )

        return adjusted, violations, actions

    def _cap_liquidity(
        self,
        targets: dict[str, float],
        portfolio: Any,
        ctx: AgentContext,
    ) -> tuple[dict[str, float], list[str], list[str]]:
        """Clip the implied trade size to ``max_participation_pct`` of ADV.

        Trade size is estimated as ``|target_weight - current_weight| * NAV``
        so an already-held position isn't penalised for its resting size —
        only the incremental order this cycle would place is checked against
        the symbol's trailing average daily dollar volume.
        """
        pit_view = getattr(ctx, "pit_view", None)
        if pit_view is None:
            return targets, [], []

        nav = getattr(portfolio, "nav", None) if portfolio is not None else None
        if not nav or nav <= 0:
            log.debug("Liquidity check skipped: no portfolio NAV available")
            return targets, [], []

        current_weights: dict[str, float] = {}
        if portfolio is not None and hasattr(portfolio, "get_weights"):
            try:
                # No fresh price dict is available here; get_weights({})
                # reuses whatever marks the portfolio last cached.
                current_weights = portfolio.get_weights({})
            except Exception as exc:
                log.debug("Liquidity check: could not read current weights (%s)", exc)

        violations: list[str] = []
        actions: list[str] = []
        adjusted = dict(targets)

        for sym, target_w in targets.items():
            current_w = current_weights.get(sym, 0.0)
            trade_notional = abs(target_w - current_w) * nav
            if trade_notional <= 0:
                continue

            adv_dollars = estimate_adv_dollars(pit_view, sym, self.adv_lookback_days)
            if not adv_dollars:
                continue

            participation = trade_notional / adv_dollars
            if participation <= self.max_participation_pct:
                continue

            max_trade_notional = self.max_participation_pct * adv_dollars
            direction = 1.0 if target_w >= current_w else -1.0
            capped_w = current_w + direction * (max_trade_notional / nav)
            violations.append(
                f"{sym} implied trade ${trade_notional:,.0f} is {participation:.1%} of "
                f"{self.adv_lookback_days}d ADV (${adv_dollars:,.0f}) — exceeds "
                f"{self.max_participation_pct:.1%} participation-rate cap"
            )
            actions.append(
                f"Clipped {sym} from {target_w:.4f} to {capped_w:.4f} (liquidity/ADV cap)"
            )
            log.warning(
                "Risk clip: %s implied trade $%.0f is %.1f%% of %dd ADV $%.0f — "
                "clipped target weight from %.4f to %.4f",
                sym, trade_notional, participation * 100, self.adv_lookback_days,
                adv_dollars, target_w, capped_w,
            )
            adjusted[sym] = capped_w

        return adjusted, violations, actions

    def _cap_correlated_exposure(
        self,
        targets: dict[str, float],
        ctx: AgentContext,
    ) -> tuple[dict[str, float], list[str], list[str]]:
        """Scale down pairs of highly-correlated, same-direction positions.

        Only reinforcing bets (both long or both short) are capped — an
        offsetting long/short pair in correlated names is a hedge, not
        concentration. Note pairs are processed independently, so a symbol
        correlated with several others in the book can be scaled down more
        than once per cycle; ``_enforce_hard_caps`` re-asserts the primary
        caps afterwards, but this pairwise pass is deliberately conservative
        rather than a single joint optimisation.
        """
        pit_view = getattr(ctx, "pit_view", None)
        if pit_view is None or len(targets) < 2:
            return targets, [], []

        symbols = list(targets.keys())
        try:
            price_df = pit_view.prices(symbols, lookback_days=self.correlation_lookback_days)
        except Exception as exc:
            log.warning(
                "Correlation check skipped: failed to load price history (%s)",
                exc, exc_info=True,
            )
            return targets, [], []

        if price_df is None or price_df.empty or "close" not in price_df.columns:
            return targets, [], []

        try:
            pivot = price_df.pivot_table(index="date", columns="symbol", values="close")
            corr = pivot.pct_change().dropna(how="all").corr()
        except Exception as exc:
            # A malformed-data failure here is at least as likely to indicate
            # a real bug as the load failure above — same level, for the same
            # reason: this is a risk-gate bypass (return unchanged targets),
            # not a routine, safe-to-ignore event.
            log.warning(
                "Correlation check: failed to compute correlation matrix (%s)",
                exc, exc_info=True,
            )
            return targets, [], []

        violations: list[str] = []
        actions: list[str] = []
        adjusted = dict(targets)

        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i + 1:]:
                w_a, w_b = adjusted.get(sym_a, 0.0), adjusted.get(sym_b, 0.0)
                if w_a == 0.0 or w_b == 0.0 or (w_a > 0) != (w_b > 0):
                    continue
                if sym_a not in corr.columns or sym_b not in corr.columns:
                    continue
                rho = corr.loc[sym_a, sym_b]
                # NaN (e.g. too few overlapping observations) compares False
                # to everything, so this also safely skips missing data.
                if not abs(rho) >= self.correlation_threshold:
                    continue

                combined = abs(w_a) + abs(w_b)
                if combined <= self.max_correlated_pair_pct:
                    continue
                scale = self.max_correlated_pair_pct / combined
                adjusted[sym_a] = w_a * scale
                adjusted[sym_b] = w_b * scale
                violations.append(
                    f"{sym_a}/{sym_b} correlation {rho:.2f} with combined same-direction "
                    f"weight {combined:.3f} exceeds cap {self.max_correlated_pair_pct}"
                )
                actions.append(
                    f"Scaled {sym_a}/{sym_b} by {scale:.4f} (correlated-pair cap)"
                )
                log.warning(
                    "Risk clip: %s/%s correlated (rho=%.2f) with combined weight %.4f "
                    "exceeds cap %.4f — scaled both by %.4f",
                    sym_a, sym_b, rho, combined, self.max_correlated_pair_pct, scale,
                )

        return adjusted, violations, actions

    def _cvar_overlay(
        self,
        targets: dict[str, float],
        ctx: AgentContext,
    ) -> tuple[dict[str, float], list[str], list[str]]:
        """De-risk when the book's historical CVaR exceeds ``cvar_limit``.

        Complements ``_vol_targeting``'s diagonal-covariance vol estimate
        with a distributional tail-risk read: reuses the same
        ``pit_view.prices(...)`` -> ``pivot.pct_change()`` idiom as
        ``_cap_correlated_exposure`` to build a real per-name return matrix,
        combines it with the current target weights into a historical
        portfolio return series *under those weights*, and computes that
        series' Conditional Value-at-Risk (expected shortfall) via
        :func:`firm.eval.metrics.conditional_value_at_risk`. Only ever
        de-risks (scale <= 1.0) — same convention as ``_vol_targeting`` —
        so it can never re-breach the hard caps applied earlier in the
        pipeline.
        """
        pit_view = getattr(ctx, "pit_view", None)
        if pit_view is None:
            return targets, [], []

        symbols = [s for s, w in targets.items() if w != 0.0]
        if not symbols:
            return targets, [], []

        try:
            price_df = pit_view.prices(symbols, lookback_days=self.cvar_lookback_days)
        except Exception as exc:
            log.warning(
                "CVaR check skipped: failed to load price history (%s)",
                exc, exc_info=True,
            )
            return targets, [], []
        if price_df is None or price_df.empty or "close" not in price_df.columns:
            return targets, [], []

        try:
            pivot = price_df.pivot_table(index="date", columns="symbol", values="close")
            rets = pivot.pct_change().dropna(how="all")
        except Exception as exc:
            # Same rationale as the correlation check above: a risk-gate
            # bypass, not a routine event — must be as visible as the load
            # failure a few lines up in this same function.
            log.warning(
                "CVaR check: failed to compute return matrix (%s)",
                exc, exc_info=True,
            )
            return targets, [], []

        port_returns = None
        for sym in rets.columns:
            w = targets.get(sym, 0.0)
            if w == 0.0:
                continue
            weighted = rets[sym].fillna(0.0) * w
            port_returns = weighted if port_returns is None else port_returns + weighted
        if port_returns is None:
            return targets, [], []

        port_cvar = conditional_value_at_risk(port_returns, self.cvar_confidence)
        if port_cvar <= self.cvar_limit:
            return targets, [], []

        scale = min(self.cvar_limit / port_cvar, 1.0)
        scaled = {s: w * scale for s, w in targets.items()}
        log.warning(
            "Risk scale: portfolio CVaR(%.0f%%) %.4f exceeds limit %.4f — "
            "scaled all weights by %.4f",
            self.cvar_confidence * 100, port_cvar, self.cvar_limit, scale,
        )
        return (
            scaled,
            [
                f"Portfolio CVaR({self.cvar_confidence:.0%}) {port_cvar:.4f} "
                f"exceeds limit {self.cvar_limit}"
            ],
            [f"Scaled all weights by {scale:.4f} (CVaR tail-risk overlay)"],
        )

    def _estimate_vol_estimates(
        self, targets: dict[str, float], ctx: AgentContext,
    ) -> dict[str, float] | None:
        """Realised annualised vol per symbol from ``ctx.pit_view``, for
        ``_vol_targeting`` when no explicit ``vol_estimates`` input is
        supplied. Mirrors ``_cap_correlated_exposure``'s own
        prices-→pivot-→pct_change idiom; returns ``None`` (no-op) on any
        missing pit_view/history, same fail-open posture as that method.
        """
        pit_view = getattr(ctx, "pit_view", None)
        if pit_view is None or not targets:
            return None
        symbols = list(targets.keys())
        try:
            price_df = pit_view.prices(symbols, lookback_days=self.vol_lookback_days)
        except Exception as exc:
            log.warning(
                "Vol targeting skipped: failed to load price history (%s)",
                exc, exc_info=True,
            )
            return None
        if price_df is None or price_df.empty or "close" not in price_df.columns:
            return None
        try:
            pivot = price_df.pivot_table(index="date", columns="symbol", values="close")
            daily_vol = pivot.pct_change().std()
        except Exception as exc:
            log.warning(
                "Vol targeting skipped: malformed price history (%s)",
                exc, exc_info=True,
            )
            return None
        annualised = (daily_vol * (252 ** 0.5)).to_dict()
        return {s: v for s, v in annualised.items() if v == v}  # drop NaN

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
        # Only ever de-risk.  Scaling weights UP to hit a vol target would
        # re-breach the per-name / gross / net / sector caps applied earlier.
        scale = min(self.vol_target / port_vol, 1.0)
        if abs(scale - 1.0) < 0.01:
            return targets, [], []
        scaled = {s: w * scale for s, w in targets.items()}
        log.info(
            "Risk scale: vol targeting — estimated portfolio vol %.4f vs target %.4f, "
            "scaled weights by %.4f",
            port_vol, self.vol_target, scale,
        )
        return (
            scaled,
            [f"Portfolio vol {port_vol:.3f} vs target {self.vol_target}"],
            [f"Scaled weights by {scale:.4f} for vol targeting"],
        )

    def _enforce_hard_caps(
        self, targets: dict[str, float]
    ) -> tuple[dict[str, float], list[str], list[str]]:
        """Re-apply per-name, gross, and net caps as a final pass.

        Returns only the *new* violations/actions a re-breach produced, so a
        compliant book reports nothing.
        """
        t, v1, a1 = self._clip_position_sizes(targets)
        t, v2, a2 = self._cap_gross_exposure(t)
        t, v3, a3 = self._cap_net_exposure(t)
        return t, v1 + v2 + v3, a1 + a2 + a3

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
        log.warning(
            "Risk drawdown breaker tripped: drawdown %.2f%% exceeds threshold %.2f%% — "
            "reduced exposure by %.0f%%",
            drawdown * 100, self.max_drawdown_pct * 100, (1 - scale) * 100,
        )
        return (
            scaled,
            [f"Drawdown {drawdown:.1%} exceeds threshold {self.max_drawdown_pct:.1%}"],
            [f"Reduced exposure by {1 - scale:.0%} (drawdown circuit breaker)"],
        )

    # ------------------------------------------------------------------
    # HMM market-regime overlay (opt-in)
    # ------------------------------------------------------------------

    def _detect_regime(self, ctx: AgentContext):
        """Detect the prevailing market regime from ``ctx.pit_view``.

        Returns a :class:`~firm.regime.model.RegimeState` or ``None`` when the
        regime cannot be determined (no data / ``hmmlearn`` missing), in which
        case the overlay is a no-op.
        """
        pit_view = getattr(ctx, "pit_view", None)
        if pit_view is None:
            return None
        if self._regime_detector is None:
            from firm.regime.detector import MarketRegimeDetector

            cfg = self._regime_overlay_cfg
            detector_kwargs: dict[str, Any] = {
                "n_states": cfg.get("n_states", 3),
                "lookback_days": cfg.get("lookback_days", 504),
                "retrain_frequency": cfg.get("retrain_frequency", 21),
                "benchmark_symbol": cfg.get("benchmark_symbol"),
                "ensemble": bool(cfg.get("ensemble", False)),
            }
            if cfg.get("ensemble_seeds"):
                detector_kwargs["ensemble_seeds"] = tuple(cfg["ensemble_seeds"])
            self._regime_detector = MarketRegimeDetector(**detector_kwargs)
        return self._regime_detector.detect(pit_view)

    def _regime_exposure_overlay(
        self, targets: dict[str, float], regime_state
    ) -> tuple[dict[str, float], list[str], list[str]]:
        """Scale gross exposure by the prevailing market regime.

        The full-confidence scale comes from :attr:`regime_exposure_map`; the
        effective scale is blended by the posterior confidence so an uncertain
        regime read barely moves sizing while a confident one applies the full
        playbook factor::

            effective = 1 + (factor - 1) * confidence * separation_damping

        ``separation_damping`` guards against label-switching (see
        ``_regime_min_state_separation``'s comment in ``__init__``): a
        high-confidence read of a label that's only noise-level distinct
        from its neighbour gets pulled back toward a no-op (1.0) instead of
        applying the full playbook factor, mirroring the per-symbol
        regime_hmm strategy's separation-based damping.
        """
        if regime_state is None:
            return targets, [], ["Regime overlay: no regime detected (no-op)"]

        label = regime_state.label
        confidence = float(regime_state.confidence)
        factor = self.regime_exposure_map.get(label, 1.0)
        separation = float(getattr(regime_state, "separation", float("inf")))
        damping = 1.0
        if self._regime_min_state_separation > 0:
            damping = max(
                self._regime_separation_damping_floor,
                min(1.0, separation / self._regime_min_state_separation),
            )
        effective = 1.0 + (factor - 1.0) * confidence * damping
        if abs(effective - 1.0) < 1e-9:
            return targets, [], [f"Regime overlay: {label} (conf {confidence:.2f}) — no change"]

        scaled = {s: w * effective for s, w in targets.items()}
        note = (
            f"Regime overlay: {label} (conf {confidence:.2f}) "
            f"scaled gross exposure by {effective:.3f}"
        )
        if damping < 1.0:
            note += (
                f" (damped {damping:.2f}x: separation={separation:.3f} < "
                f"threshold={self._regime_min_state_separation:.3f}, label-switching risk)"
            )
        return scaled, [], [note]

    def _seasonality_exposure_overlay(
        self, targets: dict[str, float], ctx: AgentContext
    ) -> tuple[dict[str, float], list[str], list[str]]:
        """Scale gross exposure by the calendar/seasonality signal.

        Reuses ``SeasonalityStrategy.generate()`` directly (not a
        reimplementation) purely for its day-of-week + turn-of-month score
        -- every symbol in its output carries the identical value by
        construction (see ``seasonality.py``'s own docstring: "applied as a
        uniform tilt to all symbols"), so any one signal's ``score`` is the
        whole answer::

            effective = 1 + seasonality_overlay_scale * score

        ``score`` is already clipped to ``[-1, 1]`` by the strategy itself,
        so ``effective`` is bounded to
        ``[1 - scale, 1 + scale]`` -- no separate damping needed here (no
        label-switching risk on a continuous calendar signal, unlike the
        discrete regime read above).
        """
        pit_view = getattr(ctx, "pit_view", None)
        if pit_view is None:
            return targets, [], ["Seasonality overlay: no pit_view available (no-op)"]

        from firm.strategies.seasonality import SeasonalityStrategy

        try:
            signals = SeasonalityStrategy(self._seasonality_params).generate(pit_view)
        except Exception:
            log.warning("Seasonality overlay: signal generation failed", exc_info=True)
            return targets, [], ["Seasonality overlay: signal generation failed (no-op)"]
        if not signals:
            return targets, [], ["Seasonality overlay: no signal (no-op)"]

        score = float(signals[0].score)
        effective = 1.0 + self.seasonality_overlay_scale * score
        if abs(effective - 1.0) < 1e-9:
            return targets, [], [f"Seasonality overlay: score={score:.3f} — no change"]

        scaled = {s: w * effective for s, w in targets.items()}
        note = f"Seasonality overlay: score={score:.3f} scaled gross exposure by {effective:.3f}"
        return scaled, [], [note]

    def _macro_exposure_overlay(
        self, targets: dict[str, float], ctx: AgentContext
    ) -> tuple[dict[str, float], list[str], list[str]]:
        """Scale gross exposure by a macro/rate signal (default: T10Y2Y,
        the 10Y-2Y Treasury yield-curve spread).

        A *level* threshold, not a z-score against the series' own trailing
        history: curve inversion (``T10Y2Y < 0``) is an absolute,
        economically meaningful signal (one of the most robustly documented
        recession-warning indicators), not something whose interpretation
        should shift with how the series happened to behave recently::

            level >= neutral_level                       -> effective = 1.0
            level <= risk_off_level                       -> effective = risk_off_scale
            otherwise                                      -> linear blend

        Requires ``ctx.pit_view`` to expose a ``.macro(series_id,
        lookback_days)`` method (``PitViewAdapter`` in backtest,
        ``LivePitViewAdapter`` in live) backed by cached FRED data
        (``scripts/backfill_macro.py`` -> ``combined/macro``); degrades to a
        no-op wherever that isn't available (older code paths, tests, or a
        live deployment before ``LivePitViewAdapter`` gained a ``.macro()``
        method) rather than raising.
        """
        pit_view = getattr(ctx, "pit_view", None)
        if pit_view is None or not hasattr(pit_view, "macro"):
            return targets, [], ["Macro overlay: pit_view has no macro data (no-op)"]

        try:
            series = pit_view.macro(self.macro_overlay_series_id, lookback_days=5)
        except Exception:
            log.warning("Macro overlay: signal lookup failed", exc_info=True)
            return targets, [], ["Macro overlay: signal lookup failed (no-op)"]
        if series is None or series.empty:
            return targets, [], [f"Macro overlay: no {self.macro_overlay_series_id} data (no-op)"]

        level = float(series.iloc[-1])
        if not math.isfinite(level):
            return targets, [], ["Macro overlay: non-finite reading (no-op)"]

        neutral = self.macro_overlay_neutral_level
        risk_off = self.macro_overlay_risk_off_level
        risk_off_scale = self.macro_overlay_risk_off_scale
        if level >= neutral:
            effective = 1.0
        elif level <= risk_off:
            effective = risk_off_scale
        else:
            span = neutral - risk_off
            frac = (neutral - level) / span if span > 0 else 1.0
            effective = 1.0 + (risk_off_scale - 1.0) * frac

        if abs(effective - 1.0) < 1e-9:
            return targets, [], [
                f"Macro overlay: {self.macro_overlay_series_id}={level:.3f} — no change"
            ]

        scaled = {s: w * effective for s, w in targets.items()}
        note = (
            f"Macro overlay: {self.macro_overlay_series_id}={level:.3f} "
            f"scaled gross exposure by {effective:.3f}"
        )
        return scaled, [], [note]
