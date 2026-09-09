"""Orchestrator – drives the per-bar agent pipeline.

Calls analysts (in parallel) -> bull/bear researchers -> debate ->
trader/PM -> risk approval loop -> execution, managing the blackboard
between steps.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any, Callable, TypeVar

import pandas as pd

from firm.agents.base import Agent, AgentContext
from firm.agents.blackboard import Blackboard
from firm.contracts.models import RiskDecision, SignalSet, TradeProposal
from firm.portfolio.state import PortfolioState

log = logging.getLogger(__name__)

_T = TypeVar("_T")


class StageTimeoutError(TimeoutError):
    """Raised when a single pipeline stage exceeds its wall-clock budget."""


def _shutdown_executor(pool: ThreadPoolExecutor, *, abandon: bool) -> None:
    """Tear down a pool without joining abandoned worker threads."""
    pool.shutdown(wait=not abandon, cancel_futures=abandon)


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
        sleeve_traders: dict[str, Agent] | None = None,
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
        cfg = config or {}
        self._analyst_timeout_seconds = self._optional_timeout(
            cfg.get("analyst_timeout_seconds"),
        )
        stage_timeout = cfg.get("orchestrator_stage_timeout_seconds")
        if stage_timeout is None:
            stage_timeout = cfg.get("analyst_timeout_seconds")
        self._stage_timeout_seconds = self._optional_timeout(stage_timeout)
        self._regime_weights_detector = None

        # ------------------------------------------------------------------
        # Per-strategy capital sleeves (capital_allocation_mode: "sleeved").
        # See docs/pattern_recognition_plan.md-style tracker or the plan this
        # was built from -- default "blended" reproduces today's single
        # shared-book behavior byte-for-byte (see step()).
        # ------------------------------------------------------------------
        self.capital_allocation_mode: str = cfg.get("capital_allocation_mode", "blended")
        # One independent TraderAgent per sleeved strategy -- required
        # because TraderAgent holds real cross-cycle instance state
        # (conviction EMA, joint_optimizer NAV history) keyed only by
        # symbol; one shared instance called once per sleeve would let
        # sleeves corrupt each other's smoothing/history state.
        self.sleeve_traders: dict[str, Agent] = sleeve_traders or {}
        self._strategy_capital_weights: dict[str, float] = cfg.get(
            "strategy_capital_weights", {},
        ) or {}
        self._total_initial_capital: float = float(cfg.get("initial_capital", 10_000_000))
        # Each sleeve's own independently-compounding virtual book -- fixed
        # initial capital at first allocation, then evolves purely from that
        # sleeve's own trades from then on (like a real per-strategy
        # sub-account), NOT re-normalized to a fraction of total NAV every
        # cycle -- that would hide a strategy's own true standalone
        # compounding track record, defeating the point of sleeving at all.
        # Restored from LiveStateStore on restart (see restore_sleeve_portfolios);
        # never derived from the broker (these are virtual, no real sub-account).
        self._sleeve_portfolios: dict[str, PortfolioState] = {}

        if self.capital_allocation_mode == "sleeved":
            self._check_sleeved_llm_cost_safety(cfg)

    @staticmethod
    def _check_sleeved_llm_cost_safety(cfg: dict[str, Any]) -> None:
        """Refuse construction rather than silently multiply LLM cost.

        bull/bear/debate normally run once per cycle over the whole blended
        blackboard; in sleeved mode they run once *per sleeve*. That's cheap
        CPU when these three stay in "quant" mode (the production default --
        see config/llm.yaml's "no per-symbol LLM fan-out" comment), but if
        any were ever switched to "llm_enhanced"/"llm_only", running them
        independently per sleeve would multiply per-cycle LLM call volume
        roughly N-fold (once per sleeve instead of once), since each sleeve's
        own top-K enhancement-budget selection has no visibility into what
        other sleeves already spent this cycle. That cross-sleeve shared
        budget coordination isn't implemented yet -- fail loud here instead
        of silently shipping an N-fold cost surprise.
        """
        agent_modes = cfg.get("agent_modes", {}) or {}
        unsafe = [
            role for role in ("bull_researcher", "bear_researcher", "debate")
            if agent_modes.get(role) in ("llm_enhanced", "llm_only")
        ]
        if unsafe:
            raise ValueError(
                f"capital_allocation_mode='sleeved' is incompatible with "
                f"agent_modes {unsafe} set to llm_enhanced/llm_only -- running "
                "these per sleeve would multiply per-cycle LLM call volume "
                "roughly N-fold (no cross-sleeve shared enhancement budget "
                "exists yet). Keep bull_researcher/bear_researcher/debate at "
                "'quant' while capital_allocation_mode is 'sleeved'."
            )

    def _resolve_market_regime(self, pit_view) -> Any | None:
        """Detect market regime when ``strategy_regime_weights`` is enabled."""
        cfg = (self.config or {}).get("strategy_regime_weights") or {}
        if not cfg.get("enabled"):
            return None
        if self._regime_weights_detector is None:
            from firm.regime.detector import MarketRegimeDetector

            detector_kwargs: dict[str, Any] = {
                "n_states": int(cfg.get("n_states", 3)),
                "lookback_days": int(cfg.get("lookback_days", 252)),
                "retrain_frequency": int(cfg.get("retrain_frequency", 21)),
                "benchmark_symbol": cfg.get("benchmark_symbol"),
                "ensemble": bool(cfg.get("ensemble", False)),
            }
            if cfg.get("ensemble_seeds"):
                detector_kwargs["ensemble_seeds"] = tuple(cfg["ensemble_seeds"])
            self._regime_weights_detector = MarketRegimeDetector(**detector_kwargs)
        return self._regime_weights_detector.detect(pit_view)

    @staticmethod
    def _optional_timeout(value: Any) -> float | None:
        if value is None:
            return None
        timeout = float(value)
        return timeout if timeout > 0 else None

    def run(self, ctx: AgentContext, **inputs: Any) -> tuple[list[dict], Blackboard]:
        """ABC-compliant entry point – delegates to :meth:`step`."""
        context: dict[str, Any] = {
            "pit_view": ctx.pit_view,
            "portfolio": ctx.portfolio,
            "prices": inputs.get("prices", ctx.config.get("prices", {})),
        }
        return self.step(context)

    def _run_analysts(self, ctx: AgentContext, bb: Blackboard) -> None:
        if not self.analysts:
            return
        workers = min(len(self.analysts), 8)
        pool = ThreadPoolExecutor(max_workers=workers)
        abandon = False
        try:
            futures = {
                pool.submit(analyst.run, ctx): analyst for analyst in self.analysts
            }
            for future in futures:
                analyst = futures[future]
                try:
                    timeout = self._analyst_timeout_seconds
                    if timeout is not None:
                        signal_set = future.result(timeout=timeout)
                    else:
                        signal_set = future.result()
                    bb.signal_sets.append(signal_set)
                    for err in getattr(analyst, "_last_errors", []) or []:
                        bb.errors.append({"agent": analyst.name, **err})
                except FuturesTimeoutError:
                    abandon = True
                    msg = (
                        f"analyst timed out after {self._analyst_timeout_seconds:.0f}s"
                    )
                    log.warning("Analyst %s %s", analyst.name, msg)
                    bb.errors.append({"agent": analyst.name, "error": msg})
                except Exception as exc:
                    log.error("Analyst %s failed", analyst.name, exc_info=True)
                    bb.errors.append({"agent": analyst.name, "error": str(exc)})
        finally:
            _shutdown_executor(pool, abandon=abandon)

    def _run_stage(
        self,
        stage: str,
        fn: Callable[..., _T],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> _T:
        timeout = self._stage_timeout_seconds
        if timeout is None:
            return fn(*args, **kwargs)

        pool = ThreadPoolExecutor(max_workers=1)
        abandon = False
        try:
            future = pool.submit(fn, *args, **kwargs)
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError as exc:
                abandon = True
                raise StageTimeoutError(
                    f"{stage} timed out after {timeout:.0f}s"
                ) from exc
        finally:
            _shutdown_executor(pool, abandon=abandon)

    def _record_stage_failure(
        self,
        bb: Blackboard,
        agent: Agent,
        exc: Exception,
    ) -> tuple[list[dict], Blackboard] | None:
        bb.errors.append({"agent": getattr(agent, "name", "?"), "error": str(exc)})
        bb.degraded = True
        log.warning("Pipeline degraded at %s: %s", getattr(agent, "name", "?"), exc)
        if self._abort_on_degraded:
            log.warning("Aborting bar (abort_on_degraded=True) – returning empty orders")
            return [], bb
        return None

    def step(self, context: dict[str, Any]) -> tuple[list[dict], Blackboard]:
        """Run the full agent pipeline for one timestep.

        Args:
            context: Must contain ``pit_view`` (:class:`PitView`),
                ``portfolio`` (:class:`PortfolioState`), and ``prices``
                (``dict[str, float]``).  Optional ``memory``
                (:class:`firm.agents.memory.TradingMemoryLog`) is forwarded
                to LLM-enhanced trader and risk agents for past-context injection.

        Returns:
            ``(orders_list, blackboard)`` where *orders_list* feeds the
            execution engine.
        """
        if self.capital_allocation_mode == "sleeved":
            return self._step_sleeved(context)

        pit_view = context["pit_view"]
        portfolio = context.get("portfolio")
        prices: dict[str, float] = context.get("prices", {})
        memory = context.get("memory")
        attribution = context.get("attribution")

        bb = Blackboard(asof=pit_view.asof)
        market_regime = self._resolve_market_regime(pit_view)
        ctx = AgentContext(
            now=pit_view.asof,
            pit_view=pit_view,
            portfolio=portfolio,
            config=self.config,
            strategy_returns=context.get("strategy_returns"),
            market_regime=market_regime,
        )

        # 1. Run analysts in parallel, deterministic merge by domain
        self._run_analysts(ctx, bb)

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
        try:
            bull_theses = self._run_stage(
                "bull_researcher", self.bull.run, ctx, blackboard=bb,
            )
        except StageTimeoutError as exc:
            early = self._record_stage_failure(bb, self.bull, exc)
            if early is not None:
                return early
            bull_theses = []

        try:
            bear_theses = self._run_stage(
                "bear_researcher", self.bear.run, ctx, blackboard=bb,
            )
        except StageTimeoutError as exc:
            early = self._record_stage_failure(bb, self.bear, exc)
            if early is not None:
                return early
            bear_theses = []

        bb.theses.extend(bull_theses)
        bb.theses.extend(bear_theses)

        # 3. Debate synthesis
        try:
            debate_results = self._run_stage(
                "debate",
                self.debate.run,
                ctx,
                bull_theses=bull_theses,
                bear_theses=bear_theses,
            )
        except StageTimeoutError as exc:
            early = self._record_stage_failure(bb, self.debate, exc)
            if early is not None:
                return early
            debate_results = []
        bb.debate_results = debate_results

        if not debate_results:
            log.warning("No debate results – returning empty orders")
            return [], bb

        # 4. Trade proposal
        try:
            proposal = self._run_stage(
                "trader",
                self.trader.run,
                ctx,
                debate_results=debate_results,
                blackboard=bb,
                memory=memory,
                # Needed by allocation_method="joint_optimizer" to compute
                # current weights (w0) for the native transaction-cost term
                # -- ctx.portfolio has holdings but PortfolioState.get_weights
                # needs fresh prices to mark them. Existing allocation
                # methods ignore this kwarg (TraderAgent.run uses **inputs),
                # so this is backward-compatible.
                prices=prices,
            )
        except StageTimeoutError as exc:
            early = self._record_stage_failure(bb, self.trader, exc)
            if early is not None:
                return early
            return [], bb
        bb.proposal = proposal

        # 5. Risk approval loop
        decision = None
        for attempt in range(self._max_risk_retries):
            try:
                decision = self._run_stage(
                    "risk",
                    self.risk.run,
                    ctx,
                    proposal=proposal,
                    portfolio=portfolio,
                    memory=memory,
                )
            except StageTimeoutError as exc:
                early = self._record_stage_failure(bb, self.risk, exc)
                if early is not None:
                    return early
                log.warning("Risk stage timed out — rejecting proposal")
                return [], bb
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
        try:
            report = self._run_stage(
                "execution",
                self.execution.run,
                ctx,
                decision=decision,
                portfolio=portfolio,
                prices=prices,
                per_strategy=proposal.per_strategy,
                attribution=attribution,
            )
        except StageTimeoutError as exc:
            early = self._record_stage_failure(bb, self.execution, exc)
            if early is not None:
                return early
            return [], bb
        bb.execution_report = report

        self._collect_llm_usage(bb)

        return report.fills, bb

    # ------------------------------------------------------------------
    # Per-strategy capital sleeves (capital_allocation_mode: "sleeved")
    # ------------------------------------------------------------------

    @staticmethod
    def _partition_blackboard(bb: Blackboard, strategy: str) -> Blackboard:
        """A blackboard containing only *strategy*'s own signals.

        Reuses the exact same ``SignalSet``/``Signal`` shape bull/bear/debate
        already consume -- with only one strategy's signals present,
        ``net_scores_for_blackboard`` degenerates to that strategy's own
        z-scored signal (nothing else to combine with), so those three
        agents need no code changes at all to work correctly per sleeve.
        """
        sleeve_bb = Blackboard(asof=bb.asof)
        for ss in bb.signal_sets:
            filtered = [sig for sig in ss.signals if sig.strategy == strategy]
            if filtered:
                sleeve_bb.signal_sets.append(
                    SignalSet(domain=ss.domain, asof=ss.asof, signals=filtered)
                )
        return sleeve_bb

    def _sleeve_capital_weights(self) -> dict[str, float]:
        """Each sleeved strategy's fraction of total initial capital.

        Explicit ``strategy_capital_weights`` entries are honored as-is; any
        strategy not listed there splits the *remaining* budget equally.
        This is the initial allocation only -- each sleeve's own
        ``PortfolioState`` then compounds independently from that starting
        point, it is not re-derived every cycle.
        """
        names = list(self.sleeve_traders.keys())
        if not names:
            return {}
        weights = {
            name: float(self._strategy_capital_weights[name])
            for name in names
            if name in self._strategy_capital_weights
        }
        remaining_names = [n for n in names if n not in weights]
        if remaining_names:
            remaining_budget = max(0.0, 1.0 - sum(weights.values()))
            equal_share = remaining_budget / len(remaining_names)
            for name in remaining_names:
                weights[name] = equal_share
        return weights

    def _get_or_create_sleeve_portfolio(self, strategy: str, weight: float) -> PortfolioState:
        portfolio = self._sleeve_portfolios.get(strategy)
        if portfolio is None:
            portfolio = PortfolioState(initial_capital=weight * self._total_initial_capital)
            self._sleeve_portfolios[strategy] = portfolio
        return portfolio

    def export_sleeve_portfolios(self) -> dict[str, dict[str, Any]]:
        """Serialize every sleeve's cash/holdings for durable persistence.

        Sleeve books are virtual (no real broker sub-account to reconcile
        against on restart), so -- unlike the real ``PortfolioState``, which
        re-derives cash/holdings from the broker every cycle -- this state
        must be explicitly saved/restored or a restart would silently reset
        every sleeve back to its initial capital split, discarding its
        entire independent compounding history.
        """
        return {
            strategy: {"cash": p.cash, "holdings": dict(p.holdings)}
            for strategy, p in self._sleeve_portfolios.items()
        }

    def restore_sleeve_portfolios(self, state: dict[str, dict[str, Any]]) -> None:
        """Restore sleeve books persisted via :meth:`export_sleeve_portfolios`.

        Called once at engine startup, before the first cycle -- any sleeve
        not yet present in *state* (e.g. a newly-added strategy) is left to
        be lazily created at its full initial-capital split on first use.
        """
        for strategy, blob in (state or {}).items():
            try:
                portfolio = PortfolioState(initial_capital=float(blob.get("cash", 0.0)))
                portfolio.cash = float(blob.get("cash", 0.0))
                portfolio.holdings = {
                    sym: float(qty) for sym, qty in (blob.get("holdings") or {}).items()
                }
                self._sleeve_portfolios[strategy] = portfolio
            except Exception:
                log.warning(
                    "Failed to restore sleeve portfolio state for %s", strategy, exc_info=True,
                )

    def get_sleeve_metrics(self) -> dict[str, dict[str, float]]:
        """Exact per-strategy performance metrics from each sleeve's own NAV
        history -- unlike ``PerformanceAttribution``'s heuristic (dominant-
        strategy-wins-the-whole-order + running-net-share-count over one
        shared book), this is a genuine standalone return series per
        strategy, since each sleeve really does hold its own capital/positions.
        """
        from firm.eval.metrics import compute_all_metrics

        result: dict[str, dict[str, float]] = {}
        for strategy, portfolio in self._sleeve_portfolios.items():
            history = portfolio.history
            if len(history) < 2:
                continue
            navs = [snap.nav for snap in history]
            returns = pd.Series(navs).pct_change().dropna()
            if returns.empty:
                continue
            result[strategy] = compute_all_metrics(returns)
        return result

    def _step_sleeved(self, context: dict[str, Any]) -> tuple[list[dict], Blackboard]:
        """Sleeved-mode pipeline: one independent bull/bear/debate/trader/risk/
        execution pass per strategy against its own ``PortfolioState``, then
        one final netted ``ExecutionAgent`` pass against the real shared
        book for actual broker submission. See the module-level design note
        this was built from for the full rationale.
        """
        pit_view = context["pit_view"]
        real_portfolio = context.get("portfolio")
        prices: dict[str, float] = context.get("prices", {})
        memory = context.get("memory")
        attribution = context.get("attribution")

        bb = Blackboard(asof=pit_view.asof)
        market_regime = self._resolve_market_regime(pit_view)
        ctx_kwargs: dict[str, Any] = dict(
            config=self.config,
            strategy_returns=context.get("strategy_returns"),
            market_regime=market_regime,
        )

        self._run_analysts(
            AgentContext(now=pit_view.asof, pit_view=pit_view, portfolio=real_portfolio, **ctx_kwargs),
            bb,
        )
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

        weights = self._sleeve_capital_weights()
        sym_dollar_targets: dict[str, float] = {}
        per_strategy_weights: dict[str, dict[str, float]] = {}

        for strategy, trader in self.sleeve_traders.items():
            sleeve_bb = self._partition_blackboard(bb, strategy)
            sleeve_portfolio = self._get_or_create_sleeve_portfolio(
                strategy, weights.get(strategy, 0.0),
            )
            sleeve_ctx = AgentContext(
                now=pit_view.asof, pit_view=pit_view, portfolio=sleeve_portfolio, **ctx_kwargs,
            )

            if not any(ss.signals for ss in sleeve_bb.signal_sets):
                # No signal from this strategy this cycle -- its book simply
                # doesn't trade, but still needs a mark-to-market snapshot so
                # its return series has no gaps on no-trade days.
                sleeve_portfolio.record_snapshot(pit_view.asof, prices)
                continue

            try:
                bull_theses = self.bull.run(sleeve_ctx, blackboard=sleeve_bb)
                bear_theses = self.bear.run(sleeve_ctx, blackboard=sleeve_bb)
            except Exception:
                log.warning("Sleeve %s bull/bear failed", strategy, exc_info=True)
                bb.errors.append({"agent": f"sleeve:{strategy}", "error": "bull/bear failed"})
                sleeve_portfolio.record_snapshot(pit_view.asof, prices)
                continue
            sleeve_bb.theses.extend(bull_theses)
            sleeve_bb.theses.extend(bear_theses)

            try:
                debate_results = self.debate.run(
                    sleeve_ctx, bull_theses=bull_theses, bear_theses=bear_theses,
                )
            except Exception:
                log.warning("Sleeve %s debate failed", strategy, exc_info=True)
                bb.errors.append({"agent": f"sleeve:{strategy}", "error": "debate failed"})
                sleeve_portfolio.record_snapshot(pit_view.asof, prices)
                continue
            sleeve_bb.debate_results = debate_results
            if not debate_results:
                sleeve_portfolio.record_snapshot(pit_view.asof, prices)
                continue

            try:
                proposal = trader.run(
                    sleeve_ctx, debate_results=debate_results, blackboard=sleeve_bb,
                    memory=memory, prices=prices,
                )
            except Exception:
                log.warning("Sleeve %s trader failed", strategy, exc_info=True)
                bb.errors.append({"agent": f"sleeve:{strategy}", "error": "trader failed"})
                sleeve_portfolio.record_snapshot(pit_view.asof, prices)
                continue
            sleeve_bb.proposal = proposal

            try:
                decision = self.risk.run(
                    sleeve_ctx, proposal=proposal, portfolio=sleeve_portfolio, memory=memory,
                )
            except Exception:
                log.warning("Sleeve %s risk failed", strategy, exc_info=True)
                bb.errors.append({"agent": f"sleeve:{strategy}", "error": "risk failed"})
                sleeve_portfolio.record_snapshot(pit_view.asof, prices)
                continue
            sleeve_bb.risk_decision = decision
            if not decision.approved:
                log.info("Sleeve %s proposal rejected: %s", strategy, decision.violations)
                sleeve_portfolio.record_snapshot(pit_view.asof, prices)
                continue

            try:
                sleeve_report = self.execution.run(
                    sleeve_ctx, decision=decision, portfolio=sleeve_portfolio, prices=prices,
                    per_strategy={strategy: decision.adjusted_targets}, attribution=None,
                )
            except Exception:
                log.warning("Sleeve %s execution failed", strategy, exc_info=True)
                bb.errors.append({"agent": f"sleeve:{strategy}", "error": "execution failed"})
                sleeve_portfolio.record_snapshot(pit_view.asof, prices)
                continue
            sleeve_bb.execution_report = sleeve_report

            # Apply this sleeve's own fills to its own book -- exactly the
            # backtest path's mechanism (PortfolioState.update), giving a
            # realistic, cost-aware, independently-compounding ledger.
            if sleeve_report.fills:
                sleeve_portfolio.update(sleeve_report.fills, prices, cost=sleeve_report.costs)
            else:
                sleeve_portfolio.record_snapshot(pit_view.asof, prices)

            sleeve_nav = sleeve_portfolio.nav
            for sym, w in decision.adjusted_targets.items():
                sym_dollar_targets[sym] = sym_dollar_targets.get(sym, 0.0) + w * sleeve_nav
            per_strategy_weights[strategy] = dict(decision.adjusted_targets)

        if not sym_dollar_targets:
            log.warning("No sleeve produced an approved allocation – returning empty orders")
            self._collect_llm_usage(bb)
            return [], bb

        real_nav = getattr(real_portfolio, "nav", None) if real_portfolio is not None else None
        if not real_nav:
            real_nav = self._total_initial_capital
        combined_targets = {
            sym: dollar / real_nav for sym, dollar in sym_dollar_targets.items()
        }

        real_ctx = AgentContext(
            now=pit_view.asof, pit_view=pit_view, portfolio=real_portfolio, **ctx_kwargs,
        )
        try:
            report = self.execution.run(
                real_ctx,
                decision=RiskDecision(approved=True, adjusted_targets=combined_targets),
                portfolio=real_portfolio,
                prices=prices,
                per_strategy=per_strategy_weights,
                attribution=attribution,
            )
        except Exception as exc:
            log.error("Real (netted) execution pass failed in sleeved mode", exc_info=True)
            bb.errors.append({"agent": "execution", "error": str(exc)})
            return [], bb
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
