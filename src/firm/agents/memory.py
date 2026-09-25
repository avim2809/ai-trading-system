"""Append-only trading decision memory with deferred LLM reflection.

Two-phase lifecycle for each decision:

  Phase A (at decision time):
    ``store_decision(date, proposal, notes)`` writes a "pending" JSONL entry
    immediately after the orchestrator produces a trade proposal.

  Phase B (when outcome is known):
    ``reflect(date, raw_return, benchmark_return, llm_service)`` reads the
    pending entry, asks the LLM to write a 2-4 sentence retrospective, and
    marks the entry "reflected".

  Injection:
    ``get_context(n)`` returns the last *n* reflected entries formatted as a
    compact markdown block for injection into LLM agent prompts.

Storage: one JSONL file (``memory_log_path`` config key, defaults to
``data/memory/decisions.jsonl``).  Each line is a self-contained JSON object.
The file is append-only; Phase B updates are written as new lines with the
same ``date`` key — ``get_context`` always uses the latest entry per date so
the original pending record is superseded without in-place mutation (safe for
concurrent readers).

Recommendation self-improvement loop (RAG):
    ``reflect_day()``'s ``DailyReflectionRecommendation`` (including a
    ``no_action`` one — the absence of a recommendation for a given
    situation is itself a useful precedent) is additionally written as a
    ``Document`` into the ``"recommendations"`` RAG collection, and
    ``reflect_day()`` retrieves the most relevant few past recommendations
    from that same collection before building its prompt — so each day's
    LLM call sees "here's what was recommended (and whether it was applied)
    for similar past situations" instead of reasoning from scratch every
    time. ``mark_recommendation_applied()`` updates that stored document's
    ``applied`` metadata in place once a human actually applies it (see
    ``firm.rag.store.VectorStore.update_metadata``), so future retrieval
    reflects reality instead of a stale ``False``.

    This is pure context enrichment for an LLM prompt: it can only ever
    change what a future recommendation-generating call *reads*, never what
    it does — applying a recommendation remains the separate, human-gated
    ``POST /api/live/recommendations/{date}/apply`` path (see
    ``firm.llm.schemas.DailyReflectionRecommendation``'s docstring). All of
    the RAG read/write calls in this module are lazily constructed and
    wrapped in defensive try/except, exactly like
    ``LLMAgentMixin._get_retriever`` — this class has no hard dependency on
    the ``llm``/``rag`` extras and must keep working (just without the
    historical-context enrichment) when they aren't installed, or when the
    vector store is otherwise unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from firm.llm.schemas import DecisionReflection, parse_llm_response

if TYPE_CHECKING:
    from firm.llm.schemas import DailyReflectionRecommendation

log = logging.getLogger("firm.agents.memory")

_DEFAULT_PATH = Path("data/memory/decisions.jsonl")
# The literal real-production-file path, distinct from `_DEFAULT_PATH` above.
# `_DEFAULT_PATH` is intentionally monkeypatchable by tests (see
# TestMemoryDecisionsAPI in tests/test_api.py) — comparing a resolved path
# against *that* would be tautological whenever `memory_log_path` is omitted
# from config, since `_DEFAULT_PATH` itself supplies the fallback. This
# constant never changes, so it's what the pytest-context guard below checks.
_REAL_PROD_PATH = Path("data/memory/decisions.jsonl")

# The default `rag.persist_dir` from config/llm.yaml — the one vector store
# every collection (news/sec_filings/research/system_docs, and now
# "recommendations") shares in production. Same rationale as
# `_REAL_PROD_PATH` above: a test that exercises reflect_day()/
# mark_recommendation_applied() end-to-end (e.g. via a real live engine)
# without mocking `_get_rag_store`/`_get_rag_retriever` must not silently
# write real embeddings into this shared production store. Unlike the JSONL
# guard, this fails *soft* (treated as "RAG unavailable," same as a missing
# extra) rather than raising past reflect_day's own fail-soft try/except —
# raising loudly here would defeat the whole point of this feature being
# fail-soft, so under-isolated tests just don't get RAG coverage instead of
# corrupting shared production data.
_REAL_PROD_VECTORDB = Path("data/vectordb")

_REFLECTION_SYSTEM = (
    "You are a portfolio manager reviewing your own past trading decision "
    "now that the outcome is known. Respond with a single JSON object "
    '(no markdown, no commentary outside the JSON): {"verdict": "correct" | '
    '"incorrect" | "partial", "what_worked": "...", "what_failed": "...", '
    '"lesson": "..."}. "verdict" judges the directional call against the '
    "return figure. \"what_worked\" and \"what_failed\" each name a specific "
    "part of the original thesis (empty string if not applicable — e.g. "
    '"what_failed": "" for a fully correct call). When a per-strategy '
    "attribution is provided, name the specific strategy/strategies "
    "responsible in \"what_worked\"/\"what_failed\" (e.g. \"momentum's long "
    "NVDA call\") instead of speaking only about \"the portfolio\" — that's "
    "what makes the lesson actionable for a specific future signal, not "
    'just a vague restatement of the return. "lesson" is one concrete '
    "takeaway to apply to the next similar decision. Be specific and terse "
    "in every field — this will be re-read by future agents, and separately "
    "aggregated across many decisions to spot recurring patterns, so each "
    "field must stand alone without the others for context. When both a "
    "target-weight breakdown and a REALIZED RETURN breakdown are given for "
    "the same strategies, the realized returns are the actual P&L ground "
    "truth — a weight is a position size, never a return or a loss figure; "
    "do not cite a weight number as if it were P&L."
)

# Confirmed live: schema-validation failures here (degenerate/garbled LLM
# output) are common enough that ~1/3 of reflected decisions were losing
# their self-assessment permanently to a single bad sample. See reflect()'s
# retry loop.
_REFLECTION_MAX_ATTEMPTS = 3

# RAG collection recommendations are written to/read from — see the module
# docstring's "Recommendation self-improvement loop" section.
_RECOMMENDATIONS_COLLECTION = "recommendations"


class TradingMemoryLog:
    """Portfolio-level decision log with outcome-triggered LLM reflection.

    Args:
        config: Dict that may contain:
            ``memory_log_path``      — path to the JSONL file (str/Path).
            ``memory_max_context``   — max entries returned by get_context (int, default 5).
            ``memory_rag_n_results`` — max past recommendations injected into
                                       reflect_day()'s prompt (int, default 3).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        raw_path = cfg.get("memory_log_path", str(_DEFAULT_PATH))
        resolved = Path(raw_path).expanduser()
        if resolved == _REAL_PROD_PATH.expanduser() and os.environ.get("PYTEST_CURRENT_TEST"):
            # A test resolving to the real production decisions file would
            # silently contaminate it (this exact bug shipped and had to be
            # cleaned up twice: commit 8822c31, then again here). Checking
            # the *resolved* path against the never-patched _REAL_PROD_PATH
            # (not `_DEFAULT_PATH`, which tests legitimately monkeypatch —
            # see TestMemoryDecisionsAPI in tests/test_api.py, whose
            # isolation *is* patching that global) means both valid
            # isolation patterns pass: explicit memory_log_path in config,
            # or monkeypatching `_DEFAULT_PATH` itself.
            raise RuntimeError(
                f"TradingMemoryLog resolved to the real production path "
                f"{_REAL_PROD_PATH} while running under pytest. Pass "
                "memory_log_path=str(tmp_path / 'decisions.jsonl') in the "
                "test's config, or monkeypatch firm.agents.memory._DEFAULT_PATH."
            )
        self._path = resolved
        self._max_context: int = int(cfg.get("memory_max_context", 5))
        self._rag_n_results: int = int(cfg.get("memory_rag_n_results", 3))
        # Lazily constructed on first use (see _get_rag_store/_get_rag_retriever)
        # so a bare-backtest context with the llm/rag extras uninstalled never
        # pays an import cost it doesn't need.
        self._rag_store: Any = None
        self._rag_retriever: Any = None

    # ── Phase A ─────────────────────────────────────────────────────────────

    def store_decision(
        self,
        date: str,
        proposal_weights: dict[str, float],
        notes: str = "",
        nav_at_decision: float | None = None,
        per_strategy: dict[str, dict[str, float]] | None = None,
        cycle_id: int | None = None,
    ) -> None:
        """Record a pending decision immediately after the orchestrator runs.

        Args:
            date:             ISO date string for this rebalance (YYYY-MM-DD).
            proposal_weights: Target weight dict {symbol: weight} from the
                              approved TradeProposal.
            notes:            Optional brief context (regime, top signals, etc.).
            nav_at_decision:  Portfolio NAV at decision time, persisted so a
                              later ``reflect()``/``reflect_day()`` call can
                              compute the return even if the caller (e.g. the
                              live engine) restarted and lost any in-memory
                              pointer to it.
            per_strategy:     {strategy: {symbol: weight}} attribution of the
                              final blended targets (``TradeProposal.
                              per_strategy``) — without this, a reflection
                              can only judge "the portfolio" as a whole, with
                              no way to say which strategy's calls were
                              actually right or wrong that cycle. Same
                              traceability gap as order history not
                              recording which strategy placed an order;
                              fed into reflect()'s own prompt below, not
                              just stored for display.
            cycle_id:         The engine's per-cycle counter (2026-09-20).
                              Real incident: under the "hourly_market_hours"
                              schedule (~7 cycles/trading day), the storage
                              key used to be *just* ``date`` with
                              first-writer-wins idempotency — only the day's
                              very first cycle's decision was ever stored;
                              the other ~6 silently vanished. Passing
                              ``cycle_id`` makes the storage key
                              ``f"{date}#{cycle_id}"`` so every cycle's
                              decision survives. ``None`` (the old call
                              shape) preserves exact legacy behavior — one
                              entry per date, keyed by date alone — so
                              existing callers/tests are unaffected.
        """
        key = self._storage_key(date, cycle_id)
        if self._idempotency_check(key):
            return
        entry = {
            "date": date,
            "cycle_id": cycle_id,
            "status": "pending",
            "proposal_weights": proposal_weights,
            "per_strategy": per_strategy or {},
            "notes": notes,
            "nav_at_decision": nav_at_decision,
            "raw_return": None,
            "benchmark_return": None,
            "reflection": None,
            "verdict": None,
            "what_worked": None,
            "what_failed": None,
            "lesson": None,
        }
        self._append(entry)
        log.debug("Memory: stored pending decision for %s (cycle_id=%s)", date, cycle_id)

    # ── Phase B ─────────────────────────────────────────────────────────────

    def reflect(
        self,
        date: str,
        raw_return: float,
        benchmark_return: float,
        llm_service: Any,
    ) -> str | None:
        """Generate and persist a reflection once the outcome is known.

        Looks up the pending entry for *date*, calls the LLM to produce a
        retrospective, and appends a "reflected" entry to the log.

        Args:
            date:             ISO date string matching the original decision.
            raw_return:       Portfolio return over the holding period (e.g. 0.023).
            benchmark_return: Benchmark (e.g. SPY) return for the same period.
            llm_service:      ``firm.llm.provider.LLMService`` instance.

        Returns:
            The reflection text, or None if no pending entry was found.
        """
        pending = self._find_pending(date)
        if pending is None:
            log.debug("Memory: no pending entry for %s — skipping reflection", date)
            return None

        alpha = raw_return - benchmark_return
        per_strategy = pending.get("per_strategy") or {}
        per_strategy_block = (
            f"\nPer-strategy attribution (which strategy targeted which symbol/weight): "
            f"{json.dumps(per_strategy, indent=2)}"
            if per_strategy
            else ""
        )
        user_prompt = (
            f"Decision date: {date}\n"
            f"Portfolio return: {raw_return:+.2%}\n"
            f"Benchmark return: {benchmark_return:+.2%}\n"
            f"Alpha: {alpha:+.2%}\n\n"
            f"Original notes: {pending.get('notes', 'none')}\n"
            f"Target weights: {json.dumps(pending.get('proposal_weights', {}), indent=2)}"
            f"{per_strategy_block}"
        )
        verdict, what_worked, what_failed, lesson, reflection, _rec = self._call_reflection_llm(
            user_prompt, label=date, raw_return=raw_return, alpha=alpha, llm_service=llm_service,
        )

        entry = {
            **pending,
            "status": "reflected",
            "raw_return": raw_return,
            "benchmark_return": benchmark_return,
            "reflection": reflection,
            "verdict": verdict,
            "what_worked": what_worked,
            "what_failed": what_failed,
            "lesson": lesson,
        }
        self._append(entry)
        log.info(
            "Memory: reflected on %s — return %+.2f%%, alpha %+.2f%%, verdict=%s",
            date, raw_return * 100, alpha * 100, verdict,
        )
        return reflection

    def reflect_day(
        self,
        date: str,
        raw_return: float,
        benchmark_return: float,
        llm_service: Any,
        strategy_performance: dict[str, dict[str, float]] | None = None,
    ) -> str | None:
        """Aggregate every cycle's still-pending decision for *date* into
        ONE daily-rollup LLM reflection call (2026-09-20).

        Replaces per-cycle reflection under the "hourly_market_hours"
        schedule: reflecting ~7 times/trading day would multiply LLM cost
        the same way running the full pipeline hourly would have (see
        Orchestrator._apply_cycle_llm_mode) — this keeps reflection cost at
        one call/day while ``store_decision``'s ``cycle_id`` key still keeps
        every individual cycle's decision on disk (see its own docstring).

        Writes ONE entry keyed ``f"{date}#rollup"`` with ``status:
        "reflected"`` — after this, ``find_all_pending()`` stops returning
        *any* of that date's per-cycle pending entries (it excludes dates
        that already have a "reflected" entry), so this is naturally
        idempotent: calling it again for an already-rolled-up date is a
        no-op via the same idempotency check every other entry uses.

        May optionally include a bounded ``DailyReflectionRecommendation``
        (see its own docstring) — never applied automatically; surfaced via
        ``list_recommendations()``/the live API for a human to act on. Any
        actionable recommendation is additionally passed through
        ``_sanity_check_recommendation`` against ``strategy_performance``
        (see its own docstring) before being persisted — a second,
        mechanical layer behind the prompt's own "one bad day is not
        sufficient evidence" instruction, since that instruction alone was
        confirmed live to not reliably stop the LLM acting on a strategy
        that didn't actually have a bad (or even negative) day.

        Args:
            strategy_performance: ``{strategy: {"return": float,
                "trailing_mean": float, "trailing_std": float,
                "trailing_n": float}}`` — REAL realized per-strategy returns
                for *date* (the trailing-stat keys are only present with
                enough history for a meaningful baseline), built by
                ``LiveTradingEngine._build_strategy_performance``. Distinct
                from ``per_strategy`` (target *weights*) above — without
                this, the LLM only ever saw weights, which a real incident
                showed it can conflate with a return figure. ``None``
                (the default) preserves the exact old prompt/behavior for
                any caller that doesn't supply it (e.g. older tests).
        """
        key = self._storage_key(date, "rollup")
        if self._idempotency_check(key):
            log.debug("Memory: %s already has a daily rollup — skipping", date)
            return None

        entries = self._load_all()
        day_entries = sorted(
            (
                e for e in entries.values()
                if e.get("date") == date and e.get("status") == "pending"
            ),
            key=lambda e: e.get("cycle_id") or 0,
        )
        if not day_entries:
            log.debug("Memory: no pending entries for %s — skipping daily rollup", date)
            return None

        alpha = raw_return - benchmark_return
        # Combine per-strategy attribution across the day's cycles -- a
        # later cycle's target for a given (strategy, symbol) pair
        # supersedes an earlier one, matching how the real book actually
        # evolved intraday.
        combined_per_strategy: dict[str, dict[str, float]] = {}
        for e in day_entries:
            for strat, weights in (e.get("per_strategy") or {}).items():
                combined_per_strategy.setdefault(strat, {}).update(weights)
        notes = "; ".join(n for n in (e.get("notes") for e in day_entries) if n)
        final_weights = day_entries[-1].get("proposal_weights", {})
        per_strategy_block = (
            f"\nTarget weights only (NOT returns — see the realized-return "
            f"section below for what each strategy actually made/lost) — "
            f"combined per-strategy attribution across the day's "
            f"{len(day_entries)} decision cycle(s): "
            f"{json.dumps(combined_per_strategy, indent=2)}"
            if combined_per_strategy
            else ""
        )
        performance_block = self._render_strategy_performance_block(strategy_performance)
        history_block = self._retrieve_recommendation_history(
            strategies=list(combined_per_strategy), alpha=alpha,
        )
        user_prompt = (
            f"Decision date: {date} ({len(day_entries)} decision cycle(s) that day)\n"
            f"Portfolio return: {raw_return:+.2%}\n"
            f"Benchmark return: {benchmark_return:+.2%}\n"
            f"Alpha: {alpha:+.2%}\n\n"
            f"Notes across the day's cycles: {notes or 'none'}\n"
            f"Final target weights at day's last cycle: {json.dumps(final_weights, indent=2)}"
            f"{per_strategy_block}"
            f"{performance_block}"
            f"{history_block}\n\n"
            "If, and only if, this day's outcome clearly points to one specific "
            "strategy that should be sized down or flagged for review, you may "
            'set "recommendation" to {"action": "reduce_position_limit"|'
            '"flag_strategy_for_review", "strategy": "<name>", "reduce_by_pct": '
            'float (0-1, only for reduce_position_limit), "rationale": "..."}. '
            'Otherwise leave it {"action": "no_action"}. Base any such call ONLY '
            "on the strategy's ACTUAL REALIZED RETURN and z-score above, never "
            "on a target weight (a weight is a position size, not a P&L figure — "
            "do not cite one as if it were a loss). One bad day is not "
            "sufficient evidence — only recommend an action for a clear, "
            "specific, named strategy failure that is genuinely unusual for "
            "that strategy (roughly z < -1), not a vague market-wide move or an "
            "ordinary day within that strategy's normal range."
        )
        verdict, what_worked, what_failed, lesson, reflection, recommendation = (
            self._call_reflection_llm(
                user_prompt, label=date, raw_return=raw_return, alpha=alpha,
                llm_service=llm_service,
            )
        )
        recommendation = self._sanity_check_recommendation(recommendation, strategy_performance)

        entry = {
            "date": date,
            # A non-None sentinel (not a real int cycle_id) so
            # _storage_key/_load_all's own re-keying reproduces exactly the
            # "date#rollup" key this method's idempotency check above uses
            # -- a bare-date key would collide with legacy pre-cycle_id log
            # entries and wouldn't match this method's own re-check.
            "cycle_id": "rollup",
            "status": "reflected",
            "proposal_weights": final_weights,
            "per_strategy": combined_per_strategy,
            "notes": notes,
            "nav_at_decision": day_entries[0].get("nav_at_decision"),
            "raw_return": raw_return,
            "benchmark_return": benchmark_return,
            "reflection": reflection,
            "verdict": verdict,
            "what_worked": what_worked,
            "what_failed": what_failed,
            "lesson": lesson,
            "recommendation": recommendation.model_dump() if recommendation else None,
            "n_cycles": len(day_entries),
        }
        self._append(entry)
        log.info(
            "Memory: daily rollup reflection for %s (%d cycles) — return %+.2f%%, "
            "alpha %+.2f%%, verdict=%s",
            date, len(day_entries), raw_return * 100, alpha * 100, verdict,
        )
        self._store_recommendation_doc(
            date=date, recommendation=recommendation, raw_return=raw_return,
            benchmark_return=benchmark_return, alpha=alpha,
            strategies=list(combined_per_strategy),
        )
        return reflection

    @staticmethod
    def _render_strategy_performance_block(
        strategy_performance: dict[str, dict[str, float]] | None,
    ) -> str:
        """Render ``strategy_performance`` (see ``reflect_day``'s own
        docstring) into a prompt block, sorted worst-return-first so the
        LLM sees the full day's cross-section at a glance — including
        whether the strategy it's about to name was actually the standout
        loser or just one of several similarly-negative strategies (a real
        incident this specifically catches: a large portfolio move blamed
        on one strategy whose own return was an order of magnitude too
        small to explain it, while several *other* strategies had a
        similar-sized move the same day).

        Returns "" when *strategy_performance* is empty/None — reflect_day()
        must produce the exact same prompt as before this feature existed
        for any caller that doesn't supply it.
        """
        if not strategy_performance:
            return ""
        lines = [
            "\nACTUAL REALIZED per-strategy returns for this date (ground "
            "truth for P&L — use these, not the target weights above, to "
            "judge which strategy actually made or lost money; z-score is "
            "how unusual that day's return was relative to that strategy's "
            "own recent history, not the portfolio's):"
        ]
        ranked = sorted(strategy_performance.items(), key=lambda kv: kv[1].get("return", 0.0))
        for strat, stats in ranked:
            ret = stats.get("return", 0.0)
            line = f"  - {strat}: {ret:+.3%}"
            std = stats.get("trailing_std")
            mean = stats.get("trailing_mean")
            n = stats.get("trailing_n")
            if std is not None and mean is not None and n:
                line += f" (trailing {int(n)}-day mean {mean:+.3%}, std {std:.3%}"
                if std > 1e-9:
                    z = (ret - mean) / std
                    line += f", z={z:+.2f}"
                line += ")"
            else:
                line += " (insufficient history for a trailing baseline yet)"
            lines.append(line)
        return "\n".join(lines) + "\n"

    # Mechanical floor for the sanity gate below -- a strategy's actual
    # return must be at least this many trailing standard deviations below
    # its own recent mean before an LLM-proposed action against it is
    # honored, whenever enough history exists to compute one. Deliberately
    # a named module constant (not buried in the method) so it's easy to
    # find and re-tune from observed false-positive/false-negative rates
    # once more days of real data accumulate.
    _RECOMMENDATION_Z_SCORE_FLOOR = -1.0

    @classmethod
    def _sanity_check_recommendation(
        cls,
        recommendation: "DailyReflectionRecommendation | None",
        strategy_performance: dict[str, dict[str, float]] | None,
    ) -> "DailyReflectionRecommendation | None":
        """Downgrade an LLM-proposed action to ``no_action`` when it isn't
        actually backed by that strategy's own realized return (2026-09-25).

        A second, mechanical/code-enforced layer behind the prompt's own
        "one bad day is not sufficient evidence" instruction — added after
        an independent audit of every pending recommendation this system
        had ever produced found 3 of 4 were wrong, and specifically wrong
        in ways this check catches directly:

        - A strategy flagged for a "large loss" that actually had a
          POSITIVE return that day (regime_hmm, +0.38%, flagged anyway) —
          caught by the "return must be negative" check below.
        - A strategy flagged over a real but tiny/immaterial loss well
          within its own normal range (volatility_breakout, -0.016%,
          lifetime Sharpe +0.36) — caught by the z-score check, which the
          sign check alone would have missed (it *was* negative).

        Never upgrades a ``no_action`` into an action, and never invents a
        new one — only ever downgrades an action-taking recommendation it
        can't independently verify, appending a note to ``rationale`` that
        explains why rather than silently discarding the original text.
        """
        if recommendation is None or recommendation.action == "no_action":
            return recommendation
        if strategy_performance is None:
            # No performance data was supplied at all -- the caller hasn't
            # opted into this check (e.g. reflect()'s legacy per-decision
            # path, which never builds strategy_performance, or an older
            # test). Preserve the exact old behavior rather than blocking
            # every caller that predates this feature; every real
            # production call site (LiveTradingEngine._maybe_reflect)
            # always supplies it, so this only matters for compatibility.
            return recommendation
        strat = recommendation.strategy
        stats = strategy_performance.get(strat) if strat else None
        if not stats:
            log.warning(
                "Memory: downgrading recommendation (%s for %r) to no_action "
                "— no realized-return data available to verify it",
                recommendation.action, strat,
            )
            return recommendation.model_copy(update={
                "action": "no_action",
                "rationale": (
                    f"[auto-downgraded from {recommendation.action}: no "
                    f"realized-return data for {strat!r} to verify against] "
                    f"{recommendation.rationale}"
                ),
            })
        ret = stats.get("return", 0.0)
        if ret >= 0:
            log.warning(
                "Memory: downgrading recommendation (%s for %s) to no_action "
                "— its actual realized return that day was %+.3f%% (not "
                "negative), contradicting the stated rationale",
                recommendation.action, strat, ret,
            )
            return recommendation.model_copy(update={
                "action": "no_action",
                "rationale": (
                    f"[auto-downgraded from {recommendation.action}: {strat} "
                    f"actually returned {ret:+.3%} that day, not a loss] "
                    f"{recommendation.rationale}"
                ),
            })
        std = stats.get("trailing_std")
        mean = stats.get("trailing_mean")
        if std is not None and mean is not None and std > 1e-9:
            z = (ret - mean) / std
            if z > cls._RECOMMENDATION_Z_SCORE_FLOOR:
                log.warning(
                    "Memory: downgrading recommendation (%s for %s) to "
                    "no_action — z-score %.2f is not unusual enough for that "
                    "strategy (floor %.2f)",
                    recommendation.action, strat, z, cls._RECOMMENDATION_Z_SCORE_FLOOR,
                )
                return recommendation.model_copy(update={
                    "action": "no_action",
                    "rationale": (
                        f"[auto-downgraded from {recommendation.action}: "
                        f"{strat}'s return that day (z={z:+.2f}) is within its "
                        f"own normal range, not a genuine anomaly] "
                        f"{recommendation.rationale}"
                    ),
                })
        return recommendation

    def _call_reflection_llm(
        self,
        user_prompt: str,
        *,
        label: str,
        raw_return: float,
        alpha: float,
        llm_service: Any,
    ) -> tuple[str, str, str, str, str, "DailyReflectionRecommendation | None"]:
        """Shared LLM-call/retry/parse logic for both ``reflect()`` and
        ``reflect_day()`` — returns
        ``(verdict, what_worked, what_failed, lesson, reflection_text, recommendation)``.
        """
        messages = [
            {"role": "system", "content": _REFLECTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        # A schema-validation failure here (confirmed live: degenerate
        # word-repetition loops, garbled JSON) permanently collapses this
        # decision's self-assessment to "unknown" with no way to recover
        # it later — unlike every other LLM+schema call site in this
        # codebase, which falls back to a quant-only value that's still
        # useful. One retry costs nothing this deferred/off-critical-path
        # call isn't latency-sensitive, and a bad sample is often a one-off
        # sampling hiccup rather than a systematic prompt problem.
        #
        # The retry only helps if it can actually reach a different model:
        # LLMService's load-balance routing is a deterministic hash of the
        # message content, so an identical retry with no override reliably
        # re-picks the exact same model that just failed. Reflection is only
        # called a couple of times a day per instance -- far too low-volume
        # to need spreading across the free-tier pool the way high-frequency
        # per-signal enhancement calls do -- so the retry explicitly forces
        # the service's own default_model rather than re-rolling the same
        # dice.
        parsed: DecisionReflection | None = None
        for attempt in range(1, _REFLECTION_MAX_ATTEMPTS + 1):
            model_override = llm_service.default_model if attempt > 1 else None
            try:
                raw = llm_service.chat_json(messages, model=model_override)
                parsed = parse_llm_response(
                    DecisionReflection, raw,
                    context=f"memory/{label} (attempt {attempt}/{_REFLECTION_MAX_ATTEMPTS})",
                )
            except Exception as exc:
                log.warning(
                    "Memory: LLM reflection call failed for %s (attempt %d/%d, model=%s): %s",
                    label, attempt, _REFLECTION_MAX_ATTEMPTS,
                    model_override or "load-balanced", exc, exc_info=True,
                )
                parsed = None
            if parsed is not None:
                break

        if parsed is not None:
            verdict, what_worked, what_failed, lesson = (
                parsed.verdict, parsed.what_worked, parsed.what_failed, parsed.lesson,
            )
            recommendation = parsed.recommendation
            # Rendered prose kept for backward-compat prompt injection
            # (get_context()) — existing consumers read a single string,
            # not the structured fields.
            reflection = (
                f"{verdict.upper()}. "
                + (f"What worked: {what_worked} " if what_worked else "")
                + (f"What failed: {what_failed} " if what_failed else "")
                + (f"Lesson: {lesson}" if lesson else "")
            ).strip()
        else:
            verdict, what_worked, what_failed, lesson = "unknown", "", "", ""
            recommendation = None
            reflection = (
                f"Outcome: {raw_return:+.2%} raw / {alpha:+.2%} alpha. "
                "(reflection unavailable)"
            )
        return verdict, what_worked, what_failed, lesson, reflection, recommendation

    # ── Recommendation RAG loop ──────────────────────────────────────────────
    # See the module docstring's "Recommendation self-improvement loop"
    # section. Every method below is best-effort: a failure here must never
    # break reflect_day() or mark_recommendation_applied(), it only means
    # today's call doesn't get the historical-context enrichment.

    def _get_rag_store(self) -> Any:
        """Lazily construct the shared vector store, or raise.

        Same lazy-construct-and-cache pattern as
        ``LLMAgentMixin._get_retriever`` — this class has no hard dependency
        on the ``rag``/``llm`` extras, so the import only happens on first
        real use.

        Refuses to construct a real store pointed at the shared production
        ``data/vectordb`` while running under pytest (see
        ``_REAL_PROD_VECTORDB``) — every caller of this method already
        treats any exception here as "RAG unavailable this call" and
        degrades gracefully, so this is a safety net, not a new failure
        mode: a test that wants real RAG coverage should mock
        ``_rag_store``/``_rag_retriever`` (see tests/test_memory.py) or
        point ``config/llm.yaml``'s ``rag.persist_dir`` elsewhere.
        """
        if self._rag_store is None:
            from firm.llm.config import rag_config
            from firm.rag.store import VectorStore

            persist_dir = Path(rag_config().get("persist_dir", "data/vectordb")).expanduser()
            if persist_dir == _REAL_PROD_VECTORDB.expanduser() and os.environ.get("PYTEST_CURRENT_TEST"):
                raise RuntimeError(
                    f"Refusing to write to the real production vector store "
                    f"{_REAL_PROD_VECTORDB} while running under pytest; mock "
                    "TradingMemoryLog._get_rag_store or set a different "
                    "rag.persist_dir for this test."
                )
            self._rag_store = VectorStore()
        return self._rag_store

    def _get_rag_retriever(self) -> Any:
        """Lazily construct the shared retriever over ``_get_rag_store()``."""
        if self._rag_retriever is None:
            from firm.llm.config import rag_config
            from firm.rag.retriever import RAGRetriever

            rag = rag_config()
            self._rag_retriever = RAGRetriever(
                self._get_rag_store(),
                reranker=bool(rag.get("reranking", True)),
                hybrid=bool(rag.get("hybrid", False)),
                reranker_provider=rag.get("reranker_provider"),
                reranker_model=rag.get("reranker_model"),
            )
        return self._rag_retriever

    @staticmethod
    def _recommendation_doc_id(date: str) -> str:
        """Stable id for *date*'s rollup recommendation doc.

        Deliberately NOT a content hash (contrast
        ``firm.rag.chunker.DocumentChunker``'s chunk ids): this id must stay
        identical across the initial write and the later
        ``mark_recommendation_applied()`` metadata update, and there's
        exactly one recommendation doc per date (``reflect_day`` is
        idempotent per date), so a plain date-keyed id is sufficient and
        lets ``VectorStore.update_metadata`` find the same row later.
        """
        return f"recommendation:{date}"

    def _store_recommendation_doc(
        self,
        *,
        date: str,
        recommendation: "DailyReflectionRecommendation | None",
        raw_return: float,
        benchmark_return: float,
        alpha: float,
        strategies: list[str],
    ) -> None:
        """Write *date*'s daily-rollup recommendation into the RAG
        ``"recommendations"`` collection (even a ``no_action`` one — the
        absence of a recommendation for a given situation is itself a
        useful precedent for a future retrieval). Skipped when
        *recommendation* is ``None``, i.e. the LLM call itself failed and
        there is no actual recommendation content to store.

        Never raises: a RAG write failure only costs this one day's
        historical-context contribution, not the reflection itself (which
        is already persisted to the JSONL log by the time this runs).
        """
        if recommendation is None:
            return
        try:
            from firm.rag.models import Document

            strategy_line = f" Strategies involved that day: {', '.join(strategies)}." if strategies else ""
            reduce_line = (
                f" (reduce by {recommendation.reduce_by_pct:.0%})"
                if recommendation.action == "reduce_position_limit"
                else ""
            )
            text = (
                f"Daily reflection recommendation for {date}. "
                f"Portfolio return {raw_return:+.2%}, benchmark {benchmark_return:+.2%}, "
                f"alpha {alpha:+.2%}.{strategy_line}\n"
                f"Recommended action: {recommendation.action}"
                + (f" for strategy \"{recommendation.strategy}\"" if recommendation.strategy else "")
                + f"{reduce_line}.\n"
                f"Rationale: {recommendation.rationale or '(none given)'}"
            )
            doc = Document(
                doc_id=self._recommendation_doc_id(date),
                text=text,
                metadata={
                    "date": date,
                    "doc_type": "recommendation",
                    "action": recommendation.action,
                    "strategy": recommendation.strategy,
                    "reduce_by_pct": recommendation.reduce_by_pct,
                    "rationale": recommendation.rationale,
                    "applied": False,
                },
            )
            self._get_rag_store().add_documents(_RECOMMENDATIONS_COLLECTION, [doc])
        except Exception:
            log.warning(
                "Memory: RAG write failed for %s's recommendation — proceeding "
                "without storing it for future self-improvement context",
                date, exc_info=True,
            )

    def _retrieve_recommendation_history(
        self, *, strategies: list[str], alpha: float,
    ) -> str:
        """Retrieve a bounded set of relevant past recommendations from RAG
        for injection into ``reflect_day()``'s prompt — the actual
        self-improvement mechanism: today's reflection gets to see what was
        recommended (and whether it was applied) for semantically similar
        past situations, instead of reasoning from scratch every day.

        Returns "" (never raises) when RAG is unavailable, the collection is
        empty, or nothing relevant is found — reflect_day() must work
        identically either way, just without this enrichment.
        """
        query = (
            "Daily trading reflection"
            + (f" for strategies: {', '.join(strategies)}" if strategies else "")
            + f". Portfolio alpha {alpha:+.2%}. Should any strategy have its "
            "position limit reduced or be flagged for review?"
        )
        try:
            docs = self._get_rag_retriever().retrieve(
                query, collection=_RECOMMENDATIONS_COLLECTION, n_results=self._rag_n_results,
            )
        except ImportError:
            return ""  # already logged by _get_rag_store/_get_rag_retriever
        except Exception:
            log.warning(
                "Memory: RAG retrieval of past recommendations failed — "
                "proceeding without historical context this reflection",
                exc_info=True,
            )
            return ""
        if not docs:
            return ""

        lines = ["\n\nPast recommendations for similar situations:"]
        for d in docs:
            meta = d.metadata
            applied = "yes" if meta.get("applied") else "no"
            lines.append(
                f"- [{meta.get('date', '?')}] {meta.get('action', '?')} for "
                f"{meta.get('strategy') or 'the portfolio'} (applied: {applied}) — "
                f"{meta.get('rationale', '')}"
            )
        return "\n".join(lines)

    def list_recommendations(self, pending_only: bool = True) -> list[dict[str, Any]]:
        """Every daily-rollup recommendation on record, most recent first.

        ``pending_only=True`` (default) filters to actionable
        (``action != "no_action"``) recommendations that haven't already
        been applied (see ``mark_recommendation_applied``) — the only
        entries actually worth a human's attention. Each dict includes
        ``date`` and ``rollup_reflection`` alongside the recommendation
        fields, so the API layer needs no separate lookup to show context
        for why it was proposed. This is intentionally read-only data
        assembly — nothing here applies a recommendation; that's a
        separate, explicit human action (see ``firm.api.routers.live``'s
        recommendations endpoints).
        """
        entries = sorted(
            self._load_all().values(), key=lambda e: e.get("date", ""), reverse=True,
        )
        out: list[dict[str, Any]] = []
        for e in entries:
            rec = e.get("recommendation")
            if not rec:
                continue
            if pending_only and (rec.get("action") == "no_action" or rec.get("applied")):
                continue
            out.append({
                "date": e.get("date"),
                "rollup_reflection": e.get("reflection"),
                **rec,
            })
        return out

    def mark_recommendation_applied(self, date: str) -> bool:
        """Mark *date*'s daily-rollup recommendation as applied.

        Called only after a human explicitly approves it via
        ``POST /api/live/recommendations/{date}/apply`` — this method
        itself never decides whether to apply anything, only records that
        it happened. Returns False if no rollup with an actionable
        recommendation exists for *date*.
        """
        entries = self._load_all()
        entry = entries.get(self._storage_key(date, "rollup"))
        if entry is None or not entry.get("recommendation"):
            return False
        entry = dict(entry)
        entry["recommendation"] = {**entry["recommendation"], "applied": True}
        self._append(entry)
        self._mark_recommendation_doc_applied(date)
        return True

    def _mark_recommendation_doc_applied(self, date: str) -> None:
        """Flip the RAG-stored recommendation doc's ``applied`` metadata to
        ``True`` in place (see ``VectorStore.update_metadata``), so a future
        retrieval sees the accurate outcome instead of the stale ``False``
        it was written with. The JSONL log above is the source of truth —
        this is a best-effort mirror for RAG retrieval only, so any failure
        here is logged and swallowed rather than propagated.
        """
        try:
            updated = self._get_rag_store().update_metadata(
                _RECOMMENDATIONS_COLLECTION,
                self._recommendation_doc_id(date),
                {"applied": True},
            )
            if not updated:
                log.warning(
                    "Memory: no RAG recommendation doc found for %s to mark applied "
                    "(JSONL log was still updated correctly)", date,
                )
        except Exception:
            log.warning(
                "Memory: RAG update failed marking %s's recommendation applied "
                "(JSONL log was still updated correctly)", date, exc_info=True,
            )

    # ── Context injection ────────────────────────────────────────────────────

    def get_context(self, n: int | None = None) -> str:
        """Return the last *n* reflected decisions formatted for LLM injection.

        Args:
            n: Number of entries to return (defaults to ``memory_max_context``).

        Returns:
            A compact markdown string, or empty string when no reflections exist.
        """
        n = n or self._max_context
        entries = self._load_all()
        reflected = [e for e in entries.values() if e.get("status") == "reflected"]
        recent = sorted(reflected, key=lambda e: e["date"])[-n:]
        if not recent:
            return ""
        lines = ["**Past decisions and outcomes:**\n"]
        for e in recent:
            alpha = (e.get("raw_return") or 0) - (e.get("benchmark_return") or 0)
            lines.append(
                f"[{e['date']}] Return: {(e.get('raw_return') or 0):+.2%} "
                f"(alpha: {alpha:+.2%})\n{e.get('reflection', '')}\n"
            )
        return "\n".join(lines)

    def summarize_lessons(self, n: int | None = None) -> dict[str, Any]:
        """Aggregate verdict counts and recent distinct lessons across every
        reflected decision — a lightweight "lessons learned" digest.

        Pure aggregation over the structured fields ``reflect()`` already
        persists (no new LLM call): what fraction of past calls were
        correct/incorrect/partial, and the *n* most recent non-empty
        ``lesson`` strings, most-recent-first — surfacing recurring
        patterns that were previously invisible inside individual
        unstructured reflection blobs.

        Args:
            n: How many recent lessons to return (default 10).

        Returns:
            ``{"total": int, "counts": {"correct", "incorrect", "partial",
            "unknown"}, "recent_lessons": [str, ...]}``.
        """
        n = n or 10
        entries = self._load_all()
        reflected = sorted(
            (e for e in entries.values() if e.get("status") == "reflected"),
            key=lambda e: e["date"],
        )
        counts = {"correct": 0, "incorrect": 0, "partial": 0, "unknown": 0}
        lessons: list[str] = []
        for e in reflected:
            verdict = e.get("verdict") or "unknown"
            counts[verdict] = counts.get(verdict, 0) + 1
            lesson = (e.get("lesson") or "").strip()
            if lesson:
                lessons.append(lesson)
        return {
            "total": len(reflected),
            "counts": counts,
            "recent_lessons": list(reversed(lessons[-n:])),
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _storage_key(date: str, cycle_id: int | str | None) -> str:
        """Idempotency/supersede key for an entry (2026-09-20).

        ``cycle_id is None`` preserves the original bare-``date`` key
        exactly (legacy callers, and any date whose entries pre-date this
        change) — only a caller that actually supplies a ``cycle_id`` gets
        the finer-grained composite key.
        """
        return date if cycle_id is None else f"{date}#{cycle_id}"

    def _load_all(self) -> dict[str, dict]:
        """Parse the JSONL file and return a dict keyed by ``_storage_key``.

        When multiple entries share the same key (pending overwritten by
        reflected), the last one wins — this is the desired supersede
        semantics. Entries written before ``cycle_id`` existed have none
        (``obj.get("cycle_id")`` is ``None``), so they key by bare date,
        identical to before this field existed.
        """
        if not self._path.exists():
            return {}
        entries: dict[str, dict] = {}
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                entries[self._storage_key(obj["date"], obj.get("cycle_id"))] = obj
        except Exception as exc:
            log.warning(
                "Memory: error reading log at %s: %s", self._path, exc, exc_info=True,
            )
        return entries

    def _find_pending(self, date: str) -> dict | None:
        """Legacy single-entry lookup — used by ``reflect()``'s original
        per-decision path. Returns the bare-``date``-keyed entry only (a
        cycle_id-keyed entry is only ever looked up via ``reflect_day``'s
        own date-scan), so this stays correct for pre-cycle_id log data."""
        entries = self._load_all()
        entry = entries.get(date)
        if entry and entry.get("status") == "pending":
            return entry
        return None

    def list_decisions(self, n: int | None = None) -> list[dict]:
        """Return decision entries (pending or reflected), most recent first.

        Used to expose the decision/reflection log to the API for GUI
        monitoring — the same data ``get_context()`` summarizes for LLM
        prompt injection, but as structured records instead of markdown.
        """
        entries = sorted(self._load_all().values(), key=lambda e: e["date"], reverse=True)
        return entries[:n] if n else entries

    def find_all_pending(self) -> list[dict]:
        """Return every decision still awaiting reflection, oldest first.

        Reads from disk rather than in-memory state, so a caller that
        restarted between the decision and the outcome becoming known (e.g.
        the live engine after a process restart) can still find and reflect
        on it — nothing is lost just because the in-process pointer was.

        Excludes any date that already has a "reflected" entry (2026-09-20)
        — once ``reflect_day()`` writes that date's rollup, its remaining
        per-cycle pending entries (there can be several, one per cycle_id)
        must stop being returned here, or every future call would keep
        re-surfacing them forever with nothing to actually do about it.
        """
        entries = self._load_all()
        reflected_dates = {e["date"] for e in entries.values() if e.get("status") == "reflected"}
        pending = [
            e for e in entries.values()
            if e.get("status") == "pending" and e["date"] not in reflected_dates
        ]
        return sorted(pending, key=lambda e: (e["date"], e.get("cycle_id") or 0))

    def _idempotency_check(self, key: str) -> bool:
        """Return True if a pending or reflected entry already exists for
        this storage key (see ``_storage_key`` — bare date for legacy
        callers, ``date#cycle_id`` otherwise)."""
        entries = self._load_all()
        return key in entries

    def _append(self, entry: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
