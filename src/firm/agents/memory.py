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
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from firm.llm.schemas import DecisionReflection, parse_llm_response

log = logging.getLogger("firm.agents.memory")

_DEFAULT_PATH = Path("data/memory/decisions.jsonl")

_REFLECTION_SYSTEM = (
    "You are a portfolio manager reviewing your own past trading decision "
    "now that the outcome is known. Respond with a single JSON object "
    '(no markdown, no commentary outside the JSON): {"verdict": "correct" | '
    '"incorrect" | "partial", "what_worked": "...", "what_failed": "...", '
    '"lesson": "..."}. "verdict" judges the directional call against the '
    "return figure. \"what_worked\" and \"what_failed\" each name a specific "
    "part of the original thesis (empty string if not applicable — e.g. "
    '"what_failed": "" for a fully correct call). "lesson" is one concrete '
    "takeaway to apply to the next similar decision. Be specific and terse "
    "in every field — this will be re-read by future agents, and separately "
    "aggregated across many decisions to spot recurring patterns, so each "
    "field must stand alone without the others for context."
)


class TradingMemoryLog:
    """Portfolio-level decision log with outcome-triggered LLM reflection.

    Args:
        config: Dict that may contain:
            ``memory_log_path``      — path to the JSONL file (str/Path).
            ``memory_max_context``   — max entries returned by get_context (int, default 5).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        raw_path = cfg.get("memory_log_path", str(_DEFAULT_PATH))
        self._path = Path(raw_path).expanduser()
        self._max_context: int = int(cfg.get("memory_max_context", 5))

    # ── Phase A ─────────────────────────────────────────────────────────────

    def store_decision(
        self,
        date: str,
        proposal_weights: dict[str, float],
        notes: str = "",
        nav_at_decision: float | None = None,
    ) -> None:
        """Record a pending decision immediately after the orchestrator runs.

        Args:
            date:             ISO date string for this rebalance (YYYY-MM-DD).
            proposal_weights: Target weight dict {symbol: weight} from the
                              approved TradeProposal.
            notes:            Optional brief context (regime, top signals, etc.).
            nav_at_decision:  Portfolio NAV at decision time, persisted so a
                              later ``reflect()`` call can compute the return
                              even if the caller (e.g. the live engine)
                              restarted and lost any in-memory pointer to it.
        """
        if self._idempotency_check(date):
            return
        entry = {
            "date": date,
            "status": "pending",
            "proposal_weights": proposal_weights,
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
        log.debug("Memory: stored pending decision for %s", date)

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
        user_prompt = (
            f"Decision date: {date}\n"
            f"Portfolio return: {raw_return:+.2%}\n"
            f"Benchmark return: {benchmark_return:+.2%}\n"
            f"Alpha: {alpha:+.2%}\n\n"
            f"Original notes: {pending.get('notes', 'none')}\n"
            f"Target weights: {json.dumps(pending.get('proposal_weights', {}), indent=2)}"
        )
        parsed: DecisionReflection | None = None
        try:
            messages = [
                {"role": "system", "content": _REFLECTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
            raw = llm_service.chat_json(messages)
            parsed = parse_llm_response(DecisionReflection, raw, context=f"memory/{date}")
        except Exception as exc:
            log.warning("Memory: LLM reflection failed for %s: %s", date, exc)

        if parsed is not None:
            verdict, what_worked, what_failed, lesson = (
                parsed.verdict, parsed.what_worked, parsed.what_failed, parsed.lesson,
            )
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
            reflection = (
                f"Outcome: {raw_return:+.2%} raw / {alpha:+.2%} alpha. "
                "(reflection unavailable)"
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

    def _load_all(self) -> dict[str, dict]:
        """Parse the JSONL file and return a dict keyed by date.

        When multiple entries share the same date (pending overwritten by
        reflected), the last one wins — this is the desired supersede semantics.
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
                entries[obj["date"]] = obj
        except Exception as exc:
            log.warning("Memory: error reading log at %s: %s", self._path, exc)
        return entries

    def _find_pending(self, date: str) -> dict | None:
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
        """
        entries = self._load_all()
        pending = [e for e in entries.values() if e.get("status") == "pending"]
        return sorted(pending, key=lambda e: e["date"])

    def _idempotency_check(self, date: str) -> bool:
        """Return True if a pending or reflected entry already exists for date."""
        entries = self._load_all()
        return date in entries

    def _append(self, entry: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
