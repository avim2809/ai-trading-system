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

log = logging.getLogger("firm.agents.memory")

_DEFAULT_PATH = Path("data/memory/decisions.jsonl")

_REFLECTION_SYSTEM = (
    "You are a portfolio manager reviewing your own past trading decision "
    "now that the outcome is known. Write exactly 2-4 sentences of plain "
    "prose (no bullets, no headers, no markdown). Cover in order:\n"
    "1. Was the directional call correct? (cite the return figure)\n"
    "2. Which part of the thesis held or failed?\n"
    "3. One concrete lesson to apply to the next similar decision.\n"
    "Be specific and terse. Your output will be re-read by future agents, "
    "so every word must earn its place."
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
    ) -> None:
        """Record a pending decision immediately after the orchestrator runs.

        Args:
            date:             ISO date string for this rebalance (YYYY-MM-DD).
            proposal_weights: Target weight dict {symbol: weight} from the
                              approved TradeProposal.
            notes:            Optional brief context (regime, top signals, etc.).
        """
        if self._idempotency_check(date):
            return
        entry = {
            "date": date,
            "status": "pending",
            "proposal_weights": proposal_weights,
            "notes": notes,
            "raw_return": None,
            "benchmark_return": None,
            "reflection": None,
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
        try:
            messages = [
                {"role": "system", "content": _REFLECTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]
            reflection = llm_service.chat(messages)
        except Exception as exc:
            log.warning("Memory: LLM reflection failed for %s: %s", date, exc)
            reflection = f"Outcome: {raw_return:+.2%} raw / {alpha:+.2%} alpha. (reflection unavailable)"

        entry = {
            **pending,
            "status": "reflected",
            "raw_return": raw_return,
            "benchmark_return": benchmark_return,
            "reflection": reflection,
        }
        self._append(entry)
        log.info("Memory: reflected on %s — return %+.2f%%, alpha %+.2f%%",
                 date, raw_return * 100, alpha * 100)
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

    def _idempotency_check(self, date: str) -> bool:
        """Return True if a pending or reflected entry already exists for date."""
        entries = self._load_all()
        return date in entries

    def _append(self, entry: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
