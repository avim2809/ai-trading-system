"""Entry-point trial capture state (ticket P1-12). STDLIB ONLY, LIVE-IMPORT-PATH.

Every backtest entry point (``execute_backtest``, ``run_backtest_from_config``,
``ExperimentRunner.run``, the ``run-backtest`` console script) runs inside
:func:`capture_run`. The OUTERMOST call records one trial and nothing else:

* **Unarmed (default, including every firm-api process)**: append one JSON line to
  ``$FIRM_RESEARCH_LEDGER_ROOT/inbox/<tag>.jsonl``. No ``firm.research`` import, no git, no pandas.
* **Armed**: importing ``firm.research.capture`` sets :data:`LEDGER_SINK`; the hook then calls it
  and the trial goes to the hash-chained ledger as ``mode="unregistered"``.

Capture can never change, delay or fail a backtest: every capture step is wrapped and logs a
WARNING instead of raising. ``IN_API_PROCESS`` is a TAG only (inbox file name); safety never
depends on it.
"""

from __future__ import annotations

import contextvars
import fcntl
import hashlib
import json
import logging
import os
import socket
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

IN_API_PROCESS: bool = False  # tag only (inbox file name / source); never selects the ledger path
LEDGER_SINK: Callable[..., None] | None = None  # set ONLY by importing firm.research.capture

LEDGER_ROOT_ENV = "FIRM_RESEARCH_LEDGER_ROOT"
_DEFAULT_ROOT = "/local/store/research-ledger"

_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar("firm_capture_depth", default=0)
_OUTER: contextvars.ContextVar[Any] = contextvars.ContextVar("firm_capture_outer", default=None)
_warned: set[str] = set()


def mark_api_process() -> None:
    """Tag this process as a firm-api process (selects the ``api-<port>`` inbox file)."""
    global IN_API_PROCESS
    IN_API_PROCESS = True


def _warn_once(key: str, msg: str, *args: Any) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


def lossy_config_hash(config: Any) -> str:
    """sha256 of the sorted-key compact JSON of ``config`` (``default=repr``); lossy by design."""
    return hashlib.sha256(_dumps(config).encode()).hexdigest()


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=repr)


def _inbox_tag() -> str:
    if IN_API_PROCESS:
        port = "".join(c for c in os.environ.get("FIRM_API_PORT", "8000") if c.isdigit()) or "8000"
        return f"api-{port}"
    host = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in socket.gethostname())
    return f"proc-{host}"


def _append_inbox_json_line(entry: str, config: dict, cap: _Capture) -> None:
    env = os.environ.get(LEDGER_ROOT_ENV)
    if not env and os.environ.get("PYTEST_CURRENT_TEST"):
        _warn_once("pytest-default-root", "trial capture: refusing the default host ledger root under pytest")
        return
    inbox = os.path.join(env or _DEFAULT_ROOT, "inbox")
    if not os.path.isdir(inbox) or not os.access(inbox, os.W_OK | os.X_OK):
        _warn_once(
            "inbox-missing",
            "trial capture: inbox %s missing or not writable; capture line dropped (owner must provision it)",
            inbox,
        )
        return
    tag = _inbox_tag()
    line = {
        "ts": time.time(),
        "entry": entry,
        "source": "api" if IN_API_PROCESS else "process",
        "tag": tag,
        "pid": os.getpid(),
        "seed": cap.seed,
        "status": cap.status,
        "error": cap.error,
        "metrics": cap.metrics,
        "config_hash": lossy_config_hash(config),
        "config_hash_kind": "lossy_json_default_repr",
        "config": json.loads(_dumps(config)),
    }
    data = (json.dumps(line, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(os.path.join(inbox, f"{tag}.jsonl"), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            os.write(fd, data)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class _Capture:
    """Handle yielded by :func:`capture_run`; ``finish`` reads a report, never raises."""

    def __init__(self, entry: str, config: Any, seed: int | None) -> None:
        self.entry = entry
        self.config = config
        self.seed = seed if isinstance(seed, int) and not isinstance(seed, bool) else None
        self.status = "completed"
        self.error: str | None = None
        self.metrics: dict | None = None
        self.returns: Any = None  # pandas Series when the report exposes daily returns

    def finish(self, result: Any) -> None:
        try:
            returns = getattr(result, "returns", None)
            if returns is not None and hasattr(returns, "dropna") and len(returns):
                self.returns = returns
                outer = _OUTER.get()
                if outer is not None and outer is not self and outer.returns is None:
                    outer.returns = returns  # nested run: hand the daily returns to the recorded outer row
            metrics = getattr(result, "metrics", None)
            if isinstance(metrics, dict):
                self.metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))}
            if self.returns is not None and self.metrics is None:
                r = self.returns.dropna()
                self.metrics = {"n_obs": len(r)}
        except Exception:
            log.debug("trial capture: finish() could not read result", exc_info=True)


class capture_run:
    """Context manager used at each entry point. Never raises; records only when outermost."""

    def __init__(self, entry: str, config: Any, seed: int | None = None) -> None:
        self._cap = _Capture(entry, config, seed)
        self._depth = 0
        self._tokens: tuple[Any, Any] | None = None

    def __enter__(self) -> _Capture:
        try:
            self._depth = _DEPTH.get()
            t_depth = _DEPTH.set(self._depth + 1)
            t_outer = _OUTER.set(self._cap) if self._depth == 0 else _OUTER.set(_OUTER.get())
            self._tokens = (t_depth, t_outer)
        except Exception:
            self._tokens = None
        return self._cap

    def __exit__(self, exc_type, exc, tb) -> bool:
        cap = self._cap
        if exc is not None:
            cap.status, cap.error = "failed", repr(exc)[:2000]
        try:
            if self._tokens is not None:
                _OUTER.reset(self._tokens[1])
                _DEPTH.reset(self._tokens[0])
        except Exception:
            _DEPTH.set(self._depth)
        if self._depth == 0:
            self._record(cap)
        return False

    @staticmethod
    def _record(cap: _Capture) -> None:
        try:
            sink = LEDGER_SINK
            config = cap.config if isinstance(cap.config, dict) else {"config_repr": repr(cap.config)}
            if sink is not None:
                try:
                    sink(
                        cap.entry, config, seed=cap.seed, status=cap.status,
                        error=cap.error, metrics=cap.metrics, returns=cap.returns,
                    )
                    return
                except Exception:
                    log.warning(
                        "trial capture: ledger sink failed; falling back to the inbox "
                        "(backtest result unaffected)", exc_info=True,
                    )
            _append_inbox_json_line(cap.entry, config, cap)
        except Exception:
            log.warning("trial capture failed (backtest result unaffected)", exc_info=True)
