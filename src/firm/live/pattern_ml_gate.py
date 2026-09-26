"""Fail-closed gate for pattern_recognition's ML knobs (CNN/XGBoost scoring).

Problem this closes: ``cnn_scoring_enabled`` was flipped live on 2026-09-20
(``63763c4``) on a single-window backtest result, self-flagged in its own
config comment as "not this codebase's usual 3-window robustness check —
revisit if live results diverge." It later failed a real 8-fold walk-forward
audit (``scripts/validate_pattern_cnn_walkforward.py``) and was rolled back
(``a8bb88f``) — but nothing in the code would have stopped that same
single-window-result mistake from happening again, or from happening to the
newer ``xgb_confirmation_enabled`` knob, which has never been walk-forward
validated at all. This module is the missing guardrail: it refuses to honor
either flag unless a passing record from ``scripts/pattern_recognition_rollout_gate.py``
exists on disk, mirroring ``firm.live.execution_safety``'s fail-closed style
(pure decision function, silent-downgrade-with-loud-log rather than a raised
exception, so a misconfigured/missing gate record degrades to the
already-live-tested rule-based scanner rather than crashing engine startup).

Deliberately NOT wired into ``firm.runtime._build_categorized_strategies``:
that path is shared with the backtest/validator harness itself
(``scripts/validate_pattern_cnn_walkforward.py`` drives real backtests with
``cnn_scoring_enabled: True`` to produce the very record this gate reads) —
guarding it there would make the harness unable to produce a passing record
in the first place. Guard at the live-engine boundary instead (see
``firm.live.engine.LiveTradingEngine.__init__`` /
``_rebuild_orchestrator``), which is the true funnel for every path that can
start or hot-swap *live* trading and covers backtests correctly by not
existing on that path at all.

Record path is repo-level (not ``FIRM_DATA_DIR``-relative), matching the
existing convention that pattern-ML model artifacts
(``firm.patterns.ml.inference.DEFAULT_MODEL_PATH`` et al.) are a shared
code-level asset identical across both live instances, not per-instance live
state — the record that validates those shared files belongs next to them.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

GATE_PATH_ENV = "FIRM_PATTERN_ML_GATE"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GATE_PATH = _PROJECT_ROOT / "data" / "models" / "pattern_recognition.rollout_gate.json"

#: strategy_params.pattern_recognition keys this gate governs, and the
#: rollback value each is forced to when the gate refuses.
_GATED_FLAGS: dict[str, bool] = {
    "cnn_scoring_enabled": False,
    "xgb_confirmation_enabled": False,
}

_REMEDIATION = (
    "run scripts/validate_pattern_cnn_walkforward.py then "
    "scripts/pattern_recognition_rollout_gate.py "
    "(writing --output to {path}) before enabling this flag"
)


def gate_path() -> Path:
    return Path(os.environ.get(GATE_PATH_ENV, str(DEFAULT_GATE_PATH)))


def _load_record(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("pattern_ml_gate: could not read gate record at %s (%s) -- treating as absent", path, exc)
        return None


def evaluate_gate(record: dict[str, Any] | None) -> dict[str, Any]:
    """Pure: given a loaded rollout-gate result JSON (or ``None`` if absent/
    unreadable), decide whether the gated ML flags may be honored.

    ``record`` is ``scripts/pattern_recognition_rollout_gate.py``'s own
    output shape — reused as-is rather than inventing a new schema. Its
    ``overall_recommendation`` is not a clean ``"KEEP"|"HOLD"|"ROLLBACK"``
    enum (e.g. it can be ``"HOLD (insufficient live sample, backtest
    verdict=fail)"``), so only the leading token is checked.
    """
    if record is None:
        return {"allowed": False, "reason": "no rollout-gate record found"}
    recommendation = str(record.get("overall_recommendation", ""))
    verdict = recommendation.split(" ", 1)[0].strip().upper()
    if verdict == "KEEP":
        return {"allowed": True, "reason": f"rollout gate recommendation={recommendation!r}"}
    return {"allowed": False, "reason": f"rollout gate recommendation={recommendation!r} (require KEEP)"}


def sanitize_strategy_params(
    strategy_params: dict[str, Any] | None,
    *,
    gate_path_override: str | Path | None = None,
) -> dict[str, Any]:
    """Return a NEW ``strategy_params`` dict with any gated
    ``pattern_recognition`` ML flag forced back to its safe default unless a
    passing rollout-gate record backs it up.

    Never mutates the input (the caller may hand this the same dict object
    other code still holds a reference to — see
    ``firm.live.provider_utils.resolve_live_startup``'s own aliasing note for
    why that matters). Never raises: an unreadable/missing gate record is
    treated as "not passing," not as an error.
    """
    strategy_params = dict(strategy_params or {})
    pr_params = dict(strategy_params.get("pattern_recognition") or {})

    requested = {flag: pr_params.get(flag) for flag in _GATED_FLAGS if pr_params.get(flag)}
    if not requested:
        return strategy_params

    path = Path(gate_path_override) if gate_path_override is not None else gate_path()
    decision = evaluate_gate(_load_record(path))

    if decision["allowed"]:
        log.info(
            "pattern_ml_gate: honoring %s (gate passed: %s)",
            sorted(requested), decision["reason"],
        )
        return strategy_params

    for flag in requested:
        pr_params[flag] = _GATED_FLAGS[flag]
    strategy_params["pattern_recognition"] = pr_params
    log.error(
        "pattern_ml_gate: REFUSED %s -- %s at %s -- forcing back to %s. %s.",
        sorted(requested), decision["reason"], path,
        {flag: _GATED_FLAGS[flag] for flag in requested},
        _REMEDIATION.format(path=path),
    )
    return strategy_params
