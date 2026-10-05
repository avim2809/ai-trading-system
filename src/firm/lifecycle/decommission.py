"""G-DECOMMISSION combiner and the persisted zero-weight override store (ticket P5-01).

``n = int(dd > m * survival_ref) + int(cusum_alarm) + int(fidelity_breach_months >= 2) + int(mechanism_invalid)``;
n == 0 -> ok, n == 1 -> probation (review; target weight unchanged), n >= 2 -> zero_weight (target weight 0 until an owner
approval file clears it). ``m`` is ``g_decommission.drawdown_multiple_of_survival_ref`` from gates.yaml; ``dd`` must be the
IMMUTABLE drawdown from ``firm.risk.kill_switch`` (``Decision.dd_immutable``), never the resettable engine peak. The survival
reference is ``max(bootstrap p95, 2.5 tau)`` (``KillSwitchConfig.survival_ref``).

The override store is an atomically written JSON file under ``$FIRM_DATA_DIR`` (outside the repo): the services restart
automatically, so an in-memory zero-weight decision would be lost and trading would resume silently.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml

from firm.lifecycle.gates import GatesConfig
from firm.lifecycle.state_machine import Registry, State, transition

log = logging.getLogger(__name__)

__all__ = [
    "DecommissionInputs",
    "apply_decommission",
    "approval_clears_zero_weight",
    "decommission_decision",
    "read_overrides",
    "write_zero_weight",
    "zero_weight_active",
]

FIDELITY_BREACH_CONSECUTIVE_MONTHS = 2  # trigger name: fidelity_breach_two_consecutive_months


@dataclass(frozen=True)
class DecommissionInputs:
    dd: float
    survival_ref: float
    cusum_alarm: bool
    fidelity_breach_months: int
    mechanism_invalid: bool


def decommission_decision(
    i: DecommissionInputs, gates: GatesConfig
) -> Literal["ok", "probation", "zero_weight"]:
    mult = float(gates.get("g_decommission", "drawdown_multiple_of_survival_ref"))
    n = (
        int(i.dd > mult * i.survival_ref)
        + int(bool(i.cusum_alarm))
        + int(i.fidelity_breach_months >= FIDELITY_BREACH_CONSECUTIVE_MONTHS)
        + int(bool(i.mechanism_invalid))
    )
    verdict = "ok" if n == 0 else ("probation" if n == 1 else "zero_weight")
    if n:
        log.warning("decommission combiner: %d trigger(s) -> %s (inputs=%s)", n, verdict, i)
    return verdict  # type: ignore[return-value]


def read_overrides(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with open(path) as fh:
        data = json.load(fh)
    return {r["family"]: r for r in data.get("overrides", [])}


def write_zero_weight(
    path: Path, family: str, triggers: list[str], now_fn: Callable[[], datetime]
) -> dict:
    """Persist a zero-weight decision (write temp + os.replace). Entries for other families are kept."""
    current = read_overrides(path)
    rec = {
        "family": family,
        "zero_weight": True,
        "triggers": list(triggers),
        "decided_at_utc": now_fn().isoformat(),
    }
    current[family] = rec
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump({"overrides": list(current.values())}, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    log.warning("zero-weight override persisted for %s (%s)", family, triggers)
    return rec


def approval_clears_zero_weight(approvals_dir: Path, family: str, decided_at_utc: str) -> bool:
    """True when an owner approval file ``<family>__clear_zero_weight.yaml`` is dated after the override decision."""
    p = approvals_dir / f"{family}__clear_zero_weight.yaml"
    if not p.exists():
        return False
    with open(p) as fh:
        doc = yaml.safe_load(fh) or {}
    if doc.get("family") != family or not doc.get("approver") or not doc.get("clears_zero_weight"):
        return False
    try:
        return datetime.fromisoformat(str(doc["approved_at_utc"])) > datetime.fromisoformat(
            decided_at_utc
        )
    except (KeyError, ValueError):
        return False


def zero_weight_active(overrides_path: Path, approvals_dir: Path, family: str) -> bool:
    rec = read_overrides(overrides_path).get(family)
    if not rec or not rec.get("zero_weight"):
        return False
    return not approval_clears_zero_weight(approvals_dir, family, rec["decided_at_utc"])


def apply_decommission(
    registry: Registry,
    overrides_path: Path,
    family: str,
    inputs: DecommissionInputs,
    gates: GatesConfig,
    now_fn: Callable[[], datetime],
) -> tuple[Literal["ok", "probation", "zero_weight"], Registry]:
    """Run the combiner and apply it: one trigger -> PROBATION (weight unchanged), two -> PROBATION plus persisted zero weight."""
    verdict = decommission_decision(inputs, gates)
    if verdict == "ok":
        return verdict, registry
    rec = registry.get(family)
    if verdict == "zero_weight":
        mult = float(gates.get("g_decommission", "drawdown_multiple_of_survival_ref"))
        triggers = [
            name
            for name, hit in (
                ("drawdown_above_1_5x_survival_ref", inputs.dd > mult * inputs.survival_ref),
                ("cusum_alarm", inputs.cusum_alarm),
                (
                    "fidelity_breach_two_consecutive_months",
                    inputs.fidelity_breach_months >= FIDELITY_BREACH_CONSECUTIVE_MONTHS,
                ),
                ("mechanism_invalidated_owner_decision", inputs.mechanism_invalid),
            )
            if hit
        ]
        write_zero_weight(overrides_path, family, triggers, now_fn)
    if rec.state in (State.PROBATION, State.DECOMMISSIONED, State.ARCHIVED):
        return verdict, registry
    return verdict, transition(
        family, State.PROBATION, None, None, registry=registry, gates=gates, now_fn=now_fn
    )
