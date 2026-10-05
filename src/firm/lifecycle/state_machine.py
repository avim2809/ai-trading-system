"""Strategy-family lifecycle state machine (ticket P5-01).

Chain: IDEA -> SPEC -> PREREG -> RESEARCH -> RESEARCH_PASS -> EMBARGO -> PAPER -> LIVE_STEP_1 -> LIVE_STEP_2 -> LIVE_FULL.
Any non-terminal state may move to PROBATION, DECOMMISSIONED or ARCHIVED; PROBATION returns only to its recorded prior state.

``transition`` is pure: it returns a new ``Registry`` and the caller persists it (``save_registry``). Forward edges need
(a) an evidence file satisfying that edge's gate (``GateNotMetError``), (b) an owner approval file under the registry's
approvals directory (``ApprovalRequiredError``), (c) a prereg hash equal to the family's recorded ``config_hash``. Agents never
create approval files; approvals are owner commits. Moving to PROBATION is the safety direction and needs no approval;
DECOMMISSIONED/ARCHIVED and leaving PROBATION do. The approval file is ``<family>__<SRC>__<DST>.yaml`` with keys ``family``,
``from_state``, ``to_state``, ``approver``, ``approved_at_utc`` and ``signature_ref``.

Approval identity (OD-06, decided (a): advisory CODEOWNERS only): ``identity_status`` reports ``unverifiable``, never ``pass``,
unless a distinct agent identity is configured (OD-06 option b). An unverifiable approval is accepted only with a non-empty
``signature_ref`` (signed commit or merged-PR reference) and is recorded as advisory; agent-named approvers are rejected.

Evidence file (YAML) keys by edge: all forward edges ``family``, ``edge`` ("SRC->DST"); SPEC->PREREG, PREREG->RESEARCH,
RESEARCH->RESEARCH_PASS, RESEARCH_PASS->EMBARGO ``prereg_hash`` (must equal the recorded config_hash); RESEARCH->RESEARCH_PASS
also ``tier`` (paper-eligible per gates.yaml) and non-empty ``trial_ids``; ->EMBARGO also ``code_commit`` and ``config_hash``
(frozen at entry); EMBARGO->PAPER the current ``code_commit`` / ``config_hash`` (must equal the frozen ones) and
``embargo_weeks_before_start`` weeks elapsed; PAPER->LIVE_STEP_1 ``gpaper_overall: pass``; LIVE_STEP_1->2 and 2->FULL
``min_months_per_step`` months of dwell and ``live_step_criteria_hold: true``. Only ``kind == candidate`` families can be
promoted (the Alpaca 92/8 allocation book, the S2 shadow ledger and legacy pipeline strategies have no promotion path).
Timestamps come from the injected ``now_fn`` (callers use git or ``date -u``), never from log lines.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml

from firm.lifecycle.gates import GatesConfig, add_months

log = logging.getLogger(__name__)

__all__ = [
    "ApprovalRequiredError",
    "FamilyRecord",
    "GateNotMetError",
    "Registry",
    "State",
    "Transition",
    "TransitionError",
    "allowed",
    "identity_status",
    "load_registry",
    "save_registry",
    "target_weight_allowed",
    "transition",
]


class TransitionError(Exception):
    pass


class GateNotMetError(TransitionError):
    pass


class ApprovalRequiredError(TransitionError):
    pass


class State(StrEnum):
    IDEA = "IDEA"
    SPEC = "SPEC"
    PREREG = "PREREG"
    RESEARCH = "RESEARCH"
    RESEARCH_PASS = "RESEARCH_PASS"
    EMBARGO = "EMBARGO"
    PAPER = "PAPER"
    LIVE_STEP_1 = "LIVE_STEP_1"
    LIVE_STEP_2 = "LIVE_STEP_2"
    LIVE_FULL = "LIVE_FULL"
    PROBATION = "PROBATION"
    DECOMMISSIONED = "DECOMMISSIONED"
    ARCHIVED = "ARCHIVED"


_CHAIN = (
    State.IDEA,
    State.SPEC,
    State.PREREG,
    State.RESEARCH,
    State.RESEARCH_PASS,
    State.EMBARGO,
    State.PAPER,
    State.LIVE_STEP_1,
    State.LIVE_STEP_2,
    State.LIVE_FULL,
)
_SINKS = (State.PROBATION, State.DECOMMISSIONED, State.ARCHIVED)
_LIVE_STATES = (State.LIVE_STEP_1, State.LIVE_STEP_2, State.LIVE_FULL)
AGENT_NAME_MARKERS = ("claude", "agent", "research", "bot", "gpt", "copilot")


@dataclass(frozen=True)
class FamilyRecord:
    family: str
    state: State
    prior_state: State | None  # set only while PROBATION
    entered_at_utc: datetime
    code_commit: str
    config_hash: str
    kind: str  # candidate | forward_test | shadow_ledger | legacy_pipeline


@dataclass(frozen=True)
class Transition:
    family: str
    src: State
    dst: State
    evidence: tuple[str, ...]
    approval_id: str | None


@dataclass(frozen=True)
class Registry:
    families: tuple[FamilyRecord, ...]
    approvals_dir: Path
    history: tuple[dict, ...] = ()

    def get(self, family: str) -> FamilyRecord:
        for r in self.families:
            if r.family == family:
                return r
        raise KeyError(f"unknown family {family!r}")

    def replace_record(self, rec: FamilyRecord) -> Registry:
        return replace(
            self, families=tuple(rec if r.family == rec.family else r for r in self.families)
        )


def allowed(src: State, dst: State) -> bool:
    """Structural legality only (gates and approvals are checked by ``transition``)."""
    if src in (State.ARCHIVED, dst):
        return False
    if src is State.DECOMMISSIONED:
        return dst is State.ARCHIVED
    if dst in _SINKS:
        return True
    if src is State.PROBATION:
        return dst in _CHAIN  # which prior state is checked against the record
    if src in _CHAIN and dst in _CHAIN:
        return _CHAIN.index(dst) == _CHAIN.index(src) + 1
    return False


def target_weight_allowed(
    rec: FamilyRecord, *, account_type: Literal["paper", "live"], zero_weight: bool
) -> bool:
    if zero_weight or rec.state in (State.DECOMMISSIONED, State.ARCHIVED):
        return False
    if rec.state is State.PROBATION:
        if rec.prior_state is None:
            return False
        return target_weight_allowed(
            replace(rec, state=rec.prior_state, prior_state=None),
            account_type=account_type,
            zero_weight=False,
        )
    if rec.state is State.PAPER:
        return account_type == "paper"
    return rec.state in _LIVE_STATES


def identity_status(
    approver: str,
    signature_ref: str | None,
    *,
    distinct_agent_identity_configured: bool = False,
) -> Literal["pass", "unverifiable", "rejected"]:
    name = (approver or "").strip().lower()
    if not name or any(m in name for m in AGENT_NAME_MARKERS):
        return "rejected"
    if distinct_agent_identity_configured and signature_ref:
        return "pass"
    return "unverifiable"


# ---- registry I/O --------------------------------------------------------------------------------------------------
def _rec_from(d: dict) -> FamilyRecord:
    return FamilyRecord(
        family=d["family"],
        state=State(d["state"]),
        prior_state=State(d["prior_state"]) if d.get("prior_state") else None,
        entered_at_utc=datetime.fromisoformat(str(d["entered_at_utc"])),
        code_commit=str(d["code_commit"]),
        config_hash=str(d["config_hash"]),
        kind=str(d["kind"]),
    )


def _rec_to(r: FamilyRecord) -> dict:
    return {
        "family": r.family,
        "state": r.state.value,
        "prior_state": r.prior_state.value if r.prior_state else None,
        "entered_at_utc": r.entered_at_utc.isoformat(),
        "code_commit": r.code_commit,
        "config_hash": r.config_hash,
        "kind": r.kind,
    }


def load_registry(path: Path) -> Registry:
    path = Path(path)
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    adir = Path(doc.get("approvals_dir", "research/approvals"))
    if not adir.is_absolute():
        adir = path.resolve().parents[2] / adir  # <repo>/research/lifecycle/registry.yaml -> <repo>
    return Registry(
        families=tuple(_rec_from(d) for d in doc.get("families", [])),
        approvals_dir=adir,
        history=tuple(doc.get("history", [])),
    )


def save_registry(reg: Registry, path: Path) -> None:
    path = Path(path)
    doc = {
        "schema": 1,
        "approvals_dir": str(reg.approvals_dir),
        "families": [_rec_to(r) for r in reg.families],
        "history": list(reg.history),
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# ---- gates per edge ------------------------------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        with open(path) as fh:
            doc = yaml.safe_load(fh)
    except OSError as exc:
        raise GateNotMetError(f"evidence file unreadable: {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise GateNotMetError(f"evidence file is not a mapping: {path}")
    return doc


def _check_evidence(
    rec: FamilyRecord, dst: State, ev: dict, gates: GatesConfig, now: datetime
) -> None:
    src = rec.state
    edge = f"{src.value}->{dst.value}"
    if ev.get("family") != rec.family or ev.get("edge") != edge:
        raise GateNotMetError(f"evidence does not match {rec.family} {edge}")
    prereg_edge = src in (State.SPEC, State.PREREG, State.RESEARCH, State.RESEARCH_PASS)
    if prereg_edge and (not ev.get("prereg_hash") or str(ev["prereg_hash"]) != rec.config_hash):
        raise GateNotMetError("prereg hash does not match the family's recorded config_hash")
    if src is State.RESEARCH:
        if ev.get("tier") not in gates.paper_eligible_tiers:
            raise GateNotMetError(f"verdict tier {ev.get('tier')!r} is not paper-eligible")
        if not ev.get("trial_ids"):
            raise GateNotMetError("RESEARCH_PASS needs verified trial_ids")
    if dst is State.EMBARGO and not (ev.get("code_commit") and ev.get("config_hash")):
        raise GateNotMetError("EMBARGO entry needs code_commit and config_hash to freeze")
    if src is State.EMBARGO:
        weeks = int(gates.get("g_paper", "embargo_weeks_before_start"))
        if now - rec.entered_at_utc < timedelta(weeks=weeks):
            raise GateNotMetError(f"EMBARGO shorter than {weeks} weeks")
        if ev.get("code_commit") != rec.code_commit or ev.get("config_hash") != rec.config_hash:
            raise GateNotMetError("code commit or config hash changed since the EMBARGO transition")
    if src is State.PAPER and ev.get("gpaper_overall") != "pass":
        raise GateNotMetError("G-PAPER overall verdict is not pass")
    if src in (State.LIVE_STEP_1, State.LIVE_STEP_2):
        months = int(gates.get("g_live_step", "min_months_per_step"))
        if now.date() < add_months(rec.entered_at_utc.date(), months):
            raise GateNotMetError(f"dwell at {src.value} is under {months} months")
        if ev.get("live_step_criteria_hold") is not True:
            raise GateNotMetError("G-LIVE-STEP criteria do not hold")


def _check_approval(
    reg: Registry, rec: FamilyRecord, dst: State, approver: str | None
) -> tuple[str, str]:
    name = f"{rec.family}__{rec.state.value}__{dst.value}"
    path = reg.approvals_dir / f"{name}.yaml"
    if not approver:
        raise ApprovalRequiredError(f"approver required for {name}")
    if not path.exists():
        raise ApprovalRequiredError(f"no approval file {path}")
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    if (
        doc.get("family") != rec.family
        or doc.get("from_state") != rec.state.value
        or doc.get("to_state") != dst.value
        or doc.get("approver") != approver
    ):
        raise ApprovalRequiredError(f"approval file {path} does not match the transition/approver")
    sig = doc.get("signature_ref")
    status = identity_status(approver, sig)
    if status == "rejected":
        raise ApprovalRequiredError(f"approver {approver!r} is not an acceptable owner identity")
    if status == "unverifiable":
        if not sig:
            raise ApprovalRequiredError(
                "approval identity is unverifiable and carries no signature_ref"
            )
        log.warning("approval %s accepted as ADVISORY (identity unverifiable, OD-06 option a)", name)
    return name, status


def transition(
    strategy_id: str,
    to_state: State,
    evidence_path: Path | None,
    approver: str | None,
    *,
    registry: Registry,
    gates: GatesConfig,
    now_fn: Callable[[], datetime],
) -> Registry:
    rec = registry.get(strategy_id)
    src, dst = rec.state, State(to_state)
    now = now_fn()
    if not allowed(src, dst):
        raise TransitionError(f"illegal transition {src.value} -> {dst.value} for {strategy_id}")
    if src is State.PROBATION and dst in _CHAIN and dst != rec.prior_state:
        raise TransitionError(f"PROBATION may return only to {rec.prior_state}, not {dst.value}")
    forward = src in _CHAIN and dst in _CHAIN
    if forward and rec.kind != "candidate":
        raise TransitionError(f"{strategy_id} is kind={rec.kind}; it has no promotion path")
    ev: dict[str, Any] = {}
    evidence: tuple[str, ...] = ()
    if forward:
        if evidence_path is None:
            raise GateNotMetError("evidence file required")
        ev = _load_yaml(Path(evidence_path))
        _check_evidence(rec, dst, ev, gates, now)
        evidence = (str(evidence_path),)
    approval_id, id_status = None, None
    if forward or dst in (State.DECOMMISSIONED, State.ARCHIVED) or src is State.PROBATION:
        approval_id, id_status = _check_approval(registry, rec, dst, approver)

    new = replace(
        rec, state=dst, entered_at_utc=now, prior_state=src if dst is State.PROBATION else None
    )
    if dst is State.EMBARGO:
        new = replace(new, code_commit=str(ev["code_commit"]), config_hash=str(ev["config_hash"]))
    entry = {
        "family": strategy_id,
        "src": src.value,
        "dst": dst.value,
        "at_utc": now.isoformat(),
        "evidence": list(evidence),
        "approval_id": approval_id,
        "identity_status": id_status,
    }
    log.info(
        "lifecycle %s: %s -> %s (approval=%s identity=%s)",
        strategy_id,
        src.value,
        dst.value,
        approval_id,
        id_status,
    )
    out = registry.replace_record(new)
    return replace(out, history=(*out.history, entry))
