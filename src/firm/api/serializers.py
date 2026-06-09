"""Convert frozen dataclasses and Blackboard to JSON-safe dicts.

Handles NaN -> None, datetime -> isoformat, and nested structures.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from firm.agents.blackboard import Blackboard
from firm.contracts.models import (
    DebateResult,
    ExecutionReport,
    RiskDecision,
    Signal,
    SignalSet,
    Thesis,
    TradeProposal,
)


def safe_value(v: Any) -> Any:
    """Convert NaN/Inf floats to None and datetimes to ISO strings."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    return v


def serialize_dict(d: dict) -> dict:
    """Recursively sanitise a dict for JSON serialization."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = serialize_dict(v)
        elif isinstance(v, (list, tuple)):
            out[k] = [safe_value(i) for i in v]
        else:
            out[k] = safe_value(v)
    return out


def serialize_signal(sig: Signal) -> dict:
    return {
        "symbol": sig.symbol,
        "strategy": sig.strategy,
        "score": safe_value(sig.score),
        "confidence": safe_value(sig.confidence),
        "horizon": sig.horizon,
        "asof": sig.asof.isoformat(),
        "meta": serialize_dict(sig.meta),
    }


def serialize_signal_set(ss: SignalSet) -> dict:
    return {
        "domain": ss.domain,
        "asof": ss.asof.isoformat(),
        "signals": [serialize_signal(s) for s in ss.signals],
    }


def serialize_thesis(t: Thesis) -> dict:
    return {
        "side": t.side,
        "symbol": t.symbol,
        "conviction": safe_value(t.conviction),
        "rationale": t.rationale,
        "supporting": list(t.supporting),
    }


def serialize_debate_result(dr: DebateResult) -> dict:
    return {
        "symbol": dr.symbol,
        "net_conviction": safe_value(dr.net_conviction),
        "bull_thesis": serialize_thesis(dr.bull_thesis) if dr.bull_thesis else None,
        "bear_thesis": serialize_thesis(dr.bear_thesis) if dr.bear_thesis else None,
    }


def serialize_proposal(p: TradeProposal) -> dict:
    return {
        "asof": p.asof.isoformat(),
        "targets": serialize_dict(p.targets),
        "per_strategy": serialize_dict(p.per_strategy),
        "notes": p.notes,
    }


def serialize_risk_decision(rd: RiskDecision) -> dict:
    return {
        "approved": rd.approved,
        "adjusted_targets": serialize_dict(rd.adjusted_targets),
        "violations": list(rd.violations),
        "actions": list(rd.actions),
    }


def serialize_execution_report(er: ExecutionReport) -> dict:
    return {
        "fills": [serialize_dict(f) for f in er.fills],
        "turnover": safe_value(er.turnover),
        "costs": safe_value(er.costs),
    }


def serialize_blackboard(bb: Blackboard) -> dict:
    """Serialize the entire blackboard pipeline to a JSON-safe dict."""
    result: dict[str, Any] = {"asof": bb.asof.isoformat()}

    result["signal_sets"] = [serialize_signal_set(ss) for ss in bb.signal_sets]
    result["theses"] = [serialize_thesis(t) for t in bb.theses]
    result["debate_results"] = [serialize_debate_result(dr) for dr in bb.debate_results]

    result["proposal"] = serialize_proposal(bb.proposal) if bb.proposal else None
    result["risk_decision"] = (
        serialize_risk_decision(bb.risk_decision) if bb.risk_decision else None
    )
    result["execution_report"] = (
        serialize_execution_report(bb.execution_report) if bb.execution_report else None
    )

    return result
