"""Fidelity monitor for candidates (P5-04): live paper results versus a shadow replay by the same engine.

Tracks daily-return correlation and tracking error (G-PAPER 2), implementation shortfall against the cost model
(G-PAPER 3), position breaks and missed reviews (G-PAPER 4). Thresholds come from ``config/gates.yaml``; the allocation
forward test keeps its own frozen I1-I5 bars (``firm.monitoring.allocation_forward``) and is never merged with these.

Alerts only: nothing here places orders, halts anything or writes outside the state dir. Never imported by live code.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from firm.costs.model import CostBreakdown, load_cost_config, spec_from_config
from firm.costs.model import cost as model_cost

_ROOT = Path(__file__).resolve().parents[3]
_GATES_PATH = _ROOT / "config" / "gates.yaml"
_STATE_ROOT = _ROOT / "data" / ("forward" + "_monitors")
_ZERO_EPS = 1e-12
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GatesConfig:
    corr_min: float = 0.95
    te_max_fraction_of_tau: float = 0.25
    window_trading_days: int = 63
    zero_exposure_rule: str = "exclude_from_corr_and_tracking_error_and_report_count"
    cost_max_multiple: float = 1.5
    cost_min_fills: int = 30
    cost_min_instruments: int = 10
    max_unreconciled_trading_days: int = 1


def load_gates_config(path: Path | None = None) -> GatesConfig:
    """Read the G-PAPER fidelity block of ``config/gates.yaml``; every value must be present (no silent defaults)."""
    g = yaml.safe_load(Path(path or _GATES_PATH).read_text())["g_paper"]
    fid = g["fidelity"]
    return GatesConfig(
        corr_min=float(fid["daily_return_corr_min"]),
        te_max_fraction_of_tau=float(fid["tracking_error_max_fraction_of_tau"]),
        window_trading_days=int(fid["window_trading_days"]),
        zero_exposure_rule=str(fid["zero_exposure_day_rule"]),
        cost_max_multiple=float(g["realised_cost_max_multiple_of_modelled"]),
        cost_min_fills=int(g["cost_test_min_fills"]),
        cost_min_instruments=int(g["cost_test_min_instruments"]),
        max_unreconciled_trading_days=int(g["position_breaks"]["max_unreconciled_trading_days"]),
    )


def assert_deterministic(candidate_cfg: Mapping[str, Any]) -> None:
    """Only deterministic engines can be shadow-replayed. Rejects IBKR and anything with an enabled LLM component."""
    if candidate_cfg.get("deterministic") is not True:
        raise ValueError("candidate config must declare deterministic: true to be shadow-replayed")
    if str(candidate_cfg.get("broker", "")).lower() == "ibkr":
        raise ValueError("IBKR candidates are not deterministic-replayable (LLM arm B)")

    def has_llm(o: Any) -> bool:
        if isinstance(o, Mapping):
            return any((("llm" in str(k).lower()) and bool(v)) or has_llm(v) for k, v in o.items())
        if isinstance(o, (list, tuple)):
            return any(has_llm(x) for x in o)
        return False

    if has_llm(candidate_cfg):
        raise ValueError("candidate config enables an LLM component; not replayable")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FidelityDay:
    date: date
    live_ret: float
    shadow_ret: float


@dataclass(frozen=True)
class Fill:
    order_id: str
    symbol: str
    side: int                # +1 buy, -1 sell
    qty: float               # absolute units
    fill_px: float
    commission: float
    exchange_fees: float = 0.0


@dataclass(frozen=True)
class PositionBreak:
    key: str                 # "<type>:<symbol>"
    opened_at: datetime
    last_seen: datetime
    closed_at: datetime | None = None
    detail: str = ""
    trading_days_open: int = 0
    breach: bool = False


@dataclass(frozen=True)
class ReviewEvent:
    date: date
    decided: bool
    explanation: str | None = None


@dataclass(frozen=True)
class MissedReview:
    date: date
    reason: str              # "no_decision" | "blocked_by_working_order"


@dataclass(frozen=True)
class FidelityReport:
    corr: float | None
    te_annualised: float | None
    te_vs_tau: float | None
    cost_ratio_median: float | None
    cost_ratio_aggregate: float | None
    n_fills: int
    n_instruments: int
    g_paper_3_evaluable: bool
    position_breaks: tuple[PositionBreak, ...] = ()
    missed_reviews: tuple[MissedReview, ...] = ()
    n_days: int = 0
    n_zero_exposure_excluded: int = 0
    corr_pass: bool | None = None
    te_pass: bool | None = None
    cost_pass: bool | None = None
    g_paper_3_status: str = "insufficient_data"
    window_complete: bool = False   # at least ``window_trading_days`` days observed

    @property
    def breach(self) -> bool:
        """G-PAPER 2 or 3 breach (feeds the P5-01 'two consecutive months' trigger). Unevaluable is not a breach.

        corr/TE count only once the full pre-registered window is observed: on a partial (cumulative) window the sample
        corr is too noisy (true corr 0.97 falls below 0.95 in roughly 1 draw in 10 at 21 days, test-measured), so earlier
        values are informational.
        """
        fid = (self.corr_pass, self.te_pass) if self.window_complete else ()
        return any(x is False for x in (*fid, self.cost_pass))


# ---------------------------------------------------------------------------
# Shadow replay
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _private_ledger(root: Path) -> Iterator[None]:
    """Point the trial ledger at a monitor-private directory so daily shadow runs never enter the research ledger."""
    root = Path(root)
    for sub in ("", "returns", "inbox"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    from firm.research import ledger as L  # lazy: monitoring-side import only
    saved = {k: os.environ.get(k) for k in (L.LEDGER_ROOT_ENV, L.ALLOW_DIRTY_ENV)}
    os.environ[L.LEDGER_ROOT_ENV] = str(root)
    os.environ[L.ALLOW_DIRTY_ENV] = "1"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _cost_fn(candidate_cfg: Mapping[str, Any]):
    cfg = load_cost_config()
    specs_by_symbol = candidate_cfg.get("cost_specs") or {}

    def fn(symbol, date_, qty_delta, price, adv, vol_pct, multiplier, is_roll, stress):
        if adv is None or vol_pct is None:
            raise ValueError("shadow replay needs adv and vol_pct in the snapshot (no silent fallback)")
        spec = spec_from_config(specs_by_symbol[symbol], cfg)
        return model_cost(spec, qty_delta, price, adv, None, vol_pct, multiplier=stress, is_roll=is_roll, cfg=cfg)

    return fn


def shadow_returns(candidate_cfg: dict, data_snapshot_id: str, start: date, end: date, *,
                   snapshot_root: Path | None = None, ledger_root: Path | None = None) -> pd.Series:
    """Daily net returns of the candidate replayed by ``firm.backtest.vector_engine`` on its own snapshot.

    Uses the monitor-only ``shadow_loader`` (NOT ``data_access``). The engine runs ``exploratory=True`` against a private
    ledger root (default: a throw-away temp dir), so it neither fails closed on post-seal dates nor pollutes the trial ledger.
    """
    from firm.backtest.vector_engine import EngineConfig, run_vector_backtest
    from firm.monitoring import shadow_loader as SL

    assert_deterministic(candidate_cfg)
    snap = SL.load_snapshot(str(candidate_cfg["name"]), data_snapshot_id, snapshot_root)
    cfg = EngineConfig(initial_capital=snap.initial_capital)

    def run(root: Path) -> pd.Series:
        with _private_ledger(root):
            res = run_vector_backtest(snap.prices, snap.targets, snap.multipliers, _cost_fn(candidate_cfg), cfg,
                                      adv=snap.adv, vol_pct=snap.vol_pct, exploratory=True)
        return res.returns

    if ledger_root is not None:
        rets = run(Path(ledger_root))
    else:
        with tempfile.TemporaryDirectory(prefix="fidelity_shadow_") as td:
            rets = run(Path(td))
    idx = rets.index
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    return rets[mask]


# ---------------------------------------------------------------------------
# Implementation shortfall
# ---------------------------------------------------------------------------

_SF_COLS = ["order_id", "symbol", "notional", "slippage_bps", "fee_bps", "realised_bps", "modelled_bps", "ratio",
            "realised_cost", "modelled_cost"]


def implementation_shortfall(fills: Sequence[Fill], arrival_mids: Mapping[str, float],
                             modelled: Sequence[CostBreakdown]) -> pd.DataFrame:
    """Per-fill realised vs modelled cost. ``arrival_mids`` maps ``order_id`` to the mid at submission (never the prior close).

    realised_bps = side*(fill-arrival)/arrival*1e4 + (commission + exchange_fees)/notional*1e4
    modelled_bps = CostBreakdown.total/notional*1e4 (commission + fees + half-spread + impact [+ roll])
    ratio        = realised_bps / modelled_bps   (NaN where the model predicts zero cost)
    """
    if len(fills) != len(modelled):
        raise ValueError("fills and modelled must be parallel sequences")
    rows = []
    for f, m in zip(fills, modelled, strict=True):
        arrival = arrival_mids.get(f.order_id)
        if arrival is None or not arrival > 0:
            raise ValueError(f"missing arrival mid for order {f.order_id!r}")
        notional = abs(f.qty) * f.fill_px
        if notional <= 0:
            raise ValueError(f"non-positive notional for order {f.order_id!r}")
        slip_bps = f.side * (f.fill_px - arrival) / arrival * 1e4
        fee_bps = (f.commission + f.exchange_fees) / notional * 1e4
        realised = slip_bps + fee_bps
        modelled_bps = m.total / notional * 1e4
        rows.append({
            "order_id": f.order_id, "symbol": f.symbol, "notional": notional,
            "slippage_bps": slip_bps, "fee_bps": fee_bps, "realised_bps": realised,
            "modelled_bps": modelled_bps,
            "ratio": realised / modelled_bps if modelled_bps > 0 else float("nan"),
            "realised_cost": realised / 1e4 * notional, "modelled_cost": m.total,
        })
    return pd.DataFrame(rows, columns=_SF_COLS)


# ---------------------------------------------------------------------------
# Position breaks and missed reviews
# ---------------------------------------------------------------------------

def _break_key(d: Mapping[str, Any]) -> str:
    return f"{d.get('type', 'unknown')}:{d.get('symbol', '')}"


def _trading_days_between(a: datetime, b: datetime) -> int:
    return int(np.busday_count(a.date(), b.date()))


def record_position_breaks(discrepancies: Sequence[dict], prior: Sequence[PositionBreak], now: datetime,
                           max_days: int = 1) -> list[PositionBreak]:
    """Open a break when a discrepancy appears, close it when it clears. Breach if open for more than ``max_days`` trading days."""
    current = {_break_key(d): d for d in discrepancies}
    out: list[PositionBreak] = []
    seen: set[str] = set()
    for p in prior:
        if p.closed_at is not None:
            out.append(p)
            continue
        days = _trading_days_between(p.opened_at, now)
        if p.key in current:
            out.append(replace(p, last_seen=now, trading_days_open=days, breach=days > max_days))
            seen.add(p.key)
        else:
            out.append(replace(p, closed_at=now, trading_days_open=days, breach=p.breach or days > max_days))
    for key, d in current.items():
        if key not in seen:
            out.append(PositionBreak(key=key, opened_at=now, last_seen=now,
                                     detail=json.dumps(d, default=str, sort_keys=True)))
    return out


def record_missed_reviews(scheduled: Sequence[date], decisions: Sequence[ReviewEvent],
                          working_orders: Sequence[dict]) -> list[MissedReview]:
    """A scheduled review is missed unless a decision is recorded (traded or not) or a logged explanation exists.

    ``working_orders`` entries carry ``since`` and optional ``until`` dates; one overlapping the review date is reported as the
    blocking reason. A stuck order counts as missed unless the review event carries an explanation (G-PAPER 4).
    """
    by_date = {e.date: e for e in decisions}
    out: list[MissedReview] = []
    for d in scheduled:
        ev = by_date.get(d)
        if ev is not None and (ev.decided or ev.explanation):
            continue
        blocked = any(o["since"] <= d and (o.get("until") is None or d <= o["until"]) for o in working_orders)
        out.append(MissedReview(date=d, reason="blocked_by_working_order" if blocked else "no_decision"))
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _window_stats(days: Sequence[FidelityDay], gates: GatesConfig) -> tuple[float | None, float | None, int, int]:
    """(corr, annualised TE, n used, n zero-exposure excluded) over the trailing window."""
    win = list(days)[-gates.window_trading_days:]
    live = np.array([d.live_ret for d in win], float)
    shadow = np.array([d.shadow_ret for d in win], float)
    excl = gates.zero_exposure_rule.startswith("exclude")
    zero = (np.abs(live) < _ZERO_EPS) & (np.abs(shadow) < _ZERO_EPS)
    keep = ~zero if excl else np.ones(len(win), bool)
    live, shadow = live[keep], shadow[keep]
    n_excl = int(zero.sum()) if excl else 0
    if len(live) < 2:
        return None, None, len(live), n_excl
    te = float(np.std(live - shadow, ddof=1) * math.sqrt(252))
    if np.std(live) < _ZERO_EPS or np.std(shadow) < _ZERO_EPS:
        corr = 1.0 if np.allclose(live, shadow, atol=_ZERO_EPS) else None
    else:
        corr = float(np.corrcoef(live, shadow)[0, 1])
    return corr, te, len(live), n_excl


def evaluate_fidelity(days: Sequence[FidelityDay], fills: Sequence[Fill], modelled: Sequence[CostBreakdown], tau: float,
                      gates: GatesConfig, *, arrival_mids: Mapping[str, float] | None = None) -> FidelityReport:
    if not tau > 0:
        raise ValueError("tau must be positive")
    corr, te, n_used, n_excl = _window_stats(days, gates)
    te_vs_tau = None if te is None else te / tau
    corr_pass = None if corr is None else corr >= gates.corr_min
    te_pass = None if te is None else te <= gates.te_max_fraction_of_tau * tau

    n_fills = len(fills)
    n_instr = len({f.symbol for f in fills})
    evaluable = n_fills >= gates.cost_min_fills and n_instr >= gates.cost_min_instruments
    med = agg = None
    cost_pass: bool | None = None
    if evaluable:
        if arrival_mids is None:
            raise ValueError("arrival_mids required to evaluate G-PAPER 3")
        sf = implementation_shortfall(fills, arrival_mids, modelled)
        med = float(sf["ratio"].median())
        tot_model = float(sf["modelled_cost"].sum())
        agg = float(sf["realised_cost"].sum() / tot_model) if tot_model > 0 else None
        vals = [v for v in (med, agg) if v is not None and not math.isnan(v)]
        cost_pass = all(v <= gates.cost_max_multiple for v in vals) if vals else None
    return FidelityReport(
        corr=corr, te_annualised=te, te_vs_tau=te_vs_tau, cost_ratio_median=med, cost_ratio_aggregate=agg,
        n_fills=n_fills, n_instruments=n_instr, g_paper_3_evaluable=evaluable,
        n_days=n_used, n_zero_exposure_excluded=n_excl, corr_pass=corr_pass, te_pass=te_pass, cost_pass=cost_pass,
        g_paper_3_status="evaluated" if evaluable else "insufficient_data",
        window_complete=len(days) >= gates.window_trading_days)


def monthly_breaches(days: Sequence[FidelityDay], tau: float, gates: GatesConfig) -> list[tuple[str, bool]]:
    """At each month-end, evaluate corr/TE over the trailing pre-registered window (not a calendar month)."""
    ds = sorted(days, key=lambda d: d.date)
    out: list[tuple[str, bool]] = []
    for i, d in enumerate(ds):
        last = i + 1 == len(ds) or (ds[i + 1].date.year, ds[i + 1].date.month) != (d.date.year, d.date.month)
        if not last:
            continue
        r = evaluate_fidelity(ds[: i + 1], [], [], tau, gates)
        out.append((f"{d.date.year:04d}-{d.date.month:02d}", r.breach))
    return out


def consecutive_breach_months(flags: Sequence[tuple[str, bool]]) -> bool:
    """P5-01 trigger input: fidelity breach in 2 consecutive months."""
    return any(a[1] and b[1] for a, b in itertools.pairwise(flags))


# ---------------------------------------------------------------------------
# Daily job
# ---------------------------------------------------------------------------

def _guard_state_dir(state_dir: Path) -> Path:
    sd = Path(state_dir).resolve()
    allowed = _STATE_ROOT.resolve()
    try:
        sd.relative_to(_ROOT)
    except ValueError:
        return sd           # outside the repo (tmp dir): cannot dirty a tracked path
    try:
        sd.relative_to(allowed)
    except ValueError:
        raise ValueError(f"state_dir {sd} is inside the repo but not under the gitignored monitor state root") from None
    return sd


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str, sort_keys=True))
    os.replace(tmp, path)


def _load_breaks(path: Path) -> list[PositionBreak]:
    if not path.exists():
        return []
    out = []
    for r in json.loads(path.read_text()):
        r["opened_at"] = datetime.fromisoformat(r["opened_at"])
        r["last_seen"] = datetime.fromisoformat(r["last_seen"])
        r["closed_at"] = datetime.fromisoformat(r["closed_at"]) if r.get("closed_at") else None
        out.append(PositionBreak(**r))
    return out


def daily_job(candidate: str, state_dir: Path, *, fetch_live: Callable[[], dict], notify: Callable[..., None],
              gates: GatesConfig | None = None, now: datetime | None = None, snapshot_root: Path | None = None) -> dict:
    """One daily evaluation. Writes only under ``state_dir``; alerts only; never halts anything.

    ``fetch_live()`` (injected; the read-only broker/API side lives with the caller) returns::

        candidate_cfg, snapshot_id, tau, start, end, live_returns (Series), fills, arrival_mids, modelled,
        discrepancies, scheduled_reviews, review_events, working_orders
    """
    if not _SLUG.match(candidate):
        raise ValueError(f"invalid candidate name {candidate!r}")
    sd = _guard_state_dir(state_dir)
    gates = gates or load_gates_config()
    now = now or datetime.now(UTC)
    p = fetch_live()
    assert_deterministic(p["candidate_cfg"])

    shadow = shadow_returns(p["candidate_cfg"], p["snapshot_id"], p["start"], p["end"],
                            snapshot_root=snapshot_root, ledger_root=sd / "shadow_ledger")
    live = pd.Series(p["live_returns"])
    live.index = pd.DatetimeIndex(live.index)
    common = live.index.intersection(shadow.index)
    days = [FidelityDay(i.date(), float(live[i]), float(shadow[i])) for i in common]
    report = evaluate_fidelity(days, p.get("fills", []), p.get("modelled", []), float(p["tau"]), gates,
                               arrival_mids=p.get("arrival_mids"))

    breaks = record_position_breaks(p.get("discrepancies", []), _load_breaks(sd / "position_breaks.json"), now,
                                    gates.max_unreconciled_trading_days)
    _write_json(sd / "position_breaks.json", [asdict(b) for b in breaks])
    missed = record_missed_reviews(p.get("scheduled_reviews", []), p.get("review_events", []), p.get("working_orders", []))
    report = replace(report, position_breaks=tuple(b for b in breaks if b.closed_at is None or b.breach),
                     missed_reviews=tuple(missed))

    flags = monthly_breaches(days, float(p["tau"]), gates)
    result = {"candidate": candidate, "run_at": now.isoformat(), "report": asdict(report), "breach": report.breach,
              "monthly_breaches": flags, "two_consecutive_breach_months": consecutive_breach_months(flags),
              "n_common_days": len(days)}
    if report.breach:
        notify(severity="warning", kind="fidelity_breach", message=f"{candidate}: G-PAPER fidelity breach", candidate=candidate)
    if any(b.breach and b.closed_at is None for b in breaks):
        notify(severity="warning", kind="position_break_overdue",
               message=f"{candidate}: position break open > {gates.max_unreconciled_trading_days} trading day(s)",
               candidate=candidate)
    if missed:
        notify(severity="warning", kind="missed_review", message=f"{candidate}: {len(missed)} missed review(s)",
               candidate=candidate)
    _write_json(sd / "state.json", result)
    _write_json(sd / "daily" / f"{now.date().isoformat()}.json", result)
    return result


def export_report(candidate: str, state_dir: Path, out_root: Path | None = None) -> Path:
    """OWNER-RUN: copy the latest daily result into the candidate's sealed monitoring directory (``reports/<date>.json``).

    The only function here that writes outside the state dir, and only under ``shadow_loader.candidate_dir``; the timer
    never calls it (P5-06 tracked-path rule).
    """
    from firm.monitoring import shadow_loader as SL

    if not _SLUG.match(candidate):
        raise ValueError(f"invalid candidate name {candidate!r}")
    state = json.loads((Path(state_dir) / "state.json").read_text())
    dest = SL.candidate_dir(candidate, out_root) / "reports" / f"{str(state['run_at'])[:10]}.json"
    _write_json(dest, state)
    return dest
