"""Ledger-side hook for entry-point trial capture (ticket P1-12).

Importing this module ARMS capture: ``firm.backtest._capture_state.LEDGER_SINK`` is set to
:func:`record_unregistered`, so every outermost backtest entry-point call in this process is
recorded in the hash-chained ledger as ``mode="unregistered"`` (counted toward N, never
promotable). Only research harnesses and ``firm.research.seal`` import this; nothing on the
live import path does (``tests/test_live_import_isolation.py``).
"""

from __future__ import annotations

import logging
import uuid

import pandas as pd

from firm.backtest import _capture_state
from firm.research import ledger

log = logging.getLogger(__name__)

__all__ = ["record_inbox_line", "record_unregistered"]

PERIODS_PER_YEAR = 252


def _dates(config: dict) -> tuple[str | None, str | None]:
    bt = config.get("backtest") if isinstance(config.get("backtest"), dict) else {}
    start = config.get("start_date") or bt.get("start_date")
    end = config.get("end_date") or bt.get("end_date")
    return (str(start) if start else None, str(end) if end else None)


def _ledger_config(config: dict) -> dict:
    """Strict-canonical config; if not serialisable use the lossy form and store both hashes."""
    try:
        ledger.canonical_json(config)
        return config
    except (TypeError, ValueError):
        import json

        lossy = json.loads(json.dumps(config, sort_keys=True, default=repr))
        lossy["_capture"] = {
            "config_is_lossy": True,
            "api_config_hash": _capture_state.lossy_config_hash(config),
        }
        try:
            ledger.canonical_json(lossy)
        except ValueError:  # NaN / inf somewhere
            lossy = {"config_repr": repr(config)[:20000], "_capture": lossy["_capture"]}
        return lossy


def record_unregistered(
    entry: str, config: dict, *, seed: int | None, status: str, error: str | None,
    metrics: dict | None, returns: pd.Series | None,
) -> None:
    cfg = _ledger_config(config)
    gross = skew = kurt = None
    n_obs = None
    ret = None
    if returns is not None:
        ret = pd.Series(returns).dropna().astype(float)
        n_obs = int(len(ret))
        if n_obs > 2 and float(ret.std(ddof=1)) > 0:
            gross = float(ret.mean() / ret.std(ddof=1))  # per-period, NOT annualised
            skew = float(ret.skew())
            kurt = float(ret.kurt()) + 3.0  # ledger stores raw kurtosis
        if n_obs == 0:
            ret = None
    start, end = _dates(config)
    rec = ledger.TrialRecord(
        trial_id=uuid.uuid4().hex, family=f"unregistered:{entry}", mode="unregistered",
        config=cfg, config_hash=ledger.config_hash(cfg), code_commit="", data_snapshot_id=None,
        seed=seed, start=start, end=end, returns_path=None, gross_sharpe=gross, net_sharpe=None,
        periods_per_year=PERIODS_PER_YEAR if gross is not None else None,
        sharpe_conversion="per_period" if gross is not None else None,
        n_obs=n_obs, skew=skew, kurt=kurt, preregistration_id=None, touched_holdout=False,
        status="failed" if status == "failed" else "completed", error=error,
    )
    ledger.record_trial(rec, returns=ret)


def record_inbox_line(line: dict, *, source_file: str, index: int) -> str:
    """Ingest one stdlib inbox line (API / unarmed process) as an ``unregistered`` ledger row.

    Used by ``scripts/sync_ledger_mirror.py``; the ``(source_file, index)`` pair is the idempotency
    key. The inbox stream is lossy by design (config hash over ``default=repr`` JSON): both hashes
    are kept when they differ. The original code commit is unknown (the unarmed hook never runs
    git), so the row carries the ingesting checkout's state.
    """
    raw_cfg = line.get("config") if isinstance(line.get("config"), dict) else {"config_repr": repr(line.get("config"))}
    cfg = dict(_ledger_config(raw_cfg))
    cap = dict(cfg.get("_capture", {}))
    cap.update({"inbox_source": line.get("source"), "inbox_tag": line.get("tag"), "inbox_ts": line.get("ts"),
                "inbox_config_hash": line.get("config_hash"), "inbox_ingest": True})
    cfg["_capture"] = cap
    metrics = line.get("metrics") if isinstance(line.get("metrics"), dict) else {}
    n_obs = metrics.get("n_obs")
    seed = line.get("seed")
    rec = ledger.TrialRecord(
        trial_id=uuid.uuid4().hex, family=f"unregistered:{line.get('entry', 'unknown')}",
        mode="unregistered", config=cfg, config_hash=ledger.config_hash(cfg), code_commit="",
        data_snapshot_id=None, seed=seed if isinstance(seed, int) else None, start=_dates(raw_cfg)[0],
        end=_dates(raw_cfg)[1], returns_path=None, gross_sharpe=None, net_sharpe=None,
        periods_per_year=None, sharpe_conversion=None, n_obs=int(n_obs) if isinstance(n_obs, int) else None,
        skew=None, kurt=None, preregistration_id=None, touched_holdout=False,
        status="failed" if line.get("status") == "failed" else "completed", error=line.get("error"),
        source_file=source_file, source_entry_index=index,
    )
    return ledger.record_trial(rec)


_capture_state.LEDGER_SINK = record_unregistered  # arming: the last statement of the module
