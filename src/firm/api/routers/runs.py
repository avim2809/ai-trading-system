"""Run CRUD and backtest launch endpoints."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from firm.api.jobs import JobManager
from firm.api.schemas import (
    CompareRequest,
    RunDetail,
    RunRequest,
    RunSummary,
    WalkForwardRequest,
)
from firm.config import get_settings
from firm.experiments.registry import RunRegistry

log = logging.getLogger(__name__)

router = APIRouter()

_registry: RunRegistry | None = None
_job_manager: JobManager | None = None


def _get_registry() -> RunRegistry:
    global _registry
    if _registry is None:
        _registry = RunRegistry()
    return _registry


def _get_job_manager() -> JobManager:
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager(_get_registry())
    return _job_manager


def _run_to_summary(run) -> dict:
    return RunSummary(
        run_id=run.run_id,
        status=run.status,
        start_time=run.start_time.isoformat(),
        end_time=run.end_time.isoformat() if run.end_time else None,
        notes=run.notes,
        metrics=run.metrics,
    ).model_dump()


def _run_to_detail(run) -> dict:
    return RunDetail(
        run_id=run.run_id,
        status=run.status,
        start_time=run.start_time.isoformat(),
        end_time=run.end_time.isoformat() if run.end_time else None,
        notes=run.notes,
        metrics=run.metrics,
        config=run.config,
        config_hash=run.config_hash,
        seed=run.seed,
        artifacts_dir=run.artifacts_dir,
    ).model_dump()


@router.get("/runs")
def list_runs(status: str | None = None):
    registry = _get_registry()
    runs = registry.list_runs(status=status)
    return [_run_to_summary(r) for r in runs]


@router.delete("/runs")
def clear_runs():
    """Delete every backtest run and its artifacts.

    Used when past runs are known to be invalid — e.g. after fixing a
    backtest-affecting bug, old results no longer reflect real strategy
    behavior and shouldn't linger in the dashboard looking valid.
    """
    registry = _get_registry()
    count = registry.clear_all()
    return {"cleared": count}


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    registry = _get_registry()
    run = registry.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return _run_to_detail(run)


@router.get("/runs/{run_id}/report")
def get_report(run_id: str):
    registry = _get_registry()
    run = registry.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    report_path = Path(run.artifacts_dir) / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report not yet available")
    return json.loads(report_path.read_text(encoding="utf-8"))


@router.get("/runs/{run_id}/equity")
def get_equity(run_id: str):
    registry = _get_registry()
    run = registry.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    equity_path = Path(run.artifacts_dir) / "equity.json"
    if not equity_path.exists():
        raise HTTPException(status_code=404, detail="Equity data not yet available")
    return json.loads(equity_path.read_text(encoding="utf-8"))


@router.get("/runs/{run_id}/tearsheet")
def get_tearsheet(run_id: str):
    """Render (and cache) a QuantStats HTML tear-sheet for a run.

    Requires the optional ``report`` extra (``pip install -e '.[report]'``);
    returns 503 with an install hint when quantstats is unavailable.
    """
    from fastapi.responses import HTMLResponse

    registry = _get_registry()
    run = registry.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")

    artifacts = Path(run.artifacts_dir)
    html_path = artifacts / "tearsheet.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    equity_path = artifacts / "equity.json"
    if not equity_path.exists():
        raise HTTPException(status_code=404, detail="Equity data not yet available")

    import pandas as pd

    data = json.loads(equity_path.read_text(encoding="utf-8"))
    dates = data.get("dates", [])
    values = data.get("values", [])
    if len(values) < 2:
        raise HTTPException(status_code=422, detail="Not enough equity data to render")
    nav = pd.Series(values, index=pd.to_datetime(dates))
    returns = nav.pct_change().dropna()

    try:
        from firm.eval.tearsheet import render_tearsheet

        log.info("Rendering tear-sheet for run %s", run_id)
        render_tearsheet(returns, out_html=str(html_path), title=f"Run {run_id}")
    except ImportError as exc:
        log.warning("Tear-sheet unavailable for run %s: %s", run_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - rendering edge cases
        log.exception("Tear-sheet rendering failed for run %s", run_id)
        raise HTTPException(status_code=500, detail=f"Tear-sheet failed: {exc}") from exc

    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@router.post("/runs")
def launch_run(req: RunRequest):
    registry = _get_registry()
    jm = _get_job_manager()

    settings = get_settings()
    bt = settings.backtest.model_dump()
    risk = settings.risk.model_dump()

    config = {
        **bt,
        **risk,
        "start_date": req.start_date,
        "end_date": req.end_date,
        "initial_capital": req.initial_capital,
        "commission_pct": req.commission_pct,
        "slippage_pct": req.slippage_pct,
        "spread_pct": req.spread_pct,
        "short_borrow_annual_pct": req.short_borrow_annual_pct,
        "market_impact_coefficient": req.market_impact_coefficient,
        "market_impact_crossover_participation": req.market_impact_crossover_participation,
        "rebalance_frequency": req.rebalance_frequency,
        "strategies": req.strategies,
        "strategy_params": req.strategy_params,
        "allocation_method": req.allocation_method or settings.allocation_method,
        "kelly_fraction": (
            req.kelly_fraction if req.kelly_fraction is not None
            else settings.kelly_fraction
        ),
        "signal_combination": req.signal_combination or settings.signal_combination,
        "strategy_circuit_breaker": (
            req.strategy_circuit_breaker or settings.strategy_circuit_breaker
        ),
        "strategy_regime_weights": (
            req.strategy_regime_weights or settings.strategy_regime_weights
        ),
        "data_source": req.data_source,
        "seed": req.seed,
        **req.risk_overrides,
    }
    if req.regime_overlay is not None:
        # Merge over the settings default so partial overrides work.
        config["regime_overlay"] = {
            **config.get("regime_overlay", {}),
            **req.regime_overlay,
        }
    if req.strategy_circuit_breaker is not None:
        config["strategy_circuit_breaker"] = {
            **config.get("strategy_circuit_breaker", {}),
            **req.strategy_circuit_breaker,
        }
    if req.universe_symbols:
        config["universe_symbols"] = req.universe_symbols

    run = registry.create_run(config, seed=req.seed, notes=req.notes)
    jm.launch(run.run_id, config)
    return {"run_id": run.run_id}


@router.post("/runs/compare")
def compare_runs(req: CompareRequest):
    registry = _get_registry()
    return registry.compare_runs(req.run_ids)


@router.post("/runs/walk_forward")
def launch_walk_forward(req: WalkForwardRequest):
    """Run a walk-forward analysis: each fold is a normal run (visible in the
    dashboard); returns the fold ids plus aggregated out-of-sample metrics."""
    jm = _get_job_manager()
    settings = get_settings()

    config = {
        "name": "walk_forward",
        "backtest": {
            "start_date": req.start_date,
            "end_date": req.end_date,
            "initial_capital": req.initial_capital,
            "commission_pct": req.commission_pct,
            "slippage_pct": req.slippage_pct,
            "spread_pct": req.spread_pct,
            "short_borrow_annual_pct": req.short_borrow_annual_pct,
            "market_impact_coefficient": req.market_impact_coefficient,
            "market_impact_crossover_participation": req.market_impact_crossover_participation,
            "rebalance_frequency": req.rebalance_frequency,
        },
        "strategies": {"enabled": req.strategies},
        "strategy_params": req.strategy_params,
        "allocation_method": req.allocation_method or settings.allocation_method,
        "kelly_fraction": (
            req.kelly_fraction if req.kelly_fraction is not None
            else settings.kelly_fraction
        ),
        "signal_combination": req.signal_combination or settings.signal_combination,
        "strategy_circuit_breaker": (
            req.strategy_circuit_breaker or settings.strategy_circuit_breaker
        ),
        "strategy_regime_weights": (
            req.strategy_regime_weights or settings.strategy_regime_weights
        ),
        "data_source": req.data_source,
        "seed": req.seed,
        "risk": {**settings.risk.model_dump(), **req.risk_overrides},
    }
    if req.regime_overlay is not None:
        config["regime_overlay"] = req.regime_overlay
    if req.universe_symbols:
        config["universe_symbols"] = req.universe_symbols

    return jm.run_walk_forward_sync(
        config,
        n_splits=req.n_splits,
        train_pct=req.train_pct,
        seed=req.seed,
        param_grid=req.param_grid,
        selection_metric=req.selection_metric,
        embargo_days=req.embargo_days,
        pbo_embargo_pct=req.pbo_embargo_pct,
    )
