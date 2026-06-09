"""Pydantic v2 request/response models for the REST API."""

from __future__ import annotations

from pydantic import BaseModel


class RunRequest(BaseModel):
    strategies: list[str] = ["momentum", "trend"]
    strategy_params: dict[str, dict] = {}
    universe_symbols: list[str] | None = None
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    initial_capital: float = 10_000_000
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    rebalance_frequency: str = "weekly"
    risk_overrides: dict[str, float] = {}
    data_source: str = "synthetic"
    seed: int = 42
    notes: str = ""


class RunSummary(BaseModel):
    run_id: str
    status: str
    start_time: str
    end_time: str | None
    notes: str
    metrics: dict[str, float]


class RunDetail(RunSummary):
    config: dict
    config_hash: str
    seed: int
    artifacts_dir: str


class StepRequest(BaseModel):
    strategies: list[str] = ["momentum", "trend"]
    strategy_params: dict[str, dict] = {}
    symbols: list[str] = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    asof_date: str = "2023-06-15"
    data_source: str = "synthetic"
    seed: int = 42


class CompareRequest(BaseModel):
    run_ids: list[str]
