"""Pydantic v2 request/response models for the REST API."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator


class RunRequest(BaseModel):
    strategies: list[str] = ["momentum", "trend"]
    strategy_params: dict[str, dict] = {}
    universe_symbols: list[str] | None = None
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    initial_capital: float = Field(default=10_000_000, gt=0)
    commission_pct: float = Field(default=0.001, ge=0, lt=1)
    slippage_pct: float = Field(default=0.0005, ge=0, lt=1)
    spread_pct: float = Field(default=0.0002, ge=0, lt=1)
    short_borrow_annual_pct: float = Field(default=0.003, ge=0, lt=1)
    market_impact_coefficient: float = Field(default=0.0, ge=0, lt=1)
    # Optional linear-below/sqrt-above crossover participation (None = pure
    # sqrt law at every size, unchanged default). See
    # firm.agents._liquidity.market_impact_pct.
    market_impact_crossover_participation: float | None = Field(default=None, gt=0, lt=1)
    rebalance_frequency: str = "weekly"
    risk_overrides: dict[str, float] = {}
    # Optional HMM market-regime exposure overlay; merged over the settings
    # default. Nested (enable flag + exposure_map), so it cannot ride in the
    # number-only ``risk_overrides`` map.
    regime_overlay: dict | None = None
    # Portfolio allocation + research combination (fall back to settings.yaml
    # defaults when omitted). allocation_method: conviction_weighted |
    # equal_weight | risk_parity | kelly. signal_combination:
    # {"method": "confidence"|"optimal"|"hrp"}.
    allocation_method: str | None = None
    kelly_fraction: float | None = None
    signal_combination: dict | None = None
    # Optional generic per-strategy rolling-Sharpe circuit breaker (off unless
    # ``enabled: true``); merged over the settings default so partial
    # overrides work. See firm.agents.research._circuit_breaker. Disabled by
    # default: docs/portfolio_construction_diagnosis.md found naive default
    # thresholds over-gate volatile-but-legitimate strategies on short
    # windows and net *hurt* portfolio Sharpe — opt in deliberately and
    # validate on your own data before relying on it.
    strategy_circuit_breaker: dict | None = None
    # Optional per-strategy regime-conditional score multipliers (off unless
    # ``enabled: true``). See firm.agents.research._regime_weights.
    strategy_regime_weights: dict | None = None
    data_source: str = "synthetic"
    seed: int = 42
    notes: str = ""

    @field_validator("start_date", "end_date")
    @classmethod
    def _valid_iso_date(cls, v: str) -> str:
        try:
            date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"invalid date {v!r}; expected YYYY-MM-DD") from exc
        return v

    @field_validator("rebalance_frequency")
    @classmethod
    def _valid_frequency(cls, v: str) -> str:
        allowed = {"daily", "weekly", "monthly"}
        if v not in allowed:
            raise ValueError(f"rebalance_frequency must be one of {sorted(allowed)}")
        return v

    @model_validator(mode="after")
    def _end_after_start(self) -> "RunRequest":
        if date.fromisoformat(self.end_date) < date.fromisoformat(self.start_date):
            raise ValueError("end_date must be on or after start_date")
        return self


class WalkForwardRequest(RunRequest):
    """A backtest request plus walk-forward split controls."""

    n_splits: int = Field(default=5, ge=2, le=20)
    train_pct: float = Field(default=0.7, gt=0.0, lt=1.0)
    # Optional parameter grid for genuine train->select->test walk-forward
    # optimization: each entry is a partial config override (e.g.
    # {"strategy_params": {"momentum": {"lookback_days": 60}}}) backtested
    # on every fold's train window; the best-by-selection_metric candidate is
    # the one actually run on that fold's test window. Omitted/empty means
    # every fold just replays the base request unchanged on the test window
    # (a plain sequential OOS check, not an optimization — there's nothing
    # to select between with fewer than 2 candidates).
    param_grid: list[dict] | None = Field(default=None, max_length=25)
    selection_metric: str = "sharpe_ratio"
    # Calendar-day gap enforced between each fold's train and test windows —
    # an embargo against train/test boundary leakage (López de Prado). 1
    # (default) matches prior hard-coded behaviour exactly.
    embargo_days: int = Field(default=1, ge=0, le=30)
    # Fraction of each CSCV block purged from the edges of out-of-sample
    # blocks adjacent to in-sample blocks when computing PBO from a genuine
    # param_grid's trial data — see firm.eval.overfitting.cscv_pbo. 0.0
    # (default) reproduces the original, unpurged CSCV split exactly.
    pbo_embargo_pct: float = Field(default=0.0, ge=0.0, lt=0.5)


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
