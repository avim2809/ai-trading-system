"""Typed application configuration.

Loads API keys from .env via pydantic-settings and merges with
config/settings.yaml for all runtime parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


class UniverseConfig(BaseModel):
    index: str = "SP500"
    min_market_cap: int = 1_000_000_000
    min_avg_volume: int = 500_000


class BacktestConfig(BaseModel):
    start_date: str = "2018-01-01"
    end_date: str = "2023-12-31"
    initial_capital: float = 10_000_000
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    rebalance_frequency: str = "weekly"


class RiskConfig(BaseModel):
    max_position_pct: float = 0.05
    max_gross_exposure: float = 2.0
    max_net_exposure: float = 0.5
    max_sector_pct: float = 0.25
    vol_target: float = 0.15
    max_drawdown_pct: float = 0.20
    # Optional HMM market-regime exposure overlay (off unless ``enabled: true``).
    # See firm.agents.risk.RiskAgent and firm.regime.detector.MarketRegimeDetector.
    regime_overlay: dict[str, Any] = {}


class DataConfig(BaseModel):
    cache_dir: str = "data/cache"
    price_provider: str = "fallback"
    fundamental_provider: str = "fallback"
    sentiment_provider: str = "fallback"


class Settings(BaseSettings):
    """Root settings – merges .env keys with settings.yaml sections."""

    tiingo_api_key: str = ""
    alphavantage_api_key: str = ""
    fmp_api_key: str = ""
    massive_api_key: str = ""
    finnhub_api_key: str = ""
    twelvedata_api_key: str = ""
    sec_edgar_user_agent: str = ""
    fred_api_key: str = ""

    request_timeout_seconds: int = 30
    max_retries: int = 3

    universe: UniverseConfig = UniverseConfig()
    backtest: BacktestConfig = BacktestConfig()
    risk: RiskConfig = RiskConfig()
    data: DataConfig = DataConfig()
    # Alpha strategies to wire into the analysts. Empty → all registered.
    strategies: list[str] = []
    # Per-strategy constructor params (e.g. stat_arb predefined_pairs).
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    # Portfolio allocation method: conviction_weighted | equal_weight |
    # risk_parity | kelly. ``kelly_fraction`` only used when method == kelly.
    allocation_method: str = "conviction_weighted"
    kelly_fraction: float = 0.5
    # Research signal combination: {"method": "confidence"|"optimal"}.
    signal_combination: dict[str, Any] = Field(default_factory=dict)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def require(self, key: str) -> str:
        """Return the value of a settings field, raising if empty."""
        value = getattr(self, key, "")
        if not value:
            raise ValueError(
                f"Settings field '{key}' is required but not set. "
                f"Add {key.upper()} to your .env file."
            )
        return value

    @classmethod
    def from_yaml(cls, path: Path | str | None = None, **overrides: Any) -> "Settings":
        """Build settings by layering YAML values under .env API keys."""
        path = Path(path) if path else _DEFAULT_SETTINGS_PATH
        yaml_data: dict[str, Any] = {}
        if path.exists():
            with open(path) as f:
                yaml_data = yaml.safe_load(f) or {}
        merged = {**yaml_data, **overrides}
        return cls(**merged)


def get_settings(yaml_path: Path | str | None = None) -> Settings:
    """Convenience factory used throughout the codebase."""
    return Settings.from_yaml(yaml_path)
