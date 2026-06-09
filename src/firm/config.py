"""Typed application configuration.

Loads API keys from .env via pydantic-settings and merges with
config/settings.yaml for all runtime parameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
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


class DataConfig(BaseModel):
    cache_dir: str = "data/cache"
    price_provider: str = "polygon"
    fundamental_provider: str = "fmp"
    sentiment_provider: str = "tiingo"


class Settings(BaseSettings):
    """Root settings – merges .env keys with settings.yaml sections."""

    polygon_api_key: str = ""
    tiingo_api_key: str = ""
    alphavantage_api_key: str = ""
    fmp_api_key: str = ""

    universe: UniverseConfig = UniverseConfig()
    backtest: BacktestConfig = BacktestConfig()
    risk: RiskConfig = RiskConfig()
    data: DataConfig = DataConfig()

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

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
