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
    # Bid-ask spread cost: an approximation of the cost of crossing the
    # quoted spread, charged per trade like commission (distinct from
    # slippage_pct, which approximates price-impact from order size/urgency
    # rather than the spread itself). See firm.backtest.commissions.
    spread_pct: float = 0.0002
    # Annualized cost of borrowing shares to sell short, charged daily on
    # short notional (see firm.backtest.firm_strategy). 0.3% is a
    # conservative "general collateral" / easy-to-borrow estimate — real
    # hard-to-borrow names can run materially higher; this is a portfolio-
    # wide approximation, not a per-symbol borrow-availability model.
    short_borrow_annual_pct: float = 0.003
    # Size/volume-aware market-impact cost (square-root law: impact scales
    # with sqrt(trade notional / ADV dollars) — see
    # firm.agents._liquidity.sqrt_impact_pct), on top of the flat
    # commission/slippage/spread rates above, which don't scale with order
    # size relative to a name's trading volume. 0.0 disables it (this
    # Python-level default keeps every existing direct-construction/test
    # backtest unchanged); config/settings.yaml opts in with a conservative
    # calibration for real runs.
    market_impact_coefficient: float = 0.0
    # Optional linear-below/sqrt-above crossover participation rate (None =
    # pure square-root law at every size, unchanged). See
    # firm.agents._liquidity.market_impact_pct.
    market_impact_crossover_participation: float | None = None
    rebalance_frequency: str = "weekly"


class RiskConfig(BaseModel):
    max_position_pct: float = 0.05
    max_gross_exposure: float = 2.0
    max_net_exposure: float = 0.5
    max_sector_pct: float = 0.25
    vol_target: float = 0.15
    max_drawdown_pct: float = 0.20
    # Optional ADV/participation-rate liquidity cap (None = disabled). See
    # firm.agents.risk.RiskAgent._cap_liquidity.
    max_participation_pct: float | None = None
    adv_lookback_days: int = 20
    # Optional pairwise-correlation concentration cap (None = disabled). See
    # firm.agents.risk.RiskAgent._cap_correlated_exposure.
    correlation_threshold: float | None = None
    max_correlated_pair_pct: float = 0.25
    correlation_lookback_days: int = 60
    # Optional CVaR (Conditional Value-at-Risk) tail-risk sizing overlay
    # (None = disabled). Complements vol_target's diagonal-covariance
    # estimate with a distributional tail read. See
    # firm.agents.risk.RiskAgent._cvar_overlay.
    cvar_limit: float | None = None
    cvar_confidence: float = 0.95
    cvar_lookback_days: int = 60
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
    danelfin_api_key: str = ""

    # Investing.com Pro (no official API — authenticated scraper). Both
    # empty and ``investing_scraper_enabled`` False by default: an unset
    # master switch so this net-new, higher-ToS-risk data source never
    # touches the network unless explicitly opted into. See
    # firm.data.investing.session.InvestingSession.
    investing_email: str = ""
    investing_password: str = ""
    investing_scraper_enabled: bool = False
    # "playwright" (default): drive the real login form with a headless
    # browser — no need to reverse-engineer the POST endpoint, but pulls in
    # the optional `investing` extra (see pyproject.toml) and launches
    # Chromium briefly (login only, not every fetch). "endpoint": skip the
    # browser entirely by POSTing directly to a reverse-engineered login
    # endpoint (see the three fields below) — lighter-weight if you've
    # already captured the real request from devtools.
    investing_auth_method: str = "playwright"
    # Only used when investing_auth_method == "endpoint". Deliberately NOT
    # hardcoded anywhere in source: this project does not guess/fabricate
    # third-party API endpoints. Supply these after inspecting the real
    # network request (browser devtools while logging in) — see the Phase 0
    # spike in docs/investing_pro_integration.md. Field names in the login
    # form (the literal HTML ``name="..."`` attributes for email/password)
    # go in ``investing_login_field_map`` as "email:<name>,password:<name>".
    investing_login_page_url: str = ""
    investing_login_post_url: str = ""
    investing_login_field_map: str = ""

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
    # Research signal combination: {"method": "confidence"|"optimal"|"hrp"}.
    # "hrp" (Hierarchical Risk Parity) is an opt-in alternative to "optimal"
    # that never inverts the correlation matrix — see
    # firm.agents.analysts.hrp_signal_weights.
    signal_combination: dict[str, Any] = Field(default_factory=dict)
    # Generic per-strategy rolling-Sharpe circuit breaker (off unless
    # ``enabled: true``). See firm.agents.research._circuit_breaker.
    strategy_circuit_breaker: dict[str, Any] = Field(default_factory=dict)
    # Per-strategy score multipliers conditioned on market regime (off unless
    # ``enabled: true``). See firm.agents.research._regime_weights.
    strategy_regime_weights: dict[str, Any] = Field(default_factory=dict)

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
