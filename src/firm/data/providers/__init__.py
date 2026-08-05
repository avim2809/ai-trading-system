"""Concrete data-provider adapters behind the :class:`DataProvider` ABC."""

from __future__ import annotations

from firm.data.providers.alpaca import AlpacaProvider
from firm.data.providers.alphavantage import AlphaVantageProvider
from firm.data.providers.base import DataProvider, ProviderError
from firm.data.providers.danelfin import DanelfinProvider
from firm.data.providers.edgar import EdgarProvider
from firm.data.providers.fallback import FallbackProvider
from firm.data.providers.finnhub import FinnhubProvider
from firm.data.providers.fmp import FMPProvider
from firm.data.providers.ibkr import IBKRProvider
from firm.data.providers.massive import MassiveProvider
from firm.data.providers.tiingo import TiingoProvider
from firm.data.providers.twelvedata import TwelveDataProvider

__all__ = [
    "DataProvider",
    "ProviderError",
    "FallbackProvider",
    "TiingoProvider",
    "AlphaVantageProvider",
    "FMPProvider",
    "FinnhubProvider",
    "EdgarProvider",
    "TwelveDataProvider",
    "IBKRProvider",
    "MassiveProvider",
    "DanelfinProvider",
    "AlpacaProvider",
    "get_provider",
]

_REGISTRY: dict[str, type[DataProvider]] = {
    FallbackProvider.name: FallbackProvider,
    MassiveProvider.name: MassiveProvider,
    TiingoProvider.name: TiingoProvider,
    AlphaVantageProvider.name: AlphaVantageProvider,
    FMPProvider.name: FMPProvider,
    FinnhubProvider.name: FinnhubProvider,
    EdgarProvider.name: EdgarProvider,
    TwelveDataProvider.name: TwelveDataProvider,
    IBKRProvider.name: IBKRProvider,
    DanelfinProvider.name: DanelfinProvider,
    AlpacaProvider.name: AlpacaProvider,
}


def get_provider(name: str, **kwargs) -> DataProvider:
    """Instantiate a provider adapter by name (``"massive"``, ``"fmp"``, ...).

    Args:
        name: Adapter identifier (see :data:`_REGISTRY` keys).
        **kwargs: Forwarded to the adapter constructor (e.g. ``settings=``).

    Raises:
        KeyError: If ``name`` is not a known adapter.
    """
    try:
        cls = _REGISTRY[name.lower()]
    except KeyError as exc:
        raise KeyError(
            f"Unknown data provider '{name}'. Known: {sorted(_REGISTRY)}"
        ) from exc
    return cls(**kwargs)
