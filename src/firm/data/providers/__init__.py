"""Concrete data-provider adapters behind the :class:`DataProvider` ABC."""

from __future__ import annotations

from firm.data.providers.alphavantage import AlphaVantageProvider
from firm.data.providers.base import DataProvider, ProviderError
from firm.data.providers.fallback import FallbackProvider
from firm.data.providers.fmp import FMPProvider
from firm.data.providers.ibkr import IBKRProvider
from firm.data.providers.massive import MassiveProvider
from firm.data.providers.tiingo import TiingoProvider

__all__ = [
    "DataProvider",
    "ProviderError",
    "FallbackProvider",
    "TiingoProvider",
    "AlphaVantageProvider",
    "FMPProvider",
    "IBKRProvider",
    "MassiveProvider",
    "get_provider",
]

_REGISTRY: dict[str, type[DataProvider]] = {
    FallbackProvider.name: FallbackProvider,
    MassiveProvider.name: MassiveProvider,
    TiingoProvider.name: TiingoProvider,
    AlphaVantageProvider.name: AlphaVantageProvider,
    FMPProvider.name: FMPProvider,
    IBKRProvider.name: IBKRProvider,
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
