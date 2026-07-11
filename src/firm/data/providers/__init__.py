"""Concrete data-provider adapters behind the :class:`DataProvider` ABC."""

from __future__ import annotations

from firm.data.providers.alphavantage import AlphaVantageProvider
from firm.data.providers.base import DataProvider, ProviderError
from firm.data.providers.fmp import FMPProvider
from firm.data.providers.ibkr import IBKRProvider
from firm.data.providers.polygon import PolygonProvider
from firm.data.providers.tiingo import TiingoProvider

__all__ = [
    "DataProvider",
    "ProviderError",
    "PolygonProvider",
    "TiingoProvider",
    "AlphaVantageProvider",
    "FMPProvider",
    "IBKRProvider",
    "get_provider",
]

_REGISTRY: dict[str, type[DataProvider]] = {
    PolygonProvider.name: PolygonProvider,
    TiingoProvider.name: TiingoProvider,
    AlphaVantageProvider.name: AlphaVantageProvider,
    FMPProvider.name: FMPProvider,
    IBKRProvider.name: IBKRProvider,
}


def get_provider(name: str, **kwargs) -> DataProvider:
    """Instantiate a provider adapter by name (``"polygon"``, ``"fmp"``, ...).

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
