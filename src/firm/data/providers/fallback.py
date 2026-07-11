"""FallbackProvider — tries Massive first, falls back per-capability.

Priority order (first non-empty result wins):
  prices            Massive → Tiingo → AlphaVantage → FMP
  news_sentiment    Massive → AlphaVantage → Tiingo
  fundamentals      Massive → FMP
  corporate_actions Massive → (none)
  universe          FMP → (none)
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Sequence

import pandas as pd

from firm.config import Settings, get_settings
from firm.data.providers.base import DataProvider, ProviderError
from firm.data.schemas import (
    CORPORATE_ACTION_COLS,
    FUNDAMENTAL_COLS,
    PRICE_COLS,
    SENTIMENT_COLS,
)

log = logging.getLogger("firm.data.providers.fallback")


def _load(name: str, settings: Settings) -> DataProvider | None:
    """Instantiate a provider by name; return None if key is missing."""
    try:
        from firm.data.providers import get_provider
        return get_provider(name, settings=settings)
    except (ProviderError, ValueError, KeyError):
        return None


class FallbackProvider(DataProvider):
    """Massive-first provider with per-capability fallback chain.

    Pass ``settings`` (or let it auto-load from .env) — no explicit api_key
    needed.  Individual providers in the chain are skipped if their key is
    absent or they raise :class:`ProviderError`.
    """

    name = "fallback"

    def __init__(self, api_key: str = "", settings: Settings | None = None) -> None:
        cfg = settings or get_settings()
        # api_key is ignored — each underlying provider reads its own key.
        super().__init__(api_key="__fallback__")
        self._cfg = cfg

        self._prices_chain: list[str] = ["massive", "tiingo", "alphavantage", "fmp"]
        self._sentiment_chain: list[str] = ["massive", "alphavantage", "tiingo"]
        self._fundamentals_chain: list[str] = ["massive", "fmp"]
        self._actions_chain: list[str] = ["massive"]
        self._universe_chain: list[str] = ["fmp"]

    def _try_chain(
        self,
        chain: list[str],
        method: str,
        empty_df: pd.DataFrame,
        *args,
    ) -> pd.DataFrame:
        for name in chain:
            provider = _load(name, self._cfg)
            if provider is None:
                log.debug("fallback_skip provider=%s method=%s reason=no_key", name, method)
                continue
            try:
                result = getattr(provider, method)(*args)
                if isinstance(result, pd.DataFrame) and not result.empty:
                    log.debug("fallback_hit provider=%s method=%s rows=%d", name, method, len(result))
                    return result
                log.debug("fallback_empty provider=%s method=%s", name, method)
            except NotImplementedError:
                log.debug("fallback_skip provider=%s method=%s reason=not_implemented", name, method)
            except ProviderError as exc:
                log.warning("fallback_error provider=%s method=%s error=%s", name, method, exc)
        log.warning("fallback_exhausted method=%s chain=%s", method, chain)
        return empty_df

    def get_prices(
        self,
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        return self._try_chain(
            self._prices_chain, "get_prices",
            pd.DataFrame(columns=PRICE_COLS),
            symbols, start, end,
        )

    def get_news_sentiment(
        self,
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        return self._try_chain(
            self._sentiment_chain, "get_news_sentiment",
            pd.DataFrame(columns=SENTIMENT_COLS),
            symbols, start, end,
        )

    def get_fundamentals(
        self,
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        return self._try_chain(
            self._fundamentals_chain, "get_fundamentals",
            pd.DataFrame(columns=FUNDAMENTAL_COLS),
            symbols, start, end,
        )

    def get_corporate_actions(
        self,
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        return self._try_chain(
            self._actions_chain, "get_corporate_actions",
            pd.DataFrame(columns=CORPORATE_ACTION_COLS),
            symbols, start, end,
        )

    def get_universe_constituents(self, index: str, date: str = "") -> list[str]:
        for name in self._universe_chain:
            provider = _load(name, self._cfg)
            if provider is None:
                continue
            try:
                result = provider.get_universe_constituents(index, date)
                if result:
                    return result
            except (NotImplementedError, ProviderError) as exc:
                log.warning("fallback_universe_error provider=%s error=%s", name, exc)
        return []
