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
    except KeyError:
        # get_provider raises KeyError for an *unknown provider name* — a
        # typo/config bug, not a missing API key. Worth a real warning since
        # it's silently identical to "this provider is just unconfigured"
        # otherwise, masking a config mistake in the fallback chain.
        log.warning("data_provider_unknown_name name=%s", name, exc_info=True)
        return None
    except (ProviderError, ValueError):
        # The provider's own constructor raises this for a genuinely missing
        # API key — expected/benign for an optional provider, so debug-only.
        log.debug("data_provider_unavailable name=%s", name, exc_info=True)
        return None
    except Exception:
        # A provider constructor bug (e.g. a signature mismatch — this has
        # happened) must not crash the whole chain: the entire point of a
        # fallback chain is that one broken/unavailable provider doesn't take
        # down every symbol behind it. Log loudly (this is a real bug, not a
        # missing key) and let the chain move on to the next provider.
        log.warning("data_provider_init_failed name=%s", name, exc_info=True)
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
        symbols: Sequence[str],
        start: datetime | str,
        end: datetime | str,
    ) -> pd.DataFrame:
        """Fill in *symbols* by querying the chain in order, requesting only
        the symbols still missing adequate coverage from each successive
        provider.

        Previously this treated the whole batch as one unit — "first
        non-empty result wins" — which meant a *partial* success from the
        primary provider (e.g. Massive's free-tier rate limit kicking in
        after 5 of 25 symbols, since MassiveProvider itself catches errors
        per-symbol and returns whatever succeeded) silently prevented every
        other provider in the chain from ever being tried for the symbols
        it didn't cover. A 25-symbol universe could end up with real cached
        data for only 5 symbols with no warning, no error — just quietly
        incomplete. Now every provider in the chain gets a shot at whatever
        symbols are still unresolved after the previous ones.

        "Resolved" also isn't just "got any non-empty rows": a provider's
        free tier can silently truncate history to its own rolling window
        (e.g. Massive returning only the last 2 years for a request going
        back to 2020) instead of erroring — a real incident where 5 of 25
        symbols happened to succeed against Massive's rate limit and were
        then never tried against Tiingo, which had the full range for every
        other symbol. So a symbol only counts as fully resolved once some
        provider's data reaches back close to *start*; otherwise later
        providers still get a shot, and whichever provider gave the
        earliest (fullest) coverage for a symbol wins — the truncated
        answer is kept only if nothing better ever turns up.
        """
        remaining = list(dict.fromkeys(symbols))  # de-dup, preserve order
        try:
            start_ts = pd.Timestamp(start)
        except (TypeError, ValueError):
            start_ts = None
        coverage_slack = pd.Timedelta(days=10)  # weekends/holidays at the boundary

        best: dict[str, tuple[pd.Timestamp, pd.DataFrame]] = {}

        for name in chain:
            if not remaining:
                break
            provider = _load(name, self._cfg)
            if provider is None:
                log.debug("fallback_skip provider=%s method=%s reason=no_key", name, method)
                continue
            try:
                result = getattr(provider, method)(remaining, start, end)
            except NotImplementedError:
                log.debug("fallback_skip provider=%s method=%s reason=not_implemented", name, method)
                continue
            except ProviderError as exc:
                log.warning("fallback_error provider=%s method=%s error=%s", name, method, exc)
                continue

            if not isinstance(result, pd.DataFrame) or result.empty:
                log.debug("fallback_empty provider=%s method=%s", name, method)
                continue

            has_symbol = "symbol" in result.columns
            has_date = start_ts is not None and "date" in result.columns
            still_missing: list[str] = []
            for sym in remaining:
                sub = result[result["symbol"] == sym] if has_symbol else result
                if sub.empty:
                    still_missing.append(sym)
                    continue
                min_date = pd.to_datetime(sub["date"], utc=True).dt.tz_localize(None).min() if has_date else pd.Timestamp.min
                prior = best.get(sym)
                if prior is None or min_date < prior[0]:
                    best[sym] = (min_date, sub)
                if has_date and min_date > start_ts + coverage_slack:
                    still_missing.append(sym)

            log.debug(
                "fallback_partial provider=%s method=%s resolved=%d/%d",
                name, method, len(remaining) - len(still_missing), len(remaining),
            )
            remaining = still_missing

        if remaining:
            truly_missing = [s for s in remaining if s not in best]
            truncated = [s for s in remaining if s in best]
            if truncated:
                log.warning(
                    "fallback_truncated_range method=%s symbols=%s requested_start=%s",
                    method, truncated, start_ts.date() if start_ts is not None else start,
                )
            if truly_missing:
                log.warning(
                    "fallback_incomplete method=%s chain=%s missing_symbols=%s",
                    method, chain, truly_missing,
                )
        if not best:
            return empty_df
        merged = pd.concat([df for _, df in best.values()], ignore_index=True)
        # Providers disagree on the "date" column's dtype (Timestamp vs.
        # python date vs. string) *and* on tz-awareness (some emit UTC-suffixed
        # timestamps, e.g. Massive/Tiingo's "published_utc"/"Z"-suffixed
        # dates) — harmless per-provider, but a real incident once two
        # providers' results land in the same merged frame: pandas refuses to
        # even compare naive and tz-aware datetimes in one column, and
        # pyarrow's parquet writer errors on the resulting mixed dtype either
        # way. utc=True treats naive values as already-UTC and converts aware
        # ones to UTC, giving one consistent dtype we can then drop tz from.
        if "date" in merged.columns:
            merged["date"] = pd.to_datetime(merged["date"], utc=True).dt.tz_localize(None).dt.normalize()
        return merged

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
