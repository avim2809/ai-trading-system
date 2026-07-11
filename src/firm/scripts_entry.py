"""Console-script entry points for the ``firm`` package.

Keeps the actual CLI logic importable (and unit-testable) inside the package,
while ``scripts/fetch_data.py`` stays a thin runnable shim. The ``firm-fetch-data``
console script (declared in ``pyproject.toml``) maps to :func:`fetch_data_main`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from firm.config import get_settings
from firm.data import schemas
from firm.data.cache import ParquetCache
from firm.data.providers import get_provider
from firm.logging_setup import configure_logging, get_logger

log = get_logger(__name__)


def build_pit_panel(
    symbols: Sequence[str],
    start: datetime,
    end: datetime,
    *,
    prices_provider: str = "fallback",
    fundamentals_provider: str | None = "fallback",
    sentiment_provider: str | None = "fallback",
    cache: ParquetCache | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch and cache a point-in-time panel keyed by ``(date, symbol)``.

    Pulls prices (always) plus optional fundamentals and news sentiment, writes
    each tidy frame to the Parquet cache, and returns them in a dict. Provider
    failures for the optional domains are logged and skipped so a partial panel
    still builds.

    Args:
        symbols: Tickers to fetch.
        start: Inclusive start date.
        end: Inclusive end date.
        prices_provider: Adapter name for prices.
        fundamentals_provider: Adapter name for fundamentals (or ``None`` to skip).
        sentiment_provider: Adapter name for news sentiment (or ``None`` to skip).
        cache: Cache to write into; a default :class:`ParquetCache` if omitted.

    Returns:
        Mapping of panel name (``"prices"``, ``"fundamentals"``,
        ``"news_sentiment"``) to its DataFrame.
    """
    cache = cache or ParquetCache()
    panels: dict[str, pd.DataFrame] = {}

    log.info(
        "fetch_start",
        extra={"context": {"symbols": list(symbols), "start": str(start), "end": str(end)}},
    )

    prices = get_provider(prices_provider).get_prices(symbols, start, end)
    cache.write(
        ParquetCache.make_key("prices", provider=prices_provider, symbols=sorted(symbols),
                              start=str(start), end=str(end)),
        prices,
    )
    panels["prices"] = prices

    if fundamentals_provider:
        try:
            funda = get_provider(fundamentals_provider).get_fundamentals(symbols, start, end)
            cache.write(
                ParquetCache.make_key("fundamentals", provider=fundamentals_provider,
                                      symbols=sorted(symbols), start=str(start), end=str(end)),
                funda,
            )
            panels["fundamentals"] = funda
        except Exception as exc:  # noqa: BLE001 - partial panel is acceptable
            log.warning("fundamentals_fetch_failed", extra={"context": {"error": str(exc)}})

    if sentiment_provider:
        try:
            news = get_provider(sentiment_provider).get_news_sentiment(symbols, start, end)
            cache.write(
                ParquetCache.make_key("news_sentiment", provider=sentiment_provider,
                                      symbols=sorted(symbols), start=str(start), end=str(end)),
                news,
            )
            panels["news_sentiment"] = news
        except Exception as exc:  # noqa: BLE001
            log.warning("sentiment_fetch_failed", extra={"context": {"error": str(exc)}})

    log.info(
        "fetch_done",
        extra={"context": {name: len(df) for name, df in panels.items()}},
    )
    return panels


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="firm-fetch-data",
        description="Build a point-in-time (date, symbol) panel into the Parquet cache.",
    )
    parser.add_argument("--symbols", required=True, help="Comma-separated tickers, e.g. AAPL,MSFT")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--prices-provider", default="fallback")
    parser.add_argument("--fundamentals-provider", default="fallback")
    parser.add_argument("--sentiment-provider", default="fallback")
    parser.add_argument("--no-fundamentals", action="store_true")
    parser.add_argument("--no-sentiment", action="store_true")
    parser.add_argument(
        "--cache-dir", default=str(settings.data.cache_dir), help="Override cache directory"
    )
    return parser.parse_args(argv)


def fetch_data_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: parse args, build the panel, report row counts."""
    configure_logging()
    args = _parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start = pd.Timestamp(args.start).to_pydatetime()
    end = pd.Timestamp(args.end).to_pydatetime()
    cache = ParquetCache(Path(args.cache_dir))

    panels = build_pit_panel(
        symbols,
        start,
        end,
        prices_provider=args.prices_provider,
        fundamentals_provider=None if args.no_fundamentals else args.fundamentals_provider,
        sentiment_provider=None if args.no_sentiment else args.sentiment_provider,
        cache=cache,
    )

    for name, df in panels.items():
        n_symbols = (
            df[schemas.COL_SYMBOL].nunique() if schemas.COL_SYMBOL in df.columns else 0
        )
        print(f"{name:16s} rows={len(df):>8d} symbols={n_symbols}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(fetch_data_main())
