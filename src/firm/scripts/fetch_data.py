"""CLI: fetch market data from configured providers and cache locally.

Usage::

    fetch-data --index SP500 --start 2018-01-01 --end 2023-12-31
    fetch-data --symbols AAPL,MSFT,GOOG --start 2020-01-01 --end 2023-12-31
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from firm.config import get_settings
from firm.data.cache import ParquetCache
from firm.data.providers.alphavantage import AlphaVantageProvider
from firm.data.providers.fmp import FMPProvider
from firm.data.providers.polygon import PolygonProvider
from firm.data.providers.tiingo import TiingoProvider
from firm.logging_setup import setup_logging

log = logging.getLogger("firm.scripts.fetch_data")

_PROVIDER_MAP = {
    "polygon": lambda key: PolygonProvider(key),
    "tiingo": lambda key: TiingoProvider(key),
    "alphavantage": lambda key: AlphaVantageProvider(key),
    "fmp": lambda key: FMPProvider(key),
}

_KEY_MAP = {
    "polygon": "polygon_api_key",
    "tiingo": "tiingo_api_key",
    "alphavantage": "alphavantage_api_key",
    "fmp": "fmp_api_key",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch and cache market data.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--symbols", help="Comma-separated ticker list")
    group.add_argument("--index", help="Index name (e.g. SP500)")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    p.add_argument(
        "--providers",
        default=None,
        help="Comma-separated provider names (default: from settings.yaml)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = _parse_args(argv)
    cfg = get_settings()

    cache = ParquetCache(cfg.data.cache_dir)

    providers_requested = (
        [p.strip() for p in args.providers.split(",")]
        if args.providers
        else [cfg.data.price_provider, cfg.data.fundamental_provider, cfg.data.sentiment_provider]
    )
    providers_requested = list(dict.fromkeys(providers_requested))

    # Resolve symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        log.info("Resolving universe for %s ...", args.index)
        p_name = cfg.data.price_provider
        key = getattr(cfg, _KEY_MAP[p_name], "")
        prov = _PROVIDER_MAP[p_name](key)
        symbols = prov.get_universe_constituents(args.index, args.end)
        if not symbols:
            log.error("Could not resolve universe; exiting.")
            sys.exit(1)
        log.info("Universe: %d symbols", len(symbols))

    all_prices: list[pd.DataFrame] = []
    all_fundamentals: list[pd.DataFrame] = []
    all_sentiment: list[pd.DataFrame] = []

    for name in providers_requested:
        if name not in _PROVIDER_MAP:
            log.warning("Unknown provider '%s'; skipping.", name)
            continue
        api_key = getattr(cfg, _KEY_MAP[name], "")
        if not api_key:
            log.warning("No API key for %s; skipping.", name)
            continue

        prov = _PROVIDER_MAP[name](api_key)
        log.info("Fetching from %s …", name)

        # Prices
        cache_key = f"prices/{name}/{args.start}_{args.end}"
        if cache.has(cache_key):
            log.info("  prices: cache hit")
            all_prices.append(cache.get(cache_key))  # type: ignore[arg-type]
        else:
            try:
                df = prov.get_prices(symbols, args.start, args.end)
                if not df.empty:
                    cache.put(cache_key, df)
                    all_prices.append(df)
                    log.info("  prices: %d rows", len(df))
            except NotImplementedError:
                log.debug("  prices: not supported by %s", name)

        # Fundamentals
        cache_key = f"fundamentals/{name}/{args.start}_{args.end}"
        if cache.has(cache_key):
            log.info("  fundamentals: cache hit")
            all_fundamentals.append(cache.get(cache_key))  # type: ignore[arg-type]
        else:
            try:
                df = prov.get_fundamentals(symbols, args.start, args.end)
                if not df.empty:
                    cache.put(cache_key, df)
                    all_fundamentals.append(df)
                    log.info("  fundamentals: %d rows", len(df))
            except NotImplementedError:
                log.debug("  fundamentals: not supported by %s", name)

        # Sentiment
        cache_key = f"sentiment/{name}/{args.start}_{args.end}"
        if cache.has(cache_key):
            log.info("  sentiment: cache hit")
            all_sentiment.append(cache.get(cache_key))  # type: ignore[arg-type]
        else:
            try:
                df = prov.get_news_sentiment(symbols, args.start, args.end)
                if not df.empty:
                    cache.put(cache_key, df)
                    all_sentiment.append(df)
                    log.info("  sentiment: %d rows", len(df))
            except NotImplementedError:
                log.debug("  sentiment: not supported by %s", name)

    # Combine
    combined_prices = pd.concat(all_prices, ignore_index=True) if all_prices else pd.DataFrame()
    combined_fundamentals = (
        pd.concat(all_fundamentals, ignore_index=True) if all_fundamentals else pd.DataFrame()
    )
    combined_sentiment = (
        pd.concat(all_sentiment, ignore_index=True) if all_sentiment else pd.DataFrame()
    )

    if not combined_prices.empty:
        cache.put("combined/prices", combined_prices)
    if not combined_fundamentals.empty:
        cache.put("combined/fundamentals", combined_fundamentals)
    if not combined_sentiment.empty:
        cache.put("combined/sentiment", combined_sentiment)

    log.info(
        "Done. Prices=%d rows, Fundamentals=%d rows, Sentiment=%d rows",
        len(combined_prices),
        len(combined_fundamentals),
        len(combined_sentiment),
    )


if __name__ == "__main__":
    main()
