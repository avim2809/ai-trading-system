"""Alpaca data provider – historical prices and news via alpaca-py.

Lets the Alpaca live instance source OHLCV price history from the same
venue it trades against (Alpaca's Market Data API), instead of a
third-party REST provider, mirroring why ``IBKRProvider`` exists for the
IBKR instance. Also exposes Alpaca's News API (Benzinga-sourced, bundled
with the same trading API key — no separate signup) as an extra sentiment
source for the shared ``FallbackProvider`` chain, not just this instance.

Unlike IBKR's persistent socket connection, Alpaca's market-data client is
a stateless REST client — no connection-sharing/threading concerns apply
here (see ``IBKRProvider``/``IBKRBroker.shared_connection`` for why that
mattered there).

:meth:`get_prices` returns daily bars, split/dividend-adjusted
(``adjustment="all"``). :meth:`get_news_sentiment` pulls headlines via the
News API and derives a sentiment score with the same lightweight lexicon
``IBKRProvider`` uses (Alpaca supplies headline/summary text, not a numeric
score). Fundamentals, corporate actions, and index constituents are not
implemented — this project already has vendor-agnostic sources for those
(see ``FallbackProvider``).
"""

from __future__ import annotations

import logging

import pandas as pd

from firm.config import Settings, get_settings
from firm.data.providers.base import DataProvider, ProviderError
from firm.data.providers.sentiment_lexicon import score_headline
from firm.data.schemas import PRICE_COLS, SENTIMENT_COLS
from firm.time_utils import utcnow

log = logging.getLogger("firm.data.providers.alpaca")

try:
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest, NewsRequest, StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    _HAS_ALPACA = True
except ImportError:
    _HAS_ALPACA = False


def _is_crypto_symbol(symbol: str) -> bool:
    """True for this codebase's canonical crypto form ("BASE/QUOTE", e.g.
    "BTC/USD") -- mirrors ``firm.brokers.alpaca``'s convention of the same
    name (kept as a local copy rather than a cross-import since the two
    modules -- broker vs data provider -- have no other dependency on each
    other)."""
    return "/" in symbol


class AlpacaProvider(DataProvider):
    """Fetch historical daily bars and news from Alpaca's Market Data API."""

    name = "alpaca"

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        settings: Settings | None = None,
    ) -> None:
        if not _HAS_ALPACA:
            raise ImportError(
                "alpaca-py is not installed. Install the live extra: "
                "pip install 'firm[live]' or pip install alpaca-py"
            )
        self.settings = settings or get_settings()
        self._api_key = api_key or self.settings.require("alpaca_api_key")
        self._secret_key = secret_key or self.settings.require("alpaca_secret_key")
        super().__init__(self._api_key)
        self._stock_client = StockHistoricalDataClient(self._api_key, self._secret_key)
        self._crypto_client = CryptoHistoricalDataClient(self._api_key, self._secret_key)
        self._news_client = NewsClient(self._api_key, self._secret_key)

    def get_prices(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        crypto_symbols = [s for s in symbols if _is_crypto_symbol(s)]
        equity_symbols = [s for s in symbols if not _is_crypto_symbol(s)]
        frames: list[pd.DataFrame] = []
        if equity_symbols:
            frames.append(self._get_equity_prices(equity_symbols, start, end))
        if crypto_symbols:
            frames.append(self._get_crypto_prices(crypto_symbols, start, end))
        frames = [f for f in frames if not f.empty]
        if not frames:
            return self.empty_prices()
        return pd.concat(frames, ignore_index=True)

    def _get_crypto_prices(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Daily UTC bars for crypto pairs (e.g. "BTC/USD") via alpaca-py's
        crypto market-data client. Unlike equities, crypto trades 24/7 with
        no splits/dividends, so there's no adjustment parameter and
        ``adj_close`` is just ``close``, mirroring this provider's own
        equity convention (``adjustment="all"`` already bakes corporate
        actions into OHLC there too).

        Only *completed* UTC daily bars are returned -- Alpaca can hand back
        a same-day, still-forming bar when ``end`` is close to "now" (its
        volume/close aren't final yet), which would leak into any
        signal computed off it (e.g. BtcTrendSleeve's weekly review). Bars
        dated on or after the current UTC calendar date are dropped.
        """
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=list(symbols),
                timeframe=TimeFrame.Day,
                start=pd.Timestamp(start).to_pydatetime(),
                end=pd.Timestamp(end).to_pydatetime(),
            )
            barset = self._crypto_client.get_crypto_bars(request)
        except Exception as exc:
            raise ProviderError(f"Alpaca get_crypto_bars failed: {exc}") from exc

        today = utcnow().date()
        frames: list[pd.DataFrame] = []
        for sym in symbols:
            bars = barset.data.get(sym, [])
            if not bars:
                log.warning("No crypto historical data for %s", sym)
                continue
            rows = [
                {
                    "date": bar.timestamp.date(),
                    "symbol": sym,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    "adj_close": bar.close,
                }
                for bar in bars
                if bar.timestamp.date() < today
            ]
            if not rows:
                log.info("No completed UTC crypto bars for %s (today's forming bar dropped)", sym)
                continue
            frames.append(pd.DataFrame(rows, columns=PRICE_COLS))

        if not frames:
            return self.empty_prices()
        return pd.concat(frames, ignore_index=True)

    def _get_equity_prices(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=list(symbols),
                timeframe=TimeFrame.Day,
                start=pd.Timestamp(start).to_pydatetime(),
                end=pd.Timestamp(end).to_pydatetime(),
                adjustment=Adjustment.ALL,
                # Free/paper accounts have no SIP subscription; the default
                # feed raises "subscription does not permit querying recent
                # SIP data" whenever *end* is close to now (confirmed live
                # 2026-08-30 — surfaced by a caller requesting data up to
                # utcnow() for the first time). IEX is free-tier-accessible
                # and sufficient for daily bars.
                feed=DataFeed.IEX,
            )
            barset = self._stock_client.get_stock_bars(request)
        except Exception as exc:
            raise ProviderError(f"Alpaca get_stock_bars failed: {exc}") from exc

        frames: list[pd.DataFrame] = []
        for sym in symbols:
            bars = barset.data.get(sym, [])
            if not bars:
                log.warning("No historical data for %s", sym)
                continue
            rows = [
                {
                    "date": bar.timestamp.date(),
                    "symbol": sym,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                    # adjustment="all" above already bakes splits/dividends
                    # into OHLC — mirrors IBKRProvider's own adj_close = close.
                    "adj_close": bar.close,
                }
                for bar in bars
            ]
            frames.append(pd.DataFrame(rows, columns=PRICE_COLS))

        if not frames:
            return self.empty_prices()
        return pd.concat(frames, ignore_index=True)

    def get_fundamentals(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("AlpacaProvider does not provide fundamentals; use FMP.")

    def get_news_sentiment(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        try:
            request = NewsRequest(
                symbols=",".join(symbols),
                start=pd.Timestamp(start).to_pydatetime(),
                end=pd.Timestamp(end).to_pydatetime(),
            )
            newsset = self._news_client.get_news(request)
        except Exception as exc:
            raise ProviderError(f"Alpaca get_news failed: {exc}") from exc

        wanted = set(symbols)
        rows: list[dict] = []
        for article in newsset.data.get("news", []):
            headline = article.headline or ""
            score = score_headline(headline)
            date = article.created_at.date()
            for sym in article.symbols:
                if sym in wanted:
                    rows.append(
                        {
                            "date": date,
                            "symbol": sym,
                            "sentiment_score": score,
                            "news_volume": 1,
                            "source": article.source or "alpaca",
                            "headline": headline,
                        }
                    )

        if not rows:
            return self.empty_news()
        df = pd.DataFrame(rows, columns=SENTIMENT_COLS)
        # Aggregate to one record per (date, symbol): mean score, summed
        # volume — same convention as IBKRProvider.get_news_sentiment.
        agg = (
            df.groupby(["date", "symbol"], as_index=False)
            .agg(
                sentiment_score=("sentiment_score", "mean"),
                news_volume=("news_volume", "sum"),
                source=("source", "first"),
                headline=("headline", "first"),
            )
        )
        return agg[SENTIMENT_COLS]

    def get_corporate_actions(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("AlpacaProvider does not provide corporate actions; use Massive.")

    def get_universe_constituents(self, index: str, date: str) -> list[str]:
        raise NotImplementedError("AlpacaProvider does not provide index constituents; use FMP.")

    def get_analyst_ratings(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("AlpacaProvider does not provide analyst ratings; use FMP.")

    def get_ai_scores(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("AlpacaProvider does not provide AI scores; use Danelfin.")

    def get_live_signals(self, symbols: list[str]) -> pd.DataFrame:
        raise NotImplementedError("AlpacaProvider does not provide live signals; use Danelfin.")

    def get_best_stocks(self) -> pd.DataFrame:
        raise NotImplementedError("AlpacaProvider does not provide best-stocks; use Danelfin.")
