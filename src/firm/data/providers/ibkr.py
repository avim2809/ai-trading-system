"""Interactive Brokers data provider – historical prices and news via ib_async.

Lets the live pipeline source OHLCV price history and news-based sentiment
directly from an IB Gateway / TWS connection instead of a third-party REST
API, reusing the same market-data / news entitlements as the trading account.

:meth:`get_prices` returns daily bars.  :meth:`get_news_sentiment` pulls
headlines via ``reqHistoricalNews`` and derives a sentiment score with a
lightweight lexicon (IBKR supplies headline text, not a numeric score).
Fundamentals, corporate actions, and index constituents are not implemented.
"""

from __future__ import annotations

import logging

import pandas as pd

from firm.data.providers.base import DataProvider
from firm.data.schemas import PRICE_COLS, SENTIMENT_COLS

log = logging.getLogger("firm.data.providers.ibkr")

# Minimal finance-oriented sentiment lexicon.  IBKR returns headline text only,
# so we score it here; this is deliberately simple and dependency-free.
_POSITIVE_WORDS = frozenset({
    "beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "rally",
    "rallies", "gain", "gains", "rise", "rises", "upgrade", "upgrades", "raised",
    "raises", "outperform", "strong", "record", "profit", "growth", "bullish",
    "boost", "boosts", "win", "wins", "approval", "approved", "beat-estimates",
    "top", "tops", "positive", "expands", "expansion", "dividend", "buyback",
})
_NEGATIVE_WORDS = frozenset({
    "miss", "misses", "missed", "plunge", "plunges", "drop", "drops", "fall",
    "falls", "slump", "slumps", "decline", "declines", "downgrade", "downgrades",
    "cut", "cuts", "loss", "losses", "weak", "warning", "warns", "bearish",
    "lawsuit", "probe", "investigation", "recall", "bankruptcy", "fraud",
    "slashes", "slash", "negative", "halts", "halt", "layoffs", "default",
})


def _score_headline(text: str) -> float:
    """Return a sentiment score in [-1, 1] from headline word polarity."""
    if not text:
        return 0.0
    tokens = [t.strip(".,!?:;'\"()[]").lower() for t in text.split()]
    pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        return 0.0
    return float((pos - neg) / (pos + neg))

try:
    from ib_async import IB, Stock, util

    _HAS_IB = True
except ImportError:
    _HAS_IB = False


class IBKRProvider(DataProvider):
    """Fetch historical daily bars from IB Gateway / TWS.

    Args:
        ib: An already-connected ``ib_async.IB`` instance to reuse.  If given,
            the connection is *not* owned by this provider (it will not be
            disconnected here).
        host / port / client_id: Connection parameters used to open a private
            connection when ``ib`` is not supplied.
        market_data_type: IB market-data type. 3 = delayed, 4 = delayed-frozen.
            Delayed data avoids live-subscription errors for daily history.
        what_to_show: IB historical data field (``TRADES``, ``MIDPOINT``, ...).
    """

    name = "ibkr"

    def __init__(
        self,
        ib: "IB | None" = None,
        host: str = "127.0.0.1",
        port: int = 4002,
        client_id: int = 2,
        market_data_type: int = 3,
        what_to_show: str = "TRADES",
    ) -> None:
        if not _HAS_IB:
            raise ImportError(
                "ib_async is not installed. Install the live extra: "
                "pip install 'firm[live]' or pip install ib_async"
            )
        # DataProvider.__init__ expects an api_key; IBKR uses none.
        self.api_key = ""
        self._host = host
        self._port = port
        self._client_id = client_id
        self._market_data_type = market_data_type
        self._what_to_show = what_to_show
        self._ib = ib
        self._owns_connection = ib is None
        self._news_codes: str | None = None

    def _ensure_ib(self) -> "IB":
        if self._ib is not None and self._ib.isConnected():
            return self._ib
        self._ib = IB()
        self._ib.connect(self._host, self._port, clientId=self._client_id)
        log.info(
            "IBKRProvider connected to %s:%d (client %d)",
            self._host, self._port, self._client_id,
        )
        return self._ib

    def get_prices(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        ib = self._ensure_ib()
        ib.reqMarketDataType(self._market_data_type)

        # IB durations are relative to an end datetime; derive a day span from
        # the requested [start, end] window and cap to what IB accepts.
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        days = max((end_ts - start_ts).days, 1)
        duration = f"{days} D" if days <= 365 else f"{(days // 365) + 1} Y"
        today = pd.Timestamp.now().normalize()
        end_dt = "" if end_ts.normalize() >= today else end_ts.strftime("%Y%m%d 23:59:59")

        frames: list[pd.DataFrame] = []
        for sym in symbols:
            try:
                contract = Stock(sym, "SMART", "USD")
                ib.qualifyContracts(contract)
                bars = ib.reqHistoricalData(
                    contract,
                    endDateTime=end_dt,
                    durationStr=duration,
                    barSizeSetting="1 day",
                    whatToShow=self._what_to_show,
                    useRTH=True,
                    formatDate=1,
                )
                if not bars:
                    log.warning("No historical data for %s", sym)
                    continue
                df = util.df(bars)
                if df is None or df.empty:
                    continue
                df = df.rename(columns={"date": "date"})
                df["date"] = pd.to_datetime(df["date"]).dt.date
                df["symbol"] = sym
                # IB TRADES bars are split/dividend adjusted by default.
                df["adj_close"] = df["close"]
                for col in PRICE_COLS:
                    if col not in df.columns:
                        df[col] = None
                frames.append(df[PRICE_COLS])
            except Exception:
                log.exception("Failed to fetch IBKR history for %s", sym)

        if not frames:
            return pd.DataFrame(columns=PRICE_COLS)
        return pd.concat(frames, ignore_index=True)

    def get_fundamentals(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("IBKRProvider does not provide fundamentals; use FMP.")

    def get_news_sentiment(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        ib = self._ensure_ib()

        if self._news_codes is None:
            try:
                self._news_codes = "+".join(p.code for p in ib.reqNewsProviders())
            except Exception:
                log.exception("Failed to list IBKR news providers")
                self._news_codes = ""
        if not self._news_codes:
            log.warning("No IBKR news providers available; returning empty sentiment.")
            return pd.DataFrame(columns=SENTIMENT_COLS)

        start_dt = f"{pd.Timestamp(start).strftime('%Y-%m-%d')} 00:00:00.0"
        end_dt = f"{pd.Timestamp(end).strftime('%Y-%m-%d')} 23:59:59.0"

        rows: list[dict] = []
        for sym in symbols:
            try:
                contract = Stock(sym, "SMART", "USD")
                ib.qualifyContracts(contract)
                if not contract.conId:
                    continue
                headlines = ib.reqHistoricalNews(
                    contract.conId,
                    self._news_codes,
                    start_dt,
                    end_dt,
                    totalResults=300,
                )
                for h in headlines or []:
                    # ib_async HistoricalNews.time is a datetime or string.
                    ts = pd.to_datetime(getattr(h, "time", None), errors="coerce")
                    headline = getattr(h, "headline", "") or ""
                    rows.append(
                        {
                            "date": ts.date() if not pd.isna(ts) else None,
                            "symbol": sym,
                            "sentiment_score": _score_headline(headline),
                            "news_volume": 1,
                            "source": getattr(h, "providerCode", "") or "",
                            "headline": headline,
                        }
                    )
            except Exception:
                log.exception("Failed to fetch IBKR news for %s", sym)

        if not rows:
            return pd.DataFrame(columns=SENTIMENT_COLS)

        df = pd.DataFrame(rows, columns=SENTIMENT_COLS)
        df = df.dropna(subset=["date"])
        if df.empty:
            return pd.DataFrame(columns=SENTIMENT_COLS)
        # Aggregate to one record per (date, symbol): mean score, summed volume.
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
        raise NotImplementedError("IBKRProvider does not provide corporate actions; use Polygon.")

    def get_universe_constituents(self, index: str, date: str) -> list[str]:
        raise NotImplementedError("IBKRProvider does not provide index constituents; use Polygon or FMP.")
