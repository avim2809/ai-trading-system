"""Interactive Brokers data provider – historical daily prices via ib_insync.

Lets the live pipeline source OHLCV price history directly from an IB Gateway
/ TWS connection instead of a third-party REST API, reusing the same market
data entitlements as the trading account.

Only :meth:`get_prices` is implemented; IBKR is not used for fundamentals,
sentiment, corporate actions, or index constituents here.
"""

from __future__ import annotations

import logging

import pandas as pd

from firm.data.providers.base import DataProvider
from firm.data.schemas import PRICE_COLS

log = logging.getLogger("firm.data.providers.ibkr")

try:
    from ib_insync import IB, Stock, util

    _HAS_IB = True
except ImportError:
    _HAS_IB = False


class IBKRProvider(DataProvider):
    """Fetch historical daily bars from IB Gateway / TWS.

    Args:
        ib: An already-connected ``ib_insync.IB`` instance to reuse.  If given,
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
                "ib_insync is not installed. Install the live extra: "
                "pip install 'firm[live]' or pip install ib_insync"
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
        raise NotImplementedError("IBKRProvider does not provide sentiment; use Tiingo.")

    def get_corporate_actions(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError("IBKRProvider does not provide corporate actions; use Polygon.")

    def get_universe_constituents(self, index: str, date: str) -> list[str]:
        raise NotImplementedError("IBKRProvider does not provide index constituents; use Polygon or FMP.")
