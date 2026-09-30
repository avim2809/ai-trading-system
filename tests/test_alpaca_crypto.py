"""Tests for AlpacaBroker's crypto support (BTC trend satellite sleeve work).

Covers the crypto-specific behaviours added alongside the BTC trend sleeve
(``src/firm/allocation/btc_trend.py``):

- symbol normalisation: Alpaca's positions endpoints return/accept the
  compact "BTCUSD" form while every caller in this codebase uses the
  canonical "BTC/USD" form (confirmed via Alpaca's own support docs, see
  ``firm.brokers.alpaca``'s module-level symbology note);
- time_in_force: crypto orders reject "day" and must go out as gtc/ioc;
- fractional, uncapped-precision quantities (no int rounding, unlike the
  equity flip-split path);
- no extended_hours concept for a 24/7 crypto market;
- market orders only (this adapter hasn't been built/tested against
  crypto limit/stop_limit, even though Alpaca itself supports them);
- crypto price lookups route through the crypto data client, not the
  equity one, and mixed equity+crypto batches are partitioned correctly.

Equity behaviour must remain byte-for-byte unchanged -- see
``test_alpaca_broker.py`` for that coverage, all still passing after this
change.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("alpaca")

from firm.brokers.alpaca import AlpacaBroker
from firm.brokers.base import BrokerError, OrderRequest
from alpaca.trading.enums import AssetClass
from alpaca.trading.requests import MarketOrderRequest


class _FakeTradingClient:
    """Minimal fake covering only what AlpacaBroker's crypto paths call."""

    def __init__(self, position_qty: float | None = None):
        self._position_qty = position_qty
        self.submit_calls: list = []
        self.position_lookups: list[str] = []
        self._n = 0

    def get_open_position(self, symbol: str):
        self.position_lookups.append(symbol)
        if self._position_qty is None:
            raise Exception("position does not exist")
        return SimpleNamespace(
            symbol=symbol,
            qty=self._position_qty,
            avg_entry_price=50_000.0,
            market_value=self._position_qty * 50_000.0,
            unrealized_pl=0.0,
        )

    def get_all_positions(self):
        return self._all_positions

    def submit_order(self, req):
        self._n += 1
        self.submit_calls.append(req)
        return SimpleNamespace(
            id=f"o{self._n}",
            symbol=req.symbol,
            side=req.side,
            qty=req.qty,
            filled_qty=req.qty,
            filled_avg_price=50_000.0,
            status="filled",
            submitted_at=None,
        )


def _crypto_broker(position_qty: float | None = None) -> tuple[AlpacaBroker, _FakeTradingClient]:
    broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
    client = _FakeTradingClient(position_qty)
    broker._trading = client
    return broker, client


class TestCryptoOrderSubmission:
    def test_market_order_uses_slash_symbol_and_gtc(self):
        broker, client = _crypto_broker()
        broker.submit_order(
            OrderRequest(symbol="BTC/USD", side="buy", quantity=0.05123456789, fractional=True)
        )
        req = client.submit_calls[0]
        assert isinstance(req, MarketOrderRequest)
        assert req.symbol == "BTC/USD"
        assert str(req.time_in_force) == "TimeInForce.GTC" or req.time_in_force.value == "gtc"

    def test_day_tif_is_overridden_to_gtc_for_crypto(self, caplog):
        broker, client = _crypto_broker()
        with caplog.at_level("WARNING"):
            broker.submit_order(
                OrderRequest(symbol="BTC/USD", side="buy", quantity=0.01, time_in_force="day")
            )
        req = client.submit_calls[0]
        assert req.time_in_force.value == "gtc"
        assert any("rejects for crypto" in r.message for r in caplog.records)

    def test_ioc_tif_is_preserved_for_crypto(self):
        broker, client = _crypto_broker()
        broker.submit_order(
            OrderRequest(symbol="BTC/USD", side="buy", quantity=0.01, time_in_force="ioc")
        )
        req = client.submit_calls[0]
        assert req.time_in_force.value == "ioc"

    def test_fractional_quantity_is_not_rounded_to_an_integer(self):
        broker, client = _crypto_broker()
        broker.submit_order(
            OrderRequest(symbol="BTC/USD", side="buy", quantity=0.123456789, fractional=True)
        )
        req = client.submit_calls[0]
        assert req.qty == 0.123456789

    def test_quantity_beyond_9_decimals_is_rounded_not_rejected(self):
        broker, client = _crypto_broker()
        broker.submit_order(
            OrderRequest(symbol="BTC/USD", side="buy", quantity=0.1234567891234, fractional=True)
        )
        req = client.submit_calls[0]
        assert req.qty == round(0.1234567891234, 9)

    def test_extended_hours_is_ignored_for_crypto(self, caplog):
        broker, client = _crypto_broker()
        with caplog.at_level("WARNING"):
            broker.submit_order(
                OrderRequest(symbol="BTC/USD", side="buy", quantity=0.01, extended_hours=True)
            )
        req = client.submit_calls[0]
        assert not getattr(req, "extended_hours", False)
        assert any("crypto trades" in r.message for r in caplog.records)

    def test_non_market_order_type_raises(self):
        broker, client = _crypto_broker()
        with pytest.raises(BrokerError, match="only 'market' is implemented"):
            broker.submit_order(
                OrderRequest(
                    symbol="BTC/USD", side="buy", quantity=0.01,
                    order_type="limit", limit_price=50_000.0,
                )
            )
        assert client.submit_calls == []

    def test_long_only_crypto_never_triggers_flip_split(self):
        """Even with an existing long position, a crypto sell larger than
        the position must submit as a single order -- Alpaca crypto has no
        short side, so the equity flip-split path must not engage."""
        broker, client = _crypto_broker(position_qty=0.05)
        broker.submit_order(OrderRequest(symbol="BTC/USD", side="sell", quantity=0.08, fractional=True))
        assert len(client.submit_calls) == 1
        assert client.submit_calls[0].qty == 0.08


class TestCryptoPositionSymbolNormalisation:
    def test_get_position_looks_up_compact_symbol_returns_canonical(self):
        broker, client = _crypto_broker(position_qty=0.25)
        pos = broker.get_position("BTC/USD")
        assert client.position_lookups == ["BTCUSD"]
        assert pos.symbol == "BTC/USD"
        assert pos.quantity == 0.25

    def test_get_position_no_position_returns_none(self):
        broker, client = _crypto_broker(position_qty=None)
        assert broker.get_position("BTC/USD") is None
        assert client.position_lookups == ["BTCUSD"]

    def test_equity_position_lookup_is_unaffected(self):
        broker, client = _crypto_broker(position_qty=10)
        pos = broker.get_position("AAPL")
        assert client.position_lookups == ["AAPL"]
        assert pos.symbol == "AAPL"

    def test_get_positions_normalises_crypto_and_leaves_equities_alone(self):
        broker, client = _crypto_broker()
        client._all_positions = [
            SimpleNamespace(
                symbol="BTCUSD", asset_class=AssetClass.CRYPTO, qty=0.1,
                avg_entry_price=50_000.0, market_value=5_000.0, unrealized_pl=10.0,
            ),
            SimpleNamespace(
                symbol="AAPL", asset_class=AssetClass.US_EQUITY, qty=10,
                avg_entry_price=200.0, market_value=2_000.0, unrealized_pl=5.0,
            ),
        ]
        positions = broker.get_positions()
        by_symbol = {p.symbol: p for p in positions}
        assert set(by_symbol) == {"BTC/USD", "AAPL"}
        assert by_symbol["BTC/USD"].quantity == 0.1
        assert by_symbol["AAPL"].quantity == 10


class TestCryptoPriceLookup:
    def _quote(self, bid, ask):
        q = MagicMock()
        q.bid_price = bid
        q.ask_price = ask
        return q

    def test_get_current_price_routes_crypto_through_crypto_client(self):
        broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
        equity_client = MagicMock()
        crypto_client = MagicMock()
        crypto_client.get_crypto_latest_quote.return_value = {"BTC/USD": self._quote(49_900.0, 50_100.0)}
        broker._data = equity_client
        broker._crypto_data = crypto_client

        price = broker.get_current_price("BTC/USD")

        assert price == 50_000.0
        equity_client.get_stock_latest_quote.assert_not_called()
        crypto_client.get_crypto_latest_quote.assert_called_once()

    def test_get_current_price_equity_unaffected(self):
        broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
        equity_client = MagicMock()
        crypto_client = MagicMock()
        equity_client.get_stock_latest_quote.return_value = {"AAPL": self._quote(199.0, 201.0)}
        broker._data = equity_client
        broker._crypto_data = crypto_client

        price = broker.get_current_price("AAPL")

        assert price == 200.0
        crypto_client.get_crypto_latest_quote.assert_not_called()

    def test_get_current_prices_partitions_mixed_batch(self):
        broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
        equity_client = MagicMock()
        crypto_client = MagicMock()
        equity_client.get_stock_latest_quote.return_value = {"AAPL": self._quote(199.0, 201.0)}
        crypto_client.get_crypto_latest_quote.return_value = {"BTC/USD": self._quote(49_900.0, 50_100.0)}
        broker._data = equity_client
        broker._crypto_data = crypto_client

        prices = broker.get_current_prices(["AAPL", "BTC/USD"])

        assert prices == {"AAPL": 200.0, "BTC/USD": 50_000.0}
        equity_call = equity_client.get_stock_latest_quote.call_args[0][0]
        assert equity_call.symbol_or_symbols == ["AAPL"]
        crypto_call = crypto_client.get_crypto_latest_quote.call_args[0][0]
        assert crypto_call.symbol_or_symbols == ["BTC/USD"]

    def test_missing_crypto_quote_omitted_with_warning(self, caplog):
        broker = AlpacaBroker(api_key="k", secret_key="s", paper=True)
        crypto_client = MagicMock()
        crypto_client.get_crypto_latest_quote.return_value = {}
        broker._data = MagicMock()
        broker._crypto_data = crypto_client

        with caplog.at_level("WARNING"):
            prices = broker.get_current_prices(["BTC/USD"])

        assert prices == {}
        assert any("crypto quote" in r.message for r in caplog.records)
