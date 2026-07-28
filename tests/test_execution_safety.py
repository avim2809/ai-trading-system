"""Tests for firm.live.execution_safety — no real order is ever transmitted."""

from __future__ import annotations

from firm.live.execution_safety import (
    Order,
    RiskProfile,
    guard_live_submission,
    guard_order,
    is_live_broker,
    trading_armed,
)


def _profile() -> RiskProfile:
    return RiskProfile(
        account_equity=100_000.0,
        max_risk_per_trade_pct=2.0,
        max_daily_loss_pct=6.0,
        max_position_notional=100_000.0,
        min_stop_atr_mult=1.0,
        symbol_allowlist=["SPY"],
    )


def _ok_order() -> Order:
    # risk = |548-540| * 100 = 800 = 0.8% of equity; notional 54,800; stop 8 = 2x ATR.
    return Order(
        symbol="SPY", side="sell", qty=100, order_type="market",
        price=548.0, stop=540.0, atr=4.0,
    )


class TestGuardOrder:
    def test_live_without_env_routes_paper(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FIRM_ALLOW_TRADING", raising=False)
        res = guard_order(
            _ok_order(), _profile(), live=True,
            confirmation="CONFIRM SELL 100 SPY @ market",
            audit=tmp_path / "a.jsonl",
        )
        assert res["routed"] == "paper"

    def test_risk_breach_blocks(self, tmp_path):
        order = Order(symbol="TSLA", side="buy", qty=10, price=100.0, stop=99.0, atr=1.0)
        res = guard_order(order, _profile(), audit=tmp_path / "a.jsonl")
        assert res["routed"] == "blocked"
        assert res["breaches"]

    def test_live_with_env_and_token_routes_live(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FIRM_ALLOW_TRADING", "1")
        order = _ok_order()
        res = guard_order(
            order, _profile(), live=True,
            confirmation=order.confirmation_token(),
            audit=tmp_path / "a.jsonl",
        )
        assert res["routed"] == "live"

    def test_live_with_env_bad_token_blocks(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FIRM_ALLOW_TRADING", "1")
        res = guard_order(
            _ok_order(), _profile(), live=True, confirmation="nope",
            audit=tmp_path / "a.jsonl",
        )
        assert res["routed"] == "blocked"

    def test_audit_written(self, tmp_path):
        audit = tmp_path / "audit.jsonl"
        guard_order(_ok_order(), _profile(), audit=audit)
        assert audit.exists()
        assert audit.read_text().strip()


class TestRequireStop:
    def test_missing_stop_blocks_by_default(self, tmp_path):
        order = Order(symbol="SPY", side="buy", qty=10, price=100.0)  # no stop
        res = guard_order(order, _profile(), audit=tmp_path / "a.jsonl")
        assert res["routed"] == "blocked"
        assert any("stop" in b.lower() for b in res["breaches"])

    def test_missing_stop_allowed_when_require_stop_false(self, tmp_path):
        """The live engine rebalances to target weights and never attaches
        a protective stop to an order — require_stop=False is how it opts
        out of a check that doesn't apply to that trading style."""
        profile = _profile()
        profile.require_stop = False
        order = Order(symbol="SPY", side="buy", qty=10, price=100.0)
        res = guard_order(order, profile, audit=tmp_path / "a.jsonl")
        assert res["routed"] != "blocked"

    def test_other_breaches_still_enforced_when_require_stop_false(self, tmp_path):
        profile = _profile()
        profile.require_stop = False
        # Not on the allowlist (["SPY"]) — must still block despite require_stop=False.
        order = Order(symbol="TSLA", side="buy", qty=10, price=100.0)
        res = guard_order(order, profile, audit=tmp_path / "a.jsonl")
        assert res["routed"] == "blocked"
        assert any("allowlist" in b for b in res["breaches"])


class TestLiveSubmissionGate:
    def test_paper_broker_always_allowed(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FIRM_ALLOW_TRADING", raising=False)
        gate = guard_live_submission(
            "ibkr_paper", {"symbol": "SPY", "side": "buy"}, audit=tmp_path / "a.jsonl"
        )
        assert gate["allowed"] is True

    def test_live_broker_blocked_without_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("FIRM_ALLOW_TRADING", raising=False)
        gate = guard_live_submission(
            "ibkr_live", {"symbol": "SPY", "side": "buy"}, audit=tmp_path / "a.jsonl"
        )
        assert gate["allowed"] is False
        assert "FIRM_ALLOW_TRADING" in gate["reason"]

    def test_live_broker_allowed_with_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FIRM_ALLOW_TRADING", "1")
        gate = guard_live_submission(
            "alpaca_live", {"symbol": "SPY", "side": "buy"}, audit=tmp_path / "a.jsonl"
        )
        assert gate["allowed"] is True


class TestHelpers:
    def test_is_live_broker(self):
        assert is_live_broker("ibkr_live")
        assert is_live_broker("alpaca_live")
        assert not is_live_broker("ibkr_paper")
        assert not is_live_broker(None)

    def test_trading_armed(self, monkeypatch):
        monkeypatch.setenv("FIRM_ALLOW_TRADING", "1")
        assert trading_armed()
        monkeypatch.setenv("FIRM_ALLOW_TRADING", "0")
        assert not trading_armed()
