"""Tests for contract dataclasses – construction and immutability."""

from datetime import datetime

import pytest

from firm.contracts.models import (
    DebateResult,
    ExecutionReport,
    PortfolioSnapshot,
    RiskDecision,
    Signal,
    SignalSet,
    Thesis,
    TradeProposal,
)


class TestSignal:
    def test_construction(self) -> None:
        s = Signal(
            symbol="AAPL",
            strategy="momentum",
            score=0.75,
            confidence=0.9,
            horizon="5d",
            asof=datetime(2023, 1, 1),
        )
        assert s.symbol == "AAPL"
        assert s.score == 0.75

    def test_frozen(self) -> None:
        s = Signal(
            symbol="AAPL",
            strategy="momentum",
            score=0.5,
            confidence=0.8,
            horizon="1d",
            asof=datetime(2023, 1, 1),
        )
        with pytest.raises(AttributeError):
            s.score = 0.99  # type: ignore[misc]


class TestSignalSet:
    def test_construction(self) -> None:
        ss = SignalSet(domain="technical", asof=datetime(2023, 6, 1))
        assert ss.signals == []

    def test_frozen(self) -> None:
        ss = SignalSet(domain="technical", asof=datetime(2023, 6, 1))
        with pytest.raises(AttributeError):
            ss.domain = "fundamental"  # type: ignore[misc]


class TestThesis:
    def test_construction(self) -> None:
        t = Thesis(
            side="bull",
            symbol="MSFT",
            conviction=0.85,
            rationale="Strong cloud growth",
        )
        assert t.side == "bull"
        assert t.supporting == []


class TestDebateResult:
    def test_construction(self) -> None:
        dr = DebateResult(symbol="GOOG", net_conviction=0.3)
        assert dr.bull_thesis is None
        assert dr.bear_thesis is None


class TestTradeProposal:
    def test_construction_and_frozen(self) -> None:
        tp = TradeProposal(asof=datetime(2023, 1, 1), targets={"AAPL": 0.05})
        assert tp.targets["AAPL"] == 0.05
        with pytest.raises(AttributeError):
            tp.notes = "changed"  # type: ignore[misc]


class TestRiskDecision:
    def test_construction(self) -> None:
        rd = RiskDecision(approved=True)
        assert rd.violations == []
        assert rd.actions == []


class TestExecutionReport:
    def test_defaults(self) -> None:
        er = ExecutionReport()
        assert er.fills == []
        assert er.turnover == 0.0


class TestPortfolioSnapshot:
    def test_construction(self) -> None:
        ps = PortfolioSnapshot(
            asof=datetime(2023, 12, 31),
            holdings={"AAPL": 100.0},
            weights={"AAPL": 0.05},
            cash=9_500_000.0,
            nav=10_000_000.0,
        )
        assert ps.nav == 10_000_000.0

    def test_frozen(self) -> None:
        ps = PortfolioSnapshot(asof=datetime(2023, 12, 31))
        with pytest.raises(AttributeError):
            ps.cash = 1.0  # type: ignore[misc]
