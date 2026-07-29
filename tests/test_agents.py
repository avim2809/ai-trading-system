"""Tests for the agent layer (Phase 2B).

Covers:
  - Blackboard: add signals, query by symbol / domain, get_all_symbols
  - Risk Manager: constraint clipping, veto on extreme violations
  - Execution Agent: correct order calculation from weight diffs
  - Orchestrator: mock agents, verify end-to-end pipeline
  - Analysts: z-scoring, signal aggregation
  - Researchers + Debate: thesis generation and synthesis
  - Trader: allocation methods
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from firm.agents.base import Agent, AgentContext
from firm.agents.blackboard import Blackboard
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

NOW = datetime(2023, 6, 15, 16, 0)


# ── helpers ─────────────────────────────────────────────────────────────
def _sig(symbol: str, strategy: str, score: float, confidence: float = 0.8) -> Signal:
    return Signal(
        symbol=symbol,
        strategy=strategy,
        score=score,
        confidence=confidence,
        horizon="5d",
        asof=NOW,
    )


def _make_signal_set(domain: str, signals: list[Signal]) -> SignalSet:
    return SignalSet(domain=domain, asof=NOW, signals=signals)


# ══════════════════════════════════════════════════════════════════════
# Blackboard
# ══════════════════════════════════════════════════════════════════════
class TestBlackboard:
    def test_empty_blackboard(self):
        bb = Blackboard(asof=NOW)
        assert bb.get_all_symbols() == set()
        assert bb.get_signals_by_symbol("AAPL") == []
        assert bb.get_signals_by_domain("technical") == []

    def test_add_signals_query_by_symbol(self):
        bb = Blackboard(asof=NOW)
        ss = _make_signal_set(
            "technical",
            [_sig("AAPL", "momentum", 1.2), _sig("GOOG", "momentum", -0.5)],
        )
        bb.signal_sets.append(ss)

        aapl = bb.get_signals_by_symbol("AAPL")
        assert len(aapl) == 1
        assert aapl[0].score == 1.2

        goog = bb.get_signals_by_symbol("GOOG")
        assert len(goog) == 1
        assert goog[0].score == -0.5

        assert bb.get_signals_by_symbol("MSFT") == []

    def test_query_by_domain(self):
        bb = Blackboard(asof=NOW)
        bb.signal_sets.append(
            _make_signal_set("technical", [_sig("AAPL", "momentum", 0.5)])
        )
        bb.signal_sets.append(
            _make_signal_set("fundamental", [_sig("AAPL", "multi_factor", 0.3)])
        )

        tech = bb.get_signals_by_domain("technical")
        assert len(tech) == 1
        assert tech[0].strategy == "momentum"

        fund = bb.get_signals_by_domain("fundamental")
        assert len(fund) == 1
        assert fund[0].strategy == "multi_factor"

    def test_get_all_symbols(self):
        bb = Blackboard(asof=NOW)
        bb.signal_sets.append(
            _make_signal_set(
                "technical",
                [_sig("AAPL", "mom", 1.0), _sig("GOOG", "mom", -0.5)],
            )
        )
        bb.signal_sets.append(
            _make_signal_set("fundamental", [_sig("MSFT", "mf", 0.2)])
        )
        assert bb.get_all_symbols() == {"AAPL", "GOOG", "MSFT"}

    def test_multi_domain_same_symbol(self):
        bb = Blackboard(asof=NOW)
        bb.signal_sets.append(
            _make_signal_set("technical", [_sig("AAPL", "momentum", 0.5)])
        )
        bb.signal_sets.append(
            _make_signal_set("sentiment", [_sig("AAPL", "news", 0.8)])
        )
        aapl = bb.get_signals_by_symbol("AAPL")
        assert len(aapl) == 2


# ══════════════════════════════════════════════════════════════════════
# Z-scoring utility
# ══════════════════════════════════════════════════════════════════════
class TestZscoring:
    def test_zscore_normalises_per_strategy(self):
        from firm.agents.analysts import zscore_signals

        raw = [
            _sig("AAPL", "momentum", 10.0),
            _sig("GOOG", "momentum", 20.0),
            _sig("MSFT", "momentum", 30.0),
        ]
        zscored = zscore_signals(raw)
        scores = sorted((s.symbol, s.score) for s in zscored)
        # mean=20, std≈8.165 → AAPL≈-1.22, GOOG≈0, MSFT≈+1.22
        assert scores[0][0] == "AAPL" and scores[0][1] < 0
        assert abs(scores[1][1]) < 0.01  # GOOG ≈ 0
        assert scores[2][0] == "MSFT" and scores[2][1] > 0

    def test_zscore_single_signal_passthrough(self):
        from firm.agents.analysts import zscore_signals

        raw = [_sig("AAPL", "momentum", 5.0)]
        assert zscore_signals(raw)[0].score == 5.0


# ══════════════════════════════════════════════════════════════════════
# Analysts
# ══════════════════════════════════════════════════════════════════════
class TestAnalysts:
    """Test analyst agents using a mock strategy."""

    @staticmethod
    def _mock_strategy(name: str, signals: list[Signal]) -> Any:
        strat = MagicMock()
        strat.name = name
        strat.generate.return_value = signals
        return strat

    def test_fundamental_analyst(self):
        from firm.agents.analysts.fundamental import FundamentalAnalyst

        signals = [_sig("AAPL", "multi_factor", 1.0), _sig("GOOG", "multi_factor", -0.5)]
        strat = self._mock_strategy("multi_factor", signals)

        analyst = FundamentalAnalyst(strategies=[strat])
        ctx = AgentContext(now=NOW, pit_view=MagicMock())
        result = analyst.run(ctx)

        assert result.domain == "fundamental"
        assert len(result.signals) == 2

    def test_technical_analyst(self):
        from firm.agents.analysts.technical import TechnicalAnalyst

        s1 = [_sig("AAPL", "momentum", 2.0), _sig("GOOG", "momentum", 1.0)]
        s2 = [_sig("AAPL", "trend", 0.5), _sig("GOOG", "trend", -0.5)]

        analyst = TechnicalAnalyst(
            strategies=[self._mock_strategy("momentum", s1), self._mock_strategy("trend", s2)]
        )
        ctx = AgentContext(now=NOW, pit_view=MagicMock())
        result = analyst.run(ctx)

        assert result.domain == "technical"
        assert len(result.signals) == 4

    def test_sentiment_analyst(self):
        from firm.agents.analysts.sentiment import SentimentAnalyst

        signals = [_sig("AAPL", "news", 0.3)]
        analyst = SentimentAnalyst(strategies=[self._mock_strategy("news", signals)])
        ctx = AgentContext(now=NOW, pit_view=MagicMock())
        result = analyst.run(ctx)

        assert result.domain == "sentiment"
        assert len(result.signals) == 1

    def test_analyst_handles_strategy_failure(self):
        from firm.agents.analysts.fundamental import FundamentalAnalyst

        broken = MagicMock()
        broken.name = "broken"
        broken.generate.side_effect = RuntimeError("boom")

        good_signals = [_sig("AAPL", "multi_factor", 1.0)]
        good = self._mock_strategy("multi_factor", good_signals)

        analyst = FundamentalAnalyst(strategies=[broken, good])
        ctx = AgentContext(now=NOW, pit_view=MagicMock())
        result = analyst.run(ctx)
        assert len(result.signals) == 1

    def test_analyst_no_pit_view(self):
        from firm.agents.analysts.fundamental import FundamentalAnalyst

        analyst = FundamentalAnalyst()
        ctx = AgentContext(now=NOW, pit_view=None)
        result = analyst.run(ctx)
        assert result.signals == []


# ══════════════════════════════════════════════════════════════════════
# Researchers + Debate
# ══════════════════════════════════════════════════════════════════════
class TestResearchers:
    @staticmethod
    def _populated_blackboard() -> Blackboard:
        bb = Blackboard(asof=NOW)
        bb.signal_sets.append(
            _make_signal_set(
                "technical",
                [
                    _sig("AAPL", "momentum", 1.5),
                    _sig("GOOG", "momentum", -1.0),
                    _sig("MSFT", "momentum", 0.3),
                ],
            )
        )
        bb.signal_sets.append(
            _make_signal_set(
                "fundamental",
                [
                    _sig("AAPL", "multi_factor", 0.8),
                    _sig("GOOG", "multi_factor", 0.2),
                    _sig("MSFT", "multi_factor", -0.7),
                ],
            )
        )
        return bb

    def test_bull_researcher(self):
        from firm.agents.research.bull import BullResearcher

        bb = self._populated_blackboard()
        bull = BullResearcher()
        ctx = AgentContext(now=NOW)
        theses = bull.run(ctx, blackboard=bb)

        symbols = {t.symbol for t in theses}
        assert "AAPL" in symbols  # strong positive signals
        for t in theses:
            assert t.side == "bull"
            assert 0.0 <= t.conviction <= 1.0

    def test_bear_researcher(self):
        from firm.agents.research.bear import BearResearcher

        bb = self._populated_blackboard()
        bear = BearResearcher()
        ctx = AgentContext(now=NOW)
        theses = bear.run(ctx, blackboard=bb)

        symbols = {t.symbol for t in theses}
        assert "GOOG" in symbols  # has negative signals
        for t in theses:
            assert t.side == "bear"
            assert 0.0 <= t.conviction <= 1.0

    def test_debate_synthesis(self):
        from firm.agents.research.debate import DebateAgent

        bull_theses = [
            Thesis(side="bull", symbol="AAPL", conviction=0.8, rationale="strong", supporting=["momentum"]),
            Thesis(side="bull", symbol="GOOG", conviction=0.2, rationale="weak", supporting=["multi_factor"]),
        ]
        bear_theses = [
            Thesis(side="bear", symbol="AAPL", conviction=0.3, rationale="minor risk", supporting=["sentiment"]),
            Thesis(side="bear", symbol="GOOG", conviction=0.6, rationale="high risk", supporting=["momentum"]),
        ]
        debate = DebateAgent()
        ctx = AgentContext(now=NOW)
        results = debate.run(ctx, bull_theses=bull_theses, bear_theses=bear_theses)

        assert len(results) == 2
        by_sym = {r.symbol: r for r in results}
        assert by_sym["AAPL"].net_conviction == pytest.approx(0.5)
        assert by_sym["GOOG"].net_conviction == pytest.approx(-0.4)
        # Sorted by net_conviction descending
        assert results[0].net_conviction >= results[1].net_conviction

    def test_bull_conviction_uses_net_signal_mass(self):
        """Regression: a lone loud positive signal with a quiet opposing one
        must not produce a saturated bull conviction; it reflects the net."""
        from firm.agents.research.bull import BullResearcher

        bb = Blackboard(asof=NOW)
        # One +3 and one -1 (equal confidence): net mean = +1.0 -> /3 ~= 0.33,
        # NOT 1.0 (which the positive-only average would have given).
        bb.signal_sets.append(
            _make_signal_set("technical", [_sig("XYZ", "momentum", 3.0)])
        )
        bb.signal_sets.append(
            _make_signal_set("fundamental", [_sig("XYZ", "multi_factor", -1.0)])
        )
        theses = BullResearcher().run(AgentContext(now=NOW), blackboard=bb)
        xyz = next(t for t in theses if t.symbol == "XYZ")
        assert xyz.conviction == pytest.approx(1.0 / 3.0, abs=0.05)

    def test_debate_one_sided(self):
        from firm.agents.research.debate import DebateAgent

        bull_only = [Thesis(side="bull", symbol="AAPL", conviction=0.9, rationale="x", supporting=[])]
        debate = DebateAgent()
        ctx = AgentContext(now=NOW)
        results = debate.run(ctx, bull_theses=bull_only, bear_theses=[])
        assert len(results) == 1
        assert results[0].net_conviction == pytest.approx(0.9)


# ══════════════════════════════════════════════════════════════════════
# Trader
# ══════════════════════════════════════════════════════════════════════
class TestTrader:
    @staticmethod
    def _debate_results() -> list[DebateResult]:
        return [
            DebateResult(symbol="AAPL", net_conviction=0.6),
            DebateResult(symbol="GOOG", net_conviction=-0.4),
            DebateResult(symbol="MSFT", net_conviction=0.2),
        ]

    def test_conviction_weighted(self):
        from firm.agents.trader import TraderAgent

        trader = TraderAgent(config={"allocation_method": "conviction_weighted"})
        ctx = AgentContext(now=NOW)
        proposal = trader.run(ctx, debate_results=self._debate_results())

        assert isinstance(proposal, TradeProposal)
        assert "AAPL" in proposal.targets
        assert proposal.targets["AAPL"] > 0  # bullish
        assert proposal.targets["GOOG"] < 0  # bearish
        total = sum(abs(w) for w in proposal.targets.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_equal_weight(self):
        from firm.agents.trader import TraderAgent

        trader = TraderAgent(config={"allocation_method": "equal_weight"})
        ctx = AgentContext(now=NOW)
        proposal = trader.run(ctx, debate_results=self._debate_results())

        weights = list(proposal.targets.values())
        assert all(abs(abs(w) - abs(weights[0])) < 1e-6 for w in weights)

    def test_max_positions(self):
        from firm.agents.trader import TraderAgent

        trader = TraderAgent(config={"max_positions": 2})
        ctx = AgentContext(now=NOW)
        proposal = trader.run(ctx, debate_results=self._debate_results())
        assert len(proposal.targets) == 2

    def test_zero_conviction_filtered(self):
        from firm.agents.trader import TraderAgent

        results = [DebateResult(symbol="AAPL", net_conviction=0.0)]
        trader = TraderAgent()
        ctx = AgentContext(now=NOW)
        proposal = trader.run(ctx, debate_results=results)
        assert len(proposal.targets) == 0

    def test_risk_parity_uses_inverse_vol(self):
        """Regression: risk_parity must inverse-vol weight, not equal-weight."""
        import pandas as pd

        from firm.agents.trader import TraderAgent

        # LO is low-vol (small moves), HI is high-vol (large moves).
        dates = pd.bdate_range("2020-01-01", periods=70)
        lo = 100 + pd.Series(range(70)) * 0.01
        hi = 100 + pd.Series(range(70)) * 0.01
        hi.iloc[::2] += 8.0  # inject large swings
        frames = []
        for sym, series in (("LO", lo), ("HI", hi)):
            frames.append(pd.DataFrame({
                "date": dates, "symbol": sym,
                "close": series.values, "adj_close": series.values,
            }))
        price_df = pd.concat(frames, ignore_index=True)

        class _PV:
            asof = NOW
            def prices(self, symbols=None, lookback_days=252):
                return price_df[price_df["symbol"].isin(symbols)]

        trader = TraderAgent(config={"allocation_method": "risk_parity"})
        ctx = AgentContext(now=NOW, pit_view=_PV())
        results = [
            DebateResult(symbol="LO", net_conviction=0.5),
            DebateResult(symbol="HI", net_conviction=0.5),
        ]
        proposal = trader.run(ctx, debate_results=results)
        # Low-vol name must receive a strictly larger allocation.
        assert abs(proposal.targets["LO"]) > abs(proposal.targets["HI"])


# ══════════════════════════════════════════════════════════════════════
# Risk Manager
# ══════════════════════════════════════════════════════════════════════
class TestRiskManager:
    def test_position_size_clipping(self):
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(config={"max_position_pct": 0.05})
        proposal = TradeProposal(
            asof=NOW,
            targets={"AAPL": 0.10, "GOOG": -0.08, "MSFT": 0.03},
        )
        ctx = AgentContext(now=NOW)
        decision = risk.run(ctx, proposal=proposal)

        assert decision.approved
        assert abs(decision.adjusted_targets["AAPL"]) <= 0.05 + 1e-9
        assert abs(decision.adjusted_targets["GOOG"]) <= 0.05 + 1e-9
        assert decision.adjusted_targets["MSFT"] == pytest.approx(0.03)
        assert len(decision.violations) >= 2

    def test_gross_exposure_cap(self):
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(config={"max_position_pct": 1.0, "max_gross_exposure": 1.0})
        proposal = TradeProposal(
            asof=NOW,
            targets={"AAPL": 0.8, "GOOG": 0.5, "MSFT": 0.7},  # gross = 2.0
        )
        ctx = AgentContext(now=NOW)
        decision = risk.run(ctx, proposal=proposal)

        gross = sum(abs(w) for w in decision.adjusted_targets.values())
        assert gross <= 1.0 + 1e-9

    def test_net_exposure_cap(self):
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(
            config={"max_position_pct": 1.0, "max_gross_exposure": 10.0, "max_net_exposure": 0.3}
        )
        proposal = TradeProposal(
            asof=NOW,
            targets={"AAPL": 0.5, "GOOG": 0.3, "MSFT": -0.1},  # net = 0.7
        )
        ctx = AgentContext(now=NOW)
        decision = risk.run(ctx, proposal=proposal)

        net = sum(decision.adjusted_targets.values())
        assert abs(net) <= 0.3 + 1e-6

    def test_veto_on_extreme_violations(self):
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(
            config={
                "max_position_pct": 0.01,
                "max_gross_exposure": 0.05,
                "veto_threshold": 0.3,
            }
        )
        proposal = TradeProposal(
            asof=NOW,
            targets={"AAPL": 0.50, "GOOG": 0.50},  # gross = 1.0, way over limits
        )
        ctx = AgentContext(now=NOW)
        decision = risk.run(ctx, proposal=proposal)

        assert not decision.approved
        assert any("VETO" in v for v in decision.violations)

    def test_drawdown_circuit_breaker(self):
        from firm.agents.risk import RiskAgent
        from firm.portfolio.state import PortfolioState

        risk = RiskAgent(config={"max_position_pct": 1.0, "max_drawdown_pct": 0.10})
        portfolio = PortfolioState(initial_capital=1_000_000)

        portfolio._history = [
            PortfolioSnapshot(asof=NOW, nav=1_000_000, cash=1_000_000),
            PortfolioSnapshot(asof=NOW, nav=850_000, cash=850_000),  # 15% dd
        ]

        proposal = TradeProposal(asof=NOW, targets={"AAPL": 0.4})
        ctx = AgentContext(now=NOW)
        decision = risk.run(ctx, proposal=proposal, portfolio=portfolio)

        assert abs(decision.adjusted_targets["AAPL"]) < 0.4
        assert any("Drawdown" in v for v in decision.violations)

    def test_no_violations_passes_cleanly(self):
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(config={"max_position_pct": 0.10, "max_gross_exposure": 2.0})
        proposal = TradeProposal(
            asof=NOW,
            targets={"AAPL": 0.05, "GOOG": -0.03},
        )
        ctx = AgentContext(now=NOW)
        decision = risk.run(ctx, proposal=proposal)

        assert decision.approved
        assert len(decision.violations) == 0
        assert decision.adjusted_targets == proposal.targets

    def test_sector_concentration(self):
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(config={"max_position_pct": 1.0, "max_sector_pct": 0.10})
        proposal = TradeProposal(
            asof=NOW,
            targets={"AAPL": 0.08, "MSFT": 0.08, "GOOG": 0.05},
        )
        sector_map = {"AAPL": "tech", "MSFT": "tech", "GOOG": "comm"}
        ctx = AgentContext(now=NOW)
        decision = risk.run(ctx, proposal=proposal, sector_map=sector_map)

        tech_total = sum(
            abs(decision.adjusted_targets[s])
            for s in ("AAPL", "MSFT")
            if s in decision.adjusted_targets
        )
        assert tech_total <= 0.10 + 1e-9

    def test_vol_targeting_never_levers_up_past_caps(self):
        """Regression: a low-vol book must not be scaled UP through the caps."""
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(config={
            "max_position_pct": 0.05,
            "max_gross_exposure": 0.20,
            "vol_target": 0.15,
        })
        proposal = TradeProposal(asof=NOW, targets={"AAPL": 0.05, "MSFT": 0.05})
        # Very low per-name vols => naive vol-targeting scale would be >> 1.
        vol_estimates = {"AAPL": 0.02, "MSFT": 0.02}
        decision = risk.run(
            AgentContext(now=NOW), proposal=proposal, vol_estimates=vol_estimates
        )
        gross = sum(abs(w) for w in decision.adjusted_targets.values())
        assert gross <= 0.20 + 1e-9
        assert all(abs(w) <= 0.05 + 1e-9 for w in decision.adjusted_targets.values())

    def test_sector_scaling_does_not_leave_net_breach(self):
        """Regression: non-uniform sector scaling must not leave net > cap."""
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(config={
            "max_position_pct": 1.0,
            "max_gross_exposure": 10.0,
            "max_net_exposure": 0.5,
            "max_sector_pct": 0.25,
        })
        # Long side spread thin across sectors; a concentrated short sector
        # gets scaled down, which would otherwise push net above 0.5.
        targets = {
            "T1": -0.20, "T2": -0.20,           # tech shorts, sector 0.40 > 0.25
            "L1": 0.20, "L2": 0.20, "L3": 0.20, "L4": 0.10,  # longs, distinct sectors
        }
        sector_map = {
            "T1": "tech", "T2": "tech",
            "L1": "a", "L2": "b", "L3": "c", "L4": "d",
        }
        proposal = TradeProposal(asof=NOW, targets=targets)
        decision = risk.run(AgentContext(now=NOW), proposal=proposal, sector_map=sector_map)
        net = sum(decision.adjusted_targets.values())
        assert abs(net) <= 0.5 + 1e-6

    def test_missing_sector_map_is_surfaced_not_silent(self):
        """Regression: skipping the sector cap must be recorded, not silent."""
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(config={"max_position_pct": 0.10})
        proposal = TradeProposal(asof=NOW, targets={"AAPL": 0.05})
        decision = risk.run(AgentContext(now=NOW), proposal=proposal)
        assert any("sector" in a.lower() for a in decision.actions)

    def test_position_clip_is_logged(self, caplog):
        """Every clip/scale/veto decision must be traceable via stdlib logging,
        not just surfaced in the returned violations/actions lists."""
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(config={"max_position_pct": 0.05})
        proposal = TradeProposal(asof=NOW, targets={"AAPL": 0.09})
        with caplog.at_level("WARNING", logger="firm.agents.risk"):
            decision = risk.run(AgentContext(now=NOW), proposal=proposal)

        assert decision.adjusted_targets["AAPL"] == pytest.approx(0.05)
        assert any("AAPL" in r.message and "clip" in r.message.lower() for r in caplog.records)

    def test_veto_decision_is_logged_at_warning(self, caplog):
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(config={
            "max_position_pct": 0.01,
            "veto_threshold": 0.05,
        })
        proposal = TradeProposal(asof=NOW, targets={"AAPL": 0.5, "MSFT": -0.5})
        with caplog.at_level("WARNING", logger="firm.agents.risk"):
            decision = risk.run(AgentContext(now=NOW), proposal=proposal)

        assert not decision.approved
        assert any("VETO" in r.message for r in caplog.records)

    def test_drawdown_breaker_is_logged(self, caplog):
        from firm.agents.risk import RiskAgent
        from firm.portfolio.state import PortfolioState

        risk = RiskAgent(config={"max_position_pct": 1.0, "max_drawdown_pct": 0.10})
        portfolio = PortfolioState(initial_capital=100_000)
        portfolio._history = [
            PortfolioSnapshot(asof=NOW, nav=100_000, cash=100_000),
            PortfolioSnapshot(asof=NOW, nav=80_000, cash=80_000),  # 20% dd
        ]
        proposal = TradeProposal(asof=NOW, targets={"AAPL": 0.10})
        with caplog.at_level("WARNING", logger="firm.agents.risk"):
            decision = risk.run(AgentContext(now=NOW), proposal=proposal, portfolio=portfolio)

        assert decision.adjusted_targets["AAPL"] == pytest.approx(0.05)
        assert any("drawdown" in r.message.lower() for r in caplog.records)

    def test_liquidity_check_clips_trade_exceeding_adv_participation(self, caplog):
        """A target weight implying a trade > max_participation_pct of ADV
        must be clipped, even though it's well within the per-name cap."""
        from firm.agents.risk import RiskAgent

        class _ThinPitView:
            asof = NOW
            universe = ["THIN"]

            def prices(self, symbols=None, lookback_days=20):
                import pandas as pd
                # $1/share * 1,000 shares/day = $1,000 ADV.
                return pd.DataFrame({
                    "symbol": ["THIN"] * 5,
                    "close": [1.0] * 5,
                    "volume": [1_000] * 5,
                })

        class _Portfolio:
            nav = 100_000.0

            def get_weights(self, prices):
                return {}

        risk = RiskAgent(config={
            "max_position_pct": 1.0,
            "max_participation_pct": 0.10,
        })
        # Target 1% of $100k NAV = $1,000 trade vs $1,000 ADV -> 100%
        # participation, way past the 10% cap.
        proposal = TradeProposal(asof=NOW, targets={"THIN": 0.01})
        ctx = AgentContext(now=NOW, pit_view=_ThinPitView())
        with caplog.at_level("WARNING", logger="firm.agents.risk"):
            decision = risk.run(ctx, proposal=proposal, portfolio=_Portfolio())

        assert decision.adjusted_targets["THIN"] < 0.01
        implied_trade = decision.adjusted_targets["THIN"] * 100_000.0
        assert implied_trade <= 100.0 + 1e-6  # 10% of $1,000 ADV
        assert any("ADV" in r.message for r in caplog.records)

    def test_liquidity_check_noop_without_pit_view(self):
        """No pit_view (most backtests/tests) means the check can't run — it
        must not raise or otherwise block the proposal."""
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(config={"max_position_pct": 1.0, "max_participation_pct": 0.10})
        proposal = TradeProposal(asof=NOW, targets={"AAPL": 0.05})
        decision = risk.run(AgentContext(now=NOW), proposal=proposal)
        assert decision.approved
        assert decision.adjusted_targets["AAPL"] == pytest.approx(0.05)

    def test_liquidity_check_disabled_by_default(self):
        """max_participation_pct is opt-in; omitting it must not change
        existing behaviour even with a pit_view present."""
        from firm.agents.risk import RiskAgent

        class _ThinPitView:
            asof = NOW
            universe = ["THIN"]

            def prices(self, symbols=None, lookback_days=20):
                import pandas as pd
                return pd.DataFrame({
                    "symbol": ["THIN"], "close": [1.0], "volume": [1_000],
                })

        risk = RiskAgent(config={"max_position_pct": 1.0})
        proposal = TradeProposal(asof=NOW, targets={"THIN": 0.5})
        ctx = AgentContext(now=NOW, pit_view=_ThinPitView())
        decision = risk.run(ctx, proposal=proposal)
        assert decision.adjusted_targets["THIN"] == pytest.approx(0.5)

    @staticmethod
    def _correlated_pit_view(n_days: int = 30):
        """Two symbols (A, B) that move in lockstep (rho ~= 1.0) and a
        third (C) with independent, uncorrelated returns."""
        import pandas as pd

        dates = pd.bdate_range("2023-01-01", periods=n_days)
        rows = []
        price_a, price_c = 100.0, 100.0
        rng_returns = [0.01, -0.02, 0.015, -0.005, 0.02] * (n_days // 5 + 1)
        for i, d in enumerate(dates):
            ret_a = rng_returns[i]
            ret_c = rng_returns[(i + 2) % len(rng_returns)] * -1.0
            price_a *= 1 + ret_a
            price_c *= 1 + ret_c
            for sym, price in (("A", price_a), ("B", price_a), ("C", price_c)):
                rows.append({"date": d, "symbol": sym, "close": price, "volume": 1_000_000})
        return pd.DataFrame(rows)

    def test_correlated_pair_is_scaled_down(self, caplog):
        from firm.agents.risk import RiskAgent

        class _View:
            asof = NOW
            universe = ["A", "B", "C"]

            def prices(self, symbols=None, lookback_days=60):
                return TestRiskManager._correlated_pit_view()

        risk = RiskAgent(config={
            "max_position_pct": 1.0,
            "correlation_threshold": 0.9,
            "max_correlated_pair_pct": 0.10,
        })
        # A and B are identical (rho=1.0, same direction) with combined
        # weight 0.30 > the 0.10 cap; C is uncorrelated and untouched.
        proposal = TradeProposal(asof=NOW, targets={"A": 0.15, "B": 0.15, "C": 0.15})
        ctx = AgentContext(now=NOW, pit_view=_View())
        with caplog.at_level("WARNING", logger="firm.agents.risk"):
            decision = risk.run(ctx, proposal=proposal)

        combined_ab = abs(decision.adjusted_targets["A"]) + abs(decision.adjusted_targets["B"])
        assert combined_ab <= 0.10 + 1e-6
        assert decision.adjusted_targets["C"] == pytest.approx(0.15)
        assert any("correlated" in r.message.lower() for r in caplog.records)

    def test_offsetting_correlated_pair_is_not_capped(self):
        """A long/short pair in correlated names is a hedge, not
        concentration, and must not be scaled down."""
        from firm.agents.risk import RiskAgent

        class _View:
            asof = NOW
            universe = ["A", "B"]

            def prices(self, symbols=None, lookback_days=60):
                return TestRiskManager._correlated_pit_view()

        risk = RiskAgent(config={
            "max_position_pct": 1.0,
            "correlation_threshold": 0.9,
            "max_correlated_pair_pct": 0.10,
        })
        proposal = TradeProposal(asof=NOW, targets={"A": 0.15, "B": -0.15})
        ctx = AgentContext(now=NOW, pit_view=_View())
        decision = risk.run(ctx, proposal=proposal)
        assert decision.adjusted_targets["A"] == pytest.approx(0.15)
        assert decision.adjusted_targets["B"] == pytest.approx(-0.15)

    def test_correlation_check_disabled_by_default(self):
        from firm.agents.risk import RiskAgent

        class _View:
            asof = NOW
            universe = ["A", "B"]

            def prices(self, symbols=None, lookback_days=60):
                return TestRiskManager._correlated_pit_view()

        risk = RiskAgent(config={"max_position_pct": 1.0, "max_net_exposure": 1.0})
        proposal = TradeProposal(asof=NOW, targets={"A": 0.3, "B": 0.3})
        ctx = AgentContext(now=NOW, pit_view=_View())
        decision = risk.run(ctx, proposal=proposal)
        assert decision.adjusted_targets["A"] == pytest.approx(0.3)
        assert decision.adjusted_targets["B"] == pytest.approx(0.3)

    @staticmethod
    def _fat_tailed_pit_view(n_days: int = 80, tail_days: tuple = (10, 30, 50)):
        """One symbol with small, varied day-to-day returns (no ties, so the
        tail-percentile cutoff isn't washed out by a run of identical
        "ordinary" values) plus a few sharp -8% drops — a fat left tail that
        inflates CVaR without inflating variance-based vol estimates nearly
        as much."""
        import pandas as pd

        pattern = [0.0008, -0.0006, 0.0011, -0.0009, 0.0007, -0.0004, 0.0013, -0.0011]
        dates = pd.bdate_range("2023-01-01", periods=n_days)
        rows = []
        price = 100.0
        for i, d in enumerate(dates):
            ret = -0.08 if i in tail_days else pattern[i % len(pattern)]
            price *= 1 + ret
            rows.append({"date": d, "symbol": "TAIL", "close": price, "volume": 1_000_000})
        return pd.DataFrame(rows)

    @staticmethod
    def _thin_tailed_pit_view(n_days: int = 80):
        """One symbol with small, smooth oscillating returns — no fat tail."""
        import pandas as pd

        dates = pd.bdate_range("2023-01-01", periods=n_days)
        rows = []
        price = 100.0
        pattern = [0.001, -0.001, 0.0015, -0.0012, 0.0008]
        for i, d in enumerate(dates):
            price *= 1 + pattern[i % len(pattern)]
            rows.append({"date": d, "symbol": "STABLE", "close": price, "volume": 1_000_000})
        return pd.DataFrame(rows)

    def test_cvar_disabled_by_default(self):
        from firm.agents.risk import RiskAgent

        class _View:
            asof = NOW
            universe = ["TAIL"]

            def prices(self, symbols=None, lookback_days=60):
                return TestRiskManager._fat_tailed_pit_view()

        risk = RiskAgent(config={"max_position_pct": 1.0})
        proposal = TradeProposal(asof=NOW, targets={"TAIL": 0.3})
        ctx = AgentContext(now=NOW, pit_view=_View())
        decision = risk.run(ctx, proposal=proposal)
        assert decision.adjusted_targets["TAIL"] == pytest.approx(0.3)

    def test_fat_tailed_book_is_scaled_down(self, caplog):
        from firm.agents.risk import RiskAgent

        class _View:
            asof = NOW
            universe = ["TAIL"]

            def prices(self, symbols=None, lookback_days=60):
                return TestRiskManager._fat_tailed_pit_view()

        risk = RiskAgent(config={
            "max_position_pct": 1.0,
            "cvar_limit": 0.01,
            "cvar_confidence": 0.95,
            "cvar_lookback_days": 60,
        })
        proposal = TradeProposal(asof=NOW, targets={"TAIL": 0.3})
        ctx = AgentContext(now=NOW, pit_view=_View())
        with caplog.at_level("WARNING", logger="firm.agents.risk"):
            decision = risk.run(ctx, proposal=proposal)
        assert abs(decision.adjusted_targets["TAIL"]) < 0.3
        assert any("cvar" in r.message.lower() for r in caplog.records)

    def test_thin_tailed_book_not_scaled(self):
        from firm.agents.risk import RiskAgent

        class _View:
            asof = NOW
            universe = ["STABLE"]

            def prices(self, symbols=None, lookback_days=60):
                return TestRiskManager._thin_tailed_pit_view()

        risk = RiskAgent(config={"max_position_pct": 1.0, "cvar_limit": 0.05})
        proposal = TradeProposal(asof=NOW, targets={"STABLE": 0.3})
        ctx = AgentContext(now=NOW, pit_view=_View())
        decision = risk.run(ctx, proposal=proposal)
        assert decision.adjusted_targets["STABLE"] == pytest.approx(0.3)

    def test_cvar_noop_without_pit_view(self):
        from firm.agents.risk import RiskAgent

        risk = RiskAgent(config={"max_position_pct": 1.0, "cvar_limit": 0.01})
        proposal = TradeProposal(asof=NOW, targets={"TAIL": 0.3})
        ctx = AgentContext(now=NOW, pit_view=None)
        decision = risk.run(ctx, proposal=proposal)
        assert decision.adjusted_targets["TAIL"] == pytest.approx(0.3)


# ══════════════════════════════════════════════════════════════════════
# Execution Agent
# ══════════════════════════════════════════════════════════════════════
class TestExecution:
    def test_order_from_empty_portfolio(self):
        from firm.agents.execution import ExecutionAgent
        from firm.portfolio.state import PortfolioState

        execution = ExecutionAgent(config={"commission_pct": 0.001, "slippage_pct": 0.0})
        portfolio = PortfolioState(initial_capital=1_000_000)
        prices = {"AAPL": 150.0, "GOOG": 2800.0}

        decision = RiskDecision(
            approved=True,
            adjusted_targets={"AAPL": 0.6, "GOOG": 0.3},
        )
        ctx = AgentContext(now=NOW)
        report = execution.run(ctx, decision=decision, portfolio=portfolio, prices=prices)

        assert isinstance(report, ExecutionReport)
        assert len(report.fills) == 2
        by_sym = {o["symbol"]: o for o in report.fills}
        assert by_sym["AAPL"]["side"] == "buy"
        assert by_sym["GOOG"]["side"] == "buy"
        assert by_sym["AAPL"]["quantity"] == pytest.approx(600_000 / 150.0, rel=0.01)
        assert report.turnover == pytest.approx(0.9)
        assert report.costs > 0

    def test_order_diff_from_existing_holdings(self):
        from firm.agents.execution import ExecutionAgent
        from firm.portfolio.state import PortfolioState

        execution = ExecutionAgent()
        portfolio = PortfolioState(initial_capital=500_000)
        portfolio.holdings = {"AAPL": 1000}  # 1000 shares at $150 = $150k
        prices = {"AAPL": 150.0, "GOOG": 100.0}

        # NAV = 500_000 + 1000*150 = 650_000
        # current AAPL weight = 150_000/650_000 ≈ 0.2308
        # target AAPL weight = 0.10 → need to sell
        decision = RiskDecision(
            approved=True,
            adjusted_targets={"AAPL": 0.10},
        )
        ctx = AgentContext(now=NOW)
        report = execution.run(ctx, decision=decision, portfolio=portfolio, prices=prices)

        aapl_orders = [o for o in report.fills if o["symbol"] == "AAPL"]
        assert len(aapl_orders) == 1
        assert aapl_orders[0]["side"] == "sell"

    def test_no_orders_when_on_target(self):
        from firm.agents.execution import ExecutionAgent
        from firm.portfolio.state import PortfolioState

        execution = ExecutionAgent()
        portfolio = PortfolioState(initial_capital=0)
        portfolio.cash = 500_000
        portfolio.holdings = {"AAPL": 1000}
        prices = {"AAPL": 500.0}
        # NAV = 500_000 + 500_000 = 1_000_000, AAPL weight = 0.5

        decision = RiskDecision(approved=True, adjusted_targets={"AAPL": 0.5})
        ctx = AgentContext(now=NOW)
        report = execution.run(ctx, decision=decision, portfolio=portfolio, prices=prices)
        assert len(report.fills) == 0
        assert report.turnover == pytest.approx(0.0)

    def test_fills_carry_signed_shares(self):
        """Regression: orders must expose a signed ``shares`` field matching
        ``side`` so downstream consumers route sells correctly."""
        from firm.agents.execution import ExecutionAgent
        from firm.portfolio.state import PortfolioState

        execution = ExecutionAgent()
        portfolio = PortfolioState(initial_capital=500_000)
        portfolio.holdings = {"AAPL": 1000}  # ~0.23 weight, target 0.10 -> sell
        prices = {"AAPL": 150.0, "GOOG": 100.0}
        decision = RiskDecision(
            approved=True, adjusted_targets={"AAPL": 0.10, "GOOG": 0.20}
        )
        report = execution.run(
            AgentContext(now=NOW), decision=decision, portfolio=portfolio, prices=prices
        )
        by_sym = {o["symbol"]: o for o in report.fills}
        # AAPL is a sell -> negative shares, positive quantity
        assert by_sym["AAPL"]["side"] == "sell"
        assert by_sym["AAPL"]["shares"] < 0
        assert by_sym["AAPL"]["quantity"] == pytest.approx(abs(by_sym["AAPL"]["shares"]))
        # GOOG is a buy -> positive shares
        assert by_sym["GOOG"]["side"] == "buy"
        assert by_sym["GOOG"]["shares"] > 0

    def test_fills_tagged_with_originating_strategy(self):
        """Regression: orders carry the dominant contributing strategy, not a
        hardcoded 'composite', so per-strategy approval routing works."""
        from firm.agents.execution import ExecutionAgent

        execution = ExecutionAgent()
        decision = RiskDecision(approved=True, adjusted_targets={"AAPL": 0.5, "MSFT": 0.5})
        per_strategy = {
            "momentum": {"AAPL": 0.4, "MSFT": 0.1},
            "trend": {"AAPL": 0.1, "MSFT": 0.4},
        }
        report = execution.run(
            AgentContext(now=NOW),
            decision=decision,
            prices={"AAPL": 100.0, "MSFT": 100.0},
            per_strategy=per_strategy,
        )
        by_sym = {o["symbol"]: o for o in report.fills}
        assert by_sym["AAPL"]["strategy"] == "momentum"
        assert by_sym["MSFT"]["strategy"] == "trend"

    def test_closing_order_attributed_via_held_position_not_composite(self):
        """Regression: a symbol dropped entirely from this cycle's targets
        (fully closed) has no *this-cycle* per_strategy entry, but should
        still be attributed to whichever strategy currently holds the
        position (via PerformanceAttribution), not silently dumped into
        'composite'."""
        from firm.agents.execution import ExecutionAgent
        from firm.portfolio.attribution import PerformanceAttribution
        from firm.portfolio.state import PortfolioState

        attribution = PerformanceAttribution()
        # Seed prior holdings as if 'momentum' opened the AAPL position.
        attribution.record_trades(
            [{"symbol": "AAPL", "shares": 100, "price": 100.0, "strategy": "momentum"}],
            {"AAPL": 100.0},
        )

        execution = ExecutionAgent()
        portfolio = PortfolioState(initial_capital=500_000)
        portfolio.holdings = {"AAPL": 100}
        prices = {"AAPL": 150.0}
        # AAPL fully closed this cycle: absent from targets and per_strategy.
        decision = RiskDecision(approved=True, adjusted_targets={})
        report = execution.run(
            AgentContext(now=NOW),
            decision=decision,
            portfolio=portfolio,
            prices=prices,
            per_strategy={},
            attribution=attribution,
        )
        by_sym = {o["symbol"]: o for o in report.fills}
        assert by_sym["AAPL"]["side"] == "sell"
        assert by_sym["AAPL"]["strategy"] == "momentum"

    def test_closing_order_falls_back_to_composite_when_untraceable(self):
        """Without attribution history (or with no matching held position),
        an untraceable close still falls back to 'composite' rather than
        raising — but this should be the rare/last-resort path."""
        from firm.agents.execution import ExecutionAgent
        from firm.portfolio.state import PortfolioState

        execution = ExecutionAgent()
        portfolio = PortfolioState(initial_capital=500_000)
        portfolio.holdings = {"AAPL": 100}
        prices = {"AAPL": 150.0}
        decision = RiskDecision(approved=True, adjusted_targets={})
        report = execution.run(
            AgentContext(now=NOW),
            decision=decision,
            portfolio=portfolio,
            prices=prices,
        )
        by_sym = {o["symbol"]: o for o in report.fills}
        assert by_sym["AAPL"]["strategy"] == "composite"

    def test_portfolio_nav_marks_to_market(self):
        """Regression: nav must value holdings at price, not sum share counts."""
        from firm.portfolio.state import PortfolioState

        p = PortfolioState(initial_capital=0)
        p.cash = 1000.0
        p.holdings = {"AAPL": 10}  # 10 shares
        # Before any prices seen: holdings contribute 0 (cash only), never 10.
        assert p.nav == pytest.approx(1000.0)
        # After marking at $150, nav reflects market value (1000 + 1500).
        p.get_weights({"AAPL": 150.0})
        assert p.nav == pytest.approx(2500.0)

    def test_fills_consumed_by_portfolio_and_attribution(self):
        """Regression: execution fills feed PortfolioState.update and
        PerformanceAttribution.record_trades without KeyError, and a sell
        actually reduces holdings (it used to route as a buy)."""
        from firm.agents.execution import ExecutionAgent
        from firm.portfolio.attribution import PerformanceAttribution
        from firm.portfolio.state import PortfolioState

        execution = ExecutionAgent()
        portfolio = PortfolioState(initial_capital=500_000)
        portfolio.holdings = {"AAPL": 1000}
        prices = {"AAPL": 150.0}
        decision = RiskDecision(approved=True, adjusted_targets={"AAPL": 0.10})
        report = execution.run(
            AgentContext(now=NOW), decision=decision, portfolio=portfolio, prices=prices
        )

        before = portfolio.holdings["AAPL"]
        portfolio.update(report.fills, prices)  # must not raise KeyError
        attribution = PerformanceAttribution()
        attribution.record_trades(report.fills, prices)  # must not raise KeyError

        # A sell reduces the position rather than increasing it.
        assert portfolio.holdings.get("AAPL", 0.0) < before

    def test_update_deducts_transaction_cost(self):
        """Transaction cost is charged to cash on top of the traded notional,
        and the default (cost=0) leaves the pre-cost behaviour unchanged."""
        from firm.portfolio.state import PortfolioState

        fills = [{"symbol": "AAPL", "shares": 100, "price": 50.0, "strategy": "m"}]
        prices = {"AAPL": 50.0}

        # Without cost: cash drops by exactly shares*price.
        p0 = PortfolioState(initial_capital=100_000)
        p0.update(fills, prices)
        assert p0.cash == pytest.approx(100_000 - 100 * 50.0)

        # With cost: cash drops by shares*price + cost.
        p1 = PortfolioState(initial_capital=100_000)
        p1.update(fills, prices, cost=25.0)
        assert p1.cash == pytest.approx(100_000 - 100 * 50.0 - 25.0)
        # The cost is charged back to the originating strategy's ledger.
        assert p1.get_strategy_pnl("m") == pytest.approx(-(100 * 50.0) - 25.0)


# ══════════════════════════════════════════════════════════════════════
# Orchestrator (end-to-end with mocks)
# ══════════════════════════════════════════════════════════════════════
class TestOrchestrator:
    @staticmethod
    def _build_orchestrator(
        risk_approves: bool = True,
        risk_violations: list[str] | None = None,
    ):
        from firm.agents.orchestrator import Orchestrator

        mock_analyst = MagicMock(spec=Agent)
        mock_analyst.name = "mock_analyst"
        mock_analyst.run.return_value = SignalSet(
            domain="technical",
            asof=NOW,
            signals=[_sig("AAPL", "momentum", 1.0), _sig("GOOG", "momentum", -0.5)],
        )

        mock_bull = MagicMock(spec=Agent)
        mock_bull.name = "bull"
        mock_bull.run.return_value = [
            Thesis(side="bull", symbol="AAPL", conviction=0.7, rationale="x", supporting=["momentum"]),
        ]

        mock_bear = MagicMock(spec=Agent)
        mock_bear.name = "bear"
        mock_bear.run.return_value = [
            Thesis(side="bear", symbol="GOOG", conviction=0.5, rationale="y", supporting=["momentum"]),
        ]

        mock_debate = MagicMock(spec=Agent)
        mock_debate.name = "debate"
        mock_debate.run.return_value = [
            DebateResult(symbol="AAPL", net_conviction=0.7),
            DebateResult(symbol="GOOG", net_conviction=-0.5),
        ]

        mock_trader = MagicMock(spec=Agent)
        mock_trader.name = "trader"
        mock_trader.run.return_value = TradeProposal(
            asof=NOW,
            targets={"AAPL": 0.04, "GOOG": -0.03},
        )

        mock_risk = MagicMock(spec=Agent)
        mock_risk.name = "risk"
        mock_risk.run.return_value = RiskDecision(
            approved=risk_approves,
            adjusted_targets={"AAPL": 0.04, "GOOG": -0.03},
            violations=risk_violations or [],
        )

        mock_exec = MagicMock(spec=Agent)
        mock_exec.name = "execution"
        mock_exec.run.return_value = ExecutionReport(
            fills=[
                {"symbol": "AAPL", "side": "buy", "quantity": 100, "strategy": "composite"},
                {"symbol": "GOOG", "side": "sell", "quantity": 50, "strategy": "composite"},
            ],
            turnover=0.07,
            costs=15.0,
        )

        pit_view = MagicMock()
        pit_view.asof = NOW

        orch = Orchestrator(
            analysts=[mock_analyst],
            bull=mock_bull,
            bear=mock_bear,
            debate=mock_debate,
            trader=mock_trader,
            risk=mock_risk,
            execution=mock_exec,
        )
        return orch, pit_view

    def test_pipeline_end_to_end(self):
        orch, pit_view = self._build_orchestrator()
        orders, bb = orch.step({"pit_view": pit_view, "portfolio": None, "prices": {"AAPL": 150, "GOOG": 100}})

        assert len(orders) == 2
        assert isinstance(bb, Blackboard)
        assert len(bb.signal_sets) == 1
        assert bb.proposal is not None
        assert bb.risk_decision is not None
        assert bb.risk_decision.approved
        assert bb.execution_report is not None

    def test_pipeline_risk_veto_returns_empty(self):
        orch, pit_view = self._build_orchestrator(risk_approves=False, risk_violations=["Too risky"])
        orders, bb = orch.step({"pit_view": pit_view, "portfolio": None, "prices": {}})

        assert orders == []
        assert not bb.risk_decision.approved

    def test_orchestrator_via_run(self):
        orch, pit_view = self._build_orchestrator()
        ctx = AgentContext(now=NOW, pit_view=pit_view)
        orders, bb = orch.run(ctx, prices={"AAPL": 150, "GOOG": 100})

        assert len(orders) == 2
        assert isinstance(bb, Blackboard)

    def test_blackboard_timestamps_match(self):
        orch, pit_view = self._build_orchestrator()
        _, bb = orch.step({"pit_view": pit_view, "portfolio": None, "prices": {}})
        assert bb.asof == NOW

    def test_analyst_failure_handled_gracefully(self):
        from firm.agents.orchestrator import Orchestrator

        failing = MagicMock(spec=Agent)
        failing.name = "failing_analyst"
        failing.run.side_effect = RuntimeError("boom")

        good = MagicMock(spec=Agent)
        good.name = "good_analyst"
        good.run.return_value = SignalSet(
            domain="technical", asof=NOW,
            signals=[_sig("AAPL", "momentum", 1.0)],
        )

        orch = Orchestrator(
            analysts=[failing, good],
            bull=MagicMock(spec=Agent, **{"name": "bull", "run.return_value": []}),
            bear=MagicMock(spec=Agent, **{"name": "bear", "run.return_value": []}),
            debate=MagicMock(spec=Agent, **{"name": "debate", "run.return_value": []}),
            trader=MagicMock(spec=Agent, **{"name": "trader", "run.return_value": TradeProposal(asof=NOW)}),
            risk=MagicMock(spec=Agent, **{"name": "risk", "run.return_value": RiskDecision(approved=True)}),
            execution=MagicMock(spec=Agent, **{"name": "exec", "run.return_value": ExecutionReport()}),
        )
        pit_view = MagicMock()
        pit_view.asof = NOW
        orders, bb = orch.step({"pit_view": pit_view, "portfolio": None, "prices": {}})

        assert len(bb.signal_sets) == 1
        assert bb.signal_sets[0].domain == "technical"
        # Regression: the failure is no longer silent.
        assert bb.degraded is True
        assert any(e.get("agent") == "failing_analyst" for e in bb.errors)

    def test_abort_on_degraded_returns_empty(self):
        from firm.agents.orchestrator import Orchestrator

        failing = MagicMock(spec=Agent)
        failing.name = "failing_analyst"
        failing.run.side_effect = RuntimeError("boom")
        good = MagicMock(spec=Agent)
        good.name = "good_analyst"
        good.run.return_value = SignalSet(
            domain="technical", asof=NOW, signals=[_sig("AAPL", "momentum", 1.0)],
        )
        orch = Orchestrator(
            analysts=[failing, good],
            bull=MagicMock(spec=Agent, **{"name": "bull", "run.return_value": []}),
            bear=MagicMock(spec=Agent, **{"name": "bear", "run.return_value": []}),
            debate=MagicMock(spec=Agent, **{"name": "debate", "run.return_value": []}),
            trader=MagicMock(spec=Agent, **{"name": "trader", "run.return_value": TradeProposal(asof=NOW)}),
            risk=MagicMock(spec=Agent, **{"name": "risk", "run.return_value": RiskDecision(approved=True)}),
            execution=MagicMock(spec=Agent, **{"name": "exec", "run.return_value": ExecutionReport()}),
            config={"abort_on_degraded": True},
        )
        pit_view = MagicMock()
        pit_view.asof = NOW
        orders, bb = orch.step({"pit_view": pit_view, "portfolio": None, "prices": {}})
        assert orders == []
        assert bb.degraded is True

    def test_analyst_timeout_records_degraded(self):
        import time

        from firm.agents.orchestrator import Orchestrator

        slow = MagicMock(spec=Agent)
        slow.name = "slow_analyst"
        slow.run.side_effect = lambda *_a, **_k: time.sleep(2) or SignalSet(
            domain="technical", asof=NOW, signals=[],
        )
        fast = MagicMock(spec=Agent)
        fast.name = "fast_analyst"
        fast.run.return_value = SignalSet(
            domain="fundamental", asof=NOW, signals=[_sig("AAPL", "multi_factor", 1.0)],
        )
        orch = Orchestrator(
            analysts=[slow, fast],
            bull=MagicMock(spec=Agent, **{"name": "bull", "run.return_value": []}),
            bear=MagicMock(spec=Agent, **{"name": "bear", "run.return_value": []}),
            debate=MagicMock(spec=Agent, **{"name": "debate", "run.return_value": []}),
            trader=MagicMock(spec=Agent, **{"name": "trader", "run.return_value": TradeProposal(asof=NOW)}),
            risk=MagicMock(spec=Agent, **{"name": "risk", "run.return_value": RiskDecision(approved=True)}),
            execution=MagicMock(spec=Agent, **{"name": "exec", "run.return_value": ExecutionReport()}),
            config={"analyst_timeout_seconds": 0.2},
        )
        pit_view = MagicMock()
        pit_view.asof = NOW
        _orders, bb = orch.step({"pit_view": pit_view, "portfolio": None, "prices": {}})

        assert bb.degraded is True
        assert any(
            e.get("agent") == "slow_analyst" and "timed out" in e.get("error", "")
            for e in bb.errors
        )
        assert any(ss.domain == "fundamental" for ss in bb.signal_sets)

    def test_stage_timeout_records_degraded(self):
        import time

        from firm.agents.orchestrator import Orchestrator

        mock_analyst = MagicMock(spec=Agent)
        mock_analyst.name = "mock_analyst"
        mock_analyst.run.return_value = SignalSet(
            domain="technical",
            asof=NOW,
            signals=[_sig("AAPL", "momentum", 1.0)],
        )
        slow_debate = MagicMock(spec=Agent)
        slow_debate.name = "debate"
        slow_debate.run.side_effect = lambda *_a, **_k: time.sleep(2)
        orch = Orchestrator(
            analysts=[mock_analyst],
            bull=MagicMock(spec=Agent, **{"name": "bull", "run.return_value": []}),
            bear=MagicMock(spec=Agent, **{"name": "bear", "run.return_value": []}),
            debate=slow_debate,
            trader=MagicMock(spec=Agent, **{"name": "trader", "run.return_value": TradeProposal(asof=NOW)}),
            risk=MagicMock(spec=Agent, **{"name": "risk", "run.return_value": RiskDecision(approved=True)}),
            execution=MagicMock(spec=Agent, **{"name": "exec", "run.return_value": ExecutionReport()}),
            config={"orchestrator_stage_timeout_seconds": 0.2},
        )
        pit_view = MagicMock()
        pit_view.asof = NOW
        _orders, bb = orch.step({"pit_view": pit_view, "portfolio": None, "prices": {}})

        assert bb.degraded is True
        assert any(
            e.get("agent") == "debate" and "timed out" in e.get("error", "")
            for e in bb.errors
        )
