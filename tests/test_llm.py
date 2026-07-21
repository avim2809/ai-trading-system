"""Tests for LLM-enhanced agents, mode switching, cost tracking, and API router.

All LLM calls are mocked – no real API keys required.
"""

from __future__ import annotations

import types
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from firm.agents.base import AgentContext
from firm.agents.blackboard import Blackboard
from firm.contracts.models import (
    DebateResult,
    RiskDecision,
    Signal,
    SignalSet,
    Thesis,
    TradeProposal,
)

NOW = datetime(2024, 1, 15, 16, 0)


# ── helpers ─────────────────────────────────────────────────────────

def _sig(symbol: str, strategy: str, score: float, confidence: float = 0.8) -> Signal:
    return Signal(
        symbol=symbol, strategy=strategy, score=score,
        confidence=confidence, horizon="5d", asof=NOW,
    )


def _make_signal_set(domain: str, signals: list[Signal]) -> SignalSet:
    return SignalSet(domain=domain, asof=NOW, signals=signals)


def _mock_strategy(name: str, signals: list[Signal]) -> Any:
    strat = MagicMock()
    strat.name = name
    strat.generate.return_value = signals
    return strat


class MockLLMService:
    """Fake LLMService that returns canned JSON responses."""

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.usage_stats = {"last_tokens": 42}
        self._calls: list[dict] = []

    def chat(self, messages: list[dict], **kw) -> str:
        self._calls.append({"messages": messages, **kw})
        return "LLM says hello"

    def chat_json(self, messages: list[dict], **kw) -> dict:
        self._calls.append({"messages": messages, **kw})
        return {"score": 0.75, "confidence": 0.9, "rationale": "Mock LLM analysis"}


class MockLLMServiceBroken:
    """LLMService that always raises."""

    def __init__(self, config: dict | None = None):
        self.usage_stats = {}

    def chat(self, messages, **kw):
        raise RuntimeError("LLM service down")

    def chat_json(self, messages, **kw):
        raise RuntimeError("LLM service down")


# Fake firm.llm and firm.rag modules for import patching
def _build_fake_llm_module():
    """Create a minimal fake firm.llm.provider module."""
    mod = types.ModuleType("firm.llm")
    provider_mod = types.ModuleType("firm.llm.provider")
    provider_mod.LLMService = MockLLMService  # type: ignore[attr-defined]
    compression_mod = types.ModuleType("firm.llm.compression")
    config_mod = types.ModuleType("firm.llm.config")

    class FakeCompressor:
        def __init__(self, target_ratio=0.5, use_llmlingua=False):
            pass

        def compress(self, text, target_ratio=None):
            return text[:500]

    def _fake_optimization_config(cfg=None):
        return {"compression_enabled": True, "compression_ratio": 0.5}

    compression_mod.TokenCompressor = FakeCompressor  # type: ignore[attr-defined]
    config_mod.optimization_config = _fake_optimization_config  # type: ignore[attr-defined]
    return {
        "firm.llm": mod,
        "firm.llm.provider": provider_mod,
        "firm.llm.compression": compression_mod,
        "firm.llm.config": config_mod,
    }


def _build_fake_rag_module():
    """Create minimal fake firm.rag modules."""
    rag_mod = types.ModuleType("firm.rag")
    store_mod = types.ModuleType("firm.rag.store")
    retriever_mod = types.ModuleType("firm.rag.retriever")

    class FakeDoc:
        def __init__(self, text, metadata):
            self.text = text
            self.metadata = metadata

    class FakeVectorStore:
        def __init__(self, *a, **kw):
            pass
        def stats(self):
            return {"collections": {"news": 100, "sec_filings": 50}, "total": 150}
        def delete_collection(self, name):
            pass

    class FakeRetriever:
        def __init__(self, store, **kw):
            pass
        def retrieve_for_symbol(self, symbol, query, n_results=3, collections=None, asof=None):
            return [FakeDoc(f"News about {symbol}: record earnings", {"source": "reuters"})]

    store_mod.VectorStore = FakeVectorStore  # type: ignore[attr-defined]
    retriever_mod.RAGRetriever = FakeRetriever  # type: ignore[attr-defined]
    return {
        "firm.rag": rag_mod,
        "firm.rag.store": store_mod,
        "firm.rag.retriever": retriever_mod,
    }


@pytest.fixture()
def mock_llm_modules(monkeypatch):
    """Patch sys.modules so LLM agent imports resolve to fakes."""
    import sys
    fake_modules = {**_build_fake_llm_module(), **_build_fake_rag_module()}
    for name, mod in fake_modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    yield fake_modules


# ══════════════════════════════════════════════════════════════════════
# LLMAgentMixin
# ══════════════════════════════════════════════════════════════════════
class TestLLMAgentMixin:
    def test_call_llm_logs_usage(self, mock_llm_modules):
        from firm.agents.llm.base_llm_agent import LLMAgentMixin

        mixin = LLMAgentMixin()
        result = mixin._call_llm("system", "user")
        assert result == "LLM says hello"
        assert len(mixin._llm_log) == 1
        assert mixin._llm_log[0]["tokens"] == 42

    def test_call_llm_json_mode(self, mock_llm_modules):
        from firm.agents.llm.base_llm_agent import LLMAgentMixin

        mixin = LLMAgentMixin()
        result = mixin._call_llm("system", "user", json_mode=True)
        assert isinstance(result, dict)
        assert "score" in result

    def test_retrieve_context_returns_string(self, mock_llm_modules):
        from firm.agents.llm.base_llm_agent import LLMAgentMixin

        mixin = LLMAgentMixin()
        ctx = mixin._retrieve_context("AAPL", "news")
        assert "AAPL" in ctx
        assert "reuters" in ctx

    def test_retrieve_context_handles_failure(self):
        from firm.agents.llm.base_llm_agent import LLMAgentMixin

        mixin = LLMAgentMixin()
        ctx = mixin._retrieve_context("AAPL", "news")
        assert ctx == ""

    def test_compress_with_mock(self, mock_llm_modules):
        from firm.agents.llm.base_llm_agent import LLMAgentMixin

        mixin = LLMAgentMixin()
        long_text = "x" * 5000
        result = mixin._compress(long_text)
        # Mock compressor truncates to 500
        assert len(result) <= 500

    def test_compress_disabled_passes_through_unchanged(self, mock_llm_modules, monkeypatch):
        # Target sys.modules directly rather than `import firm.llm.config as
        # x` — with no fromlist, that statement resolves via attribute-chain
        # traversal from the real top-level `firm` package, not a sys.modules
        # dict lookup. If some earlier test already really-imported firm.llm,
        # that chain reaches the *real* module and silently bypasses this
        # fixture's monkeypatched sys.modules entry.
        import sys
        from firm.agents.llm.base_llm_agent import LLMAgentMixin

        monkeypatch.setattr(
            sys.modules["firm.llm.config"], "optimization_config",
            lambda cfg=None: {"compression_enabled": False, "compression_ratio": 0.5},
        )

        mixin = LLMAgentMixin()
        long_text = "x" * 5000
        result = mixin._compress(long_text)
        assert result == long_text

    def test_compress_fallback_no_module(self, monkeypatch):
        import sys
        from firm.agents.llm.base_llm_agent import LLMAgentMixin

        # Ensure the compression module is NOT importable
        monkeypatch.delitem(sys.modules, "firm.llm.compression", raising=False)
        monkeypatch.setitem(sys.modules, "firm.llm.compression", None)

        mixin = LLMAgentMixin()
        mixin._compressor = None
        long_text = "x" * 5000
        result = mixin._compress(long_text)
        assert len(result) <= 3000


# ══════════════════════════════════════════════════════════════════════
# LLM-Enhanced Sentiment Analyst
# ══════════════════════════════════════════════════════════════════════
class TestLLMSentimentAnalyst:
    def test_returns_signal_set(self, mock_llm_modules):
        from firm.agents.llm.sentiment_analyst_llm import LLMSentimentAnalyst

        signals = [_sig("AAPL", "news", 0.3)]
        strat = _mock_strategy("news", signals)
        agent = LLMSentimentAnalyst(strategies=[strat])
        ctx = AgentContext(now=NOW, pit_view=MagicMock())

        result = agent.run(ctx)
        assert isinstance(result, SignalSet)
        assert result.domain == "sentiment"
        assert len(result.signals) == 1
        assert result.signals[0].meta.get("llm_enhanced") is True
        assert result.signals[0].score == pytest.approx(0.75)

    def test_falls_back_on_llm_failure(self, mock_llm_modules):
        from firm.agents.llm.sentiment_analyst_llm import LLMSentimentAnalyst

        signals = [_sig("AAPL", "news", 0.3)]
        strat = _mock_strategy("news", signals)
        agent = LLMSentimentAnalyst(strategies=[strat])
        agent._llm = MockLLMServiceBroken()
        ctx = AgentContext(now=NOW, pit_view=MagicMock())

        result = agent.run(ctx)
        assert isinstance(result, SignalSet)
        assert result.signals[0].score == pytest.approx(0.3)
        assert result.signals[0].meta.get("llm_enhanced") is None

    def test_no_pit_view_returns_empty(self, mock_llm_modules):
        from firm.agents.llm.sentiment_analyst_llm import LLMSentimentAnalyst

        agent = LLMSentimentAnalyst()
        ctx = AgentContext(now=NOW, pit_view=None)
        result = agent.run(ctx)
        assert result.signals == []


# ══════════════════════════════════════════════════════════════════════
# LLM-Enhanced Technical Analyst
# ══════════════════════════════════════════════════════════════════════
class TestLLMTechnicalAnalyst:
    def test_returns_signal_set(self, mock_llm_modules):
        from firm.agents.llm.technical_analyst_llm import LLMTechnicalAnalyst

        signals = [_sig("AAPL", "momentum", 1.5)]
        strat = _mock_strategy("momentum", signals)
        agent = LLMTechnicalAnalyst(strategies=[strat])
        ctx = AgentContext(now=NOW, pit_view=MagicMock())

        result = agent.run(ctx)
        assert isinstance(result, SignalSet)
        assert result.domain == "technical"
        assert result.signals[0].meta.get("llm_enhanced") is True


# ══════════════════════════════════════════════════════════════════════
# LLM-Enhanced Fundamental Analyst
# ══════════════════════════════════════════════════════════════════════
class TestLLMFundamentalAnalyst:
    def test_returns_signal_set(self, mock_llm_modules):
        from firm.agents.llm.fundamental_analyst_llm import LLMFundamentalAnalyst

        signals = [_sig("AAPL", "multi_factor", 0.8)]
        strat = _mock_strategy("multi_factor", signals)
        agent = LLMFundamentalAnalyst(strategies=[strat])
        ctx = AgentContext(now=NOW, pit_view=MagicMock())

        result = agent.run(ctx)
        assert isinstance(result, SignalSet)
        assert result.domain == "fundamental"


# ══════════════════════════════════════════════════════════════════════
# LLM-Enhanced Bull Researcher
# ══════════════════════════════════════════════════════════════════════
class TestLLMBullResearcher:
    def test_returns_theses(self, mock_llm_modules):
        from firm.agents.llm.bull_researcher_llm import LLMBullResearcher

        bb = Blackboard(asof=NOW)
        bb.signal_sets.append(_make_signal_set("technical", [_sig("AAPL", "momentum", 1.5)]))
        agent = LLMBullResearcher()
        ctx = AgentContext(now=NOW)

        # Mock _call_llm to return conviction + rationale
        agent._llm = MockLLMService()
        agent._call_llm = lambda s, u, json_mode=False: {"conviction": 0.85, "rationale": "Strong momentum with earnings beat"}

        theses = agent.run(ctx, blackboard=bb)
        assert isinstance(theses, list)
        assert all(isinstance(t, Thesis) for t in theses)
        for t in theses:
            assert t.side == "bull"

    def test_falls_back_on_failure(self, mock_llm_modules):
        from firm.agents.llm.bull_researcher_llm import LLMBullResearcher

        bb = Blackboard(asof=NOW)
        bb.signal_sets.append(_make_signal_set("technical", [_sig("AAPL", "momentum", 1.5)]))
        agent = LLMBullResearcher()
        agent._llm = MockLLMServiceBroken()
        ctx = AgentContext(now=NOW)

        theses = agent.run(ctx, blackboard=bb)
        assert len(theses) >= 1
        assert theses[0].side == "bull"


# ══════════════════════════════════════════════════════════════════════
# LLM-Enhanced Bear Researcher
# ══════════════════════════════════════════════════════════════════════
class TestLLMBearResearcher:
    def test_returns_theses(self, mock_llm_modules):
        from firm.agents.llm.bear_researcher_llm import LLMBearResearcher

        bb = Blackboard(asof=NOW)
        bb.signal_sets.append(_make_signal_set("technical", [_sig("GOOG", "momentum", -1.0)]))
        agent = LLMBearResearcher()
        ctx = AgentContext(now=NOW)

        theses = agent.run(ctx, blackboard=bb)
        assert isinstance(theses, list)
        for t in theses:
            assert t.side == "bear"


# ══════════════════════════════════════════════════════════════════════
# LLM-Enhanced Debate
# ══════════════════════════════════════════════════════════════════════
class TestLLMDebate:
    def test_returns_debate_results(self, mock_llm_modules):
        from firm.agents.llm.debate_llm import LLMDebateAgent

        bull = [Thesis(side="bull", symbol="AAPL", conviction=0.8, rationale="strong", supporting=["momentum"])]
        bear = [Thesis(side="bear", symbol="AAPL", conviction=0.3, rationale="minor", supporting=["sentiment"])]
        agent = LLMDebateAgent()

        # Mock to return adjusted conviction
        agent._llm = MockLLMService()
        agent._call_llm = lambda s, u, json_mode=False: {"net_conviction": 0.6, "reasoning": "Bull wins"}

        ctx = AgentContext(now=NOW)
        results = agent.run(ctx, bull_theses=bull, bear_theses=bear)
        assert isinstance(results, list)
        assert all(isinstance(r, DebateResult) for r in results)


# ══════════════════════════════════════════════════════════════════════
# LLM-Enhanced Trader
# ══════════════════════════════════════════════════════════════════════
class TestLLMTrader:
    def test_returns_trade_proposal(self, mock_llm_modules):
        from firm.agents.llm.trader_llm import LLMTraderAgent

        debate_results = [
            DebateResult(symbol="AAPL", net_conviction=0.6),
            DebateResult(symbol="GOOG", net_conviction=-0.4),
        ]
        agent = LLMTraderAgent()

        agent._llm = MockLLMService()
        agent._call_llm = lambda s, u, json_mode=False: {
            "adjusted_targets": {"AAPL": 0.55, "GOOG": -0.35},
            "notes": "Reduced GOOG weight slightly",
        }

        ctx = AgentContext(now=NOW)
        result = agent.run(ctx, debate_results=debate_results)
        assert isinstance(result, TradeProposal)
        assert "AAPL" in result.targets

    def test_falls_back_on_failure(self, mock_llm_modules):
        from firm.agents.llm.trader_llm import LLMTraderAgent

        debate_results = [DebateResult(symbol="AAPL", net_conviction=0.6)]
        agent = LLMTraderAgent()
        agent._llm = MockLLMServiceBroken()

        ctx = AgentContext(now=NOW)
        result = agent.run(ctx, debate_results=debate_results)
        assert isinstance(result, TradeProposal)


# ══════════════════════════════════════════════════════════════════════
# LLM-Enhanced Risk Agent
# ══════════════════════════════════════════════════════════════════════
class TestLLMRiskAgent:
    def test_returns_risk_decision(self, mock_llm_modules):
        from firm.agents.llm.risk_llm import LLMRiskAgent

        agent = LLMRiskAgent(config={"max_position_pct": 1.0})

        agent._llm = MockLLMService()
        agent._call_llm = lambda s, u, json_mode=False: {
            "additional_violations": ["Pending SEC investigation"],
            "additional_actions": ["Flag for manual review"],
            "override_approval": None,
        }

        proposal = TradeProposal(asof=NOW, targets={"AAPL": 0.05})
        ctx = AgentContext(now=NOW)
        result = agent.run(ctx, proposal=proposal)

        assert isinstance(result, RiskDecision)
        assert any("[LLM]" in v for v in result.violations)

    def test_falls_back_on_failure(self, mock_llm_modules):
        from firm.agents.llm.risk_llm import LLMRiskAgent

        agent = LLMRiskAgent(config={"max_position_pct": 1.0})
        agent._llm = MockLLMServiceBroken()

        proposal = TradeProposal(asof=NOW, targets={"AAPL": 0.05})
        ctx = AgentContext(now=NOW)
        result = agent.run(ctx, proposal=proposal)
        assert isinstance(result, RiskDecision)
        assert result.approved


# ══════════════════════════════════════════════════════════════════════
# Mode Switching in build_orchestrator
# ══════════════════════════════════════════════════════════════════════
class TestModeSwitching:
    def test_quant_mode_returns_base_agents(self):
        """Default (quant) mode should use base agent classes."""
        from firm.runtime import build_orchestrator

        orch = build_orchestrator({"agent_modes": {}})
        from firm.agents.analysts.sentiment import SentimentAnalyst

        assert any(isinstance(a, SentimentAnalyst) for a in orch.analysts)

    def test_llm_mode_wraps_agent(self, mock_llm_modules):
        """When mode is llm_enhanced, the agent should be the LLM variant."""
        from firm.runtime import build_orchestrator

        config = {
            "agent_modes": {"sentiment_analyst": "llm_enhanced"},
            "llm_config": {},
        }
        orch = build_orchestrator(config)

        from firm.agents.llm.sentiment_analyst_llm import LLMSentimentAnalyst
        sentiment = [a for a in orch.analysts if a.role == "sentiment_analyst"]
        assert len(sentiment) == 1
        assert isinstance(sentiment[0], LLMSentimentAnalyst)

    def test_llm_mode_wraps_researcher(self, mock_llm_modules):
        from firm.runtime import build_orchestrator

        config = {
            "agent_modes": {"bull_researcher": "llm_enhanced"},
            "llm_config": {},
        }
        orch = build_orchestrator(config)

        from firm.agents.llm.bull_researcher_llm import LLMBullResearcher
        assert isinstance(orch.bull, LLMBullResearcher)

    def test_llm_mode_wraps_debate(self, mock_llm_modules):
        from firm.runtime import build_orchestrator

        config = {"agent_modes": {"debate": "llm_enhanced"}, "llm_config": {}}
        orch = build_orchestrator(config)

        from firm.agents.llm.debate_llm import LLMDebateAgent
        assert isinstance(orch.debate, LLMDebateAgent)

    def test_llm_mode_wraps_trader(self, mock_llm_modules):
        from firm.runtime import build_orchestrator

        config = {"agent_modes": {"trader": "llm_enhanced"}, "llm_config": {}}
        orch = build_orchestrator(config)

        from firm.agents.llm.trader_llm import LLMTraderAgent
        assert isinstance(orch.trader, LLMTraderAgent)

    def test_llm_mode_wraps_risk(self, mock_llm_modules):
        from firm.runtime import build_orchestrator

        config = {"agent_modes": {"risk": "llm_enhanced"}, "llm_config": {}}
        orch = build_orchestrator(config)

        from firm.agents.llm.risk_llm import LLMRiskAgent
        assert isinstance(orch.risk, LLMRiskAgent)

    def test_fallback_when_import_fails(self):
        """If LLM class import fails, _maybe_wrap should return the original agent."""
        from firm.agents.analysts.sentiment import SentimentAnalyst
        from firm.runtime import _maybe_wrap

        quant_agent = SentimentAnalyst(config={})
        result = _maybe_wrap(
            quant_agent, "sentiment_analyst",
            "firm.agents.llm.nonexistent_module.FakeClass",
            {"sentiment_analyst": "llm_enhanced"}, {}, {},
        )
        assert result is quant_agent


# ══════════════════════════════════════════════════════════════════════
# Orchestrator LLM Cost Tracking
# ══════════════════════════════════════════════════════════════════════
class TestOrchestratorLLMTracking:
    def test_collects_llm_logs(self):
        from firm.agents.orchestrator import Orchestrator
        from firm.contracts.models import ExecutionReport

        mock_analyst = MagicMock()
        mock_analyst.name = "mock_analyst"
        mock_analyst._llm_log = [{"system_preview": "test", "tokens": 100}]
        mock_analyst.run.return_value = SignalSet(
            domain="technical", asof=NOW,
            signals=[_sig("AAPL", "momentum", 1.0)],
        )

        mock_bull = MagicMock()
        mock_bull.name = "bull"
        mock_bull._llm_log = [{"system_preview": "bull", "tokens": 50}]
        mock_bull.run.return_value = [
            Thesis(side="bull", symbol="AAPL", conviction=0.7, rationale="strong", supporting=["momentum"]),
        ]

        mock_bear = MagicMock()
        mock_bear.name = "bear"
        mock_bear.run.return_value = []
        # Ensure _llm_log is not present on this mock
        del mock_bear._llm_log

        mock_debate = MagicMock()
        mock_debate.name = "debate"
        mock_debate.run.return_value = [
            DebateResult(symbol="AAPL", net_conviction=0.7),
        ]

        mock_trader = MagicMock()
        mock_trader.name = "trader"
        mock_trader.run.return_value = TradeProposal(asof=NOW, targets={"AAPL": 0.04})

        mock_risk = MagicMock()
        mock_risk.name = "risk"
        mock_risk.run.return_value = RiskDecision(
            approved=True, adjusted_targets={"AAPL": 0.04},
        )

        mock_exec = MagicMock()
        mock_exec.name = "exec"
        mock_exec.run.return_value = ExecutionReport(
            fills=[{"symbol": "AAPL", "side": "buy", "quantity": 10, "strategy": "composite"}],
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
        orders, bb = orch.step({"pit_view": pit_view, "portfolio": None, "prices": {"AAPL": 150}})

        assert hasattr(bb, "llm_usage")
        assert bb.llm_usage["total_tokens"] == 150
        assert len(bb.llm_usage["calls"]) == 2
        assert bb.llm_usage["estimated_cost"] > 0


# ══════════════════════════════════════════════════════════════════════
# API Router
# ══════════════════════════════════════════════════════════════════════
class TestLLMRouter:
    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient
        from firm.api.app import create_app
        app = create_app()
        return TestClient(app)

    def test_providers_endpoint(self, client):
        resp = client.get("/api/llm/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert "providers" in data
        assert isinstance(data["providers"], list)
        assert len(data["providers"]) > 0

    def test_config_endpoint(self, client):
        resp = client.get("/api/llm/config")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_update_config(self, client, tmp_path, monkeypatch):
        import firm.api.routers.llm as llm_mod
        monkeypatch.setattr(llm_mod, "_CONFIG_PATH", tmp_path / "llm.yaml")

        resp = client.put("/api/llm/config", json={
            "agent_modes": {"sentiment_analyst": "llm_enhanced"},
            "default_model": "gpt-4o-mini",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "updated"

    def test_cache_stats_without_llm_installed(self, client):
        resp = client.get("/api/llm/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "hits" in data or "available" in data

    def test_cache_clear_without_llm_installed(self, client):
        resp = client.delete("/api/llm/cache")
        assert resp.status_code == 200

    def test_rag_stats_without_rag_installed(self, client):
        resp = client.get("/api/llm/rag/stats")
        assert resp.status_code == 200

    def test_rag_stats_reshapes_flat_counts_into_collections(self, client, monkeypatch):
        # Regression: VectorStore.stats() returns a flat {name: count,
        # "_total": n} dict — the router used to pass it through raw, which
        # had no "collections" key at all. The frontend's
        # Object.keys(ragStats.collections) then threw on any real data,
        # crashing the whole Configuration page (not just showing empty).
        import firm.rag.store as store_mod

        class _FakeStore:
            def stats(self):
                return {"sec_filings": 4205, "research": 104, "_total": 4309}

        monkeypatch.setattr(store_mod, "VectorStore", lambda *a, **k: _FakeStore())
        resp = client.get("/api/llm/rag/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "collections" in data
        assert data["collections"]["sec_filings"]["count"] == 4205
        assert data["collections"]["research"]["count"] == 104
        assert "_total" not in data["collections"]
        assert isinstance(data["collections"]["sec_filings"]["description"], str)

    def test_rag_ingest(self, client):
        resp = client.post("/api/llm/rag/ingest", json={"doc_type": "news"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ingestion_started"

    def test_test_connection_without_llm(self, client):
        resp = client.post("/api/llm/test", json={"prompt": "hi"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ("ok", "error")

    def test_test_connection_reports_response_time(self, client, monkeypatch):
        import firm.llm.provider as provider_mod

        class _FakeService:
            def __init__(self, *_a, **_k):
                pass

            def chat(self, *_a, **_k):
                return "hello there"

        monkeypatch.setattr(provider_mod, "LLMService", _FakeService)
        resp = client.post("/api/llm/test", json={"prompt": "hi"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert isinstance(data["response_time_ms"], (int, float))
        assert data["response_time_ms"] >= 0

    def test_providers_have_configured_field(self, client):
        resp = client.get("/api/llm/providers")
        data = resp.json()
        for p in data["providers"]:
            assert "configured" in p
            assert "label" in p
            assert "default_model" in p
            assert isinstance(p["configured"], bool)
