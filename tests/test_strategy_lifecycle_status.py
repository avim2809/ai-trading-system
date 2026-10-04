"""P0-03: strategy lifecycle statuses (advisory sidecar; live behaviour neutral)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import firm.strategies  # noqa: F401
from firm.strategies import registry
from firm.strategies.registry import (
    StrategyNotAllocatableError,
    StrategyStatus,
    get,
    list_allocatable,
    list_strategies,
    require_allocatable,
    status,
    target_status,
)

ROOT = Path(__file__).resolve().parent.parent
HARD = {
    "gann", "ml_prediction", "danelfin_ai_score", "danelfin_best_stocks_signal",
    "danelfin_live_signals", "danelfin_market_percentile", "insider_cluster",
    "investing_analyst_ratings",
}
LEGACY = {
    "momentum", "trend", "mean_reversion", "stat_arb", "multi_factor", "sentiment",
    "event_driven", "volatility_breakout", "seasonality", "regime_hmm", "pattern_recognition",
}


def test_list_strategies_unchanged():
    names = list_strategies()
    assert LEGACY <= set(names) and {"gann", "ml_prediction"} <= set(names)
    assert names == list(registry._REGISTRY)


def test_hard_status_refused_in_research_entry():
    with pytest.raises(StrategyNotAllocatableError):
        require_allocatable("gann")
    require_allocatable("gann", allow_archived=True)
    require_allocatable("momentum")  # LEGACY_LIVE is warn-only
    assert status("momentum") is StrategyStatus.LEGACY_LIVE
    assert target_status("momentum") is StrategyStatus.ARCHIVED_BENCHMARKS
    assert target_status("gann") is None


def test_unknown_name_is_active():
    @registry.register("_dummy_lifecycle_test")
    class _D:  # noqa: D401
        pass

    try:
        assert status("_dummy_lifecycle_test") is StrategyStatus.ACTIVE
        assert "_dummy_lifecycle_test" in list_allocatable()
        require_allocatable("_dummy_lifecycle_test")
    finally:
        registry._REGISTRY.pop("_dummy_lifecycle_test", None)


def test_allocatable_excludes_hard_statuses():
    alloc = set(list_allocatable())
    assert not (alloc & HARD)
    assert LEGACY <= alloc


def test_empty_list_fallback_warns_and_skips(caplog):
    from firm.runtime import _build_categorized_strategies

    with caplog.at_level(logging.WARNING, logger="firm.runtime"):
        cats = _build_categorized_strategies({})
    built = {s.name for v in cats.values() for s in v}
    assert not (built & HARD)
    assert LEGACY <= built
    assert any("non-allocatable" in r.getMessage() for r in caplog.records)


def test_explicit_list_unchanged():
    from firm.runtime import _build_categorized_strategies

    cats = _build_categorized_strategies({"strategies": ["gann"]})
    assert [s.name for v in cats.values() for s in v] == ["gann"]


def test_engine_rebuild_with_empty_strategies(tmp_path, caplog):
    from firm.live.approval import ApprovalQueue
    from firm.live.data_feed import LiveDataFeed
    from firm.live.engine import LiveTradingEngine
    from tests.test_brokers import MockBroker

    broker = MockBroker()
    feed = LiveDataFeed(providers={}, universe=["AAPL"])
    config = {
        "initial_capital": 100_000,
        "memory_log_path": str(tmp_path / "decisions.jsonl"),
        "strategy_params": {},
    }
    with patch("firm.live.engine.build_orchestrator") as mb:
        mb.return_value = MagicMock()
        engine = LiveTradingEngine(
            config=config, broker=broker, data_feed=feed, approval_queue=ApprovalQueue(broker=broker)
        )
        with caplog.at_level(logging.WARNING, logger="firm.live.engine"):
            engine.update_strategies([])
    assert engine.enabled_strategies
    assert not (set(engine.enabled_strategies) & HARD)
    assert any("non-allocatable" in r.getMessage() for r in caplog.records)


def test_registry_status_table_covers_enabled():
    cfg = yaml.safe_load((ROOT / "config" / "live.yaml").read_text())
    enabled = set(cfg["strategies"]["enabled"])
    assert enabled == LEGACY
    for n in enabled:
        assert status(n) is StrategyStatus.LEGACY_LIVE
        assert target_status(n) is not None
    # every registered, non-LEGACY name in the table is a hard status
    assert {n for n in registry._STATUS if registry._STATUS[n] is not StrategyStatus.LEGACY_LIVE} == HARD


def test_skew_tolerant_imports(monkeypatch, caplog):
    """An old registry without list_allocatable must not break the fallbacks."""
    from firm.live.engine import LiveTradingEngine
    from firm.live.pipeline_warmup import _strategy_names
    from firm.runtime import _build_categorized_strategies

    monkeypatch.delattr(registry, "list_allocatable")
    cats = _build_categorized_strategies({})
    built = {s.name for v in cats.values() for s in v}
    assert LEGACY <= built and "gann" in built  # old behaviour: everything
    assert LiveTradingEngine._all_strategy_names() == list_strategies()
    assert _strategy_names({}) == set(list_strategies())


def test_deprecated_marker_registered():
    text = (ROOT / "pyproject.toml").read_text()
    assert '"deprecated:' in text


def test_deprecations_doc_has_row_per_strategy():
    text = (ROOT / "docs" / "DEPRECATIONS.md").read_text()
    for n in HARD | LEGACY:
        assert f"`{n}`" in text, n
