"""Tests for live pipeline warmup."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from firm.live.pipeline_warmup import PipelineWarmupGate, warm_pipeline_dependencies


def test_warmup_disabled_skips_work():
    with patch("firm.live.pipeline_warmup._warm_hmm") as mock_hmm:
        warm_pipeline_dependencies({"pipeline_warmup": False, "strategies": ["regime_hmm"]})
    mock_hmm.assert_not_called()


def test_warmup_hmm_when_regime_enabled():
    with patch("firm.live.pipeline_warmup._warm_hmm") as mock_hmm:
        warm_pipeline_dependencies({"strategies": ["momentum", "regime_hmm"]})
    mock_hmm.assert_called_once()


def test_warmup_skips_hmm_without_regime():
    with patch("firm.live.pipeline_warmup._warm_hmm") as mock_hmm:
        warm_pipeline_dependencies({"strategies": ["momentum"]})
    mock_hmm.assert_not_called()


def test_warmup_rag_when_llm_agents_enabled():
    with patch("firm.live.pipeline_warmup._warm_rag_imports") as mock_rag:
        warm_pipeline_dependencies({
            "strategies": ["momentum"],
            "agent_modes": {"fundamental_analyst": "llm_enhanced"},
        })
    mock_rag.assert_called_once()


def test_warm_hmm_probe_fits():
    from firm.live.pipeline_warmup import _warm_hmm

    with patch("firm.regime.model.GaussianRegimeModel") as mock_cls:
        mock_cls.return_value.fit = MagicMock()
        _warm_hmm()
    mock_cls.return_value.fit.assert_called_once()


def test_warmup_gate_sets_ready_when_disabled():
    gate = PipelineWarmupGate()
    gate.start_background({"pipeline_warmup": False})
    assert gate.wait_ready(timeout=1.0)


def test_warmup_gate_background_sets_ready():
    gate = PipelineWarmupGate()
    with patch("firm.live.pipeline_warmup.warm_pipeline_dependencies") as mock_warm:
        gate.start_background({"pipeline_warmup": True, "strategies": []})
        assert gate.wait_ready(timeout=2.0)
    mock_warm.assert_called_once()


def test_warmup_gate_start_is_idempotent():
    gate = PipelineWarmupGate()
    with patch("firm.live.pipeline_warmup.warm_pipeline_dependencies"):
        gate.start_background({"pipeline_warmup": True})
        gate.start_background({"pipeline_warmup": True})
        assert gate.wait_ready(timeout=2.0)
    time.sleep(0.05)
