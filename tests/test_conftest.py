"""Tests for the session-wide fixtures in conftest.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import firm.live.pipeline_warmup as pipeline_warmup
from firm.live.pipeline_warmup import PipelineWarmupGate


def test_pipeline_warmup_is_mocked_by_default():
    """The autouse fixture in conftest.py must have replaced _warm_hmm with
    a mock for any test outside test_pipeline_warmup.py itself — this is
    what keeps the real sklearn/threadpoolctl fit off the deadlock-prone
    background-thread path during the rest of the suite. Reads the
    attribute off the module (not a top-level `from ... import _warm_hmm`)
    since patch() replaces it there, and a name bound at collection time
    would keep pointing at the pre-patch original."""
    assert isinstance(pipeline_warmup._warm_hmm, MagicMock)


def test_gate_background_thread_never_runs_real_hmm_fit():
    """End-to-end check of the actual path that deadlocked live: a real
    PipelineWarmupGate, spawning its real background thread, against a
    config that would trigger the HMM warmup — the thread must complete
    without ever touching the real fit."""
    gate = PipelineWarmupGate()
    gate.start_background({"pipeline_warmup": True, "strategies": ["regime_hmm"]})
    assert gate.wait_ready(timeout=5.0)
    pipeline_warmup._warm_hmm.assert_called_once()
