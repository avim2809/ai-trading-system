"""Tests for firm.live.pattern_ml_gate -- the fail-closed guard on
pattern_recognition's CNN/XGBoost ML knobs.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from firm.live.approval import ApprovalQueue
from firm.live.data_feed import LiveDataFeed
from firm.live.engine import LiveTradingEngine
from firm.live.pattern_ml_gate import evaluate_gate, sanitize_strategy_params
from tests.test_brokers import MockBroker


class TestEvaluateGate:
    def test_no_record_refuses(self):
        decision = evaluate_gate(None)
        assert decision["allowed"] is False

    def test_keep_recommendation_allows(self):
        decision = evaluate_gate({"overall_recommendation": "KEEP"})
        assert decision["allowed"] is True

    def test_hold_recommendation_refuses(self):
        decision = evaluate_gate({"overall_recommendation": "HOLD (insufficient live sample, backtest verdict=fail)"})
        assert decision["allowed"] is False

    def test_rollback_recommendation_refuses(self):
        decision = evaluate_gate({"overall_recommendation": "ROLLBACK"})
        assert decision["allowed"] is False

    def test_missing_recommendation_key_refuses(self):
        decision = evaluate_gate({"backtest": {"verdict": "pass"}})
        assert decision["allowed"] is False


class TestSanitizeStrategyParams:
    def test_no_pattern_recognition_params_passthrough(self):
        params = {"momentum": {"lookback": 20}}
        result = sanitize_strategy_params(params, gate_path_override="/nonexistent")
        assert result == params

    def test_flags_off_passthrough_even_without_gate_file(self, tmp_path):
        params = {"pattern_recognition": {"cnn_scoring_enabled": False}}
        result = sanitize_strategy_params(params, gate_path_override=tmp_path / "missing.json")
        assert result["pattern_recognition"]["cnn_scoring_enabled"] is False

    def test_cnn_flag_on_without_gate_file_is_refused(self, tmp_path):
        gate_path = tmp_path / "gate.json"
        params = {"pattern_recognition": {"cnn_scoring_enabled": True, "zigzag_pct": 0.03}}
        result = sanitize_strategy_params(params, gate_path_override=gate_path)
        assert result["pattern_recognition"]["cnn_scoring_enabled"] is False
        # Untouched keys survive.
        assert result["pattern_recognition"]["zigzag_pct"] == 0.03

    def test_xgb_flag_on_without_gate_file_is_refused(self, tmp_path):
        gate_path = tmp_path / "gate.json"
        params = {"pattern_recognition": {"xgb_confirmation_enabled": True}}
        result = sanitize_strategy_params(params, gate_path_override=gate_path)
        assert result["pattern_recognition"]["xgb_confirmation_enabled"] is False

    def test_flag_on_with_stale_hold_record_is_refused(self, tmp_path):
        gate_path = tmp_path / "gate.json"
        gate_path.write_text(json.dumps({"overall_recommendation": "HOLD (insufficient live sample, backtest verdict=fail)"}))
        params = {"pattern_recognition": {"cnn_scoring_enabled": True}}
        result = sanitize_strategy_params(params, gate_path_override=gate_path)
        assert result["pattern_recognition"]["cnn_scoring_enabled"] is False

    def test_flag_on_with_passing_keep_record_is_honored(self, tmp_path):
        gate_path = tmp_path / "gate.json"
        gate_path.write_text(json.dumps({"overall_recommendation": "KEEP"}))
        params = {"pattern_recognition": {"cnn_scoring_enabled": True}}
        result = sanitize_strategy_params(params, gate_path_override=gate_path)
        assert result["pattern_recognition"]["cnn_scoring_enabled"] is True

    def test_corrupt_gate_file_is_treated_as_absent_and_refused(self, tmp_path):
        gate_path = tmp_path / "gate.json"
        gate_path.write_text("{not valid json")
        params = {"pattern_recognition": {"cnn_scoring_enabled": True}}
        result = sanitize_strategy_params(params, gate_path_override=gate_path)
        assert result["pattern_recognition"]["cnn_scoring_enabled"] is False

    def test_never_mutates_input(self, tmp_path):
        params = {"pattern_recognition": {"cnn_scoring_enabled": True}}
        original_inner = params["pattern_recognition"]
        sanitize_strategy_params(params, gate_path_override=tmp_path / "missing.json")
        assert params["pattern_recognition"] is original_inner
        assert original_inner["cnn_scoring_enabled"] is True

    def test_env_override_used_when_no_explicit_path_given(self, tmp_path, monkeypatch):
        gate_path = tmp_path / "gate.json"
        gate_path.write_text(json.dumps({"overall_recommendation": "KEEP"}))
        monkeypatch.setenv("FIRM_PATTERN_ML_GATE", str(gate_path))
        params = {"pattern_recognition": {"cnn_scoring_enabled": True}}
        result = sanitize_strategy_params(params)
        assert result["pattern_recognition"]["cnn_scoring_enabled"] is True


def _make_engine(tmp_path):
    broker = MockBroker()
    feed = LiveDataFeed(providers={}, universe=["AAPL", "MSFT"])
    queue = ApprovalQueue(broker=broker)
    config = {
        "initial_capital": 100_000,
        "memory_log_path": str(tmp_path / "decisions.jsonl"),
        "strategy_params": {},
    }
    with patch("firm.live.engine.build_orchestrator") as mock_build:
        mock_build.return_value = MagicMock()
        return LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)


class TestLiveTradingEngineIntegration:
    """End-to-end: the gate must actually reach the live engine's config,
    both at construction and via the hot-swap ``update_*`` path, with no
    gate record present on disk (the default state after a fresh checkout
    or a rolled-back knob)."""

    def test_init_with_cnn_flag_already_on_in_config_is_downgraded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FIRM_PATTERN_ML_GATE", str(tmp_path / "missing_gate.json"))
        broker = MockBroker()
        feed = LiveDataFeed(providers={}, universe=["AAPL"])
        queue = ApprovalQueue(broker=broker)
        config = {
            "initial_capital": 100_000,
            "memory_log_path": str(tmp_path / "decisions.jsonl"),
            "strategy_params": {"pattern_recognition": {"cnn_scoring_enabled": True}},
        }
        with patch("firm.live.engine.build_orchestrator") as mock_build:
            mock_build.return_value = MagicMock()
            engine = LiveTradingEngine(config=config, broker=broker, data_feed=feed, approval_queue=queue)

        assert engine._config["strategy_params"]["pattern_recognition"]["cnn_scoring_enabled"] is False

    def test_update_strategy_params_downgrades_without_gate_record(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FIRM_PATTERN_ML_GATE", str(tmp_path / "missing_gate.json"))
        engine = _make_engine(tmp_path)

        with patch("firm.live.engine.build_orchestrator") as mock_build:
            mock_build.return_value = MagicMock()
            engine.update_strategy_params({"pattern_recognition": {"cnn_scoring_enabled": True, "xgb_confirmation_enabled": True}})

        assert engine._config["strategy_params"]["pattern_recognition"]["cnn_scoring_enabled"] is False
        assert engine._config["strategy_params"]["pattern_recognition"]["xgb_confirmation_enabled"] is False

    def test_update_strategy_params_honors_passing_gate_record(self, tmp_path, monkeypatch):
        gate_path = tmp_path / "gate.json"
        gate_path.write_text(json.dumps({"overall_recommendation": "KEEP"}))
        monkeypatch.setenv("FIRM_PATTERN_ML_GATE", str(gate_path))
        engine = _make_engine(tmp_path)

        with patch("firm.live.engine.build_orchestrator") as mock_build:
            mock_build.return_value = MagicMock()
            engine.update_strategy_params({"pattern_recognition": {"cnn_scoring_enabled": True}})

        assert engine._config["strategy_params"]["pattern_recognition"]["cnn_scoring_enabled"] is True

    def test_update_strategies_does_not_race_the_sanitized_config(self, tmp_path, monkeypatch):
        # update_strategies sets self._config *after* calling
        # _rebuild_orchestrator -- guards against that reintroducing a race
        # that overwrites the sanitized value with the raw request.
        monkeypatch.setenv("FIRM_PATTERN_ML_GATE", str(tmp_path / "missing_gate.json"))
        engine = _make_engine(tmp_path)
        engine._config = {
            **engine._config,
            "strategy_params": {"pattern_recognition": {"cnn_scoring_enabled": True}},
        }

        with patch("firm.live.engine.build_orchestrator") as mock_build:
            mock_build.return_value = MagicMock()
            engine.update_strategies(["pattern_recognition", "momentum"])

        assert engine._config["strategy_params"]["pattern_recognition"]["cnn_scoring_enabled"] is False
