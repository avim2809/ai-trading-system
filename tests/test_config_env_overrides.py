"""Tests for FIRM_LIVE_CONFIG / FIRM_LLM_CONFIG env overrides."""

from __future__ import annotations

from firm.live import provider_utils
from firm.llm import config as llm_config


def test_firm_live_config_override(monkeypatch, tmp_path):
    yaml_path = tmp_path / "custom_live.yaml"
    yaml_path.write_text("broker: alpaca_paper\nschedule: hourly\n")
    monkeypatch.setenv("FIRM_LIVE_CONFIG", str(yaml_path))
    loaded = provider_utils.load_live_yaml_defaults()
    assert loaded["broker"] == "alpaca_paper"
    assert loaded["schedule"] == "hourly"


def test_firm_llm_config_override(monkeypatch, tmp_path):
    yaml_path = tmp_path / "custom_llm.yaml"
    yaml_path.write_text("agent_modes:\n  fundamental_analyst: quant\n")
    monkeypatch.setenv("FIRM_LLM_CONFIG", str(yaml_path))
    loaded = llm_config.load_llm_config()
    assert loaded["agent_modes"]["fundamental_analyst"] == "quant"
