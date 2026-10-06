"""Frozen S2 forward-test pre-registration: status, fingerprint, power arithmetic."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import s2_forward_preregistered as pre


def test_frozen_and_fingerprint_deterministic():
    assert pre.DRAFT is False and pre.PREREGISTERED_AT.endswith("Z")
    assert pre.bars_fingerprint() == pre.bars_fingerprint() and len(pre.bars_fingerprint()) == 64


def test_rule_matches_the_frozen_s2_v1_numbers():
    assert pre.RULE["cut_weights"] == {"SPY": 0.30, "IEF": 0.70}
    assert pre.RULE["plain_weights"] == {"SPY": 0.60, "IEF": 0.40}
    assert "0.50" in pre.RULE["cut_if"]
    assert pre.FROZEN_REFERENCE["bars_fingerprint"].startswith("4fd97659")


def test_start_is_the_first_trading_day_of_october_2026():
    from firm.allocation.calendar import first_trading_day_of_month
    assert first_trading_day_of_month(2026, 10).isoformat() == pre.START["first_check_day"]


def test_power_report_matches_poisson_arithmetic():
    r = pre.power_report(24)
    assert r["expected_episodes"] == pytest.approx(19 / 24.66 * 2, rel=1e-6)
    assert 0.4 < r["p_at_least_2"] < 0.5
    assert pre.power_report(48)["p_at_least_2"] > r["p_at_least_2"]
