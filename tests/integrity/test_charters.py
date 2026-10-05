"""DRAFT of the owner-committed integrity test (P5-02). The owner copies it to tests/integrity/test_charters.py.

Reads committed files only (charters, config/gates.yaml, git history, the host ledger); no market data. All logic lives in
``firm.research.charter`` (unit-tested in tests/test_charter.py); this file only applies it to the real charters.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from firm.research import charter as C

REPO = Path(__file__).resolve().parents[2]
CHARTERS = list(C.iter_charters(REPO))


@pytest.mark.parametrize("path", CHARTERS, ids=lambda p: p.name)
def test_charter_is_valid_and_precedes_ledger(path):
    # strict: an approved charter has no unfilled approval fields. Ordering is checked against the host ledger.
    problems = C.validate_charter_file(path, repo_dir=REPO, strict=True)
    assert problems == []


def test_template_exists_and_is_not_validated_as_a_charter():
    tpl = REPO / "research" / "charters" / C.TEMPLATE_NAME
    assert tpl.is_file()
    fm, body = C.parse_charter(tpl.read_text())
    assert set(C.REQUIRED_KEYS) <= set(fm)
    assert C.validate_body(body) == []
