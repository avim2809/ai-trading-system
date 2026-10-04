"""A full-suite run must leave the host research ledger byte-unchanged (P1-12).

DRAFT for the owner-protected integrity test directory (owner commits it unchanged).

Records existence, size and last row_hash of ``/local/store/research-ledger/trials.jsonl`` plus the host
``inbox/`` and ``returns/`` listings at session start and asserts they are identical at session end.
Read-only; a missing host ledger snapshots as ``{"exists": False}``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

HOST_ROOT = Path("/local/store/research-ledger")


def _snapshot() -> dict:
    if not HOST_ROOT.is_dir():
        return {"exists": False}
    tr = HOST_ROOT / "trials.jsonl"
    last = size = None
    if tr.exists():
        size = tr.stat().st_size
        lines = tr.read_bytes().splitlines()
        if lines:
            try:
                last = json.loads(lines[-1]).get("row_hash")
            except ValueError:
                last = "unparseable-tail"
    listing = {
        d: sorted(os.listdir(HOST_ROOT / d)) if (HOST_ROOT / d).is_dir() else None for d in ("inbox", "returns")
    }
    return {"exists": True, "trials_size": size, "last_row_hash": last, "listing": listing}


_START: dict = {}


@pytest.fixture(scope="session", autouse=True)
def _host_ledger_guard():
    _START.update(_snapshot())
    yield
    assert _snapshot() == _START, "the test session modified the host research ledger (inbox/returns/trials)"


def test_host_ledger_snapshot_is_stable_now():
    assert _snapshot() == _START
