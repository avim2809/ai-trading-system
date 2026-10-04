"""DRAFT for tests/integrity/test_ledger_chain.py (P1-01). Owner-only path (P0-04): review and commit by hand.

CI check on the tracked mirror research/ledger/trials.jsonl using git history only; no real data.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from firm.research.ledger import verify_chain

REPO = Path(__file__).resolve().parents[2]
MIRROR = "research/ledger/trials.jsonl"
K = 20


def _git(*args: str) -> bytes:
    return subprocess.run(["git", "-c", "safe.directory=*", *args], cwd=REPO, check=True, capture_output=True).stdout


def _versions() -> list[bytes]:
    try:
        shas = _git("log", f"-{K}", "--format=%H", "--", MIRROR).decode().split()
    except subprocess.CalledProcessError:
        return []
    return [_git("show", f"{s}:{MIRROR}") for s in reversed(shas)]  # oldest first


def test_mirror_is_prefix_extension_of_previous_commit():
    v = _versions()
    if len(v) < 2:
        pytest.skip("fewer than two commits touch research/ledger/trials.jsonl")
    for old, new in zip(v, v[1:]):
        assert new.startswith(old), "ledger mirror was rewritten or truncated between commits"


def test_mirror_chain_verifies():
    p = REPO / MIRROR
    if not p.exists():
        pytest.skip("mirror not committed yet")
    rep = verify_chain(p)
    assert rep.ok and not rep.partial_tail, rep


def test_legacy_rows_unchanged_prefix():
    """The legacy block is the first block of the mirror and never changes."""
    import json

    v = _versions()
    if not v:
        pytest.skip("mirror not committed yet")

    def legacy_prefix(b: bytes) -> bytes:
        out = []
        for ln in b.splitlines(keepends=True):
            if json.loads(ln)["row"]["mode"] != "legacy":
                break
            out.append(ln)
        return b"".join(out)

    first = legacy_prefix(v[0])
    assert first, "first mirror version has no legacy block"
    for b in v[1:]:
        assert b.startswith(first)
        rest = b[len(first):]
        # nothing legacy may appear after a non-legacy row has appeared
        seen_other = False
        for ln in rest.splitlines():
            mode = json.loads(ln)["row"]["mode"]
            seen_other |= mode != "legacy"
            assert not (seen_other and mode == "legacy"), "legacy row after the legacy block"
