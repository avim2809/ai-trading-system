"""DRAFT of tests/integrity/test_core_v1_prereg_order.py (ticket P3-11; owner commits it under CODEOWNERS).

The core_v1 pre-registration commit must be EARLIER than the first ledger trial that carries the core_v1 preregistration id in the tracked
mirror ``research/ledger/trials.jsonl``; no post-freeze edit to ``scripts/core_v1_preregistered.py``. Git-based; skipped when git history or
the mirror is unavailable. No data is read.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
MIRROR = REPO / "research" / "ledger" / "trials.jsonl"
YAML_PATH = "research/preregistration/20261006_core_v1.yaml"
MODULE = "scripts/core_v1_preregistered.py"


def _git(*a: str) -> str:
    return subprocess.run(["git", "-c", "safe.directory=*", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def _have_git() -> bool:
    try:
        _git("rev-parse", "HEAD")
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def test_prereg_commit_precedes_the_first_core_v1_ledger_trial():
    if not _have_git() or not MIRROR.exists():
        pytest.skip("git history or the ledger mirror is unavailable")
    first_add = _git("log", "--diff-filter=A", "--format=%cI", "--", YAML_PATH).splitlines()
    if not first_add:
        pytest.skip("prereg yaml has no git history here")
    t_prereg = datetime.fromisoformat(first_add[-1])
    times = []
    for ln in MIRROR.read_text().splitlines():
        if ln.strip():
            row = json.loads(ln)["row"]
            if row.get("preregistration_id") == "core_v1" or (row.get("family") == "core_v1" and row.get("mode") == "registered"):
                times.append(datetime.fromisoformat(row["created_at"]))
    if not times:
        pytest.skip("no core_v1 trial in the mirror yet")
    assert t_prereg < min(times), "a core_v1 trial was recorded before the pre-registration commit"


def test_frozen_module_has_no_post_freeze_edit():
    if not _have_git():
        pytest.skip("git history unavailable")
    import yaml

    entries = yaml.safe_load((REPO / "research" / "preregistration" / "INDEX.yaml").read_text())["entries"]
    entry = next(e for e in entries if e["family"] == "core_v1_module")
    freeze = entry["freeze_commit"]
    changed = _git("log", "--format=%H", f"{freeze}..HEAD", "--", MODULE).splitlines()
    assert not changed, f"{MODULE} was edited after the freeze commit: {changed}"
