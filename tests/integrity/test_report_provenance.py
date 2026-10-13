"""DRAFT of tests/integrity/test_report_provenance.py (ticket P3-08; owner commits it under CODEOWNERS, agents may not write tests/integrity/).

Every ledger trial id referenced in ``research/reports/**`` (JSON, YAML and Markdown) must exist in the tracked ledger mirror
``research/ledger/trials.jsonl`` as a COMPLETED row, and every id the core_v1 run cites as registered evidence must be in mode
``registered``. Git-free; skipped while the mirror is not committed yet. No data is read.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
MIRROR = REPO / "research" / "ledger" / "trials.jsonl"
REPORTS = REPO / "research" / "reports"
ID_KEYS = {"trial_id", "trial_ids", "ledger_trial_ids", "engine_trial", "ledger_trial_id"}
HEX32 = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{32}(?![0-9a-fA-F])")
REGISTERED_ONLY_FILES = {"results.json", "constants.json", "constants_addendum.draft.yaml"}   # ids cited here are registered rows


def _mirror() -> dict[str, dict]:
    rows = {}
    for ln in MIRROR.read_text().splitlines():
        if ln.strip():
            r = json.loads(ln)["row"]
            rows[r["trial_id"]] = r
    return rows


def _walk(obj, under_id_key=False):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk(v, under_id_key or k in ID_KEYS)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, under_id_key)
    elif isinstance(obj, str) and under_id_key and HEX32.fullmatch(obj):
        yield obj


def _ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return set(_walk(json.loads(text)))
    if path.suffix in (".yaml", ".yml"):
        return set(_walk(yaml.safe_load(text)))
    if path.suffix == ".md":
        return set(HEX32.findall(text))
    return set()


def _files():
    return [p for p in sorted(REPORTS.rglob("*")) if p.is_file() and p.suffix in (".json", ".yaml", ".yml", ".md") and "drafts" not in p.parts]


def test_every_cited_trial_id_is_in_the_ledger_mirror_and_completed():
    if not MIRROR.exists():
        pytest.skip("ledger mirror not committed yet")
    rows = _mirror()
    bad = []
    for p in _files():
        for t in _ids(p):
            r = rows.get(t)
            if r is None:
                bad.append(f"{p.relative_to(REPO)}: {t} not in the mirror")
            elif r["status"] != "completed":
                bad.append(f"{p.relative_to(REPO)}: {t} status {r['status']}")
    assert not bad, bad


def test_core_v1_evidence_ids_are_registered_rows():
    if not MIRROR.exists():
        pytest.skip("ledger mirror not committed yet")
    rows = _mirror()
    bad = []
    for p in _files():
        if p.name not in REGISTERED_ONLY_FILES:
            continue
        for t in _ids(p):
            r = rows.get(t)
            if r is not None and r["family"] == "core_v1" and r["mode"] != "registered":
                bad.append(f"{p.relative_to(REPO)}: {t} is {r['mode']}, not registered")
    assert not bad, bad
