"""docs/*_trial_history.json are append-only (ticket P0-04, AGENTS.md rule 3).

The ledgers feed ``prior_trials`` into the Deflated Sharpe, so an old entry that is edited,
dropped or reordered would silently understate N. The baseline is PINNED, not ``HEAD~``: a push of
several commits (or a rewrite followed by an append) would defeat a ``HEAD~`` comparison. Two
independent pins are used:

* ``BASELINE`` below (test-held, CODEOWNERS-protected): sha256 of every baseline entry (canonical JSON,
  sorted keys) and the baseline value of each top-level counter. Works without git objects.
* ``git show c2bd5cd:docs/<file>`` (the commit at plan time): cross-checks the pins and the current entries.
  When the revision cannot be resolved this FAILS in CI (``CI=true`` or ``FIRM_INTEGRITY_CI=1``) and skips
  only on a developer machine without history.

Top-level counters (``cumulative_trials``, ``cumulative_trials_through_last_entry``) change on append, so
the check is: baseline ``entries`` are an unchanged prefix; counters never decrease and are at least the
sum of entry ``n_trials`` where entries carry it. A ledger created after the baseline must be added to
``BASELINE`` by an owner-signed change; until then ``test_every_ledger_has_a_baseline`` fails.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS = REPO_ROOT / "docs"
BASELINE_REV = "c2bd5cd"
COUNTERS = ("cumulative_trials", "cumulative_trials_through_last_entry")

# sha256 of every baseline entry at c2bd5cd (canonical JSON, sorted keys) and the baseline counters.
BASELINE: dict[str, dict] = {
    "S1_trial_history.json": {
        "counters": {"cumulative_trials": 4},
        "entries": [
            "064665d5784fc4a01923a94ea9d8e87c11040c6368a0d8069008126f47de4f78",
        ],
    },
    "S2_trial_history.json": {
        "counters": {"cumulative_trials": 4},
        "entries": [
            "6145b7731b16ecb0939e602a18fa19b9bf318593f7b6f1b8cf673785a758f87e",
        ],
    },
    "S3_trial_history.json": {
        "counters": {"cumulative_trials": 5},
        "entries": [
            "5938545312c02fbfc5b3f736ab09bf9b947f48a9b325dbbf149d99e8f10c1cbb",
        ],
    },
    "S4_trial_history.json": {
        "counters": {"cumulative_trials": 4},
        "entries": [
            "ef1628cd1fbe073188b23a266f45b0dec031f9b714604e1e086dd7380b83a0ae",
        ],
    },
    "allocation_forward_test_trial_history.json": {
        "counters": {},
        "entries": [
            "3f6ef7ad914f27e30bf05775457899c71f5ef493833d03ac27ec51d9b858dc1d",
        ],
    },
    "alt_premia_trial_history.json": {
        "counters": {"cumulative_trials": 10},
        "entries": [
            "7a0ff51ed8b28fed535ef7472172d71ece702eb0fe6c37788e72a6a0c7f76082",
        ],
    },
    "combination_trial_history.json": {
        "counters": {},
        "entries": [
            "b0c9fdda8ccc379472635cfe35a0358aa74a306decb68f25a3186a8538cc073c",
            "5df22e14fb8e3f1c42110334ffa4fd152da1e2d77b7adf29e4960c45c52c5854",
            "c44e9d10e2484716bac2b22713a814c05697b6ae287b48b703207ada58ae9a48",
            "8f2c127884d9dd3d3d738c0966a54c3a59d83e2ebec0670fcbe578e6c3aeb218",
            "e27f55ad91bc2fb99838fd47060046fb6671cc4b5b95c9d95c99095b1d392ce5",
            "5023beacd93e3c6f5c17235c28fc12b001a7f4e845a7e70c6674e41a01f52161",
            "a0a121205f2125b6115c0fc7a90de26ac4f914a3e2b702b7a5e79cd4178ae348",
            "ae74f6296217a481ad1086b5bbfb4008a609bf5251ba26fcac1a5ef981963d21",
            "7e904653c3da4362a2604d918109107562aeaeb9d8613d29875a8a94637571a6",
            "1cdd54ef35f280cbddf2ef53bba30b0adfc6116f64a3870fbdd7a2b562778805",
            "5c18636043cb421cd84c673d570d57e7f06e7ecc7874c9975a801348dd1e0c28",
            "1a0495bc1f8bba68a128c16de07241e7d842077ebe0f572af4369d0dc651ff83",
            "dc0b6eb36b2aa2482bd2e387e9c3a1bd07417d91661643541a8ae35330f3c45c",
            "73487abc117cae639dae8b5f6afcbb129d9cb16a242f634d1c7b75bf96cbe4ae",
            "4487b27c43f6246bedf8d9d7273e2526d9915ff56cafea6413e5293fed81877f",
            "915683f7311b73403fe9152a8f1dc1ff23dd9f2aba1863ed8d6d8a3850aa71cb",
        ],
    },
    "insider_cluster_trial_history.json": {
        "counters": {"cumulative_trials": 8},
        "entries": [
            "5fb50335cd9ea2dcedb9f5d3ea3727a4148d365713185ec0e022172c9dc6a91a",
        ],
    },
    "pattern_ml_trial_history.json": {
        "counters": {"cumulative_trials_through_last_entry": 104},
        "entries": [
            "80b9dab5c3a762cc8b52f63aa9384ecb315d4c9a96814aad3776926b8203b31f",
            "16a1e26d95103eb1feae16546e93979ccb42e75bec5d2d9fffbcdfe8f2fc484a",
            "023bd59bf7a4ea1f6ec2e61daeec56315ef46a53686bab94d4587ff3558d1fdb",
            "6fd7cae0babb35f02da382ab89ca7bf8513a371205a5db5f8d9aa1165bbeb1eb",
        ],
    },
    "s5_trial_history.json": {
        "counters": {},
        "entries": [
            "ddd928d28b5408f45bbda137de46cbd8ec3452397295268f8b732fa2bf1dcfdd",
        ],
    },
    "standalone_strategy_trial_history.json": {
        "counters": {"cumulative_trials": 11},
        "entries": [
            "2223c92b2cb6b489ccc0c5e395218394c232d5213ce99e854ed7eaa6cfed4cf7",
        ],
    },
}


def entry_hash(entry: object) -> str:
    blob = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def check_ledger(current: dict, baseline_hashes: list[str], baseline_counters: dict) -> list[str]:
    """Return a list of problems (empty = append-only invariant holds)."""
    problems: list[str] = []
    entries = current.get("entries")
    if not isinstance(entries, list):
        return ["'entries' is missing or not a list"]
    if len(entries) < len(baseline_hashes):
        problems.append(f"entries shrank: {len(entries)} < baseline {len(baseline_hashes)}")
    for i, expected in enumerate(baseline_hashes[: len(entries)]):
        if entry_hash(entries[i]) != expected:
            problems.append(f"baseline entry {i} was modified or reordered")
    for name, base_value in baseline_counters.items():
        value = current.get(name)
        if not isinstance(value, int) or value < base_value:
            problems.append(f"counter {name} decreased or vanished: {value!r} < baseline {base_value}")
    carried = sum(e["n_trials"] for e in entries if isinstance(e, dict) and isinstance(e.get("n_trials"), int))
    for name in COUNTERS:
        if name in current and isinstance(current[name], int) and current[name] < carried:
            problems.append(f"counter {name}={current[name]} is below the sum of entry n_trials ({carried})")
    return problems


def _current(name: str) -> dict:
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def _in_ci() -> bool:
    return os.environ.get("CI", "").lower() == "true" or os.environ.get("FIRM_INTEGRITY_CI") == "1"


def _git_baseline(name: str) -> dict | None:
    try:
        out = subprocess.run(
            ["git", "show", f"{BASELINE_REV}:docs/{name}"],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return json.loads(out)


def test_every_ledger_has_a_baseline():
    files = sorted(p.name for p in DOCS.glob("*_trial_history.json"))
    missing = [f for f in files if f not in BASELINE]
    assert not missing, f"ledgers without a pinned baseline (owner-signed change to BASELINE required): {missing}"
    gone = [f for f in BASELINE if f not in files]
    assert not gone, f"baseline ledgers deleted or renamed: {gone}"


@pytest.mark.parametrize("name", sorted(BASELINE))
def test_baseline_entries_are_an_unchanged_prefix(name):
    problems = check_ledger(_current(name), BASELINE[name]["entries"], BASELINE[name]["counters"])
    assert not problems, f"{name}: {problems}"


@pytest.mark.parametrize("name", sorted(BASELINE))
def test_git_baseline_agrees_with_pins_and_current(name):
    base = _git_baseline(name)
    if base is None:
        if _in_ci():
            pytest.fail(f"cannot resolve {BASELINE_REV}:docs/{name} in CI (checkout needs fetch-depth: 0)")
        pytest.skip(f"{BASELINE_REV} not available (developer machine without history)")
    assert [entry_hash(e) for e in base["entries"]] == BASELINE[name]["entries"], "pinned hashes differ from git"
    counters = {k: base[k] for k in COUNTERS if k in base}
    assert counters == BASELINE[name]["counters"], "pinned counters differ from git"
    assert not check_ledger(_current(name), [entry_hash(e) for e in base["entries"]], counters)


# --- the checker itself (so the invariant cannot rot silently) ---------------------------------

def _sample() -> tuple[dict, list[str], dict]:
    doc = {"family": "x", "entries": [{"n_trials": 3, "a": 1}, {"n_trials": 2, "a": 2}], "cumulative_trials": 5}
    return doc, [entry_hash(e) for e in doc["entries"]], {"cumulative_trials": 5}


def test_checker_accepts_appends_and_counter_growth():
    doc, hashes, counters = _sample()
    doc["entries"].append({"n_trials": 4})
    doc["cumulative_trials"] = 9
    assert check_ledger(doc, hashes, counters) == []


def test_checker_rejects_edit_delete_reorder_and_counter_decrease():
    doc, hashes, counters = _sample()
    edited = json.loads(json.dumps(doc))
    edited["entries"][0]["a"] = 99
    assert check_ledger(edited, hashes, counters)
    shrunk = json.loads(json.dumps(doc))
    shrunk["entries"].pop()
    assert check_ledger(shrunk, hashes, counters)
    swapped = json.loads(json.dumps(doc))
    swapped["entries"].reverse()
    assert check_ledger(swapped, hashes, counters)
    lowered = json.loads(json.dumps(doc))
    lowered["cumulative_trials"] = 4
    assert check_ledger(lowered, hashes, counters)
    understated = json.loads(json.dumps(doc))
    understated["entries"].append({"n_trials": 10})
    assert check_ledger(understated, hashes, counters)  # counter 5 < sum of n_trials 15


def test_ci_fails_instead_of_skipping_without_baseline_revision(monkeypatch):
    monkeypatch.setenv("FIRM_INTEGRITY_CI", "1")
    assert _in_ci()
    monkeypatch.delenv("FIRM_INTEGRITY_CI")
    monkeypatch.delenv("CI", raising=False)
    assert not _in_ci()
