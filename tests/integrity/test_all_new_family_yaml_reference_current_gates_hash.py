"""DRAFT (owner commits to tests/integrity/): every APPROVED yaml prereg references the current gates hash,
its recorded spec hash, and a charter committed before approval; every frozen prereg module is indexed."""

from __future__ import annotations

from pathlib import Path

from firm.research import prereg as P

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "research" / "preregistration" / "INDEX.yaml"


def test_index_exists_and_verifies_cleanly():
    assert INDEX.is_file()
    assert P.verify_index(P.load_index(INDEX), REPO) == []


def test_all_yaml_families_reference_current_gates_hash():
    current = P.gates_hash(REPO)
    for e in P.load_index(INDEX):
        if e.kind == "yaml" and e.status == "APPROVED":
            spec = P.load_spec(REPO / e.yaml_path)
            assert spec.gates_hash == current, e.family
            assert P.spec_hash(spec) == e.fingerprint, e.family
            assert P.validate_spec(spec) == [], e.family


def test_every_frozen_module_indexed():
    on_disk = {p.relative_to(REPO).as_posix() for p in (REPO / "scripts").glob("*_preregistered*.py")}
    assert on_disk == {e.module_path for e in P.load_index(INDEX) if e.kind == "frozen_module"}
