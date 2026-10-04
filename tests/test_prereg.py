"""Pre-registration index + approval check (P1-09). Synthetic tmp repos only; the real scripts/ are read-only."""

from __future__ import annotations

import dataclasses
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from firm.research import ledger as L
from firm.research import prereg as P

REPO = Path(__file__).resolve().parents[1]
REAL_INDEX = REPO / "research" / "preregistration" / "INDEX.yaml"
DRAFT_INDEX = REPO / "plan" / "drafts" / "P1-09" / "INDEX.yaml"  # owner commits it as REAL_INDEX
TEMPLATE_DRAFT = REPO / "plan" / "drafts" / "P1-09" / "TEMPLATE.yaml"
TEMPLATE_REAL = REPO / "research" / "preregistration" / "TEMPLATE.yaml"


def _index_path() -> Path:
    return REAL_INDEX if REAL_INDEX.exists() else DRAFT_INDEX


def _template_path() -> Path:
    return TEMPLATE_REAL if TEMPLATE_REAL.exists() else TEMPLATE_DRAFT


def _git(d: Path, *args: str, when: str = "2026-11-01T00:00:00+00:00") -> None:
    env = {**os.environ, "GIT_COMMITTER_DATE": when, "GIT_AUTHOR_DATE": when}
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
        cwd=d, check=True, capture_output=True, env=env,
    )


def _spec(gates: str, **over) -> dict:
    d = {
        "family": "fam", "hypothesis": "h", "mechanism_ref": "research/charters/c.md",
        "universe": ["SPY", "IEF"], "date_range": ["2010-01-01", "2026-06-30"],
        "param_grid": {"speed": [8, 16], "cap": [1.0, 2.0]}, "max_grid_size": 12, "n_planned_trials": 4,
        "metrics": ["net_sharpe"], "gates_hash": gates, "charter_ref": "research/charters/c.md",
        "falsification": "net Sharpe CI includes 0", "post_seal_data_declaration": P.POST_SEAL_DECLARATION,
        "approver": "owner", "approved_at_utc": "2026-11-10T00:00:00Z",
    }
    d.update(over)
    return d


@pytest.fixture()
def env(tmp_path, monkeypatch):
    d = tmp_path / "repo"
    (d / "config").mkdir(parents=True)
    (d / "research" / "charters").mkdir(parents=True)
    (d / "research" / "preregistration").mkdir(parents=True)
    (d / "config" / "gates.yaml").write_text("a: 1\n")
    (d / "research" / "charters" / "c.md").write_text("charter\n")
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "charter")
    spec = P.spec_from_dict(_spec(P.gates_hash(d)))
    yp = d / "research" / "preregistration" / "20261110_fam.yaml"
    yp.write_text(yaml.safe_dump(_spec(P.gates_hash(d))))
    idx = d / "research" / "preregistration" / "INDEX.yaml"

    def write_index(status="APPROVED", fp=None):
        idx.write_text(yaml.safe_dump({"entries": [{
            "family": "fam", "kind": "yaml", "module_path": None,
            "yaml_path": "research/preregistration/20261110_fam.yaml", "fingerprint": fp or P.spec_hash(spec),
            "freeze_commit": "x", "freeze_time_utc": "2026-11-10T00:00:00Z", "ledger_path": None,
            "fingerprint_source": "none", "status": status,
        }]}))

    write_index()
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "prereg", when="2026-11-10T00:00:00+00:00")  # clean tree for registered ledger rows
    monkeypatch.setattr(P, "REPO_DIR", d)
    monkeypatch.setattr(P, "DEFAULT_INDEX", idx)
    return d, spec, write_index


def ok_cfg(**kw):
    return {"speed": 8, "cap": 1.0, **kw}


# ------------------------------------------------------------------ real frozen modules

def test_every_frozen_prereg_module_is_indexed():
    on_disk = {p.relative_to(REPO).as_posix() for p in (REPO / "scripts").glob("*_preregistered*.py")}
    indexed = {e.module_path for e in P.load_index(_index_path()) if e.kind == "frozen_module"}
    assert on_disk == indexed


def test_every_frozen_prereg_module_is_indexed_detects_new_module(tmp_path):
    idx = yaml.safe_load(_index_path().read_text())
    idx["entries"] = idx["entries"][1:]
    p = tmp_path / "INDEX.yaml"
    p.write_text(yaml.safe_dump(idx))
    on_disk = {q.relative_to(REPO).as_posix() for q in (REPO / "scripts").glob("*_preregistered*.py")}
    assert on_disk != {e.module_path for e in P.load_index(p)}


@pytest.fixture(scope="module")
def real_problems():
    return P.verify_index(P.load_index(_index_path()), REPO)


def test_recomputed_fingerprint_matches_recorded(real_problems):
    # verify_index recomputes every module in a subprocess; trial_history sources must match the ledger value,
    # freeze_commit sources must equal the freeze-commit recomputation (drift check).
    assert real_problems == []


def test_every_entry_has_a_fingerprint_source_consistent_with_ledger():
    for e in P.load_index(_index_path()):
        if e.kind != "frozen_module":
            continue
        if e.fingerprint_source == "trial_history":
            assert e.ledger_path and e.fingerprint in P._ledger_fingerprints(REPO / e.ledger_path), e.family
        else:
            assert e.fingerprint_source == "freeze_commit", e.family
    by = {e.family: e for e in P.load_index(_index_path())}
    for fam in ("combination", "pattern_ml", "s2_forward", "futures_trend"):
        assert by[fam].fingerprint_source == "freeze_commit"
    assert "s2_forward" in by and "allocation_forward_test" in by


def test_index_freeze_commits_are_ancestors_of_head():
    if (REPO / ".git").is_file() or (REPO / ".git").is_dir():
        shallow = subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "--is-shallow-repository"],
                                 cwd=REPO, capture_output=True, text=True, check=False).stdout.strip()
        if shallow == "true":
            pytest.skip("shallow clone: ancestry cannot be checked")
    for e in P.load_index(_index_path()):
        if e.kind != "frozen_module":
            continue
        r = subprocess.run(["git", "-c", "safe.directory=*", "merge-base", "--is-ancestor", e.freeze_commit, "HEAD"],
                           cwd=REPO, capture_output=True, check=False)
        assert r.returncode == 0, e.family
        assert e.freeze_time_utc == P.freeze_time_utc(e.freeze_commit, REPO)
        assert e.freeze_time_utc.endswith("Z")


def test_futures_trend_marked_draft_or_superseded_with_zero_trials():
    e = {x.family: x for x in P.load_index(_index_path())}["futures_trend"]
    assert e.status in ("DRAFT", "SUPERSEDED")
    assert e.ledger_path is None  # no docs/futures_trend_trial_history.json: zero trials
    assert not (REPO / "docs" / "futures_trend_trial_history.json").exists()
    assert "superseded-by core_v1" in e.notes and "0 trials" in e.notes


def test_known_inconsistencies_recorded():
    notes = {e.family: e.notes for e in P.load_index(_index_path())}
    assert "205" in notes["S3"] and "210" in notes["S3"]
    assert "52" in notes["alt_premia"] and "57" in notes["alt_premia"]
    assert "family key" in notes["s5"]


def test_verify_index_detects_changed_fingerprint(tmp_path):
    idx = yaml.safe_load(_index_path().read_text())
    idx["entries"][0]["fingerprint"] = "0" * 64
    p = tmp_path / "INDEX.yaml"
    p.write_text(yaml.safe_dump(idx))
    probs = P.verify_index(P.load_index(p), REPO)
    assert any("recomputed fingerprint" in x for x in probs)


# ------------------------------------------------------------------ spec

def test_spec_hash_ignores_approver_fields_and_detects_grid_change(env):
    _, spec, _ = env
    h = P.spec_hash(spec)
    assert P.spec_hash(dataclasses.replace(spec, approver="someone", approved_at_utc="2030-01-01T00:00:00Z")) == h
    assert P.spec_hash(dataclasses.replace(spec, param_grid={"speed": [8, 16, 32], "cap": [1.0, 2.0]})) != h
    assert P.spec_hash(dataclasses.replace(spec, hypothesis="h2")) != h


def test_is_approved_happy_path_and_family_or_stem(env):
    assert P.is_approved("fam", ok_cfg())
    assert P.is_approved("20261110_fam", ok_cfg())
    assert P.is_approved("fam", {"params": ok_cfg(), "seed": 3})
    assert not P.is_approved("other", ok_cfg())


def test_is_approved_rejects_hash_mismatch_gates_change_oversized_grid_and_later_charter(env):
    d, _, write_index = env
    # hash mismatch (index records a different hash)
    write_index(fp="1" * 64)
    assert not P.is_approved("fam", ok_cfg())
    write_index()
    assert P.is_approved("fam", ok_cfg())
    # non-approved status
    write_index(status="DRAFT")
    assert not P.is_approved("fam", ok_cfg())
    write_index()
    # config outside the grid / extra param / missing param
    assert not P.is_approved("fam", ok_cfg(speed=9))
    assert not P.is_approved("fam", ok_cfg(extra=1))
    assert not P.is_approved("fam", {"speed": 8})
    assert not P.is_approved("fam", ok_cfg(speed=8.0 + 0j))
    # oversized grid (spec edited AND index re-hashed still fails: grid 13 > 12)
    big = _spec(P.gates_hash(d), param_grid={"speed": list(range(13)), "cap": [1.0]}, n_planned_trials=4)
    (d / "research/preregistration/20261110_fam.yaml").write_text(yaml.safe_dump(big))
    write_index(fp=P.spec_hash(P.spec_from_dict(big)))
    assert not P.is_approved("fam", {"speed": 1, "cap": 1.0})
    (d / "research/preregistration/20261110_fam.yaml").write_text(yaml.safe_dump(_spec(P.gates_hash(d))))
    write_index()
    assert P.is_approved("fam", ok_cfg())
    # gates.yaml changed after approval
    (d / "config" / "gates.yaml").write_text("a: 2\n")
    assert not P.is_approved("fam", ok_cfg())
    (d / "config" / "gates.yaml").write_text("a: 1\n")
    assert P.is_approved("fam", ok_cfg())
    # charter committed AFTER approval time
    (d / "research/charters/c.md").write_text("charter v2\n")
    _git(d, "commit", "-qam", "charter edit", when="2026-11-20T00:00:00+00:00")
    assert not P.is_approved("fam", ok_cfg())


def test_is_approved_false_when_charter_missing_or_untracked(env):
    d, _, _ = env
    (d / "research/charters/c.md").unlink()
    assert not P.is_approved("fam", ok_cfg())


def test_is_approved_false_on_unreadable_index(env):
    d, _, _ = env
    (d / "research/preregistration/INDEX.yaml").write_text("not: [a list of entries")
    assert not P.is_approved("fam", ok_cfg())


def test_integer_vs_bool_grid_values_not_conflated(env):
    d, _, write_index = env
    s = _spec(P.gates_hash(d), param_grid={"flag": [0, 1]}, n_planned_trials=2)
    (d / "research/preregistration/20261110_fam.yaml").write_text(yaml.safe_dump(s))
    write_index(fp=P.spec_hash(P.spec_from_dict(s)))
    assert P.is_approved("fam", {"flag": 1})
    assert not P.is_approved("fam", {"flag": True})


# ------------------------------------------------------------------ ledger enforcement

def test_exploratory_mode_logged_counted_not_promotable(env, tmp_path, monkeypatch):
    lroot = tmp_path / "ledger"
    lroot.mkdir()
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(lroot))
    monkeypatch.setenv(L.REPO_DIR_ENV, str(env[0]))

    @L.backtest_logged("fam", mode="registered", prereg="fam")
    def bt(speed, cap):
        return speed * cap

    assert bt(8, 1.0) == 8.0  # approved config runs (tmp repo is clean)
    with pytest.raises(L.UnapprovedPreregError):
        bt(9, 1.0)  # outside the grid
    assert len(L.trials()) == 1  # refused trial did not run, so is not recorded

    @L.backtest_logged("fam", mode="exploratory")
    def bt2(speed, cap):
        return speed * cap

    assert bt2(9, 1.0) == 9.0
    df = L.trials()
    row = df[df["mode"] == "exploratory"].iloc[0]
    assert pd.isna(row["preregistration_id"]) and len(df) == 2
    assert set(df["mode"]) == {"registered", "exploratory"}


def test_backtest_logged_registered_without_prereg_id_refused(env, tmp_path, monkeypatch):
    lroot = tmp_path / "ledger"
    lroot.mkdir()
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(lroot))

    @L.backtest_logged("fam", mode="registered", prereg=None)
    def bt(speed, cap):
        return 1

    with pytest.raises(L.UnapprovedPreregError):
        bt(8, 1.0)


def test_fixed_args_excluded_from_grid_check(env, tmp_path, monkeypatch):
    lroot = tmp_path / "ledger"
    lroot.mkdir()
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(lroot))
    monkeypatch.setenv(L.REPO_DIR_ENV, str(env[0]))

    @L.backtest_logged("fam", mode="registered", prereg="fam", fixed_args=("data",))
    def bt(data, speed, cap):
        return len(data)

    assert bt([1, 2], 8, 1.0) == 2


# ------------------------------------------------------------------ template / schema

def test_template_yaml_validates_against_schema():
    d = yaml.safe_load(_template_path().read_text())
    spec = P.spec_from_dict(d)  # all fields present, none unknown
    assert spec.post_seal_data_declaration == P.POST_SEAL_DECLARATION
    # the template is a blank form: it must NOT pass as an approved, complete spec
    assert P.validate_spec(spec) != []


def test_post_seal_declaration_required(env):
    d, spec, _ = env
    assert P.validate_spec(spec) == []
    bad = dataclasses.replace(spec, post_seal_data_declaration="")
    assert any("post_seal_data_declaration" in x for x in P.validate_spec(bad))
    bad2 = dataclasses.replace(spec, post_seal_data_declaration="Some data after 2026-09-30 was examined.")
    assert P.validate_spec(bad2)
    with pytest.raises(ValueError):
        raw = _spec(P.gates_hash(d))
        del raw["post_seal_data_declaration"]
        P.spec_from_dict(raw)


def test_validate_spec_flags_post_seal_dates_and_oversized_grid(env):
    _, spec, _ = env
    assert any("post-seal" in x for x in P.validate_spec(dataclasses.replace(spec, date_range=("2020-01-01", "2026-10-15"))))
    over = dataclasses.replace(spec, param_grid={"a": list(range(5)), "b": list(range(5))}, max_grid_size=12)
    assert any("max_grid_size" in x for x in P.validate_spec(over))


def test_load_index_rejects_duplicates_and_unknown_fields(tmp_path):
    e = {"family": "a", "kind": "yaml", "module_path": None, "yaml_path": "x", "fingerprint": "f",
         "freeze_commit": "c", "freeze_time_utc": "t", "ledger_path": None, "fingerprint_source": "none", "status": "DRAFT"}
    p = tmp_path / "i.yaml"
    p.write_text(yaml.safe_dump({"entries": [e, e]}))
    with pytest.raises(ValueError, match="duplicate"):
        P.load_index(p)
    p.write_text(yaml.safe_dump({"entries": [{**e, "bogus": 1}]}))
    with pytest.raises(ValueError, match="unknown"):
        P.load_index(p)
