"""Append-only trial ledger (P1-01). Every test redirects the ledger to tmp_path."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from firm.research import ledger as L
from firm.research import legacy_adapters as LA

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
CENSUS = REPO / "research" / "ledger" / "legacy_backfill.csv"


@pytest.fixture()
def root(tmp_path, monkeypatch):
    r = tmp_path / "ledger"
    r.mkdir()
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(r))
    monkeypatch.delenv("FIRM_LEDGER_ALLOW_DIRTY", raising=False)
    return r


def _git(d: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args],
        cwd=d, check=True, capture_output=True,
    )


@pytest.fixture()
def clean_repo(tmp_path, monkeypatch):
    d = tmp_path / "repo"
    (d / "src").mkdir(parents=True)
    (d / "runs").mkdir()
    (d / "data").mkdir()
    (d / "src" / "a.py").write_text("x = 1\n")
    (d / "runs" / "r.txt").write_text("r\n")
    (d / "data" / "d.txt").write_text("d\n")
    _git(d, "init", "-q")
    _git(d, "add", "-A")
    _git(d, "commit", "-q", "-m", "init")
    monkeypatch.setenv(L.REPO_DIR_ENV, str(d))
    return d


def mk(mode="legacy", **kw) -> L.TrialRecord:
    cfg = kw.pop("config", {"a": 1})
    base = {
        "trial_id": os.urandom(16).hex(), "family": "fam", "mode": mode, "config": cfg,
        "config_hash": L.config_hash(cfg), "code_commit": "", "data_snapshot_id": None, "seed": 1,
        "start": None, "end": None, "returns_path": None, "gross_sharpe": None, "net_sharpe": None,
        "periods_per_year": None, "sharpe_conversion": None, "n_obs": None, "skew": None, "kurt": None,
        "preregistration_id": None, "touched_holdout": False, "status": "completed", "error": None,
    }
    base.update(kw)
    return L.TrialRecord(**base)


def lines(root: Path) -> list[bytes]:
    return (root / "trials.jsonl").read_bytes().splitlines(keepends=True)


def test_chain_verifies_after_n_appends(root):
    for _ in range(20):
        L.record_trial(mk())
    rep = L.verify_chain()
    assert rep.ok and rep.n_rows == 20 and rep.first_bad_row is None


def test_tamper_detected(root):
    for _ in range(10):
        L.record_trial(mk())
    p = root / "trials.jsonl"
    ls = lines(root)
    b = bytearray(ls[5])
    i = b.index(b'"fam"') + 2
    b[i] = ord("g")
    ls[5] = bytes(b)
    p.write_bytes(b"".join(ls))
    rep = L.verify_chain()
    assert not rep.ok and rep.first_bad_row == 5


def test_deleted_middle_line_detected(root):
    for _ in range(10):
        L.record_trial(mk())
    ls = lines(root)
    del ls[4]
    (root / "trials.jsonl").write_bytes(b"".join(ls))
    rep = L.verify_chain()
    assert not rep.ok and rep.first_bad_row == 4


def test_rehashed_deletion_still_detected_by_seq(root):
    for _ in range(6):
        L.record_trial(mk())
    rows = [json.loads(x) for x in lines(root)]
    del rows[2]
    prev = "0" * 64
    out = []
    for r in rows:  # attacker re-hashes everything but cannot fix seq without rewriting it
        r["prev_hash"] = prev
        import hashlib

        prev = hashlib.sha256((prev + L.canonical_json(r["row"])).encode()).hexdigest()
        r["row_hash"] = prev
        out.append(json.dumps(r) + "\n")
    (root / "trials.jsonl").write_text("".join(out))
    rep = L.verify_chain()
    assert not rep.ok and rep.first_bad_row == 2


def test_no_mutation_api():
    for n in ("update_trial", "delete_trial", "remove_trial", "exclude_trial", "drop_trial", "filter_out"):
        assert not hasattr(L, n)
    assert set(L.__all__) == {
        "LEDGER_ROOT_ENV", "REPO_DIR_ENV", "Mode", "TrialRecord", "ChainReport", "TrialHandle",
        "canonical_json", "config_hash", "record_trial", "trials", "verify_chain",
        "backtest_logged", "run_trial", "DirtyTreeError", "UnapprovedPreregError",
        "LedgerNotProvisionedError", "LedgerCorruptError",
    }


def test_trials_returns_copy(root):
    L.record_trial(mk())
    df = L.trials()
    df.loc[:, "family"] = "zzz"
    assert (L.trials()["family"] == "fam").all()


def test_config_hash_canonical():
    assert L.config_hash({"a": 1, "b": 2}) == L.config_hash({"b": 2, "a": 1})
    with pytest.raises(TypeError):
        L.config_hash({"a": object()})
    with pytest.raises(ValueError):
        L.config_hash({"a": float("nan")})


def test_backtest_logged_records_failure(root, clean_repo):
    @L.backtest_logged("fam")
    def bad(x):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        bad(3)
    df = L.trials()
    assert len(df) == 1 and df.iloc[0]["status"] == "failed" and "boom" in df.iloc[0]["error"]
    assert df.iloc[0]["mode"] == "exploratory"


def test_backtest_logged_success_records_completed(root, clean_repo):
    @L.backtest_logged("fam")
    def ok(x, y=2):
        return x + y

    assert ok(1) == 3
    df = L.trials()
    assert len(df) == 1 and df.iloc[0]["status"] == "completed"
    assert df.iloc[0]["config"]["args"] == {"x": 1, "y": 2}


def test_run_trial_handle_stores_returns_and_stats(root, clean_repo):
    s = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))
    with L.run_trial("fam", {"p": 1}) as h:
        h.set(seed=7, periods_per_year=252, gross_sharpe=0.05)
        h.returns = s
    df = L.trials()
    r = df.iloc[0]
    assert r["seed"] == 7 and r["n_obs"] == 300 and r["returns_path"] == f"returns/{r['trial_id']}.parquet"
    assert (root / r["returns_path"]).exists()


def test_dirty_tree_raises_for_registered_only(root, clean_repo):
    (clean_repo / "runs" / "r.txt").write_text("changed\n")
    (clean_repo / "data" / "d.txt").write_text("changed\n")
    L.record_trial(mk("registered", preregistration_id="h" * 8))  # outputs dirt is ignored
    (clean_repo / "src" / "a.py").write_text("x = 2\n")
    with pytest.raises(L.DirtyTreeError):
        L.record_trial(mk("registered", preregistration_id="h" * 8))
    assert L.verify_chain().n_rows == 1


def test_registered_dirty_with_allow_env_downgrades(root, clean_repo, monkeypatch):
    (clean_repo / "src" / "a.py").write_text("x = 2\n")
    monkeypatch.setenv("FIRM_LEDGER_ALLOW_DIRTY", "1")
    L.record_trial(mk("registered", preregistration_id="abc"))
    r = L.trials().iloc[0]
    assert r["mode"] == "exploratory" and r["code_commit"].endswith("+dirty")
    assert r["preregistration_id"] is None
    assert r["config"]["_provenance"]["downgraded_from_prereg"] == "abc"


@pytest.mark.parametrize("mode", ["unregistered", "exploratory"])
def test_unregistered_dirty_tree_still_recorded(root, clean_repo, mode):
    (clean_repo / "src" / "a.py").write_text("x = 2\n")
    L.record_trial(mk(mode))
    r = L.trials().iloc[0]
    assert r["code_commit"].endswith("+dirty")
    assert len(r["config"]["_provenance"]["dirty_paths_digest"]) == 64


def test_clean_tree_commit_is_plain_sha(root, clean_repo):
    L.record_trial(mk("exploratory"))
    c = L.trials().iloc[0]["code_commit"]
    assert len(c) == 40 and "dirty" not in c


def test_refuses_to_autocreate_root(tmp_path, monkeypatch):
    missing = tmp_path / "nope"
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(missing))
    with pytest.raises(L.LedgerNotProvisionedError, match="mkdir"):
        L.record_trial(mk())
    assert not missing.exists()


def test_refuses_default_root_under_pytest(monkeypatch):
    monkeypatch.delenv(L.LEDGER_ROOT_ENV, raising=False)
    assert os.environ.get("PYTEST_CURRENT_TEST")
    with pytest.raises(L.LedgerNotProvisionedError, match="pytest"):
        L.record_trial(mk())


def test_sharpe_requires_periods_per_year(root):
    with pytest.raises(ValueError, match="periods_per_year"):
        L.record_trial(mk(gross_sharpe=0.1))
    with pytest.raises(ValueError, match="periods_per_year"):
        L.record_trial(mk(net_sharpe=0.1))
    L.record_trial(mk(gross_sharpe=0.1, periods_per_year=252))


def test_registered_requires_prereg(root, clean_repo):
    with pytest.raises(ValueError, match="preregistration_id"):
        L.record_trial(mk("registered"))
    with pytest.raises(ValueError, match="preregistration_id"):
        L.record_trial(mk("exploratory", preregistration_id="x"))
    with pytest.raises(ValueError, match="preregistration_id"):
        L.record_trial(mk("unregistered", preregistration_id="x"))


def test_config_hash_mismatch_rejected(root):
    r = mk()
    r = L.TrialRecord(**{**r.__dict__, "config_hash": "0" * 64})
    with pytest.raises(ValueError, match="config_hash"):
        L.record_trial(r)


def test_returns_written_before_row(root, monkeypatch):
    s = pd.Series([0.01, -0.02, 0.0])
    rec = mk(returns_path="returns/x.parquet")
    rec = L.TrialRecord(**{**rec.__dict__, "trial_id": "x"})
    monkeypatch.setattr(L, "_append_locked", lambda *a, **k: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        L.record_trial(rec, returns=s)
    assert (root / "returns" / "x.parquet").exists()
    assert not (root / "trials.jsonl").exists() or L.verify_chain().n_rows == 0
    monkeypatch.undo()
    monkeypatch.setenv(L.LEDGER_ROOT_ENV, str(root))
    L.record_trial(mk())
    rep = L.verify_chain()
    assert rep.ok and any("orphan" in w for w in rep.warnings)


def test_returns_path_must_match_trial_id(root):
    with pytest.raises(ValueError, match="returns_path"):
        L.record_trial(mk(returns_path="returns/other.parquet"), returns=pd.Series([0.1, 0.2]))


def test_partial_tail_refused_for_append(root):
    for _ in range(3):
        L.record_trial(mk())
    with open(root / "trials.jsonl", "ab") as f:
        f.write(b'{"seq": 3, "prev_ha')
    rep = L.verify_chain()
    assert rep.ok and rep.n_rows == 3 and rep.partial_tail
    with pytest.raises(L.LedgerCorruptError):
        L.record_trial(mk())


# ---------------------------------------------------------------- legacy adapters

EXPECTED = {
    "combination": 57, "pattern_ml": 104, "standalone_strategy": 11, "alt_premia": 10,
    "insider_cluster": 8, "S1": 4, "S2": 4, "S3": 5, "S4": 4, "s5": 3,
}


def _by_file():
    out: dict[str, list[L.TrialRecord]] = {}
    for r in LA.iter_legacy_rows(DOCS):
        out.setdefault(Path(r.source_file).name.removesuffix("_trial_history.json"), []).append(r)
    return out


def test_legacy_adapters_schemas():
    by = _by_file()
    totals = {k: sum(r.n_variants for r in v) for k, v in by.items()}
    assert totals == EXPECTED
    assert sum(totals.values()) == 210
    assert "allocation_forward_test" in LA.ADAPTERS and "allocation_forward_test" not in by
    for v in by.values():
        assert all(r.mode == "legacy" and not r.count_is_estimate for r in v)


def test_adapters_preserve_sharpes_per_file_key_map():
    by = _by_file()
    for stem, rows in by.items():
        for r in rows:
            sh = r.config.get("trial_daily_sharpes")
            if stem == "pattern_ml" or (stem == "combination" and "oos_sharpes_annualised" not in r.config):
                assert sh is None
                continue
            assert sh and r.periods_per_year == 252 and r.sharpe_conversion
    s2 = by["S2"][0]
    raw = json.loads((DOCS / "S2_trial_history.json").read_text())["entries"][0]
    assert s2.config["trial_daily_sharpes"] == raw["trial_daily_sharpes_cash_excess_governing"]
    assert s2.config["trial_daily_sharpes_bm2_excess_alt"] == raw["trial_daily_sharpes_bm2_excess_alt"]
    assert [r.source_entry_index for r in by["combination"]] == list(range(16))
    assert by["s5"][0].n_variants == 3


def test_unknown_stem_raises(tmp_path):
    (tmp_path / "zzz_trial_history.json").write_text(json.dumps({"entries": []}))
    with pytest.raises(LA.UnknownLegacyFileError):
        list(LA.iter_legacy_rows(tmp_path))


def test_pattern_ml_keeps_both_counts():
    rows = _by_file()["pattern_ml"]
    assert sum(r.n_variants for r in rows) == 104
    assert sum(r.config["n_configs"] for r in rows) == 13


def test_legacy_rows_deterministic():
    a = [(r.trial_id, r.config_hash) for r in LA.iter_legacy_rows(DOCS)]
    b = [(r.trial_id, r.config_hash) for r in LA.iter_legacy_rows(DOCS)]
    assert a == b


def test_backfill_idempotent(root):
    n1 = LA.backfill_legacy(DOCS, CENSUS)
    assert n1 > 0
    n2 = LA.backfill_legacy(DOCS, CENSUS)
    assert n2 == 0
    df = L.trials(mode="legacy")
    assert df["n_variants"].sum() >= 210
    assert L.verify_chain().ok


def test_backfill_equals_signed_census(root):
    if not CENSUS.exists():
        pytest.skip("research/ledger/legacy_backfill.csv (P0-05 signed census) absent")
    LA.backfill_legacy(DOCS, CENSUS)
    df = L.trials(mode="legacy")
    import csv

    total = sum(int(r["n_variants"]) for r in csv.DictReader(CENSUS.open()))
    assert int(df["n_variants"].sum()) == total
    est = df[df["count_is_estimate"]]
    assert int(est["n_variants"].sum()) == 253 and int(df[~df["count_is_estimate"]]["n_variants"].sum()) == 210


# ---------------------------------------------------------------- mirror sync

def _load_sync():
    spec = importlib.util.spec_from_file_location("sync_ledger_mirror", REPO / "scripts" / "sync_ledger_mirror.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_sync_mirror_extends_and_refuses_rewrite(root, tmp_path):
    sync = _load_sync()
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    s = pd.Series([0.01, 0.02, -0.01])
    rec = mk("exploratory", returns_path="returns/abc.parquet", n_obs=3)
    rec = L.TrialRecord(**{**rec.__dict__, "trial_id": "abc"})
    L.record_trial(mk())
    L.record_trial(rec, returns=s)
    assert sync.sync(mirror) == 2
    assert (mirror / "trials.jsonl").read_bytes() == (root / "trials.jsonl").read_bytes()
    man = [json.loads(x) for x in (mirror / "returns_manifest.jsonl").read_text().splitlines()]
    assert len(man) == 1 and man[0]["trial_id"] == "abc" and man[0]["n_obs"] == 3 and len(man[0]["sha256"]) == 64
    L.record_trial(mk())
    assert sync.sync(mirror) == 3
    assert len((mirror / "returns_manifest.jsonl").read_text().splitlines()) == 1  # idempotent
    (mirror / "trials.jsonl").write_bytes(b"x" + (mirror / "trials.jsonl").read_bytes())
    with pytest.raises(sync.MirrorError):
        sync.sync(mirror)
