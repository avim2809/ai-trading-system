"""scripts/build_review_package.py helpers on temporary files (no real data)."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import build_review_package as brp  # noqa: E402


def test_run_dirs_are_renamed_out_of_the_gitignored_runs_folder():
    assert brp._git_safe(Path("runs/S1/returns.parquet")) == Path("run_outputs/S1/returns.parquet")
    assert brp._git_safe(Path("freeze_shortlist.py")) == Path("freeze_shortlist.py")


def test_sha256_and_parquet_meta(tmp_path):
    f = tmp_path / "x.parquet"
    pd.DataFrame({"date": pd.to_datetime(["2001-02-03", "1999-01-04", "2005-06-07"]), "v": [1, 2, 3]}).to_parquet(f)
    assert brp.sha256(f) == hashlib.sha256(f.read_bytes()).hexdigest()
    assert brp.parquet_meta(f) == (3, "1999-01-04", "2005-06-07")
    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not parquet")
    assert brp.parquet_meta(bad) == (None, None, None)


def test_scratch_roots_resolve_under_the_scratch_dir(tmp_path):
    assert brp.resolve("scratch:data", tmp_path) == tmp_path / "data"
    assert brp.resolve("data/research/fred", tmp_path) == brp.ROOT / "data/research/fred"
