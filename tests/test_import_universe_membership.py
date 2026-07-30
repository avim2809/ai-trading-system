"""Tests for scripts/import_universe_membership.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from firm.data.cache import ParquetCache
from firm.data.schemas import UNIVERSE_COLUMNS
from firm.runtime import load_universe_membership

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "import_universe_membership.py"


def _run_import(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )


class TestImportUniverseMembership:
    def test_imports_csv_to_cache(self, tmp_path):
        csv = tmp_path / "membership.csv"
        csv.write_text(
            "index,symbol,added_date,removed_date\n"
            "SP500,AAPL,2010-01-01,\n"
            "SP500,OLD,2005-01-01,2015-06-01\n",
            encoding="utf-8",
        )
        cache_dir = tmp_path / "cache"
        proc = _run_import(
            str(csv),
            "--cache-dir", str(cache_dir),
            "--index", "SP500",
        )
        assert proc.returncode == 0, proc.stderr

        from firm.config import Settings

        settings = Settings()
        settings.data.cache_dir = str(cache_dir)
        df = load_universe_membership(settings)
        assert df is not None
        assert set(df.columns) == set(UNIVERSE_COLUMNS)
        assert set(df["symbol"]) == {"AAPL", "OLD"}

    def test_merge_with_existing(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache = ParquetCache(cache_dir)
        cache.put(
            "combined/universe_membership",
            pd.DataFrame({
                "index": ["SP500"],
                "symbol": ["MSFT"],
                "added_date": [pd.Timestamp("2012-01-01")],
                "removed_date": [pd.NaT],
            }),
        )
        csv = tmp_path / "new.csv"
        csv.write_text(
            "symbol,added_date,removed_date\nAAPL,2010-01-01,\n",
            encoding="utf-8",
        )
        proc = _run_import(
            str(csv),
            "--cache-dir", str(cache_dir),
            "--index", "SP500",
            "--merge",
        )
        assert proc.returncode == 0, proc.stderr

        merged = cache.get("combined/universe_membership")
        assert merged is not None
        assert set(merged["symbol"]) == {"MSFT", "AAPL"}

    def test_dry_run_does_not_write(self, tmp_path):
        csv = tmp_path / "membership.csv"
        csv.write_text(
            "index,symbol,added_date,removed_date\nSP500,XOM,2010-01-01,\n",
            encoding="utf-8",
        )
        cache_dir = tmp_path / "cache"
        proc = _run_import(str(csv), "--cache-dir", str(cache_dir), "--dry-run")
        assert proc.returncode == 0, proc.stderr
        assert not list(cache_dir.glob("*.parquet"))

    def test_missing_columns_fails(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("symbol,added_date\nAAPL,2010-01-01\n", encoding="utf-8")
        proc = _run_import(str(csv), "--cache-dir", str(tmp_path / "cache"))
        assert proc.returncode == 1
