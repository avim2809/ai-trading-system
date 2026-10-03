"""Schema and consistency checks for research/ledger/legacy_backfill.csv (P0-05)."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "research" / "ledger" / "legacy_backfill.csv"

COLUMNS = [
    "family", "sub_family", "description", "date_start", "date_end",
    "n_variants", "n_variants_alt_convention", "count_convention",
    "count_is_estimate", "holdout_touched", "trial_daily_sharpes_known",
    "sharpe_frequency", "sharpe_values_per_period", "conversion",
    "source_file", "source_entry_index", "evidence_ref", "notes",
]
LEDGER_TOTAL = 210
GROSS_TOTAL = 463


@pytest.fixture(scope="module")
def rows() -> list[dict[str, str]]:
    with CSV_PATH.open() as fh:
        return list(csv.DictReader(fh))


def test_schema(rows):
    assert list(rows[0].keys()) == COLUMNS
    for r in rows:
        assert r["count_is_estimate"] in {"true", "false"}
        assert int(r["n_variants"]) >= 0
        assert r["family"]


def test_ledger_total_is_210(rows):
    ledger = [r for r in rows if r["count_is_estimate"] == "false"]
    assert sum(int(r["n_variants"]) for r in ledger) == LEDGER_TOTAL


def test_gross_total_is_463_and_gann_not_understated(rows):
    assert sum(int(r["n_variants"]) for r in rows) == GROSS_TOTAL
    gann = [r for r in rows if r["family"] == "gann"]
    assert len(gann) == 1 and int(gann[0]["n_variants"]) >= 145


def test_ledger_rows_match_json_counts(rows):
    by_file: dict[str, int] = {}
    for r in rows:
        if r["count_is_estimate"] == "false":
            by_file[r["source_file"]] = by_file.get(r["source_file"], 0) + int(r["n_variants"])
    for rel, total in by_file.items():
        path = ROOT / rel
        assert path.exists(), rel
        d = json.loads(path.read_text())
        counter = d.get("cumulative_trials", d.get("cumulative_trials_through_last_entry"))
        if counter is not None:
            assert total == counter, rel
        elif "n_trials" in d["entries"][0]:
            assert total == sum(e["n_trials"] for e in d["entries"]), rel
        elif rel.endswith("s5_trial_history.json"):
            assert total == sum(len(e["variant_names"]) for e in d["entries"])
        else:
            assert total == 0, rel
    assert sum(by_file.values()) == LEDGER_TOTAL


def test_entry_rows_unique_and_indexed(rows):
    keys = [(r["source_file"], r["source_entry_index"]) for r in rows if r["count_is_estimate"] == "false"]
    assert all(k[1] != "" for k in keys)
    assert len(keys) == len(set(keys))
    assert sum(1 for k in keys if k[0].endswith("combination_trial_history.json")) == 16
    assert sum(1 for k in keys if k[0].endswith("pattern_ml_trial_history.json")) == 4


def test_pattern_ml_alt_convention_is_13(rows):
    pml = [r for r in rows if r["family"] == "pattern_ml"]
    assert sum(int(r["n_variants"]) for r in pml) == 104
    assert sum(int(r["n_variants_alt_convention"]) for r in pml) == 13


def test_estimate_rows_have_evidence_and_method(rows):
    for r in (x for x in rows if x["count_is_estimate"] == "true"):
        assert r["evidence_ref"].strip(), r["family"]
        assert r["notes"].strip(), r["family"]
        assert r["source_file"].strip(), r["family"]


def test_sources_exist_for_tracked_files(rows):
    for r in rows:
        for part in r["source_file"].split(";"):
            part = part.strip()
            if part.startswith(("docs/", "scripts/", "research/")):
                assert (ROOT / part).exists(), part
    tracked_ref = "origin/research/gann-archive"
    assert any(r["source_file"] == tracked_ref for r in rows if r["family"] == "gann")


def test_sharpes_are_per_period(rows):
    for r in rows:
        if not r["sharpe_values_per_period"]:
            continue
        vals = [float(v) for v in r["sharpe_values_per_period"].split(";")]
        assert all(abs(v) < 0.5 for v in vals), (r["family"], vals)
        if r["sharpe_frequency"] == "annualised":
            assert r["conversion"] == "SR_d = SR_a/sqrt(252)"
        else:
            assert r["sharpe_frequency"] == "daily" and r["conversion"] == "none"


def test_csv_matches_seed_script():
    out = subprocess.run(
        ["python3", str(ROOT / "scripts" / "seed_legacy_census.py")],
        capture_output=True, text=True, check=True, env={"PYTHONDONTWRITEBYTECODE": "1"},
    ).stdout
    assert out == CSV_PATH.read_text()
