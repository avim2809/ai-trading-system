"""Adapters from the frozen ``docs/*_trial_history.json`` files to ledger rows (ticket P1-01).

One adapter per schema; ``ADAPTERS`` is keyed by file stem (the file name minus
``_trial_history.json``) and an unknown stem raises, so a new file can never be skipped silently.
Legacy files hold per-trial daily Sharpe scalars, not return series: they enter as raw counts
(``n_variants``) with the Sharpes kept in ``config``. ``trial_id`` is a deterministic uuid5 of
(source_file, entry index) so backfills are reproducible; ``touched_holdout=True`` for every
legacy row (pre-seal data is burned; conservative).
"""

from __future__ import annotations

import csv
import json
import logging
import math
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from firm.research.ledger import TrialRecord, config_hash, record_trial, trials

log = logging.getLogger(__name__)

SUFFIX = "_trial_history.json"
CENSUS_REL = "research/ledger/legacy_backfill.csv"
_NS = uuid.UUID("5b0f6f58-2c8e-4c43-9a3e-4a6a1f0d7e11")
_DAILY = "daily series (trial_daily_sharpes), no conversion"


class UnknownLegacyFileError(ValueError):
    """A ``*_trial_history.json`` file has no registered adapter."""


def _rec(stem: str, family: str, idx: int, n_variants: int, config: dict, *, sharpes: bool,
         date: str | None, conversion: str = _DAILY) -> TrialRecord:
    src = f"docs/{stem}{SUFFIX}"
    return TrialRecord(
        trial_id=uuid.uuid5(_NS, f"{src}#{idx}").hex, family=family, mode="legacy", config=config,
        config_hash=config_hash(config), code_commit="legacy", data_snapshot_id=None, seed=None,
        start=date, end=date, returns_path=None, gross_sharpe=None, net_sharpe=None,
        periods_per_year=252 if sharpes else None, sharpe_conversion=conversion if sharpes else None,
        n_obs=None, skew=None, kurt=None, preregistration_id=None, touched_holdout=True,
        status="completed", error=None, count_is_estimate=False, n_variants=n_variants,
        source_file=src, source_entry_index=idx,
    )


def _schema1(stem: str, data: dict) -> Iterator[TrialRecord]:
    """{family, entries[{date, fingerprint, n_trials, trials, trial_daily_sharpes, tier|tiers|survivors}]}.

    S2 variant: ``trial_daily_sharpes_cash_excess_governing`` is the governing series; the bm2 series is
    kept as an alternative in ``config``.
    """
    family = data.get("family", stem)
    for i, e in enumerate(data["entries"]):
        cfg: dict[str, Any] = {"fingerprint": e.get("fingerprint"), "trials": e.get("trials", [])}
        if "trial_daily_sharpes" in e:
            cfg["trial_daily_sharpes"] = e["trial_daily_sharpes"]
        elif "trial_daily_sharpes_cash_excess_governing" in e:
            cfg["trial_daily_sharpes"] = e["trial_daily_sharpes_cash_excess_governing"]
            cfg["trial_daily_sharpes_bm2_excess_alt"] = e.get("trial_daily_sharpes_bm2_excess_alt")
        else:
            raise UnknownLegacyFileError(f"{stem}: entry {i} has no Sharpe list key")
        for k in ("tier", "tiers", "survivors", "official_tier_conditional_on_A7",
                  "tiers_conditional_on_A7_all_variants_descriptive", "note"):
            if k in e:
                cfg[k] = e[k]
        if "note" in data:
            cfg["file_note"] = data["note"]
        yield _rec(stem, family, i, int(e["n_trials"]), cfg, sharpes=bool(cfg["trial_daily_sharpes"]),
                   date=e.get("date"))


def _schema2_combination(stem: str, data: dict) -> Iterator[TrialRecord]:
    for i, e in enumerate(data["entries"]):
        cfg: dict[str, Any] = {"source": e["source"], "candidates": e["candidates"]}
        sharpes = False
        conv = _DAILY
        if "oos_sharpes" in e:  # annualised in the file; per-period conversion per the signed census
            cfg["oos_sharpes_annualised"] = e["oos_sharpes"]
            cfg["trial_daily_sharpes"] = [v / math.sqrt(252) for v in e["oos_sharpes"].values()]
            sharpes, conv = True, "SR_d = SR_a/sqrt(252) (file holds annualised oos_sharpes)"
        yield _rec(stem, "combination", i, int(e["n_trials"]), cfg, sharpes=sharpes, date=e.get("date"),
                   conversion=conv)


def _schema3_pattern_ml(stem: str, data: dict) -> Iterator[TrialRecord]:
    for i, e in enumerate(data["entries"]):
        cfg = {
            "source": e["source"], "candidates": e["candidates"], "n_folds": e["n_folds"],
            "n_candidates": e["n_candidates"], "n_configs": int(e["n_candidates"]),
            "count_convention": "n_variants = folds x candidates; n_configs = candidate configs",
            "reported_pbo": e.get("reported_pbo"),
            "reported_deflated_sharpe": e.get("reported_deflated_sharpe"), "verdict": e.get("verdict"),
        }
        yield _rec(stem, "pattern_ml", i, int(e["n_trials"]), cfg, sharpes=False, date=e.get("date"))


def _schema4_s5(stem: str, data: dict) -> Iterator[TrialRecord]:
    for i, e in enumerate(data["entries"]):
        cfg = {
            "fingerprint": e.get("fingerprint"), "variant_names": e["variant_names"],
            "trial_daily_sharpes": e["trial_daily_sharpes"], "prior_trials_used": e.get("prior_trials_used"),
            "tier": e.get("tier"),
        }
        yield _rec(stem, "s5", i, len(e["variant_names"]), cfg, sharpes=bool(e["trial_daily_sharpes"]),
                   date=(e.get("appended_at") or "")[:10] or None)


def _allocation_forward_test(stem: str, data: dict) -> Iterator[TrialRecord]:
    """Forward-test state record, not a trial list: ZERO variants (explicitly listed so it is not 'unknown')."""
    return iter(())


ADAPTERS: dict[str, Callable[[str, dict], Iterator[TrialRecord]]] = {
    "S1": _schema1, "S2": _schema1, "S3": _schema1, "S4": _schema1,
    "alt_premia": _schema1, "insider_cluster": _schema1, "standalone_strategy": _schema1,
    "combination": _schema2_combination, "pattern_ml": _schema3_pattern_ml, "s5": _schema4_s5,
    "allocation_forward_test": _allocation_forward_test,
}


def iter_legacy_rows(docs_dir: Path) -> Iterator[TrialRecord]:
    """Rows for every ``*_trial_history.json`` in ``docs_dir`` (sorted names, then entry index)."""
    for p in sorted(Path(docs_dir).glob(f"*{SUFFIX}")):
        stem = p.name[: -len(SUFFIX)]
        if stem not in ADAPTERS:
            raise UnknownLegacyFileError(f"no adapter for {p.name}; add one before backfilling")
        yield from ADAPTERS[stem](stem, json.loads(p.read_text()))


def iter_census_estimate_rows(csv_path: Path) -> Iterator[TrialRecord]:
    """Estimate-only rows (``count_is_estimate=true``) of the signed census; index = 0-based data row."""
    with open(csv_path, newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if r["count_is_estimate"].strip().lower() != "true":
                continue
            cfg = {k: r[k] for k in ("sub_family", "description", "count_convention", "evidence_ref", "notes")}
            hold = r["holdout_touched"].strip().lower()
            yield TrialRecord(
                trial_id=uuid.uuid5(_NS, f"{CENSUS_REL}#{i}").hex, family=r["family"], mode="legacy",
                config=cfg, config_hash=config_hash(cfg), code_commit="legacy", data_snapshot_id=None,
                seed=None, start=r["date_start"] or None, end=r["date_end"] or None, returns_path=None,
                gross_sharpe=None, net_sharpe=None, periods_per_year=None, sharpe_conversion=None,
                n_obs=None, skew=None, kurt=None, preregistration_id=None,
                touched_holdout=not hold.startswith("no"), status="completed", error=None,
                count_is_estimate=True, n_variants=int(r["n_variants"]),
                source_file=CENSUS_REL, source_entry_index=i,
            )


def backfill_legacy(docs_dir: Path, census_csv: Path) -> int:
    """Idempotently append legacy rows to the ledger; returns the number appended."""
    existing = trials(mode="legacy")
    seen = set(zip(existing["source_file"], existing["source_entry_index"]))
    n = 0
    for rec in [*iter_legacy_rows(docs_dir), *iter_census_estimate_rows(census_csv)]:
        if (rec.source_file, rec.source_entry_index) in seen:
            continue
        record_trial(rec)
        n += 1
    log.info("legacy backfill appended %d rows", n)
    return n
