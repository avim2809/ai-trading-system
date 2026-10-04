"""Hashed data manifests and ``data_snapshot_id`` (ticket P2-02).

Same columns, same order as ``scripts/build_review_package.build_manifest`` so P1-11 can fold both into
the review bundle. Only hashes and per-file counts are recorded; no vendor data is committed.
Parquet row counts and date bounds are read through ``firm.research.data_access`` (seal-checked, pre-seal rows only).
Not imported by any live module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = ["dataset", "licence", "path", "bytes", "sha256", "rows", "first_date", "last_date"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parquet_meta(path: Path) -> tuple[int | None, str | None, str | None]:
    from firm.research import data_access, seal

    df = data_access.read_parquet(path, asof=seal.max_research_date())
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"])
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = df.index.to_series()
    else:
        return len(df), None, None
    if not len(dates):
        return 0, None, None
    return len(df), dates.min().date().isoformat(), dates.max().date().isoformat()


def build_manifest(paths: Iterable[Path], *, root: Path, dataset: str, licence: str, meta: bool = True) -> pd.DataFrame:
    """One row per file; ``path`` is posix-relative to ``root``. ``meta=False`` skips row/date columns (left None)."""
    rows = []
    for p in sorted(Path(x) for x in paths):
        n, first, last = _parquet_meta(p) if meta and p.suffix == ".parquet" else (None, None, None)
        rows.append({"dataset": dataset, "licence": licence, "path": p.resolve().relative_to(Path(root).resolve()).as_posix(),
                     "bytes": p.stat().st_size, "sha256": sha256_file(p), "rows": n, "first_date": first, "last_date": last})
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def snapshot_id(manifest: pd.DataFrame) -> str:
    """sha256 over the sorted ``(path, sha256)`` pairs."""
    pairs = sorted(zip(manifest["path"], manifest["sha256"], strict=True))
    return hashlib.sha256(json.dumps(pairs, separators=(",", ":")).encode()).hexdigest()


def write_manifest(df: pd.DataFrame, out: Path, *, missing: list[str] | None = None, extra: dict | None = None) -> None:
    """JSON: ``{snapshot_id, columns, rows, missing, ...extra}`` (``missing`` = requested histories absent on disk)."""
    clean = df.astype(object).where(df.notna(), None)
    payload = {"snapshot_id": snapshot_id(df), "columns": list(df.columns), "rows": clean.to_dict(orient="records"),
               "missing": sorted(missing or [])}
    payload.update(extra or {})
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> tuple[pd.DataFrame, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame(payload["rows"], columns=payload["columns"]), payload


def verify_manifest(manifest_path: Path, root: Path) -> list[str]:
    """Paths whose file is missing or whose sha256 differs from the manifest (empty list = intact)."""
    df, _ = read_manifest(manifest_path)
    bad = []
    for rel, digest in zip(df["path"], df["sha256"], strict=True):
        p = Path(root) / rel
        if not p.is_file() or sha256_file(p) != digest:
            bad.append(rel)
    return bad
