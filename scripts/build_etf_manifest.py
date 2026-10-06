"""Build the ETF data manifest (P2-02) for the 14 universe ETFs: bars, dividends, USDILS, delisted list. Pre-seal, via data_access.

  python scripts/build_etf_manifest.py [--asof 2026-09-30] [--out research/data_manifests]

Writes ``etf_<snapshot_id>.json`` (never overwrites an existing manifest), records the sha256 of the universe yaml, then runs ``verify_manifest``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import yaml

import firm.research  # noqa: F401  (seal / wrapped entry-point check)
from firm.data.etf_loader import DEFAULT_DATA_ROOT
from firm.data.manifest import build_manifest, snapshot_id, verify_manifest, write_manifest
from firm.research import seal

LICENCE = "EODHD licensed; manifest only, data not committed"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asof", type=dt.date.fromisoformat, default=None)
    ap.add_argument("--universe", type=Path, default=ROOT / "config" / "universe_etf.yaml")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "research" / "data_manifests")
    a = ap.parse_args(argv)
    asof = a.asof or seal.max_research_date()
    seal.check_asof(asof, what="build_etf_manifest")
    root = Path(a.data_root or (ROOT / DEFAULT_DATA_ROOT))
    syms = [i["symbol"] for i in yaml.safe_load(a.universe.read_text(encoding="utf-8"))["instruments"]]
    want = [f"etfs_full/{s}.parquet" for s in syms] + [f"corporate_actions/dividends/{s}.parquet" for s in syms]
    want += ["forex/USDILS.parquet"]
    present = [root / w for w in want if (root / w).is_file()]
    missing = [w for w in want if not (root / w).is_file()]
    df = build_manifest(present, root=root, dataset="eodhd_etf_universe", licence=LICENCE)
    delisted = root / "symbols_delisted.parquet"  # no date column: hashed as bytes only (data_access would refuse it)
    df = pd.concat([df, build_manifest([delisted], root=root, dataset="eodhd_etf_universe", licence=LICENCE, meta=False)], ignore_index=True)
    df = df.sort_values("path").reset_index(drop=True)
    sid = snapshot_id(df)
    out = a.out / f"etf_{sid}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")
    extra = {"asof": asof.isoformat(), "data_root": "data/research/eodhd",
             "instruments": syms, "universe_yaml": "config/universe_etf.yaml",
             "universe_yaml_sha256": hashlib.sha256(a.universe.read_bytes()).hexdigest(),
             "note": "no post-seal data was read; hashing is of file bytes; filename key = snapshot_id"}
    write_manifest(df, out, missing=missing, extra=extra)
    bad = verify_manifest(out, root)
    print(f"{out} snapshot_id={sid} files={len(df)} missing={missing} verify={bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
