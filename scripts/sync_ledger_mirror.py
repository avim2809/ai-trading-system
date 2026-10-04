#!/usr/bin/env python3
"""Copy the canonical host ledger into the tracked mirror ``research/ledger/`` (ticket P1-01).

Takes the ledger flock, verifies the chain, refuses if the existing mirror is not a byte-prefix of the
canonical file, then writes ``trials.jsonl`` and appends a returns-manifest row (trial_id, sha256, n_obs)
for each returns parquet not yet listed. Parquet files are NOT copied (licensed-data policy). It does not
``git add`` or commit. (The P1-12 inbox ingest is added by that ticket.)
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from firm.research import ledger as L

log = logging.getLogger("sync_ledger_mirror")


class MirrorError(RuntimeError):
    pass


def sync(mirror_dir: Path) -> int:
    root = L._root(for_write=True)
    src = root / L.LEDGER_FILE
    if not src.exists():
        raise MirrorError(f"{src} does not exist")
    with open(src, "rb") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            rep = L.verify_chain(src)
            if not rep.ok or rep.partial_tail:
                raise MirrorError(f"canonical chain not clean: {rep}")
            data = lock.read()
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    mirror_dir.mkdir(parents=True, exist_ok=True)
    dst = mirror_dir / "trials.jsonl"
    if dst.exists():
        old = dst.read_bytes()
        if not data.startswith(old):
            raise MirrorError("existing mirror is not a byte-prefix of the canonical ledger (rewritten/truncated?)")
    manifest = mirror_dir / "returns_manifest.jsonl"
    listed = set()
    if manifest.exists():
        listed = {json.loads(x)["trial_id"] for x in manifest.read_text().splitlines() if x}
    new_rows = []
    for ln in data.splitlines():
        row = json.loads(ln)["row"]
        if row.get("returns_path") and row["trial_id"] not in listed:
            p = root / row["returns_path"]
            if not p.exists():
                raise MirrorError(f"returns file missing for trial {row['trial_id']}: {p}")
            new_rows.append({
                "trial_id": row["trial_id"], "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "n_obs": row.get("n_obs"),
            })
    tmp = dst.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dst)
    if new_rows:
        with open(manifest, "a") as f:
            f.writelines(json.dumps(r, sort_keys=True) + "\n" for r in new_rows)
    return rep.n_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mirror-dir", type=Path, default=REPO / "research" / "ledger")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = sync(a.mirror_dir)
    print(f"mirror now has {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
