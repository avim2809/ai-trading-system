#!/usr/bin/env python3
"""Copy the canonical host ledger into the tracked mirror ``research/ledger/`` (ticket P1-01).

Takes the ledger flock, verifies the chain, refuses if the existing mirror is not a byte-prefix of the
canonical file, then writes ``trials.jsonl`` and appends a returns-manifest row (trial_id, sha256, n_obs)
for each returns parquet not yet listed. Parquet files are NOT copied (licensed-data policy). It does not
``git add`` or commit. With ``--ingest-inbox`` (P1-12) it first appends every
``inbox/*.jsonl`` capture line (backtests run in the firm-api or any unarmed process) as an
``unregistered`` trial, idempotently keyed on ``(source_file="inbox/<tag>.jsonl", line index)``;
inbox files are never deleted so the key stays stable. Owner/ops step.
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


def ingest_inbox() -> int:
    """Append not-yet-ingested inbox lines to the ledger; return the number of new rows."""
    from firm.research import capture

    root = L._root(for_write=True)
    inbox = root / "inbox"
    if not inbox.is_dir():
        return 0
    done: set[tuple[str, int]] = set()
    df = L.trials(mode="unregistered")
    for sf, ix in zip(df["source_file"], df["source_entry_index"], strict=True):
        if isinstance(sf, str) and ix == ix and ix is not None:
            done.add((sf, int(ix)))
    n_new = 0
    for f in sorted(inbox.glob("*.jsonl")):
        sf = f"inbox/{f.name}"
        for i, ln in enumerate(f.read_bytes().split(b"\n")):
            if not ln.strip() or (sf, i) in done:
                continue
            try:
                line = json.loads(ln)
            except ValueError:
                log.warning("inbox %s line %d is not valid JSON (partial write?); skipped", sf, i)
                continue
            capture.record_inbox_line(line, source_file=sf, index=i)
            n_new += 1
    return n_new


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mirror-dir", type=Path, default=REPO / "research" / "ledger")
    ap.add_argument("--ingest-inbox", action="store_true", help="first ingest inbox/*.jsonl capture lines (P1-12)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.ingest_inbox:
        print(f"ingested {ingest_inbox()} new inbox rows")
    n = sync(a.mirror_dir)
    print(f"mirror now has {n} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
