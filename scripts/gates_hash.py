#!/usr/bin/env python3
"""Compute (and, owner-run, record) the integrity hash of config/gates.yaml.

The hash is sha256 of the canonical JSON of the parsed YAML with the top-level ``meta`` block removed (a file cannot contain its
own hash, and ``meta`` holds the freeze commit and signature stamp). The expected value lives OUTSIDE gates.yaml, on the signed line
of docs/gate_deviation_register_2026_10.md, so a threshold cannot be edited and a hash field updated in the same file.

    python scripts/gates_hash.py                 # print the hash
    python scripts/gates_hash.py --check         # compare with the value recorded in the register (exit 1 on mismatch)
    python scripts/gates_hash.py --record        # OWNER ONLY: write the hash into the register's signed line

Agents must not run ``--record``: recording the hash is the owner's second signature step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import yaml

log = logging.getLogger("gates_hash")

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "config" / "gates.yaml"
REGISTER = ROOT / "docs" / "gate_deviation_register_2026_10.md"
HASH_LINE = re.compile(
    r"^(sha256 of config/gates\.yaml with the meta block removed \(filled in at freeze\): )(\S.*)$", re.MULTILINE
)


def canonical_hash(path: Path = GATES) -> str:
    """sha256 of canonical JSON (sorted keys, compact separators) of the YAML without its ``meta`` block."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    data.pop("meta", None)
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def recorded_hash(register: Path = REGISTER) -> str | None:
    """The hash recorded on the signed line, or None while it is still the blank placeholder."""
    m = HASH_LINE.search(Path(register).read_text(encoding="utf-8"))
    if not m:
        raise ValueError(f"signed hash line not found in {register}")
    value = m.group(2).strip()
    return None if set(value) <= {"_"} else value


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--record", action="store_true", help="owner only: write the hash into the register")
    args = ap.parse_args(argv)

    h = canonical_hash()
    if args.record:
        text = REGISTER.read_text(encoding="utf-8")
        m = HASH_LINE.search(text)
        if not m:
            log.error("signed hash line not found in %s", REGISTER)
            return 2
        if set(m.group(2).strip()) > {"_"}:
            log.error("a hash is already recorded (%s); a change needs a new signed register version", m.group(2).strip())
            return 2
        REGISTER.write_text(HASH_LINE.sub(lambda mm: mm.group(1) + h, text, count=1), encoding="utf-8")
        log.info("recorded %s in %s", h, REGISTER)
        print(h)
        return 0
    if args.check:
        rec = recorded_hash()
        if rec is None:
            log.warning("no hash recorded yet in the register (owner step pending); computed %s", h)
            return 1
        ok = rec == h
        log.info("recorded %s | computed %s | %s", rec, h, "MATCH" if ok else "MISMATCH")
        return 0 if ok else 1
    print(h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
