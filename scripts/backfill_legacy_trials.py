#!/usr/bin/env python3
"""Append the legacy trial rows (11 docs/*_trial_history.json + signed census estimates) to the ledger.

Idempotent. Honours FIRM_RESEARCH_LEDGER_ROOT; the real host ledger is only written when it is unset
and the root has been provisioned by the owner. Never edits the frozen JSON files.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from firm.research.ledger import verify_chain
from firm.research.legacy_adapters import backfill_legacy

log = logging.getLogger("backfill_legacy_trials")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=Path, default=REPO / "docs")
    ap.add_argument("--census", type=Path, default=REPO / "research" / "ledger" / "legacy_backfill.csv")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    n = backfill_legacy(a.docs, a.census)
    rep = verify_chain()
    log.info("appended %d rows; chain ok=%s rows=%d", n, rep.ok, rep.n_rows)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
