"""Frozen bar-cleaning rule for EODHD research data (2026-09-30).

The insider-cluster evaluation (docs/insider_cluster_verdict_2026_09.md) showed that
raw EODHD bars can fake a pass: phantom holiday bars (SMLP 2015-12-25 at $0.0002),
recurring scale errors (XBKS, ~250x), unadjusted reverse splits (ACRX, CERN) and
zero-volume garbage quotes (QPAC). Every pre-registration on EODHD prices cites
``cleaning_fingerprint()`` and runs its bars through ``clean_bars`` before any
return is computed. The rule is fixed here, before any shortlist test ran, and
is not tuned per candidate.

Rules, applied in order:

1. Drop bars with a missing or non-positive open, close or adjusted_close.
2. Drop bars with zero or missing volume (no trade: the quote is not executable).
   Skipped for asset="nav": mutual-fund NAV series (e.g. VFITX) carry no volume.
   They serve only as benchmark/cash proxies before an ETF existed, never as a
   traded holding.
3. Equities/ETFs/NAV funds: drop bars whose date is not an exchange session, taken as the
   dates of the SPY series. Crypto trades every day: no calendar filter.
4. Spike reversal: a bar whose adjusted close moves beyond +100% / -50% against
   the previous kept bar, and where one of the next 5 kept bars is back within
   +/-25% of that previous level, is a bad print. The bars from the jump up to
   the reverting bar are dropped.
5. Unexplained up-jump: a one-day adjusted move above +150% (x2.5) that does
   not revert starts a new ``segment`` (typically an unadjusted reverse split,
   e.g. CERN 2008-12-18 x3.93, ACRX x20). Downstream code must not compute a
   return across a segment boundary. One-day drops are kept as real.

Every ambiguous extreme is resolved against long books: real non-reverting
gains above x2.5 are lost, and fake drops (an unadjusted forward split) stay.
Report long-short or short-leg results with that bias in mind.

    python scripts/eodhd_clean.py --report [--dir prices|us_universe|etfs|crypto]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

EODHD = Path(__file__).resolve().parents[1] / "data" / "research" / "eodhd"

CLEANING_RULES = {
    "version": 2,
    "frozen_at": "2026-09-30T18:23:17Z",  # UTC time of commit aa22da3 (v1: 18:07:57Z, commit 53161f6)
    "v2_change": "asset='nav' skips the volume rule (before any shortlist prereg froze)",
    "drop_nonpositive_or_missing_price": ["open", "close", "adjusted_close"],
    "drop_zero_or_missing_volume": True,
    "equity_calendar": "dates of data/research/eodhd/etfs_full/SPY.parquet (from 1993-01-29)",
    "crypto_calendar": "none (all days)",
    "spike_reversal": {"jump_up": 1.0, "jump_down": -0.5, "revert_within_bars": 5, "revert_band": 0.25},
    "segment_break_up_jump": 1.5,
    "down_jumps": "kept as real (bias resolved against long books)",
}


def cleaning_fingerprint() -> str:
    return hashlib.sha256(json.dumps(CLEANING_RULES, sort_keys=True).encode()).hexdigest()


def _spikes_and_breaks(adj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (drop, segment_start) over ``adj`` per rules 4 and 5."""
    sp = CLEANING_RULES["spike_reversal"]
    up, down, k_max, band = sp["jump_up"], sp["jump_down"], sp["revert_within_bars"], sp["revert_band"]
    brk = CLEANING_RULES["segment_break_up_jump"]
    n = len(adj)
    drop = np.zeros(n, dtype=bool)
    seg_start = np.zeros(n, dtype=bool)
    prev = 0                        # index of the previous kept bar
    t = 1
    while t < n:
        r = adj[t] / adj[prev] - 1.0
        if r > up or r < down:
            level = adj[prev]
            revert = next((j for j in range(t + 1, min(n, t + 1 + k_max))
                           if abs(adj[j] / level - 1.0) <= band), None)
            if revert is not None:
                drop[t:revert] = True
                t = revert
                continue            # the reverting bar is compared with ``prev``
            if r > brk:
                seg_start[t] = True
        prev = t
        t += 1
    return drop, seg_start


def clean_bars(d: pd.DataFrame, asset: str = "equity",
               calendar: pd.DatetimeIndex | None = None) -> tuple[pd.DataFrame, dict]:
    """Return (clean bars with a ``segment`` column, per-rule drop counts).

    ``d`` needs date, open, close, adjusted_close, volume. ``calendar`` is required
    for asset="equity"/"nav" (use ``equity_calendar()``) and ignored for "crypto".
    """
    if asset not in ("equity", "crypto", "nav"):
        raise ValueError(f"asset must be 'equity', 'crypto' or 'nav', not {asset!r}")
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for c in ("open", "close", "adjusted_close", "volume"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    rep = {"n_in": int(len(d))}
    px = d[CLEANING_RULES["drop_nonpositive_or_missing_price"]]
    bad_px = ~(px > 0).all(axis=1)
    rep["price"] = int(bad_px.sum())
    d = d[~bad_px]
    bad_vol = ~(d["volume"] > 0) if asset != "nav" else pd.Series(False, index=d.index)
    rep["zero_volume"] = int(bad_vol.sum())
    d = d[~bad_vol]
    if asset in ("equity", "nav"):
        if calendar is None:
            raise ValueError("equity cleaning needs the exchange calendar")
        off = ~d["date"].isin(calendar)
        rep["off_calendar"] = int(off.sum())
        d = d[~off]
    d = d.reset_index(drop=True)
    drop, seg_start = _spikes_and_breaks(d["adjusted_close"].to_numpy(dtype=float))
    rep["spike_reversal"] = int(drop.sum())
    d["segment"] = np.cumsum(seg_start)
    d = d[~drop].reset_index(drop=True)
    rep["segment_breaks"] = int(seg_start.sum())
    rep["n_out"] = int(len(d))
    return d, rep


def equity_calendar(subdir: str = "etfs_full") -> pd.DatetimeIndex:
    """Exchange sessions = SPY's dates (from 1993-01-29 in etfs_full/). Before SPY
    existed there is no calendar: equity bars before 1993-01-29 are dropped."""
    spy = pd.read_parquet(EODHD / subdir / "SPY.parquet", columns=["date"])
    return pd.DatetimeIndex(pd.to_datetime(spy["date"])).sort_values()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="prices", help="subfolder of data/research/eodhd")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--asset", choices=["equity", "crypto", "nav"], help="default: crypto for crypto*/, else equity")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asset = args.asset or ("crypto" if args.dir.startswith("crypto") else "equity")
    cal = equity_calendar() if asset in ("equity", "nav") else None
    totals: dict[str, int] = {}
    worst = []
    for f in sorted((EODHD / args.dir).glob("*.parquet")):
        try:
            _, rep = clean_bars(pd.read_parquet(f), asset, cal)
        except (KeyError, ValueError) as exc:
            log.warning("%s: skipped (%s)", f.stem, exc)
            continue
        for k, v in rep.items():
            totals[k] = totals.get(k, 0) + v
        worst.append((rep["spike_reversal"] + 10 * rep["segment_breaks"], f.stem, rep))
    worst.sort(key=lambda x: -x[0])
    log.info("cleaning fp %s, %s: totals %s", cleaning_fingerprint()[:12], args.dir, totals)
    if args.report:
        for _, t, rep in worst[:25]:
            log.info("  %-10s %s", t, rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
