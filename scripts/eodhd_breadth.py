"""Market-breadth signal builder for the S2 (breadth overlay on the 60/40 core)
pre-registration (scripts/eodhd_s2_breadth_overlay_preregistered_bars.py).

PHASE 1 SCOPE: this module builds the breadth SIGNAL series only (eligible-stock
counts and the two declared breadth measures) from local EODHD data. It never
reads or joins any forward return, and never looks at how breadth relates to
subsequent performance of anything -- that is the whole point of running this
before the pre-registration is frozen. Downstream evaluation code (Phase 2,
not written here) is the only place a return series may appear.

Universe: data/research/eodhd/us_universe_full/*.parquet (full-history pull,
back to 1985 where EODHD has it; ~19k+ US common-stock tickers, active AND
delisted -- survivorship-free, per docs/eodhd_shortlist_protocol_2026_10.md,
amendment 1 2026-09-30 22:40Z -- the 2005-start `us_universe/` folder is
superseded). Every ticker's bars are run through scripts/eodhd_clean.py:
clean_bars (cleaning v2; equity calendar = SPY's own trading dates in
etfs_full/, from 1993-01-29 -- bars before that have no calendar to check
against and are dropped by clean_bars itself) before anything is computed,
and no rolling window (SMA200, ADV20, the advance/decline flag) is allowed to
cross a `segment` boundary -- the same rule the protocol gives for returns,
applied here to avoid smoothing over a data discontinuity (e.g. an unadjusted
reverse split) that the cleaning rule cannot distinguish from a real,
permanent re-rating.

Point-in-time eligibility (declared, not tuned -- see UNIVERSE_SCREEN below):
a ticker is "eligible" on day t iff, within its current segment:
    1. it has >= SMA_WINDOW (200) consecutive clean trading-day bars ending at
       t (so the 200-day average itself needs no look-ahead and is never
       computed on a partial window);
    2. its raw (unadjusted) close on day t is >= PRICE_MIN_USD (excludes
       penny-stock noise; raw close is used, not adjusted_close, because a
       stock can show an adjusted_close of a few dollars purely from
       cumulative forward splits while trading at $100+ today -- adjusted
       price is the wrong number for a "is this a real, tradable, non-junk
       stock" screen);
    3. its trailing 20-session median dollar volume (adjusted_close x volume,
       per the protocol's own dollar-volume rule: volume is split-adjusted,
       open/close are raw) ending at t is >= ADV20_MIN_USD.

Breadth measures, both computed only over each day's eligible set:
    - primary: fraction of eligible stocks with adjusted_close > SMA200.
    - alt (variant D in the pre-registration): a trailing-21-session average
      net advance/decline ratio, (advances - declines) / eligible_count per
      day, smoothed over 21 sessions (a standard window, not tuned against
      any return). An "advance" is adjusted_close_t > adjusted_close_{t-1}
      for an eligible stock with a same-segment previous bar; "decline" is
      the reverse; a flat close counts as neither.

Output: a single date-indexed parquet with eligible_count and both breadth
series (`build_breadth`'s `out_path`). No survivorship, coverage, or breadth
number in this file depends on any return series.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eodhd_clean as ec

log = logging.getLogger(__name__)

EODHD = Path(__file__).resolve().parents[1] / "data" / "research" / "eodhd"
REQUIRED_COLS = ["date", "open", "close", "adjusted_close", "volume"]

# ---------------------------------------------------------------------------
# Declared, pre-return screen. Frozen in the pre-registration's UNIVERSE dict
# (kept here too so the builder and the pre-registration cite the same numbers).
# ---------------------------------------------------------------------------
UNIVERSE_SCREEN = {
    "sma_window": 200,
    "price_min_usd": 5.0,
    "adv20_min_usd": 1_000_000.0,
    "ad_window": 21,
}


@dataclass
class BreadthResult:
    frame: pd.DataFrame          # date, eligible_count, pct_above_200sma, net_ad_ratio, net_ad_21d
    n_tickers_scanned: int
    n_tickers_used: int          # had >=1 day passing the cleaning rule
    skipped: list[str]           # ticker stems that raised and were skipped


def _load_one(path: Path) -> pd.DataFrame | None:
    try:
        d = pd.read_parquet(path, columns=REQUIRED_COLS)
    except (OSError, ValueError, KeyError) as exc:
        log.warning("%s: unreadable, skipped (%s)", path.stem, exc)
        return None
    if d.empty:
        return None
    # Downcast prices to float32 (memory budget); keep volume as float64 --
    # dollar volume = adjusted_close x volume can need more than float32's
    # ~7 significant digits once volume exceeds ~10-20M shares/day.
    for c in ("open", "close", "adjusted_close"):
        d[c] = d[c].astype("float32")
    d["volume"] = d["volume"].astype("float64")
    return d


def _ticker_signal(d: pd.DataFrame, calendar: pd.DatetimeIndex, calendar_pos: dict[np.datetime64, int]
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Per-ticker (date_idx, eligible, above_sma, advance_minus_decline) arrays.

    ``advance_minus_decline`` is +1/-1/0 per eligible day (advance/decline/flat);
    the caller accumulates it only where eligible.
    """
    clean, _ = ec.clean_bars(d, asset="equity", calendar=calendar)
    if clean.empty:
        return None
    sw = UNIVERSE_SCREEN["sma_window"]
    seg = clean["segment"].to_numpy()
    adj = clean["adjusted_close"].to_numpy(dtype=np.float64)
    raw_close = clean["close"].to_numpy(dtype=np.float64)
    dollar_vol = adj * clean["volume"].to_numpy(dtype=np.float64)

    # Rolling stats computed per-segment (never across a segment break) via a
    # pandas groupby-rolling on the segment id, keeping everything positional.
    s = pd.Series(adj)
    seg_s = pd.Series(seg)
    sma = s.groupby(seg_s).transform(lambda x: x.rolling(sw, min_periods=sw).mean()).to_numpy()
    n_in_seg = seg_s.groupby(seg_s).cumcount().to_numpy() + 1  # 1-based position within segment

    dv = pd.Series(dollar_vol)
    adv20 = dv.groupby(seg_s).transform(lambda x: x.rolling(20, min_periods=20).median()).to_numpy()

    eligible = (
        (n_in_seg >= sw)
        & np.isfinite(sma)
        & (raw_close >= UNIVERSE_SCREEN["price_min_usd"])
        & np.isfinite(adv20)
        & (adv20 >= UNIVERSE_SCREEN["adv20_min_usd"])
    )
    above_sma = eligible & (adj > sma)

    prev_adj = np.roll(adj, 1)
    prev_seg = np.roll(seg, 1)
    has_prev = (prev_seg == seg) & (np.arange(len(seg)) > 0)
    ad = np.zeros(len(seg), dtype=np.int8)
    ad[has_prev & (adj > prev_adj)] = 1
    ad[has_prev & (adj < prev_adj)] = -1
    ad = np.where(eligible & has_prev, ad, 0)

    dates = clean["date"].to_numpy()
    idx = np.array([calendar_pos.get(dt, -1) for dt in dates], dtype=np.int64)
    keep = idx >= 0
    return idx[keep], eligible[keep], above_sma[keep], ad[keep]


def build_breadth(universe_dir: Path | None = None, calendar: pd.DatetimeIndex | None = None,
                   limit: int | None = None) -> BreadthResult:
    """Stream every us_universe ticker once, accumulate daily breadth counts.

    ``calendar`` defaults to ``eodhd_clean.equity_calendar()`` (SPY's own
    trading dates); tests pass a synthetic calendar instead of touching the
    real EODHD pull.

    Memory: one ticker's DataFrame is live at a time; only small int64/float64
    accumulator arrays (length = trading-calendar length) persist across the
    loop. No candidate/benchmark/placebo return is computed anywhere here.
    """
    universe_dir = universe_dir or (EODHD / "us_universe_full")
    calendar = calendar if calendar is not None else ec.equity_calendar()  # etfs_full/SPY, from 1993-01-29
    n = len(calendar)
    calendar_pos = {d: i for i, d in enumerate(calendar.to_numpy())}

    elig_count = np.zeros(n, dtype=np.int64)
    above_count = np.zeros(n, dtype=np.int64)
    adv_count = np.zeros(n, dtype=np.int64)
    decl_count = np.zeros(n, dtype=np.int64)

    files = sorted(universe_dir.glob("*.parquet"))
    if limit is not None:
        files = files[:limit]
    skipped: list[str] = []
    n_used = 0
    for i, f in enumerate(files):
        d = _load_one(f)
        if d is None:
            skipped.append(f.stem)
            continue
        try:
            out = _ticker_signal(d, calendar, calendar_pos)
        except (ValueError, KeyError) as exc:
            log.warning("%s: signal build failed, skipped (%s)", f.stem, exc)
            skipped.append(f.stem)
            continue
        del d
        if out is None:
            continue
        idx, eligible, above_sma, ad = out
        n_used += 1
        if eligible.any():
            np.add.at(elig_count, idx[eligible], 1)
            np.add.at(above_count, idx[above_sma], 1)
        adv_mask = eligible & (ad == 1)
        decl_mask = eligible & (ad == -1)
        if adv_mask.any():
            np.add.at(adv_count, idx[adv_mask], 1)
        if decl_mask.any():
            np.add.at(decl_count, idx[decl_mask], 1)
        if (i + 1) % 2000 == 0:
            log.info("breadth build: %d/%d tickers scanned (%d used)", i + 1, len(files), n_used)

    with np.errstate(invalid="ignore", divide="ignore"):
        pct_above = np.where(elig_count > 0, above_count / np.maximum(elig_count, 1), np.nan)
        net_ad_ratio = np.where(elig_count > 0, (adv_count - decl_count) / np.maximum(elig_count, 1), np.nan)
    net_ad_series = pd.Series(net_ad_ratio)
    net_ad_21d = net_ad_series.rolling(UNIVERSE_SCREEN["ad_window"], min_periods=UNIVERSE_SCREEN["ad_window"]).mean().to_numpy()

    frame = pd.DataFrame({
        "date": calendar,
        "eligible_count": elig_count,
        "pct_above_200sma": pct_above,
        "net_ad_ratio": net_ad_ratio,
        "net_ad_21d": net_ad_21d,
    })
    return BreadthResult(frame=frame, n_tickers_scanned=len(files), n_tickers_used=n_used, skipped=skipped)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output parquet path")
    ap.add_argument("--limit", type=int, default=None, help="cap # tickers scanned (debug)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = build_breadth(limit=args.limit)
    log.info("scanned %d tickers, %d usable after cleaning, %d skipped entirely",
              res.n_tickers_scanned, res.n_tickers_used, len(res.skipped))
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        res.frame.to_parquet(out_path, index=False)
        log.info("wrote %s (%d rows)", out_path, len(res.frame))
    nz = res.frame[res.frame["eligible_count"] > 0]
    if len(nz):
        log.info("eligible_count: min=%d max=%d median=%.0f, first nonzero date=%s",
                  nz["eligible_count"].min(), nz["eligible_count"].max(),
                  nz["eligible_count"].median(), nz["date"].iloc[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
