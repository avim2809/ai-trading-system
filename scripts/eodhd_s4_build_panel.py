"""S4 (52-week-high proximity) universe/feature panel builder — PHASE 1, DRAFT.

Builds the point-in-time month-end panel that ``eodhd_s4_52wk_high_preregistered_bars.py``
evaluates against in phase 2. This script computes SIGNALS and ELIGIBILITY only: it
never computes a forward return, and the candidate/benchmark/placebo return series on
the evaluation window is out of scope for phase 1 (see the pre-registration module
docstring and docs/eodhd_shortlist_protocol_2026_10.md).

Universe: every ``data/research/eodhd/us_universe_full/<TICKER>.parquet`` file (full
1985-or-inception history per shortlist protocol amendment 1, superseding the
2005-start ``us_universe/`` folder used before the amendment), all ``Type == "Common
Stock"`` per ``us_universe_symbols.parquet``, active + delisted; there is no
point-in-time index membership, so eligibility each month-end is built from the raw
OHLCV history alone, as the protocol requires.

Per ticker, streamed one file at a time (float32, only the 5 columns needed):
1. Run ``scripts/eodhd_clean.py:clean_bars`` (frozen cleaning rule, fingerprint
   ``cleaning_fingerprint()``) against the shared SPY trading calendar.
2. Track ``segment`` (clean_bars already provides this) and compute, *within each
   segment only* (a segment break is treated like a fresh IPO for lookback purposes —
   see the design-decision note in the module docstring of the pre-registration file):
   - ``n_bars_in_segment``: 1-based bar count since the segment started.
   - ``ratio_52wk``: adjusted_close / trailing-252-bar max of adjusted_close (NaN until
     the segment has >= 252 bars).
   - ``adv63``: trailing-63-bar median of adjusted_close * volume (NaN until the segment
     has >= 63 bars; note 63 < 252, so whenever ``ratio_52wk`` is defined, ``adv63`` is
     too — the per-ticker eligibility screen below only needs the 252-bar condition).
3. Extract only the rows landing exactly on a common month-end trading date (last
   session of each calendar month in the SPY calendar). A ticker with no bar on that
   exact date (pre-IPO, delisted, or a halt) simply has no row that month — this is
   how survivorship is kept honest without an explicit membership list.
4. Keep a row in the compact panel iff the per-ticker screens pass: price (raw,
   unadjusted close — see design note) >= $5, and n_bars_in_segment >= 252. The
   liquidity (top-N by adv63) cut is a *cross-sectional*, N-dependent step applied at
   evaluation time, not baked into this panel, so N can be varied (500 vs 1000) without
   rebuilding.

Output: ``$S/runs/S4/panel.parquet`` (ticker, month_end, price, adv63,
n_bars_in_segment, ratio_52wk, entry_date) and ``$S/runs/S4/eligibility_summary.parquet``
(per month-end: n_with_bar, n_pass_bars252, n_pass_price5, n_pass_both, plus the top-500/
top-1000 cutoff adv63 value when n_pass_both reaches that size) for the design decisions
in the pre-registration (N, window start, midpoint).

Design decision — price for the $5 screen: raw (unadjusted) ``close``, not
``adjusted_close``. ``adjusted_close`` is back-adjusted for *future* (relative to the
month-end in question) splits/dividends, so it is not the price that was actually
quoted/tradable at that date; using it for a penny-stock screen would apply
look-ahead-shaped scaling to a level check. ``ratio_52wk`` and ``adv63`` use
``adjusted_close`` (continuous within a segment), per protocol §1 for dollar volume and
per George & Hwang for the ratio itself (a split-unadjusted ratio would show a stock
that just split as "far from its high" when it is not).

    nice -n 10 python scripts/eodhd_s4_build_panel.py --out-dir $S/runs/S4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import eodhd_clean as ec

log = logging.getLogger(__name__)

EODHD = Path(__file__).resolve().parents[1] / "data" / "research" / "eodhd"
UNIVERSE_DIR = EODHD / "us_universe_full"

RATIO_WINDOW = 252
ADV_WINDOW = 63
PRICE_MIN = 5.0
BARS_MIN = 252
N_CANDIDATES_FOR_SUMMARY = (500, 1000)

PANEL_COLUMNS = ["ticker", "month_end", "price", "adv63", "n_bars_in_segment", "ratio_52wk", "entry_date"]


def month_end_dates(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading session of each calendar month in ``calendar`` (sorted, unique)."""
    s = pd.Series(calendar, index=calendar)
    ends = s.groupby([calendar.year, calendar.month]).max()
    return pd.DatetimeIndex(sorted(ends.to_numpy()))


def entry_dates_after(month_ends: pd.DatetimeIndex, calendar: pd.DatetimeIndex) -> pd.Series:
    """First calendar trading date strictly after each month-end (next tradable session)."""
    cal = np.asarray(calendar.sort_values())
    out = {}
    for me in month_ends:
        idx = np.searchsorted(cal, np.datetime64(me), side="right")
        out[me] = pd.Timestamp(cal[idx]) if idx < len(cal) else pd.NaT
    return pd.Series(out)


def _segment_rolling(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Add n_bars_in_segment, ratio_52wk, adv63 -- all reset at each segment boundary."""
    g = cleaned.groupby("segment", sort=False)
    cleaned["n_bars_in_segment"] = g.cumcount() + 1
    dollar_vol = (cleaned["adjusted_close"].astype("float64") * cleaned["volume"].astype("float64"))
    cleaned["_dollar_vol"] = dollar_vol
    roll_max = g["adjusted_close"].transform(lambda s: s.rolling(RATIO_WINDOW, min_periods=RATIO_WINDOW).max())
    cleaned["ratio_52wk"] = (cleaned["adjusted_close"] / roll_max).astype("float32")
    adv = cleaned.groupby("segment", sort=False)["_dollar_vol"].transform(
        lambda s: s.rolling(ADV_WINDOW, min_periods=ADV_WINDOW).median()
    )
    cleaned["adv63"] = adv.astype("float32")
    cleaned.drop(columns=["_dollar_vol"], inplace=True)
    return cleaned


def process_ticker(path: Path, calendar: pd.DatetimeIndex, month_ends: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Stream one ticker's parquet and return its month-end rows (all, unfiltered by screens)."""
    try:
        df = pd.read_parquet(path, columns=["date", "open", "close", "adjusted_close", "volume"])
    except Exception:
        log.warning("%s: unreadable parquet, skipped", path.stem)
        return None
    if df.empty:
        return None
    for c in ("open", "close", "adjusted_close"):
        df[c] = df[c].astype("float32")
    df["volume"] = df["volume"].astype("float32")
    try:
        cleaned, _rep = ec.clean_bars(df, asset="equity", calendar=calendar)
    except (KeyError, ValueError) as exc:
        log.warning("%s: clean_bars failed (%s), skipped", path.stem, exc)
        return None
    if cleaned.empty:
        return None
    cleaned = _segment_rolling(cleaned)
    sub = cleaned.set_index("date")[["close", "adv63", "n_bars_in_segment", "ratio_52wk"]].reindex(month_ends)
    sub = sub.dropna(subset=["close"])
    if sub.empty:
        return None
    sub = sub.reset_index().rename(columns={"index": "month_end", "close": "price"})
    sub.insert(0, "ticker", path.stem)
    return sub


def build_panel(universe_dir: Path = UNIVERSE_DIR, limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = ec.equity_calendar()
    month_ends = month_end_dates(calendar)
    entry = entry_dates_after(month_ends, calendar)

    files = sorted(universe_dir.glob("*.parquet"))
    if limit:
        files = files[:limit]
    log.info("S4 panel build: %d ticker files, %d month-ends (%s -> %s)",
             len(files), len(month_ends), month_ends.min().date(), month_ends.max().date())

    frames = []
    t0 = time.time()
    for i, f in enumerate(files):
        sub = process_ticker(f, calendar, month_ends)
        if sub is not None:
            frames.append(sub)
        if (i + 1) % 2000 == 0:
            log.info("  %d/%d tickers processed (%.1fs)", i + 1, len(files), time.time() - t0)
    log.info("processed %d tickers in %.1fs", len(files), time.time() - t0)

    if not frames:
        raw = pd.DataFrame(columns=["ticker", "month_end", "price", "adv63", "n_bars_in_segment", "ratio_52wk"])
    else:
        raw = pd.concat(frames, ignore_index=True)
    raw["entry_date"] = raw["month_end"].map(entry)

    summary = _eligibility_summary(raw, month_ends)

    pass_screen = (raw["n_bars_in_segment"] >= BARS_MIN) & (raw["price"] >= PRICE_MIN)
    panel = raw.loc[pass_screen, PANEL_COLUMNS].reset_index(drop=True)
    log.info("compact panel: %d rows (of %d raw month-end hits) pass bars>=%d and price>=$%.0f",
             len(panel), len(raw), BARS_MIN, PRICE_MIN)
    return panel, summary


def _eligibility_summary(raw: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    for me in month_ends:
        m = raw[raw["month_end"] == me]
        n_with_bar = len(m)
        n_bars_ok = int((m["n_bars_in_segment"] >= BARS_MIN).sum())
        n_price_ok = int((m["price"] >= PRICE_MIN).sum())
        both = m[(m["n_bars_in_segment"] >= BARS_MIN) & (m["price"] >= PRICE_MIN)]
        n_both = len(both)
        row = {"month_end": me, "n_with_bar": n_with_bar, "n_pass_bars252": n_bars_ok,
               "n_pass_price5": n_price_ok, "n_pass_both": n_both}
        adv_sorted = both["adv63"].sort_values(ascending=False) if n_both else pd.Series(dtype="float32")
        for n in N_CANDIDATES_FOR_SUMMARY:
            row[f"reaches_{n}"] = bool(n_both >= n)
            row[f"adv63_cutoff_{n}"] = float(adv_sorted.iloc[n - 1]) if n_both >= n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap the number of ticker files (debug only)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel, summary = build_panel(limit=args.limit)
    panel.to_parquet(out_dir / "panel.parquet", index=False)
    summary.to_parquet(out_dir / "eligibility_summary.parquet", index=False)
    log.info("wrote %s (%d rows) and %s (%d months)",
              out_dir / "panel.parquet", len(panel), out_dir / "eligibility_summary.parquet", len(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
