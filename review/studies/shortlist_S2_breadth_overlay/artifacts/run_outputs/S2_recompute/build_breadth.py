"""Independent recompute (bar A7) of the S2 breadth SIGNAL only.

Built from the frozen text in scripts/eodhd_s2_breadth_overlay_preregistered_bars.py
(git ff6f44a) -- NOT by reading scripts/eodhd_breadth.py or scripts/run_eodhd_s2_evaluation.py.

Eligibility screen (UNIVERSE, frozen text):
  - raw close >= $5.00
  - trailing 20-session median dollar volume (adjusted_close * volume) >= $1,000,000
  - >= 200 same-segment bars of history (so a 200-session SMA of adjusted_close is computable)
  all three required jointly for a ticker to be "eligible" on a given day.

BREADTH measures (frozen text):
  - primary: pct_above_200sma = fraction of eligible stocks with adjusted_close >
    trailing 200-session SMA of adjusted_close, same-segment only.
  - alternative: net_ad_21d = 21-session trailing average of the daily net
    advance/decline ratio, (advances - declines) / eligible_count, where an
    eligible stock advances/declines by same-segment adjusted_close_t vs
    adjusted_close_{t-1}.

Streams per ticker (22,732 files), float32/float64 only where needed, uses
numpy cumulative-sum tricks for the rolling 200-session mean (no full
rolling-window materialization) to keep memory low. Accumulates only
per-day integer counts (eligible/above/advance/decline), never a
per-ticker-per-day matrix.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/local/store/git/ai-trading-system")
sys.path.insert(0, str(REPO / "scripts"))
from eodhd_clean import clean_bars, equity_calendar  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("build_breadth")

UNIVERSE_DIR = REPO / "data" / "research" / "eodhd" / "us_universe_full"
OUT = Path("/tmp/claude-0/-local-store-git-ai-trading-system/c4c796e1-061f-42c7-9fa9-255931a43502/scratchpad/runs/S2_recompute")

PRICE_MIN = 5.0
ADV20_MIN = 1_000_000.0
SMA_WINDOW = 200
ADV_WINDOW = 20


def main() -> int:
    t0 = time.time()
    calendar = equity_calendar()  # SPY's own dates, from 1993-01-29
    cal_vals = calendar.values  # datetime64[ns], sorted
    n_days = len(cal_vals)
    log.info("calendar: %d sessions, %s .. %s", n_days, calendar[0].date(), calendar[-1].date())

    elig_count = np.zeros(n_days, dtype=np.int64)
    above_count = np.zeros(n_days, dtype=np.int64)
    adv_count = np.zeros(n_days, dtype=np.int64)
    dec_count = np.zeros(n_days, dtype=np.int64)

    files = sorted(UNIVERSE_DIR.glob("*.parquet"))
    n_scanned = 0
    n_usable = 0
    for fi, f in enumerate(files):
        n_scanned += 1
        try:
            raw = pd.read_parquet(f, columns=["date", "open", "close", "adjusted_close", "volume"])
        except Exception as exc:  # noqa: BLE001 -- a genuinely corrupt/unreadable file must not kill the whole 20-min run
            log.warning("%s: unreadable (%s), skipped", f.stem, exc)
            continue
        if raw.empty:
            continue
        try:
            df, _rep = clean_bars(raw, asset="equity", calendar=calendar)
        except (KeyError, ValueError) as exc:
            log.warning("%s: clean_bars failed (%s), skipped", f.stem, exc)
            continue
        if df.empty:
            continue
        n_usable += 1

        dates = df["date"].to_numpy()
        idx = np.searchsorted(cal_vals, dates)
        # sanity: every date must be an exact calendar hit (clean_bars already filtered off-calendar)
        bad = (idx >= n_days) | (cal_vals[np.clip(idx, 0, n_days - 1)] != dates)
        if bad.any():
            idx = idx[~bad]
            df = df[~bad]
            dates = dates[~bad]
        if len(df) == 0:
            continue

        adj = df["adjusted_close"].to_numpy(dtype=np.float64)
        close = df["close"].to_numpy(dtype=np.float64)
        vol = df["volume"].to_numpy(dtype=np.float64)
        segment = df["segment"].to_numpy()

        dollar_vol = adj * vol
        adv20 = pd.Series(dollar_vol).rolling(ADV_WINDOW, min_periods=ADV_WINDOW).median().to_numpy()
        price_ok = close >= PRICE_MIN
        adv_ok = adv20 >= ADV20_MIN

        sma200 = np.full(len(adj), np.nan, dtype=np.float64)
        for seg in np.unique(segment):
            mask = segment == seg
            sub = adj[mask]
            if len(sub) >= SMA_WINDOW:
                csum = np.concatenate(([0.0], np.cumsum(sub)))
                sma_vals = (csum[SMA_WINDOW:] - csum[:-SMA_WINDOW]) / SMA_WINDOW
                out = np.full(len(sub), np.nan)
                out[SMA_WINDOW - 1:] = sma_vals
                sma200[mask] = out

        eligible_sma = ~np.isnan(sma200)
        eligible = price_ok & adv_ok & eligible_sma
        above = eligible & (adj > sma200)

        prev_adj = np.full(len(adj), np.nan)
        prev_seg = np.full(len(segment), -1)
        prev_adj[1:] = adj[:-1]
        prev_seg[1:] = segment[:-1]
        same_seg = prev_seg == segment
        ret_valid = same_seg & ~np.isnan(prev_adj)
        advance = eligible & ret_valid & (adj > prev_adj)
        decline = eligible & ret_valid & (adj < prev_adj)

        np.add.at(elig_count, idx, eligible.astype(np.int64))
        np.add.at(above_count, idx, above.astype(np.int64))
        np.add.at(adv_count, idx, advance.astype(np.int64))
        np.add.at(dec_count, idx, decline.astype(np.int64))

        if (fi + 1) % 2000 == 0:
            log.info("processed %d/%d files (%.1f min elapsed)", fi + 1, len(files), (time.time() - t0) / 60)

    log.info("done: %d scanned, %d usable (%.1f min)", n_scanned, n_usable, (time.time() - t0) / 60)

    elig_f = elig_count.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct_above = np.where(elig_count > 0, above_count / elig_f, np.nan)
        net_ad_daily = np.where(elig_count > 0, (adv_count - dec_count) / elig_f, np.nan)
    net_ad_21d = pd.Series(net_ad_daily).rolling(21, min_periods=21).mean().to_numpy()

    out = pd.DataFrame({
        "date": calendar,
        "eligible_count": elig_count,
        "above_count": above_count,
        "advance_count": adv_count,
        "decline_count": dec_count,
        "pct_above_200sma": pct_above,
        "net_ad_daily": net_ad_daily,
        "net_ad_21d": net_ad_21d,
    })
    out.to_parquet(OUT / "breadth_mine.parquet")
    log.info("wrote %s (%d rows)", OUT / "breadth_mine.parquet", len(out))

    # quick cross-check against the frozen pre-registration's OBSERVED block (full history)
    nz = out[out["eligible_count"] > 0]
    log.info("n_scanned=%d n_usable=%d", n_scanned, n_usable)
    log.info("eligible_count first_nonzero_date=%s min_nonzero=%d max=%d median_full=%.1f",
              nz["date"].iloc[0].date(), nz["eligible_count"].min(), out["eligible_count"].max(),
              nz["eligible_count"].median())
    w = out[out["date"] >= "2002-01-02"]
    log.info("eligible_count at 2002-01-02 window start = %d", w["eligible_count"].iloc[0])
    log.info("eligible_count at data end (2026-09-29) = %d", out["eligible_count"].iloc[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
