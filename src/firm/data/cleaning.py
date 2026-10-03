"""EODHD bar cleaning v3 (credibility plan P2-08, owner decision OD-14).

Additive successor of the frozen v2 rule in ``scripts/eodhd_clean.py`` (which stays
untouched: frozen preregistrations pin its fingerprint). v3 inherits every v2 rule
unchanged and adds:

1. Constant-sentinel rule: drop bars where any price column equals 999999.9999
   (tolerance 1e-6), counted as ``sentinel``; and drop any run of >= 5 identical
   ``adjusted_close`` values that is >= 1e5 on an asset whose median price is < 1e4,
   counted as ``constant_run``.
2. ``asset="futures"``: negative and zero prices are legitimate (crude, 2020-04-20)
   and kept; only NaN prices are dropped. Zero volume is kept, NaN volume dropped.
   No exchange calendar.
3. Spike reversal on a futures series with any non-positive adjusted close is applied
   to price DIFFERENCES scaled by the series' median absolute level ``S`` (a ratio
   against a non-positive level is meaningless): ``r_t = (adj_t - adj_prev) / S`` and
   revert test ``|adj_j - adj_prev| / S <= band``. Series that stay positive use the
   v2 level rule. Pre-registered choice, not tuned. Caveat: a genuine one-day crash
   that fully reverts within 5 bars is removed as a bad print, exactly as in v2.

Nothing here reads a data store: callers pass frames (and SPY dates) in; the real
loader is ``firm.research.data_access`` (P2-02). Not imported by any live module.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SENTINEL_VALUE = 999999.9999
SEAL_DATE = date(2026, 10, 1)  # first sealed day; research bars must be strictly earlier
_PRICE_COLS = ("open", "high", "low", "close", "adjusted_close")

CLEANING_RULES_V3 = {
    "version": 3,
    "frozen_at": "2026-10-03T22:31:16Z",
    "v3_change": "additive module src/firm/data/cleaning.py (OD-14); v2 in scripts/eodhd_clean.py unchanged",
    "sentinel": {"value": SENTINEL_VALUE, "tolerance": 1e-6, "columns": list(_PRICE_COLS)},
    "constant_run": {"min_run": 5, "min_value": 1e5, "median_price_below": 1e4, "column": "adjusted_close"},
    "drop_nonpositive_or_missing_price": ["open", "close", "adjusted_close"],
    "futures_price_rule": "drop NaN only; negative and zero prices kept",
    "drop_zero_or_missing_volume": True,
    "futures_volume_rule": "drop NaN only; zero volume kept",
    "nav_volume_rule": "skipped",
    "equity_calendar": "dates of SPY etfs_full passed as an argument (sorted, strictly before 2026-10-01)",
    "crypto_calendar": "none (all days)",
    "futures_calendar": "none",
    "spike_reversal": {"jump_up": 1.0, "jump_down": -0.5, "revert_within_bars": 5, "revert_band": 0.25},
    "spike_reversal_nonpositive_futures": "differences scaled by median |adjusted_close|, same thresholds",
    "segment_break_up_jump": 1.5,
    "down_jumps": "kept as real (bias resolved against long books)",
}


def cleaning_fingerprint_v3() -> str:
    return hashlib.sha256(json.dumps(CLEANING_RULES_V3, sort_keys=True).encode()).hexdigest()


def _spikes_and_breaks(adj: np.ndarray, diff_scale: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Boolean masks (drop, segment_start) over ``adj`` (v2 rules 4 and 5).

    ``diff_scale=None``: v2 level rule (copied verbatim). Otherwise moves are
    ``(adj_t - adj_prev) / diff_scale`` (non-positive futures series).
    """
    sp = CLEANING_RULES_V3["spike_reversal"]
    up, down, k_max, band = sp["jump_up"], sp["jump_down"], sp["revert_within_bars"], sp["revert_band"]
    brk = CLEANING_RULES_V3["segment_break_up_jump"]
    n = len(adj)
    drop = np.zeros(n, dtype=bool)
    seg_start = np.zeros(n, dtype=bool)
    prev = 0  # index of the previous kept bar
    t = 1
    while t < n:
        if diff_scale is None:
            r = adj[t] / adj[prev] - 1.0
        else:
            r = (adj[t] - adj[prev]) / diff_scale
        if r > up or r < down:
            level = adj[prev]
            if diff_scale is None:
                revert = next(
                    (j for j in range(t + 1, min(n, t + 1 + k_max)) if abs(adj[j] / level - 1.0) <= band), None
                )
            else:
                revert = next(
                    (j for j in range(t + 1, min(n, t + 1 + k_max)) if abs(adj[j] - level) / diff_scale <= band),
                    None,
                )
            if revert is not None:
                drop[t:revert] = True
                t = revert
                continue  # the reverting bar is compared with ``prev``
            if r > brk:
                seg_start[t] = True
        prev = t
        t += 1
    return drop, seg_start


def _constant_run_mask(adj: pd.Series) -> np.ndarray:
    cfg = CLEANING_RULES_V3["constant_run"]
    mask = np.zeros(len(adj), dtype=bool)
    if len(adj) == 0 or not adj.median() < cfg["median_price_below"]:
        return mask
    run_id = (adj != adj.shift()).cumsum()
    sizes = adj.groupby(run_id).transform("size")
    return ((sizes >= cfg["min_run"]) & (adj >= cfg["min_value"])).to_numpy()


def clean_bars_v3(
    d: pd.DataFrame,
    asset: Literal["equity", "crypto", "nav", "futures"] = "equity",
    calendar: pd.DatetimeIndex | None = None,
    *,
    symbol: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return (clean bars with a ``segment`` column, per-rule drop counts).

    Same contract as ``scripts/eodhd_clean.clean_bars`` plus ``sentinel`` and
    ``constant_run`` counts. ``d`` needs date, open, close, adjusted_close, volume.
    ``calendar`` is required for asset="equity"/"nav" and ignored otherwise.
    """
    if asset not in ("equity", "crypto", "nav", "futures"):
        raise ValueError(f"asset must be 'equity', 'crypto', 'nav' or 'futures', not {asset!r}")
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    d = d.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    for c in (*_PRICE_COLS, "volume"):
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("open", "close", "adjusted_close", "volume"):
        if c not in d.columns:
            raise ValueError(f"missing column {c!r}")
    tag = f" [{symbol}]" if symbol else ""
    rep = {"n_in": len(d)}

    present = [c for c in _PRICE_COLS if c in d.columns]
    is_sent = (d[present].sub(SENTINEL_VALUE).abs() < CLEANING_RULES_V3["sentinel"]["tolerance"]).any(axis=1)
    rep["sentinel"] = int(is_sent.sum())
    if rep["sentinel"]:
        log.warning("sentinel %s bars dropped: %d%s", SENTINEL_VALUE, rep["sentinel"], tag)
    d = d[~is_sent]

    px = d[CLEANING_RULES_V3["drop_nonpositive_or_missing_price"]]
    bad_px = ~px.notna().all(axis=1) if asset == "futures" else ~(px > 0).all(axis=1)
    rep["price"] = int(bad_px.sum())
    d = d[~bad_px]

    if asset == "nav":
        bad_vol = pd.Series(False, index=d.index)
    elif asset == "futures":
        bad_vol = d["volume"].isna()
    else:
        bad_vol = ~(d["volume"] > 0)
    rep["zero_volume"] = int(bad_vol.sum())
    d = d[~bad_vol]

    if asset in ("equity", "nav"):
        if calendar is None:
            raise ValueError("equity cleaning needs the exchange calendar")
        off = ~d["date"].isin(calendar)
        rep["off_calendar"] = int(off.sum())
        d = d[~off]
    else:
        rep["off_calendar"] = 0

    d = d.reset_index(drop=True)
    const = _constant_run_mask(d["adjusted_close"])
    rep["constant_run"] = int(const.sum())
    if rep["constant_run"]:
        log.warning("constant-run sentinel-like bars dropped: %d%s", rep["constant_run"], tag)
    d = d[~const].reset_index(drop=True)

    adj = d["adjusted_close"].to_numpy(dtype=float)
    if asset == "futures" and len(adj) and (adj <= 0).any():
        scale = float(np.median(np.abs(adj)))
        if scale > 0:
            drop, seg_start = _spikes_and_breaks(adj, scale)
        else:  # all-zero series: nothing to test
            drop, seg_start = np.zeros(len(d), bool), np.zeros(len(d), bool)
    else:
        drop, seg_start = _spikes_and_breaks(adj)
    rep["spike_reversal"] = int(drop.sum())
    d["segment"] = np.cumsum(seg_start)
    d = d[~drop].reset_index(drop=True)
    rep["segment_breaks"] = int(seg_start.sum())
    rep["n_out"] = len(d)
    for k, v in rep.items():
        if k not in ("n_in", "n_out") and v:
            log.info("cleaning v3%s: %s=%d", tag, k, v)
    return d, rep


def equity_calendar_v3(
    spy_dates: pd.DataFrame | pd.Series | pd.DatetimeIndex, *, asof: date | None = None
) -> pd.DatetimeIndex:
    """Exchange sessions = SPY bar dates, sorted, deduplicated (same rule as v2).

    Takes the dates as an ARGUMENT (the caller loads them via ``firm.research.data_access``).
    Truncated to dates strictly before ``SEAL_DATE`` and, if given, to dates <= ``asof``.
    """
    if isinstance(spy_dates, pd.DataFrame):
        spy_dates = spy_dates["date"] if "date" in spy_dates.columns else spy_dates.index.to_series()
    idx = pd.DatetimeIndex(pd.to_datetime(spy_dates)).sort_values().unique()
    idx = idx[idx < pd.Timestamp(SEAL_DATE)]
    if asof is not None:
        idx = idx[idx <= pd.Timestamp(asof)]
    return pd.DatetimeIndex(idx)
