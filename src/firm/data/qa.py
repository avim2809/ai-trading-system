"""Data QA: report and flag, never mutate or delete (ticket P2-03).

Cleaning (``firm.data.cleaning``) drops and repairs; QA only reports. Every function takes frames/series and
returns ``QAFinding`` lists; inputs are never modified. Not imported by any live module.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from firm.allocation.calendar import nyse_holidays

log = logging.getLogger(__name__)

Severity = Literal["info", "warn", "flag"]

# Special full-day NYSE closures absent from SPY bars (1993-2026) that ``firm.allocation.calendar`` treats as trading days.
# Where QA and calendar.py disagree, this table wins for QA only. Sources: NYSE closure notices as recorded in the ticket.
SPECIAL_CLOSURES: dict[dt.date, str] = {
    dt.date(1994, 4, 27): "National day of mourning for President Nixon (verified: SPY has no bar this day)",
    dt.date(2001, 9, 11): "September 11 attacks, NYSE closed 2001-09-11 to 2001-09-14",
    dt.date(2001, 9, 12): "September 11 attacks, NYSE closed 2001-09-11 to 2001-09-14",
    dt.date(2001, 9, 13): "September 11 attacks, NYSE closed 2001-09-11 to 2001-09-14",
    dt.date(2001, 9, 14): "September 11 attacks, NYSE closed 2001-09-11 to 2001-09-14",
    dt.date(2004, 6, 11): "National day of mourning for President Reagan",
    dt.date(2007, 1, 2): "National day of mourning for President Ford",
    dt.date(2012, 10, 29): "Hurricane Sandy, NYSE closed 2012-10-29 and 2012-10-30",
    dt.date(2012, 10, 30): "Hurricane Sandy, NYSE closed 2012-10-29 and 2012-10-30",
    dt.date(2018, 12, 5): "National day of mourning for President George H. W. Bush",
    dt.date(2025, 1, 9): "National day of mourning for President Carter",
}

# Legitimate negative futures prices (report allow-list): (symbol, date) -> close.
ALLOWED_NEGATIVE: dict[tuple[str, dt.date], float] = {("CL", dt.date(2020, 4, 20)): -37.63}

_PRICE_COLS = ("open", "high", "low", "close", "adjusted_close")


@dataclass(frozen=True)
class QAFinding:
    check: str
    symbol: str
    date: dt.date | None
    severity: Severity
    detail: str


def expected_trading_days(start: dt.date, end: dt.date, calendar: str = "nyse") -> pd.DatetimeIndex:
    """Weekdays in ``[start, end]`` minus NYSE full-day holidays minus ``SPECIAL_CLOSURES``."""
    if calendar != "nyse":
        raise ValueError(f"unsupported calendar {calendar!r}")
    days = pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end))
    holidays = set()
    for y in range(start.year, end.year + 1):
        holidays |= nyse_holidays(y)
    holidays |= set(SPECIAL_CLOSURES)
    return pd.DatetimeIndex([d for d in days if d.date() not in holidays])


def _dates(bars: pd.DataFrame) -> pd.DatetimeIndex:
    if "date" in bars.columns:
        return pd.DatetimeIndex(pd.to_datetime(bars["date"]))
    return pd.DatetimeIndex(pd.to_datetime(bars.index))


def check_spikes(returns: pd.Series, *, sigma: float = 8.0, window: int = 252, symbol: str = "") -> list[QAFinding]:
    """Flag ``|r_t| > sigma * std(r_{t-window..t-1})`` (past returns only). Never deletes."""
    r = returns.astype(float)
    n_unchecked = min(len(r), window)
    out = [QAFinding("spike", symbol, None, "info", f"{n_unchecked} leading returns unchecked (insufficient history, window={window})")]
    if len(r) <= window:
        return out
    past_sd = r.shift(1).rolling(window, min_periods=window).std()
    z = r / past_sd.replace(0.0, np.nan)
    for ts in z.index[(z.abs() > sigma).to_numpy()]:
        out.append(QAFinding("spike", symbol, pd.Timestamp(ts).date(), "flag", f"z={z[ts]:.1f} (|z| > {sigma}); for human review, not removed"))
    return out


def check_stale(bars: pd.DataFrame, *, max_repeat_days: int = 5, symbol: str = "") -> list[QAFinding]:
    """Runs of identical close longer than ``max_repeat_days`` bars. ``flag`` if volume is identical too, else ``warn``."""
    close = bars["close"].reset_index(drop=True)
    dates = _dates(bars)
    run = (close != close.shift()).cumsum()
    out = []
    for idx in close.groupby(run).groups.values():
        if len(idx) <= max_repeat_days:
            continue
        same_vol = "volume" in bars.columns and bars["volume"].iloc[list(idx)].nunique(dropna=False) == 1
        out.append(QAFinding("stale", symbol, dates[idx[0]].date(), "flag" if same_vol else "warn",
                             f"{len(idx)} identical closes ({close[idx[0]]}) from {dates[idx[0]].date()}; volume identical: {bool(same_vol)}"))
    return out


def check_missing_days(bars: pd.DataFrame, *, calendar: str = "nyse", symbol: str = "") -> list[QAFinding]:
    """Expected sessions between the first and last bar that are absent (count + first 10, one finding)."""
    d = _dates(bars)
    if len(d) == 0:
        return []
    exp = expected_trading_days(d.min().date(), d.max().date(), calendar)
    missing = exp.difference(d.normalize())
    if len(missing) == 0:
        return []
    first10 = ", ".join(x.date().isoformat() for x in missing[:10])
    return [QAFinding("missing_days", symbol, missing[0].date(), "flag", f"{len(missing)} expected sessions absent; first 10: {first10}")]


def check_negative_prices(bars: pd.DataFrame, *, asset: str, symbol: str = "") -> list[QAFinding]:
    """equity/etf/crypto: price <= 0 is a ``flag``. futures: negative is ``info`` when allow-listed, else ``warn``."""
    cols = [c for c in _PRICE_COLS if c in bars.columns]
    dates = _dates(bars)
    out = []
    for i in np.flatnonzero((bars[cols] <= 0).any(axis=1).to_numpy()):
        day = dates[i].date()
        row = bars[cols].iloc[i]
        if asset == "futures":
            allowed = ALLOWED_NEGATIVE.get((symbol, day))
            if allowed is not None and "close" in row and abs(float(row["close"]) - allowed) < 1e-6:
                out.append(QAFinding("negative_price", symbol, day, "info", f"allow-listed legitimate negative close {allowed} ({symbol} {day})"))
            else:
                out.append(QAFinding("negative_price", symbol, day, "warn", "non-positive futures price not on the allow-list"))
        else:
            out.append(QAFinding("negative_price", symbol, day, "flag", f"non-positive price for asset={asset}: {row.min()}"))
    return out


def check_fx_conversion(px_local: pd.Series, fx: pd.Series, px_usd: pd.Series, *, tol: float = 1e-6, symbol: str = "") -> list[QAFinding]:
    """``px_usd == px_local * fx`` within relative ``tol``; fx strictly positive; no day-over-day fx move above 10%."""
    out = []
    for ts in fx.index[(~(fx > 0)).to_numpy()]:
        out.append(QAFinding("fx", symbol, pd.Timestamp(ts).date(), "flag", "fx rate not strictly positive"))
    jumps = fx.pct_change(fill_method=None).abs()
    for ts in jumps.index[(jumps > 0.10).to_numpy()]:
        out.append(QAFinding("fx", symbol, pd.Timestamp(ts).date(), "flag", f"fx moved {jumps[ts]:.1%} in one day"))
    df = pd.concat({"l": px_local, "f": fx, "u": px_usd}, axis=1, join="inner")
    err = (df["u"] - df["l"] * df["f"]).abs() / df["u"].abs().replace(0.0, np.nan)
    for ts in err.index[(err > tol).to_numpy()]:
        out.append(QAFinding("fx", symbol, pd.Timestamp(ts).date(), "flag", f"px_usd differs from px_local*fx by {err[ts]:.2e} (> tol {tol})"))
    return sorted(out, key=lambda f: (f.date or dt.date.min, f.detail))


def check_roll_gaps(contracts_or_adj: pd.DataFrame, rolls: pd.DataFrame, *, tol_sigma: float = 3.0, symbol: str = "") -> list[QAFinding]:
    """Flag roll dates where the ADJUSTED series still jumps by more than ``tol_sigma`` typical daily moves.

    ``contracts_or_adj`` has a date index (or ``date`` column) and an ``adjusted`` column (else ``close``); moves are price
    differences (a Panama-adjusted series can be near zero or negative, so ratios are meaningless). Typical sigma is the std of
    non-roll differences.
    """
    col = "adjusted" if "adjusted" in contracts_or_adj.columns else "close"
    s = pd.Series(contracts_or_adj[col].to_numpy(dtype=float), index=_dates(contracts_or_adj))
    diff = s.diff()
    roll_ts = pd.DatetimeIndex(pd.to_datetime(rolls["date"]))
    on_roll = diff.index.isin(roll_ts)
    sd = diff[~on_roll].std()
    if not sd > 0:
        return []
    out = []
    for ts in diff.index[on_roll]:
        if abs(diff[ts]) > tol_sigma * sd:
            out.append(QAFinding("roll_gap", symbol, ts.date(), "flag", f"adjusted move {diff[ts]:.4g} = {abs(diff[ts]) / sd:.1f} sigma on roll date"))
    return out


def run_all(series: dict[str, pd.DataFrame], *, asset: str = "etf", calendar: str = "nyse", **kw) -> list[QAFinding]:
    """Spikes, stale, missing days and negative prices for each symbol. Reports only; inputs are not modified."""
    out: list[QAFinding] = []
    for sym, bars in series.items():
        adj = bars["adjusted_close"] if "adjusted_close" in bars.columns else bars["close"]
        if "segment" in bars.columns:
            r = adj.groupby(bars["segment"]).pct_change(fill_method=None)
        else:
            r = adj.pct_change(fill_method=None)
        r = pd.Series(r.to_numpy(), index=_dates(bars)).dropna()
        out += check_spikes(r, symbol=sym, **{k: v for k, v in kw.items() if k in ("sigma", "window")})
        out += check_stale(bars, symbol=sym, **{k: v for k, v in kw.items() if k == "max_repeat_days"})
        out += check_missing_days(bars, calendar=calendar, symbol=sym)
        out += check_negative_prices(bars, asset=asset, symbol=sym)
    log.info("data QA: %d symbols, %d findings", len(series), len(out))
    return out
