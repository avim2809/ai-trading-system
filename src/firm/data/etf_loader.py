"""Survivorship-free, point-in-time ETF loader plus USD/ILS and Israeli CPI series (ticket P2-02).

Every bar read goes through ``firm.research.data_access.read_parquet`` (allow-list, seal, post-``asof`` rows
dropped). PIT contract: ``load(asof) == clean_bars_v3(raw[date <= asof])`` (truncate first, then clean).
Cleaning looks up to 5 bars ahead for a spike reversal, so ``load(asof) == load(later)[:asof]`` holds only
for ``asof`` more than 5 bars past any spike; cleaning the FULL history and then truncating would leak
future information and is deliberately not done here. Not imported by any live module.

Store layout under ``data_root`` (default ``data/research/eodhd``): ``etfs_full/<SYM>.parquet``,
``symbols_delisted.parquet`` (Code, Type, ...; no delisting date, so ``delisted_on`` is the last bar date),
``forex/USDILS.parquet``. Bank of Israel / CBS series live under ``il_macro_root`` (default
``data/research/il_macro``). OWNER ACTION: that directory is not in the research freeze allow-list yet, so
``data_access`` refuses it (fail-closed) until the owner adds it to ``allow_roots``.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from firm.data.cleaning import clean_bars_v3, equity_calendar_v3
from firm.research import data_access, seal

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = _REPO_ROOT / "data" / "research" / "eodhd"
DEFAULT_IL_MACRO_ROOT = _REPO_ROOT / "data" / "research" / "il_macro"
CPI_DEFAULT_LAG_DAYS = 45  # FLAGGED ASSUMPTION: CBS release lag not verified; used only when no publish_date column exists


@dataclass(frozen=True)
class EtfSeries:
    symbol: str
    bars: pd.DataFrame  # date index; open, high, low, close, adjusted_close, volume, segment
    inception: dt.date
    delisted_on: dt.date | None
    clean_report: dict


def _read_raw(path: Path, asof: dt.date) -> pd.DataFrame:
    df = data_access.read_parquet(path, asof=asof)
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] <= pd.Timestamp(asof)]  # PIT truncate BEFORE cleaning


def _delisted_etf_codes(root: Path) -> set[str]:
    p = root / "symbols_delisted.parquet"
    if not p.exists():
        return set()
    df = pd.read_parquet(data_access.assert_path_allowed(p))  # symbol list, no dates: allow-list checked, no bar data
    return set(df.loc[df["Type"] == "ETF", "Code"].astype(str))


def _universe_symbols(universe_yaml: Path) -> list[str]:
    cfg = yaml.safe_load(Path(universe_yaml).read_text(encoding="utf-8"))
    return [str(i["symbol"]) for i in cfg["instruments"]]


def load_etf_universe_with_report(universe_yaml: Path, *, asof: dt.date, data_root: Path | None = None,
                                  include_delisted: bool = True) -> tuple[dict[str, EtfSeries], dict]:
    """Returns ``(series, report)``; ``report["missing"]`` lists requested histories absent on disk."""
    seal.check_asof(asof, what="load_etf_universe")
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    spy = _read_raw(root / "etfs_full" / "SPY.parquet", asof)  # calendar source, also sealed
    cal = equity_calendar_v3(spy, asof=asof)
    symbols = _universe_symbols(universe_yaml)
    delisted = _delisted_etf_codes(root)
    if include_delisted:
        symbols = symbols + sorted(delisted - set(symbols))
    out: dict[str, EtfSeries] = {}
    missing: list[str] = []
    for sym in symbols:
        if sym in delisted and not include_delisted:
            continue
        path = root / "etfs_full" / f"{sym}.parquet"
        if not path.exists():
            missing.append(sym)
            continue
        bars, rep = clean_bars_v3(_read_raw(path, asof), "equity", cal, symbol=sym)
        if bars.empty:
            missing.append(sym)
            continue
        bars = bars.set_index("date")
        bars.index.name = "date"
        last = bars.index.max().date()
        out[sym] = EtfSeries(sym, bars, bars.index.min().date(), last if sym in delisted else None, rep)
    return out, {"missing": sorted(missing), "asof": asof.isoformat(), "n_loaded": len(out)}


def load_etf_universe(universe_yaml: Path, *, asof: dt.date, data_root: Path | None = None,
                      include_delisted: bool = True) -> dict[str, EtfSeries]:
    return load_etf_universe_with_report(universe_yaml, asof=asof, data_root=data_root, include_delisted=include_delisted)[0]


def total_return(series: EtfSeries) -> pd.Series:
    """``adjusted_close`` pct change within segments (NaN at each segment start). Adjusted close ONLY."""
    b = series.bars
    return b["adjusted_close"].groupby(b["segment"]).pct_change(fill_method=None)


def usd_ils(asof: dt.date, *, source: str = "boi", il_macro_root: Path | None = None,
            data_root: Path | None = None) -> pd.Series:
    """USD/ILS daily series up to ``asof``. ``source='boi'`` (authoritative, il_macro) or ``'eodhd'`` (cross-check)."""
    if source == "boi":
        p = (Path(il_macro_root) if il_macro_root is not None else DEFAULT_IL_MACRO_ROOT) / "usd_ils.parquet"
    elif source == "eodhd":
        p = (Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT) / "forex" / "USDILS.parquet"
    else:
        raise ValueError(f"source must be 'boi' or 'eodhd', not {source!r}")
    seal.check_asof(asof, what="usd_ils")
    df = data_access.read_parquet(p, asof=asof)
    df = df.assign(date=pd.to_datetime(df["date"])).sort_values("date").drop_duplicates("date")
    s = pd.Series(df["close"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name="usd_ils")
    return s[s.index <= pd.Timestamp(asof)]


def il_cpi(asof: dt.date, *, il_macro_root: Path | None = None) -> pd.Series:
    """Israeli CPI, PIT: observations whose publication date is <= ``asof``.

    Uses the ``publish_date`` column when present, else observation month-end plus ``CPI_DEFAULT_LAG_DAYS`` (flagged).
    """
    p = (Path(il_macro_root) if il_macro_root is not None else DEFAULT_IL_MACRO_ROOT) / "il_cpi.parquet"
    seal.check_asof(asof, what="il_cpi")
    raw = pd.read_parquet(data_access.assert_path_allowed(p))
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").drop_duplicates("date")
    if "publish_date" in raw.columns:
        pub = pd.to_datetime(raw["publish_date"])
    else:
        pub = raw["date"] + pd.offsets.MonthEnd(0) + pd.Timedelta(days=CPI_DEFAULT_LAG_DAYS)
    keep = ((pub <= pd.Timestamp(asof)) & (raw["date"] <= pd.Timestamp(asof))).to_numpy()
    kept = raw.loc[keep]
    seal.check_frame(kept, what=str(p))
    return pd.Series(kept["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(kept["date"]), name="il_cpi")


def load_dividends(symbols: list[str], *, asof: dt.date, data_root: Path | None = None) -> pd.DataFrame:
    """Cash dividends per share from ``corporate_actions/dividends/<SYM>.parquet`` as ``date, symbol, amount`` (ex-date <= asof).

    ``amount`` is the UNADJUSTED per-share value (``unadjustedValue`` when present, else ``value``): the column layout is
    the EODHD dividend export and is unverified offline. Missing files are skipped and logged, not silently invented.
    """
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    frames = []
    for sym in symbols:
        p = root / "corporate_actions" / "dividends" / f"{sym}.parquet"
        if not p.exists():
            log.warning("no dividend file for %s", sym)
            continue
        df = _read_raw(p, asof)
        col = "unadjustedValue" if "unadjustedValue" in df.columns else "value"
        frames.append(pd.DataFrame({"date": df["date"], "symbol": sym, "amount": pd.to_numeric(df[col], errors="coerce")}))
    if not frames:
        return pd.DataFrame({"date": pd.to_datetime([]), "symbol": [], "amount": []})
    return pd.concat(frames, ignore_index=True).dropna().sort_values(["date", "symbol"]).reset_index(drop=True)
