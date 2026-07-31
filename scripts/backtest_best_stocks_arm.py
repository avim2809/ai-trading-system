#!/usr/bin/env python
"""Walk-forward backtest of the Danelfin Best-Stocks methodology.

Discovered while building the live arm: Danelfin's ``/ranking`` endpoint
(already used for the genuinely-historical ``danelfin_ai_score`` strategy)
ALSO supports a bulk historical mode — ``date``+``sector`` with no
``ticker`` returns every matching symbol for that historical date, not one
ticker's timeline (see DanelfinProvider.get_historical_sector_scores's
docstring for the exact-match-filter / pagination / 404-means-no-data
details found live). That means the Best-Stocks sector-ranking methodology
CAN be honestly walk-forward tested, unlike the live arm's ``/v3/
trade-ideas`` (genuinely snapshot-only).

Three disclosed, deliberate simplifications versus the live arm's rules —
not silent deviations:

1. **No "Proven Buy Signal" filter.** ``/ranking`` has no buy/hold/sell
   field at any date. This backtest uses only low_risk>=5 + aiscore
   ranking — the closest available historical proxy, not the literal rule.
2. **Annual rebalancing, not quarterly replace + annual reweight.** A
   full quarterly walk-forward (~34 rebalance events over 8+ years) would
   need on the order of tens of thousands of Danelfin API calls (11
   sectors x ~6 low_risk values x multiple pages, per rebalance date) —
   well past a single test's reasonable budget/runtime. Annual rebalancing
   (~9 events) is Danelfin's own stated "reweight" cadence, cuts the call
   count roughly 4x, and is disclosed here as a first-pass simplification,
   not silently substituted for the full rule. A quarterly version is a
   natural, more expensive follow-up if this first pass looks promising.
3. **No real historical liquidity (>100k volume) filter by default.**
   Danelfin's bulk ``/ranking`` mode has no ``average_volume_3m`` field —
   a real historical volume check needs this project's own price
   providers, one fetch per unique candidate symbol. Tried it: a single
   rebalance date's candidate pool spans hundreds to 500+ symbols per
   sector (financials alone: 573 on one sample date), and the underlying
   free-tier providers (Tiingo especially) rate-limit hard enough that
   this ballooned into a many-minutes-per-symbol exponential backoff —
   not a reasonable per-run cost for a multi-year backtest. Pass
   --check-volume to opt into a bounded version anyway (it only checks
   the~25 stocks actually being chosen per rebalance date, not the full
   candidate pool — see select_best_stocks_historical's docstring for why
   that's still a real, disclosed deviation from Danelfin's literal
   filter-then-rank order), but expect it to be slow.

Checkpoints selections + NAV curve to --output after every rebalance date,
since a multi-year run touches many symbols (for forward-return
computation, even without --check-volume) and can take a while; safe to
inspect or resume-by-rereading partway through.

Usage:
    python scripts/backtest_best_stocks_arm.py
    python scripts/backtest_best_stocks_arm.py --start 2018-01-15 --end 2026-07-01 --output /tmp/best_stocks_backtest.json
    python scripts/backtest_best_stocks_arm.py --check-volume   # slow — see simplification #3 above
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.config import get_settings  # noqa: E402
from firm.data.providers.danelfin import DanelfinProvider  # noqa: E402
from firm.data.providers.fallback import FallbackProvider  # noqa: E402
from firm.live.best_stocks_arm import select_best_stocks_historical  # noqa: E402

log = logging.getLogger(__name__)

TRAILING_VOLUME_DAYS = 63  # ~3 trading months
MIN_AVG_VOLUME = 100_000
BENCHMARK_SYMBOL = "SPY"


def _annual_rebalance_dates(start: str, end: str) -> list[str]:
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    dates = []
    d = start_ts
    while d <= end_ts:
        dates.append(d.strftime("%Y-%m-%d"))
        d = d.replace(year=d.year + 1)
    return dates


def _nearest_trading_day(price_cache: "PriceCache", date: str, max_shift_days: int = 10) -> str | None:
    """Nearest real trading day on/after *date*, using the benchmark's own
    price series as the trading-day calendar (it's guaranteed accurate,
    unlike probing Danelfin: a 404 there means "zero rows match this exact
    filter", NOT "not a trading day" — see
    DanelfinProvider.get_historical_sector_scores's docstring for how an
    earlier version of this function got that wrong, using a Danelfin
    probe that could false-negative on a perfectly valid trading day)."""
    df = price_cache.get(BENCHMARK_SYMBOL)
    if df.empty:
        log.error("best_stocks_backtest: no benchmark price data to resolve trading days")
        return None
    target = pd.Timestamp(date)
    candidates = df[df["date"] >= target]["date"]
    if candidates.empty:
        return None
    return candidates.iloc[0].strftime("%Y-%m-%d")


class PriceCache:
    """Fetches each unique symbol's full daily price history ONCE for the
    whole backtest window, reused for both the volume filter and forward
    returns.

    Checks this project's existing on-disk parquet cache FIRST (real,
    already-fetched historical data — e.g. the main 25-symbol universe
    backfilled to ~2010, see docs/remediation_progress.md) before ever
    hitting the live provider chain. This matters a lot in practice: a
    live SPY fetch turned out to be silently truncated to the last ~2
    years (Massive's own tier limit, with Tiingo/FMP too rate-limited to
    fill the gap during this session) — which broke rebalance-date
    resolution outright (every pre-2024 candidate date collapsed onto the
    same truncated earliest-available date). The disk cache has real SPY
    history back to 2010, sidestepping that failure mode entirely for the
    benchmark and any Best-Stocks candidate that happens to also be in the
    main universe; genuinely new candidate symbols still fall through to
    the live chain and pay its real rate-limit cost.
    """

    def __init__(self, provider: FallbackProvider, start: str, end: str, cache_dir: str = "data/cache"):
        self._provider = provider
        # Padded before `start` so the first rebalance date's trailing
        # volume window has real data to look back on.
        self._fetch_start = (pd.Timestamp(start) - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
        self._fetch_end = end
        self._cache: dict[str, pd.DataFrame] = {}
        self._disk_prices = self._scan_disk_cache(cache_dir)
        log.info(
            "best_stocks_backtest: disk price cache covers %d symbols (%s)",
            self._disk_prices["symbol"].nunique() if not self._disk_prices.empty else 0,
            sorted(self._disk_prices["symbol"].unique()) if not self._disk_prices.empty else [],
        )

    @staticmethod
    def _scan_disk_cache(cache_dir: str) -> pd.DataFrame:
        frames = []
        for path in Path(cache_dir).glob("*.parquet"):
            try:
                df = pd.read_parquet(path)
            except Exception:
                continue
            if {"symbol", "date", "adj_close", "volume"}.issubset(df.columns):
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        merged["date"] = pd.to_datetime(merged["date"])
        return (
            merged.sort_values(["symbol", "date"])
            .drop_duplicates(subset=["symbol", "date"], keep="last")
            .reset_index(drop=True)
        )

    def get(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._cache:
            df = pd.DataFrame()
            if not self._disk_prices.empty:
                local = self._disk_prices[self._disk_prices["symbol"] == symbol]
                # Only trust the disk copy if it actually reaches back
                # close to the requested start — a partial/short disk
                # cache entry should still fall through to a live fetch
                # rather than silently truncating the backtest.
                if not local.empty and local["date"].min() <= pd.Timestamp(self._fetch_start) + pd.Timedelta(days=30):
                    df = local
                    log.debug("best_stocks_backtest: disk-cache hit for %s (%d rows)", symbol, len(df))
            if df.empty:
                try:
                    df = self._provider.get_prices([symbol], self._fetch_start, self._fetch_end)
                except Exception:
                    log.warning("best_stocks_backtest_price_fetch_failed symbol=%s", symbol, exc_info=True)
                    df = pd.DataFrame()
            self._cache[symbol] = df.sort_values("date") if not df.empty else df
        return self._cache[symbol]

    def trailing_avg_volume(self, symbol: str, asof: str) -> float | None:
        df = self.get(symbol)
        if df.empty:
            return None
        asof_ts = pd.Timestamp(asof)
        window = df[df["date"] <= asof_ts].tail(TRAILING_VOLUME_DAYS)
        if window.empty or "volume" not in window.columns:
            return None
        return float(window["volume"].mean())

    def price_series(self, symbol: str, start: str, end: str) -> pd.Series:
        df = self.get(symbol)
        if df.empty:
            return pd.Series(dtype=float)
        mask = (df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))
        sub = df.loc[mask].set_index("date")["adj_close"]
        return sub[~sub.index.duplicated(keep="last")]


def _liquid(cache: PriceCache, symbol: str, asof: str) -> bool:
    vol = cache.trailing_avg_volume(symbol, asof)
    return vol is not None and vol > MIN_AVG_VOLUME


def _period_daily_returns(cache: PriceCache, symbols: list[str], start: str, end: str) -> pd.Series:
    """Equal-weight daily portfolio returns for *symbols* over
    [start, end], from each symbol's cached price series."""
    frames = []
    for sym in symbols:
        s = cache.price_series(sym, start, end)
        if not s.empty:
            frames.append(s.rename(sym))
    if not frames:
        return pd.Series(dtype=float)
    prices = pd.concat(frames, axis=1).sort_index().ffill()
    rets = prices.pct_change().dropna(how="all")
    return rets.mean(axis=1)  # equal-weight average of held names' daily returns


def _metrics(daily_returns: pd.Series) -> dict:
    if daily_returns.empty:
        return {"sharpe": None, "cagr": None, "max_drawdown": None, "total_return": None}
    nav = (1 + daily_returns).cumprod()
    total_return = float(nav.iloc[-1] - 1)
    n_years = len(daily_returns) / 252.0
    cagr = float(nav.iloc[-1] ** (1 / n_years) - 1) if n_years > 0 else None
    sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else None
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    max_dd = float(drawdown.min())
    return {"sharpe": sharpe, "cagr": cagr, "max_drawdown": max_dd, "total_return": total_return}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2018-01-15")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--output", default="/tmp/best_stocks_backtest.json")
    parser.add_argument(
        "--check-volume", action="store_true",
        help="Apply a real historical >100k-volume filter to selected candidates "
             "(slow — see simplification #3 in this script's module docstring).",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    danelfin = DanelfinProvider(settings=settings)
    market_data = FallbackProvider(settings=settings)
    price_cache = PriceCache(market_data, args.start, args.end)

    raw_dates = _annual_rebalance_dates(args.start, args.end)
    log.info("best_stocks_backtest: %d candidate annual rebalance dates: %s", len(raw_dates), raw_dates)

    checkpoint: dict = {"rebalance_events": [], "meta": {"start": args.start, "end": args.end, "cadence": "annual"}}
    out_path = Path(args.output)

    valid_dates: list[str] = []
    for raw_date in raw_dates:
        valid = _nearest_trading_day(price_cache, raw_date)
        if valid is None:
            log.warning("best_stocks_backtest: no trading day found near %s — skipping", raw_date)
            continue
        valid_dates.append(valid)
    log.info("best_stocks_backtest: %d valid rebalance dates: %s", len(valid_dates), valid_dates)

    # Guard against a real bug hit while building this: if the benchmark's
    # own price series is silently truncated (e.g. a live fetch quietly
    # capped to the last ~2 years by a provider's tier limit), every
    # earlier candidate date resolves to the SAME earliest-available date
    # in _nearest_trading_day, corrupting the whole backtest with
    # duplicate "rebalance" events instead of a visible error. Fail loudly
    # instead of silently producing a misleadingly-short effective window.
    if len(valid_dates) != len(set(valid_dates)):
        log.error(
            "best_stocks_backtest: duplicate resolved rebalance dates %s — this means the "
            "benchmark's price history is shorter than requested (see PriceCache's docstring "
            "for the exact failure mode this guards against). Aborting rather than silently "
            "running a corrupted backtest.",
            valid_dates,
        )
        return 1

    daily_return_segments: list[pd.Series] = []
    for i, date in enumerate(valid_dates):
        t0 = time.time()
        selection = select_best_stocks_historical(
            danelfin, date,
            volume_filter=(lambda sym, d=date: _liquid(price_cache, sym, d)) if args.check_volume else None,
        )
        elapsed = time.time() - t0
        symbols = [row["symbol"] for row in selection]
        log.info(
            "best_stocks_backtest: date=%s n_holdings=%d elapsed=%.0fs symbols=%s",
            date, len(symbols), elapsed, symbols,
        )

        period_end = valid_dates[i + 1] if i + 1 < len(valid_dates) else args.end
        if symbols:
            period_returns = _period_daily_returns(price_cache, symbols, date, period_end)
            daily_return_segments.append(period_returns)
        else:
            log.warning("best_stocks_backtest: no holdings for date=%s — skipping this period", date)

        checkpoint["rebalance_events"].append({
            "date": date, "n_holdings": len(symbols), "selection": selection,
        })
        out_path.write_text(json.dumps(checkpoint, indent=2, default=str))

    if not daily_return_segments:
        log.error("best_stocks_backtest: no return segments computed at all — aborting")
        return 1

    portfolio_returns = pd.concat(daily_return_segments).sort_index()
    portfolio_returns = portfolio_returns[~portfolio_returns.index.duplicated(keep="first")]
    portfolio_metrics = _metrics(portfolio_returns)

    benchmark_series = price_cache.price_series(BENCHMARK_SYMBOL, valid_dates[0], args.end)
    benchmark_returns = benchmark_series.pct_change().dropna()
    benchmark_metrics = _metrics(benchmark_returns)

    checkpoint["portfolio_metrics"] = portfolio_metrics
    checkpoint["benchmark_metrics"] = benchmark_metrics
    checkpoint["benchmark_symbol"] = BENCHMARK_SYMBOL
    out_path.write_text(json.dumps(checkpoint, indent=2, default=str))

    print("\n=== Danelfin Best-Stocks methodology: walk-forward backtest ===\n")
    print(f"Window: {valid_dates[0]} to {args.end}, {len(valid_dates)} annual rebalance events\n")
    print(f"{'':20s} {'Sharpe':>10s} {'CAGR':>10s} {'MaxDD':>10s} {'TotalReturn':>12s}")
    for label, m in (("Best-Stocks", portfolio_metrics), (BENCHMARK_SYMBOL, benchmark_metrics)):
        sharpe = f"{m['sharpe']:.3f}" if m["sharpe"] is not None else "n/a"
        cagr = f"{m['cagr']:.3f}" if m["cagr"] is not None else "n/a"
        dd = f"{m['max_drawdown']:.3f}" if m["max_drawdown"] is not None else "n/a"
        tr = f"{m['total_return']:.3f}" if m["total_return"] is not None else "n/a"
        print(f"{label:20s} {sharpe:>10s} {cagr:>10s} {dd:>10s} {tr:>12s}")
    print(f"\nFull checkpoint (per-date selections + metrics): {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
