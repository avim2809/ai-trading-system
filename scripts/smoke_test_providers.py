"""Smoke-test all configured data providers against their live APIs.

Run from the repo root:
    python scripts/smoke_test_providers.py

Each provider makes one lightweight real request (small date range, single
symbol).  Pass/fail is printed per provider and per capability.  Exit code is
non-zero if any provider fails.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Make sure the package is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from firm.config import get_settings
from firm.data.providers.fmp import FMPProvider
from firm.data.providers.massive import MassiveProvider
from firm.data.providers.tiingo import TiingoProvider
from firm.data.providers.alphavantage import AlphaVantageProvider

SYMBOL = "AAPL"
START = "2026-06-01"
END = "2026-06-06"

OK = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m–\033[0m"


_PLAN_PHRASES = (
    "upgrade your plan", "not entitled", "premium", "subscription",
    "rate limit", "requests per day", "per second", "forbidden",
)


def _is_plan_limit(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(p in msg for p in _PLAN_PHRASES)


def check(label: str, fn) -> bool:
    try:
        result = fn()
        if result is None or (hasattr(result, "empty") and result.empty):
            print(f"  {SKIP} {label}: returned empty (key ok, no data for range?)")
            return True
        rows = len(result) if hasattr(result, "__len__") else "?"
        print(f"  {OK} {label}: {rows} rows")
        return True
    except NotImplementedError as exc:
        print(f"  {SKIP} {label}: not implemented on this provider ({exc})")
        return True
    except Exception as exc:
        if _is_plan_limit(exc):
            print(f"  {SKIP} {label}: plan limit — key valid but this requires a paid tier")
            return True
        print(f"  {FAIL} {label}: {exc}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False


def test_massive(cfg) -> bool:
    key = cfg.massive_api_key
    if not key:
        print(f"  {SKIP} MASSIVE_API_KEY not set — skipping")
        return True
    p = MassiveProvider(api_key=key)
    results = [
        check("prices",            lambda: p.get_prices([SYMBOL], START, END)),
        check("news_sentiment",    lambda: p.get_news_sentiment([SYMBOL], START, END)),
        check("fundamentals",      lambda: p.get_fundamentals([SYMBOL], START, END)),
        check("corporate_actions", lambda: p.get_corporate_actions([SYMBOL], "2020-01-01", END)),
    ]
    return all(results)


def test_tiingo(cfg) -> bool:
    key = cfg.tiingo_api_key
    if not key:
        print(f"  {SKIP} TIINGO_API_KEY not set — skipping")
        return True
    p = TiingoProvider(api_key=key)
    results = [
        check("prices",         lambda: p.get_prices([SYMBOL], START, END)),
        check("news_sentiment", lambda: p.get_news_sentiment([SYMBOL], START, END)),
    ]
    return all(results)


def test_alphavantage(cfg) -> bool:
    key = cfg.alphavantage_api_key
    if not key:
        print(f"  {SKIP} ALPHAVANTAGE_API_KEY not set — skipping")
        return True
    p = AlphaVantageProvider(api_key=key)
    results = [
        check("prices",         lambda: p.get_prices([SYMBOL], START, END)),
        check("news_sentiment", lambda: p.get_news_sentiment([SYMBOL], START, END)),
    ]
    return all(results)


def test_fmp(cfg) -> bool:
    key = cfg.fmp_api_key
    if not key:
        print(f"  {SKIP} FMP_API_KEY not set — skipping")
        return True
    p = FMPProvider(api_key=key)
    results = [
        check("prices",            lambda: p.get_prices([SYMBOL], START, END)),
        check("fundamentals",      lambda: p.get_fundamentals([SYMBOL], START, END)),
        check("corporate_actions", lambda: p.get_corporate_actions([SYMBOL], "2020-01-01", END)),
    ]
    return all(results)


def main() -> int:
    cfg = get_settings()

    suite = [
        ("Massive (Polygon)",  test_massive),
        ("Tiingo",             test_tiingo),
        ("AlphaVantage",       test_alphavantage),
        ("FMP",                test_fmp),
    ]

    failed: list[str] = []
    for name, fn in suite:
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        ok = fn(cfg)
        if not ok:
            failed.append(name)

    print(f"\n{'═'*50}")
    if failed:
        print(f"{FAIL}  Failed: {', '.join(failed)}")
        return 1
    print(f"{OK}  All providers passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
