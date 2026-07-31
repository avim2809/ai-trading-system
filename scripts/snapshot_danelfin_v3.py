#!/usr/bin/env python
"""Shadow-mode snapshot of Danelfin's /v3/* latest-snapshot endpoints.

These endpoints (best-stocks, trading-parameters, price-forecast,
performance) have no historical dates (per Danelfin's own docs) — they
can't be backtested, and are deliberately NOT wired into any strategy or
risk/execution logic (see docs/investing_pro_integration.md). This script
exists purely to build a forward observational record — run it periodically
(e.g. daily via cron) and it appends one JSON line per run to a log file, so
there's a real history to look back on later if these signals are ever
reconsidered, without ever having acted on them.

Usage:
    python scripts/snapshot_danelfin_v3.py
    python scripts/snapshot_danelfin_v3.py --symbols AAPL,MSFT --out data/danelfin_v3_shadow.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.config import get_settings  # noqa: E402
from firm.data.providers.danelfin import DanelfinProvider  # noqa: E402

log = logging.getLogger(__name__)

_DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOG", "AMZN", "META", "TSLA", "AVGO", "AMD",
    "CRM", "NFLX", "ADBE", "JPM", "GS", "BAC", "V", "MA", "JNJ", "UNH",
    "LLY", "XOM", "CVX", "SPY", "QQQ", "IWM",
]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", default=None, help="Comma-separated tickers (default: live universe)")
    p.add_argument("--out", default="data/danelfin_v3_shadow.jsonl", help="JSONL log path (append)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    symbols = [s.strip().upper() for s in args.symbols.split(",")] if args.symbols else _DEFAULT_UNIVERSE

    provider = DanelfinProvider(settings=get_settings())
    snapshot: dict = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "best_stocks": [],
        "per_symbol": {},
    }

    try:
        best = provider.get_best_stocks()
        snapshot["best_stocks"] = best.to_dict(orient="records") if not best.empty else []
        log.info("best_stocks: %d entries", len(snapshot["best_stocks"]))
    except Exception:
        log.warning("get_best_stocks failed — continuing without it", exc_info=True)

    for i, symbol in enumerate(symbols):
        if i > 0:
            time.sleep(1.0)  # same pacing as DanelfinProvider's own requests
        entry: dict = {}
        for name, fn in (
            ("trading_parameters", lambda: provider.get_trading_parameters(symbol)),
            ("price_forecast", lambda: provider.get_price_forecast(symbol, horizon="3m")),
            ("performance", lambda: provider.get_performance(symbol, signal="buy")),
        ):
            try:
                entry[name] = fn()
            except Exception:
                log.warning("%s failed for %s — skipping", name, symbol, exc_info=True)
                entry[name] = None
            time.sleep(1.0)
        snapshot["per_symbol"][symbol] = entry
        log.info("captured %s (%d/%d)", symbol, i + 1, len(symbols))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(snapshot, default=str) + "\n")
    log.info("Appended snapshot to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
