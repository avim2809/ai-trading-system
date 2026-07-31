#!/usr/bin/env python
"""Daily driver for the Danelfin Best-Stocks synthetic paper arm.

Run once/day (e.g. via cron). On any given day this either:
  - initializes the ledger (first run: select 25 stocks, equal-weight,
    full_rebalance) — Danelfin API calls: ~11 (one per sector).
  - marks the existing holdings to market (the common case) — zero
    Danelfin calls, just a price lookup for the held symbols.
  - if >= 91 days since the last quarterly replace: re-runs the selection
    and swaps out any holding that no longer qualifies — ~11 Danelfin
    calls.
  - if >= 365 days since the last full/annual rebalance: resets holdings
    back to equal dollar weighting (no Danelfin calls, price lookup only).

State persists to --ledger (default data/best_stocks_ledger.json) — a
JSON file, not firm.live.state_store.LiveStateStore (that store has no
experiment-name column; see best_stocks_arm.py's module docstring for why
this arm intentionally doesn't share the main engine's state DB).

Usage:
    python scripts/run_best_stocks_arm.py
    python scripts/run_best_stocks_arm.py --ledger data/best_stocks_ledger.json --initial-capital 100000
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.config import get_settings  # noqa: E402
from firm.data.providers.danelfin import DanelfinProvider  # noqa: E402
from firm.data.providers.fallback import FallbackProvider  # noqa: E402
from firm.live.best_stocks_arm import select_best_stocks, selection_symbols  # noqa: E402
from firm.live.best_stocks_ledger import BestStocksLedger  # noqa: E402

log = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", default="data/best_stocks_ledger.json")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    return p.parse_args(argv)


def _latest_prices(provider: FallbackProvider, symbols: list[str], asof: datetime) -> dict[str, float]:
    if not symbols:
        return {}
    start = (asof - timedelta(days=10)).strftime("%Y-%m-%d")
    end = asof.strftime("%Y-%m-%d")
    df = provider.get_prices(symbols, start, end)
    if df.empty:
        return {}
    latest = df.sort_values("date").groupby("symbol").last()
    return {sym: float(row["adj_close"]) for sym, row in latest.iterrows()}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    asof = datetime.now(timezone.utc)

    settings = get_settings()
    danelfin = DanelfinProvider(settings=settings)
    market_data = FallbackProvider(settings=settings)

    ledger = BestStocksLedger.load(args.ledger)
    if not ledger.holdings and not ledger.nav_history:
        ledger.initial_capital = args.initial_capital
        ledger.cash = args.initial_capital

    if not ledger.holdings:
        log.info("best_stocks_arm: no holdings — initial selection + full rebalance")
        selection = select_best_stocks(danelfin)
        if not selection:
            log.error("best_stocks_arm: selection returned nothing; leaving ledger uninitialized")
            return 1
        symbols = selection_symbols(selection)
        prices = _latest_prices(market_data, symbols, asof)
        ledger.full_rebalance(asof, selection, prices)
        nav = ledger.mark_to_market(asof, prices)
        log.info("best_stocks_arm: initialized with %d holdings, nav=%.2f", len(ledger.holdings), nav)
        ledger.save(args.ledger)
        return 0

    prices = _latest_prices(market_data, list(ledger.holdings), asof)

    if ledger.due_for_quarterly_replace(asof):
        log.info("best_stocks_arm: quarterly replace due")
        selection = select_best_stocks(danelfin)
        if selection:
            extra_symbols = [s for s in selection_symbols(selection) if s not in prices]
            prices.update(_latest_prices(market_data, extra_symbols, asof))
            ledger.quarterly_replace(asof, selection, prices)
        else:
            log.warning("best_stocks_arm: quarterly replace due but selection returned nothing; skipping")

    if ledger.due_for_annual_rebalance(asof):
        log.info("best_stocks_arm: annual rebalance due")
        ledger.annual_rebalance(asof, prices)

    nav = ledger.mark_to_market(asof, prices)
    log.info("best_stocks_arm: nav=%.2f n_holdings=%d", nav, len(ledger.holdings))
    ledger.save(args.ledger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
