#!/usr/bin/env python
"""Daily driver for the Danelfin Best-Stocks paper arm (synthetic or real).

Run once/day (e.g. via cron/systemd timer). On any given day this either:
  - initializes the ledger (first run: select 25 stocks, equal-weight,
    full rebalance) — Danelfin API calls: ~11 (one per sector).
  - marks the existing holdings to market (the common case) — zero
    Danelfin calls, just a price lookup for the held symbols.
  - if >= 91 days since the last quarterly replace: re-runs the selection
    and swaps out any holding that no longer qualifies — ~11 Danelfin
    calls.
  - if >= 365 days since the last full/annual rebalance: resets holdings
    back to equal dollar weighting (no Danelfin calls, price lookup only).

Two modes:
  - Default (no --live-trading): the original synthetic mode — no broker,
    hypothetical fractional shares, priced via FallbackProvider.
  - --live-trading: places REAL IBKR paper orders (whole shares) through
    the SAME IBKR account as the main engine, on a DISTINCT client_id
    (default 3 — main engine uses 1 for its broker and 2 for its data
    feed; see .cursor/rules/ibkr-integration.mdc). Guarded against
    colliding with the main engine's own tradable universe — see
    firm.live.best_stocks_execution's module docstring for why that
    guard exists and what it protects against. Uses a SEPARATE --ledger
    path by convention (data/best_stocks_ledger_live.json) so the
    synthetic and real-executed histories never overwrite each other.

State persists to --ledger — a JSON file, not
firm.live.state_store.LiveStateStore (that store has no experiment-name
column; see best_stocks_arm.py's module docstring for why this arm
intentionally doesn't share the main engine's state DB).

Usage:
    python scripts/run_best_stocks_arm.py
    python scripts/run_best_stocks_arm.py --ledger data/best_stocks_ledger.json --initial-capital 100000
    python scripts/run_best_stocks_arm.py --live-trading --ledger data/best_stocks_ledger_live.json
"""

from __future__ import annotations

import argparse
import logging
import os
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
from firm.live.best_stocks_execution import main_engine_excluded_symbols  # noqa: E402

log = logging.getLogger(__name__)

_DEFAULT_LIVE_TRADING_CLIENT_ID = 3


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", default="data/best_stocks_ledger.json")
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument(
        "--live-trading", action="store_true",
        help="Place real IBKR paper orders instead of the synthetic ledger math.",
    )
    p.add_argument(
        "--ibkr-client-id", type=int, default=_DEFAULT_LIVE_TRADING_CLIENT_ID,
        help="Distinct IBKR client_id for this arm's broker connection (default 3).",
    )
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


def _run_synthetic(args: argparse.Namespace, asof: datetime) -> int:
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


def _run_live_trading(args: argparse.Namespace, asof: datetime) -> int:
    from firm.brokers.ibkr import IBKRBroker

    settings = get_settings()
    danelfin = DanelfinProvider(settings=settings)

    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PAPER_PORT", "4002"))
    broker = IBKRBroker(host=host, port=port, client_id=args.ibkr_client_id)
    broker.connect()
    try:
        ledger = BestStocksLedger.load(args.ledger)
        if not ledger.holdings and not ledger.nav_history:
            ledger.initial_capital = args.initial_capital
            ledger.cash = args.initial_capital

        excluded = main_engine_excluded_symbols()
        log.info("best_stocks_arm: main-engine excluded symbols: %s", sorted(excluded))

        if not ledger.holdings:
            log.info("best_stocks_arm: [LIVE] no holdings — initial selection + real full rebalance")
            selection = select_best_stocks(danelfin, excluded_symbols=frozenset(excluded))
            if not selection:
                log.error("best_stocks_arm: selection returned nothing; leaving ledger uninitialized")
                return 1
            ledger.rebalance_via_broker(asof, broker, "full", target_selection=selection)
        else:
            if ledger.due_for_quarterly_replace(asof):
                log.info("best_stocks_arm: [LIVE] quarterly replace due")
                selection = select_best_stocks(danelfin, excluded_symbols=frozenset(excluded))
                if selection:
                    ledger.rebalance_via_broker(asof, broker, "quarterly", target_selection=selection)
                else:
                    log.warning("best_stocks_arm: quarterly replace due but selection returned nothing; skipping")

            if ledger.due_for_annual_rebalance(asof):
                log.info("best_stocks_arm: [LIVE] annual rebalance due")
                ledger.rebalance_via_broker(asof, broker, "annual")

        if ledger.holdings:
            prices = broker.get_current_prices(list(ledger.holdings))
        else:
            prices = {}
        nav = ledger.mark_to_market(asof, prices)
        log.info("best_stocks_arm: [LIVE] nav=%.2f n_holdings=%d", nav, len(ledger.holdings))
        ledger.save(args.ledger)
        return 0
    finally:
        broker.disconnect()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args(argv)
    asof = datetime.now(timezone.utc)

    if args.live_trading:
        return _run_live_trading(args, asof)
    return _run_synthetic(args, asof)


if __name__ == "__main__":
    raise SystemExit(main())
