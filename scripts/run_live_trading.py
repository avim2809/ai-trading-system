#!/usr/bin/env python3
"""Run the full strategy pipeline against IBKR paper account.

This drives the backend's :class:`LiveTradingEngine` directly from the main
thread (which owns an asyncio event loop), avoiding the event-loop error that
ib_insync raises when instantiated inside FastAPI's worker thread pool.

It mirrors exactly what ``POST /api/live/start`` does, but runs standalone so
it can execute cycles in a simple loop and be deployed in a screen session.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Make ``firm`` importable when run from a source checkout.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from firm.brokers.ibkr import IBKRBroker
from firm.live.approval import ApprovalQueue
from firm.live.data_feed import LiveDataFeed
from firm.live.engine import LiveTradingEngine
import firm.strategies  # noqa: F401 — ensure @register decorators fire

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("firm.live.runner")


DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "V", "JNJ",
]


def _build_providers(host: str, port: int) -> dict:
    """Wire data providers keyed as the LiveDataFeed expects.

    LiveDataFeed looks up providers by role: ``"prices"``, ``"fundamentals"``,
    ``"sentiment"``.  Prices come from IB Gateway itself (no third-party signup);
    optional fundamentals/sentiment are added if their API keys are configured.
    """
    from firm.data.providers.ibkr import IBKRProvider

    # Single IBKR connection (client_id=2, separate from the trading
    # connection on 1) serves both price history and news sentiment.
    ibkr = IBKRProvider(host=host, port=port, client_id=2)
    providers: dict = {
        "prices": ibkr,
        "sentiment": ibkr,
    }

    # Optional fundamentals – only added when a real API key is present.
    optional = (
        ("fundamentals", "firm.data.providers.fmp", "FMPProvider"),
    )
    for role, module_path, cls_name in optional:
        try:
            module = __import__(module_path, fromlist=[cls_name])
            providers[role] = getattr(module, cls_name)()
        except Exception as exc:
            log.info("%s provider not configured (%s); skipping.", role, exc)
    return providers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval", type=int, default=300,
        help="Seconds between trading cycles (default: 300).",
    )
    parser.add_argument(
        "--max-cycles", type=int, default=0,
        help="Stop after this many cycles (0 = run forever).",
    )
    parser.add_argument(
        "--approval-mode", default="auto",
        choices=["auto", "semi_auto", "manual"],
        help="auto submits every proposed order; semi_auto/manual queue them.",
    )
    parser.add_argument(
        "--auto-approve", default="momentum,trend,mean_reversion",
        help="Comma-separated strategy names auto-submitted in semi_auto mode.",
    )
    parser.add_argument(
        "--symbols", default=",".join(DEFAULT_UNIVERSE),
        help="Comma-separated trading universe.",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=1_000_000.0,
    )
    args = parser.parse_args(argv)

    universe = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    auto_approve = [s.strip() for s in args.auto_approve.split(",") if s.strip()]

    host = os.getenv("IBKR_HOST", "127.0.0.1")
    port = int(os.getenv("IBKR_PAPER_PORT", "4002"))
    client_id = int(os.getenv("IBKR_CLIENT_ID", "1"))

    log.info("Configuring IB Gateway broker at %s:%d (client %d)", host, port, client_id)
    broker = IBKRBroker(host=host, port=port, client_id=client_id)

    config = {"initial_capital": args.initial_capital, "symbols": universe}
    data_feed = LiveDataFeed(providers=_build_providers(host, port), universe=universe)
    approval_queue = ApprovalQueue(broker=broker, persist_path="data/approvals.json")

    engine = LiveTradingEngine(
        config=config,
        broker=broker,
        data_feed=data_feed,
        approval_queue=approval_queue,
        approval_mode=args.approval_mode,
        auto_approve_strategies=auto_approve,
    )
    # Engine.start() connects the broker and validates the account.
    engine.start()
    account = broker.get_account()
    log.info(
        "Engine started | equity=$%s | mode=%s | auto-approve=%s | universe=%s",
        f"{account.get('equity', 0.0):,.2f}",
        args.approval_mode, auto_approve or "(none)", universe,
    )

    cycle = 0
    try:
        while args.max_cycles == 0 or cycle < args.max_cycles:
            cycle += 1
            result = engine.run_cycle()
            log.info(
                "Cycle %d | generated=%d submitted=%d queued=%d failed=%d%s",
                cycle,
                result.orders_generated,
                result.orders_submitted,
                result.orders_queued,
                result.orders_failed,
                f" | ERROR: {result.error}" if result.error else "",
            )
            if args.max_cycles and cycle >= args.max_cycles:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down.")
    finally:
        engine.stop()
        broker.disconnect()
        log.info("Engine stopped and broker disconnected.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
