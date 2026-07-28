"""CLI: run a full backtest.

Usage::

    run-backtest --config config/settings.yaml
    python scripts/run_backtest.py --config config/settings.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from firm.config import get_settings
from firm.data.pit_store import PointInTimeDataStore
from firm.runtime import (
    build_orchestrator,
    build_universe_resolver,
    load_fundamentals,
    load_prices,
    load_sentiment,
)


log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a full backtest")
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings YAML (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for run artifacts (default: runs/<timestamp>)",
    )
    parser.add_argument(
        "--tearsheet",
        action="store_true",
        help="Also render a QuantStats HTML tear-sheet (needs the 'report' extra)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    settings = get_settings(args.config)
    bt_config = settings.backtest.model_dump()
    risk_config = settings.risk.model_dump()

    log.info("Loading price data from %s", settings.data.cache_dir)
    prices_df = load_prices(settings)

    log.info("Determining universe")
    fund_df = load_fundamentals(settings)
    if fund_df is not None:
        log.info(
            "Loaded fundamentals cache: %d rows, %d symbols",
            len(fund_df), fund_df["symbol"].nunique(),
        )
    sentiment_df = load_sentiment(settings)
    if sentiment_df is not None:
        log.info(
            "Loaded sentiment cache: %d rows, %d symbols",
            len(sentiment_df), sentiment_df["symbol"].nunique(),
        )
    else:
        log.debug(
            "No cached sentiment for this backtest; the sentiment strategy "
            "will emit no signals (see firm.strategies.sentiment)"
        )

    # A single load() call — passing fundamentals/sentiment to a *second*
    # call without `prices` would raise (prices has no default and each
    # call fully replaces prior state), which silently meant any real
    # dataset with cached fundamentals crashed this CLI entry point outright.
    pit_store = PointInTimeDataStore()
    pit_store.load(prices=prices_df, fundamentals=fund_df, sentiment=sentiment_df)

    start_dt = datetime.fromisoformat(settings.backtest.start_date)
    end_dt = datetime.fromisoformat(settings.backtest.end_date)
    fallback_symbols = sorted(prices_df["symbol"].astype(str).unique().tolist())
    pit_store.set_universe_resolver(build_universe_resolver(settings, fallback_symbols))
    # Union across the whole window (not just a start_date snapshot) so a
    # symbol added to the index mid-backtest still gets its feed loaded;
    # FirmStrategy resolves the actually-active subset every rebalance.
    universe = pit_store.get_universe_union(start_dt, end_dt)
    if not universe:
        log.error("Empty universe — nothing to backtest")
        sys.exit(1)
    log.info("Universe: %d symbols", len(universe))

    merged_config = {**bt_config, **risk_config, "strategies": settings.strategies}
    if settings.strategy_params:
        merged_config["strategy_params"] = settings.strategy_params
    # Optional signal-combination / allocation knobs (backtest parity with live).
    merged_config["allocation_method"] = settings.allocation_method
    merged_config["kelly_fraction"] = settings.kelly_fraction
    if settings.signal_combination:
        merged_config["signal_combination"] = settings.signal_combination
    orchestrator = build_orchestrator(merged_config)

    from firm.backtest.engine import BacktestEngine

    engine = BacktestEngine(bt_config)
    engine.setup(prices_df, pit_store, orchestrator, universe)

    log.info("Running backtest: %s → %s", settings.backtest.start_date, settings.backtest.end_date)
    engine.run()

    report = engine.generate_report()
    print(report.to_text())

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f"runs/{datetime.now():%Y%m%d_%H%M%S}"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    report.save(str(out / "report.json"))
    log.info("Results saved to %s", out)

    if args.tearsheet:
        try:
            from firm.eval.tearsheet import render_tearsheet

            html_path = render_tearsheet(
                report.returns,
                benchmark=(
                    report.benchmark_returns
                    if not report.benchmark_returns.empty
                    else None
                ),
                out_html=str(out / "tearsheet.html"),
            )
            log.info("Tear-sheet written to %s", html_path)
        except ImportError as exc:
            log.error("Tear-sheet skipped: %s", exc)
        except Exception as exc:
            log.error("Tear-sheet rendering failed: %s", exc)


if __name__ == "__main__":
    main()
