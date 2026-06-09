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
from firm.runtime import build_orchestrator, load_prices


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
    pit_store = PointInTimeDataStore()
    pit_store.load(prices=prices_df)

    start_dt = datetime.fromisoformat(settings.backtest.start_date)
    universe = pit_store.get_universe(start_dt)
    if not universe:
        log.error("Empty universe — nothing to backtest")
        sys.exit(1)
    log.info("Universe: %d symbols", len(universe))

    merged_config = {**bt_config, **risk_config}
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


if __name__ == "__main__":
    main()
