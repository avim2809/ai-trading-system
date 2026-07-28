#!/usr/bin/env python
"""Append an interim snapshot for the LLM A/B paper experiment.

Reads portfolio history from ``data/live_state.db`` and prints Sharpe / max
drawdown / cycle count for the current arm. Use weekly during arm A/B runs.

Usage:
    python scripts/snapshot_llm_ab_arm.py
    python scripts/snapshot_llm_ab_arm.py --append docs/llm_ab_experiment_log.md
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from firm.live.state_store import LiveStateStore  # noqa: E402

log = logging.getLogger(__name__)


def _daily_returns(nav_series: list[float]) -> list[float]:
    if len(nav_series) < 2:
        return []
    out: list[float] = []
    for i in range(1, len(nav_series)):
        prev, cur = nav_series[i - 1], nav_series[i]
        if prev and prev > 0:
            out.append((cur - prev) / prev)
    return out


def _sharpe(daily_returns: list[float]) -> float | None:
    if len(daily_returns) < 5:
        return None
    mean = sum(daily_returns) / len(daily_returns)
    var = sum((r - mean) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
    if var <= 0:
        return None
    return mean / math.sqrt(var) * math.sqrt(252)


def _max_drawdown(nav_series: list[float]) -> float | None:
    if not nav_series:
        return None
    peak = nav_series[0]
    max_dd = 0.0
    for nav in nav_series:
        peak = max(peak, nav)
        if peak > 0:
            max_dd = max(max_dd, (peak - nav) / peak)
    return max_dd


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/live_state.db")
    parser.add_argument(
        "--append",
        default=None,
        help="Append markdown snapshot block to this file",
    )
    args = parser.parse_args()

    store = LiveStateStore(args.db)
    snapshots = store.load_portfolio_history()
    navs = [float(s.nav) for s in snapshots if s.nav]
    rets = _daily_returns(navs)
    sharpe = _sharpe(rets)
    max_dd = _max_drawdown(navs)

    llm_config = os.environ.get("FIRM_LLM_CONFIG", "(default llm.yaml)")
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    snapshot = {
        "timestamp": now,
        "firm_llm_config": llm_config,
        "snapshots": len(snapshots),
        "nav_latest": navs[-1] if navs else None,
        "daily_sharpe_ann": sharpe,
        "max_drawdown": max_dd,
    }

    print(json.dumps(snapshot, indent=2, default=str))

    if args.append:
        path = Path(args.append)
        block = (
            f"\n### Snapshot {now}\n\n"
            f"- `FIRM_LLM_CONFIG`: `{llm_config}`\n"
            f"- Portfolio snapshots: {len(snapshots)}\n"
            f"- Latest NAV: {navs[-1] if navs else 'n/a'}\n"
            f"- Ann. Sharpe (daily): {sharpe if sharpe is not None else 'n/a'}\n"
            f"- Max drawdown: {max_dd if max_dd is not None else 'n/a'}\n"
        )
        with path.open("a", encoding="utf-8") as f:
            f.write(block)
        log.info("Appended snapshot to %s", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
