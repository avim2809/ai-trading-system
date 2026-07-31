"""Real IBKR paper-order execution for the Danelfin Best-Stocks arm.

The user's explicit choice (see docs/danelfin_best_stocks_arm.md) was to run
this arm's real orders through the SAME IBKR paper account as the main
engine, accepting the collision risk in exchange for not needing a second
account, PLUS a symbol-collision guard: this arm must never hold a position
in a symbol the main engine's own universe also trades.

Why this matters: IBKR nets all fills into one account-level position
regardless of which client_id submitted them (confirmed by reading
IBKRBroker.get_positions -> ib.portfolio(), which has no account/model-code
filter, and IBKRBroker.submit_order, which never sets order.account or
order.modelCode). If both arms ever held the same symbol, the main engine's
ExecutionAgent could unwind a position this arm believes it still holds
(and vice versa), silently corrupting both arms' tracking. The guard below
is the only thing standing between "two independent paper strategies" and
"one confused shared position book".

The guard is checked in TWO places for defense-in-depth:
  1. At selection time (best_stocks_arm.select_best_stocks's
     excluded_symbols param) — colliding candidates are dropped BEFORE
     ranking, so the arm fills its 25 slots from non-colliding names
     wherever possible.
  2. At order-submission time (BestStocksLedger.rebalance_via_broker) — a
     fresh exclusion check right before placing orders, in case the main
     engine's universe changed between selection and execution (e.g. a
     runtime PUT /api/live/config universe edit).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import requests
import yaml

log = logging.getLogger("firm.live.best_stocks_execution")

_LIVE_YAML_PATH_ENV = "FIRM_LIVE_CONFIG"
_DEFAULT_LIVE_YAML = "config/live.yaml"
_LIVE_API_BASE_URL_ENV = "FIRM_API_BASE_URL"
_DEFAULT_LIVE_API_BASE_URL = "http://127.0.0.1:8000"


def static_main_universe(live_yaml_path: str | Path | None = None) -> set[str]:
    """The main engine's universe as configured on disk — what's active on
    every restart, and readable without the API being up."""
    path = Path(live_yaml_path or os.getenv(_LIVE_YAML_PATH_ENV, _DEFAULT_LIVE_YAML))
    if not path.exists():
        log.warning(
            "best_stocks_execution: live.yaml not found at %s; treating main universe as empty", path,
        )
        return set()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    symbols = (cfg.get("universe") or {}).get("symbols") or []
    return {str(s).upper() for s in symbols}


def live_main_universe(base_url: str | None = None, timeout: float = 3.0) -> set[str]:
    """Best-effort: the main engine's CURRENT in-memory universe via
    GET /api/live/config — PUT /api/live/config can change the live
    universe without touching the YAML file, so the static list alone can
    be stale. Fails soft (empty set) if the API isn't reachable; the
    static YAML check still applies regardless, so this is defense in
    depth, not the only guard."""
    url = (base_url or os.getenv(_LIVE_API_BASE_URL_ENV, _DEFAULT_LIVE_API_BASE_URL)).rstrip("/") + "/api/live/config"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        symbols = (data.get("universe") or {}).get("symbols") or []
        return {str(s).upper() for s in symbols}
    except Exception:
        log.warning(
            "best_stocks_execution: could not fetch live /api/live/config universe; "
            "using static YAML only", exc_info=True,
        )
        return set()


def main_engine_excluded_symbols(live_yaml_path: str | Path | None = None) -> set[str]:
    """Union of static + live main-engine universe — the Best-Stocks arm
    must never trade any symbol in this set."""
    return static_main_universe(live_yaml_path) | live_main_universe()
