"""JSON-persisted state for dynamically-added universe symbols.

General-purpose, not tied to any one signal source: records which symbols
are *currently* held in the live universe beyond the static
``config/live.yaml`` list, why (which sector, so ``RiskAgent``'s
concentration cap can be enforced correctly), when they were added, and how
many consecutive days they've been absent from whatever drove their
addition (used for dwell-based removal — see
``firm.live.danelfin_universe_sync.compute_universe_update``). A restart
must not silently forget these additions, so this state survives process
restarts on disk, mirroring the plain read/write idiom already used for
``kill_switch_state.json`` (``LiveTradingEngine._load_kill_switch_state``/
``_persist_kill_switch_state`` in ``firm.live.engine``): fail-soft on read
(log + fall back to empty state), ``mkdir(parents=True)`` before write.

Schema: ``{symbol: {"sector": str, "added_date": "YYYY-MM-DD",
"consecutive_absent_days": int}}``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def load_dynamic_universe_state(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the persisted dynamic-universe state, or an empty dict if absent/corrupt."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        log.warning(
            "Failed to read dynamic-universe state from %s — starting "
            "empty (any previously-added dynamic symbols will need to be "
            "re-discovered by the next sync)", p, exc_info=True,
        )
        return {}
    if not isinstance(data, dict):
        log.warning("Dynamic-universe state at %s is not a dict — ignoring", p)
        return {}
    return data


def save_dynamic_universe_state(path: str | Path, state: dict[str, dict[str, Any]]) -> None:
    """Persist the dynamic-universe state, creating parent dirs as needed."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(state, indent=2, sort_keys=True))
    except Exception:
        log.warning("Failed to persist dynamic-universe state to %s", p, exc_info=True)
