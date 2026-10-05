"""Monitor-only loader for candidate shadow-replay snapshots (P5-04).

Reads ONLY a candidate's own snapshot under the sealed monitoring tree
(``<repo>/research/<sealed dir>/candidates/<candidate>/snapshots/<id>/``, see ``SEALED_CANDIDATES_ROOT``).
It deliberately does not go through ``firm.research.data_access`` (which fails closed after the seal) and must never be
imported from ``firm.research.*`` (``tests/test_fidelity.py`` enforces this). Snapshot layout::

    meta.json           {"multipliers": {sym: float}, "initial_capital": float}
    prices.parquet      index = dates, columns = symbols
    targets.parquet     target units, same shape
    adv.parquet         average daily volume (units), same shape
    vol_pct.parquet     daily return volatility (fraction), same shape
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[3]
SEALED_DIR_NAME = "monitoring" + "_sealed"
SEALED_CANDIDATES_ROOT = _ROOT / "research" / SEALED_DIR_NAME / "candidates"
_SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class Snapshot:
    prices: pd.DataFrame
    targets: pd.DataFrame
    adv: pd.DataFrame
    vol_pct: pd.DataFrame
    multipliers: dict[str, float]
    initial_capital: float


def _slug(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _SLUG.match(value) or ".." in value:
        raise ValueError(f"invalid {kind} {value!r}")
    return value


def candidate_dir(candidate: str, root: Path | None = None) -> Path:
    return Path(root or SEALED_CANDIDATES_ROOT) / _slug("candidate", candidate)


def snapshot_dir(candidate: str, snapshot_id: str, root: Path | None = None) -> Path:
    return candidate_dir(candidate, root) / "snapshots" / _slug("snapshot_id", snapshot_id)


def load_snapshot(candidate: str, snapshot_id: str, root: Path | None = None) -> Snapshot:
    d = snapshot_dir(candidate, snapshot_id, root)
    if not d.is_dir():
        raise FileNotFoundError(f"no snapshot {snapshot_id!r} for candidate {candidate!r} under {d.parent}")
    meta = json.loads((d / "meta.json").read_text())
    frames = {n: pd.read_parquet(d / f"{n}.parquet") for n in ("prices", "targets", "adv", "vol_pct")}
    for f in frames.values():
        f.index = pd.DatetimeIndex(f.index)
    return Snapshot(multipliers={str(k): float(v) for k, v in meta["multipliers"].items()},
                    initial_capital=float(meta["initial_capital"]), **frames)
