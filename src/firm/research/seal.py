"""Forward-data seal: dates, access checks and the opt-in guards (ticket P0-02).

``config/research_freeze.yaml`` declares everything up to ``burned_through`` (2026-09-30)
in-sample and burned; data from ``seal_date`` (2026-10-01) on is the forward holdout and
must never reach research code. This module

* reads that config (cached; a missing or incoherent config FAILS CLOSED),
* provides ``check_asof`` / ``check_frame``, and
* installs the guard hooks that ``firm.data.pit_store`` and ``firm.runtime`` expose
  (``_ACCESS_GUARD``, ``None`` by default).

There is no environment-variable detection, no token env var and no debug flag: the hooks
are installed only by calling :func:`install_guards` (research entry points do it, directly
or through ``firm.research.data_access``), and :func:`install_guards` refuses to run inside
a live process. Live modules never import this package, so the guard is fail-open for code
that never imports ``firm.research``; that residual risk is stated in ``docs/HOLDOUT_POLICY.md``.
"""

from __future__ import annotations

import datetime as dt
import functools
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_PATH = _REPO_ROOT / "config" / "research_freeze.yaml"

# Importing any of these means "this is a live firm-api process": the guard must stay off.
_LIVE_MODULES = ("firm.live.engine", "firm.api.app")


class HoldoutAccessError(RuntimeError):
    """Raised for any access to sealed (post-seal) data or to a path outside the research allow-list."""


def _refuse(message: str) -> HoldoutAccessError:
    log.warning("holdout guard: %s", message)
    return HoldoutAccessError(message)


def _as_date(value: Any, field: str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value)
        except ValueError:
            pass
    raise _refuse(f"{_CONFIG_PATH.name}: {field} is not a date: {value!r}")


@functools.lru_cache(maxsize=None)
def _load_config() -> dict[str, Any]:
    """Parse and validate the freeze config. Cached; call ``_load_config.cache_clear()`` after changing it."""
    path = _CONFIG_PATH
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        log.error("cannot read research freeze config %s: %s", path, exc)
        raise HoldoutAccessError(f"research freeze config unreadable ({path}): {exc}") from exc
    if not isinstance(raw, dict):
        raise _refuse(f"{path.name} is not a mapping")
    seal = _as_date(raw.get("seal_date"), "seal_date")
    burned = _as_date(raw.get("burned_through"), "burned_through")
    if burned != seal - dt.timedelta(days=1):
        raise _refuse(f"{path.name}: burned_through {burned} must be seal_date {seal} minus one day")
    allow = raw.get("allow_roots")
    deny = raw.get("deny_paths")
    for name, value in (("allow_roots", allow), ("deny_paths", deny)):
        if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
            raise _refuse(f"{path.name}: {name} must be a list of non-empty strings")
    if not allow:
        raise _refuse(f"{path.name}: allow_roots is empty")
    cfg = dict(raw, seal_date=seal, burned_through=burned)
    log.debug("research freeze loaded from %s: seal_date=%s, %d allow roots, %d deny paths", path, seal, len(allow), len(deny))
    return cfg


def config_root() -> Path:
    """Directory that relative config entries resolve against (the checkout holding the config)."""
    return _CONFIG_PATH.resolve().parent.parent


def seal_date() -> dt.date:
    """First SEALED day (2026-10-01)."""
    return _load_config()["seal_date"]


def max_research_date() -> dt.date:
    """Last usable day (``burned_through``, 2026-09-30) = ``seal_date() - 1 day``."""
    return _load_config()["burned_through"]


def config_lists() -> tuple[list[str], list[str]]:
    """``(allow_roots, deny_paths)`` exactly as written in the config."""
    cfg = _load_config()
    return list(cfg["allow_roots"]), list(cfg["deny_paths"])


def check_asof(asof: dt.datetime | dt.date | pd.Timestamp | str, *, what: str) -> None:
    """Raise :class:`HoldoutAccessError` if ``asof`` is later than the last usable day."""
    day = pd.Timestamp(asof).date()
    limit = max_research_date()
    if day > limit:
        raise _refuse(f"{what}: asof {day} > max_research_date {limit} (seal_date {seal_date()})")


def date_values(df: pd.DataFrame, date_col: str = "date") -> pd.Series | None:
    """Naive datetimes of ``df[date_col]`` (or of a DatetimeIndex); ``None`` if the frame carries no dates."""
    if date_col in df.columns:
        values = pd.to_datetime(df[date_col], errors="coerce")
        if getattr(values.dt, "tz", None) is not None:
            values = values.dt.tz_localize(None)
        return values.reset_index(drop=True)
    if isinstance(df.index, pd.DatetimeIndex):
        index = df.index.tz_localize(None) if df.index.tz is not None else df.index
        return pd.Series(index)
    return None


def check_frame(df: pd.DataFrame, *, what: str, date_col: str = "date", allow_all_null: bool = False) -> None:
    """Raise if any row is dated on or after ``seal_date``. Non-empty frames without dates fail closed.

    ``allow_all_null`` is for secondary date columns (e.g. ``removed_date``, null for current members).
    """
    if df is None or len(df) == 0:
        return
    values = date_values(df, date_col)
    if values is None:
        raise _refuse(f"{what}: frame has no '{date_col}' column or DatetimeIndex; cannot prove it is pre-seal")
    values = values.dropna()
    if values.empty:
        if allow_all_null:
            return
        raise _refuse(f"{what}: no parseable dates in '{date_col}'; cannot prove the frame is pre-seal")
    limit = pd.Timestamp(seal_date())
    latest = values.max()
    if latest >= limit:
        raise _refuse(f"{what}: row dated {latest.date()} >= seal_date {seal_date()}")


def _check_frames(kind: str, payload: dict[str, Any]) -> None:
    frames = payload.get("frames") or {}
    cols = tuple(payload.get("date_cols") or ("date",))
    for name, df in frames.items():
        what = f"{kind}[{name}]"
        present = [c for c in cols if df is not None and c in df.columns]
        if not present:
            check_frame(df, what=what, date_col=cols[0])
        for i, col in enumerate(present):
            check_frame(df, what=what, date_col=col, allow_all_null=i > 0)


def _pit_guard(kind: str, payload: dict[str, Any]) -> None:
    """Hook for ``firm.data.pit_store.PointInTimeDataStore`` (every getter, ``load`` and ``load_macro``)."""
    what = f"PointInTimeDataStore.{kind}"
    if kind in ("load", "load_macro"):
        _check_frames(what, payload)
    elif kind == "get_universe_union":
        check_asof(payload["end"], what=what)
        check_asof(payload["start"], what=what)
    elif "asof" in payload:
        check_asof(payload["asof"], what=what)
    else:
        raise _refuse(f"{what}: unrecognised guard call {sorted(payload)}; failing closed")


def _runtime_guard(kind: str, payload: dict[str, Any]) -> None:
    """Hook for ``firm.runtime.load_*``: the cache dir must not be a denied path, loaded frames must be pre-seal."""
    from firm.research import data_access

    what = f"runtime.{kind}"
    settings = payload.get("settings")
    if settings is not None:
        try:
            cache_dir = settings.data.cache_dir
        except AttributeError as exc:
            raise _refuse(f"{what}: settings carry no data.cache_dir; failing closed") from exc
        data_access.assert_not_denied(cache_dir)
    if payload.get("frames"):
        _check_frames(what, payload)


def install_guards() -> None:
    """Install the access guards into ``pit_store`` and ``runtime`` (idempotent).

    Refuses (logs ERROR, raises :class:`HoldoutAccessError`) in a live process: a process-wide
    guard there would make live fundamentals with ``asof`` past the seal raise, and that path
    swallows errors at debug level.
    """
    live = [m for m in _LIVE_MODULES if m in sys.modules]
    if live:
        log.error("refusing to install holdout guards in a live process (%s imported)", ", ".join(live))
        raise HoldoutAccessError(f"install_guards() refused: live modules already imported: {live}")
    from firm import runtime
    from firm.data import pit_store

    if pit_store._ACCESS_GUARD is _pit_guard and runtime._ACCESS_GUARD is _runtime_guard:
        return
    pit_store._ACCESS_GUARD = _pit_guard
    runtime._ACCESS_GUARD = _runtime_guard
    log.info("holdout guards installed (seal_date=%s, max_research_date=%s)", seal_date(), max_research_date())
