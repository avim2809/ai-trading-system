"""Fail-closed file-level data access for research code (ticket P0-02).

All new research harnesses read data through this module. It is an ALLOW-LIST first:
a path must resolve (symlinks followed) under one of ``allow_roots`` in
``config/research_freeze.yaml``, and only then is it checked against ``deny_paths``. Everything
else is refused, including the IBKR instance's ``data/`` state (``data/live_state.db``,
``data/logs``, ``data/cycle_history.json`` ...), which holds post-seal prices, NAV and fills.
No environment variable, heuristic or "research mode" check exists anywhere in this module.
"""

from __future__ import annotations

import datetime as dt
import fnmatch
import logging
import os
import re
from pathlib import Path

import pandas as pd

from firm.research import seal
from firm.research.seal import HoldoutAccessError, _refuse  # noqa: F401 - re-exported

log = logging.getLogger(__name__)

_KIND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_GLOB_CHARS = set("*?[")


def _resolve_entry(entry: str) -> Path:
    p = Path(entry).expanduser()
    if not p.is_absolute():
        p = seal.config_root() / p
    return p.resolve()


def _resolve_path(path: str | os.PathLike[str]) -> Path:
    return _resolve_entry(os.fspath(path))


def _is_under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _is_denied(path: Path, pattern: str) -> bool:
    resolved = _resolve_entry(pattern)
    if not _GLOB_CHARS & set(pattern):
        return _is_under(path, resolved)
    pat = str(resolved)
    return any(fnmatch.fnmatchcase(str(candidate), pat) for candidate in (path, *path.parents))


def assert_not_denied(path: str | os.PathLike[str]) -> Path:
    """Deny-list check only (used where the allow-list does not apply, e.g. a runtime cache dir)."""
    resolved = _resolve_path(path)
    _, deny = seal.config_lists()
    for pattern in deny:
        if _is_denied(resolved, pattern):
            raise _refuse(f"{path} is under deny_paths entry {pattern!r}")
    return resolved


def assert_path_allowed(path: str | os.PathLike[str]) -> Path:
    """Allow-list, then deny-list. Returns the resolved path; raises :class:`HoldoutAccessError`."""
    resolved = _resolve_path(path)
    allow, deny = seal.config_lists()
    if not any(_is_under(resolved, _resolve_entry(root)) for root in allow):
        raise _refuse(f"{path} is outside allow_roots {allow}")
    for pattern in deny:
        if _is_denied(resolved, pattern):
            raise _refuse(f"{path} is under deny_paths entry {pattern!r}")
    return resolved


def _drop_after(df: pd.DataFrame, asof: dt.date, what: str) -> pd.DataFrame:
    if len(df) == 0:
        return df
    values = seal.date_values(df)
    if values is None:
        raise _refuse(f"{what}: file has no 'date' column or DatetimeIndex; cannot prove it is pre-seal")
    keep = (values.dt.normalize() <= pd.Timestamp(asof)).to_numpy()
    if not keep.all():
        log.info("%s: dropped %d rows dated after %s", what, int((~keep).sum()), asof)
    return df.loc[keep]


def read_parquet(path: str | os.PathLike[str], *, asof: dt.date | dt.datetime | pd.Timestamp) -> pd.DataFrame:
    """The single file-level entry for research reads.

    ``assert_path_allowed`` -> ``check_asof`` -> ``pd.read_parquet`` -> rows after ``asof`` dropped ->
    ``check_frame``. The path and ``asof`` are checked before the file is opened.
    """
    seal.install_guards()
    resolved = assert_path_allowed(path)
    seal.check_asof(asof, what=str(path))
    asof_day = pd.Timestamp(asof).date()
    df = pd.read_parquet(resolved)
    df = _drop_after(df, asof_day, str(path))
    seal.check_frame(df, what=str(path))
    log.info("data_access.read_parquet %s: %d rows (asof %s)", resolved, len(df), asof_day)
    return df


def load_panel(kind: str, *, start: str, end: str, symbols: list[str] | None = None) -> pd.DataFrame:
    """Load a panel stored as ``<allow_root>/<kind>/*.parquet`` for ``start <= date <= end``.

    ``end`` must be before ``seal_date``. ``kind`` is a single directory name found directly under an
    allow root; an unknown kind raises. With ``symbols``, per-symbol files (stem == symbol) are read
    when present, and a ``symbol`` column is filtered afterwards. Dedicated typed loaders (ETF prices,
    manifests) are built on :func:`read_parquet` by P2-02.
    """
    seal.install_guards()
    end_day = pd.Timestamp(end).date()
    start_day = pd.Timestamp(start).date()
    if end_day >= seal.seal_date():
        raise _refuse(f"load_panel({kind!r}): end {end_day} >= seal_date {seal.seal_date()}")
    if start_day > end_day:
        raise ValueError(f"load_panel: start {start_day} is after end {end_day}")
    if not isinstance(kind, str) or not _KIND_RE.match(kind):
        raise _refuse(f"load_panel: invalid panel kind {kind!r}")
    allow, _ = seal.config_lists()
    directory = next(
        (d for d in (_resolve_entry(root) / kind for root in allow) if d.is_dir()),
        None,
    )
    if directory is None:
        raise _refuse(f"load_panel: unknown panel kind {kind!r} (no such directory under allow_roots)")
    assert_path_allowed(directory)
    files = sorted(directory.glob("*.parquet"))
    if symbols:
        wanted = set(symbols)
        per_symbol = [f for f in files if f.stem in wanted]
        files = per_symbol or files
    frames = [read_parquet(f, asof=end_day) for f in files]
    frames = [f for f in frames if len(f)]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    if symbols and "symbol" in out.columns:
        out = out[out["symbol"].isin(symbols)]
    values = seal.date_values(out)
    if values is not None:
        out = out.loc[(values.dt.normalize() >= pd.Timestamp(start_day)).to_numpy()]
    out = out.reset_index(drop=True)
    seal.check_frame(out, what=f"load_panel({kind!r})")
    log.info("data_access.load_panel %s: %d rows, %d files, %s..%s", kind, len(out), len(files), start_day, end_day)
    return out
