"""Survivorship-aware universe resolution.

Backtests that use *today's* index membership silently exclude companies that
were later delisted or removed -- a classic survivorship bias that inflates
returns. :class:`UniverseResolver` instead resolves the tradable set **as of a
date** from membership windows (``added_date`` / ``removed_date``), so delisted
names remain investable up to their removal.

Membership frame schema: :data:`firm.data.schemas.UNIVERSE_COLUMNS`
(``index``, ``symbol``, ``added_date``, ``removed_date``); an open membership has
``removed_date`` = ``NaT``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import pandas as pd

from firm.data import schemas
from firm.logging_setup import get_logger

log = get_logger(__name__)


class UniverseResolver:
    """Resolve point-in-time index membership without survivorship bias."""

    def __init__(self, constituents: pd.DataFrame) -> None:
        """Args:
        constituents: Membership windows conforming to
            :data:`firm.data.schemas.UNIVERSE_COLUMNS`.
        """
        missing = {schemas.COL_SYMBOL} - set(constituents.columns)
        if missing:
            raise ValueError(f"Constituents frame missing columns: {missing}")
        df = constituents.copy()
        for col in (schemas.COL_ADDED_DATE, schemas.COL_REMOVED_DATE):
            if col not in df.columns:
                df[col] = pd.NaT
            df[col] = pd.to_datetime(df[col], errors="coerce")
        if schemas.COL_INDEX not in df.columns:
            df[schemas.COL_INDEX] = "default"
        self._df = df

    @classmethod
    def from_static(cls, symbols: Sequence[str], index: str = "static") -> "UniverseResolver":
        """Build a resolver from a fixed symbol list (always-active membership)."""
        df = pd.DataFrame(
            {
                schemas.COL_INDEX: index,
                schemas.COL_SYMBOL: list(symbols),
                schemas.COL_ADDED_DATE: pd.NaT,
                schemas.COL_REMOVED_DATE: pd.NaT,
            }
        )
        return cls(df)

    def symbols_asof(self, asof: datetime, index: str | None = None) -> list[str]:
        """Return symbols that are index members on ``asof``.

        A symbol is a member when ``added_date <= asof`` (or unknown) and
        ``removed_date > asof`` (or still open). Delisted/removed names are thus
        correctly included for dates before their removal.

        Args:
            asof: As-of date.
            index: Optional index filter (e.g. ``"sp500"``).

        Returns:
            Sorted, de-duplicated list of member symbols.
        """
        asof_ts = pd.Timestamp(asof)
        df = self._df
        if index is not None:
            df = df[df[schemas.COL_INDEX] == index]
        added = df[schemas.COL_ADDED_DATE]
        removed = df[schemas.COL_REMOVED_DATE]
        mask = (added.isna() | (added <= asof_ts)) & (removed.isna() | (removed > asof_ts))
        members = df.loc[mask, schemas.COL_SYMBOL].astype(str).unique().tolist()
        return sorted(members)

    def __call__(self, asof: datetime) -> list[str]:
        """Alias for :meth:`symbols_asof` so the resolver is a drop-in callable.

        This signature matches the ``universe_resolver`` hook on
        :class:`firm.data.pit_store.PointInTimeDataStore`.
        """
        return self.symbols_asof(asof)

    def symbols_between(
        self, start: datetime, end: datetime, index: str | None = None
    ) -> list[str]:
        """Union of every symbol that was a member at *any point* within
        ``[start, end]`` — the superset a backtest needs to load data feeds
        for, since a name that joins mid-window (e.g. an IPO/index addition
        after ``start``) still needs its feed loaded even though
        :meth:`symbols_asof` at ``start`` alone wouldn't include it yet.

        A membership window ``[added_date, removed_date)`` overlaps
        ``[start, end]`` when ``added_date <= end`` (or unknown) and
        ``removed_date > start`` (or still open) — the same logic as
        :meth:`symbols_asof` but tested against the whole window instead of
        a single instant.
        """
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        df = self._df
        if index is not None:
            df = df[df[schemas.COL_INDEX] == index]
        added = df[schemas.COL_ADDED_DATE]
        removed = df[schemas.COL_REMOVED_DATE]
        mask = (added.isna() | (added <= end_ts)) & (removed.isna() | (removed > start_ts))
        members = df.loc[mask, schemas.COL_SYMBOL].astype(str).unique().tolist()
        return sorted(members)

    def delisted_between(self, start: datetime, end: datetime) -> list[str]:
        """Symbols removed from the index within ``[start, end]`` (audit helper)."""
        removed = self._df[schemas.COL_REMOVED_DATE]
        mask = removed.between(pd.Timestamp(start), pd.Timestamp(end))
        return sorted(self._df.loc[mask, schemas.COL_SYMBOL].astype(str).unique().tolist())


def build_resolver(
    membership: pd.DataFrame | None,
    fallback_symbols: Sequence[str],
) -> "UniverseResolver":
    """Build a resolver from real membership data, degrading to a static list.

    When *membership* is a non-empty frame conforming to
    :data:`firm.data.schemas.UNIVERSE_COLUMNS` it is used directly and delisted
    names remain tradable up to their removal date — the actual survivorship-bias
    fix. Without real membership data this degrades to
    :meth:`UniverseResolver.from_static`, which treats every symbol in
    *fallback_symbols* as always-active for the whole backtest window; that mode
    keeps the resolver hook consistently wired (so real data can be dropped in
    later without further code changes) but does **not** by itself eliminate
    survivorship bias — see the ``pit-universe-membership`` follow-up, which also
    needs engine-level support for symbols entering/leaving the tradable set
    mid-backtest.
    """
    if membership is not None and not membership.empty:
        log.info(
            "Universe resolver: using real membership data (%d rows, %d symbols)",
            len(membership), membership[schemas.COL_SYMBOL].nunique(),
        )
        return UniverseResolver(membership)
    log.warning(
        "Universe resolver: no historical index-membership data found — falling "
        "back to a static, always-active symbol list (%d symbols). Survivorship "
        "bias is NOT corrected in this mode; see the pit-universe-membership "
        "follow-up task.",
        len(fallback_symbols),
    )
    return UniverseResolver.from_static(fallback_symbols)
