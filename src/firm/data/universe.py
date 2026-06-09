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

    def delisted_between(self, start: datetime, end: datetime) -> list[str]:
        """Symbols removed from the index within ``[start, end]`` (audit helper)."""
        removed = self._df[schemas.COL_REMOVED_DATE]
        mask = removed.between(pd.Timestamp(start), pd.Timestamp(end))
        return sorted(self._df.loc[mask, schemas.COL_SYMBOL].astype(str).unique().tolist())
