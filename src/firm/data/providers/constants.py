"""Shared constants for data provider adapters."""

from __future__ import annotations

# Index ETFs — no stock-style quarterly fundamentals panel.
ETF_SYMBOLS: frozenset[str] = frozenset({"SPY", "QQQ", "IWM"})
