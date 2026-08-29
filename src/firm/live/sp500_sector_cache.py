"""JSON-persisted sector tags for FMP's S&P 500 dynamic-universe scanner.

Mirrors ``firm.live.dynamic_universe_state``'s fail-soft JSON idiom, but
tracks a different thing: a symbol -> sector lookup for the *whole* S&P 500
candidate pool (not just currently-held dynamic symbols), so
``firm.live.sp500_universe_sync.build_diversified_candidates`` can
sector-balance without re-fetching FMP on every daily sync — the FMP
constituent+sector call is comparatively expensive/rate-sensitive and is
refreshed on its own weekly cadence (see ``firm.live.scheduler``), separate
from the daily universe-sync job that just reads this cache.

Schema: ``{symbol: {"sector": str, "source": str, "as_of": "YYYY-MM-DD"}}``.
``source`` is ``"fmp"`` / ``"alphavantage"`` / ``"static"`` (seeded from the
existing 25-name ``risk.sector_map``) — informational only, not consumed by
any selection logic.

**Unknown-sector candidates are never persisted with a guessed/"unknown"
sector** — see ``firm.data.providers.fmp._normalize_gics_sector``'s
docstring for why an un-normalized/guessed label is a real
``max_sector_pct`` cap-bypass hazard, not a cosmetic gap.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)


def load_sector_cache(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the persisted sector cache, or an empty dict if absent/corrupt."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except Exception:
        log.warning(
            "Failed to read sp500 sector cache from %s — starting empty "
            "(the next refresh_sector_cache run will rebuild it)", p, exc_info=True,
        )
        return {}
    if not isinstance(data, dict):
        log.warning("sp500 sector cache at %s is not a dict — ignoring", p)
        return {}
    return data


def save_sector_cache(path: str | Path, cache: dict[str, dict[str, Any]]) -> None:
    """Persist the sector cache, creating parent dirs as needed."""
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache, indent=2, sort_keys=True))
    except Exception:
        log.warning("Failed to persist sp500 sector cache to %s", p, exc_info=True)


def merge_sector_records(
    existing: dict[str, dict[str, Any]],
    fmp_rows: pd.DataFrame,
    seed_map: dict[str, str] | None,
    today: str,
) -> dict[str, dict[str, Any]]:
    """Pure merge of a fresh FMP pull onto the persisted cache.

    - FMP rows overlay ``existing`` (FMP is the freshest, broadest source).
    - Rows with a missing/"unknown" normalized sector are dropped, never
      merged in — a candidate with no known sector must stay absent from
      the cache (and therefore excluded from selection), not get a guessed
      bucket.
    - ``seed_map`` (the existing 25-name ``risk.sector_map``) fills any
      symbol still missing after the FMP overlay — so the static universe's
      own names are always present even before FMP is ever reachable.
    - When ``fmp_rows`` is empty (an outage or premium-gated response),
      prior entries are preserved untouched — an FMP outage must never
      erase yesterday's good cache.
    """
    merged: dict[str, dict[str, Any]] = {sym: dict(v) for sym, v in existing.items()}

    has_rows = fmp_rows is not None and not fmp_rows.empty and "symbol" in fmp_rows.columns
    if has_rows:
        for _, row in fmp_rows.iterrows():
            symbol = row.get("symbol")
            sector = row.get("sector", "unknown")
            if not symbol or not sector or sector == "unknown":
                continue
            merged[symbol] = {"sector": sector, "source": "fmp", "as_of": today}

    for symbol, sector in (seed_map or {}).items():
        if symbol not in merged and sector and sector != "unknown":
            merged[symbol] = {"sector": sector, "source": "static", "as_of": today}

    return merged


def refresh_sector_cache(
    path: str | Path,
    *,
    fmp_provider: Any,
    seed_map: dict[str, str] | None = None,
    backfill_provider: Any | None = None,
    backfill_limit: int = 0,
    today: str,
) -> dict[str, dict[str, Any]]:
    """Fetch, merge, optionally backfill, and persist the sector cache.

    Fetch failures (missing key, transient outage, or a premium-gated
    endpoint — all indistinguishable from here, and all handled the same
    way) degrade to an empty FMP pull, so ``merge_sector_records`` just
    preserves whatever was already cached plus any still-missing
    ``seed_map`` entries — never a crash, never a wiped cache.

    ``backfill_provider`` (e.g. ``AlphaVantageProvider``, via its
    ``get_sector``) is optional and best-effort: it looks up a *bounded*
    number (``backfill_limit``) of symbols FMP listed as candidates but
    with no known sector, never the full ~500-name pool — Alpha Vantage
    has no per-index enumeration endpoint of its own, so it can only ever
    resolve names FMP already told us about.
    """
    existing = load_sector_cache(path)
    try:
        fmp_rows = fmp_provider.get_universe_constituents_with_sectors("sp500")
    except Exception:
        log.warning(
            "sp500_sector_cache: failed to fetch FMP constituents+sectors "
            "(missing key, premium-gated endpoint, or transient outage) — "
            "merging seed_map only, preserving prior cache entries",
            exc_info=True,
        )
        fmp_rows = pd.DataFrame(columns=["symbol", "sector"])

    merged = merge_sector_records(existing, fmp_rows, seed_map, today)

    if backfill_provider is not None and backfill_limit > 0 and not fmp_rows.empty:
        unknown_symbols = [
            row["symbol"]
            for _, row in fmp_rows.iterrows()
            if row.get("symbol")
            and row.get("symbol") not in merged
            and (not row.get("sector") or row.get("sector") == "unknown")
        ]
        for symbol in unknown_symbols[:backfill_limit]:
            try:
                sector = backfill_provider.get_sector(symbol)
            except Exception:
                log.warning(
                    "sp500_sector_cache: Alpha Vantage backfill failed for %s",
                    symbol, exc_info=True,
                )
                continue
            if sector and sector != "unknown":
                merged[symbol] = {"sector": sector, "source": "alphavantage", "as_of": today}

    save_sector_cache(path, merged)
    log.info(
        "sp500_sector_cache: refreshed (%d symbols known, %d from this FMP pull)",
        len(merged), len(fmp_rows),
    )
    return merged
