"""JSON-persisted sector tags for the S&P 500 dynamic-universe scanner.

Mirrors ``firm.live.dynamic_universe_state``'s fail-soft JSON idiom, but
tracks a different thing: a symbol -> sector lookup for the *whole* S&P 500
candidate pool (not just currently-held dynamic symbols), so
``firm.live.sp500_universe_sync.build_diversified_candidates`` can
sector-balance without re-fetching on every daily sync — the primary
constituent+sector source is refreshed on its own weekly cadence (see
``firm.live.scheduler``), separate from the daily universe-sync job that
just reads this cache.

Schema: ``{symbol: {"sector": str, "source": str, "as_of": "YYYY-MM-DD"}}``.
``source`` is ``"fmp"`` / ``"github"`` / ``"alphavantage"`` / ``"static"``
(seeded from the existing 25-name ``risk.sector_map``) — informational only,
not consumed by any selection logic.

**Unknown-sector candidates are never persisted with a guessed/"unknown"
sector** — see ``firm.data.providers.fmp._normalize_gics_sector``'s
docstring for why an un-normalized/guessed label is a real
``max_sector_pct`` cap-bypass hazard, not a cosmetic gap.

**FMP's ``/stable/sp500-constituent`` endpoint is confirmed premium-gated
(HTTP 402) on this project's current key (verified live 2026-08-30)** — so
in practice every refresh falls through to :func:`fetch_sp500_constituents_
from_github` below, a free, keyless, no-rate-limit fallback. FMP is tried
first anyway (no cost to trying, and it starts working for real the moment
the plan is ever upgraded) rather than removed outright.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from firm.data.providers.fmp import _normalize_gics_sector

log = logging.getLogger(__name__)

# Community-maintained, Wikipedia-derived, MIT-licensed dataset — no API key,
# no rate limit. Verified live 2026-08-30 (HTTP 200, current GICS sector per
# row, last committed 2026-08-20). See https://github.com/datasets/
# s-and-p-500-companies for provenance/license.
_GITHUB_SP500_CSV_URL = (
    "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/"
    "main/data/constituents.csv"
)


def fetch_sp500_constituents_from_github(
    url: str = _GITHUB_SP500_CSV_URL, timeout: float = 10.0,
) -> pd.DataFrame:
    """Free, keyless S&P 500 membership + GICS sector — the fallback source
    used whenever FMP's constituent endpoint is unavailable (confirmed
    premium-gated on this project's current key, see module docstring).

    Columns: ``symbol``, ``sector`` (already normalized via
    :func:`firm.data.providers.fmp._normalize_gics_sector`) — a drop-in
    match for :meth:`FMPProvider.get_universe_constituents_with_sectors`'s
    return shape. Raises ``requests.HTTPError``/``ValueError`` on any
    network or parsing failure so callers apply the exact same
    graceful-skip handling as any other constituent source — never a
    silent empty result mistaken for "no constituents."
    """
    import requests

    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    if "Symbol" not in df.columns or "GICS Sector" not in df.columns:
        raise ValueError(
            f"unexpected columns from {url}: {list(df.columns)} "
            "(expected at least 'Symbol' and 'GICS Sector')"
        )
    rows = [
        {"symbol": str(sym).strip().upper(), "sector": _normalize_gics_sector(raw_sector)}
        for sym, raw_sector in zip(df["Symbol"], df["GICS Sector"])
        if isinstance(sym, str) and sym.strip()
    ]
    if not rows:
        return pd.DataFrame(columns=["symbol", "sector"])
    return pd.DataFrame(rows, columns=["symbol", "sector"])


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
    source: str = "fmp",
) -> dict[str, dict[str, Any]]:
    """Pure merge of a fresh constituents+sector pull onto the persisted cache.

    - *fmp_rows* overlay ``existing`` (the freshest pull is the broadest
      source, regardless of which provider it actually came from — see
      *source*).
    - Rows with a missing/"unknown" normalized sector are dropped, never
      merged in — a candidate with no known sector must stay absent from
      the cache (and therefore excluded from selection), not get a guessed
      bucket.
    - ``seed_map`` (the existing 25-name ``risk.sector_map``) fills any
      symbol still missing after the overlay — so the static universe's own
      names are always present even before any external source is reachable.
    - When *fmp_rows* is empty (every source failed), prior entries are
      preserved untouched — an outage must never erase yesterday's good
      cache.
    - *source*: the provenance tag recorded on each merged row (``"fmp"`` or
      ``"github"`` — see :func:`refresh_sector_cache`, which picks whichever
      source actually produced *fmp_rows* this run). Kept as a parameter
      name for backwards compatibility with existing callers/tests that
      don't care about provenance.
    """
    merged: dict[str, dict[str, Any]] = {sym: dict(v) for sym, v in existing.items()}

    has_rows = fmp_rows is not None and not fmp_rows.empty and "symbol" in fmp_rows.columns
    if has_rows:
        for _, row in fmp_rows.iterrows():
            symbol = row.get("symbol")
            sector = row.get("sector", "unknown")
            if not symbol or not sector or sector == "unknown":
                continue
            merged[symbol] = {"sector": sector, "source": source, "as_of": today}

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

    Tries *fmp_provider* first (no cost to trying, and it starts working
    for real the moment the FMP plan is ever upgraded — see module
    docstring for the confirmed HTTP 402 on the current key), then falls
    back to the free, keyless :func:`fetch_sp500_constituents_from_github`
    if FMP raises or returns nothing. Only if *both* fail does this degrade
    to an empty pull, so ``merge_sector_records`` just preserves whatever
    was already cached plus any still-missing ``seed_map`` entries — never
    a crash, never a wiped cache.

    ``backfill_provider`` (e.g. ``AlphaVantageProvider``, via its
    ``get_sector``) is optional and best-effort: it looks up a *bounded*
    number (``backfill_limit``) of symbols the fetch listed as candidates
    but with no known sector, never the full ~500-name pool — Alpha Vantage
    has no per-index enumeration endpoint of its own, so it can only ever
    resolve names the primary fetch already told us about. In practice this
    should rarely fire now: the GitHub fallback already includes sector for
    every row, unlike a hypothetical FMP response missing just that field.
    """
    existing = load_sector_cache(path)
    source = "fmp"
    try:
        fmp_rows = fmp_provider.get_universe_constituents_with_sectors("sp500")
        if fmp_rows is None or fmp_rows.empty:
            raise ValueError("FMP returned no rows")
    except Exception:
        log.info(
            "sp500_sector_cache: FMP constituents+sectors unavailable "
            "(missing key, premium-gated endpoint, or transient outage) — "
            "falling back to the free GitHub S&P 500 dataset",
            exc_info=True,
        )
        source = "github"
        try:
            fmp_rows = fetch_sp500_constituents_from_github()
        except Exception:
            log.warning(
                "sp500_sector_cache: GitHub fallback also failed — merging "
                "seed_map only, preserving prior cache entries",
                exc_info=True,
            )
            fmp_rows = pd.DataFrame(columns=["symbol", "sector"])

    merged = merge_sector_records(existing, fmp_rows, seed_map, today, source=source)

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
