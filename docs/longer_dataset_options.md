# Longer, delisting-inclusive historical dataset — vendor scoping

Remediation plan item: **`longer-dataset`**. The current cache (~29 symbols,
2020–2026) is too short and too survivorship-clean to validate 50+ tunable
parameters across 12 concurrent strategies. This doc scopes what to buy/build
and how it plugs into the existing PIT stack.

**Status (2026-07-27):** engineering is ready (`UniverseResolver`,
`load_universe_membership`, `fetch_data.py`); **data acquisition is not**.

---

## Minimum requirements

| Requirement | Why |
|-------------|-----|
| **10+ years** daily OHLCV | Cover multiple regimes (2020 crash, 2022 bear, 2021 bull) |
| **Delistings + index changes** | `UniverseResolver` needs `added_date` / `removed_date` windows — without them we fall back to static lists (survivorship bias) |
| **Point-in-time fundamentals** | Real filing dates already supported (`EdgarProvider`, `FMPProvider`); vendor must expose `filed` / `fillingDate` or equivalent |
| **Corporate actions** | Splits/dividends for adjusted prices; current providers vary |
| **Reasonable US equity breadth** | S&P 500 membership history minimum; Russell 1000/3000 preferred for stat_arb breadth |

---

## Vendor shortlist (ranked for this codebase)

### Tier 1 — best fit (membership + prices + fundamentals)

| Vendor | Strengths | Gaps / cost | Integration path |
|--------|-----------|-------------|-------------------|
| **Sharadar (via Nasdaq Data Link / Quandl)** | Long US equity history, fundamentals, delistings table | Paid; separate tables for prices vs fundamentals | New provider or batch ETL → `data/cache/combined/{prices,fundamentals,universe_membership}` |
| **Norgate Data** | Purpose-built for survivorship-free backtests; index membership | Windows/desktop API; not REST-native | Nightly export script → Parquet cache (same schema as `UNIVERSE_COLUMNS`) |
| **Compustat / CRSP (academic)** | Gold standard survivorship research | Expensive; licensing | Batch export only — overkill unless institutional budget |

### Tier 2 — partial (good prices, weak membership history)

| Vendor | Strengths | Gaps | Notes |
|--------|-----------|------|-------|
| **Polygon.io / Massive** | Already integrated (`MassiveProvider`); solid daily bars | Index membership history limited; delistings need separate product tier | Extend `fetch_data.py` for wider symbol lists; **still need membership CSV/Parquet from elsewhere** |
| **Tiingo** | Already integrated; clean daily data | No historical S&P membership windows | Prices only unless paired with membership source |
| **FMP** | Fundamentals + current constituents endpoint | `get_universe_constituents` is **current snapshot only**, not historical windows | Keep for fundamentals refresh; not sufficient alone for `longer-dataset` |

### Tier 3 — not recommended as primary

| Vendor | Issue |
|--------|-------|
| **Yahoo / free APIs** | Delisting gaps, corporate-action errors, ToS risk for production research |
| **Alpha Vantage / Finnhub** | No historical index membership; rate limits on bulk history |

---

## Recommended acquisition path

1. **Buy membership history** (Norgate or Sharadar delistings/membership table) — this unblocks the engineering already shipped in `pit-universe-membership`.
2. **Bulk-download daily prices** for the union of all symbols ever in the target index(es) over 10y — use `scripts/etl_sharadar_to_cache.py` (Sharadar SEP), `scripts/backfill_tiingo_prices.py` (free Tiingo tier, live 25-symbol universe), or existing `scripts/fetch_data.py` with Tiingo/Massive keys.
3. **Write `data/cache/combined/universe_membership.parquet`** conforming to `firm.data.schemas.UNIVERSE_COLUMNS`:
   - `index`, `symbol`, `added_date`, `removed_date` (`NaT` = still active)
4. **Re-run portfolio-construction diagnosis** (`docs/portfolio_construction_diagnosis.md` windows) on the longer panel before trusting any new strategy weights.
5. **Document source + as-of date** in `data/cache/README.md` (create when data lands).

---

## Acceptance checklist (close `longer-dataset` todo)

- [ ] `combined/universe_membership` Parquet covers ≥10y with real removal dates for delisted names
- [ ] Price cache covers the same symbol union (no silent drops)
- [ ] At least one backtest walk-forward completes on the new panel without falling back to `from_static`
- [ ] `docs/portfolio_construction_diagnosis.md` re-run on ≥3 non-overlapping 18mo windows

---

## Budget / decision needed

See **`docs/longer_dataset_vendor_decision.md`** for the recommended pick
(Sharadar primary, Norgate alternative) and operator approval checklist.

No vendor is selected yet — pick based on:

- **Lowest integration friction on Linux headless:** Sharadar batch → Parquet
- **Best survivorship research quality:** Norgate export
- **Lowest cash cost (more engineering):** CRSP academic license if available, else manual SEC + Tiingo prices + hand-curated membership CSV for S&P 500 only
