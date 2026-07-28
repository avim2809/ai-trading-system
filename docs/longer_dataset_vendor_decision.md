# Longer dataset — vendor decision (draft)

**Status:** awaiting operator purchase decision (updated 2026-07-29). Engineering
is ready; see `docs/longer_dataset_options.md` for the full shortlist.

**Operator context (2026-07-29):** subscribed to Sharadar free tier; evaluating
**Bundle 10Y at ~$49/mo** ([subscribe page](https://sharadar.com/subscribe?plan=bundle)).
Alternatives reviewed (EODHD, Norgate, Valuein, Tiingo, FMP, Massive) — Sharadar
still best fit for this Linux/Parquet stack. **10Y tier is sufficient** for current
diagnostic/PBO windows (earliest backtest date **2020-12-01**); only upgrade to
15Y+ if targeting **2010-start** research panels.

**Interim prices (done):** `scripts/backfill_tiingo_prices.py` extended live
25-symbol universe to ~2010–present in `combined/prices` (does **not** fix
survivorship — still need membership + delisted names).

## Recommendation

| Priority | Choice | Rationale |
|----------|--------|-----------|
| **Primary** | **Sharadar Bundle 10Y (~$49/mo)** | Membership + delisted prices + PIT fundamentals; `scripts/etl_sharadar_to_cache.py` already wired |
| **Alternative** | **Norgate Platinum (~$630/yr prepaid)** | Best survivorship quality; Windows export → SCP (not cheaper, more ops) |
| **Budget fallback** | **Tiingo prices (done) + free S&P membership CSV** | $0 ongoing; high validation risk; no PIT fundamentals bundle |

## Alternatives reviewed (2026-07-29) — not better deals for this repo

| Vendor | Cost | Why skip |
|--------|------|----------|
| EODHD EOD | $20/mo | No membership; survivorship DIY |
| EODHD Fundamentals / All-in-One | $60–100/mo | More expensive than Sharadar; weaker PIT than SF1 |
| Valuein Pro | $49/mo | PIT fundamentals only — no bundled prices/membership |
| FMP Premium | $59/mo | Not PIT; no historical S&P membership |
| Massive (Polygon) | $79+/mo | Tick/microstructure; no membership history |
| Norgate Platinum | ~$52/mo (annual) | Same ballpark; Windows-only export |

## ROI framing (operator, not engineering)

- **Low ROI** if goal is “improve current 25-name paper bot unchanged” — live uses
  static `UniverseResolver.from_static`; Tiingo already covers those prices to 2010.
- **Worth $49/mo** if goal is survivorship-clean S&P backtests, PIT fundamentals for
  `multi_factor`, or go/no-go before promoting strategies / real capital.
- Data does not create alpha; first PBO audit **failed** (PBO≈0.69). Honest panel may
  look worse — value is avoiding false confidence.

## Decision matrix

| Criterion | Sharadar 10Y | Norgate | Tiingo + manual membership |
|-----------|--------------|---------|----------------------------|
| Survivorship-free membership | Yes (S&P table since 1957) | Yes (core product) | Partial (manual curation) |
| Headless / Linux CI | Yes (REST/bulk) | No (desktop export) | Yes |
| Fundamentals PIT (`datekey`) | Yes (SF1) | Limited | FMP/cache (already live) |
| Integration effort | Low (`etl_sharadar_to_cache.py`) | Medium (export pipeline) | High |
| Ongoing cost | **~$49/mo** (10Y bundle) | ~$52/mo (annual prepaid) | $0–30/mo |

## Approval checklist (operator)

- [ ] Budget owner signs off on **Sharadar Bundle 10Y** (or 15Y+ if 2010 panels needed)
- [ ] API key / export credentials stored in `.env` (not committed)
- [ ] Target index: **S&P 500** first
- [ ] Target span for **current remediation**: **≥2020-01-01** (diagnostics/PBO); stretch **≥2010** needs 15Y+ tier
- [ ] Run `scripts/etl_sharadar_to_cache.py` + `scripts/import_universe_membership.py` after first export
- [ ] Re-run `docs/portfolio_construction_diagnosis.md` windows + `scripts/run_walk_forward_pbo_audit.py`

## After purchase — engineering tasks

1. Bulk ETL script: vendor → `data/cache/combined/{prices,fundamentals,universe_membership}`
2. Update `data/cache/README.md` with source, as-of date, symbol count
3. Acceptance run per checklist in `docs/longer_dataset_options.md`
4. Close `longer-dataset` todo in remediation plan
