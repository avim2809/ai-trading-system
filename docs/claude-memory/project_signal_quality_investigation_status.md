---
name: project-signal-quality-investigation-status
description: "Consolidated status of the 'is the strategy edge real' investigation: 6/6 failed combination-layer gates, current data coverage, Sharadar declined, FMP scanner + capital-gate endpoint shipped 2026-08-30 — scanner works on the free tier via a GitHub S&P 500 fallback (FMP itself is premium-gated)"
metadata: 
  node_type: memory
  type: project
  originSessionId: ff1d5776-266d-4d49-812e-1d60eb6fc60c
  modified: 2026-08-30T08:49:27.314Z
---

The repo's own docs (`docs/remediation_progress.md`, `docs/formal_pbo_audit.md`,
`docs/portfolio_construction_diagnosis.md`, `docs/longer_dataset_vendor_decision.md`)
contain a much fuller research history on this than auto-memory had captured
before 2026-08-30 — read those directly for detail; this is the summary plus
the decisions made since.

**The core finding (unchanged across ~3 sessions, 2026-07-27 through 08-26):**
the combination/allocation layer is not the bottleneck. `docs/portfolio_
construction_diagnosis.md` Finding 2: the blended portfolio has never beaten
its own best single strategy, 6/6 diagnostic runs. Six independent attempts
to fix this at the combination layer have all failed the walk-forward+PBO
gate (PBO<0.5 *and* DSR>0.95 required, none cleared it): `zscore_demean`,
strategy concentration, the full `joint_optimizer` QP redesign (2026-08-23,
[[project_joint_optimizer_redesign_aug23]]), then in a further PART 3 session
(2026-08-24/25, not originally in auto-memory): dropping 2 stat_arb ETF
pairs, a seasonality→RiskAgent exposure overlay, and a macro (yield-curve)
exposure overlay. A real bug was found and reverted in that same PART 3 pass
(overlays stacking with the live `regime_overlay` could zero every target
weight below the no-trade band and lock the book at 0% invested — see the
commit `e23ee9d`).

**Likely actual bottleneck: data, not mechanism.** Checked live 2026-08-30
directly against `data/cache`:

| Panel | Range | Symbols |
|---|---|---|
| prices | 2010-01-04 → 2026-07-30 | 29 (Tiingo backfill, already done, free) |
| fundamentals | 2020-07-30 → 2026-08-14 | 22 of 25 |
| sentiment | **2025-11-19 → 2026-08-28 (~9 months)** | 25 |
| universe_membership (point-in-time) | **missing entirely** | — |

The existing walk-forward+PBO audit window is 2020-01-01→2026-06-30 — so
fold 1 (train 2020-01→2021-02) runs fundamentals-starved and effectively
9-strategy (sentiment contributes nothing that early), and every audit's
universe is `UniverseResolver.from_static` (today's 25 names, no
survivorship-bias correction — no historical membership data exists at any
length to fix this).

**Decision 2026-08-30: no new paid data.** Sharadar Bundle 10Y (~$49/mo) was
the standing recommendation in `docs/longer_dataset_vendor_decision.md`
("awaiting operator purchase decision") — the ETL is fully built
(`scripts/etl_sharadar_to_cache.py`, tested, needs no API key, just a manual
CSV export from a Sharadar/Nasdaq-Data-Link subscription) but **user declined
after seeing the actual coverage table above — proceeding with the data as
it stands.** Don't re-raise the Sharadar purchase unprompted; the
survivorship-bias / short-sentiment-history caveat above should just travel
with any PBO/attribution number quoted from here on.

**SHIPPED 2026-08-30 (commit `4e21897`, pushed to `origin/main`): both agreed
next steps, implemented and tested (1592 tests pass, full suite).** External
research this session (Grinold-Kahn IR=IC×√Breadth, HRP/Black-Litterman/
shrinkage being estimation-error tools not breadth generators, a published
near-identical case study) validated that breadth expansion — not more
combination-layer engineering — is the theoretically-justified next lever,
which reprioritized this from "deferred" to "build now."

1. **Sector-balanced FMP dynamic universe** — reuses
   `compute_universe_update()` in `danelfin_universe_sync.py` (Danelfin
   itself is dead, see [[project_danelfin_integration]]) unchanged, fed by a
   new `src/firm/live/sp500_universe_sync.py` + `sp500_sector_cache.py`.
   Selection is sector water-fill (front-load sectors thin/absent in the
   static 25 — materials/industrials/utilities/staples/real-estate) then
   highest trailing dollar-volume within a sector — deliberately liquidity,
   not this system's own alpha signals, to avoid a look-ahead/selection-bias
   trap. New `config/live.yaml` block `sp500_dynamic_universe`, `enabled:
   false` by default.
   - `FMPProvider.get_universe_constituents_with_sectors` confirmed live
     2026-08-30 that FMP's `/stable/sp500-constituent` returns **HTTP 402**
     ("Restricted Endpoint... upgrade your plan") on the current key — not
     just a missing sector field, the *entire* endpoint.
   - **Fixed same-day, still free**: added
     `firm.live.sp500_sector_cache.fetch_sp500_constituents_from_github` — a
     keyless, no-rate-limit, community-maintained (Wikipedia-derived,
     MIT-licensed) S&P 500 + GICS-sector CSV
     (`github.com/datasets/s-and-p-500-companies`, verified live 2026-08-30:
     503 rows, 0 unknown sectors, last committed 2026-08-20). Both
     `sync_once` and `refresh_sector_cache` now try FMP first (costs
     nothing, starts working for real if the plan is ever upgraded), fall
     back to this GitHub source on any FMP failure — **the scanner is fully
     functional on the free tier now**, confirmed via a real end-to-end run
     (`refresh_sector_cache` against the live network → 503 symbols cached,
     source=github for all). AlphaVantage's OVERVIEW `Sector` field also
     works live but has no index-enumeration endpoint, so it remains only a
     bounded per-symbol backfill, rarely needed now.
   - New file `data/sp500_sector_map.json` (gitignored, added to
     `.gitignore` alongside the other live-state files) is what this
     produces on a real run — don't be surprised to find it present.
   - Still needs, before it does anything real even if FMP is upgraded: the
     forward paper A/B (one engine only, Alpaca) — no backtest validation is
     possible for this, by design.
2. **Real-capital allocation gate automation** — new `GET
   /api/live/capital-gate` (`src/firm/live/capital_gate.py`, pure/testable),
   automates the manual checklist in `docs/PROJECT_CONTEXT.md` (~L670-694):
   duration, trade count, bootstrap-Sharpe-CI (new
   `MonteCarloAnalyzer.sharpe_confidence_interval` in `eval/robustness.py`),
   max drawdown, kill-switch trips. **Kill-switch-trips criterion is
   `durable: false`** — trip history isn't persisted across restarts today
   (only current-state blob is), so it's best-effort from in-session alerts
   only; durable trip-history persistence is an explicitly deferred
   follow-up, not done. This endpoint is live now on both engines (no
   `enabled` flag — pure read-only observability, no live-trading behavior
   change).

Neither change altered any running engine's live behavior at ship time
(scanner disabled by default; gate endpoint is additive/read-only) — both
engines kept running through this, untouched. A pre-existing GitHub
Dependabot alert (5, 1 high/4 moderate — first seen after the
`joint_optimizer` push, see [[project_joint_optimizer_redesign_aug23]])
reappeared on this push too; still not investigated/fixed, still flag
rather than silently upgrade/pin.

**Enabled live 2026-08-30, Alpaca only** (per the forward-A/B design —
IBKR/`config/live.yaml` stays on the static 25-name universe as the
control). Before flipping `enabled: true`, ran a real dry-run simulation
(`sync_once` against a stub engine, but real `AlpacaProvider` prices + a
real GitHub-sourced sector cache, no engine mutation) per the config
comment's own documented discipline — and it caught a real bug:
`AlpacaProvider.get_prices` raised `"subscription does not permit querying
recent SIP data"` (a 403) whenever the request's end date is close to now —
free/paper Alpaca accounts have no SIP subscription. This is a
previously-latent bug (nothing had called `get_prices` with `end=utcnow()`
before this scanner's liquidity check), not something this session
introduced — **fixed at the source** by passing `feed=DataFeed.IEX`
explicitly in `AlpacaProvider.get_prices`'s `StockBarsRequest` (verified
live against the real API before and after the fix). This fix benefits
every caller of `AlpacaProvider.get_prices`, not just this scanner.

Post-fix dry-run picks (real liquidity ranking, real S&P 500 data) filled
exactly the 5 sectors with zero static representation
(materials/industrials/utilities/real_estate/consumer_staples), 2 large,
genuinely liquid names each (e.g. WMT/KO for staples, CAT/GEV for
industrials, FCX/LIN for materials, EQIX/WELL for real estate, NEE/CEG for
utilities) — exactly the intended water-fill behavior.

**LIVE as of 2026-08-30 (commit `c92bbfa`, pushed):** `config/live_alpaca.yaml`
has `sp500_dynamic_universe.enabled: true` (`data_alpaca/`-scoped state/cache
paths); `ai-trading-alpaca.service` restarted to pick it up, confirmed
healthy post-restart (broker connected, no alerts, no errors) and the
scheduler startup log explicitly shows `sp500_universe_sync=07:00
(sector_cache_refresh=sun)` plus the `_refresh_sp500_sector_cache_safe` job
registered. `config/live.yaml` (IBKR) is untouched — still the static
25-name control arm. The sector cache (`data_alpaca/sp500_sector_map.json`)
is already warm (503 symbols, populated during the pre-enable dry run), so
Monday's job doesn't start cold.

**First real scheduled run: Monday 2026-08-31, 07:00 US/Eastern** (mon-fri
only — today, 8/30, is a weekend, so nothing fires until then; don't assume
it already ran just because the config went live on a Sunday). Check
`/api/live/status` and service logs on Alpaca (port 8001) for
`sp500_universe_sync` results after that time to see the first real
(non-dry-run) additions/removals.

Related: [[project_joint_optimizer_redesign_aug23]],
[[project_live_pause_not_persistent]], [[project_danelfin_integration]].
