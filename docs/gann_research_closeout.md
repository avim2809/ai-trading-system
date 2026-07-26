# Gann Strategy Research — Closeout Memo

**Status:** Permanently retired (July 2026)  
**Verdict:** No detectable systematic timing value on equities, commodities, FX, or crypto.

Experiment scripts and run artefacts were removed from `main` after closeout. This document is the sole record on the default branch.

**Archived code + results** (for line-level audit): branch `research/gann-archive` — see `research/gann-archive/README.md` on that branch for raw GitHub URLs.

---

## Summary

The Gann composite (`src/firm/strategies/gann.py`) remains in the repository for reference and ablation, but is **disabled in live trading** and will not be re-enabled. After a multi-phase research programme — correcting the original implementation, walk-forward IC tests, pipeline Sharpe analysis, and permutation-controlled event studies on two universes — the conclusion is unambiguous: **Gann's geometric cycle mechanisms do not predict swing timing better than chance.**

---

## Study arc

| Phase | Finding |
|-------|---------|
| IC ablation (25 US large-caps) | Best variant `angles_swing` IC +0.059; `cycles_only` negative |
| Pipeline Sharpe (holdout 2024-07 → 2026-06) | +Gann ΔSharpe −0.026 vs 10-strategy baseline |
| Legacy cycles event study | NO TIMING VALUE (convergence ≈ baseline, lift ~0.98) |
| Correct implementation (equities, calendar-day) | Exp1 lift 0.95, Exp2 lift 1.01, permutation p = 1.000 |
| Multi-asset expansion (weekly bars) | 0/5 assets pass; Exp2 convergence lifts **below 1.0** on all assets |

---

## Methodology (final, correct implementation)

The original `_time_cycles` component tested the wrong mechanism (fixed universal cycles, trading-bar conversion, dense pivots, fade-last-move direction). The corrected framework used:

1. **Sparse major pivots** — `argrelextrema` with order 8–15 and minimum swing filter (target 4–8 pivots/year).
2. **Calendar-day or weekly-bar arithmetic** — no `252/365` bar conversion for projections.
3. **Natural cycles** — Gann-documented counts (days or weeks from pivot).
4. **Price-derived squaring** — `T = sqrt(P / scale) × multiplier` with per-asset scale calibration on a training window.
5. **Confirmation gate** — volume spike, range expansion, or strong close within the projection window.
6. **Permutation tests** — 500 shuffles of projection offsets with pivot positions held fixed.

Pass criteria were pre-registered: lift > 1.15 (Exp1) or convergence lift > 1.20 (Exp2) with p < 0.10, on ≥3/5 multi-asset names.

---

## Key results (multi-asset, weekly bars, holdout 2020–2026)

| Asset | Exp1 lift (gate) | Exp2 convergence lift (gate) | Perm p (Exp2) |
|-------|------------------|------------------------------|---------------|
| Gold | 1.02 | **0.16** | 1.000 |
| S&P 500 | 0.88 | **0.53** | 0.976 |
| Oil | 1.00 | **0.40** | 1.000 |
| EUR/USD | 0.99 | **0.33** | 1.000 |
| Bitcoin | 0.87 | **0.58** | 0.996 |

Bitcoin 208-week halving cycle lift: **0.42**. Confirmation gate improved lift on only 3/5 assets in Exp1 and **worsened** it on 4/5 in Exp2.

Experiment 2 lifts below 1.0 mean sqrt(P) projections land *less often* near confirmed swings than random dates.

---

## Interpretation

- **Not an implementation bug.** Failure reproduced after fixing calendar arithmetic, sparse pivots, price-derived cycles, and confirmation gating.
- **Not an asset-class effect.** Commodities, equity index, FX, and crypto all failed.
- **Practitioner edge is discretionary.** The automated confirmation gate did not increase precision; any edge in discretionary Gann trading likely comes from pivot selection and trade management, not rule-based geometric timing.

Pivot density ran below target (2.3–3.7 pivots/year vs 4–8). Sparser pivots should have helped if the mechanism were real; worse results with fewer pivots supports retirement.

---

## Configuration

- **Live:** `gann` excluded from `config/live.yaml` `strategies.enabled`.
- **Code:** `src/firm/strategies/gann.py` — disabled; default weights angles 50% + retracement 50% (cycles weight 0) for ablation only.

**Do not reopen** this research line without a new, falsifiable hypothesis and pre-registered pass criteria.
