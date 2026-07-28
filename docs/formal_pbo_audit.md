# Formal PBO walk-forward audit

First end-to-end run of `scripts/run_walk_forward_pbo_audit.py` on the cache
panel (2020-01-01 → 2026-06-30), using a **genuine** `param_grid` that mirrors
real firm tuning choices:

| Candidate | `signal_combination` | `allocation_method` |
|-----------|---------------------|---------------------|
| 0 | `confidence` | `conviction_weighted` |
| 1 | `optimal` | `conviction_weighted` |
| 2 | `optimal` | `equal_weight` |

Setup: 10-strategy roster, 25-symbol universe, `data_source: cache`, 3 folds,
70% train / 30% test per fold, `selection_metric: sharpe_ratio`.

Reproduce:

```bash
python scripts/run_walk_forward_pbo_audit.py --n-splits 3
python scripts/run_pbo_trial_audit.py --fold-ids <ids from output>
```

## Results (2026-07-27)

| Metric | Value |
|--------|-------|
| **PBO** | 0.686 |
| **Deflated Sharpe (DSR)** | 0.285 |
| **Probabilistic Sharpe (PSR)** | 0.631 |
| **Verdict** | `fail` |
| **PBO folds** | 3 |

Fold run ids:

- `20260727_174614_024361_74dd75ad`
- `20260727_175123_164255_09ac5df9`
- `20260727_175622_175468_db8d4e77`

OOS Sharpe (per fold): -0.24, 0.58, 0.41 (mean ≈ 0.25).

## Interpretation

High PBO (~0.69) indicates that **in-sample winner selection among these three
candidates is not reliable** — the train-window best config often underperforms
on the subsequent test window. This is expected on a short cache panel with only
three competing trials and does **not** by itself invalidate `optimal` combination
(which won the separate 3-window A/B in `docs/portfolio_construction_diagnosis.md`);
it flags that **grid search over these knobs without a longer panel / hold-out
folds risks overfitting**.

**Next steps:**

1. Re-run after `longer-dataset` (10y+ with delistings) lands.
2. Expand `param_grid` only with candidates you would genuinely deploy (avoid
   fishing on many variants).
3. Treat PBO `fail` as a gate before promoting any new hyperparameter set to
   `config/live.yaml`.

Raw JSON: `/tmp/walk_forward_pbo_audit.json` (local run artifact).
