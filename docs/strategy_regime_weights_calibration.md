# Strategy regime weights calibration

A/B comparison of `strategy_regime_weights` **off** vs **on** (example weights from
`config/settings.yaml`) on the three diagnostic windows from
`docs/portfolio_construction_diagnosis.md`. All runs used:

- `data_source: cache`
- 10-strategy roster (gann/ml_prediction off)
- `signal_combination: optimal`
- `seed: 42`

Reproduce:

```bash
python scripts/calibrate_strategy_regime_weights.py
```

## Results (2026-07-27)

| Window | Weights | Port Sharpe | Port return | Max DD |
|--------|---------|-------------|-------------|--------|
| `run_18mo_2025_2026` | off | **0.256** | 0.021 | 0.079 |
| `run_18mo_2025_2026` | on | -0.157 | -0.018 | 0.099 |
| `wf_fold0_2020_2021` | off | **0.269** | 0.007 | 0.056 |
| `wf_fold0_2020_2021` | on | -0.319 | -0.008 | 0.053 |
| `wf_fold1` | off | -0.981 | -0.018 | 0.037 |
| `wf_fold1` | on | **-0.053** | -0.002 | 0.029 |

**Delta (on − off) portfolio Sharpe:**

| Window | Δ Sharpe |
|--------|----------|
| `run_18mo_2025_2026` | **−0.413** |
| `wf_fold0_2020_2021` | **−0.589** |
| `wf_fold1` | **+0.928** |

## Verdict

The shipped example weights are **not** safe to enable globally: they materially
**hurt** portfolio Sharpe in 2/3 windows while helping the short/low-signal
`wf_fold1` window substantially. This mirrors the circuit-breaker finding — naive
playbook multipliers need per-window calibration, not copy-paste defaults.

**Recommendation:** keep `strategy_regime_weights.enabled: false` in both
`config/settings.yaml` and `config/live.yaml` until a revised weight table is
validated on held-out folds. The feature remains wired for research; use
`scripts/calibrate_strategy_regime_weights.py` to iterate on candidate weights,
or `scripts/suggest_strategy_regime_weights.py` to draft weights from empirical
strategy × regime Sharpe on a train window (always validate hold-out).

Raw JSON: `/tmp/strategy_regime_weights_calibration.json` (local run artifact).

## v2 — data-driven weights (2026-07-28)

`scripts/suggest_strategy_regime_weights.py` derives multipliers from train-window
(2020–2024) strategy × regime Sharpe on SPY-labeled days.

### Full 3-window calibration (`--weights-json` from suggest output)

| Window | Off Sharpe | On Sharpe | Δ |
|--------|------------|-----------|---|
| `run_18mo_2025_2026` | 0.256 | **0.307** | **+0.051** |
| `wf_fold0_2020_2021` | 0.269 | **0.480** | **+0.211** |
| `wf_fold1` | -0.981 | -1.686 | **−0.705** |

**Verdict:** v2 beats the hand-tuned v1 example on 2/3 windows but **still fails**
the short/low-signal `wf_fold1` window badly (v1 helped that fold; v2 hurts it).
Keep `enabled: false` until a scheme that is stable across all folds is found
(e.g. softer multipliers, longer train panel post-`longer-dataset`, or fold-aware
ensembling).

```bash
python scripts/suggest_strategy_regime_weights.py --output /tmp/suggested.json
python scripts/calibrate_strategy_regime_weights.py --weights-json /tmp/suggested.json
```

### v2-soft — damped multipliers (33% strength toward 1.0)

Same v2 table scaled toward neutral (`1 + 0.33 × (mult − 1)`):

| Window | Δ Sharpe (soft vs off) |
|--------|------------------------|
| `run_18mo_2025_2026` | **+0.218** |
| `wf_fold0_2020_2021` | **+0.111** |
| `wf_fold1` | **−1.390** |

Softening **helps** the two longer windows more than full v2, but **worsens**
`wf_fold1` further (−1.39 vs −0.71 full v2). No free lunch — keep disabled.
