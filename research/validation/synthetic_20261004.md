# Synthetic GARCH-t validation of the statistics pipeline (P1-08), 20261004

**Verdict: PASS** (marginal items listed below)

## Provenance

- date_utc: 20261004
- quick: False
- non_evidence: False
- git_commit: d5f8e1b1bc556b733d41c5ad372beda47a0e1c62
- git_tree_clean: True
- gates_yaml_sha256_(meta removed): f212120e7613372643aad3124977a4404a34dae26d0bb67bf2a8b820e5ce4869
- gates_sha256: f212120e7613372643aad3124977a4404a34dae26d0bb67bf2a8b820e5ce4869
- host_ledger_before: absent
- host_ledger_after: absent
- throwaway_ledger_root: /tmp/claude-1001/-local-store-research/9c98bdc3-591b-4263-a2b6-ffa9aca169f2/scratchpad/synthetic_ledger_fgjz1794
- workers: 2
- runtime_s: 5683.3
- children_user_cpu_s: 0.0
- peak_rss_children_MB: 192.9
- python: 3.14.4
- master_seed: 20261004
- cross_correlation: 0.3
- base_seeds: {'null_size': 20261004, 'null_pbo': 20262004, 'power_single': 20263004, 'power_multi': 20264004, 'strong_drift': 20265004, 'reported_sr1_k50': 20266004, 'reported_k10_sr15': 20267004, 'trend_k50': 20268004, 'carry_k50': 20269004, 'autocorr_null': 20270004, 'stress_null': 20271004, 'plumbing_ledger': 20272004}
- gates_yaml_sha256: f212120e7613372643aad3124977a4404a34dae26d0bb67bf2a8b820e5ce4869

- scenarios block sha256: `826fb9b304160225199991be29469e8bd801b1fd6ced2e7e26d2d2f3ecea9094`
- cross_correlation / master_seed source: /local/store/research/ai-trading-system/.claude/worktrees/w5a/plan/drafts/P1-08/scenario_freeze.yaml (gates.yaml defers these two values to P1-08)

## Failures

- none

## Marginal (point estimate meets the bar, Monte-Carlo interval does not)

- size reality_check: 0.070 (95% CI 0.051-0.096, n=500) vs max 0.07
- size spa: 0.070 (95% CI 0.051-0.096, n=500) vs max 0.07
- size romano_wolf_fwer: 0.070 (95% CI 0.051-0.096, n=500) vs max 0.07

## Scenario results

### null_size

kind=null K=50 T=2520 SR=0.0 B=2000 sims=500 base_seed=20261004 corr=0.3 ar_rho=0.0 garch={'omega': 5e-06, 'alpha': 0.05, 'beta': 0.9, 'nu': 6.0, 'mu': 0.0}

- reality_check: 0.070 (95% CI 0.051-0.096, n=500)
- spa: 0.070 (95% CI 0.051-0.096, n=500)
- romano_wolf_fwer: 0.070 (95% CI 0.051-0.096, n=500)
- dsr_false_pass: 0.006 (95% CI 0.002-0.017, n=500)
- psr0_ks_p: 0.2888250590445729

### null_pbo

kind=null K=50 T=2520 SR=0.0 B=2000 sims=200 base_seed=20262004 corr=0.3 ar_rho=0.0 garch={'omega': 5e-06, 'alpha': 0.05, 'beta': 0.9, 'nu': 6.0, 'mu': 0.0}

- pbo_mean: 0.4815322455322455
- pbo_ci: [0.45904944235956696, 0.504015048704924]

### power_single

kind=drift K=1 T=2520 SR=1.0 B=2000 sims=500 base_seed=20263004 corr=0.3 ar_rho=0.0 garch={'omega': 5e-06, 'alpha': 0.05, 'beta': 0.9, 'nu': 6.0, 'mu': 0.0}

- t_test: 0.930 (95% CI 0.904-0.949, n=500)
- psr: 0.930 (95% CI 0.904-0.949, n=500)

### power_multi

kind=drift K=50 T=2520 SR=1.3 B=2000 sims=500 base_seed=20264004 corr=0.3 ar_rho=0.0 garch={'omega': 5e-06, 'alpha': 0.05, 'beta': 0.9, 'nu': 6.0, 'mu': 0.0}

- reality_check: 0.862 (95% CI 0.829-0.889, n=500)
- spa: 0.870 (95% CI 0.838-0.897, n=500)
- romano_wolf: 0.860 (95% CI 0.827-0.888, n=500)
- romano_wolf_any: 0.862 (95% CI 0.829-0.889, n=500)
- dsr_pass: 0.536 (95% CI 0.492-0.579, n=500)
- analytic_bonferroni_power: 0.8463084921014048
- oracle max-t power: 0.858 (95% CI 0.825-0.886, n=500) (crit 3.068); oracle max-SR pass: 0.858 (95% CI 0.825-0.886, n=500)

### strong_drift

kind=drift K=30 T=1600 SR=3.0 B=2000 sims=500 base_seed=20265004 corr=0.3 ar_rho=0.0 garch={'omega': 5e-06, 'alpha': 0.05, 'beta': 0.9, 'nu': 6.0, 'mu': 0.0}

- dsr_pass: 1.000 (95% CI 0.992-1.000, n=500)
- pbo_mean: 5.8275058275058275e-05
- pbo_ci: [7.1201903522686625e-06, 0.00010942992619784788]
- analytic_bonferroni_power: 0.999998118767475
- oracle max-t power: 1.000 (95% CI 0.992-1.000, n=500) (crit 3.001); oracle max-SR pass: 1.000 (95% CI 0.992-1.000, n=500)

### reported_sr1_k50 (reported only, not a pass bar)

kind=drift K=50 T=2520 SR=1.0 B=2000 sims=500 base_seed=20266004 corr=0.3 ar_rho=0.0 garch={'omega': 5e-06, 'alpha': 0.05, 'beta': 0.9, 'nu': 6.0, 'mu': 0.0}

- reality_check: 0.558 (95% CI 0.514-0.601, n=500)
- spa: 0.566 (95% CI 0.522-0.609, n=500)
- romano_wolf: 0.546 (95% CI 0.502-0.589, n=500)
- romano_wolf_any: 0.558 (95% CI 0.514-0.601, n=500)
- dsr_pass: 0.160 (95% CI 0.130-0.195, n=500)
- pbo_mean: 0.1750257964257964
- pbo_ci: [0.16211455102060646, 0.18793704183098636]
- cpcv_pos_paths_mean: 8.368
- analytic_bonferroni_power: 0.5287170928346143
- oracle max-t power: 0.538 (95% CI 0.494-0.581, n=500) (crit 3.063); oracle max-SR pass: 0.538 (95% CI 0.494-0.581, n=500)
- CPCV >= 7 of 9 paths positive: 0.902 (95% CI 0.873-0.925, n=500)

### reported_k10_sr15 (reported only, not a pass bar)

kind=drift K=10 T=2520 SR=1.5 B=2000 sims=500 base_seed=20267004 corr=0.3 ar_rho=0.0 garch={'omega': 5e-06, 'alpha': 0.05, 'beta': 0.9, 'nu': 6.0, 'mu': 0.0}

- reality_check: 0.988 (95% CI 0.974-0.994, n=500)
- spa: 0.988 (95% CI 0.974-0.994, n=500)
- romano_wolf: 0.988 (95% CI 0.974-0.994, n=500)
- romano_wolf_any: 0.988 (95% CI 0.974-0.994, n=500)
- dsr_pass: 0.758 (95% CI 0.719-0.793, n=500)
- analytic_bonferroni_power: 0.9849049452395132
- oracle max-t power: 0.990 (95% CI 0.977-0.996, n=500) (crit 2.449); oracle max-SR pass: 0.990 (95% CI 0.977-0.996, n=500)

### trend_k50 (reported only, not a pass bar)

kind=trend K=50 T=2520 SR=1.3 B=2000 sims=500 base_seed=20268004 corr=0.3 ar_rho=0.0 garch={'omega': 5e-06, 'alpha': 0.05, 'beta': 0.9, 'nu': 6.0, 'mu': 0.0}

- reality_check: 0.128 (95% CI 0.102-0.160, n=500)
- spa: 0.128 (95% CI 0.102-0.160, n=500)
- romano_wolf: 0.080 (95% CI 0.059-0.107, n=500)
- romano_wolf_any: 0.128 (95% CI 0.102-0.160, n=500)
- dsr_pass: 0.032 (95% CI 0.020-0.051, n=500)
- analytic_bonferroni_power: 0.8463084921014048
- oracle max-t power: 0.122 (95% CI 0.096-0.154, n=500) (crit 3.016); oracle max-SR pass: 0.122 (95% CI 0.096-0.154, n=500)

### carry_k50 (reported only, not a pass bar)

kind=carry K=50 T=2520 SR=1.3 B=2000 sims=500 base_seed=20269004 corr=0.3 ar_rho=0.0 garch={'omega': 5e-06, 'alpha': 0.05, 'beta': 0.9, 'nu': 6.0, 'mu': 0.0}

- reality_check: 0.816 (95% CI 0.780-0.848, n=500)
- spa: 0.818 (95% CI 0.782-0.849, n=500)
- romano_wolf: 0.812 (95% CI 0.775-0.844, n=500)
- romano_wolf_any: 0.816 (95% CI 0.780-0.848, n=500)
- dsr_pass: 0.528 (95% CI 0.484-0.571, n=500)
- analytic_bonferroni_power: 0.8463084921014048
- oracle max-t power: 0.788 (95% CI 0.750-0.822, n=500) (crit 3.164); oracle max-SR pass: 0.788 (95% CI 0.750-0.822, n=500)

### autocorr_null (reported only, not a pass bar)

kind=null K=50 T=2520 SR=0.0 B=2000 sims=500 base_seed=20270004 corr=0.3 ar_rho=0.3 garch={'omega': 5e-06, 'alpha': 0.05, 'beta': 0.9, 'nu': 6.0, 'mu': 0.0}

- reality_check: 0.066 (95% CI 0.047-0.091, n=500)
- spa: 0.066 (95% CI 0.047-0.091, n=500)
- romano_wolf_fwer: 0.066 (95% CI 0.047-0.091, n=500)
- reality_check_iid_control: 0.314 (95% CI 0.275-0.356, n=500)

### stress_null (reported only, not a pass bar)

kind=null K=50 T=2520 SR=0.0 B=2000 sims=500 base_seed=20271004 corr=0.3 ar_rho=0.0 garch={'omega': 1e-06, 'alpha': 0.08, 'beta': 0.9, 'nu': 5.0, 'mu': 0.0}

- reality_check: 0.036 (95% CI 0.023-0.056, n=500)
- spa: 0.036 (95% CI 0.023-0.056, n=500)
- romano_wolf_fwer: 0.036 (95% CI 0.023-0.056, n=500)
- dsr_false_pass: 0.000 (95% CI 0.000-0.008, n=500)

### plumbing_ledger (reported only, not a pass bar)

kind=null K=50 T=2520 SR=0.0 B=2000 sims=20 base_seed=20272004 corr=0.3 ar_rho=0.0 garch={'omega': 5e-06, 'alpha': 0.05, 'beta': 0.9, 'nu': 6.0, 'mu': 0.0}

- reality_check: 0.050 (95% CI 0.009-0.236, n=20)
- plumbing_ok: 1.000 (95% CI 0.839-1.000, n=20)

---
Harness thresholds were read from gates.yaml frozen at f212120e7613372643aad3124977a4404a34dae26d0bb67bf2a8b820e5ce4869; none were changed after viewing results.
