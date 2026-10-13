# core_v1 research run: G-RESEARCH report

**Tier D.** A point failure or an H4 ENB miss. Outcome: passive (BM2, or the owner's 92/8 book); the candidate stops.

Reason: H4 ENB miss

> all data to 2026-09-30 is in-sample; no post-seal data examined.

## Provenance

- data_snapshot_id: `ab5d0a987833b9e7f24bf1cf7d3a5d01661b5c8a4a9b843feaa03b1829491a0f`
- code commit: `e6b7dd0d01714f0f1a4a1a9e201a1ea33d761750`
- prereg hash: `6274c43219d3c7fe1fe9d9686bd6a8cfe3c643fc32aaf68d86116279746c2bba`
- gates sha256: `bcecaec9ef44163596e27459779a124747d382c8068bf0607d4793df79f71540`

## Trial count and variance (DSR)

- N (gate, family-N): 31; raw count in the ledger: 843; N=463 and N=210 shown in the sensitivities.
- var_sr: 0.00075543 (length-adjusted over legacy_trend_standalone, alt_premia_T2, alt_premia_C1, eodhd_s3_trial_1, eodhd_s3_trial_2, eodhd_s3_trial_3, eodhd_s3_trial_4, eodhd_s3_trial_5, core_v1_grid; legacy trial lengths assumed equal to the candidate's; floored at the grid variance)

## H4 diversification

- ENB (asset-class P&L, Meucci): 2.4482 vs minimum 2.5

## Gates

| test | status | reason |
|---|---|---|
| G-RESEARCH-1 | **INSUFFICIENT** | confidence miss: DSR 0.6278 < 0.95 at family N=31 |
| G-RESEARCH-2 | **INSUFFICIENT** | PBO uninformative (effective grid N 1.115290427791857 < 4; median pairwise column correlation 0.9832546449620854 > 0.95); never a pass, never a hard fail |
| G-RESEARCH-3 | **PASS** | median path Sharpe 0.05886; 9 of 9 paths positive |
| G-RESEARCH-4 | **INSUFFICIENT** | confidence miss: DSR 0.5351 < 0.9 at 2x cost |
| G-RESEARCH-5 | **FAIL** | point failure: Volmageddon: 0.030 > 1.5 x 0.020 |
| G-RESEARCH-6 | **FAIL** | point failure: gross-cap days 0.423 > 0.2 |
| G-RESEARCH-7 | **FAIL** | point failure: gap -0.562 <= 0 vs BM2 and the correlation branch does not apply (corr 0.55) [FX-covered sub-window 2005-01-03 .. 2026-09-29, 5469 days; USD/ILS begins 2005-01-03; full window starts 1994-02-02] |
| G-RESEARCH-8 | **PASS** | charter research/charters/core_v1.md approved at commit 763031041a50ff78fef9466a538a021373cce57a; verified before the first core_v1 row |
| KELLY-TAU-BOUND | **FAIL** | point failure: tau 0.09 > max tau 0.0144 (Kelly bound on the deflated Sharpe, 50% haircut) |

### Values

**G-RESEARCH-1** (insufficient)

```
sharpe: 0.060997
sharpe_annual: 0.9683
skew: -0.26533
kurt: 8.7909
n_obs: 8218
n_gate: 31
n_raw: 843
var_sr: 0.00075543
var_sr_unadjusted: 0.00075547
var_sr_all_family: 0.0037488
var_sr_members: 20
var_sr_source: length-adjusted over legacy_trend_standalone, alt_premia_T2, alt_premia_C1, eodhd_s3_trial_1, eodhd_s3_trial_2, eodhd_s3_trial_3, eodhd_s3_trial_4, eodhd_s3_trial_5, core_v1_grid; legacy trial lengths assumed equal to the candidate's; floored at the grid variance
dsr: 0.62782
dsr_raw_n: 0.0075365
dsr_unadjusted_var: 0.62777
dsr_all_family_var: 1.0924e-09
n_family_plus_prior_attempt: 55
dsr_family_plus_prior_attempt: 0.40912
dsr_raw_N_463: 0.023023
dsr_ledgered_N_210: 0.083112
threshold: 0.95
```

**G-RESEARCH-2** (insufficient)

```
pbo: 0.70917
prob_oos_loss: 0
selected_median_oos_sharpe: 0.058977
n_splits: 12870
n_blocks: 16
grid_n: 12
effective_grid_n: 1.1153
median_pairwise_corr: 0.98325
family_variant_count: 31
perf_degradation_slope: -0.97943
rows_used: 8208
threshold: 0.2
```

**G-RESEARCH-3** (pass)

```
path_sharpes: [0.06027739010537851, 0.05811954744643429, 0.059229150285861176, 0.05869986891185344, 0.05888200224065628, 0.05885806846306987, 0.058027879795178205, 0.05907111635735945, 0.057973085150032265]
n_paths: 9
n_splits: 45
median: 0.058858
n_positive: 9
selection_counts: {0: 1, 2: 13, 4: 21, 8: 1, 10: 9}
embargo_pct: 0.01
n_paths_required: 9
min_positive_paths: 7
```

**G-RESEARCH-4** (insufficient)

```
net_sharpe: {'1': 0.06099715433086371, '2': 0.058340920768913826, '3': 0.05566805876923497}
dsr: {'1': 0.6278247554365867, '2': 0.5351080370517047, '3': 0.43978649957452604}
n_obs: 8218
flat_bps_net_sharpe_comparison_only: 0.057906
```

**G-RESEARCH-5** (fail)

```
min_active_fraction: 0.5
episodes: [{'name': 'GFC', 'status': 'ok', 'max_drawdown': 0.02924897642473534, 'reference_max_dd': 0.027928891283971042, 'n_days': 146, 'n_active_instruments': 14, 'breach': False, 'total_return': -0.01837518609982891}, {'name': 'Flash crash / euro', 'status': 'ok', 'max_drawdown': 0.012039222847810449, 'reference_max_dd': 0.0196273770926658, 'n_days': 64, 'n_active_instruments': 14, 'breach': False, 'total_return': -0.0009382918423850306}, {'name': 'US downgrade', 'status': 'ok', 'max_drawdown': 0.0069368910535606565, 'reference_max_dd': 0.019949796078568668, 'n_days': 65, 'n_active_instruments': 14, 'breach': False, 'total_return': 0.003710560277598862}, {'name': 'Volmageddon', 'status': 'ok', 'max_drawdown': 0.030238741535130842, 'reference_max_dd': 0.019949796078568668, 'n_days': 65, 'n_active_instruments': 14, 'breach': True, 'total_return': -0.025030638395532434}, {'name': 'Q4 2018', 'status': 'ok', 'max_drawdown': 0.012585362714383597, 'reference_max_dd': 0.019584768012979488, 'n_days': 63, 'n_active_instruments': 14, 'breach': False, 'total_return': -0.008407274572559076}, {'name': 'COVID', 'status': 'ok', 'max_drawdown': 0.01872887603014073, 'reference_max_dd': 0.01780634613002638, 'n_days': 52, 'n_active_instruments': 14, 'breach': False, 'total_return': -0.009319759973308361}, {'name': '2022 inflation', 'status': 'ok', 'max_drawdown': 0.03317958619718109, 'reference_max_dd': 0.031385180931053955, 'n_days': 209, 'n_active_instruments': 14, 'breach': False, 'total_return': -0.03036514050457495}, {'name': 'Regional banks', 'status': 'ok', 'max_drawdown': 0.004000743679847818, 'reference_max_dd': 0.012331641822322518, 'n_days': 23, 'n_active_instruments': 14, 'breach': False, 'total_return': 0.005394638249858641}, {'name': 'Yen-carry', 'status': 'ok', 'max_drawdown': 0.007301401414700037, 'reference_max_dd': 0.014805561307728326, 'n_days': 35, 'n_active_instruments': 14, 'breach': False, 'total_return': 0.007723789654690449}, {'name': 'Liberation Day', 'status': 'ok', 'max_drawdown': 0.014762934011426032, 'reference_max_dd': 0.018035960990543914, 'n_days': 53, 'n_active_instruments': 14, 'breach': False, 'total_return': -0.0006112660748004428}]
summary: {'worst_episode': '2022 inflation', 'worst_max_drawdown': 0.03317958619718109, 'any_breach': True, 'n_breached': 1, 'n_unusable': 0, 'n_not_applicable': 0, 'n_low_coverage': 0, 'all_ok': True}
survival_reference_p95_2520d: 0.063349
survival_ref_max_with_2p5_tau: 0.225
bootstrap_draws: 10000
bootstrap_seed: 20261005
multiple: 1.5
breaches: ['Volmageddon: 0.030 > 1.5 x 0.020']
non_ok: []
```

**G-RESEARCH-6** (fail)

```
chosen_net_sharpe: 0.060997
perturbations: [{'param': 'vol_span', 'direction': '-25%', 'net_sharpe': 0.06098458096983488, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42235071984894973}, {'param': 'vol_span', 'direction': '+25%', 'net_sharpe': 0.06103179558947906, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42341279206986077}, {'param': 'vol_blend_long_weight', 'direction': '-25%', 'net_sharpe': 0.061032545715012085, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4231767760207694}, {'param': 'vol_blend_long_weight', 'direction': '+25%', 'net_sharpe': 0.0610874260145867, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4225867358980411}, {'param': 'vol_long_window_days', 'direction': '-25%', 'net_sharpe': 0.06091540714366251, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4224687278734954}, {'param': 'vol_long_window_days', 'direction': '+25%', 'net_sharpe': 0.06092545287640542, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42329478404531506}, {'param': 'vol_floor_percentile', 'direction': '-25%', 'net_sharpe': 0.060980743766898195, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4243568562662261}, {'param': 'vol_floor_percentile', 'direction': '+25%', 'net_sharpe': 0.06102976064381896, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4228227519471324}, {'param': 'ewmac_fast_spans[0]', 'direction': '-25%', 'net_sharpe': 0.06099715433086371, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4228227519471324}, {'param': 'ewmac_fast_spans[0]', 'direction': '+25%', 'net_sharpe': 0.06099715433086371, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4228227519471324}, {'param': 'ewmac_fast_spans[1]', 'direction': '-25%', 'net_sharpe': 0.06103874347930403, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4227047439225867}, {'param': 'ewmac_fast_spans[1]', 'direction': '+25%', 'net_sharpe': 0.0608340121668732, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4221147037998584}, {'param': 'ewmac_fast_spans[2]', 'direction': '-25%', 'net_sharpe': 0.0610315542525033, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42329478404531506}, {'param': 'ewmac_fast_spans[2]', 'direction': '+25%', 'net_sharpe': 0.06077562779588067, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4228227519471324}, {'param': 'ewmac_fast_spans[3]', 'direction': '-25%', 'net_sharpe': 0.06066062176684742, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4222327118244041}, {'param': 'ewmac_fast_spans[3]', 'direction': '+25%', 'net_sharpe': 0.06081426062831322, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42376681614349776}, {'param': 'ewmac_fast_spans[4]', 'direction': '-25%', 'net_sharpe': 0.060662468008867904, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4189284871371253}, {'param': 'ewmac_fast_spans[4]', 'direction': '+25%', 'net_sharpe': 0.0610753923024053, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4255369365116828}, {'param': 'ewmac_fast_spans[5]', 'direction': '-25%', 'net_sharpe': 0.06080229582313403, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.420462591456219}, {'param': 'ewmac_fast_spans[5]', 'direction': '+25%', 'net_sharpe': 0.06125503665188189, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4231767760207694}, {'param': 'ewmac_slow_to_fast_ratio', 'direction': '-25%', 'net_sharpe': 0.060652160982007465, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4175123908425773}, {'param': 'ewmac_slow_to_fast_ratio', 'direction': '+25%', 'net_sharpe': 0.060904191921887445, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42530092046259144}, {'param': 'forecast_cap', 'direction': '-25%', 'net_sharpe': 0.061153122313628384, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4188104791125797}, {'param': 'forecast_cap', 'direction': '+25%', 'net_sharpe': 0.06082656935826613, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.41562426244984657}, {'param': 'breakout_lookbacks_N[0]', 'direction': '-25%', 'net_sharpe': 0.06112738323260174, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42530092046259144}, {'param': 'breakout_lookbacks_N[0]', 'direction': '+25%', 'net_sharpe': 0.06072665754358271, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4221147037998584}, {'param': 'breakout_lookbacks_N[1]', 'direction': '-25%', 'net_sharpe': 0.06094738576652006, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42329478404531506}, {'param': 'breakout_lookbacks_N[1]', 'direction': '+25%', 'net_sharpe': 0.06105007415475955, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42294075997167807}, {'param': 'breakout_lookbacks_N[2]', 'direction': '-25%', 'net_sharpe': 0.06084962267584176, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.419046495161671}, {'param': 'breakout_lookbacks_N[2]', 'direction': '+25%', 'net_sharpe': 0.06030783697066216, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4255369365116828}, {'param': 'breakout_lookbacks_N[3]', 'direction': '-25%', 'net_sharpe': 0.0609294552960204, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42034458343167336}, {'param': 'breakout_lookbacks_N[3]', 'direction': '+25%', 'net_sharpe': 0.06100041612633108, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42530092046259144}, {'param': 'breakout_lookbacks_N[4]', 'direction': '-25%', 'net_sharpe': 0.06057847843006907, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42117063960349305}, {'param': 'breakout_lookbacks_N[4]', 'direction': '+25%', 'net_sharpe': 0.06101505986329867, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42530092046259144}, {'param': 'breakout_smoothing_fraction_of_N', 'direction': '-25%', 'net_sharpe': 0.06102714224974649, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42069860750531035}, {'param': 'breakout_smoothing_fraction_of_N', 'direction': '+25%', 'net_sharpe': 0.06091729177288229, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42447486429077175}, {'param': 'fdm_cap', 'direction': '-25%', 'net_sharpe': 0.06099715433086371, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4228227519471324}, {'param': 'fdm_cap', 'direction': '+25%', 'net_sharpe': 0.06099715433086371, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4228227519471324}, {'param': 'idm_cap', 'direction': '-25%', 'net_sharpe': 0.06101278753065391, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4225867358980411}, {'param': 'idm_cap', 'direction': '+25%', 'net_sharpe': 0.06099715433086371, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4228227519471324}, {'param': 'buffer_fraction', 'direction': '-25%', 'net_sharpe': 0.061621157008742086, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4248288883644088}, {'param': 'buffer_fraction', 'direction': '+25%', 'net_sharpe': 0.06053923711610346, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.41951852725985367}, {'param': 'tau', 'direction': '-25%', 'net_sharpe': 0.061632207081651534, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.3667689402879396}, {'param': 'tau', 'direction': '+25%', 'net_sharpe': 0.06021733168366872, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.44512626858626386}, {'param': 'gross_cap', 'direction': '-25%', 'net_sharpe': 0.05986677880481059, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.44854850129808826}, {'param': 'gross_cap', 'direction': '+25%', 'net_sharpe': 0.06178760838258472, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.3795138069388718}, {'param': 'speed_cost_max_fraction', 'direction': '-25%', 'net_sharpe': 0.06106806703944068, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4304932735426009}, {'param': 'speed_cost_max_fraction', 'direction': '+25%', 'net_sharpe': 0.059659868205665705, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4228227519471324}, {'param': 'vol_ewma_span', 'direction': '-25%', 'net_sharpe': 0.061422684008599106, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.4228227519471324}, {'param': 'vol_ewma_span', 'direction': '+25%', 'net_sharpe': 0.060989614887598, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.42376681614349776}, {'param': 'max_vol_scale', 'direction': '-25%', 'net_sharpe': 0.06164503808206363, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.3991031390134529}, {'param': 'max_vol_scale', 'direction': '+25%', 'net_sharpe': 0.060450028900680554, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.43061128156714656}, {'param': 'instrument_risk_cap_multiple', 'direction': '-25%', 'net_sharpe': 0.0592106753232897, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.3397451026669814}, {'param': 'instrument_risk_cap_multiple', 'direction': '+25%', 'net_sharpe': 0.060208435038730516, 'mean_abs_forecast_after_reestimation': 10.0, 'gross_cap_bound_share': 0.47628038706632053}]
unassessed: []
gross_cap_bound_share: 0.42282
n_perturbation_rows: 54
floor_ratio: 0.7
n_breaches: 0
```

**G-RESEARCH-7** (fail)

```
sharpe_candidate: 0.013408
sharpe_bm2: 0.57514
correlation: 0.54651
margin: 0.5512
rationale_claimed: False
gate_variant: annual
sensitivity_variant: monthly
gap_ci: {'estimate': -0.5617276200818588, 'ci_low': -0.9507589661974303, 'ci_high': -0.1795294362261564}
gap_standard_error: 0.19675
alpha: 0.05
power: 0.8
minimum_detectable_sharpe_gap: 0.5512
underpowered: False
sensitivity_monthly_bm2: {'sharpe_bm2': 0.5523546952256757, 'gap': {'estimate': -0.5389468206552538, 'ci_low': -0.9315333984309513, 'ci_high': -0.14831276386787415}}
all_states_gate_variant: {'gross': {'candidate': {'cagr': 0.0046995314027695745, 'vol': 0.08156191105146958, 'sharpe': 0.09830777335127151, 'max_dd': 0.23130803566201452}, 'bm2': {'cagr': 0.06423795815372157, 'vol': 0.1080574537090509, 'sharpe': 0.6301881558813652, 'max_dd': 0.3342378235127187}}, 'after_cost': {'candidate': {'cagr': 0.0036894488129628478, 'vol': 0.08154102496675689, 'sharpe': 0.0859765159016266, 'max_dd': 0.23386625895982327}, 'bm2': {'cagr': 0.06422372946582455, 'vol': 0.1080585548611928, 'sharpe': 0.6300590768272986, 'max_dd': 0.33424296595199066}}, 'after_tax': {'candidate': {'cagr': -0.002421112848612794, 'vol': 0.08426502724323524, 'sharpe': 0.013407874570421936, 'max_dd': 0.2618097529581389}, 'bm2': {'cagr': 0.05413354479794119, 'vol': 0.100425883742489, 'sharpe': 0.5751354946522808, 'max_dd': 0.329882930680224}}}
core_only_100: {'net_sharpe_annual_pre_tax': 0.7446646850250996, 'n_trades': 924}
tax_assumptions: ["INFORMATION ONLY - not tax advice and not the owner's actual liability.", 'Active assumptions:', '  convention: deferred_mtm [plan-fixed-ex-ante]', '  dividend_rate: UNSET (reuses rate_real_gain) [unset]', '  foreign_tax_credit: False [unset]', '  fx_translation: trade_date [agent-knowledge]', '  inflation_adjust: False (FORCED: Israeli CPI unavailable; documented sensitivity; overstates tax)', '  loss_carryforward: False [agent-knowledge]', '  lot_method: fifo [agent-knowledge]', '  rate_real_gain: 0.25 [repo-brief+source-plan]', '  surtax_rate_1: 0.03 [source-plan]', '  surtax_rate_2: 0.02 [source-plan]', '  surtax_threshold_ils: 721560 [source-plan(current-year value open)]', '  withholding.IE: UNSET (adviser Q10): modelled as 0.0 placeholder, not a tax fact', '  withholding.US: UNSET (adviser Q10): modelled as 0.0 placeholder, not a tax fact']
block_length: 6.4597
n_boot: 10000
seed: 20261005
window_start: 2005-01-03
window_end: 2026-09-29
n_days: 5469
full_window_start: 1994-02-02
fx_first_date: 2005-01-03
fx_truncated: True
deviation: gate 7 (after-tax benchmark and comparison) is judged on the FX-covered sub-window 2005-01-03 .. 2026-09-29 (5469 of 8218 trading days) because USD/ILS begins 2005-01-03; the candidate's record before it is excluded from the after-tax comparison only; every other gate and every pre-tax number keeps the full window 1994-02-02 .. 2026-09-29
gap: -0.56173
correlation_max: 0.3
benchmark_variant: annual
```

**G-RESEARCH-8** (pass)

```
ok: True
detail: charter research/charters/core_v1.md approved at commit 763031041a50ff78fef9466a538a021373cce57a; verified before the first core_v1 row
```

**KELLY-TAU-BOUND** (fail)

```
tau: 0.09
deflated_sharpe_annual: 0.057772
haircut: 0.5
max_tau: 0.014443
planning_sharpe: 0.028886
```

## Comparators (same engine, same Sharpe convention)

- BM2 annual (gate, after tax, annualised): sharpe=0.57514
- BM2 monthly (sensitivity): sharpe=0.55235
- core_only_100 (engine, pre-tax, annualised): sharpe=0.74466

## Sensitivities

- DSR at family N 31: 0.6278247554365867
- DSR at raw N 843: 0.0075365256421577875
- DSR at family N + prior attempt's 24 duplicate grid rows = 55: 0.40912394729075563
- DSR at N=463: 0.023023021136654498
- DSR at N=210: 0.08311228432118617
- DSR with unadjusted var_sr 0.000755: 0.6277685577520458
- DSR with all-family var_sr 0.00375: 1.092431367588631e-09
- monthly BM2 sensitivity: Sharpe gap {'estimate': -0.5389468206552538, 'ci_low': -0.9315333984309513, 'ci_high': -0.14831276386787415}
- flat-bps comparison net Sharpe (never gates): 0.057906373610055656
- Kelly bound on the raw (undeflated) annual net Sharpe, informational: tau <= 0.2421

## Deviations

- gate 7 (after-tax benchmark and comparison) is judged on the FX-covered sub-window 2005-01-03 .. 2026-09-29 (5469 of 8218 trading days) because USD/ILS begins 2005-01-03; the candidate's record before it is excluded from the after-tax comparison only; every other gate and every pre-tax number keeps the full window 1994-02-02 .. 2026-09-29
- family membership of N=31 is provisional (gates family_membership_confirmed_by_owner: false)
- legacy trial lengths are unknown: var_sr assumes they equal the candidate's (conservative)
- H4 family streams: one active family, so the family-level ENB is diagnostic only
- robustness: all 27 parameters are perturbed (P4-03 layer wired in; a parameter that cannot be perturbed would be listed unassessed, gate 6 insufficient)
- P4-03 sizing layer: vol scale, instrument risk cap and gross cap; the 0.40 class cap is not applied (not a gate-6 parameter)
- rf is None throughout (Sharpe versus zero), as in P2-05
- 24 registered grid rows from an earlier incomplete attempt are in the ledger (earlier evaluation attempt registered the grid, then stopped before any gate was computed (legacy var_sr source unresolved: StartupError); its rows are kept, never deleted or edited); the grid was registered again; gate N stays the frozen family N 31

## Known leakage

- pooled scalars, speed filter, FDM and IDM were estimated on the full window (P3-11) and are not re-estimated on the CPCV training groups
- the instrument weights are fixed ex ante (handcrafted, one group per asset class); the universe was chosen without performance input
- all data to 2026-09-30 is in-sample (OD-04): CPCV and PBO address selection among the grid only

## Decision note

Tier C/D or ENB miss: stop. The outcome is passive (BM2, or the 92/8 book with its active satellite noted); this is a legitimate result.

all data to 2026-09-30 is in-sample; no post-seal data examined.
