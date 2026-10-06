# P2-05 after-tax BM2 dry run (60/40 SPY/IEF, US-listed)

**INFORMATION ONLY - not tax advice and not the owner's actual liability.**

No post-seal data was read (asof 2026-09-30, pre-seal, through `firm.research.data_access`). Exploratory ledger trial family `p2_05_bm2_dry_run` (run once, counted). Seed 13, 10000 paired stationary-bootstrap draws (Politis-White).

## Substitutions and assumptions (read first)

- Israeli CPI unavailable: inflation_adjust forced False, CPI flat 1.0 (overstates modelled tax)
- USD/ILS from EODHD forex (Bank of Israel series not available)
- dividends: real EODHD files for SPY and IEF (all 14 ETF files on disk); implied series used only as a cross-check
- withholding US/IE unset (adviser Q10): modelled 0.0 placeholder

Tax assumptions printed by the model:

```
INFORMATION ONLY - not tax advice and not the owner's actual liability.
Active assumptions:
  convention: deferred_mtm [plan-fixed-ex-ante]
  dividend_rate: UNSET (reuses rate_real_gain) [unset]
  foreign_tax_credit: False [unset]
  fx_translation: trade_date [agent-knowledge]
  inflation_adjust: False (FORCED: Israeli CPI unavailable; documented sensitivity; overstates tax)
  loss_carryforward: False [agent-knowledge]
  lot_method: fifo [agent-knowledge]
  rate_real_gain: 0.25 [repo-brief+source-plan]
  surtax_rate_1: 0.03 [source-plan]
  surtax_rate_2: 0.02 [source-plan]
  surtax_threshold_ils: 721560 [source-plan(current-year value open)]
  withholding.IE: UNSET (adviser Q10): modelled as 0.0 placeholder, not a tax fact
  withholding.US: UNSET (adviser Q10): modelled as 0.0 placeholder, not a tax fact
```

## Window core_window: 2015-02-02..2026-09-28 (2931 days); ILS NAV

Gate benchmark variant (pinned in the gates file, Amendment 1): **annual**; variant with the higher after-tax Sharpe (information only, never gates): monthly

| variant | state | CAGR | vol | Sharpe (vs 0) | maxDD |
|---|---|---|---|---|---|
| annual_alpaca | gross | 6.48% | 10.46% | 0.653 | 12.74% |
| annual_alpaca | after_cost | 6.48% | 10.46% | 0.653 | 12.74% |
| annual_alpaca | after_tax | 5.21% | 8.81% | 0.621 | 10.21% |
| monthly_alpaca | gross | 6.52% | 10.54% | 0.653 | 13.54% |
| monthly_alpaca | after_cost | 6.52% | 10.54% | 0.652 | 13.54% |
| monthly_alpaca | after_tax | 5.25% | 8.85% | 0.623 | 10.85% |
| annual_ibkr | gross | 6.48% | 10.46% | 0.653 | 12.74% |
| annual_ibkr | after_cost | 6.48% | 10.46% | 0.653 | 12.74% |
| annual_ibkr | after_tax | 5.21% | 8.81% | 0.621 | 10.21% |
| annual_alpaca_terminal_liquidation | gross | 6.48% | 10.46% | 0.653 | 12.74% |
| annual_alpaca_terminal_liquidation | after_cost | 6.48% | 10.46% | 0.653 | 12.74% |
| annual_alpaca_terminal_liquidation | after_tax | 5.21% | 11.00% | 0.517 | 13.15% |

USD pre-tax BM2 (no FX, no tax):

| rebalance | state | CAGR | vol | Sharpe (vs 0) | maxDD |
|---|---|---|---|---|---|
| annual | gross | 8.77% | 10.23% | 0.873 | 21.32% |
| annual | after_cost | 8.77% | 10.23% | 0.873 | 21.32% |
| monthly | gross | 8.81% | 10.33% | 0.870 | 21.18% |
| monthly | after_cost | 8.81% | 10.33% | 0.869 | 21.19% |

Daily-rebalanced adjusted_close total-return reference (USD): CAGR 8.85%, vol 10.50%, Sharpe 0.861, maxDD 21.02%.

Paired bootstrap, monthly (sensitivity, as 'system') minus annual (BM2 annual variant), ILS NAV, block length 28.2:

| state | dSharpe [95% CI] | dCAGR [95% CI] | dMaxDD [95% CI] |
|---|---|---|---|
| gross | -0.0003 [-0.0232, +0.0237] | +0.0004 [-0.0023, +0.0036] | +0.0080 [-0.0054, +0.0116] |
| after_cost | -0.0006 [-0.0235, +0.0234] | +0.0004 [-0.0024, +0.0035] | +0.0080 [-0.0054, +0.0116] |
| after_tax | +0.0019 [-0.0205, +0.0253] | +0.0004 [-0.0018, +0.0030] | +0.0065 [-0.0051, +0.0093] |

Annual-rebalance trades: 24. Modelled tax per year (ILS, annual gate run):

| year | net real gain | tax gain | surtax | tax dividends | total |
|---|---|---|---|---|---|
| 2015 | 0 | 0 | 0 | 1,882 | 1,882 |
| 2016 | -27 | 0 | 0 | 2,005 | 2,005 |
| 2017 | 1,144 | 286 | 0 | 1,964 | 2,250 |
| 2018 | 2,535 | 634 | 0 | 2,295 | 2,929 |
| 2019 | -786 | 0 | 0 | 2,415 | 2,415 |
| 2020 | 7,009 | 1,752 | 0 | 1,948 | 3,700 |
| 2021 | 2,324 | 581 | 0 | 1,701 | 2,282 |
| 2022 | 20,783 | 5,196 | 0 | 2,450 | 7,645 |
| 2023 | -2,400 | 0 | 0 | 3,370 | 3,370 |
| 2024 | 18,229 | 4,557 | 0 | 4,098 | 8,655 |
| 2025 | 26,783 | 6,696 | 0 | 4,418 | 11,114 |
| 2026 | 11,040 | 2,760 | 0 | 2,926 | 5,686 |

## Window long_window: 2005-01-03..2026-09-29 (5469 days); ILS NAV

Gate benchmark variant (pinned in the gates file, Amendment 1): **annual**; variant with the higher after-tax Sharpe (information only, never gates): annual

| variant | state | CAGR | vol | Sharpe (vs 0) | maxDD |
|---|---|---|---|---|---|
| annual_alpaca | gross | 6.42% | 10.81% | 0.630 | 33.42% |
| annual_alpaca | after_cost | 6.42% | 10.81% | 0.630 | 33.42% |
| annual_alpaca | after_tax | 5.41% | 10.04% | 0.575 | 32.99% |
| monthly_alpaca | gross | 6.38% | 11.11% | 0.613 | 34.68% |
| monthly_alpaca | after_cost | 6.38% | 11.11% | 0.612 | 34.69% |
| monthly_alpaca | after_tax | 5.34% | 10.39% | 0.552 | 34.42% |
| annual_ibkr | gross | 6.42% | 10.81% | 0.630 | 33.42% |
| annual_ibkr | after_cost | 6.42% | 10.81% | 0.630 | 33.42% |
| annual_ibkr | after_tax | 5.41% | 10.04% | 0.575 | 32.99% |
| annual_alpaca_terminal_liquidation | gross | 6.42% | 10.81% | 0.630 | 33.42% |
| annual_alpaca_terminal_liquidation | after_cost | 6.42% | 10.81% | 0.630 | 33.42% |
| annual_alpaca_terminal_liquidation | after_tax | 5.41% | 11.60% | 0.513 | 34.75% |

USD pre-tax BM2 (no FX, no tax):

| rebalance | state | CAGR | vol | Sharpe (vs 0) | maxDD |
|---|---|---|---|---|---|
| annual | gross | 8.17% | 10.25% | 0.817 | 30.19% |
| annual | after_cost | 8.17% | 10.25% | 0.817 | 30.19% |
| monthly | gross | 8.13% | 10.63% | 0.788 | 32.18% |
| monthly | after_cost | 8.12% | 10.63% | 0.788 | 32.19% |

Daily-rebalanced adjusted_close total-return reference (USD): CAGR 8.32%, vol 10.88%, Sharpe 0.789, maxDD 31.39%.

Paired bootstrap, monthly (sensitivity, as 'system') minus annual (BM2 annual variant), ILS NAV, block length 7.1:

| state | dSharpe [95% CI] | dCAGR [95% CI] | dMaxDD [95% CI] |
|---|---|---|---|
| gross | -0.0177 [-0.0497, +0.0120] | -0.0004 [-0.0038, +0.0029] | +0.0126 [-0.0077, +0.0274] |
| after_cost | -0.0180 [-0.0499, +0.0117] | -0.0004 [-0.0039, +0.0028] | +0.0126 [-0.0076, +0.0275] |
| after_tax | -0.0228 [-0.0600, +0.0132] | -0.0007 [-0.0045, +0.0029] | +0.0143 [-0.0093, +0.0310] |

Annual-rebalance trades: 44. Modelled tax per year (ILS, annual gate run):

| year | net real gain | tax gain | surtax | tax dividends | total |
|---|---|---|---|---|---|
| 2005 | 0 | 0 | 0 | 2,961 | 2,961 |
| 2006 | 518 | 129 | 0 | 3,326 | 3,456 |
| 2007 | 1,281 | 320 | 0 | 3,518 | 3,839 |
| 2008 | -720 | 0 | 0 | 3,107 | 3,107 |
| 2009 | -311 | 0 | 0 | 2,964 | 2,964 |
| 2010 | -6,641 | 0 | 0 | 2,870 | 2,870 |
| 2011 | -948 | 0 | 0 | 2,865 | 2,865 |
| 2012 | 1,138 | 285 | 0 | 3,027 | 3,312 |
| 2013 | 567 | 142 | 0 | 2,944 | 3,086 |
| 2014 | 7,843 | 1,961 | 0 | 3,574 | 5,534 |
| 2015 | 3,225 | 806 | 0 | 4,025 | 4,831 |
| 2016 | 314 | 79 | 0 | 4,152 | 4,231 |
| 2017 | 10,621 | 2,655 | 0 | 4,067 | 6,722 |
| 2018 | 15,726 | 3,932 | 0 | 4,752 | 8,684 |
| 2019 | 799 | 200 | 0 | 5,000 | 5,199 |
| 2020 | 26,441 | 6,610 | 0 | 4,033 | 10,643 |
| 2021 | 8,118 | 2,029 | 0 | 3,523 | 5,552 |
| 2022 | 59,707 | 14,927 | 0 | 5,072 | 19,999 |
| 2023 | -1,666 | 0 | 0 | 6,977 | 6,977 |
| 2024 | 48,620 | 12,155 | 0 | 8,484 | 20,639 |
| 2025 | 66,547 | 16,637 | 0 | 9,148 | 25,784 |
| 2026 | 27,119 | 6,780 | 0 | 6,059 | 12,839 |

## Versus `core_only_100` (docs/allocation_replay_2026_09.json, published summary)

published summary only (no daily series on disk): core_only_100 is a monthly drift-band rebalanced 60/40 SPY/IEF; its Sharpe is ex-T-bill, ours is vs zero, so Sharpe is not comparable; CAGR/vol/maxDD are.

core_only_100 (2015-02-02..2026-09-28, 2931 days): CAGR 8.66%, vol 10.38%, Sharpe ex-T-bill 0.650, maxDD 21.04%, 279 orders.

Daily-rebalanced SPY/IEF adjusted-close reference CAGR 8.85%; gap to published 0.19 pp; within the 1 pp bound of the opt-in `real_data` sanity test: **True** (computed here, equivalent to that test).

## Real vs implied dividends (cross-check; the run uses the REAL files)

| symbol | real | implied | matched on date | median rel. err | max rel. err |
|---|---|---|---|---|---|
| SPY | 87 | 136 | 87 | 3.74e-05 | 5.31e-03 |
| IEF | 290 | 290 | 290 | 2.12e-04 | 1.36e-02 |
