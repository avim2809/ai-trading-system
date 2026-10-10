# core_v1 constants (P3-11)

All data to 2026-09-30 is in-sample; No data after 2026-09-30 was examined.

- constants.json sha256: `77cbac0332e03d44077e4ea207746d80d3cd0924dac52ad555a8ea835926373b`
- window: 1994-02-02 .. 2026-09-29; data snapshot `ab5d0a987833b9e7f24bf1cf7d3a5d01661b5c8a4a9b843feaa03b1829491a0f`
- tau (from the charter): 0.09; code commit `514cd3c7bfbdc78e501e7371010b87503b1e956a`; seed 20261005
- instrument weights: handcrafted_one_group_per_asset_class (sha256 `6eb94e426e90a832b4a7c1778f36212bd2e1503d6cdb3e89141d45f8df3d0d83`)
- expected rule Sharpe 0.3 (research/charters/core_v1.md section 3: 'The plan's own range is 0.3 to 0.6 net' (low end taken))

## Pooled scalars (mean |raw*scalar| = 10 on the uncapped raw forecast; the capped mean is reported separately)

| rule | scalar | mean abs uncapped | mean abs capped |
|---|---|---|---|
| ewmac_2 | 14.0885 | 10.000000 | 9.4202 |
| ewmac_4 | 10.0713 | 10.000000 | 9.5090 |
| ewmac_8 | 6.96919 | 10.000000 | 9.5658 |
| ewmac_16 | 4.68994 | 10.000000 | 9.5701 |
| ewmac_32 | 3.05285 | 10.000000 | 9.5387 |
| ewmac_64 | 1.93348 | 10.000000 | 9.4821 |
| breakout_20 | 0.857654 | 10.000000 | 10.0000 |
| breakout_40 | 0.88619 | 10.000000 | 10.0000 |
| breakout_80 | 0.891388 | 10.000000 | 10.0000 |
| breakout_160 | 0.873981 | 10.000000 | 10.0000 |
| breakout_320 | 0.845217 | 10.000000 | 10.0000 |

## Surviving speeds per instrument (cost and turnover only)

| instrument | ewmac fast spans | breakout lookbacks | FDM (capped) | FDM (uncapped) |
|---|---|---|---|---|
| SPY | [4, 8, 16, 32, 64] | [20, 40, 80, 160, 320] | 1.2435 | 1.2435 |
| IWM | [4, 8, 16, 32, 64] | [20, 40, 80, 160, 320] | 1.2435 | 1.2435 |
| QQQ | [4, 8, 16, 32, 64] | [20, 40, 80, 160, 320] | 1.2435 | 1.2435 |
| EFA | [4, 8, 16, 32, 64] | [20, 40, 80, 160, 320] | 1.2435 | 1.2435 |
| EEM | [4, 8, 16, 32, 64] | [20, 40, 80, 160, 320] | 1.2435 | 1.2435 |
| SHY | [32, 64] | [160, 320] | 1.0589 | 1.0589 |
| IEF | [16, 32, 64] | [40, 80, 160, 320] | 1.1472 | 1.1472 |
| TLT | [8, 16, 32, 64] | [20, 40, 80, 160, 320] | 1.2121 | 1.2121 |
| TIP | [16, 32, 64] | [40, 80, 160, 320] | 1.1472 | 1.1472 |
| LQD | [8, 16, 32, 64] | [40, 80, 160, 320] | 1.1751 | 1.1751 |
| HYG | [8, 16, 32, 64] | [40, 80, 160, 320] | 1.1751 | 1.1751 |
| GLD | [4, 8, 16, 32, 64] | [20, 40, 80, 160, 320] | 1.2435 | 1.2435 |
| DBC | [4, 8, 16, 32, 64] | [20, 40, 80, 160, 320] | 1.2435 | 1.2435 |
| IYR | [4, 8, 16, 32, 64] | [20, 40, 80, 160, 320] | 1.2435 | 1.2435 |

## IDM: 1.8855 (uncapped 1.8855, cap 2.5); FDM cap 2.5

## Choices left open by the tickets (owner review)

- expected rule Sharpe is the low end of the charter's stated range
- equal group weights for ewmac and breakout
- Sharpe versus zero (rf=None); engine prices are adjusted closes
- IDM estimated once on the 'all' speed set
