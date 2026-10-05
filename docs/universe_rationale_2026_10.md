# Instrument universe rationale (P2-01, 2026-10)

Status: candidate for owner review. Written before any result of the new research programme exists. **No return, Sharpe, correlation,
volatility or drawdown statistic was computed or consulted; none appears in this document.** No data dated on or after 2026-10-01 was read
(asof <= 2026-09-30). Owner decisions consumed: OD-01 (capital under 25k: the ETF path is the only one built), OD-02 (undecided: no futures data),
OD-12 (undecided: US-listed tickers only, `ucits_variant: null`), OD-13 (futures file is a DRAFT candidate list).

## Selection rule (fixed before the list was written)

1. Ten asset-class cells, fixed in advance: equity_us, equity_dm_ex_us, equity_em, treasury, tips, credit_ig, credit_hy, gold, commodities, reit.
2. Candidates per cell are the US-listed ETFs already owned in `data/research/eodhd/etfs_full/` (names listed in the P2-01 ticket) that represent that cell.
3. Include an ETF if (a) it is the earliest-inception vehicle in its cell (or among the earliest up to 3 distinct exposures; ties by expense ratio,
   owner-supplied and dated, none supplied so far; no tie needed beyond the exact 3-way treasury tie which fits the cap), (b) it has at least 5 years of
   history, and (c) its median dollar ADV (close x volume) over the first 3 years after inception exceeds the screen floor.
4. At most 3 per cell; cap 25 instruments; minimum 15. Near-duplicate exposure inside a cell (e.g. VWO next to EEM, JNK next to HYG) is excluded.
5. No current AUM is used (none is held in the repo, today's AUM is survivorship- and performance-correlated).

The only statistic computed is the early-window median dollar ADV (volume times close, first 3 years after inception) plus the dates in the
data calendar. Inception comes from `extras_manifest_full.json` (`etf:<SYM>.first`), cross-checked against the first parquet row.

### Deviation / disclosure about the screen floor
The ticket text reuses one number ($20M ADV) as both the trade-time floor and the early-window screen. Applied to the early window it would remove
SPY itself (early median about 12M), IEF and LQD, which is absurd for liquidity reasons that have nothing to do with returns. I therefore
fixed the early-window **screen floor at $1M** (a round number) and kept $20M as the per-instrument `liquidity_floor_adv_usd` for trade-time checks
(P2-04/P5). Disclosure: I chose $1M after seeing the early-window ADV figures (a liquidity statistic only, not performance). The only candidate affected
is EWJ (early median about 0.27M, excluded). A different floor between about 0.3M and 2.8M changes nothing else. The owner may override.

## Warm-up rule (pre-registered here)
The vol estimator (P3-01) uses a 2520-day long window. Until an instrument has 2520 days, use an expanding mean of the same quantity. An
instrument enters the portfolio only after 256 return days. `first_trade_date` = the date of the 257th data row (256 returns) = portfolio entry,
read by P3-07 for active-instrument counts. `full_vol_window_date` = the 2521st row, reporting only.

## Result (14 instruments; amended 2026-10-05)

Owner review on 2026-10-05 dropped VGK: it is a regional subset of EFA (Europe is over 60% of EFA), so under equal-within-class weights it would make Europe about 80% of the developed ex-US bucket without adding an independent bet. The structural rule (no ETF that is a regional or sector subset of another in the same cell) uses no performance data. `min_instruments` was lowered from 15 to 14 accordingly, before any ledger trial. QQQ stays: it overlaps SPY by about 54% of weight but is not a subset, and its tech tilt is roughly offset by IWM inside the class.

| Cell | Members | Reason |
|---|---|---|
| equity_us | SPY, IWM, QQQ | large cap, small cap, growth/tech index vehicles; all pass screen |
| equity_dm_ex_us | EFA | EWJ fails the screen; EFA earliest broad vehicle; VGK dropped at owner review (subset of EFA) |
| equity_em | EEM | earliest broad EM; VWO duplicate |
| treasury | SHY, IEF, TLT | exact earliest tie (2002-07-26), three durations |
| tips | TIP | earliest linker; SCHP later |
| credit_ig | LQD | earliest IG; AGG is a blend with no cell |
| credit_hy | HYG | earliest HY; JNK duplicate |
| gold | GLD | earliest physical gold |
| commodities | DBC | earliest broad commodity; GSG/DJP duplicate |
| reit | IYR | earliest REIT (VNQ later, so excluded even though prior prereg scripts used VNQ) |

Per-instrument inception, `first_trade_date`, `full_vol_window_date`, prior exposure and one-line reason are in `config/universe_etf.yaml`.
Excluded candidates and reasons are listed there too. The Vanguard mutual funds VUSTX, VFITX, VFISX are pre-inception proxies only, never members.

## Prior exposure
Generated mechanically: any frozen `scripts/*_preregistered*.py` containing the quoted ticker. Most members have been seen by earlier
preregistered evaluations (flags mean the data is not fresh, and tell P3 which legacy trials enter the trend-family var_sr and N). IWM's
flag comes from the insider-cluster prereg, QQQ, TIP, LQD, HYG have none.

## Futures
`config/universe_futures.yaml` is a DRAFT candidate list seeded from the 19 markets of the frozen DRAFT `futures_trend_preregistered_bars.py`.
No futures data exists or is purchased (OD-02), the futures path is not built under OD-01. Contract specs there are unverified; margins are not stored.
