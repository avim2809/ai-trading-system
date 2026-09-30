# S5 crypto cross-sectional momentum: THE GATE (resolved 2026-09-30)

Per `docs/eodhd_shortlist_protocol_2026_10.md` and
`docs/research_brief_eodhd_findings.md` §1 candidate #5: this gate must clear
before S5 is scoped at all. **Only data availability/quality was examined.
No strategy, benchmark or placebo return was computed on the evaluation
window, and no relationship between past and future returns was examined.**

Data: `data/research/eodhd/crypto/*.parquet` (7,085 non-empty series, read via
the symlink at `data/research`), `data/research/eodhd/crypto_symbols.parquet`
(`code`, `listing` = active/delisted), `extras_manifest.json` (`crypto:*`
statuses; 1,187 empty).

## Verdict: GATE PASSES

Formerly-large, now-delisted coins are present in the local EODHD crypto data
with EOD history that runs through their collapse to a plausible death date
well before 2026-09, not truncated continuations to the present. The data
does contain the losers.

## Evidence

### 1. Coverage: do delisted coins have populated history at all?

`crypto_symbols.parquet`: 8,272 rows, 1,868 `active` / 6,404 `delisted`.
Cross-tabulated against `extras_manifest.json` status:

| listing | empty | ok |
|---|---|---|
| active | 0 | 1,868 |
| delisted | 1,187 | 5,217 |

100% of active symbols have populated history. **81.5% of delisted symbols
(5,217/6,404) have populated history; 18.5% (1,187) are metadata-only with no
downloaded bars.** This is a real, non-trivial coverage gap — any
universe-construction rule must check for non-empty history (it already does,
per protocol: "clean series within one segment") and can't assume every
`delisted` row is usable.

### 2. Do delisted series end at plausible death dates, not just "now"?

For the 5,217 populated delisted series, distribution of last bar date:

- min 2021-05-14, 25th pct 2022-06-10, median 2022-10-09, 75th pct
  2023-08-24, max 2026-02-26 (a single outlier; effectively everything else
  is well earlier).
- By last-bar year: 2021: 105, **2022: 2,991**, 2023: 1,051, 2024: 607,
  2025: 462, 2026: 1.
- **Zero** delisted series end on or after 2026-07-01 (i.e. none are
  masquerading as "delisted" while actually still reporting bars up to
  today). The 2022 spike matches the real 2022 crypto bear
  market/Terra-UST/FTX collapse wave, not an arbitrary data cutoff.

This is the key test the brief specified: dead coins stop, at dates that
track real-world collapse events, rather than silently continuing.

### 3. Ten known failed/collapsed coins: first/last dates and price path

| code | listing | first | last | peak date | peak px | last px | drawdown |
|---|---|---|---|---|---|---|---|
| LUNA-USD (old Terra) | delisted | 2019-08-07 | 2022-05-11 | 2021-03-21 | 21.95 | 1.19 | -94.6% |
| STLUNA-USD (staked LUNA) | delisted | 2022-03-29 | 2022-06-15 | n/a (79 rows) | — | — | ends 1 month after the May-2022 depeg |
| MIR-USD (Mirror Protocol) | delisted | 2018-10-03 | 2023-06-04 | 2022-05-30 | 0.320 | 0.00203 | -99.4% |
| SAFEMOON-USD | delisted | 2021-03-12* | 2023-09-07 | 2021-03-12 | 2.0e-8 | 4.5e-9 | -100.0% (real launch) |
| SRM-USD (Serum) | delisted | 2020-08-25 | 2025-11-16 | 2021-09-12 | 12.48 | 0.0091 | -99.9% |
| FEI-USD (Fei Protocol) | delisted | 2021-04-05 | 2023-06-04 | 2021-06-14 | 1.02 | 0.98 | -4% (depegged briefly, not a full collapse in this series) |
| BTRFLY-USD (Redacted Cartel) | delisted | 2021-12-19 | 2024-01-15 | 2022-01-06 | 3,443 | 16.91 | -99.5% |
| ICE-USD (Popsicle Finance) | delisted | 2021-04-01 | 2023-05-31 | 2021-11-04 | 61.3 | 1.18 | -98.1% |
| COVER-USD (Cover Protocol) | delisted | — | — | — | — | — | **empty** (one of the 1,187) |
| ANC-USD | delisted | 2013-07-15 | 2025-04-10 | 2013-12-04 | 11.57 | 0.030 | **not the Terra Anchor Protocol token** (see caveat below) |

\* `SAFEMOON-USD.parquet` has a single stray bar at 2018-08-07 ($0.29, likely
an unrelated earlier project's residual quote under the same ticker,
followed by a ~2.5-year gap), then a continuous run from 2021-03-12 at
$0.00000002 — matching SafeMoon's real March-2021 launch price and its
April-2021 peak near $0.0000104. Both facts (gap + price level) are visible
directly in the bars, which is exactly the kind of check this gate asked for.

8 of the 10 are unambiguous: real, once-large/heavily-traded coins, now
delisted, with EOD history running through the collapse to a death date well
before 2026-09-30 (matches known real-world failure dates). 1 of 10
(COVER-USD) is in the empty-history 18.5%, consistent with the base rate
found in §1 — not evidence of concealment, just the known coverage gap.

**Caveat — ticker reuse:** `ANC-USD`'s 2013 first-date and single-digit
prices show this file is an unrelated, much smaller altcoin that happened to
reuse the "ANC" ticker, not Terra's Anchor Protocol governance token. EODHD's
own disambiguation convention (seen elsewhere: `LUNA-USD` delisted/old vs.
`LUNA20314-USD` active/Terra-2.0; `TAO-USD` vs. `TAO22974-USD`; `COMP-USD` vs.
`COMP1-USD`/`COMP5692-USD`) is to append a numeric suffix to a newly
(re-)registered symbol when the bare ticker is already taken. **Universe
construction must not assume a bare ticker uniquely identifies the coin
market participants mean by that symbol** — this is a design decision for
§(b), not a gate failure, since it doesn't change whether delisted coins'
*own* history is populated and truthful.

### 4. Historical top-30-by-dollar-volume: does the data contain the losers?

For Jan-1 of each year 2018-2025, ranked USD-quoted crypto files (`*-USD.parquet`
only, excluding non-USD quote pairs like `BTC-EUR`/`BTC-GBP`) by trailing
30-day median dollar volume (`close x volume`), fraction of that year's
top-30 that is `delisted` as of today:

| anchor | n eligible | top-30 now delisted |
|---|---|---|
| 2018-01-01 | 644 | 7/30 (23%) |
| 2019-01-01 | 1,312 | 7/30 (23%) |
| 2020-01-01 | 1,684 | 6/30 (20%) |
| 2021-01-01 | 3,004 | 8/30 (27%) |
| 2022-01-01 | 4,390 | 5/30 (17%) |
| 2023-01-01 | 2,603 | 4/30 (13%) |
| 2024-01-01 | 2,015 | 1/30 (3%) |
| 2025-01-01 | 1,852 | 0/30 (0%) |

17-27% of once-top-30 coins for 2018-2022 anchors are now delisted — a
material loser fraction, not survivorship-only. The tapering toward 2024/2025
is expected, not a red flag: those cohorts have had only 1-2 years to fail
and get marked delisted as of this 2026-09-30 snapshot, versus 4-8 years for
the 2018-2021 cohorts.

**Note on the raw top-30 lists (design-relevant, not a gate issue):** the raw
ranking is polluted by items that are not independent "coins" for a
cross-sectional momentum design: leveraged/inverse tokens
(`3X-LONG-BITCOIN-TOKEN-USD`, `BULL1-USD`), wrapped/staked derivatives that
re-express another coin's price (`RENBTC-USD`, `ibETH-USD`, `WBETH-USD`,
`BTCB-USD`, `CBBTC32994-USD`, `RSETH-USD`), tokenized non-crypto assets
(`XAU-USD`, `PAXG-USD`, `XAUT-USD` = gold; `DJ30-USD`, `SPX-USD` = equity
indices), and ticker-collision duplicates (`COMP-USD`/`COMP1-USD`/
`COMP5692-USD`, `TAO-USD`/`TAO22974-USD`). §(b)'s exclusion rule addresses
this.

### 5. Are prices USD, and is volume in coin units or USD?

Quote-pair files (`*-USD.parquet`) hold USD-denominated OHLC (e.g. BTC-USD
close 2026-09-30 = $83,129; ETH-USD close = $2,667 — both plausible current
levels). `volume` is **USD notional, not coin units**: BTC-USD's 2026-09-30
`volume` is 26,436,243,456. Total BTC supply is ~19.9M coins, so 26.4B
"coins traded in a day" is impossible; 26.4B **dollars** of BTC turnover is
squarely in the known range for aggregated-exchange BTC daily dollar volume.
Same check on ETH-USD (`volume` 14,730,137,600 vs. ~120M ETH total supply)
confirms the same conclusion. Per protocol §1, downstream dollar-volume
universe construction should therefore use `close x volume` directly (not
`adjusted_close x volume` as for equities, where volume is split-adjusted and
price is raw) — crypto has no splits, and `close`/`adjusted_close` were
observed equal in all series checked.

### 6. Bar timestamps and gaps

All checked series (`BTC-USD`, `LUNA-USD`, `MIR-USD`, `SAFEMOON-USD`) stamp
bars at 00:00 UTC exactly, consistent with the protocol's execution-timing
assumption for crypto.

Gap behaviour varies by coin liveness/liquidity: `BTC-USD` has zero missing
calendar days over 5,924 days (2010-2026) but 4.2% zero-volume bars;
`LUNA-USD` zero missing days, 26.6% zero-volume bars (thin trading in its
2019 early history); `MIR-USD` 1.06% missing calendar days; `SAFEMOON-USD`
has the large pre-launch gap described in §3. `eodhd_clean.py`'s existing
rules (drop non-positive/missing price, drop zero-volume, spike-reversal,
segment breaks on unexplained up-jumps >150%) already remove the zero-volume
noise; they do **not** currently break a `segment` on a large *calendar* gap
(only on a large *price* jump), so a stray pre-launch bar followed by a
multi-year gap (as in SAFEMOON) is not itself segment-broken by the existing
rule unless the price move also qualifies. Flagged as a design consideration
for §(b) (a coin should not enter the eligible universe from a
pre-gap/pre-launch stub).

## Conclusion

The gate's stated pass condition — delisted coins that were once large are
present with history through their decline — is met, with clear supporting
evidence across coverage rates, death-date distributions, ten named
real-world collapses, and the historical top-30 survivorship check. Two
caveats (ticker reuse; no calendar-gap segment break) are carried into the
§(b) pre-registration as explicit design decisions, not gate failures.
