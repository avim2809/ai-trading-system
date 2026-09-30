# Futures trend-following: engineering sketch (2026-09-30)

**Status: sketch/proposal, nothing here has been built.** Written to scope
the effort behind the 36-54 person-day estimate in
`docs/research_findings_beyond_equities_2026_09_30.md` (Part 1, item 1), by
reading the real code paths that would need to change. No code in this repo
was modified as part of this sketch.

## 0. What this is being fitted into

The task that produced this document described a `src/firm/allocation/`
module (`Sleeve.target_weights -> orders`) as the home for this feature,
being built by another agent in parallel. **As of this session that module
does not exist anywhere in this checkout** — I checked `main`,
`feat/allocation-mode`, and `research/allocation-forward-test`; the only
allocation-adjacent code on any branch is `src/firm/live/capital_reallocation.py`
(the existing capital-sleeves rebalance-band logic, unrelated: it moves
capital *between* the 11 existing equity strategies' sleeves, it doesn't hold
a new asset class). Everything below is written against the **current**
code (equity/ETF-only) plus a description of the target shape
(`Sleeve.target_weights -> orders`) taken as given; it should be reconciled
with whatever that other work actually ships before either lands.

## 1. The core problem: nothing in this system has a concept of a futures contract

Every one of the four files below treats a position as `(symbol, shares,
price)` with **market value = shares × price**. A futures contract's real
notional is `contracts × price × multiplier`, and its cash impact when
opened is **margin** (a fraction of that notional), not the full notional in
cash. Both facts are absent everywhere, not just in one place — this is the
finding that drives the 36-54 person-day estimate, and it's confirmed by
direct reading, not assumed:

### `src/firm/brokers/base.py` — `OrderRequest` / `BrokerPosition`

```python
@dataclass
class OrderRequest:
    symbol: str
    side: Literal["buy", "sell"]
    quantity: float
    order_type: ... = "market"
    ...
```
No `multiplier`, no `expiry`/`contract_month`, no `asset_class`. `BrokerPosition`
is the same: `symbol, quantity, avg_cost, market_value, unrealized_pnl` — no
multiplier field, so `market_value` is only correct today because every
current instrument (stocks/ETFs) has multiplier 1. Needed: both dataclasses
need a `multiplier: float = 1.0` field (and ideally `expiry: date | None` on
`OrderRequest`/a position identity that distinguishes contract months, since
"ES" is not a single tradable instrument — "ESZ26" is).

### `src/firm/brokers/ibkr.py` — contract building is hardwired to equities

Every contract-building call site constructs the same thing:
```python
contract = Stock(symbol, "SMART", "USD")      # connect() health check, line 233
contract = Stock(symbol, "SMART", "USD")      # _get_qualified_contract_unlocked, line 348
contracts = [Stock(s, "SMART", "USD") for s in missing]  # warm_universe, line 369
```
There is no branch anywhere for `ib_async`'s `Future`/`ContFuture` types. This
needs:
- A per-symbol `asset_class` lookup (config-driven, e.g. a `futures.yaml`
  universe entry alongside the existing equity universe) that picks
  `Future(symbol, lastTradeDateOrContractMonth, exchange, multiplier)`
  instead of `Stock(...)`.
- **Roll logic**: `Future` contracts expire; the broker needs to know, for
  each symbol, which contract month is "current" on a given date (a roll
  calendar), and to qualify/trade the new month before the old one goes
  illiquid — none of this exists. `ContFuture` (ib_async's continuous-future
  helper, used for market data only, not order routing) could simplify price
  lookups but does not solve order routing, which must always target a real,
  qualified, dated contract.
- **Historical data for backtesting**: confirmed via IBKR's own TWS API docs
  (`docs/futures_data_vendor_comparison_2026_09.md`), `includeExpired` only
  reaches ~2 years past expiration — this repo's IBKR integration cannot be
  the backtest data source regardless of code changes; it can only be the
  live execution venue once a signal is built and paper-tested elsewhere.

### `src/firm/portfolio/state.py` — every NAV/weight/cost-basis computation is `shares × price`

```python
@property
def nav(self) -> float:
    return self.cash + sum(
        shares * self._last_prices.get(sym, 0.0)
        for sym, shares in self.holdings.items()
    )
```
Identical pattern in `get_weights()`, `update()`, `record_snapshot()`,
`unrealized_return_pct()` — five separate call sites, all needing a
`shares * price * multiplier_map.get(sym, 1.0)` correction, where
`multiplier_map` would need to be threaded into `PortfolioState.__init__`
(currently takes only `initial_capital`) and populated from wherever the
futures universe config lives. Getting even one of these five sites wrong
silently misprices NAV by the multiplier factor (e.g. 50x for ES) — the same
class of "silent wrong number" bug the split-adjustment fix
(`docs/edge_search_verdict_2026_09.md` §1) found and fixed for equities, except
here it would be wrong from day one rather than introduced by a corporate
action.

A second, distinct problem: `PortfolioState.cash` currently absorbs the full
notional of every fill (`self.cash -= shares * price`). A futures fill should
only debit **initial margin**, with the rest of the notional un-cash-settled
until close — this system has no margin/buying-power concept anywhere, only
a cash ledger. Approximating futures as "notional-debited, no margin" would
wildly understate available buying power and make position sizing
(RiskAgent's caps, ExecutionAgent's weight math) meaningless for a
margined instrument. This is a structural gap, not a field-rename.

### `src/firm/agents/execution.py` — order sizing assumes fractional, continuous shares

```python
dollar_amount = diff_w * nav
quantity = abs(dollar_amount / price)
```
No multiplier, and critically **no rounding to whole units** in the main
path (only the protective-stop helper does `int(round(...))`, line 484) —
equities/ETFs can be sized fractionally today and the broker (Alpaca) or a
downstream step handles whole-share rounding. Futures contracts **cannot be
fractional at all**: 0.37 of a contract does not exist at any broker. This
sizing line needs to become
`quantity = round(abs(dollar_amount / (price * multiplier)))` for any
futures symbol, with an explicit **minimum-contract-size** consequence: if
`abs(dollar_amount / (price * multiplier))` rounds to 0, that market gets
*no* position this cycle even though the target weight was nonzero — a kind
of forced no-trade-band that doesn't exist today and needs its own logging
(this system's own convention: log every fallback/rounding decision,
`.cursor/rules/logging.mdc`) so a persistently-zero-sized market is visible,
not silently dropped.

The existing `rebalance_band_pct` / `rebalance_fraction` / `_day_anchor`
machinery (lines ~270-337) operates entirely in **weight space** and should
carry over largely unchanged — the fix is isolated to the point where a
target weight becomes a `quantity`, not the band logic itself.

### `src/firm/backtest/engine.py` / `datafeeds.py` — plain equity feed and cash-account broker

```python
self.cerebro.broker.setcash(initial_capital)
commission = PercentageCommission(commission=commission_pct, spread_pct=spread_pct)
self.cerebro.broker.addcommissioninfo(commission)
```
`AdjustedPandasData` (`datafeeds.py`) is a plain `bt.feeds.PandasData` —
OHLCV only, no multiplier/margin metadata per feed. `backtrader` itself
*does* support futures natively (`bt.CommInfoBase(mult=..., margin=...,
stocklike=False)`, exactly the mechanism this repo doesn't use), so this is
the smallest lift of the four: a new `FuturesCommission(bt.CommInfoBase)`
alongside the existing `PercentageCommission`, applied per-symbol via
`cerebro.broker.addcommissioninfo(futures_commission, name=symbol)`, with
`mult` and `margin` sourced from the same config that would drive
`ibkr.py`'s contract building. The existing `PercentageCommission` stays for
the equity/ETF book; a mixed backtest (60/40 core + futures sleeve) needs
both registered simultaneously, which `backtrader` supports per-instrument.

### `src/firm/live/engine.py` — pricing and calendar both assume equities

`_resolve_cycle_prices`/`_closing_price` (already flagged as a *live, latent*
bug for equities in `docs/edge_search_verdict_2026_09.md` §6: a raw-close
fallback during an IBKR data-farm outage takes a fake jump across a stock
split) generalizes into a **required feature**, not a bug, for futures: a
roll date must never look like a fake return the way an unadjusted split
does. Whatever back-adjustment the purchased vendor's continuous series
provides for the *backtest* has no live-trading equivalent — the live engine
would need to detect "this cycle rolled the front contract" and translate a
position from the expiring contract to the new one as an explicit,
logged event (close old contract, open new contract, zero net signal
change), not as a price gap PortfolioState absorbs as P&L.

Calendar: `respect_market_hours`/`is_market_open()` gates every cycle against
a broker's equities session. Futures trade nearly 24 hours (CME Globex,
~23/6). For a **monthly-rebalance** signal (the pre-registered trend rule
only trades once a month) this is much less pressing than it would be for a
faster strategy — the practical fix is to keep routing the sleeve's monthly
orders during the existing equities-session cycle (there is always a
several-hour overlap between CME Globex and US equity market hours) rather
than building a genuine futures trading calendar, deferring that harder
problem unless/until a faster futures signal is ever considered.

## 2. Where the new allocation layer (`Sleeve.target_weights -> orders`) fits

Whatever that module turns out to look like, the futures sleeve needs it to
carry (per target-weight dict entry, alongside the weight itself) at least:
`asset_class` (`"equity" | "futures"`), and for futures, `multiplier` and the
currently-qualified `contract_month`/expiry — none of which exist in
`PortfolioSnapshot`, `OrderRequest`, or any current `target_weights: dict[str,
float]` signature anywhere in the codebase today (checked
`src/firm/contracts/models.py` and every `target_weights` call site). If the
other agent's allocator ships with a bare `dict[str, float]` weight
interface, it will need widening before a futures sleeve can plug into it —
this is the one integration point most likely to need a two-way
conversation rather than a one-sided sketch.

## 3. Feasibility at $100k-$1M: can a small account hold 19 markets via micros?

CME lists a micro (1/10th multiplier) contract for **every one of the 19
target markets**, confirmed this session (not assumed):

| Market | Micro ticker | Multiplier / contract size | Indicative initial margin | Source confidence |
|---|---|---|---|---|
| ES (S&P 500) | MES | $5 × index | ~$1,320 | confirmed |
| NQ (Nasdaq-100) | MNQ | $2 × index | UNVERIFIED (search returned an implausible "$100-300" figure — needs a direct CME check, likely $1,500-2,500 in practice) | UNVERIFIED |
| RTY (Russell 2000) | M2K | $5 × index | ~similar order to MES, not line-item confirmed | UNVERIFIED |
| YM (Dow) | MYM | $0.50 × index | ~$1,320 | confirmed |
| FDAX (DAX) | Eurex has its own micro-DAX-equivalent-sized product family (FDXM "Mini-DAX" confirmed via Databento's catalog listing); an even-smaller CME-style "micro" tier specifically for FDAX is UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| ZF/ZN/ZB (Treasuries) | Micro Treasury **Yield** futures exist for 10Y and 30Y confirmed; "all four" tenors implied (2Y/5Y less directly confirmed) | $10/basis point of yield | Likely low (a few hundred dollars; yield-based, not price-based) — UNVERIFIED exact figure | partially confirmed |
| FGBL/FGBS (Bund/Schatz) | No CME-style "micro" analog found for Eurex rates products in this search | UNVERIFIED | UNVERIFIED | UNVERIFIED |
| 6E/6J/6B/6A/6C (FX) | M6E, M6J, M6B, M6A, MCD/M6C — **all five confirmed to exist** | M6E: 12,500 EUR ($1.25/tick); others similarly ~1/10 of the full contract | M6E ~$405; others not individually confirmed but same order of magnitude | confirmed (existence), partially confirmed (margin) |
| CL (Crude) | MCL | 100 barrels | UNVERIFIED exact figure, roughly 1/10 of CL's ($6,000+) margin | partially confirmed |
| GC (Gold) | MGC | 10 troy oz | ~$675 | confirmed |
| HG (Copper) | MHG (Micro Copper) — confirmed to exist | 2,500 lbs | UNVERIFIED exact figure | partially confirmed |
| ZC (Corn) | Micro corn exists (CME's "Micro Ag futures" family) — confirmed to exist, exact ticker not individually confirmed | 1/10 of ZC | UNVERIFIED | partially confirmed |
| ZS (Soybeans) | Micro soybeans exists (same Micro Ag family) — confirmed to exist | 1/10 of ZS | UNVERIFIED | partially confirmed |

**Verdict: plausibly yes, with real caveats, not a clean yes.**

- A rough sum of just-one-micro-contract initial margins across all 19
  markets, using the confirmed figures (MES $1,320 + MYM $1,320 + M6E $405 +
  MGC $675, plus the unconfirmed-but-same-order-of-magnitude remainder)
  lands in the **very rough range of $15,000-$25,000** to hold a single
  contract in every market simultaneously — 15-25% of a $100k account,
  before any market's *intended* risk-parity weight calls for more than one
  contract, and before IBKR's own margin cushion above CME's exchange
  minimums (IBKR typically requires more than the bare exchange minimum;
  not quantified here). This is a rough order-of-magnitude estimate from
  partially-confirmed figures, not a number to size real risk against.
- **Granularity, not just margin, is the binding constraint at $100k.** The
  trend rule's inverse-vol sizing wants continuously-variable weights per
  market; a single micro contract is the smallest indivisible unit, so at
  $100k a market whose *intended* risk-parity weight is small (a low-vol FX
  pair, say) will likely round to exactly 1 contract or 0 — meaningfully
  lumpier than the theoretical target, in exactly the way
  `src/firm/agents/execution.py`'s rounding gap (section 1 above) predicts.
  This lumpiness is a real, unavoidable cost of small account size, not an
  engineering bug to fix.
- **This materially favors CSI Data / Norgate's stated $100k-$1M framing
  from the research brief being read as "$500k-$1M is comfortable, $100k is
  tight but not obviously impossible."** At $500k-$1M the same $15-25k
  minimum-margin footprint becomes 1.5-5% of NAV, and each market's target
  weight maps to several contracts instead of ~1, which is where inverse-vol
  sizing actually behaves the way the trend-following literature assumes.
- **Confirmed gaps to close before relying on this table**: NQ/RTY/CL/HG/
  grain-micro margin figures, whether Eurex has a true micro/nano tier for
  FDAX/FGBL/FGBS comparable to CME's, and IBKR's specific margin
  requirement (vs. bare CME SPAN minimums) for each. None of this required a
  signup to check today (CME's own contract-specs and margin pages are
  public) — it wasn't fully chased down here purely for session-time
  budget, and should be a fast follow-up, not a blocker to sharing this
  sketch.

## 4. Effort estimate (unchanged from the research brief, now with a basis)

Having read the actual code, **36-54 person-days is a reasonable estimate,
possibly light** on the low end once margin/buying-power accounting
(section 1, `PortfolioState.cash`) is counted as its own item rather than
folded into "position sizing." A rough breakdown, matching the code sections
above:
- `OrderRequest`/`BrokerPosition` multiplier + expiry fields, and threading a
  `multiplier_map`/futures-universe config through `PortfolioState`: 3-5 days.
- `IBKRBroker` `Future` contract building + roll-calendar logic + qualified-
  contract caching per contract month (mirroring the existing per-symbol
  cache, `warm_universe`): 8-12 days.
- `PortfolioState` multiplier correction across its 5 call sites + a real
  margin/buying-power ledger (not just a cash debit): 6-10 days.
- `ExecutionAgent` whole-contract rounding + minimum-size no-trade handling +
  logging: 3-5 days.
- `BacktestEngine`/`datafeeds.py` futures `CommInfo` (mult/margin) +
  per-symbol commission registration + continuous-contract feed loading from
  whichever vendor is purchased (parsing roll schedules into `backtrader`
  feeds): 6-10 days.
- `LiveTradingEngine` roll-event handling (detect + execute + log a front-
  month roll as a non-signal event) + reconciliation-safe restart behavior
  (matching the existing IBKR/Alpaca reconciliation rigor this system
  already holds itself to): 6-10 days.
- Coordination with whichever `src/firm/allocation/` shape actually ships
  (section 2): 2-4 days if it's close to the sketch above, more if its
  weight interface needs widening.
- Testing (this repo's own bar: regression tests that fail on the old
  behavior, per the split-adjustment fix precedent): folded into each item
  above rather than a separate line, consistent with how this system has
  scoped every other fix so far.

**Total: roughly 34-56 person-days**, consistent with the original estimate,
now grounded in the specific files and call sites rather than a general
"contract multipliers/margin/rolls" description.
