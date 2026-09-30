---
name: project-edge-search-sep29
description: "9/29-30 edge search: backtest split bug found+fixed (branch research/edge-search, NOT merged to main); 0/11 strategies have standalone edge; all 7 alt-premia Tier C; recommendation = trade 60/40 benchmark (proposal only); BTC trend the only lead"
metadata:
  node_type: memory
  type: project
  originSessionId: c4c796e1-061f-42c7-9fa9-255931a43502
  modified: 2026-09-30T00:24:03.556Z
---

Session 2026-09-29 → 30. The owner asked for "an approach that actually makes money OOS, even if it means abandoning the architecture". Full verdict: `docs/edge_search_verdict_2026_09.md`, on branch `research/edge-search` (worktree under that session's scratchpad).

**Backtest split bug, real and fixed on the branch only.**
- The backtest broker marked and filled at the cache's raw, split-unadjusted `close`, so every split booked a fake ±75–95% day. C3's worst day, −6.96% on 2024-06-10, was the NVDA split.
- Fix: `backtest/datafeeds.total_return_adjust_panel`, applied in `execute_backtest`.
- All earlier cache backtests that held stocks through a split are contaminated.
- Branch commits: `7e9fedb` (fix), `bd11bd9` (preregs), plus the results commit. **Not merged to main and not pushed**: merging changes tracked `src/` under the running services, see [[feedback-never-edit-live-checkout]].

**Results:**
- **9/28 combination eval on the fixed engine:** still FAIL. This is a re-measurement on a corrected feed: every bar is adjusted and portfolio paths diverge, so old and new daily returns correlate only 0.56–0.80 even on non-split days. OOS C0 −0.69→−0.22, PBO 0.97→0.46, the placebo is still indistinguishable. Ledger 52→57.
- **Live config vs cash and SPY, 2020–26:** Sharpe above T-bills 0.14, +25% total, beta ≈ 0. SPY +152%.
- **Step 1 (fp 1d54517a):** 0/11 strategies survive. Most are negative even gross, and turnover of 60–100%/day kills the rest. `regime_hmm` has gross +0.88, net −1.41 (post-hoc lead only).
- **Step 2 (fp 76598975; alternative premia vs SPY / 60-40 / vol-targeted SPY, 2007–2026, free Tiingo/CBOE/FRED data):** all 7 Tier C, mostly negative point estimates. BTC 4-week trend vs BTC held: +0.28 Sharpe, passes 6/7 bars, fails significance. The power analysis showed the minimum detectable gap is 0.4–0.9.
- Both steps were independently recomputed by agents.

**Recommendation, a proposal awaiting owner sign-off:**
- Move one instance, suggested Alpaca, to 60/40 SPY/IEF monthly (the pre-registered fallback, D1 default) via a standalone allocator. The current risk caps (5% per name, net 0.5) can't express it.
- Keep the other instance as control.
- Optionally run a 5–10% BTC-trend forward paper sleeve.

**Also found:**
- IBKR live latent split issue: when the REST fallback serves raw OHLC, `_resolve_cycle_prices`/`_closing_price` prefer raw close. Not fixed.
- Tests still write `data/llm_cache.db` and `data/vectordb` in the working directory.
- Vendor quotes for Step 3 survivorship-free data: Norgate Platinum USD 630/yr, but it's Windows-only. EODHD £200–1000/yr. Not purchased.

**Owner context:** research briefs for non-equity instruments are in `docs/research_brief_new_instruments{,_standalone}.md`. See [[user-profile]].

Related: [[project-optimal-combination-lockout-sep28]], [[feedback-out-of-box-fact-backed-research]].

**Update 2026-09-30 (later): external research on non-equity instruments.**
Owner ran `docs/research_brief_new_instruments.md` (repo-access version)
against an external agent; full report + my corrections in
`docs/research_findings_beyond_equities_2026_09_30.md`.

- **Report's shortlist:** futures trend-following (strongest, but 36-54
  person-days — no contract-multiplier/margin/roll concept anywhere in the
  code), a passive bond/commodity sleeve (claimed cheapest at 2-4 days),
  insider-purchase clustering (PEAD explicitly dead per Tier-1 evidence,
  drop it), FX momentum (borderline; FX carry confirmed dead post-2008,
  Sharpe 0.04-0.16), crypto carry beyond BTC (BIS: >50% of months would
  hit forced liquidation at 10x — don't build). CFDs, VIX-futures basis,
  covered-call/put-write variants, individual commodities, spin-offs,
  ETF-structure arbitrage all landed on its "don't bother" list — same
  premium as something already tested, or decayed post-2010.
- **I corrected its #2 pick (the diversification sleeve) after a direct
  repo read**, which the report itself flagged as unverified: `allocation_method:
  risk_parity` (`agents/trader.py:256`) only reweights symbols the 11
  strategies already signalled on that cycle — it cannot host an always-on
  bond/commodity position. It needs the same standalone-allocator pattern
  as the 60/40 proposal, not a config change.
- **Quick empirical check** (cached Step 2 data, a 50/30/20 IEF/GLD/DBC
  sleeve at 30% NAV vs. the real split-fixed book): the live book's beta is
  already ~0, so it barely draws down on its own — in the 2022 selloff the
  book was +6% and the sleeve dragged the blend down to +1.8%. The
  "diversification helps in a crisis" argument doesn't obviously apply to
  *this* book. Single scenario, not pre-registered.
- Nothing here has been pre-registered or tested against the evaluation
  standard yet. It's an input to that process, not a result of it.
