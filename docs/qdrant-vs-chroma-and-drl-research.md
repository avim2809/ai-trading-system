# Research: Qdrant-vs-ChromaDB and DRL-for-trading

**Date:** 2026-06-30 · **Method:** deep-research (fan-out search → fetch →
adversarial verification → synthesis). Verifies two contested recommendations
from an external architecture review (Glean) before acting on them.

Pipeline: 2 questions → ~26 sources → claims extracted → adversarially verified
(2-of-3 refute kills a claim) → synthesized. Three claims were **refuted** in
verification (listed below).

---

## Q1 — Migrate ChromaDB → Qdrant for filter correctness? **Verdict: No (over-engineering).**

The review's premise is **largely outdated** for current ChromaDB:

- **Pre-filtering (medium confidence, vote 2-1):** Current ChromaDB (Feb 2026
  docs) **pre-filters** — `where`/`where_document` predicates build an
  eligible-ID set *before* the ANN/KNN stage; it degrades to brute force when
  the eligible set is small. So the "post-filter recall loss" premise is no
  longer accurate. — [Chroma Cookbook](https://cookbook.chromadb.dev/core/advanced/queries/)
- **Hybrid search (high, 3-0):** Chroma now ships **native sparse+dense hybrid**
  (BM25 + SPLADE, fused via RRF). Hybrid is *not* Qdrant-only. ⚠️ Best-documented
  for Chroma **Cloud** — verify open-source single-machine availability + version.
  — [trychroma](https://www.trychroma.com/project/sparse-vector-search)
- **The real, engine-agnostic issue (high, 3-0):** recall degrades under
  **low-selectivity filters** — when a restrictive `ticker AND date-range AND
  filing-type` AND-filter yields a tiny qualifying subset, those vectors form a
  sparsely connected HNSW subgraph and recall drops (affects HNSW *and* IVFFlat,
  pre- *and* post-filter, via different mechanisms). Broad filters: negligible
  impact. — [arXiv 2602.11443](https://arxiv.org/pdf/2602.11443),
  [FAVOR 2605.07770](https://arxiv.org/html/2605.07770v1)
- **Qdrant superiority was REFUTED here (vote 1-2):** the claim that Qdrant's
  payload-index bitset pre-filtering yields higher recall did not survive
  verification. Qdrant being strictly better on recall is **unproven** in this corpus.

**Decision rule:** migrate **only if** you *measure* unacceptable recall@k on
representative highly-restrictive filters in current ChromaDB **and** a
specialized filtered-ANN engine demonstrably recovers it. For a
thousands-to-low-millions single-machine corpus, that bar is rarely met. Note:
PR #4 already routes the worst case (date/ticker/numeric filters on runs/trades)
to **DuckDB SQL**, and added a **BM25+dense hybrid** retriever — so the two
classic Qdrant justifications are largely already addressed in-repo.

*Caveat:* no head-to-head ChromaDB-vs-Qdrant-vs-pgvector recall benchmark on a
financial ticker/date/filing workload was found; concrete per-engine scale
ceilings were not established by surviving claims.

---

## Q2 — Add a DRL trading strategy? **Verdict: Defer; research-only behind strict overfitting discipline.**

The credible peer-reviewed literature is **overwhelmingly skeptical**:

- **Headline returns without trial counts are worthless (high, 3-0):** best-of-N
  Sharpe inflates with the number of trials even when true skill is zero (False
  Strategy Theorem). A backtest that doesn't control for search extent is
  "worthless, regardless of how excellent the reported performance" — directly
  implicating cited figures like **2240% ROI** or **Sharpe 1.70** that omit trial
  counts. — [Deflated Sharpe Ratio, Bailey & López de Prado 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)
- **False positives are near-certain under multiple testing (high, 3-0):** in a
  controlled overfit example **100% of in-sample Sharpes were positive yet ~78%
  of out-of-sample Sharpes were negative** — high backtest Sharpe alone is
  uninformative. — [Probability of Backtest Overfitting 2017](https://www.researchgate.net/publication/318600389_The_probability_of_backtest_overfitting)
- **DRL adds its own fragility (high, 3-0):** reproducing SOTA deep RL is "seldom
  straightforward"; seeds/hyperparameters/reward-scale swing results; hard to
  separate real improvement from noise — sharper still in non-stationary markets.
  — [Henderson et al., Deep RL That Matters](https://arxiv.org/pdf/1709.06560)
- **The best rigorous DRL result is modest and conditional (high):** the leading
  overfitting-aware paper shows only that *less-overfitted* agents beat
  *more-overfitted* agents + an equal-weight strategy + a passive index over an
  **~8-week** crypto window — **not** that DRL beats strong active baselines
  (momentum, mean-variance, GBR/Ridge) out-of-sample after costs. Its real
  contribution is an overfitting *detector*. — [arXiv 2209.05559](https://arxiv.org/pdf/2209.05559)
- **Regime-conditioned DRL (e.g. HMM posterior):** robustness improvements appear
  mainly in single-author backtests; **no independent replication** found.

**Required discipline before trusting any DRL result:** disclose effective trial
count; apply **Deflated Sharpe Ratio** and/or **Probability of Backtest
Overfitting (CSCV)**; reject plain single hold-out; **walk-forward with strict
point-in-time** info sets; **realistic costs**; Harvey et al. `t > 3.0` hurdle.
A credible example reports an honest **null** (Sharpe 0.33, aggregate p=0.34)
under exactly this discipline. — [arXiv 2512.12924](https://arxiv.org/pdf/2512.12924)

**Bottom line:** adding DRL is far more likely to manufacture backtest-overfit
risk than real out-of-sample alpha. If pursued, it's a *research experiment*
gated behind DSR/PBO + the repo's existing walk-forward PIT infrastructure and
realistic costs — never a production alpha source judged on backtest ROI.

---

## Refuted claims (failed verification)

- ChromaDB docs "acknowledge a recall-correctness risk" in filtered ANN — **0-3**.
- Qdrant payload-index bitset pre-filtering yields higher recall — **1-2**.
- A *negative* in-sample/out-of-sample Sharpe relationship — **1-2** (the correct
  framing is that high IS Sharpe is *uninformative*, not negatively predictive).

## Net effect on the architecture-review plan

- **Tier B (Qdrant): drop** unless a measured recall problem appears on
  restrictive filters. The justifications are already largely addressed in-repo.
- **Tier C (DRL): keep deferred** as research-only behind DSR/PBO/walk-forward.
