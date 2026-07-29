# Regime ensemble — research spike scoping (not implemented)

Deferred item from the 2026-07-29 P/L-improvement research pass (Phase 7 of
6 implemented phases — see the CPCV/HRP/market-impact/CVaR/LLM-audit/
structured-reflection work landed alongside this). This is scope-only:
recommend picking it up as an independent follow-up once Phases 1–6 have had
time to be validated (walk-forward A/Bs, paper-trading observation), not
bundled into the same change set — it's genuine research with an uncertain
payoff, unlike the other phases which had a clear, bounded implementation.

## Why

`GaussianRegimeModel` (`src/firm/regime/model.py`) already had one real bug
fixed this session: label instability (Bull/Bear labels swapping between
refits on thin separation) was addressed via a separation-based damping
factor in `HMMRegimeStrategy` (`src/firm/strategies/regime_hmm.py`). That
fix flipped `regime_hmm`'s own attributed Sharpe from negative to positive
in all 3 diagnostic windows. But the *complementary* feature —
`strategy_regime_weights` (per-strategy multipliers conditioned on the
detected market regime, `src/firm/agents/research/_regime_weights.py`) — was
A/B'd with real thresholds and **hurt** portfolio Sharpe, and remains
disabled by default in both `config/settings.yaml` and `config/live.yaml`.
The likely cause: a single Gaussian HMM's regime *label* is still noisier
than the strategy-weighting feature assumes, even with separation damping.
2025–2026 research on ensemble regime detection (HMM + boosting/bagging
voting) reports materially better Sharpe and fewer false regime-switch
signals than a standalone HMM — directly relevant to derisking the
regime-weights feature enough to reconsider enabling it.

## Proposed approach (not started)

- New module `firm/regime/ensemble.py` (or extend `model.py`) implementing
  an ensemble regime classifier behind the **same interface** `RegimeState`/
  `GaussianRegimeModel.classify()`/`MarketRegimeDetector.detect()` already
  expose, so every consumer (`ctx.market_regime`, `RiskAgent`'s
  `regime_overlay`, `strategy_regime_weights`) needs zero changes — only the
  detector's internals change.
- Candidate designs, roughly in order of effort:
  1. **Ensemble-HMM voting** — fit several `GaussianHMM`s with different
     random seeds / `n_states` / covariance types, take a majority vote (or
     average posterior) on the current label. Cheapest change, reuses
     existing `hmmlearn` dependency and `GaussianRegimeModel` almost as-is.
  2. **HMM + boosted/bagged classifier** — train a gradient-boosted or
     bagged classifier (features: rolling return/vol/momentum stats) on the
     HMM's own historical labels as a smoothing/confirmation layer,
     matching the literature's "ensemble-HMM" framing more closely. Needs a
     labeled training loop, not just inference-time voting.
  3. **Statistical jump model** as a lower-complexity alternative to HMM
     specifically for the regime *signal* (not necessarily replacing the
     detector used for `RiskAgent`'s exposure overlay) — designed
     specifically for downside-risk-aware regime switching, worth
     evaluating as a simpler substitute rather than an addition.
- Validation: reuse `scripts/calibrate_strategy_regime_weights.py` (already
  built this session) against the same 3 diagnostic windows from
  `docs/portfolio_construction_diagnosis.md`, so the new detector is judged
  against the same bar that made the current feature fail its A/B.

## Explicit non-goals for this spike

- Not replacing `HMMRegimeStrategy`'s per-symbol alpha signal (already
  fixed and net-positive this session) — this is scoped to the *market-level*
  regime detector feeding `strategy_regime_weights`/`RiskAgent.regime_overlay`.
- Not enabling `strategy_regime_weights` or `regime_overlay` in
  `config/live.yaml` until a new A/B shows a genuine improvement — same
  promotion-gate discipline as every other knob in this system.
