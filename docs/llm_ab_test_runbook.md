# LLM layer A/B paper test — runbook

Remediation plan item: **`llm-ab-test`**. Goal: determine whether
`llm_enhanced` fundamental/sentiment analysts add measurable, attributable
alpha vs pure quant on the **same** live universe, risk, and strategy stack.

**Status (2026-07-27):** config presets + env overrides scaffolded; **requires
weeks of sequential paper-trading time** to complete.

---

## Design constraints

- Production runs **one** `LiveTradingEngine` per `firm-api` instance (single IBKR
  paper account). True simultaneous A/B on one account is not possible.
- **Sequential A/B** is the supported model: run arm A for N weeks, then arm B
  for N weeks (or alternate monthly), comparing NAV/attribution from
  `data/live_state.db` and cycle logs.
- Both arms share **`config/live.yaml`** (universe, risk, strategies, costs).
  Only **`agent_modes`** differ via `FIRM_LLM_CONFIG`.

---

## Config presets

| Arm | `FIRM_LLM_CONFIG` | Analyst modes |
|-----|-------------------|---------------|
| **Quant-only** | `config/llm_ab_quant.yaml` | All agents `quant`; `enhancement.policy: cache_only` |
| **LLM-enhanced** | `config/llm_ab_llm.yaml` | `fundamental_analyst` + `sentiment_analyst` → `llm_enhanced`; `enhancement.policy: live_calls` |

Separate LLM cache DBs (`data/llm_cache_ab_{quant,llm}.db`) avoid cross-contamination.

Optional: `FIRM_LIVE_CONFIG` can point at an alternate live YAML if you need
experiment metadata blocks — defaults to `config/live.yaml`.

---

## Procedure

### Weekly monitoring (arm A / arm B)

```bash
FIRM_LLM_CONFIG=config/llm_ab_quant.yaml python scripts/snapshot_llm_ab_arm.py \
  --append docs/llm_ab_experiment_log.md
```

---
### 1. Baseline snapshot

```bash
# Record starting NAV, positions, and config hashes before switching arms
curl -s localhost:8000/api/live/status | jq .
sha256sum config/live.yaml config/llm_ab_quant.yaml config/llm_ab_llm.yaml
```

### 2. Arm A — quant-only (recommended first: fewer moving parts)

```bash
export FIRM_LLM_CONFIG=config/llm_ab_quant.yaml
# Restart production service or cycle live engine:
sudo systemctl restart ai-trading.service
# Or via API:
curl -X POST localhost:8000/api/live/stop
curl -X POST localhost:8000/api/live/start -H 'Content-Type: application/json' -d '{"broker":"ibkr_paper"}'
```

Run **≥8 weeks** (minimum for a noisy Sharpe read; 12+ preferred).

### 3. Arm B — LLM-enhanced

```bash
export FIRM_LLM_CONFIG=config/llm_ab_llm.yaml
sudo systemctl restart ai-trading.service
```

Run the **same calendar length** as arm A. Ensure `GROQ_API_KEY` / `VOYAGE_API_KEY`
are set; monitor `GET /api/llm/cache/stats` for cost/latency.

### 4. Compare

| Metric | Source |
|--------|--------|
| NAV / Sharpe / max DD | `data/live_state.db` portfolio history |
| Per-strategy attribution | Same DB + `PerformanceAttribution` export |
| LLM-specific edge | Compare `fundamental_analyst` / `sentiment_analyst` attributed returns; check signal meta `llm_enhanced: true` in cycle snapshots |
| Costs | LLM cache stats + broker commission reports |

Document results in a dated note under `docs/` (or a new run registry entry with
`notes: "llm-ab arm quant"`).

---

## systemd wiring

Add to `deploy/ai-trading.service` `Environment=` block when running an arm:

```ini
Environment=FIRM_LLM_CONFIG=/local/store/git/ai-trading-system/config/llm_ab_quant.yaml
```

Switch the path when changing arms; always `systemctl restart` after editing.

---

## Acceptance checklist (close `llm-ab-test` todo)

- [ ] Both arms ran ≥8 weeks on identical `live.yaml` universe/risk/strategies
- [ ] NAV curves exported and Sharpe/max-DD computed on matching calendar lengths
- [ ] Per-analyst attribution compared (quant vs llm_enhanced arms)
- [ ] Written conclusion: keep LLM layer, disable it, or restrict to one analyst

---

## Related

- Formal overfitting on strategy *selection* (not LLM): `scripts/run_pbo_trial_audit.py`
- Regime-conditional strategy weights (research): `strategy_regime_weights` in `config/settings.yaml`
