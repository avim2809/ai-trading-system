# LLM A/B experiment log

Tracks the sequential quant-only vs llm_enhanced paper-trading experiment.
See `docs/llm_ab_test_runbook.md` for procedure.

## Arm A — quant-only

| Field | Value |
|-------|-------|
| **Started** | 2026-07-27 (UTC) |
| **`FIRM_LLM_CONFIG`** | `config/llm_ab_quant.yaml` |
| **Live config** | `config/live.yaml` (unchanged) |
| **Target duration** | ≥8 weeks (12+ preferred) |

### Baseline at arm start (pre-restart snapshot)

- Engine state: `running`, broker: `ibkr_paper`, connected: yes
- Active strategies: 10 (momentum … regime_hmm)
- Schedule: `market_open`
- Config hashes (sha256):
  - `live.yaml`: `613605efcf76dff517232981bdb1bc8311168f22eee314597d173021cee6b788`
  - `llm.yaml` (prior production): `7890a2d4ac0b86fe50c66ca2d9786b27384f54db5e1fe330a8950e570176bdda`
  - `llm_ab_quant.yaml`: `c4632940bbc8320720ca255071a70442868ed8899e885225e16b3458ef961995`

### End criteria / notes

- [ ] Run until **≥2026-09-21** (8 weeks) before switching to arm B
- [ ] Export NAV curve from `data/live_state.db` at arm end
- [ ] Record Sharpe, max drawdown, per-analyst attribution vs arm B

## Arm B — llm_enhanced

_Not started._ Switch `FIRM_LLM_CONFIG` to `config/llm_ab_llm.yaml` and restart
`ai-trading.service` when arm A completes.

### Snapshot 2026-07-27 14:42 UTC

- `FIRM_LLM_CONFIG`: `config/llm_ab_quant.yaml`
- Portfolio snapshots: 1
- Latest NAV: 998984.02
- Ann. Sharpe (daily): n/a
- Max drawdown: 0.0

### Snapshot 2026-07-28 08:10 UTC

- `FIRM_LLM_CONFIG`: `config/llm_ab_quant.yaml`
- Portfolio snapshots: 1
- Latest NAV: 998984.02
- Ann. Sharpe (daily): n/a
- Max drawdown: 0.0
