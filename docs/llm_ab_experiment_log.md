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

- [x] ~~Run until **≥2026-09-21** (8 weeks) before switching to arm B~~ —
      **ended early, 2026-09-08, per explicit user request** (43 days
      elapsed of the planned 56; 13 days short). Not a data/incident-driven
      stop — logged here so the shorter-than-planned arm A window is a
      documented, deliberate deviation, not silently missing context for
      whoever reads this log next.
- [x] Export NAV curve from `data/live_state.db` at arm end — see final
      snapshot below.
- [ ] Record per-analyst attribution vs arm B (not done at cutover; can be
      pulled from `PerformanceAttribution` state if needed later)

### Final snapshot at arm A end (2026-09-08 05:58 UTC, via `scripts/snapshot_llm_ab_arm.py`)

- Portfolio snapshots: 32
- Latest NAV: 950221.67 (started ~998984.02 on 2026-07-27 — approx -4.9%
  over the arm)
- Ann. Sharpe (daily): -5.625
- Max drawdown: 0.0488
- Caveat: consistent with this project's own repeated walk-forward/PBO
  findings (`docs/remediation_progress.md` #58-67) that the current
  12-strategy/25-name combination layer has not cleared a profitability
  gate in six independent attempts — this arm's negative live Sharpe should
  be read in that context, not necessarily as evidence against quant-only
  analysts specifically vs. arm B.

## Arm B — llm_enhanced

| Field | Value |
|-------|-------|
| **Started** | 2026-09-08 (IDT), ~05:57 UTC |
| **`FIRM_LLM_CONFIG`** | `config/llm_ab_llm.yaml` |
| **Live config** | `config/live.yaml` (unchanged) |
| **Started early** | Yes — at explicit user request, 13 days before arm A's planned 8-week end date. See arm A's "End criteria / notes" above. |

### Baseline at arm start (post-restart snapshot, 2026-09-08 08:57 IDT)

- Both `ai-trading.service` (IBKR paper) and `ai-trading-alpaca.service`
  (Alpaca paper) restarted together, same arm, to keep the broker comparison
  unconfounded (per `ai-trading-alpaca.service`'s own config comment).
- Engine state: `running`, broker: `ibkr_paper` / `alpaca_paper`, both connected
- Active strategies: 10 (momentum … regime_hmm) on both — unchanged from arm A
- Next scheduled cycle: 2026-09-08 09:30 ET (market open)
- No errors in either service's startup logs; clean restart.

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
