# LLM policy

Status: P5-05, 2026-10. Documentation only: no config, unit or `src/` file changed. Owner decision OD-03 (signed 2026-10-03): IBKR :8000 and LLM arm B stay running, untouched, as the control.

## 1. Scope and who is bound

Bound: all research sessions, and any strategy managed by `firm.lifecycle`. **Grandfathered** until OD-03 is acted on (via P0-07, not built): the IBKR 11-strategy pipeline and LLM arm B. AGENTS.md rules 8/10 (P0-04) are scoped the same way.

### 1b. Permitted research-tooling uses

- Literature summarisation into charters, with human verification of every cited claim.
- Code review.
- Anomaly explanations for alerts.
- Drafting preregistrations for human approval.

Constraint: an agent must not write a charter's mechanism section (consistent with P5-02).

## 2. Facts as they are today

Checked against `deploy/*.service` and `config/llm_ab_llm.yaml` at the time of writing; the owner should re-verify with `grep FIRM_LLM_CONFIG deploy/*.service`.

- Both units (`deploy/ai-trading.service` IBKR, `deploy/ai-trading-alpaca.service`) set `FIRM_LLM_CONFIG=config/llm_ab_llm.yaml` (arm B, `llm_enhanced`, since 2026-09-08). `config/llm_ab_quant.yaml` was arm A (ended 2026-09-08).
- Agent modes (`config/llm.yaml`): `fundamental_analyst` and `sentiment_analyst` are `llm_enhanced`; technical, bull, bear, debate, trader and risk are `quant`. `config/live.yaml` sets `llm_open_close_only: true`.
- Models: default `groq/openai/gpt-oss-120b` with a load-balanced fallback chain (`gemini/gemini-3.7-flash`, `groq/qwen/qwen3.8-27b` (Preview tier) and several OpenRouter `:free` models ending in `openrouter/openrouter/free`, an auto-router over whatever free model is live). Models are therefore **not pinned** to an exact version, and which model answered a given call is not fixed.
- Anonymisation exists: `src/firm/agents/llm/news_anonymizer.py` strips company name and ticker (Glasserman & Lin 2023) from retrieved text, but it is applied to the RAG `news` collection only, not to other prompt inputs.
- The A/B (`docs/llm_ab_experiment_log.md`) is a forward test, consistent with section 3. Its NAV results are **not evidence of an LLM edge**: one window, arm A (quant) NAV about -4.9% (approx. 998,984 to 950,222, 2026-07-27 to 2026-09-08), no pre-registered gate.
- LLM layers are agents, not registered strategies; they carry no registry status. This section is their status record: arm B / LLM-enhanced analysts = RUNNING (grandfathered control); no other LLM component is part of any candidate.

## 3. Rules for any future LLM-derived signal

1. The prereg records the exact pinned model id and its date.
2. Entity anonymisation is mandatory for text inputs.
3. No historical backtest is accepted as evidence (training-data look-ahead and memorisation: Lopez-Lira & Tang 2023 and critiques).
4. Only a post-cutoff evaluation window counts, and it is recorded in the prereg (post-cutoff evaluation).
5. Forward paper test only, on a dedicated paper account.
6. Promotion requires passing the full G-RESEARCH and G-PAPER from the gates config; a forward paper test alone is not sufficient.
7. A model change resets the clock and counts as a new trial.

## 4. Correction to the source plan

"Pinned model" is **not satisfied** by the current config (unpinned, load-balanced fallback chain). This is recorded here, not fixed: arm B is grandfathered and its config is frozen.

## 5. Retirement path

P0-07 under OD-03 (deferred; the owner executes it when the IBKR paper account is needed or the LLM A/B ends).
