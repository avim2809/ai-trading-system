---
name: project-groq-model-deprecation-fix
description: Groq deprecated llama-3.3-70b-versatile (2026-06-17); was silently degrading every live LLM-enhancement call to a last-resort OpenRouter fallback until fixed 2026-08-30
metadata: 
  node_type: memory
  type: project
  originSessionId: ff1d5776-266d-4d49-812e-1d60eb6fc60c
  modified: 2026-08-29T21:03:27.517Z
---

Found during a 2026-08-30 status check: `groq/llama-3.3-70b-versatile` — the
primary model in `config/llm.yaml`, `llm_ab_quant.yaml`, `llm_ab_llm.yaml`,
plus the hardcoded defaults in `src/firm/llm/{provider.py,config.py,
providers.py}` — was fully decommissioned by Groq for free/developer-tier
keys on 2026-06-17. Every enhancement call had been failing at the primary
hop (`model_not_found`) since at least then. Both running services
(`ai-trading.service`, `ai-trading-alpaca.service`) actually load
`config/llm_ab_quant.yaml` via `FIRM_LLM_CONFIG` — its fallback chain is
short (`gemini/gemini-3.7-flash` → `openrouter/openrouter/free`), and Gemini
was also intermittently `ServiceUnavailableError`, so most calls were
quietly landing on the last-resort free OpenRouter model instead of the
intended primary. Non-fatal (caught, falls back to the quant-only score)
but a real, silent quality degradation running for ~2+ months.

**Why:** the failure is graceful (`except Exception` around the LLM call in
e.g. `sentiment_analyst_llm.py`), so nothing paged or broke a cycle — it
would never have surfaced without reading the actual journalctl output
during a routine status check.

**How to apply:**
- Fixed: `default_model` → `groq/openai/gpt-oss-120b` everywhere above
  (Groq's own recommended successor — confirmed live 2026-08-30 via a
  direct `litellm.completion()` call, including a realistic `json_mode`
  enhancement-shaped prompt). `groq/llama-3.1-8b-instant`, listed as a
  fallback in the old provider registry, is *also* dead now — don't reuse it.
  `groq/openai/gpt-oss-20b` and `groq/qwen/qwen3.6-27b` were verified live
  too, as lighter alternatives if gpt-oss-120b ever needs replacing.
- gpt-oss/qwen3.6 are reasoning models — they spend part of the token
  budget on hidden reasoning before the visible answer. Confirmed
  `max_tokens: 2000` (the existing config value) is enough for clean JSON
  output in a realistic test call; don't drop `max_tokens` below that
  without re-verifying JSON completion isn't truncated.
- Both services were restarted 2026-08-30 to pick up the new config
  (LLMService reads `default_model` once at construction, no hot-reload
  path) — came back up clean, both brokers connected, no alerts.
- General lesson: Groq's free-tier model roster rotates unannounced (this
  is the second such deprecation found, after `llama-3.1-8b-instant` in a
  prior session) — when a Groq/Gemini/OpenRouter model_not_found error shows
  up in logs, verify the replacement empirically against the real API
  (`litellm.completion(...)`) rather than trusting a web search or docs
  page, since docs can lag actual availability by months either direction.

Related: [[project_live_pause_not_persistent]], [[feedback_production_incident_priority]].
