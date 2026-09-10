---
name: project-reflection-and-llm-fallback-fixes-sep8
description: "9/8 evening session: fixed benchmark_return false-precision bug, dead LLM fallback model, added verified qwen3.8-27b"
metadata: 
  node_type: memory
  type: project
  originSessionId: cd4b929f-9803-4fd1-ab03-9b8800a53d3d
  modified: 2026-09-08T20:36:35.872Z
---

Three commits, same session, both services restarted+verified after each: `983c5a2`, `6405749`, `5322009`.

**1. `benchmark_return` still broken after this morning's own fix (983c5a2).** This morning's [[project_llm_ab_experiment]]-adjacent commit (7abdbe9) replaced a hardcoded `benchmark_return=0.0` in live reflection with a real PIT-panel lookup — but it silently collapsed back to an exact 0.0 almost every time anyway. Root cause: `LiveDataFeed`'s `exclude_forming_bar` drops *today's* bar (required for IBKR — quotes can't be pulled off the worker thread), so the panel's "most recent" row is really the last *completed* session. `_maybe_reflect` fires on the very next cycle after a decision with **no minimum-age gating at all** (`find_all_pending()` has no date filter) — so reflection routinely runs before a new session has closed, making start-price and end-price land on the identical row. Verified against real IBKR SPY data reproducing both of that day's actual production reflections exactly. Fixed with a guard requiring the end-price row to be strictly after the decision date (falls back to the same 0.0, now honestly labeled via debug log instead of masquerading as computed alpha). Added a regression test; 133/133 pass.

**Important unresolved caveat — told to the user, not yet addressed:** because reflection has no minimum-age gating, the fixed guard will likely still fire on *most* reflections under the current cadence — meaning real nonzero alpha decomposition may rarely materialize until either (a) reflection is deferred until a full session has closed, or (b) a genuinely live/intraday benchmark quote is wired in (independent of the IBKR-thread constraint, e.g. via an existing REST provider like twelvedata). This is a real design gap, not just a bug — flag it if asked to revisit reflection/alpha quality again.

**2. Dead LLM fallback model removed (6405749).** `openrouter/openai/gpt-oss-20b:free` confirmed dead (404, graduated to paid-only) in both `config/llm.yaml` and `config/llm_ab_llm.yaml` — same shape as the [[project_groq_model_deprecation_fix]] incident: with `load_balance`'s deterministic-hash routing, a dead entry isn't inert, some fraction of calls hash directly onto it as primary pick, forcing the whole remaining chain to be walked every time. Was happening on every single cycle, both live instances, confirmed in logs.

**3. Verified-live model research + addition (5322009).** User has a **paid Google API tier 1 with ~100 NIS (~$27) credit** — worth remembering when discussing Gemini cost/quality tradeoffs. Researched current (Sep 2026) free-tier reasoning-model landscape on Groq + OpenRouter via WebSearch/WebFetch, then live-tested candidates before touching config (established project convention). Findings:
- `groq/qwen/qwen3.8-27b` — verified live, clean JSON, ~0.5s, genuinely newer than `gpt-oss-120b`. Added as a **fallback**, not default — it's on Groq's "Preview" tier (docs.groq.com/models), which can be pulled with less notice than a full deprecation.
- `groq/qwen/qwen3.6-27b` — rejected: returns raw unstripped `<think>...</think>` tags inline in content, would break `chat_json`/`parse_llm_response`.
- `minimaxai/minimax-m2.7` (listed on Groq's own docs page) — rejected: 404 live, docs/registry mismatch.
- Discovered `gemini-3.7-flash` already runs hybrid reasoning **by default even when not requested** (`reasoning_tokens` showed up unprompted in a real API response, e.g. 275 of 296 completion tokens on one test call) — real cost-relevant fact given the limited credit budget. Explicit `reasoning_effort="low"` cut reasoning tokens roughly in half (275→117) while still producing a clean, correct answer. Not yet wired into the codebase (would need checking whether `firm/llm/provider.py`'s completion call forwards extra kwargs) — a real, cheap lever if the user wants tighter cost control on that key, worth raising if Gemini/cost comes up again.
- No OpenRouter free-tier model found that clearly beats what's already in the fallback list for reasoning quality; `openrouter/google/gemma-4-31b-it:free` no longer appears on OpenRouter's current official free-models collection page (may be rotating out) but still returns 429 rather than 404 live, so left in place rather than removed on weak evidence — worth rechecking if it starts hard-failing.
