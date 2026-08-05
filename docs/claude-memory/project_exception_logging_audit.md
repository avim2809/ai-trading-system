---
name: project-exception-logging-audit
description: Silent-exception-logging audit across the codebase (2026-07-19) — what was fixed and the critical /live/start bug found along the way
metadata: 
  node_type: memory
  type: project
  originSessionId: 2ceda047-fc14-4536-ae62-6e8b77b7ca59
  modified: 2026-07-19T06:28:45.876Z
---

Completed a full-codebase audit for the silent-exception-swallowing anti-pattern (`except Exception: return <default>` with no log), triggered by the user noticing `news_ingestor.py` did this. Fixed ~20 files: data providers (fred, fallback, ibkr, fmp, tiingo), RAG ingestors (sec, research, earnings, system), `rag/assistant.py`, `llm/compression.py`, `base_llm_agent.py`, the 8 `agents/llm/*_llm.py` enhancement wrappers (bumped debug→warning since prod runs at INFO level), `backtest/engine.py`'s 9 analyzer-extraction blocks, `api/routers/llm.py`, `runtime.py`, `live/data_feed.py`, and brokers `ibkr.py`/`alpaca.py`. Committed as `28d247c` and pushed to `origin/main`.

**Critical bug found (not just a logging gap):** `src/firm/api/routers/live.py`'s `/live/start` endpoint always built `LiveDataFeed(providers={}, ...)` — a permanently empty dict — regardless of which broker was selected. Every engine started via that documented HTTP endpoint ran with zero price/fundamentals/sentiment data every cycle, completely silently. My own earlier successful live-engine testing this session used `scripts/run_live_trading.py` directly (which builds its own IBKR-specific providers), so this bug was never exercised by that testing.

**Why:** discovered while tracing why `LiveDataFeed.refresh()`'s `if price_prov:` guards had no log for the "not configured" branch — traced the only production constructor and found it was always empty.

**How to apply:** Fixed by wiring `FallbackProvider()` (the existing Massive→Tiingo→AlphaVantage→FMP chain, see [[project_rag_subsystem_status]]) into `providers={"prices":..., "fundamentals":..., "sentiment":...}` at `live.py:139`, broker-agnostic. User explicitly chose "fix it now" over "just log it" via AskUserQuestion — confirms user wants real bugs found during audits fixed immediately, not just flagged, when the fix is well-scoped and low-risk. If `/live/start` is used again for paper/live trading, verify this fix is still in place and that FallbackProvider actually has usable keys configured (Massive/Tiingo/AV/FMP) — see [[project_ibkr_paper_trading_setup]] for current key/setup status.
