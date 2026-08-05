---
name: project-live-monitoring-and-security
description: "Real IBKR live-trade run + monitoring session (2026-07-21) — 6 bugs found/fixed, reverse proxy + firewall set up"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2ceda047-fc14-4536-ae62-6e8b77b7ca59
  modified: 2026-07-21T22:20:53.935Z
---

User asked to "run the auto trade now and monitor it, monitor every aspect." Ran a real cycle against the live IBKR paper account (12 strategies, 25 symbols) — it completed successfully, generated 20 real orders, correctly routed to manual approval by the daily-turnover cap (a first rebalance from all-cash needs ~100% turnover, over the 25% cap). Monitoring itself turned up 6 more real bugs, all fixed/tested/committed same session:

1. **34-hour silent hang**: a scheduled cycle from IB Gateway's mandatory daily-restart window hung forever with zero error/alert, blocking every cycle since. Added a watchdog (`cycle_watchdog_seconds`, default 30min) that fires a critical alert if a cycle runs abnormally long — purely observational, doesn't touch the lock (a resumed stuck thread could otherwise race a new cycle).
2. **Decision-log contamination**: several `test_live_engine.py` fixtures weren't isolated from the real `data/memory/decisions.jsonl` — a leaked test entry for today's date silently blocked a real decision from being recorded (same-day idempotency check in `store_decision()`). Fixed all fixtures to use `tmp_path`; wiped the real file (every entry in it turned out to be fake test noise going back to 2026-07-11).
3. **`/live/account` crashed 100% of the time**: `ib_async`'s `reqAccountSummary()` needs an asyncio event loop in the calling thread; FastAPI's anyio threadpool doesn't guarantee the same thread (or any loop) across requests. Fix: subscribe once in `connect()`, read the cache via `accountSummary()` thereafter (same safe pattern `get_positions()` already used). **Residual risk**: `get_current_price(s)`/`is_market_open()` use the same reqXXX pattern and could theoretically hit this too — not yet observed, only ever called within a single `run_cycle()` on one thread so far.
4. Plus the 3 from the prior session turn: Massive rate-limit retry storm, fake-zero-price bug (IBKR `last=0.0` sentinel), and `is_market_open()`'s UTC-vs-ET timezone bug.

**Security finding + fix**: the API was on `0.0.0.0:8000` with zero auth and already being scanned by internet bots (`/mcp`, `/sse` probes in the logs). Fixed:
- `firm-api` now binds `127.0.0.1` by default (`FIRM_API_HOST` env var to override; docker-compose sets it to `0.0.0.0` explicitly since the container network is the isolation boundary there).
- nginx reverse proxy on this host: self-signed 10-year cert (`/etc/nginx/ssl/ai-trading.{crt,key}`), HTTP Basic Auth (`/etc/nginx/.htpasswd`, user `admin`), HTTP→HTTPS redirect, config at `/etc/nginx/sites-available/ai-trading`.
- `ufw` enabled, default-deny incoming, only 22 (SSH) and 443 (HTTPS) open.
- Access going forward: `https://<vps-public-ip>/` (re-check the VPS's current public IP, don't trust a cached value blindly) with the `admin` credentials (password was shown to the user once in-chat and has since been changed by the user directly via `htpasswd` — check with them or a password manager rather than assuming any historical value; **redacted here on purpose, do not re-add a literal password to this file**).

**How to apply**: if `/live/*` endpoints seem unreachable next session, check (a) `systemctl status ai-trading` binds to loopback now — direct `curl localhost:8000` still works from *on the VPS*, only *external* access requires going through nginx; (b) `systemctl status nginx`; (c) `ufw status`. See [[project_ibkr_paper_trading_setup]] for the IBKR-specific setup this builds on, and [[project_reflection_and_gui_parity]] for the GUI/API work these bugs were found while exercising.

**One more real bug found live**: the user hit a blank page at `/live/config` — `GET /llm/rag/stats` returned a flat `{name: count}` dict (`VectorStore.stats()`'s real contract), but the frontend's `RAGStats` type/component assumed `{"collections": {name: {count, description}}}` (matching only the error-path fallback). `Object.keys(undefined)` threw with no error boundary, blanking the whole page. Fixed by reshaping at the API layer (`src/firm/api/routers/llm.py`), plus made `LiveConfig.tsx` defensive (`ragStats?.collections`) so a future regression degrades gracefully instead of crashing.

**Frontend test suite added** (previously zero coverage — directly how the above bug went undetected): Vitest + jsdom + React Testing Library + MSW, configured in `frontend/vite.config.ts`. Fixtures in `frontend/src/test/mockData.ts` are built from real curl'd API responses, not just the TS types — same discipline that would have caught the rag/stats mismatch. 56 tests across all pages + api/client.ts + shared components. Run via `npm test` inside `frontend/`. `onUnhandledRequest: 'error'` means any new API call a component makes without a corresponding MSW handler fails the test loudly — keep it that way, don't loosen it to silence failures.
