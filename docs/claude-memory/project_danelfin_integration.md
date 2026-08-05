---
name: project-danelfin-integration
description: "Danelfin API integration status — ai_scores strategy (enabled, A/B'd), live_signals strategy (enabled, unvalidated), and a separate Best-Stocks synthetic paper arm"
metadata: 
  node_type: memory
  type: project
  originSessionId: d187bdf3-58a0-49c4-ae8b-81294f53899d
  modified: 2026-07-31T07:53:53.215Z
---

User subscribed to Danelfin (Expert plan, $149/mo, 10,000 calls/mo, 120/min)
as a real-API replacement for Investing.com Pro's per-stock data (which
turned out Cloudflare-blocked and unreachable — see
[[project_investing_pro_scraper]] if it exists, otherwise
`docs/investing_pro_integration.md`). Base URL `https://apirest.danelfin.com`,
header auth `x-api-key`. As of 2026-07-31 three things are live:

1. **`danelfin_ai_score` strategy** — backed by `GET /ranking` (genuine daily
   history back to ~2016, undocumented `page=N` pagination). A/B'd across
   the standard 3 windows, consistently positive at the portfolio level —
   enabled in `config/live.yaml`.
2. **`danelfin_live_signals` strategy** — backed by `/v3/trading-parameters`
   + `/v3/price-forecast` + `/v3/performance` (all latest-snapshot-only, no
   historical dates — **structurally unbacktestable**, `pit_view
   .live_signals()` is always empty in a backtest). Enabled anyway, per the
   user's explicit request, documented plainly as unvalidated (no A/B
   possible). Real field names were verified live (not guessed) —
   `stop_loss_pct`/`take_profit_pct` are percentage points,
   `median_3m`/`q05_3m`/`q95_3m` are 0-1 decimals — a real unit mismatch
   between the two endpoints.
3. **Danelfin "Best Stocks Strategy"** — a **separate, synthetic
   paper-tracking arm** (NOT inside the main engine), implementing
   Danelfin's own published methodology (sector-ranked, 25-stock
   equal-weight, quarterly/annual rebalance) via `/v3/trade-ideas`. Runs as
   a JSON-persisted mark-to-market ledger (`data/best_stocks_ledger.json`,
   gitignored) on a daily systemd timer (`best-stocks-arm.timer`,
   `OnCalendar=*-*-* 22:00:00 UTC` — explicit UTC matters, host is
   `Asia/Jerusalem`). Deliberately **not** a real broker-connected second
   engine — see the "no concurrent multi-arm infra" note below. Full detail
   in `docs/danelfin_best_stocks_arm.md`.

**Key finding: this project's "LLM A/B" experiment is sequential, not
concurrent.** One `LiveTradingEngine` singleton per `firm-api` process, one
IBKR paper account, one `data/live_state.db` (no experiment-name column —
`INSERT OR REPLACE` on 3 fixed keys). Switching arms = editing
`FIRM_LLM_CONFIG` + `systemctl restart ai-trading.service`. There is **no
existing infrastructure for running two live-trading arms at once** — that
would need a second IBKR client ID, a second systemd process, and a
separate state-store file. Don't assume multi-arm support exists; it
doesn't, as of this date.

**Real bugs found by verifying live before building** (a recurring theme
this session): `/v3/trade-ideas`'s actual response shape
(`{date: {symbol: {...}}}` with sibling `total`/`limit`/`offset` int keys,
not the originally-guessed `{"items": [...]}`) silently broke
`get_trade_ideas` for months until exercised; `get_live_signals` originally
always queried `/v3/performance` with `signal="buy"` regardless of the
actual trading-parameters call. Both fixed. **Lesson: always make one real
live call to verify a third-party endpoint's field names/shape before
writing a parser against it — Danelfin's own docs describe endpoints only
at a high level, not the field-level shape.**

See also [[feedback_autonomous_scope_calls]].
