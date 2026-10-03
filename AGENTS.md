# AGENTS.md — Agent context and research-integrity rules

Pointers and rules for AI coding agents working in this repository. `CLAUDE.md` imports this file
(`@AGENTS.md`) and adds a Claude-specific tail; Cursor and other agents read this file directly.

## Pointers

| Resource | Purpose |
|----------|---------|
| [PLAN.md](PLAN.md) | Credibility plan (repo root): expectations, gates, phases, ticket index; tickets in `plan/tickets/`, owner decisions in `plan/OWNER_DECISIONS.md` |
| [docs/REPO_MAP.md](docs/REPO_MAP.md) | Verified map of the repo and the plan-to-actual path table; worktree and venv policy (section 12) |
| [docs/HOLDOUT_POLICY.md](docs/HOLDOUT_POLICY.md) | Forward-data seal: what is burned, what is sealed, every on-host post-seal location |
| [docs/PROJECT_CONTEXT.md](docs/PROJECT_CONTEXT.md) | Full architecture, deployment, live config, REST/UI wiring, IBKR pitfalls |
| [docs/README.md](docs/README.md) | Index of docs/: current state, verdicts, evaluation ledgers, research, prompts, archive |
| [.cursor/rules/](.cursor/rules/) | Concise agent memory (project context, live trading, strategies, IBKR, logging, frontend, research integrity) |
| [docs/claude-memory/](docs/claude-memory/) | Repo mirror of Claude's persistent session memory (`MEMORY.md` index + per-topic files) — keep in sync when it changes |
| [deploy/ai-trading.service](deploy/ai-trading.service) | Production systemd unit (`firm-api` + auto-start live) |
| [config/live.yaml](config/live.yaml) | Canonical live paper trading configuration (IBKR, `:8000`, blended 11-strategy pipeline, the control) |
| [config/live_alpaca.yaml](config/live_alpaca.yaml) | Second live paper instance (Alpaca, `:8001`): `strategy_mode: allocation`, 60/40 SPY/IEF core plus 8% BTC 4-week-trend satellite (the 92/8 book) |
| [config/research_freeze.yaml](config/research_freeze.yaml) | Seal date, burned-through date, research allow-list and deny-list (source of truth for rule 1) |

**Production on bare-metal:** two live instances run as separate systemd units/ports — `:8000` (IBKR, `config/live.yaml`) and `:8001`
(Alpaca, `config/live_alpaca.yaml`), each `firm-api` with `FIRM_AUTO_START_LIVE=1` and its own `FIRM_API_PORT`/`FIRM_DATA_DIR`/
`FIRM_LIVE_CONFIG`. Neither runs `scripts/run_live_trading.py` directly. Alpaca has run the allocation 92/8 forward test since 2026-09-30;
IBKR is the blended 11-strategy control.

## Session types (which rules bind which)

- **OPS session:** operating the two live paper instances (`:8000` IBKR blended control, `:8001` Alpaca allocation 92/8). Routine edits to
  `config/live*.yaml`, service restarts, and log/API inspection stay allowed as today, in a worktree or with services stopped, never by editing
  the running checkout in place. Rules 1-7 below do not restrict ops work; Rules 11-13 do. Emergency hotfix path: the owner commits and pushes
  directly to `main`; the integrity CI workflow then runs on that push.
- **RESEARCH session:** anything under `plan/tickets/`, `src/firm/{validation,research,signals,costs,risk,monitoring,lifecycle,reporting}`,
  `research/`, new `scripts/<family>_preregistered*.py`. Rules 1-13 all bind. Research sessions run as the non-root `research` unix user, never
  restart services, never edit `config/live*.yaml`, `config/llm_ab_*.yaml` or `deploy/`, and never push to `main`/`master`/tags (a root-owned
  local pre-push hook blocks it; they push branches only and the owner merges).

## Mission

Build validation infrastructure and a boring, diversified, low-turnover core. Do NOT search for new alpha. Do NOT optimise for backtest
performance. Failure (a Tier C/D verdict, the passive outcome) is a legitimate result.

## Hard rules (enforced by tooling where possible; violations fail CI)

1. **Holdout:** never read, list, grep, load, or infer anything from data after `config/research_freeze.yaml` `seal_date` (2026-10-01), or from
   the `deny_paths` in that file: `data/cache/`, `data/research/s2_forward/`, `data/forward_monitors/`, `data_alpaca/`,
   `research/monitoring_sealed/`, `docs/s2_forward_snapshot.json`, and the IBKR live state and logs under `data/` (`data/logs`,
   `data/live_state.db`, `data/cycle_history.json`, `data/order_history.json`, `data/execution_audit.jsonl`, `data/memory`,
   `data/llm_cache*.db`). Pre-seal research inputs (`data/research/eodhd`, `data/research/fred`, `data/research/insider`) ARE readable, only
   through `firm.research.data_access`. The list is identical to `config/research_freeze.yaml`. If a task seems to need anything else, STOP and
   ask the human.
2. **Ledger:** every research backtest runs through `firm.research` `backtest_logged` (P1-01, not built yet). Never call engine internals
   directly. Never delete or modify ledger rows (`/local/store/research-ledger/trials.jsonl`). Exploratory runs are logged and counted.
3. **Pre-registration:** no backtest without an approved pre-registration whose hash matches the config, except `mode="exploratory"` (counted,
   never promotable). Frozen `scripts/*_preregistered*.py` are never edited; `docs/*_trial_history.json` are append-only.
4. **Grid only:** parameters may be chosen only from the pre-registered grid. No re-running with "one more variant": that is a new
   pre-registration.
5. **Protected files:** never edit files under `tests/integrity/`, `tests/test_live_import_isolation.py`, `config/gates.yaml`,
   `config/research_freeze.yaml`, `research/preregistration/`, `research/charters/`, `research/approvals/`, `.claude/`,
   `deploy/claude-managed-settings.json`, `deploy/claude-research-user-settings.json`, `deploy/claude-research-hooks/`, `.github/CODEOWNERS`, or
   this file. If a test seems wrong, report it; do not change it.
6. **No test gaming:** implement general solutions; never special-case test inputs, hard-code expected values, or weaken assertions.
7. **Seeds:** every stochastic routine takes an explicit seed; record it in the ledger.
8. **Frozen strategies:** do not change parameters of a strategy managed by `firm.lifecycle` while it is in EMBARGO, PAPER or LIVE. (The legacy
   11-strategy IBKR pipeline is grandfathered until OD-03; its routine config edits are ops work.)
9. **No live orders:** never place live orders. Paper orders only through the existing services; `FIRM_ALLOW_TRADING` stays unset in agent
   sessions.
10. **LLMs are research tools only:** no LLM output enters the signal path of a `firm.lifecycle` strategy (`docs/LLM_POLICY.md`, P5-05). LLM arm
    B on IBKR (`config/llm_ab_llm.yaml`) is grandfathered until OD-03.
11. **Never edit the live checkout in place.** Work in the research clone `/local/store/research/ai-trading-system`, in its
    `.claude/worktrees/<TICKET>` (`scripts/new_research_worktree.sh`), test with the worktree's own interpreter (never the main `.venv`, whose
    editable install points at the live `src`), and never pip install into the live `.venv`.
12. **LIVE-IMPORT-PATH tickets** follow the protocol in PLAN.md section 8 (self-contained, full suite, `scripts/live_import_smoke.py`,
    acknowledged merge window, verified after restart). No other ticket may import `firm.research` etc. from `firm.live`, `firm.api` or
    `firm.runtime` (`tests/test_live_import_isolation.py`).
13. **Timestamps:** for preregistrations, approvals and seal events, take them from `date -u` or git, never from log lines (the host clock is
    Asia/Jerusalem).

## Rules of thumb (carried over from CLAUDE.md)

- **Logging/traceability**: every module uses stdlib `logging` (`log = logging.getLogger(__name__)`); log decisions, fallbacks, external-I/O
  outcomes, and safety events (execution-gate blocks, news-guard, kill-switch). No `print`, no bare `except: pass`. See
  `.cursor/rules/logging.mdc`.
- Edit `config/live.yaml` for live universe/risk/strategies/params (include `risk.sector_map` for sector caps); use `resolve_live_startup()` in
  `src/firm/live/provider_utils.py` — do not duplicate YAML merge logic.
- Backtests with **Cache** data source load prices + fundamentals from `data/cache` (not live API).
- On a running engine, change behavioural knobs via engine setters (`update_news_guard` / `update_signal_combination` / `update_allocation`).
- Never call `IBKRBroker.connect()` from uvicorn's asyncio loop — use a sync handler thread or `asyncio.to_thread()`.
- **Frontend is mobile-first responsive** (usable at ~375px): grids start at `grid-cols-1/2` and scale with `md:`/`lg:`; header/action rows wrap
  (`flex flex-wrap … gap`, `flex-shrink-0`, `min-w-0`); tables `overflow-x-auto`. See `.cursor/rules/frontend.mdc`.
- Keep backend endpoints and `frontend/src/api/{types,client}.ts` in sync when extending features.

## Workflow per ticket

- Read `plan/tickets/<ID>.md` and `docs/REPO_MAP.md`. Plan first. List the files you will touch.
- Write failing tests first; confirm they fail; implement; run the suite under `nice -n 10 ionice -c3` (xdist `-n 2` at most, research venv only).
- Report: what changed, test output, deviations, open questions. Stop at every "Human checkpoint".
- At most 3 research agents at once on this host.

## Definition of done (all tickets)

Tests added and passing; ruff clean; no ledger-bypassing path; no reads of sealed data; docs updated (`docs/README.md`, `REPO_MAP.md`, ticket
status).
