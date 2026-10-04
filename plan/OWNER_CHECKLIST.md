# Owner checklist: everything still pending from you

Updated 2026-10-04. All 20 owner decisions (OD-01 to OD-20) are recorded in `plan/OWNER_DECISIONS.md`. What remains is action,
not decisions. Items are in the order that unblocks the most. Commands run as root on the host.

## A. Do now (each takes minutes)

| # | What | How | Why it matters |
|---|---|---|---|
| A1 | **Install the guardrails (stage 7)** | `/root/setup_research_isolation.sh --guardrails /local/store/git/ai-trading-system/deploy` then `/root/setup_research_isolation.sh --verify`. First make the research venv exist (see A2) or the research Stop hook will block turns. | Turns on the Claude Code deny rules, hooks and sandbox for the `research` user, and the host-wide edit-denies on protected files. An agent was blocked from running it (it changes permission settings), so it must be you. |
| A2 | **Create the shared research venv** | `cd /local/store/research/ai-trading-system && runuser -u research -- scripts/new_research_worktree.sh VENV-INIT core --create-venv` (usage: `<TICKET_ID> [dependency-set] [--create-venv]`; it creates a worktree and builds a non-root, non-editable venv at `/local/store/research-venvs/core`; read the script header first, and delete the throwaway `VENV-INIT` worktree afterwards). | The research Stop hook refuses the live `.venv` and fails closed without a safe interpreter. |
| A3 | **Write the charter mechanism** (about 15 minutes) | Copy `plan/drafts/P5-02/charter_core_v1_DRAFT.md` to `research/charters/core_v1.md`, write the three short mechanism paragraphs in your own words, commit it. | Gate 8 needs a human-written mechanism whose git timestamp precedes the first `core_v1` ledger row. This is the one gate no agent can satisfy. |
| A4 | **Review the draft ETF universe and costs** | Open `config/universe_etf.yaml` (15 ETFs, 10 cells, each with an inclusion rationale) and `config/costs.yaml` (5 fee entries all marked `verified: false`). Check each ETF is something you would hold and that each fee matches your broker's current schedule; change `verified` to `true` when it does. | P2-01 says the universe must be chosen without looking at performance; the fees feed every cost-stress result. |

## B. Allocation monitor timer (OD-15: yes, after a dry run; you install it)

The monitor is read-only: it GETs `127.0.0.1:8001` over loopback and sends alerts. It never halts or trades.

1. Create the user and credentials file (root-owned, mode 0600):
   ```bash
   useradd --system --no-create-home allocmon
   install -d -m 0755 /etc/allocation-forward-monitor
   install -m 0600 /dev/null /etc/allocation-forward-monitor/credentials.env
   # then edit it to contain exactly: TIINGO_API_KEY=...  FRED_API_KEY=...  ALERT_WEBHOOK_URL=...
   install -d -o allocmon -m 0755 /local/store/git/ai-trading-system/data/forward_monitors/allocation
   ```
2. Dry run as the monitor user, no timer yet:
   ```bash
   cd /local/store/git/ai-trading-system
   sudo -u allocmon env $(cat /etc/allocation-forward-monitor/credentials.env | xargs) PYTHONPATH=src \
     .venv/bin/python3 scripts/allocation_forward_monitor.py validate
   sudo -u allocmon env $(cat /etc/allocation-forward-monitor/credentials.env | xargs) PYTHONPATH=src \
     .venv/bin/python3 scripts/allocation_forward_monitor.py run
   ```
   Expect `validate` to pass (the `live_nav` rule entry is already recorded in `docs/allocation_forward_test_trial_history.json`).
3. Only then install: copy `deploy/allocation-forward-monitor.{service,timer}` to `/etc/systemd/system/`,
   `systemctl daemon-reload`, `systemctl enable --now allocation-forward-monitor.timer`.

## C. When the autopilot finishes (`/local/store/research/reports/AUTOPILOT.md`)

| # | What | How |
|---|---|---|
| C1 | **Review and merge `integ/auto`** | Read the summary; if the suite is green and you are happy, fast-forward `main` to `origin/integ/auto` in a quiet window (it adds new modules only; run `scripts/live_import_smoke.py` afterwards). Nothing restarts. |
| C2 | **Fold in the P1-08 scenario values** | Wave 5 job `w5a` commits its two deferred values (cross-correlation, seed) in `plan/drafts/P1-08/scenario_freeze.yaml`. They belong in `config/gates.yaml`, which is hash-pinned, so this is a new signed register version: re-run `scripts/gates_hash.py --record` flow after reviewing. |
| C3 | **Install the pre-registration index** | `w5b` drafts `research/preregistration/INDEX.yaml` entries under `plan/drafts/P1-09/`; `research/preregistration/` is owner-protected, so you copy them in. |

## D. Housekeeping

- **GitHub reports 22 dependency vulnerabilities** on the default branch (2 critical, 6 high). See https://github.com/avim2809/ai-trading-system/security/dependabot. Unrelated to this plan, but worth a pass.
- **S2 forward-test timer** (`deploy/s2-forward-shadow.timer`): still your separate pending install decision from before this plan.
- **Tax adviser** (OD-12): an Israeli tax professional's answer on estate tax, withholding and reporting is required before any live step. This plan treats tax figures as information only.

## E. Blocked until A3 (and then agents continue)

Charter approval and `tau` confirmation (about 9%, working value) unblock P3-11 (estimating the constants) and then P3-08, the
real research run that produces the Tier A/C/D verdict. At capital under $25k the expected outcome is the passive portfolio.
