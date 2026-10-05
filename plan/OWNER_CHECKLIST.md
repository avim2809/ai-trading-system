# Owner checklist: what is still pending from you

Updated 2026-10-05. All 20 owner decisions are recorded in `plan/OWNER_DECISIONS.md`. The gates are signed and frozen
(`config/gates.yaml`, hash recorded). Stage 7 guardrails and the research venv are installed. Everything below is action, in
order of what unblocks the most. Commands run as root on the host.

## A. Needed for the research verdict (the long pole)

| # | What | How |
|---|---|---|
| A1 | **Write and commit the charter** (about 15 minutes) | Copy `plan/drafts/P5-02/charter_core_v1_DRAFT.md` to `research/charters/core_v1.md`. The mechanism, falsification and the three "agent draft" paragraphs are drafted; rewrite them in your own words and delete the "AGENT DRAFT" labels. Fill the blanks (expected worst year, longest flat period, correlation, turnover and cost). Confirm tau (working value 9%). Fill the approval block, commit. |
| A2 | **Approve the pre-registration** | Job `w12a` drafts `plan/drafts/P3-11/core_v1_prereg_DRAFT.yaml` (at most 12 configs, gates hash pinned). Review it, copy it to `research/preregistration/<date>_core_v1.yaml`, set `approved_by` and the date, commit. |
| A3 | **Tell me both are committed** | I then run P3-11 (constants) and P3-08 (the real pre-seal research run) and report the Tier A/C/D verdict. |

## B. Live services (tonight, after the US close, about 23:30 server time)

Waves 7 and 9 put live-path code on `main` (P0-03 registry statuses, P1-10, P1-12 trial capture in `engine.py`, `runtime.py`,
`api/app.py`, `registry.py`). The running services still use the code they loaded earlier; a restart activates it. All new imports
are guarded and the capture is fail-open. Restart one at a time, IBKR first:

```bash
systemctl restart ai-trading.service            # IBKR :8000
sleep 60; curl -s 127.0.0.1:8000/api/live/status | head -c 300
systemctl restart ai-trading-alpaca.service     # Alpaca :8001
sleep 60; curl -s 127.0.0.1:8001/api/live/status | head -c 300
```
Then tell me and I run `scripts/live_import_smoke.py`, check both status endpoints, the logs, and the next cycle. Rollback if anything
looks wrong: `git -C /local/store/git/ai-trading-system reset --hard 330a0e2` (the commit before the live-path waves) and restart again.

## C. Timers you install (read-only monitors; agents never install timers)

| What | Steps |
|---|---|
| **Allocation forward-test monitor** (OD-15, yes after a dry run) | 1) `useradd --system --no-create-home allocmon`; `install -d -m 0755 /etc/allocation-forward-monitor`; `install -m 0600 /dev/null /etc/allocation-forward-monitor/credentials.env` and put `TIINGO_API_KEY=`, `FRED_API_KEY=`, `ALERT_WEBHOOK_URL=` in it; `install -d -o allocmon /local/store/git/ai-trading-system/data/forward_monitors/allocation`. 2) Dry run as allocmon: `scripts/allocation_forward_monitor.py validate` then `run`. 3) Copy `deploy/allocation-forward-monitor.{service,timer}` to `/etc/systemd/system/`, `systemctl daemon-reload`, `systemctl enable --now allocation-forward-monitor.timer`. |
| **S2 shadow forward test** (pending since before this plan) | Paper ledger only, places no orders. First do one manual run: `cd /local/store/git/ai-trading-system && set -a && . ./.env && set +a && PYTHONPATH=src .venv/bin/python scripts/s2_forward_shadow.py run`. If it records the day, install: `cp deploy/s2-forward-shadow.{service,timer} /etc/systemd/system/ && systemctl daemon-reload && systemctl enable --now s2-forward-shadow.timer` (07:00 UTC daily). Note the service reads the repo `.env` (EODHD key); the monthly check pulls about 6,100 tickers and needs the EODHD subscription active, otherwise checks are logged as missed. |

## D. Optional

- **Bank of Israel USD/ILS series** (instead of the EODHD cross-check): add `data/research/il_macro` to `allow_roots` in `config/research_freeze.yaml`
  (protected: no agent or assistant may edit it) and to the matching `allow_roots` in `/etc/claude-code/research_freeze.deny.json`, update
  `tests/integrity` expectations if they pin the list, then fetch the series with the P2-02 script (it takes the source URLs or CSVs as
  arguments, so you supply the Bank of Israel link). Until then the after-tax FX uses EODHD USD/ILS.
- **Tax adviser (OD-12):** an Israeli tax professional's answer on estate tax, withholding and reporting is required before any live step.
- **Dependency vulnerabilities:** a branch bumping the flagged packages is being prepared by a research job; you review and merge.

## E. Decisions recorded 2026-10-05

- Kelly bound `tau <= 0.5 x Kelly vol` is evaluated once in the P3-08 report (deflated Sharpe with the 50% haircut); a breach is Tier D.
- tau stays 9% (working value); the 20% cap-days ceiling (gate 6) decides whether it is too high for the ETF path.
- Handcrafted weights use one group per asset class (equal across classes, equal within).
- The P1-08 harness result is accepted: PASS, with test size 0.070 against the 0.07 bar marginal.
- Real-data steps of P2-02, P2-03, P2-05 may be run by a research job (pre-seal, through the fail-closed loader only).
- Live services are restarted by you tonight after the US close (section B).
