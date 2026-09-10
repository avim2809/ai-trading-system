---
name: project-live-pause-not-persistent
description: Stopping a live engine via /api/live/stop does not survive a service restart — FIRM_AUTO_START_LIVE=1 always auto-resumes from config/live.yaml on boot
metadata: 
  node_type: memory
  type: project
  originSessionId: ff1d5776-266d-4d49-812e-1d60eb6fc60c
  modified: 2026-08-29T22:07:53.769Z
---

Calling `/api/live/stop` (or the UI equivalent) to pause `ai-trading.service` /
`ai-trading-alpaca.service` is **not a durable state** — it only stops the
in-memory engine of the current process. Both systemd units set
`FIRM_AUTO_START_LIVE=1`, so the very next service restart (a deploy, a
crash, a host reboot, `wait_for_ibgateway.sh` retry) calls
`bootstrap_live_from_yaml()` again and silently resumes live trading from
`config/live.yaml` / `config/live_alpaca.yaml`, full_auto, no confirmation.

**Why this matters:** discovered 2026-08-30 — trade history showed live
activity resumed 2026-08-24/25, which looked at first (from auto-memory
alone) like it might have silently overridden the 2026-08-23 session's
"keep both engines paused" decision ([[project_joint_optimizer_redesign_aug23]]).
Correction after reading `docs/remediation_progress.md` directly: this
*was* a deliberate resume, done "by explicit user request mid-session"
during the 2026-08-24/25 PART 3 investigation — it just wasn't captured in
auto-memory, only in the repo's own docs. The general risk this memory
describes (a restart silently re-enabling live trading regardless of
intent) is still real and worth knowing — it just didn't actually fire
that particular time. Lesson: when auto-memory and repo docs
(`docs/remediation_progress.md`, `docs/formal_pbo_audit.md`) disagree or
one is silent, check the repo docs — they're the more complete record for
this project's research/ops history.

**How to apply:**
- Before restarting either service for *any* reason (a deploy, a config
  change, a crash recovery), check whether the engine is supposed to be
  paused first — restarting will un-pause it regardless of intent.
- If a genuine "stay paused across restarts" state is ever needed, that
  requires either unsetting `FIRM_AUTO_START_LIVE` or adding a persisted
  halt flag `bootstrap_live_from_yaml()` checks before auto-starting —
  neither exists today. Flag this to the user rather than assuming one
  restart-safe pause mechanism already covers it.
- When auditing "is live trading paused or running," don't trust a stale
  memory note — always verify against `GET /api/live/status` on both ports
  (8000 IBKR, 8001 Alpaca), since a restart between sessions can flip it.

Related: [[project_joint_optimizer_redesign_aug23]], [[project_concurrent_sessions_aug21]].
