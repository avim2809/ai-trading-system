---
name: feedback-research-sessions-scope
description: Research sessions (credibility plan) never restart services or edit live config, whatever "run on your own" grants; they run as the research unix user in a separate clone
metadata:
  node_type: memory
  type: feedback
---

A RESEARCH session (anything under plan/tickets, `src/firm/{validation,research,signals,costs,risk,monitoring,lifecycle,reporting}`,
`research/`, new `scripts/<family>_preregistered*.py`) runs as the non-root `research` unix user in `/local/store/research/ai-trading-system`
(one worktree per ticket under `.claude/worktrees/<ID>`), never restarts or stops services, never edits `config/live*.yaml`,
`config/llm_ab_*.yaml` or `deploy/*.service`, never reads the sealed post-seal paths in `config/research_freeze.yaml`, and never pushes to
`main`/`master`/tags (it pushes branches; the owner merges).

**Why:** the credibility plan (PLAN.md, OD-04/OD-07) needs an enforceable seal on forward data and an audit trail. The older memory entries
[[feedback-autonomous-session-workflow]] ("run on your own" grants no-questions plus git push) and [[feedback-autonomous-scope-calls]]
(overnight builds, service restarts) were written for OPS work on the live paper instances and conflict with that. They still hold for OPS
sessions; they do NOT extend to research sessions.

**How to apply:** at the start of a session decide OPS or RESEARCH (AGENTS.md "Session types"). In a research session an autonomy grant
means "do not ask questions", not "restart services or touch live config". Ops sessions keep routine config edits and restarts, in a
worktree or with services stopped, never by editing the running checkout in place ([[feedback-never-edit-live-checkout]]).
