---
name: feedback-repo-doc-and-memory-upkeep
description: "User wants durable project knowledge (docs + agent memory) kept in the repo, not left stale or VPS-local"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d0819ea0-0b87-4dce-9dd6-90d93db8461e
  modified: 2026-09-10T11:46:57.950Z
---

User explicitly asked (2026-09-10) to refresh repo documentation and move
agent memory into the repo, noting some of it currently only lives on this
VPS. Clarified it's not just this session — the repo docs in general
haven't been updated in a long time (confirmed: `CLAUDE.md`/`AGENTS.md`/
`.cursor/rules/` untouched since 2026-07-23/24; `docs/PROJECT_CONTEXT.md`
since 2026-08-23 — ~7 weeks and ~2.5 weeks respectively of real changes not
reflected).

**Why:** VPS-only Claude memory is a durability risk (lost if this VPS is
ever lost/reset) and isn't visible to anyone reading the repo directly. A
prior session already established `docs/claude-memory/` (mirroring this
memory system's own file format) for exactly this reason, but it was only
ever synced once (2026-08-05) and had drifted badly out of date.

**How to apply:** don't let this drift happen again — when a project
memory here captures something durably true about the codebase (a real
architectural decision, a fixed bug with lasting relevance, a feature's
current status), also reflect it in the appropriate repo doc
(`docs/PROJECT_CONTEXT.md`, `CLAUDE.md`/`AGENTS.md`, `.cursor/rules/*.mdc`)
at the time, rather than only in VPS-local memory — and periodically check
`git log -1 -- <doc>` against how much has actually changed since, the way
staleness was caught this time. `docs/claude-memory/` should be kept in
sync with this memory system's `MEMORY.md`/files whenever either changes
meaningfully, not just when asked.
