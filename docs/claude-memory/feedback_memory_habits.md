---
name: feedback-memory-habits
description: "User wants proactive, thorough memory-keeping of technical work in this repo"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 2ceda047-fc14-4536-ae62-6e8b77b7ca59
  modified: 2026-07-18T19:32:31.961Z
---

Keep memory updated proactively as consequential technical work happens in this repo, not just when explicitly asked.

**Why:** in one session the user asked twice ("save all your knowledge in the repo memory", then later "save everything you do to memory") to persist findings — a clear signal they view the memory system as the durable record of infra/debugging work here, not just conversation-scoped detail.

**How to apply:** after any nontrivial fix, bug found via testing, or environment-state discovery (dependency swaps, config gaps, credentials status, scripts rewritten) in this repo, write/update the relevant memory file before considering the task done — don't wait for an explicit "remember this." Keep entries factual and dated so future sessions can tell what's still true vs. superseded. See [[ibkr-paper-trading-setup]] for the running example of this — it's been updated multiple times in one session as new facts (bugs found, fixes applied, live verification results) came in, rather than left as a stale one-shot snapshot.
