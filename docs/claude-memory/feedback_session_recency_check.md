---
name: feedback-session-recency-check
description: "When asked to check state 'since our last session', verify against git log timestamps, not memory's last-written date"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: cd4b929f-9803-4fd1-ab03-9b8800a53d3d
  modified: 2026-09-08T20:13:25.758Z
---

Memory's most recent entry is not the same thing as "our last session" — memory can lag behind real sessions that happened but weren't yet written up.

**Why:** On 2026-09-08, asked to "check system state since our last work," I read MEMORY.md's newest entry (8/30 Groq fix) and reported everything since as one undifferentiated blob "since 8/30." In fact `git log` showed a distinct session that morning (9/8 09:23, 5 commits) that was the real last session — memory just hadn't caught up on it yet. The user caught this ("i meant you wrote our last session was in 30/8").

**How to apply:** before reporting "since our last session," run `git log --format='%h %ci %s'` and look for commit-timestamp clusters, not just the newest memory file's date. Treat the most recent cluster as the actual last session and call it out distinctly, even if memory hasn't recorded it — then backfill that memory gap as part of the check-in (see [[project_llm_ab_experiment]] for an example of catching up a session memory retroactively).
