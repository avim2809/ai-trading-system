---
name: feedback-never-edit-live-checkout
description: "Both live services run from /local/store/git/ai-trading-system itself; uncommitted edits there get lazily imported mid-run. Do dev work in a git worktree, or only while the services are stopped"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 7e68870f-025c-467b-8177-1f4914055172
  modified: 2026-09-28T12:45:19.853Z
---

On 2026-09-28 my uncommitted edits in the repo checkout broke a live cycle. IBKR
cycle 109, the 08:06 ET premarket cycle, failed with an ImportError:
`_combine.py` asked for `combine_signals_optimal_robust`.

Why it broke: both `firm-api` services import from this same checkout. The
running process already held the OLD `firm.agents.analysts` in memory. It then
lazily imported my NEW `firm.agents.research._combine` from disk; `bull.py` does
a function-level import, and this was the first cycle since the Sunday restart.
The result was a mixed-version process. "Off by default" doesn't protect you
here: the breakage came from import-time version skew, not from the feature
itself.

A second hazard of the same kind: a service restart, including an unattended one, would
load half-finished uncommitted code.

**How to apply:**
- Never modify tracked `src/` files in `/local/store/git/ai-trading-system` while
  either service (`ai-trading`, `ai-trading-alpaca`) is active. Do dev and
  experiment work in a `git worktree` under the scratchpad instead.
- Only commit or fast-forward the live checkout once the full suite is green.
- Restart only with the user's go-ahead.
- If an edit is already in place: `git stash -u` to restore HEAD. A failed
  import is not cached, so the next cycle retries cleanly.
- The user is fine with stopping both paper services for a whole work session
  ("start the service only when this work is completely finished",
  2026-09-28). Stopping them is a legitimate alternative to a worktree.

Related: [[project_optimal_combination_lockout_sep28]], [[feedback_production_incident_priority]].
