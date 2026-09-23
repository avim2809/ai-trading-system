---
name: feedback-exhaustive-audit-closure
description: "User got frustrated (2026-09-23) that repeated system checks kept surfacing one new issue at a time — do the full-depth sweep in one parallel pass, not incrementally"
metadata:
  node_type: memory
  type: feedback
  modified: 2026-09-23T21:56:28.811Z
  originSessionId: 403dab55-8f5b-43a3-abc3-7533df0c5692
---

When asked to "check the system end to end," don't stop at the first thing
found, fix it, and report success — then get asked again and find another
thing. That pattern ("every time i ask you to check the system and you fix
something, you find something else on another round, this has to
conclude!!!") reads as a real process failure, not just impatience: each
individual fix was fine, but reporting them one at a time made the session
feel endless and made the user distrust that "done" ever really meant done.

**How to apply**: when asked for a system health check (especially after
already finding and fixing one thing), do ONE genuinely exhaustive pass
across every plausible area before reporting back — don't report "fixed X"
and wait to be asked again before checking Y. Use parallel subagents
(Agent tool) to cover independent areas at once (ops/infra, scheduled
jobs/alerts, specific bug investigation, full test suite) rather than
serially discovering them. The user explicitly authorized "use multiagents
to speed things up" for this — treat that as standing permission for this
kind of parallel sweep in this repo, not a one-off.

**Be honest about what "conclude" can mean, don't just promise it**: for a
live system with a real external dependency (IBKR's own data-farm
reliability), it is dishonest to promise zero issues will ever surface
again — say so plainly rather than over-committing. What's fair to promise:
this session's *own* bugs get found and closed in one pass, and an external
outage recurring later is clearly labeled as that, not reported as a new
"finding" needing the same alarm treatment. See
[[project_reconciliation_audit_sep23]] for the specific audit this lesson
came from — all 4 real issues found were eventually genuinely closed
(including one whose first fix attempt made things worse and had to be
caught, rolled back, and corrected before it actually worked) — see
[[feedback_verify_before_trusting_a_heuristic]] for that specific lesson.

Related: [[feedback_autonomous_scope_calls]], [[feedback_production_incident_priority]],
[[feedback_verify_before_trusting_a_heuristic]].
