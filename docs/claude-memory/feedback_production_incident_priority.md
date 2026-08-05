---
name: feedback-production-incident-priority
description: "User wants live-trading production bugs fixed immediately once flagged as currently impacting the system, not scoped into a deferred follow-up"
metadata:
  node_type: memory
  type: feedback
  originSessionId: d187bdf3-58a0-49c4-ae8b-81294f53899d
  modified: 2026-08-05T19:29:25.848Z
---

When a bug is found that is *currently* affecting the live-trading system (not
a latent/theoretical risk, but something actively broken right now — e.g. a
hung broker call freezing positions/account/reconciliation), fix it in the
same session rather than proposing to scope it as a separate future task.

**Why:** after root-causing a stuck IBKR `qualifyContracts()` call that had
frozen the entire broker layer (2026-08-03/04, see
[[project_ibkr_paper_trading_setup]]), I suggested treating the real fix
(bounded request/lock timeouts) as a follow-up to pick up later rather than
implementing it immediately. The user's response was direct: "this needs to
be handled now." This is distinct from
[[feedback_autonomous_scope_calls]] (which is about *how much* new capability
to wire in) — this is about *urgency*: don't defer a fix for something that is
presently broken in production, even if it's a nontrivial, well-scoped piece
of work.

**How to apply:** when diagnosing a live incident, if the root cause has a
clear, boundable fix (not requiring a large redesign), implement + test +
commit + deploy it in the same session by default. Reserve "let's scope this
as a separate task" for genuinely large/risky architectural work, not for a
concrete, well-understood bug affecting the live system right now. Still ask
before the actual production restart/deploy step itself (that authorization
norm is unaffected) — the "handle it now" preference is about not stalling on
*starting* the fix, not about skipping deploy confirmation.
