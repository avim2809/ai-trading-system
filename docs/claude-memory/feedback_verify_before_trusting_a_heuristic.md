---
name: feedback-verify-before-trusting-a-heuristic
description: "2026-09-23: adopted a heuristic estimate as ground truth with no sum-invariant check, made live reconciliation drift worse; caught by verifying right after, fixed by adding an explicit constraint + adversarial review before retrying"
metadata:
  node_type: memory
  type: feedback
  modified: 2026-09-23T21:56:48.549Z
  originSessionId: 403dab55-8f5b-43a3-afa3-7533df0c5692
---

Built `seed_sleeves_from_attribution(force=True)` to fix a live Alpaca
sleeved-mode reconciliation drift, ran it against the real account, and it
made the drift *worse* (cash gap $12.8k -> $15.8k). The mistake:
`PerformanceAttribution.get_strategy_holdings()` is a heuristic estimate
(records *decided* trade quantities each cycle, not verified fills) — I
adopted its numbers directly as each sleeve's new holdings with no check
that the 11 sleeves' numbers summed to the broker's one real number per
symbol. It looked reasonable on the first read because the *shape* of the
logic (reuse the existing cutover-seed method) was sound; the actual defect
was a missing invariant, not a missing feature.

**How to apply, generally**: before trusting any heuristic/estimate as the
basis for correcting a live system's state, ask explicitly "what
constraint must the output satisfy against ground truth, and does the code
actually enforce it — or does it just claim to in a docstring?" A heuristic
can be a reasonable signal for *shape* (relative proportions, which of
several sub-accounts is bigger) while being unsafe to trust for *totals*
(the aggregate that must match a real, externally-verifiable number). This
distinction — shape vs. total — is the actual fix that worked the second
time: rescale the heuristic to force the sum to match broker truth, rather
than adopting it wholesale.

**Process that caught and fixed it, worth repeating**:
1. Verified reconciliation immediately after the mutating call, rather than
   assuming success from a 200 response — caught the regression within the
   same turn, not days later.
2. Found the exact pre-mutation state already logged (added that logging
   specifically as a rollback aid before ever running the risky action —
   worth doing this proactively for any live-state-mutating operator
   action, not just after getting burned once) and used it to build a
   real, minimal, already-tested rollback path rather than guessing at a
   fix under pressure.
3. Before retrying with a corrected algorithm, spawned an independent
   adversarial-review subagent specifically *because* the first version
   "looked reasonable on a first read too" — it's a cheap, parallel step
   and it caught two real remaining gaps (a silent zero-sum-weights
   landmine, and a NaN-unsafe verification check) that a self-review would
   likely have missed a second time for the same reason it was missed the
   first time.
4. Built a read-only preview + self-verifying check (compares the proposed
   result against real ground truth, checking for NaN/inf, checking every
   side of the invariant — not just the one that happened to be the
   symptom) as a *permanent* artifact, not a one-off manual check — so the
   same class of mistake can't recur next time this needs to run.
5. Independently spot-checked a few concrete values against known ground
   truth before applying, even after the automated self-check passed —
   didn't just trust the tool's own verification blindly a second time.

See [[project_reconciliation_audit_sep23]] for the full incident.
Related: [[feedback_exhaustive_audit_closure]], [[feedback_production_incident_priority]].
