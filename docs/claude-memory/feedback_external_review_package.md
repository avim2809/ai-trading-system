---
name: feedback-external-review-package
description: Owner wants research reviewable by a third party — data, scripts and results preserved, not just verdicts
metadata:
  type: feedback
---
The owner (2026-10-01): "for external review you must save all the needed info for 3rd party to review your work … not just the results but the data and scripts used to evaluate".

**Why:** scratchpad artifacts are ephemeral, and branches and logs scatter. Without the exact inputs, scripts and run outputs, nobody can audit a verdict.

**How to apply:** for every research study, keep the review package current with `scripts/build_review_package.py --scratch <scratchpad>`. Add new run dirs to its `STUDIES` map. It produces:
- **in git:** `review/` (artifacts, scratch scripts, input SHA-256 manifest, environment, UTC timeline);
- **outside git:** a private data bundle with the exact licensed and public inputs.

Merge study branches into main. Vendor data stays out of git (licences); the owner decides who gets the bundle. Related: [[project-eodhd-data-and-insider-verdict-sep30]], [[feedback-host-clock-is-local-time]].
