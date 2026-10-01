---
name: project-s2-forward-test-oct01
description: S2 breadth-overlay SHADOW forward test started 2026-10-01 (paper ledger only); timer install pending owner approval; EODHD data dependency
metadata:
  type: project
---
Owner asked (2026-10-01) to forward-test the S2 breadth overlay. I did it as a **shadow ledger** (no orders, no live config change), because a live version needs allocator support and sign-off on the config diff.

- **Design:** frozen `scripts/s2_forward_preregistered.py` (a5142f2, fp cbd6ad32). Rule = S2 V1. Tracker `scripts/s2_forward_shadow.py`. Ledger `data/research/s2_forward/`. Plan `docs/s2_forward_test_plan.md`.
- **First check 10/1:** breadth 47.8% → cut state (SPY 30 / IEF 70). The signal sat just under 50% on 9/29 too.
- **Decision rule:** nothing before 24 months. It needs ≥ 2 completed episodes, which has roughly a 45% chance within 24 months. The rule often answers CONTINUE.
- **Open:**
  - The daily timer (`deploy/s2-forward-shadow.*`) install was blocked by the classifier ("unauthorized persistence"). It needs the owner's go-ahead; until then run `run` by hand.
  - EODHD is a one-month subscription; a lapse means missed checks, and a replacement source needs a validation.

**Why:** the overlay switches rarely, so a forward test is slow by construction.
**How to apply:** don't read early shadow results as evidence; don't promote before the frozen conditions. Related: [[project-eodhd-data-and-insider-verdict-sep30]], [[feedback-external-review-package]].
