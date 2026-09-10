---
name: project-pattern-recognition-feature
description: "Chart pattern recognition (Strategy #13) — built across 5 phases plus a full follow-up pass, now enabled live on both production instances"
metadata: 
  node_type: memory
  type: project
  originSessionId: d0819ea0-0b87-4dce-9dd6-90d93db8461e
  modified: 2026-09-10T11:45:50.627Z
---

Chart-pattern recognition (Head & Shoulders, triangles, flags, cup & handle,
etc. — Lo/Mamaysky/Wang 2000 geometric framework) shipped as Strategy #13
(`pattern_recognition`), the first genuinely new alpha strategy added this
cycle. Full history/design/bugs found: `docs/pattern_recognition_plan.md`.

**Original 5 phases** (committed `53d5345`..`da08323`): extrema/rule
detectors/scorer, strategy registration, pattern-aware LLM validation,
on-demand `/api/patterns/*` REST + `/patterns` frontend page.

**Follow-up pass** (2026-09-09, commits `bbb72c8`..`660203f`): small bug
fixes (pivot-window detectors now collect every valid match, not just the
first; a pre-existing `test_llm.py` fixture gap), an optional scheduled
scan job + persistent SQLite scan history, a real XGBoost training run +
ONNX export, and — in an isolated Python 3.12 env (`uv`-managed `.venv-ml`,
since torch/stable-baselines3/gymnasium have no Python 3.14 wheels) — a
CNN/GAF image validator and a PPO RL position sizer. The PPO sizer's reward
needed a quadratic risk-aversion penalty to avoid a degenerate "always bet
max size" policy; the first calibration attempt (0.5) still saturated at
max size 99.4% of the time on real cached data — 1.0 is the smallest value
verified to actually break that saturation.

**Enabled live** on both `:8000` (IBKR) and `:8001` (Alpaca) via
`PUT /api/live/config` hot-swap, no restart — added to both
`config/live.yaml` and `config/live_alpaca.yaml`'s `strategies.enabled`
(each instance reads a *different* YAML file via `FIRM_LIVE_CONFIG`, a real
gotcha worth remembering before assuming one edit covers both). Its
per-cycle scan-summary log was bumped `DEBUG`→`INFO` (`660203f`) after
discovering production's logging level meant it was silently invisible —
found while manually verifying it was actually running via a forced cycle.

**How to apply:** if debugging pattern_recognition's live behavior, check
`pattern_recognition: N symbols scanned, M signals` in either instance's
log first. If extending it, `.venv-ml` is a separate, isolated environment
for the CNN/PPO pieces only — never imported by the live `firm-api`
process itself.
