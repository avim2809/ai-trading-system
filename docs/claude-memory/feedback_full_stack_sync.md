---
name: feedback_full_stack_sync
description: "User wants frontend/backend kept in sync as one unit, not backend-only fixes"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d187bdf3-58a0-49c4-ae8b-81294f53899d
  modified: 2026-07-28T23:07:17.851Z
---

When changing or auditing this system, always check frontend impact too, not just backend/API — verify `frontend/src/api/{types,client}.ts` and any consuming page actually match the real runtime shape the backend returns, not just that `tsc --noEmit` passes (TypeScript types on `fetch` responses aren't checked against reality, so a type can be simply wrong and compile fine).

**Why:** Found 2026-07-29 while fixing a live crash: `LiveDashboard.tsx` called `c.cycle_id.slice(0, 8)` assuming a truncated UUID string, but the backend (`CycleResult.cycle_id: int` in `src/firm/live/engine.py`) has always sent a plain integer — `.slice()` on a number throws, and with no error boundary the whole page unmounted (rendered fine for ~1s until the cycles query resolved, then went blank). The test mocks in `frontend/src/test/mockData.ts` had also wrongly used `cycle_id: '1'` (string), which happened to make `.slice()` work in tests — masking the bug from the suite entirely. Separately, the deployed `frontend/dist` was also found stale (built before the last edits to `LiveDashboard.tsx`/`LiveConfig.tsx`), compounding the outage.

**How to apply:**
- After any backend schema/field-shape change, grep the frontend for every place that field is consumed and re-verify by eye, not just by typecheck.
- Rebuild the frontend (`npm run build`) after every session that touches `frontend/src`, since `frontend/dist` is what's actually served (`src/firm/api/app.py` mounts it via `StaticFiles`) and is gitignored/never auto-rebuilt.
- When a test mock and the real API could plausibly diverge (any `fetchJson<T>()` call), sanity-check the mock's field *types* against an actual `curl` of the real endpoint, not just against the TS interface — the interface can be wrong in exactly the same way the mock is.
- See [[project_frontend_log_monitor]] and [[project_reflection_and_gui_parity]] for prior frontend/backend parity gaps found the same way.
