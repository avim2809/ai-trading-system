---
name: project-frontend-log-monitor
description: Frontend live log monitor added 2026-07-19 — /logs page + GET /api/logs/tail endpoint
metadata: 
  node_type: memory
  type: project
  originSessionId: 2ceda047-fc14-4536-ae62-6e8b77b7ca59
  modified: 2026-07-19T06:48:17.516Z
---

Added a live-tailing log console to the frontend (`frontend/src/pages/Logs.tsx`, route `/logs`, "Monitoring" nav section) backed by a new `GET /api/logs/tail?offset=N` endpoint (`src/firm/api/routers/logs.py`) that incrementally reads `data/logs/api.log`. Committed `4ad9dc0`, pushed to `origin/main`.

**Why:** user asked to "fix my frontend and add a logging monitor... to see live progress." Investigation found the frontend itself had no static bug (clean type-check/build, every API call matched a real route) — the actual ask was the missing live-progress feature, confirmed with the user via AskUserQuestion.

**Related fix while wiring this up:** `src/firm/api/app.py` previously only called `setup_logging(log_file=...)` inside the `firm-api` console-script `run()` function — so starting the API any other way (bare `uvicorn firm.api.app:app`, Docker CMD) wrote **no log file at all**, which would have made the new /logs page permanently empty in that case. Moved the call to module import time so it always runs. See [[project_exception_logging_audit]] for the broader silent-exception-logging work this session, including the same `except: pass` pattern also found on the LLM router import in this same file.

**How to apply:** if live trading or backtest runs seem to produce no output in the new Logs page, first check the API process was actually started via a path that imports `firm.api.app` (any uvicorn/gunicorn entry point now works) and that `data/logs/api.log` exists and is growing. Uvicorn's own HTTP access logs do NOT appear here (uvicorn.access has `propagate=False`) — only the app's own `log.info/warning/error` calls do, which is intentional (this is a progress/activity monitor, not an HTTP access log).
