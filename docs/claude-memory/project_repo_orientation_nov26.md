---
name: project-repo-orientation
description: Stable architecture, research/ops boundaries, and source-of-truth pointers for future work in this repository
metadata:
  node_type: memory
  type: project
---

## Architecture and mission

- The Python package lives under `src/firm/`; the backend is FastAPI and the UI is React under `frontend/`.
- The documented production design has an IBKR pipeline control on `:8000` and an Alpaca allocation instance on `:8001`. Treat this as documented architecture, not proof of current runtime state; verify before operational actions.
- The project prioritizes credible validation and a diversified, low-turnover core, not searching for alpha. A passive outcome is acceptable.

## Boundaries for future work

- Start with `AGENTS.md` to classify work as OPS or RESEARCH and apply the corresponding rules.
- Never edit the live checkout in place. Research work belongs in the separate research clone and its ticket worktrees, with its own interpreter.
- Research work must respect the holdout policy, approved preregistrations, and append-only trial ledger. Do not inspect sealed runtime data.
- For architecture and operations consult `docs/PROJECT_CONTEXT.md`; use `PLAN.md` and `plan/tickets/` for the validation roadmap and ticket workflow.
- `docs/REPO_MAP.md` identifies itself as verified in early October 2026 and can lag later changes; cross-check its claims against current source and newer docs.
