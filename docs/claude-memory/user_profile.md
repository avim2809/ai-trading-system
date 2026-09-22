---
name: user-profile
description: Who the user is and how they operate this trading-system repo
metadata: 
  node_type: memory
  type: user
  originSessionId: 2ceda047-fc14-4536-ae62-6e8b77b7ca59
  modified: 2026-09-22T21:20:07.334Z
---

Runs `ai-trading-system` as an independent/personal project — git author "Avi Milner". Comfortable with low-level infra, not just application code: has been hand-building `setup.sh` to bare-metal-deploy the stack (Python 3.14, Ubuntu 26, headless IB Gateway + IBC install, systemd units, uninstall paths) across recent commits, and runs things directly on the host rather than only through wrappers. Also cost-conscious about infrastructure — actively watches API usage/quotas (Voyage free-tier %, rate caps) and pushed back on a resource-heavy dependency once its real operational cost became clear.

**How to apply:** it's safe to go straight to root-cause infra fixes (dependency swaps, service/process checks, config file paths) without over-explaining basic Linux/trading concepts. Still always defer to them for anything requiring their own credentials/secrets (e.g. IBKR paper login) rather than trying to work around it.

**Updated 2026-09-22** ([[project_vps_migration_sep22]]): migrated off the old 2-core/3.3GB box to a new VPS (157.173.96.157) — 6 CPU cores, 11GB RAM, 193GB disk. The old tight-resource constraint no longer applies; CPU/RAM headroom is no longer the binding concern it used to be for ML-inference-heavy or resource-intensive live-path work (still worth a sanity check under sustained load, just not the hard blocker it was on the old host). System timezone on the new host is Asia/Jerusalem (fixed post-migration; the VPS provider default was wrong).
