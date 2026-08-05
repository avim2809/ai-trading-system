---
name: user-profile
description: Who the user is and how they operate this trading-system repo
metadata: 
  node_type: memory
  type: user
  originSessionId: 2ceda047-fc14-4536-ae62-6e8b77b7ca59
  modified: 2026-07-18T21:33:06.335Z
---

Runs `ai-trading-system` as an independent/personal project — git author "Avi Milner". Comfortable with low-level infra, not just application code: has been hand-building `setup.sh` to bare-metal-deploy the stack (Python 3.14, Ubuntu 26, headless IB Gateway + IBC install, systemd units, uninstall paths) across recent commits, and runs things directly on the host rather than only through wrappers. Also cost-conscious about infrastructure — actively watches API usage/quotas (Voyage free-tier %, rate caps) and pushed back on a resource-heavy dependency once its real operational cost became clear.

**How to apply:** it's safe to go straight to root-cause infra fixes (dependency swaps, service/process checks, config file paths) without over-explaining basic Linux/trading concepts. Still always defer to them for anything requiring their own credentials/secrets (e.g. IBKR paper login) rather than trying to work around it. The deployment VPS is small — **2 CPU cores, 3.3GB RAM** (confirmed 2026-07-18) — and also runs IB Gateway (a Java process) during live trading. Weigh CPU/RAM cost before recommending anything ML-inference-heavy or otherwise resource-intensive for the live path; prefer noting the tradeoff and asking over silently adding it.
