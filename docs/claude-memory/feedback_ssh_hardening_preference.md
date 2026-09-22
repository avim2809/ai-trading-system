---
name: feedback-ssh-hardening-preference
description: "User wants to keep SSH password authentication enabled — don't propose disabling it broadly even when a real misconfiguration is found"
metadata:
  node_type: memory
  type: feedback
  modified: 2026-09-22T22:23:38.067Z
  originSessionId: 403dab55-8f5b-43a3-afa3-7533df0c5692
---

An audit found a real SSH misconfiguration (2026-09-23, [[project_vps_migration_sep22]]):
conflicting cloud-init drop-ins meant root login with password auth was
effectively enabled despite one file trying to disable it. I proposed a fix
that disabled `PasswordAuthentication` entirely plus `PermitRootLogin
prohibit-password`. The user pushed back: "why ssh hardening - i want to
continue using ssh password as well." Also declined an unrelated, genuinely
low-risk nginx `server_tokens off` hardening suggestion without much
explanation — password-based access generally seems to be the user's
preferred/needed workflow on this box, not an oversight to be corrected.

**Why:** the user has an established Linux/infra-comfort profile
([[user_profile]]) and evidently relies on SSH password login as their
actual access method to this VPS — a security audit finding "this is
possible" doesn't mean the user wants it closed off, especially when it's
their own primary access path.

**How to apply:** don't unilaterally apply SSH/access-hardening changes from
an audit finding, even a "critical" one, without asking first — this is
exactly the kind of thing to flag and let the user decide, not assume.
When they decline, a narrower option (e.g. disable password auth for
*root* specifically while leaving it enabled for other accounts) can be
offered once, but don't push if declined again. General security hardening
suggestions (nginx server_tokens, etc.) are worth mentioning but should be
presented as low-priority/optional, not framed as something that needs
fixing.
