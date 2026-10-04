# Owner Decisions

Companion to [PLAN.md](../PLAN.md). Each decision has the question, the options, the trade-offs, a recommendation, the tickets it blocks, and a blank for your answer. Write your decision on the `Decision:` line and commit it; a blank Decision line means every ticket listed under Blocks is blocked; agents never substitute the recommendation. In particular OD-03, OD-07, OD-15, OD-19 and OD-20 can never be defaulted. Recommendations are the plan's defaults, not choices already made.

Suggested order: OD-04..OD-08 and OD-09, OD-10, OD-14, OD-15, OD-19 first (they unblock P0 and the early waves), then OD-16 (it freezes `config/gates.yaml`), then OD-17, OD-01, and the rest as their tickets approach.

Ticket headers list every OD a ticket consumes. Only the Blocks column gates a ticket's start; "Also referenced by" lists tickets that read the decision's outcome (for example a value it records) but can start before it is made.

| ID | Topic | Blocks | Also referenced by (non-blocking) |
|---|---|---|---|
| [OD-01](#od-01) | Real investable capital | P2-07, P3-04 (research use), P3-08 (path choice), P6-01, P7-01 (path decision), P7-02 | P2-01, P5-02 |
| [OD-02](#od-02) | Buy futures history? | P2-07, P3-04 (research use) | P2-01 |
| [OD-03](#od-03) | Retire the IBKR pipeline and LLM layer? | P0-07, P6-03 | P0-03, P5-05, P6-01 |
| [OD-04](#od-04) | Meaning of seal_date | P0-02, P0-08, P3-10, P3-11, P3-08 | P5-04 (`exempt_monitors` amendment), P7-03 |
| [OD-05](#od-05) | Post-seal data location and unseal token | P0-02, P0-04, P3-10, P5-06 | P3-08, P3-11, P5-04 |
| [OD-06](#od-06) | CODEOWNERS and branch protection | P0-04, P0-08, P1-09 | P5-01, P5-02 (approval identity) |
| [OD-07](#od-07) | Sandbox, deny settings, bypass mode, research user | P0-02, P0-04 | P0-06, P3-08, P3-11 |
| [OD-08](#od-08) | AGENTS.md / CLAUDE.md structure | P0-04 | |
| [OD-09](#od-09) | Canonical trial count N | P0-05, P1-01, P0-08, P3-08 | P1-03, P1-12 |
| [OD-10](#od-10) | Ledger store format and location | P1-01, P1-12 | |
| [OD-11](#od-11) | Candidate paper account | P6-01, P6-02, P6-03 | P5-04 |
| [OD-12](#od-12) | UCITS vs US-listed, Israeli tax | P2-05 (UCITS variant only), P7-01 | P2-01, P2-06 |
| [OD-13](#od-13) | Futures_trend DRAFT prereg | P1-09, P2-01 | |
| [OD-14](#od-14) | Freeze EODHD cleaning v3 | P2-08, P2-02, P3-11 | P3-08 |
| [OD-15](#od-15) | New systemd timers or instances | P5-06, P6-03, P5-04 | |
| [OD-16](#od-16) | Sign the gate deviation register | P0-08, P1-08, P3-08, P6-02 | P4-02, P4-04, P5-01, P5-02 |
| [OD-17](#od-17) | ETF-path target vol tau | P5-02, P3-11, P3-08 | P0-08, P3-06, P4-03 |
| [OD-18](#od-18) | Forward-holdout pass before first live step | P3-10, P7-02 | P0-08 |
| [OD-19](#od-19) | Allocation monitor NAV source and halt authority | P5-06 | |
| [OD-20](#od-20) | Real-money broker, account and host | P7-02 | P7-01 |

---

## OD-01

**Real investable capital**

**Question.** What is your real investable capital? The $1M IBKR and ~$100k Alpaca balances are paper. This decides ETF path vs futures path.

**Options.**
- (a) under $25k: passive only
- (b) $25k-100k: ETF path (15-25 ETFs, long/flat, no carry)
- (c) $100k-250k: micro futures or hybrid
- (d) over $250k: full futures trend + carry

**Trade-offs.** Lower capital makes futures lumpy (one micro in each of 19 markets is roughly 15-25% of $100k in margin) and costs more in fixed fees; higher capital justifies the 34-56 person-day futures plumbing.

**Recommendation.** Build and evaluate the ETF path first whatever the capital: the data is owned, it runs on Alpaca-class brokers, and it needs no futures plumbing. Revisit futures only if capital is at least about $100k and the ETF core result or the charter argues for carry. The P7-01 tool is built now with no performance input.

**Blocks only.** P2-07, the P3-08 path choice, P6-01, P7-01's path decision (not the tool itself), P7-02 (no real capital is committed before this is answered), and P3-04 research use only (P3-04 is a pure fixture-tested function and is scheduled right after P3-01).

**Decision:** (a) under $25k: passive only. Recorded 2026-10-03. Consequence: per P7-01 a bespoke system is not run live at this size; P0-P1 validation tooling and the P3-08 research verdict remain useful, live phases (P6-P7) are not planned. Revisit if capital grows.

---

## OD-02

**Buy futures history?**

**Question.** Should we buy futures history: Norgate Futures $270/yr (Windows NDU updater), CSI $200/yr (Linux support unconfirmed), or Databento (data from about 2010)?

**Options.**
- Buy Norgate now
- Buy CSI
- Buy Databento
- Defer

**Trade-offs.** Buying now unblocks P2-07 and carry research but spends money before knowing whether the ETF core has any verdict; Eurex coverage and second-contract (carry) data are unconfirmed for Norgate/CSI; Norgate's updater is Windows-only.

**Recommendation.** Defer until OD-01 says futures and the ETF core has a verdict. If you buy, first confirm Eurex coverage and second-contract data, and draw the sealed instrument subset yourself (recorded seed) before the data reaches the host.

**Blocks.** P2-07, P3-04

**Decision:** Defer / not applicable (owner, 2026-10-04): no futures data is bought; P2-07 stays deferred. Revisit only if OD-01 capital rises to at least about $100k and the ETF core earns it.

---

## OD-03

**Retire the IBKR pipeline and LLM layer?**

**Question.** Should the IBKR :8000 11-strategy pipeline and its LLM layer (FIRM_LLM_CONFIG=config/llm_ab_llm.yaml) be taken out of the live signal path, and when? Should pattern infrastructure go too, given FIRM_ENABLE_PATTERN_SCAN=1 on the Alpaca unit?

**Options.**
- (a) now
- (b) keep as an untouched control until P6, then retire to free the IBKR paper account
- (c) keep indefinitely with statuses advisory only

**Trade-offs.** (a) ends the blended control and LLM arm B mid-experiment and guts the pipeline (analyst buckets empty); (b) preserves the comparison and costs nothing now; (c) leaves failed strategies live indefinitely. All 11 enabled strategies failed the standalone evaluation.

**Recommendation.** (b). Leave both units unchanged. Publish DEPRECATIONS.md and LLM_POLICY.md now (no live effect). Retire via P0-07 (an explicit restart ticket) when the IBKR paper account is needed or the LLM A/B ends. Record stat_arb's (enabled, -3.41, missing from the source) target status as ARCHIVED in DEPRECATIONS.md; its effective status stays LEGACY_LIVE (warn-only) until P0-07 executes under OD-03. Decide separately whether the Alpaca pattern scan stays.

**Blocks.** P0-07, P6-03

**Decision:** (b) keep as untouched control; retire later via P0-07. Recorded 2026-10-03.

---

## OD-04

**Meaning of seal_date**

**Question.** What does seal_date mean, given forward tests already consume data from 2026-09-30/10-01, research code has seen data through 2026-09-30, and post-seal data already sits on the host?

**Options.**
- (a) seal 2026-10-01 for research code, ACL-based on this host, forward-test monitors exempt
- (b) seal an earlier date (pointless, already seen)
- (c) no seal, forward paper only

**Trade-offs.** (a) is honest about what is protectable but is fail-open for code that never imports firm.research; (b) protects nothing; (c) gives up any pre-registered forward check.

**Recommendation.** (a). Declare all data up to 2026-09-30 in-sample and burned, including the 2024-07..2026-06 window and the whole 2020-01..2026-06-30 panel. Accept the fail-open residual risk, mitigated by ACL and P1-12 capture. The forward holdout may be unsealed once per family, no earlier than 12 months of accrued data (P3-10). Correct '0.277 best holdout Sharpe' to '0.277 = baseline pipeline Sharpe on the Gann follow-up window'. Monitor exceptions are listed in HOLDOUT_POLICY.md (P0-02): `exempt_monitors` (allocation_forward_test, s2_forward); the P5-04 candidate shadow replay reads only through the monitor-only `firm.monitoring.shadow_loader` (or, if you prefer, a signed amendment adding the candidate monitors to `exempt_monitors`); and the P7-03 annual review reads post-seal data only through that owner-approved monitoring path.

**Blocks.** P0-02, P0-08, P3-10, P3-11, P3-08

**Decision:** (a) accepted as recommended. Recorded 2026-10-03.

---

## OD-05

**Post-seal data location and unseal token**

**Question.** Where does post-seal research data live, and who holds the unseal token?

**Options.**
- On-host ACL-restricted directories plus off-host unseal copy
- On-host encrypted volume
- On-host plain directory

**Trade-offs.** Services need live and monitor data on-host, so physical absence is impossible for those; an off-host or encrypted unseal copy is the only strong control for the one-time evaluation but adds a manual step.

**Recommendation.** Live and monitor data stay on host, restricted by ACL from the research user. The data for the one-time unseal evaluation is pulled by a script you run yourself to an off-host or encrypted, not-auto-mounted volume. Only you hold the token preimage; config stores its sha256. Sealed monitor snapshots (`research/monitoring_sealed/`, written only by owner-run `export` subcommands of P5-04/P5-06) stay on the host, gitignored and ACL-restricted from the research user; they are not committed, because a tracked file would be checked out into every research clone.

**Blocks.** P0-02, P0-04, P3-10, P5-06

**Decision:** Accepted as recommended. Recorded 2026-10-03.

---

## OD-06

**CODEOWNERS and branch protection**

**Question.** CODEOWNERS and branch protection for a solo owner who commits straight to main under the same identity the agent uses.

**Options.**
- (a) advisory CODEOWNERS only
- (b) separate GitHub machine user or fine-grained PAT for research agents, plus branch protection requiring a PR and code-owner review, with you merging (GitHub rulesets)
- (c) local pre-push hook only

**Trade-offs.** (a) is decorative; (b) gives real separation but changes your workflow (PRs, a second credential); (c) is cheap but local and bypassable. CI is PR-only today, so a push trigger and integrity job are needed either way.

**Recommendation.** (b) for research agents, plus a push trigger and an integrity job in CI. Ops sessions keep your identity. Use a GitHub ruleset whose bypass list includes your account (or repository admins) and apply the PR and code-owner requirement to the research machine user only, because branch protection otherwise binds every identity and a code owner cannot approve their own PR, which would block your in-session production hotfixes. Document a hotfix path (direct push by you, then a post-hoc integrity CI run) and add a GUARDRAIL_REDTEAM item confirming you can still push a hotfix to main. Without a second identity CODEOWNERS is decorative, and the plan says so rather than claiming enforcement; likewise the approval-identity checks in P5-01/P5-02 report 'unverifiable' (advisory only) unless (b) is adopted.

**Blocks.** P0-04, P0-08, P1-09

**Decision:** (a) advisory CODEOWNERS only. Amended 2026-10-03: no separate GitHub machine account; the research user shares the owner's GitHub identity and Claude login. Enforcement comes from the unix user, file ACLs, managed settings and a root-owned local pre-push hook that blocks pushes to main/master/tags; approval-identity checks (P5-01/P5-02) are advisory ('unverifiable'). No GitHub ruleset is created.

---

## OD-07

**Sandbox, deny settings, bypass mode**

**Question.** Sandbox, deny settings and bypass mode vs existing ops workflows (systemctl, curl to :8000/:8001, ufw/nginx, IB Gateway), with the agent running as root.

**Options.**
- (a) the source's global sandbox with allowUnsandboxedCommands=false
- (b) narrow committed denies plus a sandboxed research profile
- (c) a separate non-root unix user for research agents plus root-owned managed settings in /etc/claude-code

**Trade-offs.** (a) would break ops workflows; (b) alone is bypassable by shell redirection as root; (c) gives real separation but needs root-side setup and a user with no read access to post-seal paths.

**Recommendation.** (b)+(c). The committed .claude/settings.json and /etc/claude-code/managed-settings.json apply to EVERY session including root ops sessions, so they carry only rules safe for all of them: Edit/Write denies on settings.json, settings.local.json, .claude/hooks/**, the guardrail files, tests/integrity, gates.yaml and preregistration files, plus a force-push deny. The post-seal Read denies (data_alpaca, data/forward_monitors, data/research/s2_forward, research/monitoring_sealed), the deny_holdout hook, the sandbox and disableBypassPermissionsMode='disable' go in the research user's own ~/.claude/settings.json, owned by root and read-only to that user (the user scope is then the research session scope, and the ACL already denies that user). Research agents run as a non-root user. Ops sessions are unaffected: add GUARDRAIL_REDTEAM items confirming a root ops session can still read data_alpaca/kill_switch_state.json and data_alpaca/logs/ and can still run systemctl and curl :8000/:8001. GUARDRAIL_REDTEAM must try the settings.local.json/disableAllHooks bypass and shell-redirect writes. Update memory entries ('run on your own', autonomous restarts) so research sessions exclude service restarts and live-config edits. You install the managed settings; the agent drafts.

**Blocks.** P0-02 (its `allow_roots`/`deny_paths` are the research user's ACL basis), P0-04

**Decision:** (b)+(c) non-root research user + managed settings, as recommended. Recorded 2026-10-03.

---

## OD-08

**AGENTS.md / CLAUDE.md structure**

**Question.** AGENTS.md/CLAUDE.md structure and rule scope. The source wants CLAUDE.md = '@AGENTS.md' and AGENTS.md agent-uneditable; B5 rules 8 and 10 are already violated by ops practice and by LLM arm B.

**Options.**
- (a) verbatim source text
- (b) merge into AGENTS.md, CLAUDE.md = @AGENTS.md plus an editable Claude tail, rules scoped by session type
- (c) a separate protected RESEARCH_RULES.md imported by both

**Trade-offs.** (a) would drop CLAUDE.md's Rules of thumb (including the IBKR asyncio rule and resolve_live_startup rule) and overwrite AGENTS.md; (b) keeps safety rules and needs a scoping section; (c) adds a file but keeps existing ones untouched.

**Recommendation.** (b). Fold CLAUDE.md's Rules of thumb into AGENTS.md first. Add B5 rules adapted to real paths. Scope rules 8 (no parameter changes in EMBARGO/PAPER/LIVE) and 10 (no LLM in signal path) to strategies managed by firm.lifecycle; the legacy IBKR pipeline and LLM arm B are explicitly grandfathered until OD-03. Correct the stale Alpaca description in both files during the merge (CLAUDE.md and AGENTS.md still say 'strategy_mode: allocation (opt-in, not live yet)' and 'Alpaca runs sleeved live today'; config/live_alpaca.yaml has been strategy_mode: allocation, 60/40 core plus 8% BTC trend, since 2026-09-30; also an input to P0-04). Add an 'ops session vs research session' section; routine ops edits to config/live.yaml stay allowed in ops sessions.

**Blocks.** P0-04

**Decision:** (b) accepted as recommended. Recorded 2026-10-03.

---

## OD-09

**Canonical trial count N**

**Question.** Legacy trial count and canonical N for DSR.

**Options.**
- Ledger-only 210
- Ledger plus unledgered estimate, about 460
- The per-prereg placeholders 190/205/206/207

**Trade-offs.** A lower N is easier to pass and understates multiple testing; a higher N is conservative but includes estimates (Gann about 145 vs 8). The owner has historically preferred conservative counting.

**Recommendation.** Carry both counting conventions (pattern_ml 104 folds x candidates vs 13 configs). Gate N = the raw count of all trials (returns-bearing, Sharpe-only, estimate-only legacy rows, counted unregistered and API-capture rows): about 460 to start; effective_n (P1-03) is reported only, marked [ADJ] in the register, because effective-N estimates are <= the raw count and a max() rule would always reduce to it. Alternative for you to choose: the source-faithful rule N = max over {onc, enb, mp} of effective_n on returns-bearing trials + raw count of Sharpe-only/estimate-only/unregistered/API rows. P0-08 asserts the gate's N formula matches the signed text. Report sensitivity at 210. Sign off Gann at about 145 rather than 8. Also see OD-16 for the implied minimum Sharpe.

**Blocks.** P0-05, P1-01, P0-08, P3-08

**Decision:** AMENDED 2026-10-04 (owner): the DSR gate uses a FAMILY-SCOPED N (time-series trend/carry trials, provisionally 31, list signed in gates.yaml) with the raw count 463 (210 ledgered-only) reported alongside in every verdict. Supersedes the 2026-10-03 answer (raw count 463 as the gate N). Gann stays counted at about 145 in the raw figure. See docs/gate_deviation_register_2026_10.md row 2. CENSUS SIGNED OFF as drafted (owner, 2026-10-04): Gann at about 145, the seven estimate-only rows, gross N = 463 and 210 ledgered-only. The 31-trial trend/carry family list is CONFIRMED including S5 crypto momentum; the flag n_counts.family_membership_confirmed_by_owner in gates.yaml stays false only because that file is hash-pinned, and this line is the confirmation.

---

## OD-10

**Ledger store format and location**

**Question.** Ledger store format and location.

**Options.**
- Per-branch JSONL in git
- Host-level JSONL outside worktrees, mirrored to git
- Per-trial files merged by CI
- SQLite

**Trade-offs.** Per-branch files fork across parallel or abandoned worktrees; SQLite is binary and not diffable; a host-level file needs flock and a sync step but is parallel-safe.

**Recommendation.** Make /local/store/research-ledger/trials.jsonl canonical: hash-chained, flock-serialised, optionally chattr +a, writable by the research user. Returns as returns/<trial_id>.parquet. A serial owner/ops sync copies trials.jsonl into tracked research/ledger/ plus a returns manifest (trial_id, sha256, n_obs); the parquet files themselves never enter git (returns derived from licensed EODHD data) and travel to reviewers only in the private review bundle (P1-01, P1-11). CI checks the mirror is only extended (prefix check). You provision the root before first use: `returns/` and `inbox/` (P1-12 capture lines from every backtest entry point, including firm-api), group-owned by the research group, mode 2775. P1-12 also needs your acknowledgement that `src/firm/api/app.py` joins the LIVE-IMPORT-PATH set. An SQLite mirror is optional later.

**Blocks.** P1-01, P1-12

**Decision:** Host-level JSONL ledger at /local/store/research-ledger/, as recommended. Recorded 2026-10-03.

---

## OD-11

**Candidate paper account**

**Question.** Paper account for a candidate, and which fidelity thresholds apply.

**Options.**
- Reuse Alpaca :8001
- Second Alpaca paper account on a new :8002 instance
- IBKR paper after OD-03

**Trade-offs.** Reusing :8001 conflicts with the BTC sleeve and the running forward test; :8002 costs a third service but keeps evidence clean; IBKR waits on OD-03.

**Recommendation.** Separate paper account and a third instance (:8002, data_candidate/), provisioned only after G-RESEARCH passes. The source G-PAPER thresholds apply to candidates. The allocation forward test keeps its frozen I1-I5 bars. A 'rebalance' is a scheduled review event.

**Blocks.** P6-01, P6-02, P6-03

**Decision:** Not applicable for now (owner, 2026-10-04): capital is under $25k, where the plan stays passive. Nothing is bought, provisioned or installed. Revisit if capital exceeds $25k or before any live step. No candidate paper account or :8002 instance; P6 stays unplanned.

---

## OD-12

**UCITS vs US-listed, Israeli tax**

**Question.** UCITS vs US-listed ETFs, and Israeli tax treatment (estate tax, withholding, and the 4 open questions in docs/research_findings_beyond_equities_2026_09_30.md L256).

**Options.**
- US-listed now
- UCITS now
- Decide after a tax adviser

**Trade-offs.** US-listed ETFs have owned history and simple access but possible US estate-tax and withholding exposure; UCITS avoids some of that but has data and broker-access gaps in this repo.

**Recommendation.** Run P2-05 now on US-listed BM2 with a UCITS-variant placeholder so research is not blocked. Get an Israeli tax adviser's answer before P7. This is information, not advice.

**Blocks only.** the P2-05 UCITS variant and P7-01 (the US-listed BM2 P2-05 run is not blocked)

**Decision:** US-listed ETFs now (owner, 2026-10-04): research and BM2 run on owned US-listed history; the UCITS variant stays a placeholder. An Israeli tax adviser's answer (estate tax, withholding, reporting) is required before any live step. Information, not advice.

---

## OD-13

**Futures_trend DRAFT prereg**

**Question.** Is the frozen DRAFT scripts/futures_trend_preregistered_bars.py (19 markets, single 12-month TSMOM sign rule, 10%/20% overlay on 60/40, never run, 0 trials) superseded?

**Options.**
- Supersede
- Run separately as a satellite test
- Keep as an option

**Trade-offs.** Superseding avoids two overlapping futures-trend hypotheses; running it separately adds trials to N.

**Recommendation.** Leave the file frozen and untouched. Mark it 'superseded-by core_v1 (DRAFT, never run, 0 trials)' in INDEX.yaml. Its market list seeds universe_futures.yaml only as a candidate list chosen without performance input.

**Blocks.** P1-09, P2-01

**Decision:** Supersede; file stays frozen and untouched. Recorded 2026-10-03.

---

## OD-14

**Freeze EODHD cleaning v3**

**Question.** Freeze EODHD cleaning v3 (the 999999.9999 sentinel rule; negative prices allowed only for futures) before any further EODHD-based test?

**Options.**
- Freeze v3 now as a new module
- Patch v2 in place

**Trade-offs.** Patching v2 would break the fingerprint pinned by frozen preregistrations; a new module is additive.

**Recommendation.** Freeze v3 now, additively, in src/firm/data/cleaning.py with its own fingerprint. v2 stays.

**Blocks.** P2-08, P2-02, P3-11

**Decision:** Freeze v3 now as a new module. Recorded 2026-10-03.

---

## OD-15

**New systemd timers or instances**

**Question.** Should new systemd timers or instances be installed (allocation forward-test monitor, daily reconcile, candidate instance)?

**Options.**
- Approve now
- Approve per ticket
- Never

**Trade-offs.** The frozen I1/I2 bars are currently computed by nothing, so delay means retroactive reconstruction; any new unit on a shared host needs your go-ahead.

**Recommendation.** Approve installing the read-only P5-06 monitor timer after its code review and dry run; the timer is installed by you, not by an agent. This covers the monitor unit and its daily `refresh-inputs` step (fetches the replay inputs inside the same unit, credentials from a root-owned file via `os.environ` only, never `.env`). Approve the candidate instance and the P5-04 reconcile timer per ticket at P6. The S2 timer remains your separate pending decision.

**Blocks.** P5-06, P6-03, P5-04

**Decision:** Yes, after a dry run (owner, 2026-10-04): the read-only allocation-forward-test monitor timer may be installed by the OWNER (not an agent) once `validate` and a dry `run` have passed. Candidate-instance and reconcile timers stay per-ticket at P6. The S2 timer remains a separate pending decision.

---

## OD-16

**Sign the gate deviation register**

**Question.** Sign the gate deviation register (docs/gate_deviation_register_2026_10.md) before gates.yaml is frozen. Each row gives the source threshold, the reconciled value and the rationale.

**Options.**
- Sign as drafted
- Amend specific rows
- Revert to source text verbatim

**Trade-offs.** Signing fixes the gates before any result is seen. Reverting to source text leaves ambiguities (75% of 9 paths, N definition, two max-DD uses).

**Recommendation.** Sign with these rows: var_sr = length-adjusted variance (var(SR_i) minus mean sampling variance plus sampling variance at the candidate's T, floored at the grid variance, family ledger Sharpes converted to per-period, T_i recorded per trial), with unadjusted and all-family sensitivities; N rule as OD-09 (raw count, effective_n reported only); implied minimum annualised SR for DSR >= 0.95 stated explicitly (about 1.65/yr at N=460, T about 5000; Tier A is effectively out of reach for a 0.3-0.6 SR strategy); two-sided DSR calibration; tier mapping with PBO Tier D only when PBO >= 0.5 and prob_oos_loss >= 0.5, effective-N guard for uninformative PBO, gate 7 decision table, gate 8 failure = Tier D; BM2 annual rebalancing for the after-tax benchmark; CPCV selection rule applied inside each split; PBO < 0.30, and < 0.20 because the family exceeds 20 variants; CPCV at least 7 of 9 paths positive; robustness applied to EVERY numeric core parameter; gate 7 restores the correlation <= 0.3 branch plus a mandatory power analysis; tier mapping pass=A, insufficient=C, fail=D, only A paper-eligible; charter max-DD split rule with a fixed 10-year bootstrap path (episode-length p95 for the stress gate, max(p95_10y, 2.5 tau) for survival; 2.5 tau labelled a rule of thumb); G-PAPER 1 as 26 weekly review events and at least 6 months; G-PAPER 3 only after at least 30 fills across at least 10 instruments; 12-month minimum before unseal; H5 met by a dry-run Allocator enforcement test; ENB >= 2.5 (ETF) as a hard H4 exit with no post-hoc universe changes; P1-08 size/power acceptance rates as recomputed (power at K=1 SR 1.0/yr and at K=50 SR >= 1.3/yr or within 0.05 of a simulated oracle; PBO < 0.1 under the pinned strong drift N=30, SR 3.0/yr, T=1600, S=16), frozen with their scenarios in `gates.yaml` `p1_08_acceptance`; one bootstrap procedure (`max_dd_procedure`, 2520-day paths) serves both the episode-length stress reference and the survival reference; P4-04 deviations (2.5 tau floor via survival_ref, audited reset baseline and NAV correction, block new orders rather than flatten); the Meucci weight-dependent ENB on asset-class P&L as the H4 measure with the 0.7 stress-correlation flag; the Kelly bound `tau <= 0.5 x Kelly vol` enforced as a charter-integrity check; G-LIVE-STEP drawdown reference survival_ref x 1.0 and G-PAPER 3 minimum of 30 fills across 10 instruments on live fills; the P5-04 fidelity window (trailing 63 trading days); with a deviation-register row each.

**Blocks.** P0-08, P1-08, P3-08, P6-02

**Decision:**  Post-signature confirmation (owner, 2026-10-04): the gate 7 margin is the minimum detectable Sharpe gap from the mandatory power analysis (a rule, no fixed number), as frozen in gates.yaml.

---

## OD-17

**ETF-path target vol tau**

**Question.** ETF-path target vol tau under gross <= 1.0x and long/flat, given that half the universe is bonds/linkers/credit.

**Options.**
- Source 12-15% (positions mostly pinned at the gross cap)
- tau set ex ante from asset vols (about 8-10%)
- Source tau with a ceiling on cap-bound days

**Trade-offs.** A tau far above achievable vol under a 1.0x cap makes the strategy effectively static; a lower tau departs from the source but is honest about the constraint.

**Recommendation.** Set tau in the ETF charter ex ante from long-run asset vols, without performance input; about 8-10% expected. Report the fraction of days at the gross cap with a pre-set ceiling of 20%; exceeding it is a G-RESEARCH 6 robustness failure. Record the choice before P3-11.

**Blocks.** P5-02, P3-11, P3-08

**Decision:** 8-10% target volatility, set ex ante without performance input (owner, 2026-10-04). Working value tau = 9% (midpoint), to be confirmed in the ETF charter (P5-02) before P3-11; fraction of days at the gross cap must stay <= 20% (gate 6). gates.yaml keeps tau as a deferred value frozen by the charter.

---

## OD-18

**Forward-holdout pass before first live step**

**Question.** Forward-holdout evaluation (P3-10): is a pass required before P7-02's first live step?

**Options.**
- Required before LIVE_STEP_1
- Reported only
- Skip

**Trade-offs.** Required adds a 12-month wait but is the only post-seal evidence; with 12 months it is a no-contradiction check, not proof of edge.

**Recommendation.** Required. Gates pre-set in gates.yaml: forward net SR not below the 5th percentile of the block-bootstrapped distribution of 12-month backtest SRs, and forward max-DD within the charter envelope, where the reference is the survival_dd and the forward check uses a 12-month-horizon bootstrap p95. Labelled a no-contradiction check.

**Blocks.** P3-10, P7-02

**Decision:** Report only (owner, 2026-10-04): the 12-month forward-data check is reported but not required before the first live step. Overrides the recommendation. Follow-up when gates.yaml is written: update G-LIVE-STEP in PLAN.md and ticket P7-02.

---

## OD-19

**Allocation monitor NAV source and halt authority**

**Question.** Allocation forward-test monitor: the source for live_nav, and who executes the prereg's IMPLEMENTATION_ACTION.any_fail.

**Options.**
- live_nav from the :8001 local API / from Alpaca portfolio-history with service credentials
- any_fail run automatically by the monitor / alert-only with you or incident response deciding

**Trade-offs.** The API route avoids reading .env; automatic halts risk false-positive trading stops.

**Recommendation.** No read-only credential exists today: both nginx server blocks use one /etc/nginx/.htpasswd for every method and path (no limit_except), so that credential could also POST /api/live/stop or reset the kill switch. Choose (a) the monitor calls 127.0.0.1:8001 directly over loopback (firm-api binds 127.0.0.1, no credential, no .env) with a hard-coded GET-only endpoint allowlist (e.g. GET /api/live/portfolio-history) and a unit test that every request is a GET to an allowlisted path; or (b) P5-06 adds a GET-only nginx location (limit_except GET { deny all; }) with its own htpasswd user, applied and reloaded by you as a touches_live step. Any stored credential lives in a root-owned file read by the unit, never .env and never read by an agent. Recommended: (a). The monitor only alerts. Any halt is your decision or an incident action, never automatic. Also decide the `live_nav` selection rule (P5-06 recommends the last portfolio-history snapshot after 16:00 ET for the trading day) and whether `initial_nav` is the pre-fill or post-fill value; the rule is appended to `docs/allocation_forward_test_trial_history.json` by you.

**Blocks.** P5-06

**Decision:** (a) loopback GET-only allowlist, alert-only, as recommended. Recorded 2026-10-03.

---

## OD-20

**Real-money broker, account and host**

**Question.** Real-money broker, account and host security posture for P7.

**Options.**
- IBKR live
- Alpaca live (availability to an Israeli resident unconfirmed)
- Other regulated broker
- Run on this internet-exposed VPS vs a separate hardened host

**Trade-offs.** IBKR accepts Israeli residents; this VPS has been exposed to the internet before (nginx+ufw+basic-auth added later), so real money warrants a fresh review or a separate host.

**Recommendation.** Decide at P7-01. Prefer IBKR live. Run real-money trading only on a separate hardened host, or on this VPS only after a fresh security review. FIRM_ALLOW_TRADING plus a live broker type remain the two locks. Whatever the host: arming and live credentials live only in a separate root-owned 0600 EnvironmentFile (never the shared `.env`), the real-money account is distinct from the P6-03 paper account, and the instance runs from a separate pinned checkout with its own venv whose commit hash is recorded in the step approval (P7-02).

**Blocks.** P7-02

**Decision:** Not applicable for now (owner, 2026-10-04): capital is under $25k, where the plan stays passive. Nothing is bought, provisioned or installed. Revisit if capital exceeds $25k or before any live step. No real-money broker, account or host is chosen; P7 stays unplanned.
