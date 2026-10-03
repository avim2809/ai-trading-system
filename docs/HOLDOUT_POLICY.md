# Holdout policy: discovery freeze and forward-data seal

Ticket P0-02 (see [PLAN.md](../PLAN.md) section 8, [plan/OWNER_DECISIONS.md](../plan/OWNER_DECISIONS.md) OD-04, OD-05, OD-07).
The machine-readable source of truth is [`config/research_freeze.yaml`](../config/research_freeze.yaml); this page explains it.
If the two ever disagree, the YAML wins and this page is wrong.

## 1. What is burned and what is sealed

| Period | Status | Why |
|---|---|---|
| everything up to and including **2026-09-30** (`burned_through`, `max_research_date()`) | **in-sample, burned** | The 2020-01-01..2026-06-30 panel was reused by about 28 evaluations, and the EODHD studies read data through 2026-09-30. |
| **2026-10-01** (`seal_date`) and later | **sealed forward holdout** | Never available to research code. Unsealed once per family (section 6). |

There is no historical holdout and none can be created. The "holdout" of 2024-07..2026-06 was only the Gann backtest window, and
"0.277" is the baseline 10-strategy pipeline Sharpe on that window, not a best candidate and not out-of-sample.
Hold-out reuse is unreliable (Bailey, Borwein, Lopez de Prado and Zhu, 2017). Only forward data from 2026-10-01 is clean.

## 2. How the seal is enforced (layers, strongest first)

| Layer | What it does | Binds | Who installs it |
|---|---|---|---|
| Unix user + file ACL (OD-05, OD-07) | The non-root `research` user has no read access to the post-seal locations in section 3 (traverse-only on `data/` and `data/research/`, read on the three allow roots) | research sessions | owner, as root |
| Research-user settings and deny hook (P0-04) | `Read` denies, sandbox and the `deny_holdout` hook live in the research user's own root-owned `~/.claude/settings.json` and `/etc/claude-code/hooks/` (not in the committed settings, which also bind root ops sessions) | research sessions | owner, as root (templates under `deploy/`) |
| `firm.research.data_access` | Allow-list first (`allow_roots`), then `deny_paths`, then `asof` and frame checks; fails closed; no env var, heuristic or debug flag | every new research harness | agent code |
| `firm.research.seal.install_guards()` | Sets the default-`None` `_ACCESS_GUARD` hooks in `firm.data.pit_store` (all nine `asof`/range getters, `load`, `load_macro`) and `firm.runtime` (all eight `load_*` helpers) | any process that calls it | research entry points, via `data_access` first use |
| Ledger capture (P1-12, later) | Marks backtests that never touched `firm.research` as non-promotable | all backtests | later ticket |

`install_guards()` **refuses to run** (raises `HoldoutAccessError`, logs ERROR) if `firm.live.engine` or `firm.api.app` is already imported:
a process-wide guard in a live service would make live fundamentals with `asof` after the seal raise, and that path swallows errors at debug level.
Live modules never import `firm.research` (checked by `tests/test_live_import_isolation.py` and `tests/integrity/test_holdout_guard.py`).
There is no `HOLDOUT_UNSEAL_TOKEN` environment variable and no temporary bypass; setting such a variable changes nothing (tested).

How to read data in research code:

```python
from firm.research import data_access
df = data_access.read_parquet("data/research/eodhd/etfs_full/SPY.parquet", asof="2026-09-30")
panel = data_access.load_panel("etfs_full", start="2020-01-01", end="2026-09-30", symbols=["SPY", "IEF"])
```

## 3. Post-seal data on this host (the complete list)

The services keep writing post-seal data, so the seal on this host is by ACL, deny rules and a fail-closed loader, not physical absence.
These are exactly the `deny_paths` of the config (plus the ledger inbox):

| Location | Contents | Notes |
|---|---|---|
| `data/cache` | price/fundamental caches refreshed by the live services | |
| `data_alpaca/` | Alpaca instance state: logs, `kill_switch_state.json`, orders, execution audit, dynamic universe | `FIRM_DATA_DIR` of `:8001` |
| `data/logs`, `data/live_state.db`, `data/cycle_history.json`, `data/order_history.json`, `data/execution_audit.jsonl`, `data/memory`, `data/llm_cache*.db` | IBKR instance state: prices, NAV, fills, decisions | `FIRM_DATA_DIR` of `:8000` is `data/` itself, which is why `data/` is not an allow root |
| `data/research/s2_forward` | S2 shadow forward test paper ledger | written by the frozen S2 job |
| `docs/s2_forward_snapshot.json` | S2 forward snapshot | **tracked in git**, so it is checked out into every clone and worktree; protected only by the deny rules and hook, not by ACL |
| `data/forward_monitors/` | daily forward-monitor state (P5-06) | does not exist yet; gitignored |
| `research/monitoring_sealed/` | post-seal monitor snapshots, written only by owner-run `export` subcommands | gitignored and ACL-restricted; never tracked, because a tracked file would be checked out into every research clone |
| `/local/store/research-ledger/inbox/` | one stdlib JSON line per API backtest (P1-12) | owner-provisioned; read by the owner/ops sync step, not by research code |
| service logs (`journald`, `*.log`) | everything the services print | not under any allow root |

Pre-seal research inputs that research code **may** read, only through `data_access`: `data/research/eodhd` (including `etfs_full`),
`data/research/fred`, `data/research/insider`, and `/local/store/research-ledger/returns`. Relative entries resolve against the checkout that
contains the config, and symlinks are resolved on both sides before comparing.

## 4. Monitor exceptions

Forward-test monitors read post-seal data by design. They are monitoring tools and are never research evidence.

- `exempt_monitors` in the config: `allocation_forward_test` and `s2_forward`.
- The P5-04 candidate shadow replay reads only through the monitor-only `firm.monitoring.shadow_loader` (never `data_access`), or the owner signs an
  amendment adding the candidate monitors to `exempt_monitors` (OD-04).
- The P7-03 annual review reads post-seal data only through that owner-approved monitoring path.
- Monitor output is written to `data/forward_monitors/` and `research/monitoring_sealed/`, both sealed from research code.

## 5. Residual risks (stated, not hidden)

1. **Fail-open for code that never imports `firm.research`.** A script that calls `pd.read_parquet` on a post-seal file is stopped only by the ACL
   and the research-user sandbox, not by this module. Mitigations: the research unix user (OD-07), P1-12 (such runs are marked non-promotable).
   Issues #24846 (Read deny not enforced for `.env`) and #61208 (sandbox `denyRead` not working) mean tool-level denies are not a complete control.
2. **Human knowledge of post-seal market moves cannot be avoided.** The owner and any agent session that has read repo notes or monitor output
   knows something about 2026-10 onward. Therefore every `core_v1` choice must be frozen and dated in its pre-registration **before P3-08**, with a
   declaration that no post-seal data was examined. Timestamps come from `date -u` or git, never from log lines (the host clock is Asia/Jerusalem).
3. **Tracked post-seal text.** `docs/s2_forward_snapshot.json` and some dated project notes in `docs/` and `docs/claude-memory/` were committed after
   2026-10-01 and sit in every clone.
4. **Approvals are advisory.** Per OD-06 as amended, the research user shares the owner's GitHub identity and no branch protection or second identity exists, so
   CODEOWNERS and approval-identity checks cannot be enforced; they report "unverifiable". Real enforcement is the unix user, ACLs, managed settings and
   the root-owned local pre-push hook.

## 6. Unseal rules

- Each family unseals **at most once**, no earlier than 12 months of accrued forward data (P3-10: earliest 2027-10-01), by an owner-run script.
  With 12 months the check can only reject a contradiction; it does not prove edge.
- Only the owner holds the token. `unseal_token_sha256` in the config stores its sha256; the agent never generates, sees or stores the preimage.
  **Status: PENDING, the owner has not yet set it** (`null` in the config).
- The config reserves `unseal_log` (one entry per family) and `sealed_instruments` (an optional owner-drawn instrument holdout, deferred to P2-01 / OD-02).
- Changing `seal_date`, `burned_through`, `allow_roots`, `deny_paths` or the token hash is an owner-signed, CODEOWNERS-protected change.

## 7. Open items for the owner (recorded, not blocking the code)

- Confirm the research user and ACLs (OD-05, OD-07) are in place, or record the gap here. They were **not verifiable from the agent side**.
- The research clone (`/local/store/research/ai-trading-system`) has no `data/research/` of its own; relative allow roots resolve inside the clone, so
  either point them at the live checkout's data with absolute entries, or make `data/research/eodhd` etc. in the clone symlinks to the real directories
  (roots and paths are both resolved before comparison, so that works; a symlink *inside* an allow root that points elsewhere is refused).
  P2-02 needs a decision here.
- Set `unseal_token_sha256` (or record that it stays pending).
