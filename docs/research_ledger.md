# Research trial ledger (P1-01)

Canonical file: `/local/store/research-ledger/trials.jsonl` (host level, outside every worktree; OD-10). Hash-chained
JSONL (`h_n = sha256(h_{n-1} + canonical_json(row))`, contiguous `seq`), appended only via
`firm.research.ledger.record_trial` under `flock`. No update/delete API exists. Returns parquet stay on the host in
`returns/` (licensed-data policy); git holds the mirror `research/ledger/trials.jsonl` plus `returns_manifest.jsonl`,
written by `scripts/sync_ledger_mirror.py` (owner/serial step; does not commit).

- Modes: `registered` (prereg hash, clean tree required), `exploratory`, `unregistered`, `legacy`. Non-registered
  modes are always recorded, on a dirty tree as `<sha>+dirty` with `config._provenance.dirty_paths_digest`.
- `scripts/backfill_legacy_trials.py` loads the 11 frozen `docs/*_trial_history.json` (210 trials) plus the signed
  census estimates (253), idempotently. Counting rule: OD-09 (`plan/OWNER_DECISIONS.md`).
- Owner provisioning: `mkdir -p /local/store/research-ledger/returns /local/store/research-ledger/inbox`, group
  `research`, mode 2775; `chattr +a` only after the first verified row.
- Tests must set `FIRM_RESEARCH_LEDGER_ROOT` (the default root is refused under pytest).
