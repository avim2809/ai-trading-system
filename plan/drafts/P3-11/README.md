# P3-11 / P3-08 owner actions: install and approve the core_v1 pre-registration

Draft: `core_v1_prereg_DRAFT.yaml` (schema-valid except the two owner fields `approver`, `approved_at_utc`; validated 2026-10-05
against `firm.research.prereg` on a temp copy: 12 grid configs, all covered, gates hash current, an off-grid value or an extra `tau`
key is rejected by `covers()` and `is_approved()`).

Order matters (`is_approved` and gate 8 need the charter commit to be older than `approved_at_utc`):

1. Write and sign the charter (`plan/drafts/P5-02/charter_core_v1_DRAFT.md` -> `research/charters/core_v1.md`; mechanism in your own
   words, tau confirmed, expected correlation, falsification). Commit it FIRST. Note its commit time:
   `git log -1 --format=%cI -- research/charters/core_v1.md`.
2. Review the grid (forecast_cap x buffer_fraction x speed_subset = 2 x 3 x 2). Change values now or never, not after any run. tau is not
   a dimension. `covers()` rejects any config key outside the grid, so tau, gross cap etc. must not be passed in the params of a
   registered run (they live in the frozen module).
3. Confirm `sha256sum config/gates.yaml` still equals the draft's `gates_hash` (else update the field). Confirm the three sha256 values in
   the draft's header comment (universe, stress periods, tax) still match.
4. Create and freeze `scripts/core_v1_preregistered.py` (P3-11 deliverable: `bars_fingerprint()`, GRID, seed, embargo, the hashes from
   step 3, expected ledger-row counts per kind). Not part of this job.
5. Copy the draft to `research/preregistration/<YYYYMMDD>_core_v1.yaml`; set `approver` and `approved_at_utc`
   (`date -u +%Y-%m-%dT%H:%M:%SZ`, later than the charter commit time). Check:
   `python -c "from firm.research import prereg as P; s=P.load_spec('<file>'); print(P.validate_spec(s), P.spec_hash(s))"` (expect `[]`).
6. Add the entry to `research/preregistration/INDEX.yaml`: `family: core_v1, kind: yaml, yaml_path: <file>, fingerprint: <spec_hash>,
   status: APPROVED, fingerprint_source: none, module_path: null, ledger_path: null`, plus `freeze_commit` / `freeze_time_utc` (git
   committer time, UTC). Run `verify_index` (expect `[]`). Commit YAML + INDEX in one CODEOWNERS-reviewed commit. `futures_trend` stays
   SUPERSEDED (OD-13).
7. Any later edit to the YAML or the gates file changes a hash and invalidates approval: that is a new pre-registration.

## Notes on the frozen module (w15a re-freeze, 2026-10-06)

- `scripts/core_v1_preregistered.py` was re-frozen (commit a305ea6; fingerprint `9fe358a8...a56e` in the draft INDEX entry in
  `plan/drafts/P1-09/INDEX.yaml`). The earlier w14a freeze (fingerprint 495e6d70...) is superseded; it never ran anything.
- Instrument weights: handcrafted, ONE GROUP PER ASSET CLASS (owner decision 2026-10-05, addendum "P4-01 handcrafting tree"): equal across the
  10 asset classes of the universe config, equal within each class, via `firm.portfolio.weights.handcraft_weights`. The weight-vector hash is
  frozen and pinned in `tests/test_core_v1_preregistered.py`. The charter must state the same scheme.
- Trial counts: the frozen gates `n_rule` defines the FAMILY N = legacy 19 (trend 1, alt_premia 10, S3 5, S5 3, futures_trend 0) + core_v1 grid
  up to 12 = 31 (`n_counts.family_provisional`). Estimation, robustness, cost-stress, stress-suite, benchmark and comparison rows (110 expected)
  are diagnostics: RAW count only, never family N. The module constants `FAMILY_N*`, `DIAGNOSTIC_ROWS_RAW_ONLY`, `NEW_RAW_ROWS` say so, and a
  test derives them from the gates file.
- Owner action: the real (protected) prereg INDEX has no `core_v1` frozen_module entry, so
  `tests/test_prereg.py::test_every_frozen_prereg_module_is_indexed` fails until the owner copies the draft INDEX entry across
  (freeze_commit is valid only if this branch is merged without squashing; otherwise recompute it).

## Notes on the gates-v2 refresh (w16a, 2026-10-06)

- Owner Amendment 1 (2026-10-06) changed the gates file (sha256 now `bcecaec9...1540`, see the draft `gates_hash`): the gate-7 benchmark is the
  pinned ANNUAL BM2 variant; monthly stays a reported sensitivity. `scripts/core_v1_preregistered.py` records the new hash and derives
  `BENCHMARK_GATE_VARIANT` / `BENCHMARK_SENSITIVITY_VARIANT` from the gates file (refuses if the pin is missing). Both enter the fingerprint.
- Re-frozen at commit f5f9b01 (2026-10-06T08:05:53Z); new fingerprint `7826fb03...b46a` (the w15a value `9fe358a8...` is superseded; it never ran
  anything). Weights scheme, seed 20261005 and family-N semantics are unchanged. Universe, stress and tax hashes were re-verified unchanged.
- Step 3 above applies to the new hash. Stale elsewhere: `plan/drafts/P5-02/charter_core_v1_DRAFT.md` front matter still carries the old
  `gates_yaml_sha256` (a2ad5245...); the charter must record `bcecaec9...` or `verify_charter` will refuse it. Owner action.
- Owner action (unchanged): index the module in the protected prereg INDEX (draft entry in `plan/drafts/P1-09/INDEX.yaml`, freeze_commit valid only
  without squashing), else `test_every_frozen_prereg_module_is_indexed` and `test_every_frozen_module_indexed` keep failing.

## Notes on the batch-20 RE-FREEZE (w20a, 2026-10-06, before any real run) and the exact owner steps

Why: the approved freeze (module fingerprint `7826fb03...`, YAML spec hash `23ab08a4...`, 0 real trials, never run) left `min_active_fraction`
(gate 5) and `vol_ewma_span` / `max_vol_scale` / `instrument_risk_cap_multiple` (gate 6) unfrozen, so a Tier A was unreachable by construction. This
is a NEW pre-registration; nothing was run on real data and no value was chosen from performance. Full list with sources: `reports/w20a_report.md`
and the comments in `scripts/core_v1_preregistered.py`.

Frozen values (owner-confirm the two conventions NOW; after any run they are fixed):

| value | frozen as | non-performance source |
|---|---|---|
| `MIN_ACTIVE_FRACTION` (gate 5) | 0.5 | simple majority of the 14-ETF universe; any 7 ETFs span >= 3 asset classes (ceil(1/0.40) = 3, `config/risk.yaml`). Convention, **owner-confirm** |
| `VOL_EWMA_SPAN` (gate 6) | 252 | P4-03 "slow EWMA"; one year vs the 35-day instrument span. Convention, **owner-confirm** |
| `MAX_VOL_SCALE` | 1.5 | gates `robustness_parameters` (= `config/risk.yaml`) |
| `INSTRUMENT_RISK_CAP_MULTIPLE` | 2.0 | gates `robustness_parameters` (= `config/risk.yaml`) |
| class risk cap (0.40) | NOT applied (1.0) | not a gate-6 parameter; applying it would be an unperturbed parameter. **owner-confirm** |

Unchanged: grid (12), weights scheme (handcrafted, one group per asset class), seed 20261005, family N 31, universe, tau (charter), gates hash
`bcecaec9...`, annual gate-7 variant, expected ledger rows (robustness 54: all 27 parameters now perturbable; total 122).
The charter needs NO edit (it carries the gates hash only, and the gates file is untouched).

Owner steps, in order:

1. Review the five values above. Change them now or never (a change = another new freeze). Optionally merge the branch
   `batch20/core-v1-refreeze` WITHOUT squashing (the INDEX `freeze_commit` is only valid if its history survives).
2. Copy `plan/drafts/P3-11/core_v1_prereg_DRAFT.yaml` to `research/preregistration/<YYYYMMDD>_core_v1.yaml` with today's date (NOT the existing
   `20261006_core_v1.yaml`, which stays on disk untouched; its stem must differ). Fill `approver` and `approved_at_utc` (`date -u +%Y-%m-%dT%H:%M:%SZ`,
   later than the charter's commit time). Check: `python -c "from firm.research import prereg as P; s=P.load_spec('<file>'); print(P.validate_spec(s), P.spec_hash(s))"`
   (expect `[]` and `b964385057cf7a7ef25b5f14f36ce2551f0511adff86dbca968ce1874a9bdbf0`; a different hash means a field was edited).
3. Edit `research/preregistration/INDEX.yaml` (family names are unique, so the old entries are renamed and the new ones reuse the names). Use
   `plan/drafts/P1-09/INDEX.yaml` as the template:
   a. the old YAML entry `family: core_v1` (spec hash `23ab08a4...`, file `20261006_core_v1.yaml`): rename to `core_v1_20261006`, `status: SUPERSEDED`,
      notes "superseded-by core_v1 re-freeze 2026-10-06 (0 trials, never run)". Keep every other field (the file is immutable).
   b. the module entry `core_v1_module`: replace `fingerprint`, `freeze_commit`, `freeze_time_utc` and notes with the values in the draft INDEX
      (the module path stays; the old fingerprint `7826fb03...` is recorded in the notes; one file has one fingerprint at HEAD).
   c. add the new YAML entry `family: core_v1`, `kind: yaml`, the new `yaml_path`, `fingerprint` = the spec hash from step 2, `freeze_commit` / `freeze_time_utc`
      = the commit that adds the YAML and its committer time in UTC (`git log -1 --format=%cI <commit>`), `fingerprint_source: none`, `status: APPROVED`.
4. Run `python -c "from firm.research import prereg as P; print(P.verify_index(P.load_index()))"` (expect `[]`) and commit YAML + INDEX in one
   CODEOWNERS-reviewed commit. The drivers pass `prereg='core_v1'`, which resolves to the new entry.
5. Everything else in `reports/w19a_report.md` section 5 (ACL, red-team rows, integrity-test drafts, trial-history baseline, ADAPTERS entry)
   still applies unchanged.
