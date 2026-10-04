# P1-12 drafts (owner actions)

The integrity test directory is human-owned (P0-04), so these two files are drafts. Commit them there unchanged:

- `test_entry_points_wrapped.py`
- `test_host_ledger_untouched.py` (its session-scoped autouse fixture only guards the whole suite if it
  lives in a `conftest.py` that applies to every test; as a plain test module it guards the session it
  is collected in. Move `_host_ledger_guard` into `tests/conftest.py` if you want a full-suite guard.)

Both were run from this location against the branch (the repo root is found by walking up to `src/firm`).
