# EODHD cleaning v3 (frozen, OD-14)

Module: `src/firm/data/cleaning.py` (`clean_bars_v3`, `equity_calendar_v3`, `cleaning_fingerprint_v3`).
v2 (`scripts/eodhd_clean.py`, fingerprint `fc0690f0...3054`) is untouched and stays the rule for frozen preregistrations.

- **Fingerprint v3:** `2deb690edb2db0bd4eb64c06e61bc25a37d6534da718dcc07472083e6cb0c0af`
- **Freeze commit (module + tests):** `c721d3213be862029df1cccfde2ccdfc6ce79eb4` (branch `batch2/c`)
- **`frozen_at` in the rule dict:** `2026-10-03T22:31:16Z` (`date -u` at authoring); this doc written `2026-10-03T22:33:03Z`.
- Any change to `CLEANING_RULES_V3` changes the fingerprint and fails `tests/test_cleaning_v3.py::test_fingerprint_v3_is_frozen`.

## Rules (v2 inherited unchanged, plus)
1. Sentinel: drop bars with any price column within 1e-6 of 999999.9999 (`sentinel`); drop runs of >= 5 identical `adjusted_close` >= 1e5 when the asset's median price < 1e4 (`constant_run`).
2. `asset="futures"`: NaN prices dropped, negative/zero kept; NaN volume dropped, zero kept; no calendar.
3. Futures series with a non-positive adjusted close: spike reversal on differences scaled by median |adjusted_close| (thresholds as v2). Positive series use the v2 level rule.
4. `equity_calendar_v3` takes SPY dates as an argument, drops dates >= 2026-10-01 (seal) and, if given, dates > `asof`.

## Caveats for the owner
- Rule 3 removes a genuine one-day crash that fully reverts within 5 bars (e.g. crude 2020-04-20 if the next days return near 18) as a bad print; a collapse that does not revert is kept. Test: `test_futures_negative_series_spike_uses_differences`.
- The seal bound is a local constant (`firm.research` is not on this base); P2-02 should tighten it to `max_research_date()`.
- Not run on real data in this ticket.
