"""Run guards shared by the two core_v1 drivers: seal preflight, run window, memory cap. Hermetic: injected euid / access / clock."""

from __future__ import annotations

import datetime as dt
import resource
from pathlib import Path

import pytest

from firm.research import run_guards as G

UTC = dt.UTC


def at(y, mo, d, h, mi=0):
    return dt.datetime(y, mo, d, h, mi, tzinfo=UTC)


# 2026-10-05 is a Monday. October: US/Eastern is EDT = UTC-4 until 2026-11-01, then EST = UTC-5.
@pytest.mark.parametrize(
    "when,ok",
    [
        (at(2026, 10, 6, 0, 30), True),    # Mon 20:30 ET (Tue 00:30 UTC)
        (at(2026, 10, 6, 7, 59), True),    # Tue 03:59 ET
        (at(2026, 10, 6, 8, 0), False),    # Tue 04:00 ET: window closed
        (at(2026, 10, 6, 15, 0), False),   # Tue 11:00 ET: market hours
        (at(2026, 10, 6, 23, 59), False),  # Tue 19:59 ET
        (at(2026, 10, 7, 0, 0), True),     # Tue 20:00 ET
        (at(2026, 10, 10, 6, 0), True),    # Sat 02:00 ET (still Fri evening's window; Saturday UTC)
        (at(2026, 10, 10, 15, 0), True),   # Saturday 11:00 ET
        (at(2026, 10, 10, 23, 59), True),  # Sat 19:59 ET == Sat 23:59 UTC: still Saturday in both
        (at(2026, 10, 11, 0, 0), False),   # Sat 20:00 ET == Sunday 00:00 UTC: BTC review day, refused
        (at(2026, 10, 11, 12, 0), False),  # Sunday
        (at(2026, 10, 12, 3, 0), False),   # Sunday 23:00 ET / Monday 03:00 UTC
        (at(2026, 10, 12, 20, 0), False),  # Mon 16:00 ET
        (at(2026, 10, 9, 23, 0), False),   # Fri 19:00 ET
        (at(2026, 10, 10, 0, 0), True),    # Fri 20:00 ET (Sat 00:00 UTC)
    ],
)
def test_window_ok(when, ok):
    got, why = G.window_ok(when)
    assert got is ok, why


def test_window_uses_winter_time_after_dst_ends():
    # 2026-11-03 is a Tuesday; EST = UTC-5: 20:00 ET = 01:00 UTC on the 4th, 04:00 ET = 09:00 UTC
    assert G.window_ok(at(2026, 11, 4, 1, 0))[0] is True
    assert G.window_ok(at(2026, 11, 4, 9, 0))[0] is False
    assert G.window_ok(at(2026, 11, 3, 0, 30))[0] is False   # Mon 19:30 EST


def test_window_rejects_naive_datetime():
    with pytest.raises(ValueError):
        G.window_ok(dt.datetime(2026, 10, 6, 1, 0))  # noqa: DTZ001 - naive on purpose


def test_assert_in_window_raises_with_injected_clock():
    with pytest.raises(G.PreflightError, match="window"):
        G.assert_in_window(clock=lambda: at(2026, 10, 6, 15, 0))
    G.assert_in_window(clock=lambda: at(2026, 10, 7, 1, 0))


def test_current_utc_parses_date_u_output(monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="1791000000\n", stderr=""))
    assert G.current_utc() == dt.datetime.fromtimestamp(1791000000, tz=UTC)


def test_there_is_no_way_to_bypass_the_window():
    import inspect

    assert not any("allow" in p or "override" in p or "skip" in p for p in inspect.signature(G.assert_in_window).parameters)


# ---- seal preflight --------------------------------------------------------------------------------------------------------
PASS_DOC = """## 4. Red-team matrix
| # | Attempt | Expected | Actual (UTC) | Result |
|---|---|---|---|---|
| a | read it | denied | 2026-10-07T00:00Z denied | PASS |
| b | edit it | denied | x | PASS |
**Ops (root) session checks**
| # | Attempt | Expected | Actual (UTC) | Result |
|---|---|---|---|---|
| o1 | tail | allowed | y | PASS |
"""
NOT_RUN_DOC = PASS_DOC.replace("| b | edit it | denied | x | PASS |", "| b | edit it | denied |  | NOT RUN |")
FAIL_DOC = PASS_DOC.replace("| a | read it | denied | 2026-10-07T00:00Z denied | PASS |", "| a | read it | denied | read ok | FAIL |")


@pytest.mark.parametrize("doc,ok", [(PASS_DOC, True), (NOT_RUN_DOC, False), (FAIL_DOC, False), ("no table here", False)])
def test_redteam_passed(tmp_path, doc, ok):
    p = tmp_path / "GUARDRAIL_REDTEAM.md"
    p.write_text(doc)
    assert G.redteam_passed(p) is ok


def test_redteam_passed_missing_file_is_false(tmp_path):
    assert G.redteam_passed(tmp_path / "nope.md") is False


def _mk(tmp_path: Path, doc: str = PASS_DOC) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "GUARDRAIL_REDTEAM.md").write_text(doc)
    return tmp_path


def _access(readable: set[str]):
    """os.access stand-in: a path is readable iff it ends with one of the given repo-relative suffixes."""

    def access(path, mode):
        return any(str(path).rstrip("/").endswith(r) for r in readable)

    return access


ETF = G.ETF_STORE


def test_preflight_passes_when_everything_holds(tmp_path):
    repo = _mk(tmp_path)
    rep = G.seal_preflight(repo, euid=lambda: 1001, access=_access({ETF}))
    assert rep["euid"] == 1001 and rep["etf_store_readable"] is True and rep["redteam_pass"] is True


def test_preflight_refuses_root(tmp_path):
    repo = _mk(tmp_path)
    with pytest.raises(G.PreflightError, match="root"):
        G.seal_preflight(repo, euid=lambda: 0, access=_access({ETF}))


@pytest.mark.parametrize("leak", list(G.SEALED_PATHS))
def test_preflight_refuses_when_a_sealed_path_is_readable(tmp_path, leak):
    repo = _mk(tmp_path)
    with pytest.raises(G.PreflightError, match="readable"):
        G.seal_preflight(repo, euid=lambda: 1001, access=_access({ETF, leak}))


def test_preflight_refuses_when_etf_store_unreadable(tmp_path):
    repo = _mk(tmp_path)
    with pytest.raises(G.PreflightError, match="etfs_full"):
        G.seal_preflight(repo, euid=lambda: 1001, access=_access(set()))


def test_preflight_refuses_without_a_recorded_redteam_pass(tmp_path):
    repo = _mk(tmp_path, NOT_RUN_DOC)
    with pytest.raises(G.PreflightError, match="GUARDRAIL_REDTEAM"):
        G.seal_preflight(repo, euid=lambda: 1001, access=_access({ETF}))


def test_preflight_checks_every_ticket_listed_sealed_path(tmp_path):
    repo = _mk(tmp_path)
    seen: list[str] = []

    def access(path, mode):
        seen.append(str(path))
        return str(path).endswith("etfs_full")

    G.seal_preflight(repo, euid=lambda: 1001, access=access)
    assert len(G.SEALED_PATHS) == 6
    for frag in (*G.SEALED_PATHS, G.ETF_STORE):
        assert any(s.endswith(frag) for s in seen), frag


# ---- memory cap ------------------------------------------------------------------------------------------------------------
def test_memory_cap_required():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    with pytest.raises(G.PreflightError, match="memory cap"):
        G.require_memory_cap(limit=lambda: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    assert G.require_memory_cap(limit=lambda: (2_500_000_000, 2_500_000_000)) == 2_500_000_000
    with pytest.raises(G.PreflightError, match="memory cap"):
        G.require_memory_cap(limit=lambda: (64 * 2**30, 64 * 2**30))  # a "cap" above the ceiling is not a cap
    assert (soft, hard) == resource.getrlimit(resource.RLIMIT_AS)  # nothing changed the process limits
