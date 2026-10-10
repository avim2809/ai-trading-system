import pytest

from firm.research import run_guards as G


def test_preflight_judges_sealed_paths_and_etf_store_under_data_root(tmp_path):
    wt, live = tmp_path / "wt", tmp_path / "live"
    (wt / "docs").mkdir(parents=True)
    (wt / "docs" / "GUARDRAIL_REDTEAM.md").write_text("| a | x | y | z | PASS |\n")

    def access(p, m):   # worktree copies are readable; the live sealed paths are not; the live ETF store is
        return p.startswith(str(wt)) or p == str(live / G.ETF_STORE)

    got = G.seal_preflight(wt, data_root=live, euid=lambda: 1000, access=access)
    assert got["etf_store_readable"] is True
    with pytest.raises(G.PreflightError):
        G.seal_preflight(wt, euid=lambda: 1000, access=access)   # without data_root the worktree copies count as leaks
