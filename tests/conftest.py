"""Session-wide pytest fixtures — safety nets, not test logic.

See each fixture's docstring for what it guards against.
"""

from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import pytest

# --- Hermetic test environment -------------------------------------------------
# The suite must give the same result on a clean clone (CI, a restricted user,
# a research worktree) as on the production checkout, so it may neither need
# nor be able to read the developer's real .env or secrets. Must run before
# any test module imports litellm or builds Settings.
#
# * LITELLM_MODE=PRODUCTION: litellm otherwise calls load_dotenv() at import,
#   which walks up from the venv's location and opens the first .env it finds
#   (the live checkout's, when the venv lives there).
# * Placeholder broker keys: Settings.require("alpaca_api_key") is exercised by
#   the /api/live/start tests. Unconditional on purpose, so tests never run
#   with real credentials even when a real .env is present. The values are
#   obviously fake and no test may rely on them reaching a network.
# * MPLCONFIGDIR: matplotlib writes its cache under $HOME, which a restricted
#   user may not own.
os.environ["LITELLM_MODE"] = "PRODUCTION"
os.environ["ALPACA_API_KEY"] = "test-alpaca-key-not-real"
os.environ["ALPACA_SECRET_KEY"] = "test-alpaca-secret-not-real"
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig-"))


@pytest.fixture(autouse=True)
def _no_real_pipeline_warmup(request):
    """Prevent the real pipeline-warmup background thread from running.

    Confirmed live: that thread's sklearn/hmmlearn fit (firm.live.
    pipeline_warmup._warm_hmm) imports threadpoolctl, which enumerates
    loaded shared libraries via dl_iterate_phdr() — a call that holds the
    dynamic linker's internal lock. Any test that concurrently triggers a
    fresh module import on another thread (e.g. a TestClient booting
    uvicorn/watchfiles for the first time) can deadlock: the import holds
    the linker lock via dlopen() while dl_iterate_phdr() waits on it, and
    vice versa. Reproduced running the full suite (py-spy dump showed the
    main thread blocked in importlib, two "pipeline-warmup" threads blocked
    inside threadpoolctl's library scan).

    Warmup is a pure production startup-latency optimization — pre-loading
    heavy deps before the first live cycle — with no test-observable
    effect, so tests carried the deadlock risk with none of the benefit.
    Excluded for test_pipeline_warmup.py itself, which exercises this
    exact code directly (including the real HMM fit in
    test_warm_hmm_probe_fits) and never spawns the background thread with
    unmocked work — see that file's tests for why each is already safe.
    """
    if request.module.__name__ == "tests.test_pipeline_warmup":
        yield
        return
    with (
        patch("firm.live.pipeline_warmup._warm_hmm"),
        patch("firm.live.pipeline_warmup._warm_rag_imports"),
    ):
        yield


@pytest.fixture(autouse=True)
def _no_real_fundamentals_refresh(request):
    """Stop tests from starting the real background fundamentals refresh.

    ``POST /api/live/start`` calls ``maybe_refresh_fundamentals_cache_on_start``,
    which spawns a thread that fetches real vendor fundamentals (FMP /
    AlphaVantage, real API keys from ``.env``) and merges them into
    ``settings.data.cache_dir`` whenever that cache looks stale. Found
    2026-09-29 running the suite from a fresh git worktree (no ``data/cache``,
    so always "stale"): the API tests made real AlphaVantage calls (one got an
    HTTP 503), and the leaked threads then called
    ``test_refresh_fundamentals_cache_writes_parquet``'s patched
    ``ParquetCache.put`` 31 times, failing it. From the live checkout the same
    thread would write into the running instances' real ``data/cache`` whenever
    that cache ages past its refresh window. Excluded for
    test_fundamentals_refresh.py, which exercises this code directly and
    patches ``_refresh_in_background`` itself where needed.
    """
    if request.module.__name__ == "tests.test_fundamentals_refresh":
        yield
        return
    with patch("firm.live.fundamentals_refresh._refresh_in_background"):
        yield


@pytest.fixture(autouse=True)
def _isolated_execution_audit(tmp_path):
    """Keep tests out of the live instance's real execution audit log.

    ``firm.live.execution_safety.audit_path`` defaults to
    ``$FIRM_DATA_DIR/execution_audit.jsonl`` -- i.e. ``data/`` in this
    checkout, which is the running IBKR instance's own immutable decision
    log. Only a handful of tests redirected it, so the suite had appended
    ~9k fixture records (``broker_type`` "" and fake ``alpaca_paper``
    orders) into production's audit trail (found 2026-09-28). Tests that
    set ``FIRM_EXECUTION_AUDIT`` themselves still win (their setenv runs
    after this fixture).

    Deliberately does NOT request ``monkeypatch``: an autouse conftest
    fixture that does forces monkeypatch to be set up before -- and so torn
    down after -- every module-level autouse fixture, which broke modules
    whose own teardown relies on a test's ``monkeypatch.setattr`` already
    being undone (tests/test_xgb_inference.py's ``reset_cache()`` called
    ``.cache_clear()`` on a still-patched lambda: 22 errors).
    """
    with patch.dict(os.environ, {"FIRM_EXECUTION_AUDIT": str(tmp_path / "execution_audit.jsonl")}):
        yield


@pytest.fixture(autouse=True)
def _isolated_research_ledger(tmp_path):
    """Keep tests out of the canonical host research ledger (P1-12).

    Wrapped backtest entry points append one capture line per outermost call, and the armed path
    writes ``unregistered`` rows into the append-only hash-chained ledger (no delete API), which
    would irreversibly inflate N on every full-suite run. Redirects ``FIRM_RESEARCH_LEDGER_ROOT``
    to a temp dir beside ``tmp_path`` (``inbox/`` and ``returns/`` pre-created) and resets the capture module state
    afterwards. Same ``patch.dict`` style as ``_isolated_execution_audit`` (deliberately no
    ``monkeypatch``). Tests that set the env var themselves still win.
    """
    import sys

    # A SIBLING of tmp_path, not a child: tests that assert on tmp_path's exact contents
    # (e.g. test_allocation_forward_monitor) must not see the pre-created ledger dirs.
    root = tmp_path.parent / f"{tmp_path.name}-research-ledger"
    (root / "inbox").mkdir(parents=True)
    (root / "returns").mkdir()
    try:
        with patch.dict(os.environ, {"FIRM_RESEARCH_LEDGER_ROOT": str(root)}):
            yield
    finally:
        cs = sys.modules.get("firm.backtest._capture_state")
        if cs is not None:
            cs.IN_API_PROCESS = False
            cs.LEDGER_SINK = None
