"""Session-wide pytest fixtures — safety nets, not test logic.

See each fixture's docstring for what it guards against.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


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
