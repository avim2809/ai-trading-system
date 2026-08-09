"""Session-wide pytest fixtures — safety nets, not test logic.

See each fixture's docstring for what it guards against.
"""

from __future__ import annotations

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
