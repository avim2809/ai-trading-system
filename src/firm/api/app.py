"""FastAPI application factory and entry point."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles

# Wired at import time (not just inside run()) so a rotating log file is
# written regardless of how the process is launched — `firm-api`, a bare
# `uvicorn firm.api.app:app`, or the Docker CMD. This is also what the
# /api/logs/tail endpoint reads from, so without it the frontend log
# monitor would have nothing to show.
from firm.logging_setup import setup_logging

# FIRM_DATA_DIR lets a second firm-api instance (same checkout) log to an
# isolated directory instead of interleaving with another instance's log —
# see the matching default in firm.api.routers.live.
_DATA_DIR = os.environ.get("FIRM_DATA_DIR", "data")
setup_logging(log_file=f"{_DATA_DIR}/logs/api.log")

log = logging.getLogger(__name__)


class SPAStaticFiles(StaticFiles):
    """StaticFiles subclass that falls back to index.html for SPA routing."""

    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                # Serve index.html for any path not matching a real file,
                # letting the React Router handle client-side routing.
                return await super().get_response(".", scope)
            raise


def create_app() -> FastAPI:
    """Build and return the configured FastAPI application."""

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        import asyncio

        async def _auto_start_live() -> None:
            # IBKR connect uses ib_async on a worker thread — must not run on
            # uvicorn's main asyncio loop (same constraint as POST /live/start).
            # See auto_start_live_with_retries's docstring for the retry/
            # alerting rationale (safety net against a 2026-07-29 boot-race
            # outage; primary fix is scripts/wait_for_ibgateway.sh).
            from firm.api.routers.live import auto_start_live_with_retries

            await auto_start_live_with_retries(application)

        task = asyncio.create_task(_auto_start_live())
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        from firm.api.routers.live import shutdown_live_engine

        await asyncio.to_thread(shutdown_live_engine, application)

    application = FastAPI(title="AI Trading System", version="0.1.0", lifespan=lifespan)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    import firm.strategies  # noqa: F401 — ensure @register decorators fire at startup

    from firm.api.routers import agents, decisions, live, logs, meta, runs, system
    application.include_router(meta.router, prefix="/api", tags=["meta"])
    application.include_router(runs.router, prefix="/api", tags=["runs"])
    application.include_router(agents.router, prefix="/api", tags=["agents"])
    application.include_router(live.router, prefix="/api", tags=["live"])
    application.include_router(logs.router, prefix="/api", tags=["logs"])
    application.include_router(decisions.router, prefix="/api", tags=["memory"])
    application.include_router(system.router, prefix="/api", tags=["system"])

    try:
        from firm.api.routers import llm
        application.include_router(llm.router, prefix="/api", tags=["llm"])
    except Exception:
        log.warning(
            "llm router unavailable — /api/llm/* endpoints will 404", exc_info=True
        )

    _instrument_metrics(application)

    dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if dist.is_dir():
        application.mount(
            "/",
            SPAStaticFiles(directory=str(dist), html=True),
            name="frontend",
        )

    return application


def _instrument_metrics(application: FastAPI) -> None:
    """Expose Prometheus metrics at ``/metrics`` when the optional dep is present.

    Provides per-endpoint latency/throughput for a local Prometheus+Grafana
    stack (cycle latency, error rates, etc.). Degrades silently when
    ``prometheus-fastapi-instrumentator`` is not installed, so the API runs
    without the ``api`` extra.
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except Exception:
        return  # metrics are optional; API works without them
    try:
        Instrumentator().instrument(application).expose(
            application, endpoint="/metrics", include_in_schema=False
        )
    except Exception:  # never let metrics wiring break app startup
        pass


def run() -> None:
    """Entry point for the ``firm-api`` console script.

    Binds to loopback by default — this process controls live trading
    (start/stop, order approval, account data) and should only ever be
    reached through a reverse proxy that adds TLS + auth, not directly.
    Set FIRM_API_HOST=0.0.0.0 explicitly (e.g. in docker-compose, where the
    container network already isolates it) if you need it to listen on all
    interfaces.
    """
    import uvicorn
    host = os.environ.get("FIRM_API_HOST", "127.0.0.1")
    port = int(os.environ.get("FIRM_API_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


app = create_app()
