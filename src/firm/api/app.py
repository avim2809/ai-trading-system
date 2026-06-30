"""FastAPI application factory and entry point."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.staticfiles import StaticFiles


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
    application = FastAPI(title="AI Trading System", version="0.1.0")

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=True,
    )

    import firm.strategies  # noqa: F401 — ensure @register decorators fire at startup

    from firm.api.routers import agents, live, meta, runs
    application.include_router(meta.router, prefix="/api", tags=["meta"])
    application.include_router(runs.router, prefix="/api", tags=["runs"])
    application.include_router(agents.router, prefix="/api", tags=["agents"])
    application.include_router(live.router, prefix="/api", tags=["live"])

    try:
        from firm.api.routers import llm
        application.include_router(llm.router, prefix="/api", tags=["llm"])
    except Exception:
        pass  # llm router not available

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
    """Entry point for the ``firm-api`` console script."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


app = create_app()
