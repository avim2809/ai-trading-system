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

    dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if dist.is_dir():
        application.mount(
            "/",
            SPAStaticFiles(directory=str(dist), html=True),
            name="frontend",
        )

    return application


def run() -> None:
    """Entry point for the ``firm-api`` console script."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


app = create_app()
