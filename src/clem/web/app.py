"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import domains, projects, sessions, stats


def create_app() -> FastAPI:
    """Create and configure FastAPI application.

    Returns:
        Configured FastAPI application instance

    Example:
        >>> app = create_app()
        >>> # Run with: uvicorn app:app
    """
    app = FastAPI(
        title="CLEM API",
        description="Claude Learning & Experience Manager - Session memory and knowledge API",
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # Configure CORS for local React dev server
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://localhost:3000"],  # Vite default + CRA
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routers
    app.include_router(stats.router, prefix="/api", tags=["stats"])
    app.include_router(domains.router, prefix="/api/domains", tags=["domains"])
    app.include_router(projects.router, prefix="/api/projects", tags=["projects"])
    app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "clem-api"}

    return app
