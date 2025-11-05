"""FastAPI application factory."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routers import domains, events, projects, sessions, stats


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
    app.include_router(events.router, prefix="/api/sessions", tags=["events"])

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "clem-api"}

    # Serve static frontend files
    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists():
        # Mount static assets (JS, CSS, images)
        app.mount("/assets", StaticFiles(directory=frontend_dir / "assets"), name="assets")

        # Serve index.html for root path
        @app.get("/")
        async def serve_root():
            """Serve frontend application root."""
            index_file = frontend_dir / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            raise HTTPException(status_code=404, detail="Frontend not found")

        # Catch-all route to serve index.html for client-side routing
        @app.get("/{full_path:path}")
        async def serve_frontend(full_path: str):
            """Serve frontend application for all non-API routes."""
            # Serve index.html for all non-API routes (client-side routing)
            if not full_path.startswith("api/"):
                index_file = frontend_dir / "index.html"
                if index_file.exists():
                    return FileResponse(index_file)
            # If we get here, return 404
            raise HTTPException(status_code=404, detail="Not found")

    return app
