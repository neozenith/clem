"""Web API for CLEM.

FastAPI-based REST API for browsing domains, projects, and sessions.
"""

from .app import create_app

__all__ = ["create_app"]
