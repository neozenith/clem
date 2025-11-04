"""CLEM - Claude Learning & Experience Manager.

AI-powered session memory and knowledge extraction from Claude Code sessions.
"""

__version__ = "0.1.0"

from .database.builder import DatabaseBuilder
from .database.manager import DatabaseManager

__all__ = ["DatabaseBuilder", "DatabaseManager", "__version__"]
