"""Database package for CLEM.

Manages DuckDB connection, schema, and data operations.
Database is treated as disposable cache.
"""

from .manager import DatabaseManager

__all__ = ['DatabaseManager']
