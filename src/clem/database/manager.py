"""Database connection manager for CLEM.

Handles DuckDB connection lifecycle with lazy initialization.
"""

from pathlib import Path
from typing import Any, Optional
import duckdb

from ..config import get_database_path


class DatabaseManager:
    """Manages DuckDB connection lifecycle.

    Features:
    - Lazy connection initialization
    - Context manager support
    - Query execution wrapper
    - Automatic cleanup

    Example:
        >>> with DatabaseManager() as manager:
        ...     results = manager.execute("SELECT * FROM domains")
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database manager.

        Args:
            db_path: Path to database file. If None, uses default from config.
        """
        self.db_path = db_path or get_database_path()
        self._connection: Optional[duckdb.DuckDBPyConnection] = None

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection (lazy initialization)."""
        if self._connection is None:
            self._connection = duckdb.connect(str(self.db_path))
        return self._connection

    def execute(self, query: str, params: Optional[dict] = None) -> duckdb.DuckDBPyRelation:
        """Execute a SQL query.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            DuckDB relation result

        Example:
            >>> manager.execute("SELECT * FROM domains WHERE domain_id = ?", {'id': 'play'})
        """
        if params:
            return self.connection.execute(query, params)
        return self.connection.execute(query)

    def query(self, query: str, params: Optional[dict] = None) -> Any:
        """Execute query and fetch all results.

        Args:
            query: SQL query string
            params: Optional query parameters

        Returns:
            Query results as Python objects
        """
        result = self.execute(query, params)
        return result.fetchall()

    def close(self) -> None:
        """Close database connection."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def __enter__(self) -> 'DatabaseManager':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()

    def database_exists(self) -> bool:
        """Check if database file exists."""
        return self.db_path.exists()

    def delete_database(self) -> None:
        """Delete the database file (nuclear option).

        Warning: This destroys all cached data. Can be rebuilt from source.
        """
        self.close()
        if self.db_path.exists():
            self.db_path.unlink()
