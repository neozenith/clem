"""Domain query operations."""

from typing import NamedTuple

from ..database.manager import DatabaseManager


class DomainStats(NamedTuple):
    """Statistics for a domain."""

    domain_id: str
    domain_path: str
    project_count: int
    session_count: int


class DomainQuery:
    """Query interface for domain operations."""

    def __init__(self, manager: DatabaseManager):
        """Initialize domain query interface.

        Args:
            manager: Database manager instance
        """
        self.manager = manager

    def list_all(self) -> list[DomainStats]:
        """List all domains with their statistics.

        Returns:
            List of domain statistics, ordered by domain path

        Example:
            >>> query = DomainQuery(manager)
            >>> domains = query.list_all()
            >>> for domain in domains:
            ...     print(f"{domain.domain_path}: {domain.project_count} projects")
        """
        results = self.manager.query(
            """
            SELECT domain_id, domain_path, project_count, session_count
            FROM domains
            ORDER BY domain_path
        """
        )

        return [DomainStats(*row) for row in results]

    def get_by_id(self, domain_id: str) -> DomainStats | None:
        """Get domain statistics by ID.

        Args:
            domain_id: Domain identifier

        Returns:
            Domain statistics or None if not found
        """
        results = self.manager.query(
            """
            SELECT domain_id, domain_path, project_count, session_count
            FROM domains
            WHERE domain_id = ?
        """,
            [domain_id],
        )

        return DomainStats(*results[0]) if results else None

    def count(self) -> int:
        """Get total number of domains.

        Returns:
            Count of domains
        """
        result = self.manager.connection.execute("SELECT COUNT(*) FROM domains").fetchone()
        return result[0] if result else 0
