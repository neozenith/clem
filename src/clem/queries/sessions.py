"""Session query operations."""

from typing import NamedTuple

from ..database.manager import DatabaseManager


class SessionStats(NamedTuple):
    """Statistics for a session."""

    session_id: str
    project_name: str
    domain_path: str
    event_count: int
    started_at: str | None
    file_path: str


class SessionQuery:
    """Query interface for session operations."""

    def __init__(self, manager: DatabaseManager):
        """Initialize session query interface.

        Args:
            manager: Database manager instance
        """
        self.manager = manager

    def list_all(
        self, project_name: str | None = None, domain_id: str | None = None, limit: int = 20
    ) -> list[SessionStats]:
        """List sessions with their statistics.

        Args:
            project_name: Optional project name filter
            domain_id: Optional domain filter
            limit: Maximum number of results (default: 20)

        Returns:
            List of session statistics, ordered by start time (newest first)

        Example:
            >>> query = SessionQuery(manager)
            >>> sessions = query.list_all()
            >>> sessions = query.list_all(project_name="clem", limit=10)
        """
        query_parts = [
            """
            SELECT
                s.session_id,
                p.project_name,
                d.domain_path,
                s.event_count,
                s.started_at,
                s.file_path
            FROM sessions s
            JOIN projects p ON s.project_id = p.project_id
            JOIN domains d ON s.domain_id = d.domain_id
        """
        ]

        params = []

        if project_name:
            query_parts.append("WHERE p.project_name = ?")
            params.append(project_name)
        elif domain_id:
            query_parts.append("WHERE d.domain_id = ?")
            params.append(domain_id)

        query_parts.append("ORDER BY s.started_at DESC")
        query_parts.append(f"LIMIT {limit}")

        query = " ".join(query_parts)
        results = self.manager.query(query, params if params else None)

        return [SessionStats(*row) for row in results]

    def get_by_id(self, session_id: str) -> SessionStats | None:
        """Get session statistics by ID.

        Args:
            session_id: Session identifier

        Returns:
            Session statistics or None if not found
        """
        results = self.manager.query(
            """
            SELECT
                s.session_id,
                p.project_name,
                d.domain_path,
                s.event_count,
                s.started_at,
                s.file_path
            FROM sessions s
            JOIN projects p ON s.project_id = p.project_id
            JOIN domains d ON s.domain_id = d.domain_id
            WHERE s.session_id = ?
        """,
            [session_id],
        )

        return SessionStats(*results[0]) if results else None

    def count(self, project_name: str | None = None, domain_id: str | None = None) -> int:
        """Get total number of sessions.

        Args:
            project_name: Optional project name filter
            domain_id: Optional domain filter

        Returns:
            Count of sessions
        """
        if project_name:
            result = self.manager.connection.execute(
                """
                SELECT COUNT(*)
                FROM sessions s
                JOIN projects p ON s.project_id = p.project_id
                WHERE p.project_name = ?
            """,
                [project_name],
            ).fetchone()
        elif domain_id:
            result = self.manager.connection.execute(
                "SELECT COUNT(*) FROM sessions WHERE domain_id = ?", [domain_id]
            ).fetchone()
        else:
            result = self.manager.connection.execute("SELECT COUNT(*) FROM sessions").fetchone()

        return result[0] if result else 0
