"""Project query operations."""

from typing import NamedTuple

from ..database.manager import DatabaseManager


class ProjectStats(NamedTuple):
    """Statistics for a project."""

    project_name: str
    domain_id: str
    session_count: int
    cwd: str


class ProjectQuery:
    """Query interface for project operations."""

    def __init__(self, manager: DatabaseManager):
        """Initialize project query interface.

        Args:
            manager: Database manager instance
        """
        self.manager = manager

    def list_all(self, domain_id: str | None = None) -> list[ProjectStats]:
        """List all projects with their statistics.

        Args:
            domain_id: Optional domain filter

        Returns:
            List of project statistics, ordered by domain and project name

        Example:
            >>> query = ProjectQuery(manager)
            >>> projects = query.list_all()
            >>> projects = query.list_all(domain_id="play")
        """
        if domain_id:
            results = self.manager.query(
                """
                SELECT project_name, domain_id, session_count, cwd
                FROM projects
                WHERE domain_id = ?
                ORDER BY domain_id, project_name
            """,
                [domain_id],
            )
        else:
            results = self.manager.query(
                """
                SELECT project_name, domain_id, session_count, cwd
                FROM projects
                ORDER BY domain_id, project_name
            """
            )

        return [ProjectStats(*row) for row in results]

    def get_by_id(self, project_id: str) -> ProjectStats | None:
        """Get project statistics by ID.

        Args:
            project_id: Project identifier

        Returns:
            Project statistics or None if not found
        """
        results = self.manager.query(
            """
            SELECT project_name, domain_id, session_count, cwd
            FROM projects
            WHERE project_id = ?
        """,
            [project_id],
        )

        return ProjectStats(*results[0]) if results else None

    def count(self, domain_id: str | None = None) -> int:
        """Get total number of projects.

        Args:
            domain_id: Optional domain filter

        Returns:
            Count of projects
        """
        if domain_id:
            result = self.manager.connection.execute(
                "SELECT COUNT(*) FROM projects WHERE domain_id = ?", [domain_id]
            ).fetchone()
        else:
            result = self.manager.connection.execute("SELECT COUNT(*) FROM projects").fetchone()

        return result[0] if result else 0
