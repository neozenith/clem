"""Database builder for CLEM.

Rebuilds the database from source (Claude Code session files).
"""

from datetime import datetime
from pathlib import Path

import duckdb

from ..config import SCHEMA_VERSION, get_database_path
from ..core.domain import generate_domain_id
from ..core.project import ProjectInfo, discover_projects, get_unique_domains
from .manager import DatabaseManager
from .schema import drop_all_tables, init_schema


class DatabaseBuilder:
    """Builds and rebuilds the CLEM database from source.

    Database is disposable cache - can be deleted and rebuilt at any time.
    """

    def __init__(self, db_path: Path | None = None):
        """Initialize database builder.

        Args:
            db_path: Path to database file. If None, uses default.
        """
        self.db_path = db_path or get_database_path()
        self.manager = DatabaseManager(self.db_path)

    def rebuild(self, full: bool = True) -> None:
        """Rebuild database from source files.

        Steps:
        1. Drop all tables (if full=True)
        2. Create schema
        3. Discover projects and domains
        4. Populate domains table
        5. Populate projects table
        6. Populate sessions table
        7. Update metadata

        Args:
            full: If True, nuclear rebuild (drop all tables first).
                  If False, incremental (preserve existing data).

        Example:
            >>> builder = DatabaseBuilder()
            >>> builder.rebuild(full=True)
        """
        print(f"{'🔥 Nuclear' if full else '🔄 Incremental'} rebuild starting...")

        conn = self.manager.connection

        if full:
            print("  Dropping all tables...")
            drop_all_tables(conn)

        print("  Creating schema...")
        init_schema(conn)

        print("  Discovering projects...")
        projects = discover_projects()
        domains = get_unique_domains(projects)

        print(f"  Found: {len(projects)} projects across {len(domains)} domains")

        print("  Populating domains...")
        self._populate_domains(domains)

        print("  Populating projects...")
        self._populate_projects(projects)

        print("  Populating sessions...")
        session_count = self._populate_sessions(projects)

        print(f"  Indexed: {session_count} sessions")

        # Update metadata
        conn.execute(
            """
            INSERT INTO metadata (key, value)
            VALUES ('last_rebuild', ?)
            ON CONFLICT (key) DO UPDATE SET
                value = excluded.value,
                updated_at = NOW()
        """,
            [datetime.now().isoformat()],
        )

        conn.execute(
            """
            INSERT INTO metadata (key, value)
            VALUES ('sessions_indexed', ?)
            ON CONFLICT (key) DO UPDATE SET
                value = excluded.value,
                updated_at = NOW()
        """,
            [str(session_count)],
        )

        print("\n✅ Rebuild complete!")
        print(f"   Database: {self.db_path}")
        print(f"   Domains: {len(domains)}")
        print(f"   Projects: {len(projects)}")
        print(f"   Sessions: {session_count}")

    def _populate_domains(self, domains: dict[str, list[ProjectInfo]]) -> None:
        """Populate domains table.

        Args:
            domains: Dictionary mapping domain_path to list of projects
        """
        conn = self.manager.connection

        for domain_path, projects_in_domain in domains.items():
            domain_id = generate_domain_id(domain_path)

            conn.execute(
                """
                INSERT INTO domains (
                    domain_id,
                    domain_path,
                    project_count,
                    session_count,
                    last_scan
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (domain_id) DO UPDATE SET
                    project_count = excluded.project_count,
                    session_count = excluded.session_count,
                    last_scan = excluded.last_scan
            """,
                [
                    domain_id,
                    domain_path,
                    len(projects_in_domain),
                    sum(len(p.session_files) for p in projects_in_domain),
                    datetime.now(),
                ],
            )

    def _populate_projects(self, projects: dict[str, ProjectInfo]) -> None:
        """Populate projects table.

        Args:
            projects: Dictionary mapping project_id to ProjectInfo
        """
        conn = self.manager.connection

        for project_info in projects.values():
            domain_id = generate_domain_id(project_info.domain_path)

            conn.execute(
                """
                INSERT INTO projects (
                    project_id,
                    project_name,
                    domain_id,
                    cwd,
                    claude_project_id,
                    session_count,
                    last_scan
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (project_id) DO UPDATE SET
                    session_count = excluded.session_count,
                    last_scan = excluded.last_scan
            """,
                [
                    project_info.project_id,
                    project_info.project_name,
                    domain_id,
                    project_info.cwd,
                    project_info.claude_project_id,
                    len(project_info.session_files),
                    datetime.now(),
                ],
            )

    def _populate_sessions(self, projects: dict[str, ProjectInfo]) -> int:
        """Populate sessions table.

        Reads session metadata from .jsonl files.

        Args:
            projects: Dictionary mapping project_id to ProjectInfo

        Returns:
            Total number of sessions indexed
        """
        conn = self.manager.connection
        session_count = 0

        for project_info in projects.values():
            domain_id = generate_domain_id(project_info.domain_path)

            for session_file in project_info.session_files:
                # Extract session metadata
                session_id = session_file.stem  # Filename without .jsonl
                session_meta = self._get_session_metadata(session_file)

                conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id,
                        project_id,
                        domain_id,
                        file_path,
                        git_branch,
                        started_at,
                        last_event_at,
                        event_count,
                        indexed
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT (session_id) DO UPDATE SET
                        event_count = excluded.event_count,
                        last_event_at = excluded.last_event_at
                """,
                    [
                        session_id,
                        project_info.project_id,
                        domain_id,
                        str(session_file),
                        session_meta.get("git_branch"),
                        session_meta.get("started_at"),
                        session_meta.get("last_event_at"),
                        session_meta.get("event_count", 0),
                        False,  # Not indexed for semantic search yet
                    ],
                )

                session_count += 1

        return session_count

    def _get_session_metadata(self, session_file: Path) -> dict:
        """Extract metadata from a session file.

        Args:
            session_file: Path to session .jsonl file

        Returns:
            Dictionary with metadata: git_branch, started_at, last_event_at, event_count
        """
        try:
            conn = duckdb.connect(":memory:")

            # Get git branch - wrap in try/except since column might not exist
            git_branch = None
            try:
                git_branch_result = conn.execute(f"""
                    SELECT gitBranch
                    FROM read_ndjson_auto('{session_file}')
                    WHERE gitBranch IS NOT NULL
                    LIMIT 1
                """).fetchone()
                git_branch = git_branch_result[0] if git_branch_result else None
            except Exception:
                # Column doesn't exist in this file - that's okay
                pass

            # Get timestamp range and count - wrap in try/except since column might not exist
            started_at, last_event_at, event_count = None, None, 0
            try:
                stats_result = conn.execute(f"""
                    SELECT
                        MIN(timestamp) as started_at,
                        MAX(timestamp) as last_event_at,
                        COUNT(*) as event_count
                    FROM read_ndjson_auto('{session_file}')
                    WHERE timestamp IS NOT NULL
                """).fetchone()

                if stats_result:
                    started_at, last_event_at, event_count = stats_result
            except Exception:
                # Column doesn't exist in this file - that's okay
                pass

            return {
                "git_branch": git_branch,
                "started_at": started_at,
                "last_event_at": last_event_at,
                "event_count": event_count,
            }

        except Exception as e:
            print(f"Warning: Could not extract session stats from {session_file}: {e}")
            return {}

    def get_stats(self) -> dict:
        """Get current database statistics.

        Returns:
            Dictionary with counts and last rebuild time
        """
        conn = self.manager.connection

        try:
            domain_result = conn.execute("SELECT COUNT(*) FROM domains").fetchone()
            project_result = conn.execute("SELECT COUNT(*) FROM projects").fetchone()
            session_result = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
            memory_result = conn.execute("SELECT COUNT(*) FROM memories").fetchone()

            domain_count = domain_result[0] if domain_result else 0
            project_count = project_result[0] if project_result else 0
            session_count = session_result[0] if session_result else 0
            memory_count = memory_result[0] if memory_result else 0

            last_rebuild_result = conn.execute("""
                SELECT value FROM metadata WHERE key = 'last_rebuild'
            """).fetchone()

            last_rebuild = last_rebuild_result[0] if last_rebuild_result else None

            return {
                "domains": domain_count,
                "projects": project_count,
                "sessions": session_count,
                "memories": memory_count,
                "last_rebuild": last_rebuild,
                "schema_version": SCHEMA_VERSION,
            }

        except Exception as e:
            return {"error": str(e)}

    def close(self) -> None:
        """Close database connection."""
        self.manager.close()
