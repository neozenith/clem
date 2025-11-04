"""Tests for display layer."""

from clem.display import format_domain_table, format_project_table, format_session_table, format_stats_table
from clem.queries.domains import DomainStats
from clem.queries.projects import ProjectStats
from clem.queries.sessions import SessionStats


class TestFormatStatsTable:
    """Tests for format_stats_table."""

    def test_format_basic_stats(self):
        """Test formatting basic statistics."""
        stats = {
            "domains": 6,
            "projects": 15,
            "sessions": 53,
            "memories": 0,
            "schema_version": "1.0.0",
        }

        table = format_stats_table(stats)

        assert table.title == "Database Statistics"
        assert len(table.columns) == 2
        assert len(table.rows) == 5  # Without last_rebuild

    def test_format_stats_with_rebuild(self):
        """Test formatting statistics with last rebuild time."""
        stats = {
            "domains": 6,
            "projects": 15,
            "sessions": 53,
            "memories": 0,
            "schema_version": "1.0.0",
            "last_rebuild": "2024-01-01T10:00:00",
        }

        table = format_stats_table(stats)

        assert len(table.rows) == 6  # With last_rebuild


class TestFormatDomainTable:
    """Tests for format_domain_table."""

    def test_format_empty_domains(self):
        """Test formatting empty domain list."""
        table = format_domain_table([])

        assert len(table.columns) == 3
        assert len(table.rows) == 0

    def test_format_single_domain(self):
        """Test formatting single domain."""
        domains = [DomainStats("play", "play", 5, 15)]

        table = format_domain_table(domains)

        assert len(table.rows) == 1

    def test_format_multiple_domains(self):
        """Test formatting multiple domains."""
        domains = [
            DomainStats("play", "play", 5, 15),
            DomainStats("work", "work", 3, 8),
        ]

        table = format_domain_table(domains)

        assert len(table.rows) == 2

    def test_format_empty_domain_path(self):
        """Test formatting domain with empty path."""
        domains = [DomainStats("root", "", 2, 5)]

        table = format_domain_table(domains)

        assert len(table.rows) == 1


class TestFormatProjectTable:
    """Tests for format_project_table."""

    def test_format_empty_projects(self):
        """Test formatting empty project list."""
        table = format_project_table([])

        assert len(table.columns) == 3
        assert len(table.rows) == 0

    def test_format_single_project(self):
        """Test formatting single project."""
        projects = [ProjectStats("clem", "play", 10, "/Users/test/play/clem")]

        table = format_project_table(projects)

        assert len(table.rows) == 1

    def test_format_multiple_projects(self):
        """Test formatting multiple projects."""
        projects = [
            ProjectStats("clem", "play", 10, "/Users/test/play/clem"),
            ProjectStats("api", "work", 5, "/Users/test/work/api"),
        ]

        table = format_project_table(projects)

        assert len(table.rows) == 2

    def test_format_empty_domain_id(self):
        """Test formatting project with empty domain."""
        projects = [ProjectStats("standalone", "", 3, "/Users/test/standalone")]

        table = format_project_table(projects)

        assert len(table.rows) == 1


class TestFormatSessionTable:
    """Tests for format_session_table."""

    def test_format_empty_sessions(self):
        """Test formatting empty session list."""
        table = format_session_table([])

        assert len(table.columns) == 5
        assert len(table.rows) == 0

    def test_format_single_session(self):
        """Test formatting single session."""
        sessions = [SessionStats("abc123def456", "clem", "play", 42, "2024-01-01T10:00:00")]

        table = format_session_table(sessions)

        assert len(table.rows) == 1

    def test_format_multiple_sessions(self):
        """Test formatting multiple sessions."""
        sessions = [
            SessionStats("session1", "clem", "play", 42, "2024-01-01T10:00:00"),
            SessionStats("session2", "api", "work", 38, "2024-01-02T11:30:00"),
        ]

        table = format_session_table(sessions)

        assert len(table.rows) == 2

    def test_format_session_without_started_at(self):
        """Test formatting session with no start time."""
        sessions = [SessionStats("session1", "clem", "play", 42, None)]

        table = format_session_table(sessions)

        assert len(table.rows) == 1

    def test_format_empty_domain_path(self):
        """Test formatting session with empty domain."""
        sessions = [SessionStats("session1", "standalone", "", 42, "2024-01-01T10:00:00")]

        table = format_session_table(sessions)

        assert len(table.rows) == 1
