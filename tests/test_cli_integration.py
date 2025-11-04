"""Integration tests for CLI commands."""

import argparse
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from clem.cli import cmd_domains, cmd_projects, cmd_rebuild, cmd_sessions, cmd_stats
from clem.database.builder import DatabaseBuilder
from clem.database.manager import DatabaseManager
from clem.database.schema import init_schema


@pytest.fixture
def test_db(tmp_path):
    """Create test database with sample data for CLI testing."""
    db_path = tmp_path / "test_cli.duckdb"
    manager = DatabaseManager(db_path)

    # Initialize schema
    init_schema(manager.connection)

    # Insert test data
    manager.connection.execute(
        """
        INSERT INTO domains (domain_id, domain_path, project_count, session_count)
        VALUES
            ('play', 'play', 2, 5),
            ('work', 'work', 1, 3)
    """
    )

    manager.connection.execute(
        """
        INSERT INTO projects (project_id, project_name, domain_id, cwd, claude_project_id, session_count)
        VALUES
            ('play::clem', 'clem', 'play', '/Users/test/play/clem', 'abc123', 3),
            ('play::demo', 'demo', 'play', '/Users/test/play/demo', 'def456', 2),
            ('work::api', 'api', 'work', '/Users/test/work/api', 'ghi789', 3)
    """
    )

    manager.connection.execute(
        """
        INSERT INTO sessions (session_id, project_id, domain_id, file_path, started_at, event_count)
        VALUES
            ('session1', 'play::clem', 'play', '/path/session1.jsonl', '2024-01-01T10:00:00', 42),
            ('session2', 'play::clem', 'play', '/path/session2.jsonl', '2024-01-02T10:00:00', 38),
            ('session3', 'play::clem', 'play', '/path/session3.jsonl', '2024-01-03T10:00:00', 25),
            ('session4', 'play::demo', 'play', '/path/session4.jsonl', '2024-01-04T10:00:00', 15),
            ('session5', 'play::demo', 'play', '/path/session5.jsonl', '2024-01-05T10:00:00', 20),
            ('session6', 'work::api', 'work', '/path/session6.jsonl', '2024-01-06T10:00:00', 30),
            ('session7', 'work::api', 'work', '/path/session7.jsonl', '2024-01-07T10:00:00', 28),
            ('session8', 'work::api', 'work', '/path/session8.jsonl', '2024-01-08T10:00:00', 35)
    """
    )

    # Update metadata (schema_version already inserted by init_schema)
    manager.connection.execute(
        """
        INSERT INTO metadata (key, value)
        VALUES
            ('last_rebuild', '2024-01-01T00:00:00'),
            ('sessions_indexed', '8')
    """
    )

    manager.close()

    yield db_path


class TestCmdStats:
    """Integration tests for stats command."""

    def test_stats_with_database(self, test_db):
        """Test stats command with existing database."""
        args = Mock()

        with (
            patch("clem.cli.get_database_path", return_value=test_db),
            patch("clem.cli.console.print"),
        ):
            result = cmd_stats(args)

        assert result == 0

    def test_stats_without_database(self, tmp_path):
        """Test stats command without database."""
        non_existent = tmp_path / "does_not_exist.duckdb"
        args = Mock()

        with (
            patch("clem.cli.get_database_path", return_value=non_existent),
            patch("clem.cli.console.print"),
        ):
            result = cmd_stats(args)

        assert result == 1


class TestCmdDomains:
    """Integration tests for domains command."""

    def test_domains_lists_all(self, test_db):
        """Test domains command lists all domains."""
        args = Mock()

        with (
            patch("clem.cli.get_database_path", return_value=test_db),
            patch("clem.cli.console.print"),
        ):
            result = cmd_domains(args)

        assert result == 0

    def test_domains_without_database(self, tmp_path):
        """Test domains command without database."""
        non_existent = tmp_path / "does_not_exist.duckdb"
        args = Mock()

        with (
            patch("clem.cli.get_database_path", return_value=non_existent),
            patch("clem.cli.console.print"),
        ):
            result = cmd_domains(args)

        assert result == 1


class TestCmdProjects:
    """Integration tests for projects command."""

    def test_projects_lists_all(self, test_db):
        """Test projects command lists all projects."""
        args = Mock()
        args.domain = None

        with (
            patch("clem.cli.get_database_path", return_value=test_db),
            patch("clem.cli.console.print"),
        ):
            result = cmd_projects(args)

        assert result == 0

    def test_projects_filtered_by_domain(self, test_db):
        """Test projects command filtered by domain."""
        args = Mock()
        args.domain = "play"

        with (
            patch("clem.cli.get_database_path", return_value=test_db),
            patch("clem.cli.console.print"),
        ):
            result = cmd_projects(args)

        assert result == 0

    def test_projects_without_database(self, tmp_path):
        """Test projects command without database."""
        non_existent = tmp_path / "does_not_exist.duckdb"
        args = Mock()
        args.domain = None

        with (
            patch("clem.cli.get_database_path", return_value=non_existent),
            patch("clem.cli.console.print"),
        ):
            result = cmd_projects(args)

        assert result == 1


class TestCmdSessions:
    """Integration tests for sessions command."""

    def test_sessions_lists_all(self, test_db):
        """Test sessions command lists all sessions."""
        args = Mock()
        args.project = None
        args.domain = None
        args.limit = 20

        with (
            patch("clem.cli.get_database_path", return_value=test_db),
            patch("clem.cli.console.print"),
        ):
            result = cmd_sessions(args)

        assert result == 0

    def test_sessions_filtered_by_project(self, test_db):
        """Test sessions command filtered by project."""
        args = Mock()
        args.project = "clem"
        args.domain = None
        args.limit = 20

        with (
            patch("clem.cli.get_database_path", return_value=test_db),
            patch("clem.cli.console.print"),
        ):
            result = cmd_sessions(args)

        assert result == 0

    def test_sessions_filtered_by_domain(self, test_db):
        """Test sessions command filtered by domain."""
        args = Mock()
        args.project = None
        args.domain = "work"
        args.limit = 20

        with (
            patch("clem.cli.get_database_path", return_value=test_db),
            patch("clem.cli.console.print"),
        ):
            result = cmd_sessions(args)

        assert result == 0

    def test_sessions_with_custom_limit(self, test_db):
        """Test sessions command with custom limit."""
        args = Mock()
        args.project = None
        args.domain = None
        args.limit = 5

        with (
            patch("clem.cli.get_database_path", return_value=test_db),
            patch("clem.cli.console.print"),
        ):
            result = cmd_sessions(args)

        assert result == 0

    def test_sessions_without_database(self, tmp_path):
        """Test sessions command without database."""
        non_existent = tmp_path / "does_not_exist.duckdb"
        args = Mock()
        args.project = None
        args.domain = None
        args.limit = 20

        with (
            patch("clem.cli.get_database_path", return_value=non_existent),
            patch("clem.cli.console.print"),
        ):
            result = cmd_sessions(args)

        assert result == 1


class TestCmdRebuild:
    """Integration tests for rebuild command."""

    def test_rebuild_creates_database(self, tmp_path):
        """Test rebuild command creates database."""
        db_path = tmp_path / "test_rebuild.duckdb"
        args = Mock()

        # Create a mock DatabaseBuilder that uses our test path
        mock_builder = DatabaseBuilder(db_path)

        # Mock project discovery to return empty results and use test builder
        with (
            patch("clem.cli.DatabaseBuilder", return_value=mock_builder),
            patch("clem.database.builder.discover_projects", return_value={}),
            patch("clem.cli.console.print"),
        ):
            result = cmd_rebuild(args)

        assert result == 0
        assert db_path.exists()
