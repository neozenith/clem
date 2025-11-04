"""Tests for query layer."""

import pytest

from clem.database.manager import DatabaseManager
from clem.database.schema import init_schema
from clem.queries import DomainQuery, ProjectQuery, SessionQuery


@pytest.fixture
def query_db(tmp_path):
    """Create test database with sample data."""
    db_path = tmp_path / "test_queries.duckdb"
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

    yield manager

    manager.close()


class TestDomainQuery:
    """Tests for DomainQuery."""

    def test_list_all(self, query_db):
        """Test listing all domains."""
        query = DomainQuery(query_db)
        domains = query.list_all()

        assert len(domains) == 2
        assert domains[0].domain_path == "play"
        assert domains[0].project_count == 2
        assert domains[0].session_count == 5
        assert domains[1].domain_path == "work"

    def test_get_by_id(self, query_db):
        """Test getting domain by ID."""
        query = DomainQuery(query_db)
        domain = query.get_by_id("play")

        assert domain is not None
        assert domain.domain_path == "play"
        assert domain.project_count == 2
        assert domain.session_count == 5

    def test_get_by_id_not_found(self, query_db):
        """Test getting non-existent domain."""
        query = DomainQuery(query_db)
        domain = query.get_by_id("nonexistent")

        assert domain is None

    def test_count(self, query_db):
        """Test counting domains."""
        query = DomainQuery(query_db)
        count = query.count()

        assert count == 2


class TestProjectQuery:
    """Tests for ProjectQuery."""

    def test_list_all(self, query_db):
        """Test listing all projects."""
        query = ProjectQuery(query_db)
        projects = query.list_all()

        assert len(projects) == 3
        assert projects[0].project_name == "clem"
        assert projects[0].domain_id == "play"
        assert projects[0].session_count == 3

    def test_list_all_filtered_by_domain(self, query_db):
        """Test listing projects filtered by domain."""
        query = ProjectQuery(query_db)
        projects = query.list_all(domain_id="play")

        assert len(projects) == 2
        assert all(p.domain_id == "play" for p in projects)

    def test_get_by_id(self, query_db):
        """Test getting project by ID."""
        query = ProjectQuery(query_db)
        project = query.get_by_id("play::clem")

        assert project is not None
        assert project.project_name == "clem"
        assert project.domain_id == "play"
        assert project.session_count == 3

    def test_get_by_id_not_found(self, query_db):
        """Test getting non-existent project."""
        query = ProjectQuery(query_db)
        project = query.get_by_id("nonexistent")

        assert project is None

    def test_count(self, query_db):
        """Test counting projects."""
        query = ProjectQuery(query_db)
        count = query.count()

        assert count == 3

    def test_count_filtered_by_domain(self, query_db):
        """Test counting projects filtered by domain."""
        query = ProjectQuery(query_db)
        count = query.count(domain_id="play")

        assert count == 2


class TestSessionQuery:
    """Tests for SessionQuery."""

    def test_list_all(self, query_db):
        """Test listing all sessions."""
        query = SessionQuery(query_db)
        sessions = query.list_all(limit=10)

        assert len(sessions) == 8
        # Should be ordered by started_at DESC
        assert sessions[0].session_id == "session8"
        assert sessions[-1].session_id == "session1"

    def test_list_all_filtered_by_project(self, query_db):
        """Test listing sessions filtered by project."""
        query = SessionQuery(query_db)
        sessions = query.list_all(project_name="clem")

        assert len(sessions) == 3
        assert all(s.project_name == "clem" for s in sessions)

    def test_list_all_filtered_by_domain(self, query_db):
        """Test listing sessions filtered by domain."""
        query = SessionQuery(query_db)
        sessions = query.list_all(domain_id="work")

        assert len(sessions) == 3
        assert all(s.domain_path == "work" for s in sessions)

    def test_list_all_with_limit(self, query_db):
        """Test listing sessions with limit."""
        query = SessionQuery(query_db)
        sessions = query.list_all(limit=3)

        assert len(sessions) == 3

    def test_get_by_id(self, query_db):
        """Test getting session by ID."""
        query = SessionQuery(query_db)
        session = query.get_by_id("session1")

        assert session is not None
        assert session.project_name == "clem"
        assert session.domain_path == "play"
        assert session.event_count == 42

    def test_get_by_id_not_found(self, query_db):
        """Test getting non-existent session."""
        query = SessionQuery(query_db)
        session = query.get_by_id("nonexistent")

        assert session is None

    def test_count(self, query_db):
        """Test counting sessions."""
        query = SessionQuery(query_db)
        count = query.count()

        assert count == 8

    def test_count_filtered_by_project(self, query_db):
        """Test counting sessions filtered by project."""
        query = SessionQuery(query_db)
        count = query.count(project_name="clem")

        assert count == 3

    def test_count_filtered_by_domain(self, query_db):
        """Test counting sessions filtered by domain."""
        query = SessionQuery(query_db)
        count = query.count(domain_id="work")

        assert count == 3
