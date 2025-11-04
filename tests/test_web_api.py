"""Tests for web API endpoints."""

import pytest
from fastapi.testclient import TestClient

from clem.config import CLEM_HOME
from clem.database.manager import DatabaseManager
from clem.database.schema import init_schema
from clem.web import create_app


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """Create test database with sample data for API testing."""
    # Create test database in a temporary clem home
    test_clem_home = tmp_path / ".clem"
    test_clem_home.mkdir()
    db_path = test_clem_home / "memory.duckdb"

    # Patch CLEM_HOME to point to our test directory
    monkeypatch.setattr("clem.config.CLEM_HOME", test_clem_home)
    monkeypatch.setattr("clem.config.DATABASE_PATH", db_path)

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
            ('session3', 'play::clem', 'play', '/path/session3.jsonl', '2024-01-03T10:00:00', 25)
    """
    )

    # Update metadata
    manager.connection.execute(
        """
        INSERT INTO metadata (key, value)
        VALUES
            ('last_rebuild', '2024-01-01T00:00:00')
    """
    )

    manager.close()

    yield db_path


@pytest.fixture
def client(test_db):
    """Create test client."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "clem-api"


class TestStatsEndpoint:
    """Tests for stats endpoint."""

    def test_get_stats(self, client, test_db):
        """Test getting database statistics."""
        response = client.get("/api/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["domains"] == 2
        assert data["projects"] == 3
        assert data["sessions"] == 3
        assert "schema_version" in data


class TestDomainsEndpoint:
    """Tests for domains endpoints."""

    def test_list_domains(self, client, test_db):
        """Test listing all domains."""
        response = client.get("/api/domains")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["domain_path"] == "play"
        assert data[0]["project_count"] == 2

    def test_get_domain_by_id(self, client, test_db):
        """Test getting specific domain."""
        response = client.get("/api/domains/play")
        assert response.status_code == 200
        data = response.json()
        assert data["domain_id"] == "play"
        assert data["project_count"] == 2

    def test_get_nonexistent_domain(self, client, test_db):
        """Test getting non-existent domain returns 404."""
        response = client.get("/api/domains/nonexistent")
        assert response.status_code == 404


class TestProjectsEndpoint:
    """Tests for projects endpoints."""

    def test_list_projects(self, client, test_db):
        """Test listing all projects."""
        response = client.get("/api/projects")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["project_name"] == "clem"

    def test_list_projects_filtered_by_domain(self, client, test_db):
        """Test listing projects filtered by domain."""
        response = client.get("/api/projects?domain_id=play")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all(p["domain_id"] == "play" for p in data)

    def test_get_project_by_id(self, client, test_db):
        """Test getting specific project."""
        response = client.get("/api/projects/play::clem")
        assert response.status_code == 200
        data = response.json()
        assert data["project_name"] == "clem"
        assert data["domain_id"] == "play"

    def test_get_nonexistent_project(self, client, test_db):
        """Test getting non-existent project returns 404."""
        response = client.get("/api/projects/nonexistent")
        assert response.status_code == 404


class TestSessionsEndpoint:
    """Tests for sessions endpoints."""

    def test_list_sessions(self, client, test_db):
        """Test listing all sessions."""
        response = client.get("/api/sessions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert data[0]["session_id"] == "session3"  # Most recent first

    def test_list_sessions_filtered_by_project(self, client, test_db):
        """Test listing sessions filtered by project."""
        response = client.get("/api/sessions?project_name=clem")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        assert all(s["project_name"] == "clem" for s in data)

    def test_list_sessions_with_limit(self, client, test_db):
        """Test listing sessions with limit."""
        response = client.get("/api/sessions?limit=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_get_session_by_id(self, client, test_db):
        """Test getting specific session."""
        response = client.get("/api/sessions/session1")
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "session1"
        assert data["event_count"] == 42

    def test_get_nonexistent_session(self, client, test_db):
        """Test getting non-existent session returns 404."""
        response = client.get("/api/sessions/nonexistent")
        assert response.status_code == 404


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is accessible."""
        response = client.get("/api/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "CLEM API"

    def test_docs_page(self, client):
        """Test Swagger UI docs page is accessible."""
        response = client.get("/api/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()
