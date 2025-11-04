"""Tests for database modules."""

import tempfile
from pathlib import Path
import pytest

from clem.database.manager import DatabaseManager
from clem.database.schema import (
    init_schema,
    drop_all_tables,
    create_domains_table,
    create_projects_table,
    create_sessions_table,
    create_memories_table,
    setup_vss_extension,
)
from clem.database.builder import DatabaseBuilder
from clem.config import SCHEMA_VERSION


class TestDatabaseManager:
    """Tests for DatabaseManager."""

    def test_create_manager_with_temp_db(self, tmp_path):
        """Test creating a database manager with temporary database."""
        db_path = tmp_path / "test.duckdb"
        manager = DatabaseManager(db_path)

        assert manager.db_path == db_path
        assert not db_path.exists()  # Not created until connection is accessed

    def test_lazy_connection_initialization(self, tmp_path):
        """Test that connection is created lazily."""
        db_path = tmp_path / "test.duckdb"
        manager = DatabaseManager(db_path)

        assert manager._connection is None

        # Access connection
        conn = manager.connection

        assert conn is not None
        assert manager._connection is not None

    def test_execute_query(self, tmp_path):
        """Test executing a query."""
        db_path = tmp_path / "test.duckdb"
        manager = DatabaseManager(db_path)

        result = manager.execute("SELECT 42 as answer")
        assert result.fetchone()[0] == 42

    def test_query_shorthand(self, tmp_path):
        """Test query() method."""
        db_path = tmp_path / "test.duckdb"
        manager = DatabaseManager(db_path)

        result = manager.query("SELECT 42 as answer")
        assert result[0][0] == 42

    def test_close_connection(self, tmp_path):
        """Test closing connection."""
        db_path = tmp_path / "test.duckdb"
        manager = DatabaseManager(db_path)

        # Create connection
        _ = manager.connection
        assert manager._connection is not None

        # Close it
        manager.close()
        assert manager._connection is None

    def test_context_manager(self, tmp_path):
        """Test using manager as context manager."""
        db_path = tmp_path / "test.duckdb"

        with DatabaseManager(db_path) as manager:
            result = manager.query("SELECT 42")
            assert result[0][0] == 42
            assert manager._connection is not None

        # Connection should be closed after context
        assert manager._connection is None

    def test_database_exists(self, tmp_path):
        """Test database_exists method."""
        db_path = tmp_path / "test.duckdb"
        manager = DatabaseManager(db_path)

        assert not manager.database_exists()

        # Create the database
        _ = manager.connection

        assert manager.database_exists()

    def test_delete_database(self, tmp_path):
        """Test deleting database."""
        db_path = tmp_path / "test.duckdb"
        manager = DatabaseManager(db_path)

        # Create the database
        _ = manager.connection
        assert db_path.exists()

        # Delete it
        manager.delete_database()
        assert not db_path.exists()


class TestDatabaseSchema:
    """Tests for database schema functions."""

    def test_init_schema_creates_all_tables(self, tmp_path):
        """Test that init_schema creates all required tables."""
        db_path = tmp_path / "test.duckdb"
        manager = DatabaseManager(db_path)

        init_schema(manager.connection)

        # Check that all tables exist
        tables = manager.query("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
        """)

        table_names = [row[0] for row in tables]

        assert 'domains' in table_names
        assert 'projects' in table_names
        assert 'sessions' in table_names
        assert 'memories' in table_names
        assert 'metadata' in table_names

    def test_schema_version_stored(self, tmp_path):
        """Test that schema version is stored in metadata."""
        db_path = tmp_path / "test.duckdb"
        manager = DatabaseManager(db_path)

        init_schema(manager.connection)

        version = manager.query("""
            SELECT value FROM metadata WHERE key = 'schema_version'
        """)

        assert version[0][0] == SCHEMA_VERSION

    def test_drop_all_tables(self, tmp_path):
        """Test dropping all tables."""
        db_path = tmp_path / "test.duckdb"
        manager = DatabaseManager(db_path)

        # Create schema
        init_schema(manager.connection)

        tables_before = manager.query("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'main'
        """)
        assert tables_before[0][0] >= 5

        # Drop all tables
        drop_all_tables(manager.connection)

        tables_after = manager.query("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'main'
        """)
        assert tables_after[0][0] == 0

    def test_vss_extension_loaded(self, tmp_path):
        """Test that VSS extension is loaded."""
        db_path = tmp_path / "test.duckdb"
        manager = DatabaseManager(db_path)

        setup_vss_extension(manager.connection)

        # Try to query extensions (VSS should be loaded)
        # If VSS isn't loaded, creating HNSW index would fail
        manager.execute("CREATE TABLE test_vss (id INT, embedding FLOAT[384])")
        # This would fail if VSS isn't loaded
        manager.execute("""
            CREATE INDEX test_idx ON test_vss USING HNSW (embedding)
            WITH (metric = 'cosine')
        """)

        # If we got here, VSS is working
        assert True


class TestDatabaseBuilder:
    """Tests for DatabaseBuilder."""

    def test_create_builder(self, tmp_path):
        """Test creating a database builder."""
        db_path = tmp_path / "test.duckdb"
        builder = DatabaseBuilder(db_path)

        assert builder.db_path == db_path
        assert isinstance(builder.manager, DatabaseManager)

    def test_get_stats_empty_database(self, tmp_path):
        """Test getting stats from empty database."""
        db_path = tmp_path / "test.duckdb"
        builder = DatabaseBuilder(db_path)

        # Initialize schema
        init_schema(builder.manager.connection)

        stats = builder.get_stats()

        assert stats['domains'] == 0
        assert stats['projects'] == 0
        assert stats['sessions'] == 0
        assert stats['memories'] == 0
        assert stats['schema_version'] == SCHEMA_VERSION

    def test_close(self, tmp_path):
        """Test closing builder."""
        db_path = tmp_path / "test.duckdb"
        builder = DatabaseBuilder(db_path)

        # Access connection
        _ = builder.manager.connection
        assert builder.manager._connection is not None

        # Close
        builder.close()
        assert builder.manager._connection is None
