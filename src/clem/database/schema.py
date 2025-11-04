"""Database schema definitions for CLEM.

Defines all tables and indexes. Database is disposable cache.
"""

from typing import TYPE_CHECKING

from ..config import SCHEMA_VERSION, VSS_INDEX_PARAMS

if TYPE_CHECKING:
    import duckdb


def init_schema(conn: "duckdb.DuckDBPyConnection") -> None:
    """Initialize complete database schema.

    Creates all tables, indexes, and extensions.
    Safe to call multiple times (uses IF NOT EXISTS).

    Args:
        conn: DuckDB connection
    """
    # Install and load VSS extension
    setup_vss_extension(conn)

    # Create tables
    create_metadata_table(conn)
    create_domains_table(conn)
    create_projects_table(conn)
    create_sessions_table(conn)
    create_memories_table(conn)

    # Store schema version
    conn.execute(
        """
        INSERT OR REPLACE INTO metadata (key, value)
        VALUES ('schema_version', ?)
    """,
        [SCHEMA_VERSION],
    )


def setup_vss_extension(conn: "duckdb.DuckDBPyConnection") -> None:
    """Install and load VSS extension for vector similarity search.

    Args:
        conn: DuckDB connection
    """
    conn.execute("INSTALL vss")
    conn.execute("LOAD vss")
    # Enable experimental HNSW persistence for file-based databases
    conn.execute("SET hnsw_enable_experimental_persistence = true")


def create_metadata_table(conn: "duckdb.DuckDBPyConnection") -> None:
    """Create metadata table for tracking database state.

    Args:
        conn: DuckDB connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key VARCHAR PRIMARY KEY,
            value VARCHAR,
            updated_at TIMESTAMP DEFAULT NOW()
        )
    """)


def create_domains_table(conn: "duckdb.DuckDBPyConnection") -> None:
    """Create domains table.

    Domain represents a collection of projects (e.g., 'play', 'clients/nine').

    Args:
        conn: DuckDB connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS domains (
            domain_id VARCHAR PRIMARY KEY,
            domain_path VARCHAR NOT NULL UNIQUE,
            project_count INTEGER DEFAULT 0,
            session_count INTEGER DEFAULT 0,
            last_scan TIMESTAMP DEFAULT NOW()
        )
    """)


def create_projects_table(conn: "duckdb.DuckDBPyConnection") -> None:
    """Create projects table.

    Project is a git repository that Claude Code works with.

    Args:
        conn: DuckDB connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS projects (
            project_id VARCHAR PRIMARY KEY,
            project_name VARCHAR NOT NULL,
            domain_id VARCHAR NOT NULL,
            cwd VARCHAR NOT NULL UNIQUE,
            claude_project_id VARCHAR NOT NULL,
            session_count INTEGER DEFAULT 0,
            last_scan TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (domain_id) REFERENCES domains(domain_id)
        )
    """)

    # Indexes for efficient queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_projects_domain
        ON projects(domain_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_projects_claude_id
        ON projects(claude_project_id)
    """)


def create_sessions_table(conn: "duckdb.DuckDBPyConnection") -> None:
    """Create sessions table.

    Session is a single Claude Code conversation (one .jsonl file).

    Args:
        conn: DuckDB connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id VARCHAR PRIMARY KEY,
            project_id VARCHAR NOT NULL,
            domain_id VARCHAR NOT NULL,
            file_path VARCHAR NOT NULL UNIQUE,
            git_branch VARCHAR,
            started_at TIMESTAMP,
            last_event_at TIMESTAMP,
            event_count INTEGER DEFAULT 0,
            indexed BOOLEAN DEFAULT FALSE,
            FOREIGN KEY (project_id) REFERENCES projects(project_id),
            FOREIGN KEY (domain_id) REFERENCES domains(domain_id)
        )
    """)

    # Indexes for efficient queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sessions_project
        ON sessions(project_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sessions_domain
        ON sessions(domain_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_sessions_indexed
        ON sessions(indexed)
    """)


def create_memories_table(conn: "duckdb.DuckDBPyConnection") -> None:
    """Create memories table with vector search support.

    Memories are extracted insights from sessions.

    Args:
        conn: DuckDB connection
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            memory_id VARCHAR PRIMARY KEY,
            domain_id VARCHAR NOT NULL,
            project_id VARCHAR,
            session_id VARCHAR,
            topic VARCHAR NOT NULL,
            content TEXT NOT NULL,
            embedding FLOAT[384],
            confidence FLOAT DEFAULT 1.0,
            extracted_at TIMESTAMP DEFAULT NOW(),
            FOREIGN KEY (domain_id) REFERENCES domains(domain_id),
            FOREIGN KEY (project_id) REFERENCES projects(project_id),
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)

    # Indexes for efficient queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_domain
        ON memories(domain_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_project
        ON memories(project_id)
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memories_topic
        ON memories(topic)
    """)

    # VSS index for vector similarity search
    # Note: This uses HNSW (Hierarchical Navigable Small World) algorithm
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_memories_embedding
        ON memories USING HNSW (embedding)
        WITH (metric = '{VSS_INDEX_PARAMS["metric"]}')
    """)


def drop_all_tables(conn: "duckdb.DuckDBPyConnection") -> None:
    """Drop all tables (nuclear rebuild).

    Args:
        conn: DuckDB connection
    """
    # Drop in reverse dependency order
    tables = ["memories", "sessions", "projects", "domains", "metadata"]
    for table in tables:
        conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
