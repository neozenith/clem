"""Configuration module for CLEM.

Defines paths, constants, and settings for the CLEM system.
Database is treated as disposable cache - can be rebuilt from source at any time.
"""

from pathlib import Path
from typing import Final

# === Paths ===

#: CLEM home directory containing all generated/cached data
CLEM_HOME: Final[Path] = Path("~/.clem/").expanduser()

#: Main database file (disposable cache - can be rebuilt anytime)
DATABASE_PATH: Final[Path] = CLEM_HOME / "memory.duckdb"

#: Claude Code projects directory (source of truth)
CLAUDE_PROJECTS_DIR: Final[Path] = Path("~/.claude/projects/").expanduser()

#: User home directory (for path normalization)
HOME_DIR: Final[Path] = Path.home()


# === Model Configuration ===

#: Sentence transformer model for embeddings
EMBEDDING_MODEL: Final[str] = "sentence-transformers/all-MiniLM-L6-v2"

#: Flan-T5 model for memory extraction
EXTRACTION_MODEL: Final[str] = "google/flan-t5-base"

#: Embedding vector dimensionality
EMBEDDING_DIM: Final[int] = 384


# === Memory Extraction ===

#: Topics for memory extraction
MEMORY_TOPICS: Final[list[str]] = [
    "facts",  # Factual information about the project
    "preferences",  # User preferences and patterns
    "goals",  # Project goals and objectives
    "patterns",  # Code patterns and architectural decisions
]


# === Search & Indexing ===

#: Similarity threshold for memory consolidation (0-1, cosine similarity)
SIMILARITY_THRESHOLD: Final[float] = 0.85

#: Default number of results for semantic search
DEFAULT_TOP_K: Final[int] = 10

#: Batch size for embedding generation
EMBEDDING_BATCH_SIZE: Final[int] = 32


# === Database ===

#: Schema version for tracking migrations/rebuilds
SCHEMA_VERSION: Final[str] = "1.0.0"

#: VSS index parameters
VSS_INDEX_PARAMS: Final[dict] = {
    "metric": "cosine",
    "ef_construction": 128,
    "M": 16,
}


def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    CLEM_HOME.mkdir(parents=True, exist_ok=True)


def get_database_path() -> Path:
    """Get the database path, ensuring parent directories exist."""
    ensure_directories()
    return DATABASE_PATH
