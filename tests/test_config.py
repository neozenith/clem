"""Tests for config module."""

from pathlib import Path
import pytest

from clem import config


def test_paths_are_pathlib():
    """Test that all path constants are Path objects."""
    assert isinstance(config.CLEM_HOME, Path)
    assert isinstance(config.DATABASE_PATH, Path)
    assert isinstance(config.CLAUDE_PROJECTS_DIR, Path)
    assert isinstance(config.HOME_DIR, Path)


def test_clem_home_in_user_home():
    """Test that CLEM_HOME is under user home directory."""
    assert str(config.HOME_DIR) in str(config.CLEM_HOME)
    assert config.CLEM_HOME.name == ".clem"


def test_database_path_is_in_clem_home():
    """Test that database is in CLEM_HOME."""
    assert config.DATABASE_PATH.parent == config.CLEM_HOME
    assert config.DATABASE_PATH.name == "memory.duckdb"


def test_embedding_config():
    """Test embedding configuration."""
    assert config.EMBEDDING_MODEL == 'sentence-transformers/all-MiniLM-L6-v2'
    assert config.EMBEDDING_DIM == 384
    assert isinstance(config.EMBEDDING_DIM, int)


def test_memory_topics():
    """Test memory topics configuration."""
    expected_topics = ['facts', 'preferences', 'goals', 'patterns']
    assert config.MEMORY_TOPICS == expected_topics
    assert len(config.MEMORY_TOPICS) == 4


def test_schema_version():
    """Test schema version is set."""
    assert config.SCHEMA_VERSION == '1.0.0'
    assert isinstance(config.SCHEMA_VERSION, str)


def test_ensure_directories(tmp_path, monkeypatch):
    """Test that ensure_directories creates CLEM_HOME."""
    # Mock CLEM_HOME to use temp directory
    test_clem_home = tmp_path / ".clem"
    monkeypatch.setattr(config, 'CLEM_HOME', test_clem_home)

    assert not test_clem_home.exists()

    config.ensure_directories()

    assert test_clem_home.exists()
    assert test_clem_home.is_dir()


def test_get_database_path_creates_directories(tmp_path, monkeypatch):
    """Test that get_database_path ensures parent directories exist."""
    test_clem_home = tmp_path / ".clem"
    test_db_path = test_clem_home / "memory.duckdb"

    monkeypatch.setattr(config, 'CLEM_HOME', test_clem_home)
    monkeypatch.setattr(config, 'DATABASE_PATH', test_db_path)

    assert not test_clem_home.exists()

    result = config.get_database_path()

    assert test_clem_home.exists()
    assert result == test_db_path
