"""Project discovery for CLEM.

Discovers projects by scanning Claude Code session directories and extracting
cwd information from session files.
"""

from pathlib import Path
from typing import NamedTuple

import duckdb

from ..config import CLAUDE_PROJECTS_DIR
from .domain import encode_to_claude_id, extract_domain_and_project, generate_project_id


class ProjectInfo(NamedTuple):
    """Information about a discovered project."""

    project_id: str
    project_name: str
    domain_path: str
    cwd: str
    claude_project_id: str
    session_files: list[Path]


def get_cwd_from_session(session_file: Path) -> str | None:
    """Extract cwd from a session file.

    Reads the session file and finds the first event with a cwd field.
    Uses DuckDB for efficient JSONL parsing.

    Args:
        session_file: Path to session .jsonl file

    Returns:
        Current working directory from session, or None if not found
    """
    try:
        conn = duckdb.connect(":memory:")
        result = conn.execute(f"""
            SELECT cwd
            FROM read_ndjson_auto('{session_file}')
            WHERE cwd IS NOT NULL
            LIMIT 1
        """).fetchone()

        return result[0] if result else None
    except Exception:
        # Column doesn't exist in this file - that's okay
        return None


def discover_projects() -> dict[str, ProjectInfo]:
    """Discover all projects from Claude Code session directories.

    Scans ~/.claude/projects/ to find all project directories and extracts
    project information from session files.

    Algorithm:
    1. Iterate through ~/.claude/projects/* directories
    2. For each directory, read one session file to find cwd
    3. Extract domain and project from cwd
    4. Verify encoding matches directory name
    5. Group all session files by project

    Returns:
        Dictionary mapping project_id to ProjectInfo

    Example:
        >>> projects = discover_projects()
        >>> projects['play::clem'].project_name
        'clem'
        >>> projects['play::clem'].domain_path
        'play'
    """
    if not CLAUDE_PROJECTS_DIR.exists():
        return {}

    projects: dict[str, ProjectInfo] = {}

    # Iterate through project directories
    for project_dir in CLAUDE_PROJECTS_DIR.iterdir():
        if not project_dir.is_dir():
            continue

        claude_project_id = project_dir.name

        # Find all session files in this project
        session_files = list(project_dir.glob("*.jsonl"))
        if not session_files:
            continue

        # Extract cwd from first session file
        cwd = None
        for session_file in session_files:
            cwd = get_cwd_from_session(session_file)
            if cwd:
                break

        if not cwd:
            print(f"Warning: Could not extract cwd from {project_dir}")
            continue

        # Extract domain and project information
        domain_proj = extract_domain_and_project(cwd)

        # Verify encoding matches (sanity check)
        encoded = encode_to_claude_id(cwd)
        if encoded != claude_project_id:
            print(f"Warning: Encoding mismatch for {claude_project_id}")
            print(f"  Expected: {encoded}")
            print(f"  Got: {claude_project_id}")
            # Still proceed - Claude Code might have different encoding

        # Generate project ID
        project_id = generate_project_id(domain_proj.domain_path, domain_proj.project_name)

        # Store project info
        projects[project_id] = ProjectInfo(
            project_id=project_id,
            project_name=domain_proj.project_name,
            domain_path=domain_proj.domain_path,
            cwd=cwd,
            claude_project_id=claude_project_id,
            session_files=session_files,
        )

    return projects


def get_project_sessions(project_dir: Path) -> list[Path]:
    """Get all session files for a project directory.

    Args:
        project_dir: Path to project directory in ~/.claude/projects/

    Returns:
        List of session file paths
    """
    if not project_dir.exists() or not project_dir.is_dir():
        return []

    return sorted(project_dir.glob("*.jsonl"))


def get_unique_domains(projects: dict[str, ProjectInfo]) -> dict[str, list[ProjectInfo]]:
    """Group projects by domain.

    Args:
        projects: Dictionary of discovered projects

    Returns:
        Dictionary mapping domain_path to list of projects in that domain

    Example:
        >>> projects = discover_projects()
        >>> domains = get_unique_domains(projects)
        >>> len(domains['play'])
        5  # 5 projects in 'play' domain
    """
    domains: dict[str, list[ProjectInfo]] = {}

    for project_info in projects.values():
        domain = project_info.domain_path
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(project_info)

    return domains
