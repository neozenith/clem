"""Domain detection and path manipulation for CLEM.

Extracts domain and project information from Claude Code session paths.
"""

import os
from pathlib import Path
from typing import NamedTuple

from ..config import HOME_DIR


class DomainProject(NamedTuple):
    """Domain and project information extracted from path."""

    domain_path: str
    project_name: str
    full_path: str


def extract_domain_and_project(cwd: str, home_dir: Path = HOME_DIR) -> DomainProject:
    """Extract domain and project from working directory path.

    Domain represents a collection of projects (e.g., 'play', 'clients/nine').
    Project is the leaf directory name (usually a git repository).

    Algorithm:
    1. Strip home directory from cwd
    2. Split remaining path into segments
    3. Last segment = project_name
    4. All preceding segments = domain_path (joined with '/')
    5. Empty domain if only one segment

    Examples:
        /Users/joshpeak/play/clem
        → domain: "play", project: "clem"

        /Users/joshpeak/clients/nine/agt-nam-self-service-agent
        → domain: "clients/nine", project: "agt-nam-self-service-agent"

        /Users/joshpeak/foss/adk-samples/python/agents/short-movie-agents
        → domain: "foss/adk-samples/python/agents", project: "short-movie-agents"

        /Users/joshpeak/work/poc-nine-veo3-ad-generator
        → domain: "work", project: "poc-nine-veo3-ad-generator"

    Args:
        cwd: Current working directory path
        home_dir: User home directory (for normalization)

    Returns:
        DomainProject with domain_path, project_name, and full_path

    Note:
        If cwd is just one level (e.g., ~/project), domain_path will be empty string.
    """
    # Normalize paths
    cwd = os.path.expanduser(cwd)
    home_str = str(home_dir)

    # Strip home directory prefix
    if cwd.startswith(home_str):
        rel_path = cwd[len(home_str) :].lstrip("/")
    else:
        rel_path = cwd

    # Split into segments
    segments = [s for s in rel_path.split("/") if s]

    # Handle edge cases
    if not segments:
        return DomainProject("", "", cwd)

    if len(segments) == 1:
        # No domain, just project (e.g., ~/myproject)
        return DomainProject("", segments[0], cwd)

    # Standard case: domain = all but last, project = last
    project_name = segments[-1]
    domain_path = "/".join(segments[:-1])

    return DomainProject(domain_path, project_name, cwd)


def encode_to_claude_id(path: str) -> str:
    """Encode a path to Claude Code's project ID format.

    Claude Code replaces '/' with '-' in paths, creating lossy encoding.
    This is used for directory names in ~/.claude/projects/.

    Examples:
        /Users/joshpeak/play/clem
        → -Users-joshpeak-play-clem

        /Users/joshpeak/clients/nine/agt-nam-self-service-agent
        → -Users-joshpeak-clients-nine-agt-nam-self-service-agent

    Args:
        path: Full path to encode

    Returns:
        Encoded path with '/' replaced by '-'

    Warning:
        This encoding is lossy - cannot reliably decode back to original path
        if the path contains hyphens (e.g., 'adk-samples' vs 'adk/samples').
    """
    return path.replace("/", "-")


def verify_encoding(cwd: str, claude_project_id: str) -> bool:
    """Verify that a cwd matches Claude Code's project directory name.

    Args:
        cwd: Current working directory path
        claude_project_id: Directory name from ~/.claude/projects/

    Returns:
        True if encoding matches, False otherwise

    Example:
        >>> verify_encoding('/Users/joshpeak/play/clem', '-Users-joshpeak-play-clem')
        True
        >>> verify_encoding('/Users/joshpeak/play/clem', '-Users-joshpeak-work-clem')
        False
    """
    encoded = encode_to_claude_id(cwd)
    return encoded == claude_project_id


def generate_domain_id(domain_path: str) -> str:
    """Generate a unique domain ID from domain path.

    Simply returns the domain path as-is, which serves as a natural key.

    Args:
        domain_path: Domain path (e.g., 'play', 'clients/nine')

    Returns:
        Domain ID (same as domain_path)

    Example:
        >>> generate_domain_id('clients/nine')
        'clients/nine'
    """
    return domain_path


def generate_project_id(domain_path: str, project_name: str) -> str:
    """Generate a unique project ID from domain and project name.

    Combines domain path and project name to create globally unique identifier.

    Args:
        domain_path: Domain path (e.g., 'play', 'clients/nine')
        project_name: Project name (e.g., 'clem', 'agt-nam-self-service-agent')

    Returns:
        Project ID in format 'domain_path::project_name'

    Examples:
        >>> generate_project_id('play', 'clem')
        'play::clem'
        >>> generate_project_id('clients/nine', 'agt-nam-self-service-agent')
        'clients/nine::agt-nam-self-service-agent'
        >>> generate_project_id('', 'standalone')
        '::standalone'
    """
    return f"{domain_path}::{project_name}"
