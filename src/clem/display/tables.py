"""Rich table formatters for CLEM display."""

from rich.table import Table

from ..queries.domains import DomainStats
from ..queries.projects import ProjectStats
from ..queries.sessions import SessionStats


def format_stats_table(stats: dict) -> Table:
    """Format database statistics as a Rich table.

    Args:
        stats: Dictionary containing database statistics

    Returns:
        Formatted Rich Table

    Example:
        >>> stats = {"domains": 6, "projects": 15, "sessions": 53}
        >>> table = format_stats_table(stats)
    """
    table = Table(title="Database Statistics", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Domains", str(stats["domains"]))
    table.add_row("Projects", str(stats["projects"]))
    table.add_row("Sessions", str(stats["sessions"]))
    table.add_row("Memories", str(stats["memories"]))
    table.add_row("Schema Version", stats["schema_version"])

    if stats.get("last_rebuild"):
        table.add_row("Last Rebuild", stats["last_rebuild"])

    return table


def format_domain_table(domains: list[DomainStats]) -> Table:
    """Format domain statistics as a Rich table.

    Args:
        domains: List of domain statistics

    Returns:
        Formatted Rich Table

    Example:
        >>> domains = [DomainStats("play", "play", 5, 15)]
        >>> table = format_domain_table(domains)
    """
    table = Table(show_header=True)
    table.add_column("Domain", style="cyan")
    table.add_column("Projects", justify="right", style="green")
    table.add_column("Sessions", justify="right", style="blue")

    for domain in domains:
        display_name = domain.domain_path if domain.domain_path else "(no domain)"
        table.add_row(display_name, str(domain.project_count), str(domain.session_count))

    return table


def format_project_table(projects: list[ProjectStats]) -> Table:
    """Format project statistics as a Rich table.

    Args:
        projects: List of project statistics

    Returns:
        Formatted Rich Table

    Example:
        >>> projects = [ProjectStats("clem", "play", 10, "/Users/test/play/clem")]
        >>> table = format_project_table(projects)
    """
    table = Table(show_header=True)
    table.add_column("Project", style="cyan")
    table.add_column("Domain", style="magenta")
    table.add_column("Sessions", justify="right", style="green")

    for project in projects:
        display_domain = project.domain_id if project.domain_id else "(no domain)"
        table.add_row(project.project_name, display_domain, str(project.session_count))

    return table


def format_session_table(sessions: list[SessionStats]) -> Table:
    """Format session statistics as a Rich table.

    Args:
        sessions: List of session statistics

    Returns:
        Formatted Rich Table

    Example:
        >>> sessions = [SessionStats("abc123", "clem", "play", 42, "2024-01-01T10:00:00", "/path/to/session.jsonl")]
        >>> table = format_session_table(sessions)
    """
    table = Table(show_header=True)
    table.add_column("Session ID", style="cyan", no_wrap=True)
    table.add_column("Project", style="magenta")
    table.add_column("Domain", style="blue")
    table.add_column("Events", justify="right", style="green")
    table.add_column("Started", style="dim")

    for session in sessions:
        # Truncate session ID for display
        short_id = session.session_id[:8] if len(session.session_id) > 8 else session.session_id
        display_domain = session.domain_path if session.domain_path else "(no domain)"

        # Format datetime
        if session.started_at:
            if isinstance(session.started_at, str):
                display_started = session.started_at[:19]
            else:
                display_started = str(session.started_at)[:19]
        else:
            display_started = "N/A"

        table.add_row(
            short_id,
            session.project_name,
            display_domain,
            str(session.event_count),
            display_started,
        )

    return table
