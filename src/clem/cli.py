"""Command-line interface for CLEM."""

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import __version__
from .database.builder import DatabaseBuilder
from .database.manager import DatabaseManager
from .config import get_database_path

console = Console()


def cmd_rebuild(args) -> int:
    """Rebuild database from source."""
    console.print("\n[bold]CLEM - Database Rebuild[/bold]\n")

    builder = DatabaseBuilder()

    try:
        builder.rebuild(full=True)
        return 0
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        return 1
    finally:
        builder.close()


def cmd_stats(args) -> int:
    """Show database statistics."""
    console.print("\n[bold]CLEM - Database Statistics[/bold]\n")

    db_path = get_database_path()

    if not db_path.exists():
        console.print("[yellow]Database not found. Run 'clem rebuild' first.[/yellow]")
        return 1

    builder = DatabaseBuilder()

    try:
        stats = builder.get_stats()

        if 'error' in stats:
            console.print(f"[bold red]Error:[/bold red] {stats['error']}")
            return 1

        # Create stats table
        table = Table(title="Database Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Domains", str(stats['domains']))
        table.add_row("Projects", str(stats['projects']))
        table.add_row("Sessions", str(stats['sessions']))
        table.add_row("Memories", str(stats['memories']))
        table.add_row("Schema Version", stats['schema_version'])
        if stats.get('last_rebuild'):
            table.add_row("Last Rebuild", stats['last_rebuild'])

        console.print(table)
        console.print(f"\n[dim]Database: {db_path}[/dim]")

        return 0

    finally:
        builder.close()


def cmd_domains(args) -> int:
    """List all domains."""
    console.print("\n[bold]CLEM - Domains[/bold]\n")

    db_path = get_database_path()

    if not db_path.exists():
        console.print("[yellow]Database not found. Run 'clem rebuild' first.[/yellow]")
        return 1

    manager = DatabaseManager()

    try:
        domains = manager.query("""
            SELECT domain_id, domain_path, project_count, session_count
            FROM domains
            ORDER BY domain_path
        """)

        if not domains:
            console.print("[yellow]No domains found.[/yellow]")
            return 0

        # Create domains table
        table = Table(show_header=True)
        table.add_column("Domain", style="cyan")
        table.add_column("Projects", justify="right", style="green")
        table.add_column("Sessions", justify="right", style="blue")

        for domain_id, domain_path, project_count, session_count in domains:
            display_name = domain_path if domain_path else "(no domain)"
            table.add_row(display_name, str(project_count), str(session_count))

        console.print(table)
        console.print(f"\n[dim]Total: {len(domains)} domains[/dim]")

        return 0

    finally:
        manager.close()


def cmd_projects(args) -> int:
    """List all projects."""
    console.print("\n[bold]CLEM - Projects[/bold]\n")

    db_path = get_database_path()

    if not db_path.exists():
        console.print("[yellow]Database not found. Run 'clem rebuild' first.[/yellow]")
        return 1

    manager = DatabaseManager()

    try:
        # Build query
        query = """
            SELECT
                p.project_name,
                p.domain_id,
                p.session_count,
                p.cwd
            FROM projects p
        """

        params = []

        if args.domain:
            query += " WHERE p.domain_id = ?"
            params.append(args.domain)

        query += " ORDER BY p.domain_id, p.project_name"

        projects = manager.query(query, params) if params else manager.query(query)

        if not projects:
            console.print("[yellow]No projects found.[/yellow]")
            return 0

        # Create projects table
        table = Table(show_header=True)
        table.add_column("Project", style="cyan")
        table.add_column("Domain", style="magenta")
        table.add_column("Sessions", justify="right", style="green")

        for project_name, domain_id, session_count, cwd in projects:
            display_domain = domain_id if domain_id else "(no domain)"
            table.add_row(project_name, display_domain, str(session_count))

        console.print(table)
        console.print(f"\n[dim]Total: {len(projects)} projects[/dim]")

        return 0

    finally:
        manager.close()


def cmd_sessions(args) -> int:
    """List sessions."""
    console.print("\n[bold]CLEM - Sessions[/bold]\n")

    db_path = get_database_path()

    if not db_path.exists():
        console.print("[yellow]Database not found. Run 'clem rebuild' first.[/yellow]")
        return 1

    manager = DatabaseManager()

    try:
        # Build query
        query = """
            SELECT
                s.session_id,
                p.project_name,
                d.domain_path,
                s.event_count,
                s.started_at
            FROM sessions s
            JOIN projects p ON s.project_id = p.project_id
            JOIN domains d ON s.domain_id = d.domain_id
        """

        params = []

        if args.project:
            query += " WHERE p.project_name = ?"
            params.append(args.project)
        elif args.domain:
            query += " WHERE d.domain_id = ?"
            params.append(args.domain)

        query += " ORDER BY s.started_at DESC"

        if args.limit:
            query += f" LIMIT {args.limit}"

        sessions = manager.query(query, params) if params else manager.query(query)

        if not sessions:
            console.print("[yellow]No sessions found.[/yellow]")
            return 0

        # Create sessions table
        table = Table(show_header=True)
        table.add_column("Session ID", style="cyan", no_wrap=True)
        table.add_column("Project", style="magenta")
        table.add_column("Domain", style="blue")
        table.add_column("Events", justify="right", style="green")
        table.add_column("Started", style="dim")

        for session_id, project_name, domain_path, event_count, started_at in sessions:
            # Truncate session ID for display
            short_id = session_id[:8] if len(session_id) > 8 else session_id
            display_domain = domain_path if domain_path else "(no domain)"

            # Format datetime
            if started_at:
                if isinstance(started_at, str):
                    display_started = started_at[:19]
                else:
                    display_started = started_at.isoformat()[:19]
            else:
                display_started = "N/A"

            table.add_row(
                short_id,
                project_name,
                display_domain,
                str(event_count),
                display_started
            )

        console.print(table)
        console.print(f"\n[dim]Showing {len(sessions)} sessions[/dim]")

        return 0

    finally:
        manager.close()


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CLEM - Claude Learning & Experience Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--version',
        action='version',
        version=f'clem {__version__}'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # rebuild command
    parser_rebuild = subparsers.add_parser(
        'rebuild',
        help='Rebuild database from source (nuclear option)'
    )
    parser_rebuild.set_defaults(func=cmd_rebuild)

    # stats command
    parser_stats = subparsers.add_parser(
        'stats',
        help='Show database statistics'
    )
    parser_stats.set_defaults(func=cmd_stats)

    # domains command
    parser_domains = subparsers.add_parser(
        'domains',
        help='List all domains'
    )
    parser_domains.set_defaults(func=cmd_domains)

    # projects command
    parser_projects = subparsers.add_parser(
        'projects',
        help='List all projects'
    )
    parser_projects.add_argument(
        '--domain',
        help='Filter by domain'
    )
    parser_projects.set_defaults(func=cmd_projects)

    # sessions command
    parser_sessions = subparsers.add_parser(
        'sessions',
        help='List sessions'
    )
    parser_sessions.add_argument(
        '--project',
        help='Filter by project name'
    )
    parser_sessions.add_argument(
        '--domain',
        help='Filter by domain'
    )
    parser_sessions.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Limit number of results (default: 20)'
    )
    parser_sessions.set_defaults(func=cmd_sessions)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
