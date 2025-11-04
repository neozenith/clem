"""Command-line interface for CLEM."""

import argparse
import sys
from typing import cast

from rich.console import Console

from . import __version__
from .config import get_database_path
from .database.builder import DatabaseBuilder
from .database.manager import DatabaseManager
from .display import (
    format_domain_table,
    format_project_table,
    format_session_table,
    format_stats_table,
)
from .queries import DomainQuery, ProjectQuery, SessionQuery

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

        if "error" in stats:
            console.print(f"[bold red]Error:[/bold red] {stats['error']}")
            return 1

        # Use display layer for formatting
        table = format_stats_table(stats)
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
        # Use queries layer
        query = DomainQuery(manager)
        domains = query.list_all()

        if not domains:
            console.print("[yellow]No domains found.[/yellow]")
            return 0

        # Use display layer for formatting
        table = format_domain_table(domains)
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
        # Use queries layer
        query = ProjectQuery(manager)
        projects = query.list_all(domain_id=args.domain if hasattr(args, "domain") else None)

        if not projects:
            console.print("[yellow]No projects found.[/yellow]")
            return 0

        # Use display layer for formatting
        table = format_project_table(projects)
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
        # Use queries layer
        query = SessionQuery(manager)
        project_name = args.project if hasattr(args, "project") and args.project else None
        domain_id = args.domain if hasattr(args, "domain") and args.domain else None
        limit = args.limit if hasattr(args, "limit") else 20

        sessions = query.list_all(project_name=project_name, domain_id=domain_id, limit=limit)

        if not sessions:
            console.print("[yellow]No sessions found.[/yellow]")
            return 0

        # Use display layer for formatting
        table = format_session_table(sessions)
        console.print(table)
        console.print(f"\n[dim]Showing {len(sessions)} sessions[/dim]")

        return 0

    finally:
        manager.close()


def cmd_web(args) -> int:
    """Start web server."""
    import webbrowser

    import uvicorn

    from .web import create_app

    console.print("\n[bold]CLEM - Starting Web Server[/bold]\n")

    host = args.host
    port = args.port
    open_browser = not args.no_browser

    console.print(f"[cyan]Starting server at http://{host}:{port}[/cyan]")
    console.print(f"[dim]API docs: http://{host}:{port}/api/docs[/dim]\n")

    if open_browser:
        console.print("[dim]Opening browser...[/dim]\n")
        webbrowser.open(f"http://{host}:{port}/api/docs")

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")

    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CLEM - Claude Learning & Experience Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version=f"clem {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # rebuild command
    parser_rebuild = subparsers.add_parser(
        "rebuild", help="Rebuild database from source (nuclear option)"
    )
    parser_rebuild.set_defaults(func=cmd_rebuild)

    # stats command
    parser_stats = subparsers.add_parser("stats", help="Show database statistics")
    parser_stats.set_defaults(func=cmd_stats)

    # domains command
    parser_domains = subparsers.add_parser("domains", help="List all domains")
    parser_domains.set_defaults(func=cmd_domains)

    # projects command
    parser_projects = subparsers.add_parser("projects", help="List all projects")
    parser_projects.add_argument("--domain", help="Filter by domain")
    parser_projects.set_defaults(func=cmd_projects)

    # sessions command
    parser_sessions = subparsers.add_parser("sessions", help="List sessions")
    parser_sessions.add_argument("--project", help="Filter by project name")
    parser_sessions.add_argument("--domain", help="Filter by domain")
    parser_sessions.add_argument(
        "--limit", type=int, default=20, help="Limit number of results (default: 20)"
    )
    parser_sessions.set_defaults(func=cmd_sessions)

    # web command
    parser_web = subparsers.add_parser("web", help="Start web server")
    parser_web.add_argument(
        "--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)"
    )
    parser_web.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    parser_web.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )
    parser_web.set_defaults(func=cmd_web)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    return cast(int, args.func(args))


if __name__ == "__main__":
    sys.exit(main())
