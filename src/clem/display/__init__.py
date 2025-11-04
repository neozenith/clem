"""Display layer for CLEM CLI output formatting.

Provides Rich-based formatters for tables and statistics.
"""

from .tables import (
    format_domain_table,
    format_project_table,
    format_session_table,
    format_stats_table,
)

__all__ = [
    "format_domain_table",
    "format_project_table",
    "format_session_table",
    "format_stats_table",
]
