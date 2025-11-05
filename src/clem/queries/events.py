"""Query layer for session events.

Reads events directly from session .jsonl files.
"""

from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass
class EventRecord:
    """Event record from session file."""

    timestamp: str | None
    type: str | None
    role: str | None
    content: str | None
    tool_name: str | None
    tool_input: str | None
    tool_output: str | None
    error: str | None


class EventQuery:
    """Query session events from .jsonl files."""

    def __init__(self, session_file_path: str):
        """Initialize event query.

        Args:
            session_file_path: Path to session .jsonl file
        """
        self.session_file_path = Path(session_file_path)
        if not self.session_file_path.exists():
            raise FileNotFoundError(f"Session file not found: {session_file_path}")

    def list_all(self, limit: int = 100, offset: int = 0) -> list[EventRecord]:
        """List all events in the session.

        Args:
            limit: Maximum number of events to return
            offset: Number of events to skip

        Returns:
            List of event records ordered by timestamp
        """
        conn = duckdb.connect(":memory:")

        try:
            # Session files have nested JSON structure
            # message.role, message.content, etc.
            result = conn.execute(
                f"""
                SELECT
                    timestamp,
                    type,
                    json_extract_string(message, '$.role') as role,
                    json_extract_string(message, '$.content') as content,
                    NULL as tool_name,
                    NULL as tool_input,
                    NULL as tool_output,
                    NULL as error
                FROM read_ndjson_auto('{self.session_file_path}')
                WHERE type IN ('user', 'assistant')
                ORDER BY timestamp ASC
                LIMIT {limit}
                OFFSET {offset}
            """
            ).fetchall()

            return [
                EventRecord(
                    timestamp=row[0],
                    type=row[1],
                    role=row[2],
                    content=row[3],
                    tool_name=row[4],
                    tool_input=row[5],
                    tool_output=row[6],
                    error=row[7],
                )
                for row in result
            ]

        finally:
            conn.close()

    def count(self) -> int:
        """Count total events in session.

        Returns:
            Total number of events
        """
        conn = duckdb.connect(":memory:")

        try:
            result = conn.execute(
                f"""
                SELECT COUNT(*) as count
                FROM read_ndjson_auto('{self.session_file_path}')
            """
            ).fetchone()

            return result[0] if result else 0

        finally:
            conn.close()
