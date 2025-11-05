"""Events API router."""

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from ...config import get_database_path
from ...database.manager import DatabaseManager
from ...queries import EventQuery
from ...queries.sessions import SessionQuery
from ..models import EventResponse

router = APIRouter()


@router.get("/{session_id}/events", response_model=list[EventResponse])
async def list_events(
    session_id: str,
    limit: int = Query(100, ge=1, le=500, description="Maximum number of events"),
    offset: int = Query(0, ge=0, description="Number of events to skip"),
):
    """List events for a specific session.

    Events are read directly from the session .jsonl file.
    """
    db_path = get_database_path()
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found. Run 'clem rebuild' first.")

    manager = DatabaseManager()
    try:
        # Get session to find file path
        query = SessionQuery(manager)
        session = query.get_by_id(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Get events from session file
        session_file_path = session.file_path
        if not Path(session_file_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Session file not found: {session_file_path}",
            )

        event_query = EventQuery(session_file_path)
        events = event_query.list_all(limit=limit, offset=offset)

        return [
            EventResponse(
                timestamp=e.timestamp,
                type=e.type,
                role=e.role,
                content=e.content,
                tool_name=e.tool_name,
                tool_input=e.tool_input,
                tool_output=e.tool_output,
                error=e.error,
            )
            for e in events
        ]

    finally:
        manager.close()


@router.get("/{session_id}/events/count")
async def count_events(session_id: str):
    """Get event count for a session."""
    db_path = get_database_path()
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found. Run 'clem rebuild' first.")

    manager = DatabaseManager()
    try:
        # Get session to find file path
        query = SessionQuery(manager)
        session = query.get_by_id(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        # Get event count from session file
        session_file_path = session.file_path
        if not Path(session_file_path).exists():
            raise HTTPException(
                status_code=404,
                detail=f"Session file not found: {session_file_path}",
            )

        event_query = EventQuery(session_file_path)
        count = event_query.count()

        return {"session_id": session_id, "event_count": count}

    finally:
        manager.close()
