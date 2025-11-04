"""Sessions API endpoints."""

from fastapi import APIRouter, HTTPException, Query

from ...config import get_database_path
from ...database.manager import DatabaseManager
from ...queries import SessionQuery
from ..models import SessionResponse

router = APIRouter()


@router.get("", response_model=list[SessionResponse])
async def list_sessions(
    project_name: str | None = Query(None, description="Filter by project name"),
    domain_id: str | None = Query(None, description="Filter by domain ID"),
    limit: int = Query(20, ge=1, le=100, description="Maximum number of results"),
):
    """List sessions with optional filters.

    Args:
        project_name: Optional project name filter
        domain_id: Optional domain filter
        limit: Maximum number of results (1-100, default: 20)

    Returns:
        List of sessions with statistics
    """
    db_path = get_database_path()

    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found. Run 'clem rebuild' first.")

    manager = DatabaseManager()

    try:
        query = SessionQuery(manager)
        sessions = query.list_all(project_name=project_name, domain_id=domain_id, limit=limit)

        return [
            SessionResponse(
                session_id=s.session_id,
                project_name=s.project_name,
                domain_path=s.domain_path,
                event_count=s.event_count,
                started_at=str(s.started_at) if s.started_at else None,
            )
            for s in sessions
        ]

    finally:
        manager.close()


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get a specific session by ID.

    Args:
        session_id: Session identifier

    Returns:
        Session statistics
    """
    db_path = get_database_path()

    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found. Run 'clem rebuild' first.")

    manager = DatabaseManager()

    try:
        query = SessionQuery(manager)
        session = query.get_by_id(session_id)

        if not session:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        return SessionResponse(
            session_id=session.session_id,
            project_name=session.project_name,
            domain_path=session.domain_path,
            event_count=session.event_count,
            started_at=str(session.started_at) if session.started_at else None,
        )

    finally:
        manager.close()
