"""Projects API endpoints."""

from fastapi import APIRouter, HTTPException, Query

from ...config import get_database_path
from ...database.manager import DatabaseManager
from ...queries import ProjectQuery
from ..models import ProjectResponse

router = APIRouter()


@router.get("", response_model=list[ProjectResponse])
async def list_projects(domain_id: str | None = Query(None, description="Filter by domain ID")):
    """List all projects, optionally filtered by domain.

    Args:
        domain_id: Optional domain filter

    Returns:
        List of projects with statistics
    """
    db_path = get_database_path()

    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found. Run 'clem rebuild' first.")

    manager = DatabaseManager()

    try:
        query = ProjectQuery(manager)
        projects = query.list_all(domain_id=domain_id)

        return [
            ProjectResponse(
                project_name=p.project_name,
                domain_id=p.domain_id,
                session_count=p.session_count,
                cwd=p.cwd,
            )
            for p in projects
        ]

    finally:
        manager.close()


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: str):
    """Get a specific project by ID.

    Args:
        project_id: Project identifier (format: domain::project_name)

    Returns:
        Project statistics
    """
    db_path = get_database_path()

    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found. Run 'clem rebuild' first.")

    manager = DatabaseManager()

    try:
        query = ProjectQuery(manager)
        project = query.get_by_id(project_id)

        if not project:
            raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")

        return ProjectResponse(
            project_name=project.project_name,
            domain_id=project.domain_id,
            session_count=project.session_count,
            cwd=project.cwd,
        )

    finally:
        manager.close()
