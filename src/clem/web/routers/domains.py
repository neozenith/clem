"""Domains API endpoints."""

from fastapi import APIRouter, HTTPException

from ...config import get_database_path
from ...database.manager import DatabaseManager
from ...queries import DomainQuery
from ..models import DomainResponse

router = APIRouter()


@router.get("", response_model=list[DomainResponse])
async def list_domains():
    """List all domains.

    Returns:
        List of domains with statistics
    """
    db_path = get_database_path()

    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found. Run 'clem rebuild' first.")

    manager = DatabaseManager()

    try:
        query = DomainQuery(manager)
        domains = query.list_all()

        return [
            DomainResponse(
                domain_id=d.domain_id,
                domain_path=d.domain_path,
                project_count=d.project_count,
                session_count=d.session_count,
            )
            for d in domains
        ]

    finally:
        manager.close()


@router.get("/{domain_id}", response_model=DomainResponse)
async def get_domain(domain_id: str):
    """Get a specific domain by ID.

    Args:
        domain_id: Domain identifier

    Returns:
        Domain statistics
    """
    db_path = get_database_path()

    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found. Run 'clem rebuild' first.")

    manager = DatabaseManager()

    try:
        query = DomainQuery(manager)
        domain = query.get_by_id(domain_id)

        if not domain:
            raise HTTPException(status_code=404, detail=f"Domain '{domain_id}' not found")

        return DomainResponse(
            domain_id=domain.domain_id,
            domain_path=domain.domain_path,
            project_count=domain.project_count,
            session_count=domain.session_count,
        )

    finally:
        manager.close()
