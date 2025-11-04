"""Statistics API endpoints."""

from fastapi import APIRouter, HTTPException

from ...config import get_database_path
from ...database.builder import DatabaseBuilder
from ..models import StatsResponse

router = APIRouter()


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics.

    Returns:
        Database statistics including counts and metadata
    """
    db_path = get_database_path()

    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found. Run 'clem rebuild' first.")

    builder = DatabaseBuilder()

    try:
        stats = builder.get_stats()

        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])

        return StatsResponse(**stats)

    finally:
        builder.close()
