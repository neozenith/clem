"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field


class DomainResponse(BaseModel):
    """Domain statistics response."""

    domain_id: str = Field(..., description="Domain identifier")
    domain_path: str = Field(..., description="Domain path (e.g., 'play', 'work')")
    project_count: int = Field(..., description="Number of projects in domain")
    session_count: int = Field(..., description="Number of sessions in domain")


class ProjectResponse(BaseModel):
    """Project statistics response."""

    project_name: str = Field(..., description="Project name")
    domain_id: str = Field(..., description="Domain identifier")
    session_count: int = Field(..., description="Number of sessions for project")
    cwd: str = Field(..., description="Working directory path")


class SessionResponse(BaseModel):
    """Session statistics response."""

    session_id: str = Field(..., description="Session identifier")
    project_name: str = Field(..., description="Project name")
    domain_path: str = Field(..., description="Domain path")
    event_count: int = Field(..., description="Number of events in session")
    started_at: str | None = Field(None, description="Session start timestamp")


class StatsResponse(BaseModel):
    """Database statistics response."""

    domains: int = Field(..., description="Total number of domains")
    projects: int = Field(..., description="Total number of projects")
    sessions: int = Field(..., description="Total number of sessions")
    memories: int = Field(..., description="Total number of memories")
    schema_version: str = Field(..., description="Database schema version")
    last_rebuild: str | None = Field(None, description="Last rebuild timestamp")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Health status")
    service: str = Field(..., description="Service name")


class EventResponse(BaseModel):
    """Event response from session."""

    timestamp: str | None = Field(None, description="Event timestamp")
    type: str | None = Field(None, description="Event type")
    role: str | None = Field(None, description="Message role (user/assistant)")
    content: str | None = Field(None, description="Message content")
    tool_name: str | None = Field(None, description="Tool name if tool use")
    tool_input: str | None = Field(None, description="Tool input if tool use")
    tool_output: str | None = Field(None, description="Tool output if tool use")
    error: str | None = Field(None, description="Error message if error occurred")
