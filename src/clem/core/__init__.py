"""Core domain logic for CLEM.

Handles domain detection, project discovery, and session management.
"""

from .domain import encode_to_claude_id, extract_domain_and_project, verify_encoding

__all__ = [
    "extract_domain_and_project",
    "encode_to_claude_id",
    "verify_encoding",
]
