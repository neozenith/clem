"""Tests for core.domain module."""

from pathlib import Path
import pytest

from clem.core.domain import (
    extract_domain_and_project,
    encode_to_claude_id,
    verify_encoding,
    generate_domain_id,
    generate_project_id,
)


class TestExtractDomainAndProject:
    """Tests for extract_domain_and_project function."""

    def test_simple_project(self):
        """Test extraction with simple path (one level)."""
        home = Path("/Users/test")
        cwd = "/Users/test/play/clem"

        result = extract_domain_and_project(cwd, home)

        assert result.domain_path == "play"
        assert result.project_name == "clem"
        assert result.full_path == cwd

    def test_nested_domain(self):
        """Test extraction with nested domain."""
        home = Path("/Users/test")
        cwd = "/Users/test/clients/nine/agt-nam-self-service-agent"

        result = extract_domain_and_project(cwd, home)

        assert result.domain_path == "clients/nine"
        assert result.project_name == "agt-nam-self-service-agent"
        assert result.full_path == cwd

    def test_deeply_nested_domain(self):
        """Test extraction with deeply nested domain."""
        home = Path("/Users/test")
        cwd = "/Users/test/foss/adk-samples/python/agents/short-movie-agents"

        result = extract_domain_and_project(cwd, home)

        assert result.domain_path == "foss/adk-samples/python/agents"
        assert result.project_name == "short-movie-agents"
        assert result.full_path == cwd

    def test_project_in_home_root(self):
        """Test extraction when project is directly in home."""
        home = Path("/Users/test")
        cwd = "/Users/test/myproject"

        result = extract_domain_and_project(cwd, home)

        assert result.domain_path == ""
        assert result.project_name == "myproject"
        assert result.full_path == cwd

    def test_path_with_tilde(self):
        """Test that tilde is expanded."""
        home = Path.home()
        cwd = "~/play/clem"

        result = extract_domain_and_project(cwd, home)

        assert result.domain_path == "play"
        assert result.project_name == "clem"

    def test_empty_path(self):
        """Test handling of empty path."""
        home = Path("/Users/test")
        cwd = "/Users/test/"

        result = extract_domain_and_project(cwd, home)

        assert result.domain_path == ""
        assert result.project_name == ""


class TestEncodeToClaudeId:
    """Tests for encode_to_claude_id function."""

    def test_encode_simple_path(self):
        """Test encoding a simple path."""
        path = "/Users/test/play/clem"
        result = encode_to_claude_id(path)

        assert result == "-Users-test-play-clem"

    def test_encode_nested_path(self):
        """Test encoding a nested path."""
        path = "/Users/test/clients/nine/agt-nam-self-service-agent"
        result = encode_to_claude_id(path)

        assert result == "-Users-test-clients-nine-agt-nam-self-service-agent"

    def test_encode_path_with_hyphens(self):
        """Test that existing hyphens are preserved (lossy encoding)."""
        path = "/Users/test/my-project/sub-dir"
        result = encode_to_claude_id(path)

        # This demonstrates the lossy nature - can't distinguish hyphens from slashes
        assert result == "-Users-test-my-project-sub-dir"


class TestVerifyEncoding:
    """Tests for verify_encoding function."""

    def test_verify_matching_encoding(self):
        """Test that matching encoding returns True."""
        cwd = "/Users/test/play/clem"
        claude_id = "-Users-test-play-clem"

        assert verify_encoding(cwd, claude_id) is True

    def test_verify_mismatched_encoding(self):
        """Test that mismatched encoding returns False."""
        cwd = "/Users/test/play/clem"
        claude_id = "-Users-test-work-clem"

        assert verify_encoding(cwd, claude_id) is False


class TestGenerateDomainId:
    """Tests for generate_domain_id function."""

    def test_generate_domain_id_simple(self):
        """Test generating domain ID for simple domain."""
        domain_path = "play"
        result = generate_domain_id(domain_path)

        assert result == "play"

    def test_generate_domain_id_nested(self):
        """Test generating domain ID for nested domain."""
        domain_path = "clients/nine"
        result = generate_domain_id(domain_path)

        assert result == "clients/nine"


class TestGenerateProjectId:
    """Tests for generate_project_id function."""

    def test_generate_project_id_with_domain(self):
        """Test generating project ID with domain."""
        domain_path = "play"
        project_name = "clem"

        result = generate_project_id(domain_path, project_name)

        assert result == "play::clem"

    def test_generate_project_id_nested_domain(self):
        """Test generating project ID with nested domain."""
        domain_path = "clients/nine"
        project_name = "agt-nam-self-service-agent"

        result = generate_project_id(domain_path, project_name)

        assert result == "clients/nine::agt-nam-self-service-agent"

    def test_generate_project_id_no_domain(self):
        """Test generating project ID without domain."""
        domain_path = ""
        project_name = "standalone"

        result = generate_project_id(domain_path, project_name)

        assert result == "::standalone"
