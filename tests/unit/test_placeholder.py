"""Placeholder tests — replaced as modules are built."""

from src.api.app import app


def test_project_imports() -> None:
    """Verify core modules can be imported."""
    assert app is not None


def test_placeholder() -> None:
    """Remove this when real tests exist."""
    assert 1 + 1 == 2
