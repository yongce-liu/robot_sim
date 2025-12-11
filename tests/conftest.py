from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture
def project_root() -> Path:
    """Root directory of the project (repository root)."""

    return PROJECT_ROOT


@pytest.fixture
def default_config_path(project_root: Path) -> Path:
    """Path to the default simulator configuration file."""

    return project_root / "configs" / "default.yaml"
