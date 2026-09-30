# -*- coding: utf-8 -*-
"""Shared fixtures. Paths are resolved relative to the repo, never hardcoded."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Make src/ importable as top-level modules (validation, ui_theme).
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def app_path() -> str:
    return str(ROOT / "src" / "app.py")


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    d = project_root / "data"
    if not (d / "clients_to_contact.csv").exists():
        pytest.skip("data/clients_to_contact.csv missing — run the pipeline first")
    return d


@pytest.fixture
def scratch_dir() -> Path:
    """A writable throwaway directory.

    pytest's built-in tmp_path is unusable here: %LOCALAPPDATA%\\Temp\\pytest-of-*
    raises PermissionError on this machine. Keeping the scratch space inside the
    repo (gitignored) avoids depending on the system temp directory at all.
    """
    import shutil
    import uuid

    d = ROOT / ".pytest_tmp" / uuid.uuid4().hex[:8]
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
