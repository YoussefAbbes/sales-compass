"""Run contract tests exclusively against committed anonymised fixtures."""
import os
import sys
from pathlib import Path


API_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = API_ROOT.parent

os.environ["MODEL_RESULTS_SOURCE_ROOT"] = str(WORKSPACE_ROOT / "fixtures")
os.environ.pop("DATA_ROOT", None)
sys.path.insert(0, str(WORKSPACE_ROOT))


import shutil  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture
def scratch_dir() -> Path:
    """A writable throwaway directory.

    Mirrors tests/conftest.py: pytest's tmp_path raises PermissionError under
    %LOCALAPPDATA%\Temp on this machine, so scratch space stays inside the repo
    (gitignored) rather than depending on the system temp directory.
    """
    d = WORKSPACE_ROOT / ".pytest_tmp" / uuid.uuid4().hex[:8]
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)
