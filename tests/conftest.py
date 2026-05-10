from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest


_TMP_ROOT = Path(__file__).resolve().parents[1] / ".pytest-local-tmp"


@pytest.fixture
def tmp_path() -> Iterator[Path]:
    """Workspace-local tmp_path override for Windows sandbox temp-dir cleanup issues."""

    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = _TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
