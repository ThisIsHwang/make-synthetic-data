from __future__ import annotations

import pytest
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def configs_dir() -> Path:
    return CONFIGS_DIR
