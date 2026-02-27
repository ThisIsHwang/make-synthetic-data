"""Ensure the package root is importable by tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root so `import qwen35_moe_rl` works without pip install.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
