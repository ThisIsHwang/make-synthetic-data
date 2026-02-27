from __future__ import annotations

import sys
from pathlib import Path

# When deepspeed launcher runs this as `python -u __main__.py`,
# there is no package context. Add the project root to sys.path
# so that absolute imports work.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from qwen35_moe_rl.cli import main  # noqa: E402

raise SystemExit(main())
