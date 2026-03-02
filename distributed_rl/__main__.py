"""Entry point for ``python -m distributed_rl`` and DeepSpeed launcher.

Why this file exists:
  When you run ``python -m distributed_rl``, Python executes this ``__main__.py``.
  Both ``torchrun`` and ``deepspeed`` launchers also invoke modules this way:

    torchrun --nproc_per_node=6 -m distributed_rl --config configs/foo.yaml

  ``torchrun`` spawns N worker processes, each running this file.  Each process
  gets its own ``LOCAL_RANK`` and ``RANK`` environment variables set automatically
  by torchrun.  For example, with ``--nproc_per_node=6``:
    - Process 0: LOCAL_RANK=0, RANK=0
    - Process 1: LOCAL_RANK=1, RANK=1
    - ...
    - Process 5: LOCAL_RANK=5, RANK=5

sys.path manipulation:
  torchrun may launch this from an arbitrary working directory, so we add the
  project root (parent of ``distributed_rl/``) to ``sys.path`` to ensure
  ``import distributed_rl`` always works.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so imports work when launched with
# deepspeed or torchrun from outside the package directory.
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from distributed_rl.cli import main  # noqa: E402

raise SystemExit(main())
