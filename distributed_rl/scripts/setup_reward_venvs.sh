#!/usr/bin/env bash
set -euo pipefail

# Setup reward model venvs (MetricX + XComet) for distributed_rl.
#
# Usage (from project root):
#   bash distributed_rl/scripts/setup_reward_venvs.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

# uv's default index strategy only considers versions from the FIRST index
# that contains a given package. The PyTorch cu126 index has torchmetrics==1.0.3
# but unbabel-comet needs torchmetrics>=0.10.2,<0.11.0 from PyPI.
# --index-strategy unsafe-best-match picks the best version across ALL indexes.
UV_INDEX_STRATEGY="--index-strategy unsafe-best-match"

echo "Project root: ${PROJECT_ROOT}"
echo ""

# ── MetricX venv ──────────────────────────────────────────────
echo "[1/2] Setting up MetricX venv (.venv-metrics) ..."
uv venv .venv-metrics
uv pip install --python .venv-metrics/bin/python ${UV_INDEX_STRATEGY} \
    -r distributed_rl/requirements-metrics.txt
echo "  MetricX venv ready."
echo ""

# ── XComet venv ───────────────────────────────────────────────
echo "[2/2] Setting up XComet venv (.venv-xcomet) ..."
uv venv .venv-xcomet
uv pip install --python .venv-xcomet/bin/python ${UV_INDEX_STRATEGY} \
    -r distributed_rl/requirements-xcomet.txt
echo "  XComet venv ready."

# ── Verify ────────────────────────────────────────────────────
echo ""
echo "=== Verifying ==="
.venv-xcomet/bin/python -c "import pkg_resources; print('  pkg_resources: OK')"
.venv-xcomet/bin/python -c "from comet import download_model; print('  unbabel-comet: OK')"
echo ""
echo "=== Done ==="
echo "MetricX python : ${PROJECT_ROOT}/.venv-metrics/bin/python"
echo "XComet python  : ${PROJECT_ROOT}/.venv-xcomet/bin/python"
