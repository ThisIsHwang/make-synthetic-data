#!/usr/bin/env bash
set -euo pipefail

# Setup training venv (.distributed_rl) for distributed_rl.
#
# Installs PyTorch with the correct CUDA version first, then the rest of the
# dependencies. This avoids the CUDA 12.6 vs 12.8 mismatch issue.
#
# Usage (from project root):
#   bash distributed_rl/scripts/setup_training_venv.sh          # auto-detect CUDA
#   bash distributed_rl/scripts/setup_training_venv.sh cu126    # force CUDA 12.6
#   bash distributed_rl/scripts/setup_training_venv.sh cu128    # force CUDA 12.8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

VENV_DIR=".distributed_rl"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: uv not found. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

# ── Detect CUDA version ──────────────────────────────────────
if [[ -n "${1:-}" ]]; then
  CUDA_TAG="$1"
  echo "Using user-specified CUDA tag: ${CUDA_TAG}"
else
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_FULL=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
  elif command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_FULL=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
  else
    echo "ERROR: Cannot detect CUDA version. Neither nvcc nor nvidia-smi found."
    echo "Usage: $0 <cuda-tag>  (e.g. cu126, cu128)"
    exit 1
  fi
  CUDA_TAG="cu$(echo "${CUDA_FULL}" | tr -d '.')"
  echo "Auto-detected CUDA ${CUDA_FULL} → ${CUDA_TAG}"
fi

TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
echo "PyTorch index: ${TORCH_INDEX}"
echo ""

# ── Create venv ───────────────────────────────────────────────
echo "[1/4] Creating venv (${VENV_DIR}) ..."
uv venv "${VENV_DIR}"
PYTHON="${VENV_DIR}/bin/python"
echo ""

# ── Install PyTorch (CUDA-specific) ──────────────────────────
echo "[2/4] Installing PyTorch (${CUDA_TAG}) ..."
uv pip install --python "${PYTHON}" \
    torch --index-url "${TORCH_INDEX}"
echo ""

# ── Install remaining dependencies ───────────────────────────
echo "[3/4] Installing dependencies ..."
uv pip install --python "${PYTHON}" \
    --index-strategy unsafe-best-match \
    -r distributed_rl/requirements.txt
echo ""

# ── Install flash-attn (built from source against current torch) ──
echo "[4/4] Installing flash-attn (source build, may take several minutes) ..."
uv pip install --python "${PYTHON}" \
    flash-attn --no-build-isolation --force-reinstall --no-cache

# Verify flash-attn actually loads with the current torch ABI
"${PYTHON}" -c "from flash_attn import flash_attn_func; print('  flash-attn import: OK')" || {
  echo ""
  echo "ERROR: flash-attn installed but failed to import (ABI mismatch)."
  echo "  Possible causes:"
  echo "    1. CUDA toolkit version mismatch (check: nvcc --version)"
  echo "    2. gcc/g++ version incompatibility"
  echo "    3. Pre-built wheel was used instead of source build"
  echo ""
  echo "  Try manually:"
  echo "    uv pip install --python ${PYTHON} flash-attn --no-build-isolation --force-reinstall --no-cache --no-binary flash-attn"
  exit 1
}
echo ""

# ── Install distributed_rl as editable package ────────────────
# --no-deps: dependencies are already installed above via requirements.txt.
# Without this flag, setup.py's install_requires would re-resolve versions
# and could downgrade packages (e.g. transformers, huggingface-hub).
echo "Installing distributed_rl in editable mode ..."
uv pip install --python "${PYTHON}" --no-deps -e distributed_rl/
echo ""

# ── Verify ────────────────────────────────────────────────────
echo "=== Verifying ==="
"${PYTHON}" -c "import torch; print(f'  torch {torch.__version__}  CUDA {torch.version.cuda}')"
"${PYTHON}" -c "import transformers; print(f'  transformers {transformers.__version__}')"
"${PYTHON}" -c "import trl; print(f'  trl {trl.__version__}')"
"${PYTHON}" -c "import deepspeed; print(f'  deepspeed {deepspeed.__version__}')"
"${PYTHON}" -c "from flash_attn import flash_attn_func; print('  flash-attn: OK')"
echo ""
echo "=== Done ==="
echo "Training python: ${PROJECT_ROOT}/${PYTHON}"
echo ""
echo "Launch training:"
echo "  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ${PYTHON} -m torchrun --nproc_per_node=6 \\"
echo "      -m distributed_rl --config distributed_rl/configs/gemma27b_8gpu.yaml"
