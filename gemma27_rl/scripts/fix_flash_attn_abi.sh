#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Usage:
#   VENV_BIN=/abs/path/to/.venv_train/bin ./scripts/fix_flash_attn_abi.sh
#
# Defaults to .venv_train/bin if present, otherwise .venv/bin.
if [[ -z "${VENV_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv_train/bin/python" ]]; then
    VENV_BIN="${ROOT_DIR}/.venv_train/bin"
  else
    VENV_BIN="${ROOT_DIR}/.venv/bin"
  fi
fi

PY="${VENV_BIN}/python"
PIP="${VENV_BIN}/pip"
DS="${VENV_BIN}/deepspeed"

if [[ ! -x "${PY}" ]]; then
  echo "[error] python not found at ${PY}"
  echo "        Set VENV_BIN to your training venv bin directory."
  exit 1
fi
if [[ ! -x "${PIP}" ]]; then
  echo "[error] pip not found at ${PIP}"
  exit 1
fi

echo "[info] ROOT_DIR=${ROOT_DIR}"
echo "[info] VENV_BIN=${VENV_BIN}"
echo "[info] python=${PY}"
echo "[info] pip=${PIP}"
echo "[info] deepspeed=${DS}"
echo

echo "[step] binary versions"
"${PY}" -V
"${PIP}" -V
if [[ -x "${DS}" ]]; then
  "${DS}" --version || true
else
  echo "[warn] deepspeed not installed in this venv"
fi
echo

echo "[step] before-state import probe"
"${PY}" - <<'PY'
import sys
print("python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda)
except Exception as e:
    print("torch import error:", repr(e))
for m in ("flash_attn", "flash_attn_2_cuda"):
    try:
        __import__(m)
        print(m, "import OK")
    except Exception as e:
        print(m, "import error:", repr(e))
PY
echo

echo "[step] purge stale torch extension cache"
rm -rf "${HOME}/.cache/torch_extensions"
echo

echo "[step] uninstall old flash-attn artifacts (if any)"
"${PIP}" uninstall -y flash-attn flash_attn || true
echo

echo "[step] reinstall flash-attn for current torch ABI"
"${PIP}" install --upgrade pip setuptools wheel packaging ninja
"${PIP}" install --no-cache-dir --no-build-isolation "flash-attn==2.7.4.post1"
echo

echo "[step] after-state import probe"
"${PY}" - <<'PY'
import torch
import flash_attn, flash_attn_2_cuda
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("flash_attn:", getattr(flash_attn, "__version__", "unknown"))
print("flash_attn_2_cuda import OK")
PY
echo

echo "[done] FlashAttention ABI repair completed."
echo
echo "If build/import still fails:"
echo "  1) check nvcc: nvcc --version"
echo "  2) verify this same venv is used for training launch"
echo "  3) temporary fallback: attn_implementation=sdpa"
