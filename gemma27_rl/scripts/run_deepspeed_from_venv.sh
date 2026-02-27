#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Usage:
#   VENV_BIN=/abs/path/to/.venv_train/bin \
#   INCLUDE=localhost:0,1,2,3,4,5,6,7 \
#   CONFIG=configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_mqm_scale8gpu.yaml \
#   ./scripts/run_deepspeed_from_venv.sh

if [[ -z "${VENV_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv_train/bin/python" ]]; then
    VENV_BIN="${ROOT_DIR}/.venv_train/bin"
  else
    VENV_BIN="${ROOT_DIR}/.venv/bin"
  fi
fi

INCLUDE="${INCLUDE:-localhost:0,1,2,3,4,5,6,7}"
CONFIG="${CONFIG:-configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_mqm_scale8gpu.yaml}"

PY="${VENV_BIN}/python"
DS="${VENV_BIN}/deepspeed"
ENTRY="${VENV_BIN}/gemma27_rl"

if [[ ! -x "${PY}" ]]; then
  echo "[error] python not found at ${PY}"
  exit 1
fi
if [[ ! -x "${DS}" ]]; then
  echo "[error] deepspeed not found at ${DS}"
  exit 1
fi
if [[ ! -x "${ENTRY}" ]]; then
  echo "[error] gemma27_rl entrypoint not found at ${ENTRY}"
  exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
  echo "[error] config not found: ${CONFIG}"
  exit 1
fi

echo "[info] ROOT_DIR=${ROOT_DIR}"
echo "[info] VENV_BIN=${VENV_BIN}"
echo "[info] INCLUDE=${INCLUDE}"
echo "[info] CONFIG=${CONFIG}"
echo

echo "[step] verify runtime imports in same venv"
"${PY}" - <<'PY'
import sys
import torch
import flash_attn_2_cuda
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("flash_attn_2_cuda import OK")
PY
echo

echo "[step] launch deepspeed"
exec "${DS}" --include "${INCLUDE}" "${ENTRY}" --config "${CONFIG}"
