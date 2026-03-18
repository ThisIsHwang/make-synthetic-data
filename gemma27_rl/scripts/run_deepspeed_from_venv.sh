#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# Usage:
#   VENV_BIN=/abs/path/to/.venv_train/bin \
#   INCLUDE=localhost:0,1,2,3,4,5,6,7 \
#   CONFIG=configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_mqm_scale8gpu.yaml \
#   ./scripts/run_deepspeed_from_venv.sh
#
# Multi-node example:
#   HOSTFILE=/abs/path/to/hostfile \
#   INCLUDE='policy-node-01:0,1,2,3,4,5,6,7@policy-node-02:0,1,2,3,4,5,6,7' \
#   CONFIG=configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_metricx_xcomet_multinode8p1aux.yaml \
#   ./scripts/run_deepspeed_from_venv.sh

if [[ -z "${VENV_BIN:-}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv_train/bin/python" ]]; then
    VENV_BIN="${ROOT_DIR}/.venv_train/bin"
  else
    VENV_BIN="${ROOT_DIR}/.venv/bin"
  fi
fi

HOSTFILE="${HOSTFILE:-}"
INCLUDE="${INCLUDE:-}"
EXCLUDE="${EXCLUDE:-}"
CONFIG="${CONFIG:-configs/qwen35_mqm/train_wmt24pp_enko_qwen35_27b_mqm_scale8gpu.yaml}"

if [[ -z "${HOSTFILE}" && -z "${INCLUDE}" ]]; then
  INCLUDE="localhost:0,1,2,3,4,5,6,7"
fi

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
if [[ -n "${HOSTFILE}" && ! -f "${HOSTFILE}" ]]; then
  echo "[error] hostfile not found: ${HOSTFILE}"
  exit 1
fi

echo "[info] ROOT_DIR=${ROOT_DIR}"
echo "[info] VENV_BIN=${VENV_BIN}"
echo "[info] PYTHONPATH=${PYTHONPATH}"
echo "[info] HOSTFILE=${HOSTFILE:-<none>}"
echo "[info] INCLUDE=${INCLUDE}"
echo "[info] EXCLUDE=${EXCLUDE:-<none>}"
echo "[info] CONFIG=${CONFIG}"
echo

echo "[step] verify runtime imports in same venv"
"${PY}" - <<'PY'
import sys
import torch
import flash_attn_2_cuda
import gemma27_rl
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("flash_attn_2_cuda import OK")
print("gemma27_rl:", gemma27_rl.__file__)
PY
echo

DS_ARGS=()
if [[ -n "${HOSTFILE}" ]]; then
  DS_ARGS+=(--hostfile "${HOSTFILE}")
fi
if [[ -n "${INCLUDE}" ]]; then
  DS_ARGS+=(--include "${INCLUDE}")
fi
if [[ -n "${EXCLUDE}" ]]; then
  DS_ARGS+=(--exclude "${EXCLUDE}")
fi

# This trainer has rank0-only eval/reward sections, so workers can wait in
# collectives longer than the default NCCL watchdog heartbeat timeout.
if [[ -z "${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-}" ]]; then
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
fi

echo "[step] launch deepspeed"
echo "[info] TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC}"
exec "${DS}" "${DS_ARGS[@]}" "${ENTRY}" --config "${CONFIG}"
