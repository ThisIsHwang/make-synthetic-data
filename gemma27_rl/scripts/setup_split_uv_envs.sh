#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv command not found. Install uv first."
  exit 1
fi

PYTHON_VER="${PYTHON_VER:-3.10}"

echo "[1/5] Creating uv environments..."
uv venv .venv_train --python "${PYTHON_VER}"
uv venv .venv_metricx --python "${PYTHON_VER}"
uv venv .venv_xcomet --python "${PYTHON_VER}"

echo "[2/5] Installing training environment packages..."
uv pip install --python .venv_train/bin/python -r requirements.txt
uv pip install --python .venv_train/bin/python -e .

echo "[3/5] Installing MetricX environment packages..."
uv pip install --python .venv_metricx/bin/python \
  torch transformers sentencepiece accelerate huggingface_hub
uv pip install --python .venv_metricx/bin/python -e .

echo "[4/5] Installing xCOMET environment packages..."
uv pip install --python .venv_xcomet/bin/python \
  torch unbabel-comet
uv pip install --python .venv_xcomet/bin/python -e .

echo "[5/5] Done."
echo "train python   : ${ROOT_DIR}/.venv_train/bin/python"
echo "metricx python : ${ROOT_DIR}/.venv_metricx/bin/python"
echo "xcomet python  : ${ROOT_DIR}/.venv_xcomet/bin/python"

